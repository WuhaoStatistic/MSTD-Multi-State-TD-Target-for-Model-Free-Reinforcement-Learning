import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import copy
import torch
import os
import csv
import gymnasium as gym
from torch.distributions import Normal
from datetime import datetime
from collections import deque
from utils import Actor, Double_Q_Critic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tempbuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.bs = deque(maxlen=capacity + 1)
        self.ba = deque(maxlen=capacity)
        self.br = deque(maxlen=capacity)
        self.bs_ = deque(maxlen=capacity)
        self.bdw = deque(maxlen=capacity)

    def append(self, s, a, r, s_, dw):
        if len(self.bs) == 0:
            self.bs.append(s)
            self.bs.append(s_)
        else:
            self.bs.append(s_)
        self.ba.append(a)
        self.br.append(r)

        # we make dw a matrix to match the shape of value matrix and the operartor
        self.bdw.append((dw == True))
        # diagnal = 1 if end ;

        # left means the last state is an end state
        # right means the last state is not an end state
        # 0 0 1      0 0 0
        # 0 1 0      0 0 0
        # 1 0 0      0 0 0

    def __len__(self):
        return len(self.ba)


class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.ca = capacity
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def add(self, tempbuffer: Tempbuffer):
        '''
        For a long trajectory, our collection is:
         st,  st+1, st+2, st+3 st+4....st+n
            at     at+1   at+2   
            rt+1   rt+2 rt+3 rtt+4 ...rt+n
                                       dw_n
        '''
        s = np.array(list(tempbuffer.bs))
        a = np.array((tempbuffer.ba))
        r = np.array(list(tempbuffer.br))
        dw = np.array((tempbuffer.bdw)[-1])
        self.buffer.append((s, a, r, dw))

    def sample(self, batch_size):
        s, a, r, d = zip(*self.rng.sample(self.buffer, batch_size))
        return np.stack(s), np.stack(a), np.array(r), np.stack(d)

    def __len__(self):
        return len(self.buffer)


def build_net(layer_shape, hidden_activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hidden_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super(Actor, self).__init__()
        layers = [state_dim] + list(hid_shape)

        self.a_net = build_net(layers, hidden_activation, output_activation)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic, with_logprob):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        # we learn log_std rather than std, so that exp(log_std) is always > 0
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        a = torch.tanh(u)
        if with_logprob:
            # Get probability density of logp_pi_a from probability density of u:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2


class MSSAC:
    def __init__(self, state_dim, action_dim, max_action, lr, gamma, tau, batch_size, hid_size, buffer, adaptive_alpha,
                 alpha, pre_steps):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."

        self.tau = tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.lr_a, self.lr_c = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.net_width = hid_size
        self.adaptive_alpha = adaptive_alpha
        self.actor = Actor(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.alpha = float(alpha)
        self.npre = pre_steps
        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(DEVICE)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr_c)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = buffer

        self.voperator_actor = torch.from_numpy(
            np.array([1 / self.npre * self.gamma ** (i + 1) for i in range(self.npre)])).float().to(DEVICE)
        self.roperrator_actor = torch.from_numpy(
            np.array([1 / self.npre * (self.npre - i) * self.gamma ** i for i in range(self.npre)])).float().to(DEVICE)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=DEVICE)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=DEVICE)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr_c)

    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(DEVICE)
            a, _ = self.actor(state, deterministic, with_logprob=True)
        return a.cpu().numpy()[0]

    def train(self):
        s, a, r, dw = self.replay_buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(DEVICE)  # (Batchsize,n_prediction+1,state_dim)

        # we need only need to evaluate the action taken at state t.
        a = torch.FloatTensor(a).to(DEVICE)  # (Batchsize,ac_dim)

        r = torch.FloatTensor(r).to(DEVICE)  # (Batchsize,n_prediction)

        not_d = 1 - torch.FloatTensor(dw).to(DEVICE)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            matrix = torch.zeros((self.batch_size, self.npre)).to(DEVICE)
            for j in range(self.npre):

                if j != self.npre - 1:
                    q1, q2 = self.q_critic_target(s[:, j + 1, :], a[:, j + 1, :])
                    q1 = q1.squeeze(1) * not_d
                    q2 = q2.squeeze(1) * not_d
                    matrix[:, j] = torch.min(q1, q2)
                else:
                    a_, log_a_ = self.actor(s[:, j + 1, :], deterministic=False, with_logprob=True)
                    q1, q2 = self.q_critic_target(s[:, j + 1, :], a_)
                    q1 = q1.squeeze(1) * not_d
                    q2 = q2.squeeze(1) * not_d
                    matrix[:, j] = torch.min(q1, q2) - self.alpha * log_a_.squeeze(1)

            res = torch.matmul(matrix, self.voperator_actor).float()
            res += (r * self.roperrator_actor).sum(dim=1).float()
        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s[:, 0, :], a[:, 0, :])

        critic_loss = torch.nn.functional.mse_loss(current_Q1.squeeze(1), res) + torch.nn.functional.mse_loss(
            current_Q2.squeeze(1), res)
        self.q_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters():
            params.requires_grad = False

        ca, log_pi_a = self.actor(s[:, 0, :], deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s[:, 0, :], ca)
        Q = torch.min(current_Q1, current_Q2).squeeze(1)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters(): params.requires_grad = True

        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def reward_shaping(r, s_, env_index):
    if env_index == 2:
        return (r + 8) / 8
    else:
        return r


def record(csv_path, dictm):
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for k in dictm.keys():
            writer.writerow([k, dictm[k]])
    csv_file.close()


def evaluate_policy(env, agent, max_step):
    times = 1  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        steps = 0
        s, info = env.reset()
        done = False
        t = False
        episode_reward = 0
        while not ((done or t) or steps >= max_step):
            a = agent.select_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, t, _ = env.step(a)
            episode_reward += r
            s = s_
            steps += 1
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


def main(seed):
    EnvName = ['Walker2d-v4', 'Ant-v4', 'InvertedDoublePendulum-v4', 'Humanoid-v4',
               'HalfCheetah-v4', 'Hopper-v4', 'Swimmer-v4']

    BriefEnvName = ['Walker2d', 'Ant', 'IvPendulum', 'Humanoid', 'HalfCheetah', 'Hopper', 'Swimmer']
    env_index = 2
    
    env = gym.make(EnvName[env_index])
    env_evaluate = gym.make(EnvName[env_index])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_episode_step = env._max_episode_steps
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_episode_steps={}".format(max_episode_step))
    print('seed: {}'.format(seed))
    # Set random seed
    seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env_evaluate.reset(seed=seed)
    # -----------------------------hyper parameter--------------------------------------
    max_train_steps = 1e6  # Maximum number of training steps
    first_eval_step = 1
    evaluate_freq = 2e3  # Evaluate the policy every 'evaluate_freq' steps
    buffer_length = 100000
    predictive_step = 2  # one trajectory is then st,s(t+1),s(t+2),....s(t+predictive); minimum value 1 and conventional
    # ddpg is applied
    hid_width = 256
    tau = 0.005
    lr_a = 4e-4
    lr_c = 4e-4
    gamma = 0.99
    save_interval = 1e5
    batch_size = 128
    alpha = min(1 / action_dim, 0.12)

    # ----------------------------------------------------------------------------------
    # state_dim, action_dim, max_action, lr, gamma, tau, batch_size, buffer
    max_action = float(env.action_space.high[0])
    buffer = ReplayBuffer(capacity=buffer_length)
    agent = MSSAC(state_dim=state_dim,
                  action_dim=action_dim,
                  max_action=max_action,
                  lr=(lr_a, lr_c),
                  gamma=gamma,
                  tau=tau,
                  batch_size=batch_size,
                  hid_size=hid_width,
                  buffer=buffer,
                  adaptive_alpha=True,
                  alpha=alpha,
                  pre_steps=predictive_step
                  )
    total_steps = 0
    evaluate_num = 0  # Record the number of evaluations

    timenow = str(datetime.now()).split('.')[0].replace(':', '_')
    p = '{}'.format('MSSAC action loaded ' + BriefEnvName[env_index] + '_') + timenow
    wdir = os.path.join('runs', BriefEnvName[env_index], 'MSSAC action loaded {}steps'.format(predictive_step))
    writer_path = os.path.join(wdir, p + '.csv')
    if not os.path.exists(wdir):
        os.makedirs(wdir, exist_ok=True)
    temp_record_dict = {}
    # ------------------------------------------------------------------------------------
    while total_steps < max_train_steps:
        episode_steps = 0
        s, info = env.reset()
        done = False
        t = False
        agent.I = 1
        temp_buffer = Tempbuffer(capacity=predictive_step)
        while not ((done or t) or episode_steps > max_episode_step):
            episode_steps += 1
            if total_steps < 10000:
                a = env.action_space.sample()  # warm-up to alleviate the impact from random seed
            else:
                a = agent.select_action(s, deterministic=False)
            s_, r, done, t, _ = env.step(a)
            r = reward_shaping(r, s_, env_index)
            # When dead or win or reaching the max_epsiode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if (done or t) and episode_steps != max_episode_step:
                dw = True
            else:
                dw = False
            temp_buffer.append(s, a, r, s_, dw)
            if len(temp_buffer) == predictive_step:
                buffer.add(temp_buffer)
            total_steps += 1
            if len(buffer) > batch_size:
                agent.train()
            s = s_
            if total_steps % evaluate_freq == 0 and total_steps > first_eval_step:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent, max_episode_step)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                temp_record_dict[total_steps] = evaluate_reward
                # Save the rewards
            if total_steps % save_interval == 0 and total_steps > first_eval_step:
                # torch.save(agent.critic.state_dict(), "./model/{}_critic_{}.pth".format(p, total_steps))
                # torch.save(agent.actor.state_dict(), "./model/{}_actor_{}.pth".format(p, total_steps))
                record(writer_path, temp_record_dict)
                temp_record_dict.clear()
        for i in range(predictive_step-1):
            temp_buffer.append(s, a, 0, s_, True)
            buffer.add(temp_buffer)
        # if out == 1:
        #     break


if __name__ == '__main__':
    s = 10088
    for i in range(s, s + 40, 2):
        main(seed=i)
