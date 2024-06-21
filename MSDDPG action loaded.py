import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
import csv
import os
from datetime import datetime
from collections import deque
from torch import optim

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
            at
            rt+1   rt+2 rt+3 rtt+4 ...rt+n
                                       dw_n
        When updating St, only the action taken at St is evaluated, other action will be evaluated when its state
        is St. For dw, if St+n is collected, then it means former states are not end states.
        '''
        s = np.array(list(tempbuffer.bs))
        a = np.array(list(tempbuffer.ba))
        r = np.array(list(tempbuffer.br))
        dw = np.array((tempbuffer.bdw)[-1])
        self.buffer.append((s, a, r, dw))

    def sample(self, batch_size):
        s, a, r, d = zip(*self.rng.sample(self.buffer, batch_size))
        return np.stack(s), np.stack(a), np.array(r), np.stack(d)

    def __len__(self):
        return len(self.buffer)


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


class Actor(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class Critic(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) 
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class MSDDPG:
    def __init__(self, state_dim, action_dim, max_action, lr, gamma, tau, batch_size, hid_size, buffer, random_process,
                 n_seq):
        self.actor = Actor(state_dim, hid_size, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, hid_size, action_dim, max_action).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr[0])

        self.critic = Critic(state_dim, hid_size, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, hid_size, action_dim).to(DEVICE)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr[1], weight_decay=0.01)
        # Initialize biases (if needed) from the same distribution
        self.slen = n_seq
        # Initialize target networks with the same weights as the online networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.buffer = buffer
        self.batch_size = batch_size
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

        self.target_update_count = 0
        self.voperator = torch.from_numpy(
            np.array([1 / self.slen * self.gamma ** (i + 1) for i in range(self.slen)])).float().to(DEVICE)

        self.roperator = torch.from_numpy(
            np.array([1 / self.slen * (self.slen - i) * self.gamma ** i for i in range(self.slen)])).float().to(DEVICE)

        # Ornstein-Uhlenbeck process for exploration noise
        mu, theta, sigma = random_process
        self.noise = OrnsteinUhlenbeckProcess(action_dim, mu, theta, sigma)

    def select_action(self, state, deterministic=0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        # Add exploration noise using the Ornstein-Uhlenbeck process
        action += self.noise.sample() * (1 - deterministic)
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def train(self):

        # when predictive step is n, it means we will consider n actions from state st,
        # so we should have n+1 state intotal and n action ,n reward.
        s, a, r, dw = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(DEVICE)  # (Batchsize,n_prediction+1,state_dim)

        # we need only need to evaluate the action taken at state t.
        a = torch.FloatTensor(a).to(DEVICE)  # (Batchsize,ac_dim)

        r = torch.FloatTensor(r).to(DEVICE)  # (Batchsize,n_prediction)

        not_d = 1 - torch.FloatTensor(dw).to(DEVICE)  # (Batchsize,n_prediction,n_prediction)
        # we change done flag into a Antidiagonal Matrix form. Since the antidiagonal in our voperator
        # indicates the coefficient of the value funuction of the last state.

        # Update critic
        # DDPG is deterministic, so instead of predicting the next state and next action
        # we can just predict next q value.

        cq1 = self.critic(s[:, 0, :], a[:, 0, :])  # (batch, npre)
        with torch.no_grad():
            matrix = torch.zeros((self.batch_size, self.slen)).to(DEVICE)

            for j in range(self.slen):
                if j != self.slen - 1:
                    matrix[:, j] = self.critic_target(s[:, j + 1, :], a[:, j + 1, :]).squeeze(1)
                else:
                    matrix[:, j] = self.critic_target(s[:, j + 1, :], self.actor_target(s[:, j + 1, :])).squeeze(1) * not_d

            res = torch.matmul(matrix, self.voperator)
            res += (r * self.roperator).sum(dim=1)
        critic_loss = torch.nn.functional.mse_loss(cq1.squeeze(1), res)

        self.critic_optimizer.zero_grad()

        critic_loss.backward()
        self.critic_optimizer.step()

        cq1 = self.critic(s[:, 0, :], self.actor(s[:, 0, :]))
        actor_loss = -cq1.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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
            a = agent.select_action(s, deterministic=1)  # We use the deterministic policy during the evaluating
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
    lr_a = 3e-4
    lr_c = 3e-4
    gamma = 0.99
    save_interval = 1e5
    batch_size = 100
    mu, theta, sigma = 0, 0.2, 0.3
    # ----------------------------------------------------------------------------------
    # state_dim, action_dim, max_action, lr, gamma, tau, batch_size, buffer
    max_action = float(env.action_space.high[0])
    buffer = ReplayBuffer(capacity=buffer_length)
    agent = MSDDPG(state_dim=state_dim,
                   action_dim=action_dim,
                   max_action=max_action,
                   lr=(lr_a, lr_c),
                   gamma=gamma,
                   tau=tau,
                   batch_size=batch_size,
                   hid_size=hid_width,
                   buffer=buffer,
                   random_process=(mu, theta, sigma),
                   n_seq=predictive_step,
                   )
    total_steps = 0
    evaluate_num = 0  # Record the number of evaluations
    timenow = str(datetime.now()).split('.')[0].replace(':', '_')
    p = '{}'.format('MSDDPG action loaded ' + BriefEnvName[env_index] + '_') + timenow
    wdir = os.path.join('runs', BriefEnvName[env_index], 'MSDDPG action loaded {}steps'.format(predictive_step))
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
                a = env.action_space.sample()
            else:
                a = agent.select_action(s)
            s_, r, done, t, _ = env.step(a)
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
    s = 10000
    for i in range(s, s + 30, 1):
        main(seed=i)
