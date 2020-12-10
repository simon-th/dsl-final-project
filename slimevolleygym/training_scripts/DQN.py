import gym
import slimevolleygym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import random
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

SEED = 2020
ɣ = 0.99
MAX_EPISODES = 500
N_STEPS = 10000

LR_BEGIN = 0.005
LR_END = 0.005

EPS_BEGIN = 1
EPS_END = 0.05


class DQN():
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(state_dim, hidden_dim),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim*2),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(hidden_dim*2, action_dim)
                    )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def print_weights(self):
        for param in self.model.parameters():
            print(param.data)

    def save_model(self, DIR):
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        last_saved = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        torch.save(self.model.state_dict(), f'{DIR}/torch_weights_{last_saved + 1}.pth')

    def load_model(self, path):
        proceed = 'y'
        try:
            self.model.load_state_dict(torch.load(f'{os.path.abspath(os.getcwd())}/{path}'))
        except Exception as e:
            print(e)
            proceed = input(f'\nFailed to load model state from {path}\nDo you wish to proceed? [y/n] ')
        return proceed


class ReplayBuffer():
    def __init__(self, len):
        self.replay_buffer = deque(maxlen=len)

    def push(self, state, reward, done, info):
        self.replay_buffer.append((state, reward, done, info))

    def update(self, observation, state_next, model):
        experience_state, reward, done, info = observation

        q_values = model.predict(experience_state)
        q_values_next = model.predict(next_state)

        target = q_values.tolist()
        target[action] = reward
        if not done:
            target[action] += ɣ*torch.max(q_values_next).item()

        model.update(experience_state, target)

    def replay(self, size, state_next, model):
        length = len(self.replay_buffer)
        batch = random.sample(list(self.replay_buffer), min(length, size))
        for obs in batch:
            self.update(obs, state_next, model)

    def clear(self):
        self.replay_buffer.clear()


def get_action_from_policy(model, state, t):
    if t <= N_STEPS:
        ε = EPS_BEGIN - t*(EPS_BEGIN - EPS_END)/N_STEPS
    else:
        ε = EPS_END

    if np.random.uniform() <= ε:
        action = env.action_space.sample()
    else:
        action = torch.argmax(model.predict(state)).item()

    return action


env = gym.make("CartPole-v1")
env.seed(SEED)
random.seed(SEED)


if __name__ == "__main__":
    replay_buffer = ReplayBuffer(500)

    t = 0
    plot_episodes = []
    plot_rewards = []

    model = DQN(env.observation_space.shape[0], env.action_space.n, 50, LR_BEGIN)

    proceed = 'y'
    if len(sys.argv) > 2 and sys.argv[1] == 'load':
        proceed = model.load_model(sys.argv[2])

    if proceed == 'y':
        try:
            for episode in range(MAX_EPISODES):
                current_state = env.reset()

                total_reward = 0
                done = False

                while not done:
                    action = get_action_from_policy(model, current_state, t)

                    next_state, reward, done, info = env.step(action)

                    replay_buffer.push(current_state, reward, done, info)
                    replay_buffer.replay(20, next_state, model)

                    current_state = next_state
                    total_reward += reward

                    t += 1
                    # env.render()

                print("Episode number: " + str(episode) + "; Total Reward: " + str(total_reward) + "; t: " + str(t))
                plot_rewards.append(total_reward)
                plot_episodes.append(episode)

            model.save_model(f'{os.path.abspath(os.path.dirname(__file__))}/../zoo/dqn')

        except KeyboardInterrupt:
            save = input(f'\nDo you want to save this model? [y/n] ')
            if save == 'y':
                model.save_model(f'{os.path.abspath(os.path.dirname(__file__))}/../zoo/dqn')

        env.close()
        model.print_weights()
        print("Average reward: ", np.mean(plot_rewards))
        plt.plot(plot_episodes, plot_rewards)
        plt.title("DQN")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.show()
