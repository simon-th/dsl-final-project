import gym
import slimevolleygym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

ɣ = 0.99
target_update_freq = 1
N_STEPS = 10000

MAX_EPISODES = 100

SEED = 2020
LR_BEGIN = 0.005
LR_END = 0.005

EPS_BEGIN = 1
EPS_END = 0.05

env = gym.make("CartPole-v1")
# env = gym.make("MountainCar-v0")

env.seed(SEED)

# action_space = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1],
#                 3: [1,0,1], 4: [0,1,1], 5: [0,0,0]}

def get_action_from_policy(policy_weights, state, ε):
    if t <= N_STEPS:
        ε = EPS_BEGIN - t*(EPS_BEGIN - EPS_END)/N_STEPS
    else:
        ε = EPS_END

    if t <= N_STEPS:
        𝛼 = LR_BEGIN - t*(LR_BEGIN - LR_END)/N_STEPS
    else:
        𝛼 = LR_END

    if np.random.uniform() >= ε:
        i = np.argmax(np.matmul(policy_weights.T, state))
    else:
        i = np.random.randint(0,2)
    out = i
    return out

class ReplayBuffer():
    def __init__(self, len):
        self.replay_buffer = deque(maxlen=len)

    def push(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

    def get_gradient(self, observation, w, w_target):
        state, a, sn, reward, done = observation

        # print(state, a, sn, state_next)
        q_values = np.matmul(w.T, state)
        q_values_next = np.matmul(w_target.T, state_next)

        state = np.reshape(state, (4,1))

        target = q_values.copy()
        if not done:
            target[action] = reward + ɣ*np.max(q_values_next)
        else:
            target[action] = reward

        loss = target - q_values
        loss = np.reshape(loss, (1, 2))
        gradient = 𝛼 * np.matmul(state, loss)

        norm = np.linalg.norm(gradient)
        if norm > 10:
            gradient *= 10 / norm;

        return gradient

    def replay(self, size, w, w_target):
        length = len(self.replay_buffer)
        if length < size:
            batch = random.sample(list(self.replay_buffer), (len(self.replay_buffer) + 1)//2)
        elif size == 1:
            batch = [self.replay_buffer[-1]]
        else:
            batch = random.sample(list(self.replay_buffer), size)

        gradient = 0
        for observation in batch:
            gradient += self.get_gradient(observation, w, w_target)

        w += gradient / len(batch)
        return w

    def clear(self):
        self.replay_buffer.clear()


if __name__ == "__main__":
    # w = np.zeros((4,2))
    w = np.array([[-2.185, -2.165],
 [-0.323, -0.333],
 [ 0.113,  0.382],
 [-0.059,  0.098]])
    w_target = np.zeros((4,2))
    t = 0
    𝛼 = LR_BEGIN
    ε = EPS_BEGIN

    replay_buffer = ReplayBuffer(50)

    plot_episodes = []
    plot_rewards = []
    for episode in range(MAX_EPISODES):
        state_current = env.reset()
        total_reward = 0
        done = False

        while not done:
            t += 1

            action = get_action_from_policy(w, state_current, ε)
            state_next, reward, done, info = env.step(action)

            # replay_buffer.push(state_current, action, state_next, reward, done)
            # w = replay_buffer.replay(20, w, w_target)

            state_current = state_next
            total_reward += reward

            # print(w)
            if t % 100 == 0:
                w_target = w.copy()
            # env.render()

        print("Episode number: " + str(episode) + "; Total Reward: " + str(total_reward) + "; t: " + str(t))
        plot_rewards.append(total_reward)
        plot_episodes.append(episode)

    env.close()
    print(w)
    print("Average reward: ", np.mean(plot_rewards))
    plt.plot(plot_episodes, plot_rewards)
    plt.title("LFA with target Q, experience replay, batch stochastic updating")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()
