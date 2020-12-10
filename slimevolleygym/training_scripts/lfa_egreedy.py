import gym
import numpy as np
import matplotlib.pyplot as plt

ɣ = 0.99
𝛼 = 0.01
EPS_BEGIN = 1
EPS_END = 0.05
N_STEPS = 1000
MAX_EPISODES = 5000
SEED = 2020

env = gym.make("CartPole-v1")
env.seed(SEED)
np.random.seed(SEED)


def get_action_from_policy(policy_weights, state, t):
    if t <= N_STEPS:
        ε = EPS_BEGIN - t*(EPS_BEGIN - EPS_END)/N_STEPS
    else:
        ε = EPS_END

    if np.random.uniform() >= ε:
        return np.argmax(np.matmul(policy_weights.T, state))
    return np.random.choice([0, 1])


def compute_gradient(w, action, state_current, state_next, reward, done):
    q_values = np.matmul(w.T, state_current)

    td_target_q = q_values.copy()
    td_target_q[action] = reward
    if not done:
        td_target_q[action] += ɣ*np.max(np.matmul(w.T, state_next))

    loss = td_target_q - q_values

    loss = np.reshape(loss, (1, 2))
    grad_Q = np.reshape(state_current, (4, 1))
    gradient = 𝛼 * np.matmul(grad_Q, loss)

    norm = np.linalg.norm(gradient)
    if norm > 10:
        gradient *= 10 / norm;

    return gradient



if __name__ == "__main__":
    w = np.random.uniform(0, 1, (4,2))
    # w = np.array([[ 1.37214434,  0.20685448], [ 5.23775575, -4.74334533], [ 1.28942481, -0.92188776], [-3.06873135,  3.28309075]])
    t = 0
    plot_episodes = []
    plot_rewards = []

    for episode in range(MAX_EPISODES):
        state_current = env.reset()
        total_reward = 0

        done = False
        while not done:
            t += 1

            action = get_action_from_policy(w, state_current, t)
            state_next, reward, done, info = env.step(action)

            w += compute_gradient(w, action, state_current, state_next, reward, done)

            state_current = state_next
            total_reward += reward
            # env.render()

        print("Episode number: " + str(episode) + "; Total Reward: " + str(total_reward) + "; t: " + str(t))
        plot_rewards.append(total_reward)
        plot_episodes.append(episode)

    env.close()
    print("Weights of network", w)
    print("Average reward: ", np.mean(plot_rewards))
    plt.plot(plot_episodes, plot_rewards)
    plt.title("LFA ε-greedy: Total Reward During Training")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()
