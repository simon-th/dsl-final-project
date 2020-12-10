import random
import numpy as np
import gym
import slimevolleygym

class ObservationWrapperMulti(gym.ObservationWrapper):
    def __init__(self, env, epsilon = 0.2):
        super().__init__(env)
        self.epsilon = epsilon

    def step(self, action, otherAction = None):
        observation, reward, done, info = self.env.step(action, otherAction)
        info['otherObs'] = self.observation(info['otherObs'])
        return self.observation(observation),reward,done,info

    def observation(self, obs):
        # if random.random() < self.epsilon:
            #print("Observation Wrapper!")
            # new_obs = obs
        noise = np.random.normal(0,0.1,12)
        new_obs = obs + noise
            #new_obs = obs + noise
        return new_obs
        # return obs

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, epsilon = 0.2):
        super(ObservationWrapper, self).__init__(env)
        self.epsilon = epsilon

    def step(self, action, otherAction = None):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation),reward,done,info

    def observation(self, obs):
        # if random.random() < self.epsilon:
            #print("Observation Wrapper!")
            # new_obs = obs
        noise = np.random.normal(0,0.1,12)
        new_obs = obs + noise
            #new_obs = obs + noise
        return new_obs
        # return obs

class ObservationWrapperOne(gym.ObservationWrapper):
    def __init__(self, env, epsilon = 0.2):
        super(ObservationWrapper, self).__init__(env)
        self.epsilon = epsilon

    def step(self, action, otherAction = None):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation),reward,done,info

    def observation(self, obs):
        if random.random() < self.epsilon:
            #print("Observation Wrapper!")
            new_obs = obs
            noise = np.random.normal(0,1,12)
            new_obs = new_obs + noise
            #new_obs = obs + noise
            return new_obs
        return obs

class ObservationWrapperHalf(gym.ObservationWrapper):
    def __init__(self, env, epsilon = 0.2):
        super(ObservationWrapperHalf, self).__init__(env)
        self.epsilon = epsilon

    def step(self, action, otherAction = None):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation),reward,done,info

    def observation(self, obs):
        if random.random() < self.epsilon:
            #print("Observation Wrapper!")
            #new_obs = obs
            noise = np.random.normal(0,.5,12)
            #new_obs = new_obs + noise
            new_obs = obs + noise
            return new_obs
        return obs