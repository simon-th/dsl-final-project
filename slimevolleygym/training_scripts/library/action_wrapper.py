import numpy as np
import gym
import random

class RandomNoiseActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.07):
        super(RandomNoiseActionWrapper,self).__init__(env)
        self.epsilon = epsilon

    def reset(self, **kwargs):
            return self.env.reset(**kwargs)

    def step(self, action, otherAction = None):
        return self.env.step(self.action(action), otherAction)

    def reverse_action(self, action):
        raise NotImplementedError

    def action(self, action):
        if random.random() < self.epsilon:
            action = random.choice([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0]])
        return action

class BitNoiseActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(BitNoiseActionWrapper,self).__init__(env)
        self.epsilon = epsilon

    def reset(self, **kwargs):
            return self.env.reset(**kwargs)

    def step(self, action, otherAction = None):
        return self.env.step(self.action(action), otherAction)

    def reverse_action(self, action):
        raise NotImplementedError

    def action(self, action):
        if random.random() < self.epsilon:
            index = random.randint(0,2)
            noise = np.random.normal(0,1,1)
            # print('ori action: '+ str(action) + ' | index: ' + str(index) + ' | noise: ', str(noise))
            action[index] = action[index] + noise[0]
            # print('final action: ', action)
        return action

class ConstantNoiseActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ConstantNoiseActionWrapper,self).__init__(env)

    def reset(self, **kwargs):
            return self.env.reset(**kwargs)

    def step(self, action, otherAction = None):
        return self.env.step(self.action(action), otherAction)

    def reverse_action(self, action):
        raise NotImplementedError

    def action(self, action):
        noise = np.random.normal(0,0.1,3)
        # print('ori action: ' + str(action) + ' | noise: ' + str(noise))
        action = action + noise
        # print('final action: ', action)
        return action

class ConstantNoiseActionWrapperMulti(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
            return self.env.reset(**kwargs)

    def step(self, action, otherAction = None):
        return self.env.step(self.action(action), self.action(otherAction))

    def reverse_action(self, action):
        raise NotImplementedError

    def action(self, action):
        noise = np.random.normal(0,0.1,3)
        # print('ori action: ' + str(action) + ' | noise: ' + str(noise))
        action = action + noise
        # print('final action: ', action)
        return action