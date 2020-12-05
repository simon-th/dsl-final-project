import gym

class SpikeWrapper(gym.RewardWrapper):
  def __init__(self, env):
    super(SpikeWrapper, self).__init__(env)
    self.criteria = {
      "Height": 1.5,
      "HorizontalSpeed": -1.3,
    }

  def step(self, action, otherAction=None):
    observation, reward, done, info = self.env.step(action, otherAction)
    checkedreward = self.checkreward(observation, reward)
    return observation, checkedreward, done, info

  def checkreward(self, observation, reward):
    xPosBall, yPosBall, xVelBall, yVelBall = observation[4:8]
    if (yPosBall > self.criteria["Height"] and xVelBall < self.criteria["HorizontalSpeed"]):
      reward = self.reward(reward)
    return reward

  def reward(self, reward):
    return reward + 0.4
