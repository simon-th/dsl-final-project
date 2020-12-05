import gym

class spikewrapper(gym.RewardWrapper):
  def __init__(self, env):
    super(spikewrapper, self).__init__(env)
    self.criteria = {
      "Height": 1.5,
      "HorizontalSpeed": -1.3,
    }

  def step(self, action, otherAction=None):
    observation, reward, done, info = self.env.step(action, otherAction)
    checkedreward = self.checkreward(observation, reward)
    return observation, checkedreward, done, info

  def checkreward(self, observation, reward):
    # print("observation:", observation, "reward:", reward)
    xPosBall, yPosBall, xVelBall, yVelBall = observation[4:8]
    # print(xPosBall, yPosBall, xVelBall, yVelBall)
    if (yPosBall > self.criteria["Height"] and xVelBall < self.criteria["HorizontalSpeed"]):
      reward = self.reward(reward)
      print(xPosBall, yPosBall, xVelBall, yVelBall)

    return reward

  def reward(self, reward):
    if reward >= 0:
      reward += 0.4
      print("got reward")
    return reward
