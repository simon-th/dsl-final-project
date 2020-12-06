import gym

RESTING_HEIGHT = 0.15
AGENT_BALL_EUCLID_DIST = 0.044
AGENT_AGAINST_FENCE_POS = 0.2
AGENT_FENCE_RANGE = 1.0
BALL_MAX_VELOCITY = -2
SPIKE_REWARD = 0.1

class SpikeWrapper(gym.RewardWrapper):
  def __init__(self, env, printSpikes = False):
    super(SpikeWrapper, self).__init__(env)
    self.pastObservation = [0 for _ in range(12)]
    self.numSpikes = 0
    self.printSpikes = printSpikes

  def satisfiedRewardCondition(self, obs):
    xAgent, yAgent, xVelAgent, yVelAgent, xBall, yBall, xVelBall, yVelBall = obs[0:8]
    if self.ballOnAgentSide(xBall) and \
       self.ballCollidedWithAgent(obs) and \
       self.agentJumping(yAgent) and \
       self.agentCloseToFence(xAgent) and \
       self.agentMovingLeftOrTouchingFence(xAgent, xVelAgent) and \
       self.ballMovingFastEnoughLeft(xVelBall):
      return True
    else:
      return False

  def ballOnAgentSide(self, xBall):
    return xBall > 0

  def ballCollidedWithAgent(self, obs):
    pastXVelBall = self.pastObservation[6]
    pastYVelBall = self.pastObservation[7]
    xAgent, yAgent = obs[0:2]
    xBall, yBall, currXVelBall, currYVelBall = obs[4:8]

    # Ball has collided with agent if the ball's y velocity goes from neg to pos
    if (pastYVelBall < 0 and currYVelBall >= 0) or \
       ((pastXVelBall != currXVelBall) and getEuclidDistSquared(xAgent, yAgent, xBall, yBall) < AGENT_BALL_EUCLID_DIST):
      return True
    else:
      return False

  def agentJumping(self, yAgent):
    return yAgent > (RESTING_HEIGHT+0.05)

  def agentCloseToFence(self, xAgent):
    return xAgent < AGENT_FENCE_RANGE

  def agentMovingLeftOrTouchingFence(self, xAgent, xVelAgent):
    return xAgent == AGENT_AGAINST_FENCE_POS or xVelAgent < 0

  def ballMovingFastEnoughLeft(self, xVelBall):
    return xVelBall < BALL_MAX_VELOCITY

  def testCollision(self, obs):
    xAgent, yAgent, xVelAgent, yVelAgent, xBall, yBall, xVelBall, yVelBall = obs[0:8]
    if self.ballCollidedWithAgent(obs) and self.ballOnAgentSide(xBall):
      self.i += 1
      print("Ball Collided With Agent", self.i, getEuclidDistSquared(xAgent, yAgent, xBall, yBall))

  def step(self, action, otherAction=None):
    observation, reward, done, info = self.env.step(action, otherAction)
    updatedReward = self.checkForUpdatedReward(observation, reward)
    self.pastObservation = observation
    return observation, updatedReward, done, info

  def checkForUpdatedReward(self, observation, reward):
    if self.satisfiedRewardCondition(observation):
      self.numSpikes += 1
      if self.printSpikes:
        print("Spike!", self.numSpikes)
      reward = self.reward(reward)
    return reward

  def reward(self, reward):
    return reward + SPIKE_REWARD

def getEuclidDistSquared(xAgent, yAgent, xBall, yBall):
  return (xAgent - xBall)**2 + (yAgent - yBall)**2
