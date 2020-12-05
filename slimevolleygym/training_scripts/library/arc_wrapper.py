import math

from gym import RewardWrapper

BASE = 180
MIN_ANGLE = 30
MAX_ANGLE = 60
OPTIMAL_ANGLE = 45

class ArcWrapper(RewardWrapper):

    def __init__(self, env):
        super(ArcWrapper, self).__init__(env)
        self.i = 0
        self.last_obs = [0 for _ in range(12)]
        self.new_reward = 0
        self.v_thresh = 2.5

    def debug(self, message, observation=None, print_obs=False):
        self.i += 1
        print(message, self.i)
        if print_obs:
            print('-------- [ x_a    y_a    xv_a   yv_a  x_b     y_b    xv_b   yv_b   x_o    y_o    xv_o   yv_o ]')
            print('last obs', self.last_obs)
            print('curr obs', observation)
        print()


    def step(self, action, otherAction=None):
        observation, reward, done, info = self.env.step(action, otherAction)
        return observation, self.check_reward(observation, reward), done, info

    def check_reward(self, observation, reward):
        x, y, xv, yv = observation[4:8]
        x_prev, y_prev, xv_prev, yv_prev = self.last_obs[4:8]

        has_bounced = yv_prev < 0 and yv > 0
        # if has_bounced:
        #     self.debug('bounce!', observation, print_obs=True)

        # has_crossed = x <= 0 and x_prev > 0
        # if has_crossed:
        #     self.debug('crossed!', observation)

        # if has_crossed:
        #     self.last_obs = observation
        #     return self.reward(reward)

        if not has_bounced:
            self.last_obs = observation
            return reward

        self.new_reward = 0

        angle = math.atan2(yv, xv) * 180 / math.pi
        # self.debug('bounce angle ' + str(angle))
        # if x > 0:
        #     self.debug('angle: ' + str(180 - angle))

        min_angle = BASE - MAX_ANGLE
        max_angle = BASE - MIN_ANGLE

        if min_angle <= angle <= max_angle:

            ball_v = ((xv ** 2) + (yv ** 2)) ** 0.5
            # self.debug('ball_v ' + str(ball_v))

            self.new_reward = 0.5
            optimal_angle = BASE - OPTIMAL_ANGLE
            angle_diff = abs(optimal_angle - angle)
            self.new_reward -= (angle_diff / 100)

            if ball_v < self.v_thresh:
                self.new_reward *= 0.5

        self.last_obs = observation
        return self.reward(reward)

    def reward(self, reward):
        increased = reward + self.new_reward
        # if self.new_reward > 0:
        #     self.debug('rewarding ' + str(increased))
        return increased