import math

from gym import RewardWrapper

BASE = 180
MIN_ANGLE = 30
MAX_ANGLE = 60
OPTIMAL_ANGLE = 45

V_THRESH = 0.25
COLLISION_DIST_SQ = 0.044

def dist_squared(x1, y1, x2, y2):
    return (((y2 - y1) ** 2) + ((x2 - x1) ** 2))

class ArcWrapper(RewardWrapper):

    def __init__(self, env):
        super(ArcWrapper, self).__init__(env)
        self.i = 0
        self.last_obs = [0 for _ in range(12)]
        self.new_reward = 0

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
        new_reward = self.check_reward(observation, reward)
        self.last_obs = observation
        return observation, new_reward, done, info

    def has_bounced(self, observation):
        x_agent, y_agent = observation[0:2]
        x, y, xv, yv = observation[4:8]
        xv_prev, yv_prev = self.last_obs[6:8]

        if yv == 0 or xv == 0:
            return False

        y_to_pos = yv_prev < 0 and yv > 0
        x_changed = xv_prev != xv
        agent_ball_collided = dist_squared(x_agent, y_agent, x, y) < COLLISION_DIST_SQ

        return y_to_pos or (x_changed and agent_ball_collided)

    def should_add_reward(self, observation):
        x = observation[4]
        if x <= 0:
            return False

        has_bounced = self.has_bounced(observation)
        # if has_bounced:
        #     self.debug('bounce!', observation, print_obs=True)
        if not has_bounced:
            return False

        # self.debug('Should add reward!')
        return True

    def change_new_reward(self, observation):
        x, y, xv, yv = observation[4:8]
        x_prev, y_prev, xv_prev, yv_prev = self.last_obs[4:8]

        self.new_reward = 0

        angle = math.atan2(yv, xv) * 180 / math.pi
        # self.debug('bounce angle ' + str(angle))

        min_angle = BASE - MAX_ANGLE
        max_angle = BASE - MIN_ANGLE

        if min_angle <= angle <= max_angle:
            self.new_reward += 0.1
            optimal_angle = BASE - OPTIMAL_ANGLE
            angle_diff = abs(optimal_angle - angle)
            self.new_reward -= (angle_diff / 1000)
            ball_v = ((xv ** 2) + (yv ** 2)) ** 0.5
            # self.debug('ball_v ' + str(ball_v))
            if ball_v < V_THRESH:
                self.new_reward *= 0.5

    def check_reward(self, observation, reward):
        if self.should_add_reward(observation):
            self.change_new_reward(observation)
            return self.reward(reward)
        else:
            return reward

    def reward(self, reward):
        increased = reward + self.new_reward
        # if self.new_reward > 0:
        #     self.debug('rewarding ' + str(increased))
        return increased