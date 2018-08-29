import random

import gym
import marlo

from .utils import rgb2gray, imresize


class Environment(object):

    def __init__(self, config):
        self.action_repeat = config.action_repeat
        self.random_start = config.random_start
        self.display = config.display
        self.dims = (config.screen_width, config.screen_height)
        self._screen = None
        self.terminal = True
        self.reward = 0

    def new_game(self, from_random_game=False):
        # if self.lives == 0:
        self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in range(random.randint(0, self.random_start - 1)):
            self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @property
    def screen(self):
        # TODO this may not work if imresize comes from cv2 instead of skimage in utils
        return imresize(rgb2gray(self._screen) / 255., self.dims)
        # return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def lives(self):
        return self.env.env.ale.lives()

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def render(self):
        if self.display:
            self.env.render()

    def act(self, action, is_training=True):
        pass

    def after_act(self, action):
        self.render()


class MarloEnvironment(Environment):

    def __init__(self, config):
        super().__init__(config)

        self._client_pool = marlo.launch_clients(1)
        join_tokens = marlo.make(config.env_name,
                                 params={"client_pool": self._client_pool,
                                         "videoResolution": [config.screen_width, config.screen_height]})
        self.env = marlo.init(join_tokens[0])

    # TODO if this code works unchanged, find a way so that MarloEnvironment can inherit from GymEnvironment
    def act(self, action, is_training=True):
        cumulative_reward = 0
        # TODO lives may not work here for Marlo
        start_lives = self.lives

        for _ in range(self.action_repeat):
            self._step(action)
            cumulative_reward = cumulative_reward + self.reward

            if is_training and start_lives > self.lives:
                cumulative_reward -= 1
                self.terminal = True

            if self.terminal:
                break

        self.reward = cumulative_reward

        self.after_act(action)
        return self.state


class GymEnvironment(Environment):

    def __init__(self, config):
        super().__init__(config)
        self.env = gym.make(config.env_name)

    def act(self, action, is_training=True):
        cumulative_reward = 0
        start_lives = self.lives

        for _ in range(self.action_repeat):
            self._step(action)
            cumulative_reward = cumulative_reward + self.reward

            if is_training and start_lives > self.lives:
                cumulative_reward -= 1
                self.terminal = True

            if self.terminal:
                break

        self.reward = cumulative_reward

        self.after_act(action)
        return self.state


class SimpleGymEnvironment(Environment):

    def __init__(self, config):
        super().__init__(config)
        self.env = gym.make(config.env_name)

    def act(self, action, is_training=True):
        self._step(action)

        self.after_act(action)
        return self.state
