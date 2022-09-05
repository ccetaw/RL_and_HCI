import gym
from gym import spaces
import numpy as np
from Interface import Interface
from utils import (
    compute_stochastic_position,
    minjerk_trajectory,
    compute_width_distance_fast,
    jerk_of_minjerk_trajectory
    )

class ButtonPanel(gym.Env, Interface):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 60}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, config) -> None:
        super().__init__(config)
        self.action_space = spaces.Discrete(self.n_buttons)
        obs_dict = {
            'current_pattern': spaces.MultiBinary(self.n_buttons),
            'goal_pattern': spaces.MultiBinary(self.n_buttons)
        }
        self.observation_space = spaces.Dict(obs_dict)
        self.state = {
            'current_pattern': spaces.MultiBinary(self.n_buttons),
            'goal_pattern': spaces.MultiBinary(self.n_buttons),
            'cursor_position': spaces.Box(low=0., high=1., shape=(2,)),
            'panel': self.buttons
        }

        self.counter = 0
        self.done = False
        self.add = {}
        self.dt = 0.01

        self.window = None
        self.clock = None
        self.isopen = True

    def get_obs(self):
        obs = {}
        for key in self.observation_space:
            obs[key] = self.state[key]
        return obs

    def step(self, action, after_train=False):
        if after_train:
            self.add['previous_pattern'] = self.state['current_pattern'].copy()
            self.state['target'] = self.normalized_button_position(self.buttons[action]) + 0.5 * self.normalized_button_size(self.buttons[action])
            sigma, _ = compute_width_distance_fast(self.state['cursor_position'], self, action)
            self.add['move_from'] = self.state['cursor_position']
            self.add['move_to'], self.add['move_time'] = compute_stochastic_position(self.state['target'], self.state['cursor_position'], sigma)
            self.add['jerk'] = jerk_of_minjerk_trajectory(self.add['move_time'], self.add['move_from'], self.add['move_to'])
            in_buttons = self.check_within_button(self.add['move_to'])
            for in_button in in_buttons:
                self.press_button(in_button)
            self.state['current_pattern'] = self.button_pattern()
        else:
            self.press_button(action)
            self.state['current_pattern'] = self.button_pattern()
        reward = -1
        self.counter += 1
        if np.array_equal(self.state['current_pattern'], self.state['goal_pattern']):
            self.done = True
            reward += self.n_buttons * 2
        elif self.counter >= self.n_buttons * 2:
            self.done = True
            reward -= self.n_buttons * 2
        
        return self.get_obs(), reward, self.done, {}

    def reset(self, after_train=False, start_pattern=None, goal_pattern=None):
        if after_train:
            if start_pattern is not None and goal_pattern is not None:
                self.state['current_pattern'] = start_pattern
                self.state['goal_pattern'] = goal_pattern
                self.set_button_pattern(self.state['current_pattern'])
        else:
            self.state['current_pattern'] = self.sample_possible_pattern()
            self.state['goal_pattern'] = self.sample_possible_pattern()
            while np.array_equal(self.state['current_pattern'], self.state['goal_pattern']):
                self.state['goal_pattern'] = self.sample_possible_pattern()
            self.set_button_pattern(self.state['current_pattern'])

        self.state['cursor_position'] = np.array([0.,0.])
        self.counter = 0
        self.done = False

        return self.get_obs()

    def render(self, show_grid=True):
        import pygame
        from pygame import gfxdraw

        if self.state is None:
            return None

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        time = 0
        arrived = False

        while not arrived:
            if time + self.dt < self.add['move_time']:
                time += self.dt
            else:
                time = self.add['move_time']
                arrived = True
    