import gym
from gym import spaces
import numpy as np
from Interface import Interface
from utils import (
    compute_stochastic_position,
    minjerk_trajectory,
    compute_width_distance_fast,
    jerk_of_minjerk_trajectory,
    trial_name_string
    )

class ButtonPanel(gym.Env, Interface):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 15}
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
            'previous_patttern': spaces.MultiBinary(self.n_buttons),
            'current_pattern': spaces.MultiBinary(self.n_buttons),
            'goal_pattern': spaces.MultiBinary(self.n_buttons),
            'cursor_position': spaces.Box(low=0., high=1., shape=(2,)),
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

    def get_reward(self):
        reward = 0
        reward -= 1 # Penalize every step
        flag = True
        done = False

        if self.counter >= self.n_buttons * 2: # If episode lasts too long, terminate with big penalty
            reward -= self.n_buttons * 2
            done = True
            return reward, done

        # We don't care about status of central button(s)
        for idx in self.button_group['normal']:
            if self.buttons[idx].on != self.state['goal_pattern'][idx]:
                flag = False
        for idx in self.button_group['mutual_exclu']:
            if self.buttons[idx].on != self.state['goal_pattern'][idx]:
                flag = False
        if flag:
            reward += self.n_buttons * 2
            done = True
        return reward, done
        

    def step(self, action, after_train=False):
        if after_train:
            self.state['previous_pattern'] = self.state['current_pattern'].copy()
            self.state['target'] = self.normalized_button_position(self.buttons[action]) + 0.5 * self.normalized_button_size(self.buttons[action])
            width, _ = compute_width_distance_fast(self.state['cursor_position'], self, action)
            sigma = 1/4 * width
            self.add['move_from'] = self.state['cursor_position']
            self.add['move_to'], self.add['move_time'] = compute_stochastic_position(self.state['target'], self.state['cursor_position'], sigma)
            self.add['jerk'] = jerk_of_minjerk_trajectory(self.add['move_time'], self.add['move_from'], self.add['move_to'])
            in_buttons = self.check_within_button(self.add['move_to'])
            for in_button in in_buttons:
                self.press_button(in_button)
            self.state['current_pattern'] = self.button_pattern()
            self.state['cursor_position'] = self.add['move_to']
        else:
            self.press_button(action)
            self.state['current_pattern'] = self.button_pattern()
        self.counter += 1
        reward, self.done = self.get_reward()
        
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

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        font = pygame.font.Font('freesansbold.ttf', 16)
        canvas.fill((255, 255, 255))

        if show_grid:
            # Draw the margin
            pygame.draw.line(
                    canvas,
                    0,
                    (0, self.margin_size),
                    (self.screen_width, self.margin_size),
                    width=3,
                )
            pygame.draw.line(
                    canvas,
                    0,
                    (0, self.screen_height - self.margin_size),
                    (self.screen_width, self.screen_height - self.margin_size),
                    width=3,
                )
            pygame.draw.line(
                    canvas,
                    0,
                    (self.margin_size, 0),
                    (self.margin_size, self.screen_height),
                    width=3,
                )
            pygame.draw.line(
                    canvas,
                    0,
                    (self.screen_width - self.margin_size, 0),
                    (self.screen_width - self.margin_size, self.screen_height),
                    width=3,
                )

            # Draw the grid
            for y in range(self.grid.shape[1]):
                pygame.draw.line(
                    canvas,
                    0,
                    (self.margin_size, self.margin_size + self.grid_size * y),
                    (self.screen_width - self.margin_size, self.margin_size + self.grid_size * y),
                    width=2,
                )
            for x in range(self.grid.shape[0]):
                pygame.draw.line(
                    canvas,
                    0,
                    (self.margin_size + self.grid_size * x, self.margin_size),
                    (self.margin_size + self.grid_size * x, self.screen_height - self.margin_size),
                    width=2,
                )


        # Draw the previous button pattern
        self.set_button_pattern(self.state['previous_pattern'])
        for button in self.buttons.values():
            if button.on:
                pygame.draw.rect(
                    canvas,
                    (255, 165, 0),
                    pygame.Rect(
                        button.position,
                        (button.size[0], button.size[1]),
                    ),
                )
            else:
                pygame.draw.rect(
                    canvas,
                    (200, 200, 200),
                    pygame.Rect(
                        button.position,
                        (button.size[0], button.size[1]),
                    ),
                )
            button_id = font.render(button.id, True, (150, 0, 200))
            button_type = font.render(button.type[0], True, (0, 178, 238))
            canvas.blit(button_id, button.position + 5)
            canvas.blit(button_type, [button.position[0] + button.size[0] - 15, button.position[1] + 5])

        # Draw the goal button pattern
        self.set_button_pattern(self.state['goal_pattern'])
        for button in self.buttons.values():
            if button.on:
                pygame.draw.rect(
                    canvas,
                    (255, 0, 0),
                    pygame.Rect(
                        button.position,
                        (button.size[0], button.size[1]),
                    ),
                    width=3
                )

        # Draw the cursor
        x = self.add['move_from'][0] * self.screen_width
        y = self.add['move_from'][1] * self.screen_height
        pygame.draw.circle(canvas, color=(0,0,255), center=[x, y], radius=10)

        # Draw the target
        x = self.add['move_to'][0] * self.screen_width
        y = self.add['move_to'][1] * self.screen_height
        pygame.draw.circle(canvas, color=(255,0,0), center=[x, y], radius=10)

        # Move the cursor
        while not arrived:
            if time + self.dt < self.add['move_time']:
                time += self.dt
            else:
                time = self.add['move_time']
                arrived = True
            # print(time)
            cursor_position = minjerk_trajectory(time, self.add['move_time'], self.add['move_from'] ,self.add['move_to'])
            x = cursor_position[0] * self.screen_width
            y = cursor_position[1] * self.screen_height
            # print(x, y)
            # cursor.move(x, y)
            pygame.draw.circle(canvas, color=(126,192,238), center=[x, y], radius=10)
            # print(cursor.center)

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        # Draw the current button pattern
        self.set_button_pattern(self.state['current_pattern'])
        for button in self.buttons.values():
            if button.on:
                pygame.draw.rect(
                    canvas,
                    (255, 165, 0),
                    pygame.Rect(
                        button.position,
                        (button.size[0], button.size[1]),
                    ),
                )
            else:
                pygame.draw.rect(
                    canvas,
                    (200, 200, 200),
                    pygame.Rect(
                        button.position,
                        (button.size[0], button.size[1]),
                    ),
                )
            button_id = font.render(button.id, True, (150, 0, 200))
            button_type = font.render(button.type[0], True, (0, 178, 238))
            canvas.blit(button_id, button.position + 5)
            canvas.blit(button_type, [button.position[0] + button.size[0] - 15, button.position[1] + 5])

    def close(self):
        import pygame
        pygame.display.quit()
        pygame.quit()
        self.isopen = False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Show a window of the system")
    group.add_argument("--dry_train", action="store_true", help="Train the agent")
    group.add_argument("--show", action="store_true", help="Show the trained agent")
    args = parser.parse_args()

    env_config = {
        'n_buttons': 7,
        'random': False
    }

    if args.demo:
        env = ButtonPanel(env_config)

        for _ in range(5):
            env.reset(after_train=True)
            for _ in range(10):
                env.step(env.action_space.sample(), after_train=True)
                env.render()
        env.close()

    if args.dry_train:
        import ray
        from ray import tune
        from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
        
        ray.init(local_mode=False, num_cpus=24, num_gpus=0)
        stop = {
            "training_iteration": 300,
        }
        config = {
            "num_workers": 3,
            "env": ButtonPanel,
            "env_config": env_config,
            "gamma": 0.9,
            "framework": "torch",
            "log_level": "CRITICAL",
            "num_gpus": 0,
            "seed": 31415,
        }
        config = {**DEFAULT_CONFIG, **config}
        results = tune.run(
            PPOTrainer,
            trial_name_creator=trial_name_string,
            stop=stop,
            config=config,
            local_dir="./trained_user",
            verbose=1,
            checkpoint_freq=50,
            checkpoint_at_end=True,
            num_samples=1
        )
        ray.shutdown()

    if args.show:
        from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
        env = ButtonPanel(env_config)
        config = {
            "num_workers": 3,
            "env": ButtonPanel,
            "env_config": env_config,
            "gamma": 0.9,
            "framework": "torch",
            "log_level": "CRITICAL",
            "num_gpus": 0,
            "seed": 31415,
        }
        config = {**DEFAULT_CONFIG, **config}
        agent = PPOTrainer(config=config)
        agent.restore("./trained_user/PPOTrainer_2022-09-07_09-55-46/7b6bb_00000-n_buttons_7-random_False_0_2022-09-07_09-55-47/checkpoint_000300/checkpoint-300")
        for _ in range(5):
            env.reset(after_train=True)
            while not env.done:
                action = agent.compute_single_action(env.get_obs(), unsquash_action=True)
                env.step(action, after_train=True)
                env.render()
        env.close()

