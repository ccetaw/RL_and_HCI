import gym
from gym import spaces
import pygame
import numpy as np
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
import argparse

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--demo", action="store_true", help="Show a window of the system")
group.add_argument("--train", action="store_true", help="Train the agent")
args = parser.parse_args()

class WindyGrid(gym.Env):

    metadata = {"render_mode": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, config):
        super().__init__()
        self.window_size = config["window_size"]
        self.size = config["size"]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=int),
                "target": spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=int)
            }
        )
        self._action_mapping = {
            0: np.array([1, 0]),    # down
            1: np.array([0, 1]),    # right
            2: np.array([-1,0]),    # up
            3: np.array([0,-1])     # left
        }
        self.agent_location = np.array([3,0])
        self.target_location = np.array([3,7])
        self.wind = np.array([0,0,0,1,1,1,2,2,1,0])
        self.counter = 0
        self.done = False
        self.window = None
        self.clock = None


    def _get_obs(self):
        return {"agent": self.agent_location, "target": self.target_location}
    
    def step(self, action):
        displacement = self._action_mapping[action] + np.array([self.wind[self.agent_location[1]], 0])
        self.agent_location = np.clip(
            self.agent_location + displacement, 0, self.size-1
        )
        if np.array_equal(self.agent_location, self.target_location):
            self.done = True
            reward = -self.counter
        elif self.counter >= 19:
            self.done = True
            reward = -2*self.counter
        observation = self._get_obs()

        return observation, reward, self.done, {}

    def reset(self):
        self.agent_location = np.array([3,0])
        self.target_location = np.array([3,7])
        self.counter = 0
        self.done = False
        observation = self._get_obs()
        return observation
    
    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.target_location[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    
    if args.demo:
        env_config = {
            "window_size": 512,
            "size": 10
        }
        env = WindyGrid(config=env_config)
        for i in range(5):
            env.reset()
            for j in range(10):
                env.render()
                env.step(env.action_space.sample())
        env.close()

    if args.train:
        config = {
            "env": WindyGrid,  # or "corridor" if registered above
            "env_config": {
                "window_size": 512,
                "size": 10
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_workers": 1,  # parallelism
            "framework": "torch",
            "gamma": 0.9
        }
        stop = {
            "training_iteration": 100,
            "timesteps_total": 100000,
            "episode_reward_mean": 0.1,
        }
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        # use fixed learning rate instead of grid search (needs tune)
        ppo_config["lr"] = 1e-3
        trainer = ppo.PPOTrainer(config=ppo_config, env=WindyGrid)
        # run manual training loop and print results after each iteration
        for _ in range(stop["training_iteration"]):
            result = trainer.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= stop["timesteps_total"]
                or result["episode_reward_mean"] >= stop["episode_reward_mean"]
            ):
                break

        

    
        