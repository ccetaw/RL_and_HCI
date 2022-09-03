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
group.add_argument("--show", action="store_true", help="Show the trained agent")
args = parser.parse_args()

class WindyGrid(gym.Env):

    metadata = {"render_mode": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, config):
        super().__init__()
        self.size = config["size"] # size should be at least 5 
        self.window_size = 1024
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "agent_location": spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=int),
                "target_location": spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=int)
            }
        )
        self._action_mapping = {
            0: np.array([1, 0]),    # down
            1: np.array([0, 1]),    # right
            2: np.array([-1,0]),    # up
            3: np.array([0,-1])     # left
        }
        self.state = {
            "agent_location": np.random.randint(low=0, high=self.size, size=2),
            "target_location": np.random.randint(low=0, high=self.size, size=2)
        }
        self.wind = np.random.randint(low=-1, high=2, size=self.size)
        self.wind[0] = 0
        self.wind[-1] = 0  # This is to make sure that the problem is solvable
        self.counter = 0
        self.done = False
        self.window = None
        self.clock = None


    def get_obs(self):
        obs = {}
        for key in self.observation_space:
            obs[key] = self.state[key]
        return obs
    
    def step(self, action):
        displacement = self._action_mapping[action] + np.array([0, self.wind[self.state["agent_location"][1]]])
        self.state["agent_location"] = np.clip(
            self.state["agent_location"] + displacement, 0, self.size-1
        )
        reward = -1 # Every step is penalized
        if np.array_equal(self.state["agent_location"], self.state["target_location"]):
            self.done = True
        elif self.counter >= 20:
            self.done = True
            reward = -2*self.counter # If the episode takes too much time, it's terminated automatically and a big penalty is assigned
        observation = self.get_obs()
        self.counter += 1

        return observation, reward, self.done, {}

    def reset(self):
        self.state["agent_location"] = np.random.randint(low=0, high=self.size, size=2)
        self.state["target_location"] = np.random.randint(low=0, high=self.size, size=2)
        while np.array_equal(self.state["agent_location"], self.state["target_location"]):
            self.state["target_location"] = np.random.randint(low=0, high=self.size, size=2)
        self.counter = 0
        self.done = False
        observation = self.get_obs()
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
                pix_square_size * self.state["target_location"][::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.state["agent_location"][::-1] + 0.5) * pix_square_size,
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
        env_config = {
            "window_size": 512,
            "size": 10
        }
        env = WindyGrid(config=env_config)
        config = {
            "env": WindyGrid,  # or "corridor" if registered above
            "env_config": env_config,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_workers": 1,  # parallelism
            "framework": "torch",
            "gamma": 0.9
        }
        stop = {
            "training_iteration": 1000,
            # "timesteps_total": 100000,
            # "episode_reward_mean": -10,
        }
        config = {**ppo.DEFAULT_CONFIG, **config}
        trainer = ppo.PPOTrainer(config=config)
        # run manual training loop and print results after each iteration
        for _ in range(stop["training_iteration"]):
            result = trainer.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            # if (
            #     result["timesteps_total"] >= stop["timesteps_total"] or
            #     result["episode_reward_mean"] >= -stop["episode_reward_mean"]
            # ):
            #     break

        trainer.save("./windy_grid_trained")
        
        for i in range(10):
            env.reset()
            while not env.done:
                env.render()
                action = trainer.compute_single_action(env.get_obs())
                env.step(env.action_space.sample())
        env.close()

    if args.show:
        env_config = {
            "size": 10
        }
        env = WindyGrid(config=env_config)
        config = {
            "env": WindyGrid,  # or "corridor" if registered above
            "env_config": env_config,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_workers": 1,  # parallelism
            "framework": "torch",
            "gamma": 0.9
        }
        config = {**ppo.DEFAULT_CONFIG, **config}
        agent = ppo.PPOTrainer(config=config)
        agent.restore("./windy_grid_trained")

        


        

    
        