from gym import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from utils import policy_mapping_fn
import pygame
from pygame import gfxdraw

class Arena(MultiAgentEnv):

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 1}
    reward_range = (-float("inf"), float("inf"))
    obstacles = [[1,0], [7,0], [1,1], [5,1], [7,1], 
                [5,2], [7,2],
                [2,3], [3,3], [4,3], [5,3], [7,3], [8,3],
                [1,5], [4,5], [7,5],
                [1,6], [4,6], [7,6],
                [1,7], [4,7], [7,7],
                [1,8], [4,8], [5,8], [6,8], [7,8],
                [1,9]]  # Cells that can't be occupied

    def __init__(self, env_config):
        self.random = env_config['random']
        self.size = 10
        self.window_size = 1000
        self._agent_ids = ['chaser', 'escaper']
        self.action_space = {
            'chaser': spaces.Discrete(4),
            'escaper': spaces.Discrete(4)
        }
        obs_chaser = {
            'chaser_position': spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=int),
            'escaper_position': spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=int)
        }
        obs_escaper = {
            'chaser_position': spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=int),
            'escaper_position': spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=int)
        }
        
        self.observation_space = {
            'chaser': spaces.Dict(obs_user_high_dict),
            'escaper': spaces.Dict(obs_user_low_dict)
        }

        self.state = {
            'chaser_position': np.array([0,0]),
            'escaper_position': np.array([2,2])
        }
        
        self._action_mapping = {
            0: np.array([1, 0]),    # right
            1: np.array([0, 1]),    # down
            2: np.array([-1,0]),    # left
            3: np.array([0,-1])     # up
        }

        self.counter = 0
        self.done = False
        

        self.window = None
        self.clock = None
        self.isopen = True

        self.reset()

    def get_obs(self, agent_id):
        obs = {}
        if agent_id == 'full':
            for agent in self._agent_ids:
                obs[agent] = {}
                for key in self.observation_space[agent]:
                    obs[agent][key] = self.state[key]
        else:
            obs[agent_id] = {}
            for key in self.observation_space[agent_id]:
                obs[agent_id][key] = self.state[key]
        return obs

    def get_reward(self, agent_id):
        pass
        
    def check_legal_position(self, move_to):
        if (np.any(move_to < 0) or np.any(move_to >= self.size)):
            return False
        elif move_to in obstacles:
            return False
        else:
            return True 

    def reset(self):
        if self.random:
            chaser_position = np.random.randint(low=0, high=self.size, size=2)
            escaper_position = np.random.randint(low=0, high=self.size, size=2)
            while not self.check_legal_position(chaser_position):
                chaser_position = np.random.randint(low=0, high=self.size, size=2)
            while ((not self.check_legal_position) or np.array_equal(chaser_position, escaper_position)):
                 escaper_position = np.random.randint(low=0, high=self.size, size=2)
        else:
            self.state['chaser_position'] = np.array([0,0])
            self.state['escaper_position'] = np.array([2,2])
        self.counter = 0
        self.done = False

        return self.get_obs('escaper')

    def step(self, action_dict):
        agent_active = str(list(action_dict.keys())[0])
        action = list(action_dict.values())[0]
        reward = {}
        if agent_active = 'chaser':
            move_to = self.state['chaser_position'] + self._action_mapping[action]
            if self.check_legal_position(move_to):
                self.state['chaser_position'] = move_to
            else:
                pass # Move to obstacles will be bounced back
                
            if np.array_equal(self.state['chase_position'], self.state['escaper_position']):
                reward['chaser'] = 100
                self.done = True
            else:
                reward['chaser'] = -1
 
        if agent_actve = 'escaper':
            move_to = self.state['escaper_position'] + self._action_mapping[action]
            if self.check_legal_position(move_to):
                self.state['escaper_position'] = move_to
            else:
                pass # Move to obstacles will be bounced back
                
            if np.array_equal(self.state['chase_position'], self.state['escaper_position']):
                reward['escaper'] = -100
                self.done = True
            else:
                reward['escaper'] = 0
        
        done = {'__all__': self.done} 
                

        return self.get_obs(agent_active), reward, done, {}


    def render(self, mode='human'):
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

        # First we draw the excaper
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.state['escaper_position'],
                (pix_square_size, pix_square_size),
            ),
        )
        
        # Now we draw the chaser
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.state['chaser_position'] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        
        # Then we draw the obstacles
        for obstacle in obstacles:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * np.array(obstacle),
                    (pix_square_size, pix_square_size),
                ),
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

        



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Show a window of the system")
    group.add_argument("--dry_train", action="store_true", help="Train the agent")
    args = parser.parse_args()

    uiconfig = {
        'random': False,
        'n_buttons': 6,
        'save_ui': False,
        'mode': 'all_exclu'
    }
    if args.demo:
        env = MultiAgentUI(uiconfig)

        for _ in range(3):
            env.reset()
            for _ in range(10):
                env.step({'user_high': env.action_space['user_high'].sample()})
                env.step({'user_low': env.state['target']})
                env.render()
        env.close()

    if args.dry_train:
        from ray.rllib.agents import ppo
        from ray.tune.logger import pretty_print
        env = MultiAgentUI(uiconfig)
        policies = {
            "user_high": (
                None,
                env.observation_space['user_high'],
                env.action_space['user_high'],
                {}
            ),
            "user_low":(
                None,
                env.observation_space['user_low'],
                env.action_space['user_low'],
                {}
            )
        }

        config = {
            "num_workers": 3,
            "env": MultiAgentUI,
            "env_config": uiconfig,
            "gamma": 0.9,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "framework": "torch",
            "log_level": "CRITICAL",
            "num_gpus": 0,
            "seed": 31415,
        }

        stop = {
            "training_iteration": 150,
        }

        config = {**ppo.DEFAULT_CONFIG, **config}
        trainer = ppo.PPOTrainer(config=config)

        for _ in range(stop["training_iteration"]):
            result = trainer.train()
            print(pretty_print(result))

        trainer.save("test4_1")

        for i in range(10):
            env.reset()
            while not env.done:
                action = trainer.compute_single_action(observation=list(env.get_obs('user_high').values())[0], policy_id='user_high', unsquash_action=True)
                env.step({'user_high': action})
                env.step({'user_low': env.state['target']})
                env.render()
        env.close()


