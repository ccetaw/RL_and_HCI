from simanneal import Annealer
from ButtonPanel import ButtonPanel
import random
import numpy as np
from ray.rllib.agents import ppo
import matplotlib.pyplot as plt


# Using simulated annealing to find the best ui for the user
class UIOptimizer(Annealer, ButtonPanel):

    def __init__(self, config):
        super(Annealer, self).__init__(config)
        super().__init__(initial_state=self.status())
        self.agent = config['agent']
        self.data = config['data']
        self.history = []

    def move(self):
        button_args = self.status().copy()
        # Every time only alter one button
        idx = np.random.randint(low=0, high=self.n_buttons)
        button_args[idx]['position'] += np.random.randint(low=-1, high=2, size=2) * self.grid_size
        button_args[idx]['size'] += np.random.randint(low=-1, high=2, size=2) * int(self.grid_size / 2)
        self.generata_customized_ui(button_args)
        self.state = self.status()

    
    def energy(self):
        overlapped = self.generata_customized_ui(self.state)
        cost = 0
        for _ in range(200):
            start_end = random.choices(patterns, weights=weights, k=2)
            while np.array_equal(start_end[0], start_end[1]):
                start_end = random.choices(patterns, weights=weights, k=2)
            self.reset(after_train=True, start_pattern=start_end[0], goal_pattern=start_end[1])
            while not self.done:
                action = self.agent.compute_single_action(self.get_obs(), unsquash_action=True)
                self.step(action, after_train=True)
                cost += self.add['move_time']
        if overlapped:
            cost += 500
        self.history.append(cost)
        return cost


if __name__ == '__main__':
    env_config = {
        'n_buttons': 9,
        'random': False
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

    agent = ppo.PPOTrainer(config=config)
    # agent.restore('')

    # Generate fake user data, use true data when available
    n_buttons = env_config['n_buttons']
    n_mutual_exclu = int(env_config['n_buttons']/2)-1
    n_normal = n_buttons - n_mutual_exclu -1
    weights = []
    patterns = []
    # We assume that the first button is always central, and the last n_buttons/2 - 1 are mutually exclusive
    pattern1 = np.zeros(n_buttons, dtype=int) # All OFF
    patterns.append(pattern1)
    weights.append(50)

    pattern2 = np.ones(n_buttons, dtype=int) # All ON
    if n_mutual_exclu > 0:
        pattern2[-n_mutual_exclu:] = 0
    patterns.append(pattern2)
    weights.append(50)

    for i in range(n_normal): # One button in normal group is ON
        pattern = np.zeros(n_buttons, dtype=int)
        pattern[i+1] = 1
        patterns.append(pattern)
        weights.append(10)

    for i in range(n_mutual_exclu): # One button in mutlal_exclu group is ON
        pattern = np.zeros(env_config['n_buttons'], dtype=int)
        pattern[-(i+1)] = 1
        patterns.append(pattern)
        weights.append(5)

    print(weights)
    print(patterns)


    optimizer_config = {
        **env_config,
        'agent': agent,
        'data': {
            'probabilities': weights,
            'patterns': patterns
        }
    }

    optimizer = UIOptimizer(optimizer_config)

    # x = np.array(range(len(optimizer.history)))
    # y = np.array(optimizer.history)

    # fig, ax = plt.subplots()
    # ax.plot(x, y, linewidth=2.0)
    # plt.savefig('test.png')