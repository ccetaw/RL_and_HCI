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
        # Every time either switch two buttons or alter the position and the size of one button
        switch = random.choice([True, False])
        if switch:
            idx1 = np.random.randint(low=0, high=self.n_buttons)
            idx2 = np.random.randint(low=0, high=self.n_buttons)
            while idx1 == idx2:
                idx2 = np.random.randint(low=0, high=self.n_buttons)
            temp_position = button_args[idx1]['position']
            temp_size = button_args[idx1]['size']
            button_args[idx1]['position'] = button_args[idx2]['position']
            button_args[idx1]['size'] = button_args[idx2]['size']
            button_args[idx2]['position'] = temp_position
            button_args[idx2]['size'] = temp_size
        else:
            p_or_s = random.choice([True, False]) # To change position or size
            idx = np.random.randint(low=0, high=self.n_buttons)
            if p_or_s:
                new_position = button_args[idx]['position'] + np.random.randint(low=-1, high=2, size=2) * self.grid_size
                grid_new_position = self.grid_button_position(new_position)
                grid_button_size = self.grid_button_size(button_args[idx]['size'])
                if np.all(grid_new_position + grid_button_size < self.grid.shape) and np.all(grid_new_position > 0):
                    button_args[idx]['position'] = new_position
            else:
                grid_button_position = self.grid_button_position(button_args[idx]['position'])
                new_size = np.random.randint(low=1, high=6, size=2) * int(self.grid_size / 2)
                grid_new_size = self.grid_button_size(new_size)
                if np.all(grid_button_position + grid_new_size < self.grid.shape):
                    button_args[idx]['size'] = new_size
                
        self.generata_customized_ui(button_args)
        self.state = self.status()

    
    def energy(self):
        overlap = self.generata_customized_ui(self.state)
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
        if overlap:
            cost += 500
        self.history.append(cost)
        return cost


if __name__ == '__main__':
    env_config = {
        'n_buttons': 3,
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
    agent.restore('./trained_user/PPOTrainer_2022-09-16_15-52-00/bd125_00000-random_False-n_buttons_3_0_n_buttons=3_2022-09-16_15-52-01/checkpoint_001000/checkpoint-1000')

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

    # print(weights)
    # print(patterns)


    optimizer_config = {
        **env_config,
        'agent': agent,
        'data': {
            'probabilities': weights,
            'patterns': patterns
        }
    }

    optimizer = UIOptimizer(optimizer_config)

    optimizer.Tmax = 20
    optimizer.Tmin = 0.1
    optimizer.steps = 5000
    optimizer.updates = 1000


    optimizer.anneal()
    optimizer.save('./sa_2.json')

    x = np.array(range(len(optimizer.history)))
    y = np.array(optimizer.history)

    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    plt.savefig('test.png')