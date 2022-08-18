# Reinforcement Learning and Application in Human Computer Interaction
This repository aims to provide a simple and concrete tutorial on how to build and run your own reinforcement learning environment using ray rllib, and some applications in the domain of HCI. 

## RL basics
For basics in reinforcement learning, please refer to the famous book 
[Sutton Reinforcement Learning Book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) and some other very useful resources: [Introduction to SMDP](https://www.sciencedirect.com/science/article/pii/S0004370299000521), [Multi Agent Reinforcement Learning](https://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/10_003.pdf).

And here are some other resources that you might be interested in:  
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Reinforcement Learning - State of the Art](https://link.springer.com/book/10.1007/978-3-642-27645-3)
- [Goal Conditioned Reinforcement Learning](https://proceedings.neurips.cc/paper/2019/file/c8cc6e90ccbff44c9cee23611711cdc4-Paper.pdf)


## Set up conda environment
### Install conda/miniconda
If you don't have conda or miniconda installed on your computer, install it first following the [official tutorial](https://docs.conda.io/en/latest/miniconda.html). If you have it already, ignore this section.

For linux/wsl user, run 
```bash
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"
```
to download the miniconda installer. Install miniconda by running
```bash
bash Miniconda3.sh
```
Restart your Terminal. Now your prompt should list which environment is active. After installation, delete the installer
```bash
rm Miniconda3.sh
```

The `(base)` environment is thereafter activated by default the next time you start your terminal. To disable auto-activation, run
```bash
conda config --set auto_activate_base false
```

### Create environment from .yml file
To install all pakages required for this repository, run
```bash
conda env create -f environment.yml
conda activate RL
```
If you want to play with conda environments, see [conda cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

### Utilities
We will use mainly `gym` and `ray-rllib` pakages. I list their officil documentation here:
- [gym](https://www.gymlibrary.ml/)
- [ray-rllib](https://docs.ray.io/en/releases-1.11.0/rllib/index.html)

However, the official website of `ray` does not provide a very comprehensive tutorial and API documentation. That's what we will see in the next two sections.

In the end, we use [wandb](https://wandb.ai/site) to visulize the experiment results.

## How to build a single-agent environment
First of all, to ensure that ray library is installed correctly and to get a first look, run the `taxi.py` python script
```bash
cd src/RL_tutorial
python taxi.py
```
Normally you should see something like this
```bash
(RolloutWorker pid=2302) +---------+
(RolloutWorker pid=2302) |R: | : :G|
(RolloutWorker pid=2302) | : | : : |
(RolloutWorker pid=2302) | : : : : |
(RolloutWorker pid=2302) | | : | : |
(RolloutWorker pid=2302) |Y| : |B: |
(RolloutWorker pid=2302) +---------+
(RolloutWorker pid=2302)   (South)
(RolloutWorker pid=2302) +---------+
(RolloutWorker pid=2302) |R: | : :G|
(RolloutWorker pid=2302) | : | : : |
(RolloutWorker pid=2302) | : : : : |
(RolloutWorker pid=2302) | | : | : |
(RolloutWorker pid=2302) |Y| : |B: |
(RolloutWorker pid=2302) +---------+
(RolloutWorker pid=2302)   (South)
```
### Common Paradigm
<!-- We all know that a reinforcement learning problem is mainly composed of **state**, **action**, **reward**, **policy**. An agent takes an action against the current state following a policy and then the state transits into another state.  -->
You can find a tutorial [on the official webside of `gym`](https://www.gymlibrary.ml/content/environment_creation/). We will improve the example it provides by adding wind to the grid world. In a word, in this example an agent is trying to reach a randomly generated goal starting from a random position. 

In Python language, to customize your own environment, you should define a class inherited from `gym.Env`. There are four components that your must specify in your class: two properties `action_space` and `observation_space`, and two methods `step(action)` and `reset()`. Besides, it's a good practice to define the state of the environment a property of the class. 
```python
import gym
class YourEnv(gym.Env):

def __init__(self):
    self.action_space = some_action_space # Define your action space here
    self.observation_space = some_observation_space # Define your observation space here
    self.state = some_state # Optional

def get_obs(self): # Get the current observation. Optional
    obs = {}
        for key in self.observation_space:
            obs[key] = self.state[key]
    return obs

def step(self,action):
    # Take a step here
    reward: double = some_reward
    done: bool = done_or_not
    add: dict = {}
    return self.get_obs(), reward, done, add

def reset(self):
    # Reset the environment here
    return self.get_obs()

def render(self): # Visulize your environment. Optional
    pass
```
#### Implementation details
- `action_space`: This is exactly the same thing as you see in reinforcement learning language, i.e. $\mathcal{A}$, the space of actions that the agent can take. A general way to specify the space is using what `gym.spaces` provides [here](https://www.gymlibrary.ml/content/spaces/). For example, in [windy_grid.py](src/RL_tutorial/windy_grid.py), there are four actions that the agent can take: go up, down, left,right. We could thus define the action space as a discrete space of 4 choices and define a action mapping function. 
- `observation_space`: This is what the agent sees. The agent takes an action based on the its observation on the current state. An observation could be a subset of the state or the same as the state w.r.t. whether it's MDP or POMDP. The observation_space is defined the same way as the action space. In [windy_grid.py](src/RL_tutorial/windy_grid.py), since it's a goal-conditioned reinforcement learning problem, the agent can see its own position and the goal position. 
- `step(action)`: In this function, the agent takes the `action` provided in the parameter. During training, the `action` is supplied by the trained policy. You need to update the state of the environment and return four elements:
    - `observation`: a dictionary that must have the same keys as the `observation_space`. You could define additionally a method `self._get_obs()` to get the current observation.
    - `reward`: the reward(a scalar, positive or negative) the agent get for taking the action.
    - `done`: a boolean indicates if the episode is over(the agent finishs its task or terminated because the epsiode is too long).
    - `add`: a dictionary that contains anything else that you want the function to return.
- `reset()`: reset the environment. In [windy_grid.py](src/RL_tutorial/windy_grid.py), the agent and the goal are regenerated randomly. A current observation needs to be returned.

You can now play with the [windy_grid.py](src/RL_tutorial/windy_grid.py) example. Run
```bash
python windy_grid.py --demo
```
to take a look at the environment. To train the agent, run 
```bash
python windy_grid.py --train
```
and a visualization of the trained model will be showed. As the agent needs to learn every possible case of starting location and goal position, the state space is quite large, so it needs some time to learn. 

There is another example [corridor.py](src/RL_tutorial/corridor.py) that ray-rllib provides, which is simpler than the windy grid. You could also play around with it. 

## Interlude: construct a graphical user interface

## How to build a multi-agent environment

## Application in HCI: button panel