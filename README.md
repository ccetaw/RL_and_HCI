# Reinforcement Learning and Application in Human Computer Interaction
This repository aims to provide a simple and concrete tutorial on how to build and run your own reinforcement learning environment using ray rllib, and some applications in the domain of HCI. 

## RL basics
For basics in reinforcement learning, please refer to the famous book 
[Sutton Reinforcement Learning Book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) and some other very useful resources: [Introduction to SMDP](https://www.sciencedirect.com/science/article/pii/S0004370299000521), [Multi Agent Reinforcement Learning](https://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/10_003.pdf).

And here are some other resources that you might be interested in:  
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Reinforcement Learning - State of the Art](https://link.springer.com/book/10.1007/978-3-642-27645-3)


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

In fact, the official website of `ray` does not provide a very comprehensive tutorial and API documentation. That's what we will see in the next two sections.

Besides, we use [wandb](https://wandb.ai/site) to visulize the experiment results.

## How to build a single-agent environment
First of all, to ensure that ray library is installed correctly and to get a first look, run the `taxi.py` python script
```bash
python taxi.py
```

## How to build a multi-agent environment

## Application in HCI: Button Panel