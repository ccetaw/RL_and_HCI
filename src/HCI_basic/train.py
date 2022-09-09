import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ButtonPanel import ButtonPanel
from utils import trial_name_string

ray.init(local_mode=False, num_cpus=24, num_gpus=0)

env_config = {
    'random': True,
    'n_buttons': tune.grid_search([3,5,7,9]),
}

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
    local_dir="trained_user",
    verbose=1,
    checkpoint_freq=50,
    checkpoint_at_end=True,
    num_samples=1,
    callbacks=[
        WandbLoggerCallback(
                project="button_panel",
                entity="cetaceanw",
                group="test",
                job_type="train",
                log_config=True,
                save_code=True,
        )
    ],
)
ray.shutdown()