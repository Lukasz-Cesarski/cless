import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Optional

import torch
import wandb
from cless import WandbProjects, cless_model_ensamble_sweep, get_wandb_tags
from transformers import HfArgumentParser


SWEEP_CONFIG = {
    'method': 'random',  # grid, random
    'description': "|".join(get_wandb_tags()),
    'metric': {
        'name': 'micro.test_mcrmse',
        'goal': 'minimize',
    },
    'parameters': {
        'seed': {
            'distribution': "int_uniform",
            'min': 0,
            'max': 1000,
        },
        'learning_rate': {
            'distribution': "log_uniform_values",
            'min': 5e-6,
            'max': 1e-4,
        },
        "hidden_dropout_prob": {
            'distribution': "log_uniform_values",
            'min': 5e-4,
            'max': 5e-2,
        },
        "attention_probs_dropout_prob": {
            'distribution': "log_uniform_values",
            'min': 5e-4,
            'max': 5e-2,
        },
        "weight_decay": {
            'distribution': "log_uniform_values",
            'min': 5e-5,
            'max': 1e-2,
        },
        "batch_size": {
            'value': 8
        },
    },
}


@dataclass
class CommandLine:
    sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb Sweep ID. If not passed new sweep will be created."},
    )
    free_cublas: Optional[bool] = field(
        default=False,
        metadata={"help": "Suppress CUBLAS lock (this may affect experiments reproduction!)"},
    )
    count: Optional[int] = field(
        default=20,
        metadata={"help": "Number of hyperparameters search runs"},
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = HfArgumentParser((CommandLine,))
    (cl,) = parser.parse_args_into_dataclasses()

    if not cl.free_cublas:
        # deberta processing long texts is very sensitive for randomness
        # https://pytorch.org/docs/stable/notes/randomness#avoiding-nondeterministic-algorithms
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    try:
        if any(k.startswith("KAGGLE") for k in os.environ.keys()):
            # in kaggle environment import key from secrets
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            key = user_secrets.get_secret("wandb_api")
            wandb.login(key=key)
        else:
            # locally use env variable to pass key "export WANDB_API_KEY=...."
            wandb.login()
    except:
        print("Could not log in to WandB")
        exit(1)

    warnings.simplefilter("ignore")
    start = datetime.now()
    if cl.sweep_id is None:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=WandbProjects.WANDB_DEBERTA_SWEEPS)
    else:
        sweep_id = cl.sweep_id

    wandb.agent(sweep_id, cless_model_ensamble_sweep, count=cl.count)
    pprint(f"Script timer: {datetime.now() - start}")
