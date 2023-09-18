from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Optional

import wandb
from cless import WandbProjects, get_wandb_tags, general_setup, cless_single_fold_sweep
from transformers import HfArgumentParser


SWEEP_CONFIG = {
    'method': 'random',  # grid, random
    'description': "|".join(get_wandb_tags()),
    'metric': {
        'name': 'test_mcrmse',
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
    #TODO consider change to folds (list input)
    fold: int = field(
        metadata={"help": "Fold to train model"},
    )
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
    parser = HfArgumentParser((CommandLine,))
    (cl,) = parser.parse_args_into_dataclasses()
    general_setup(free_cublas=cl.free_cublas)

    start = datetime.now()
    if cl.sweep_id is None:
        seep_config = SWEEP_CONFIG.copy()
        seep_config["description"] = seep_config.get("description") + f"|fold_{cl.fold}"
        sweep_id = wandb.sweep(seep_config, project=WandbProjects.WANDB_DEBERTA_SWEEPS)
    else:
        sweep_id = cl.sweep_id

    wandb.agent(sweep_id, lambda: cless_single_fold_sweep(fold=cl.fold), count=cl.count)
    pprint(f"Script timer: {datetime.now() - start}")
