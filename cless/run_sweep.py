from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Optional

import wandb
from cless import (
    KEEP_BEST_MODELS,
    Config,
    WandbProjects,
    cless_ensemble_sweep,
    general_setup,
    get_wandb_tags,
)
from transformers import HfArgumentParser

SWEEP_CONFIG_BASE = {
    "method": "random",  # grid, random
    "description": "|".join(get_wandb_tags()),
    "metric": {
        "name": "micro.test_mcrmse",
        "goal": "minimize",
    },
    "parameters": {
        "seed": {
            "distribution": "int_uniform",
            "min": 0,
            "max": 1000,
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 5e-7,
            "max": 1e-4,
        },
        "hidden_dropout_prob": {
            "distribution": "log_uniform_values",
            "min": 5e-4,
            "max": 5e-2,
        },
        "attention_probs_dropout_prob": {
            "distribution": "log_uniform_values",
            "min": 5e-4,
            "max": 5e-2,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-2,
        },
        "batch_size": {"value": 8},
    },
}

SWEEP_CONFIG_LARGE = {
    "method": "random",  # grid, random
    "description": "|".join(get_wandb_tags()),
    "metric": {
        "name": "micro.test_mcrmse",
        "goal": "minimize",
    },
    "parameters": {
        "seed": {
            "distribution": "int_uniform",
            "min": 0,
            "max": 1000,
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 5e-5,
        },
        "hidden_dropout_prob": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-2,
        },
        "attention_probs_dropout_prob": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-2,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-2,
        },
        "batch_size": {"value": 6},
    },
}

SWEEP_CONFIG_XLARGE = {
    "method": "random",  # grid, random
    "description": "|".join(get_wandb_tags()),
    "metric": {
        "name": "micro.test_mcrmse",
        "goal": "minimize",
    },
    "parameters": {
        "seed": {
            "distribution": "int_uniform",
            "min": 0,
            "max": 1000,
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-4,
        },
        "hidden_dropout_prob": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-2,
        },
        "attention_probs_dropout_prob": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-2,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-2,
        },
        "freeze_layers": {"values": [3, 6, 9]},
    },
}


@dataclass
class CommandLine:
    # TODO consider change to folds (list input)
    sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb Sweep ID. If not passed new sweep will be created."},
    )
    free_cublas: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Suppress CUBLAS lock (this may affect experiments reproduction!)"
        },
    )
    count: Optional[int] = field(
        default=20,
        metadata={"help": "Number of hyperparameters search runs"},
    )
    keep_best_models: Optional[int] = field(
        default=KEEP_BEST_MODELS,
        metadata={"help": "Number of experiments to keep"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((CommandLine, Config))
    (cli, config) = parser.parse_args_into_dataclasses()
    general_setup(free_cublas=cli.free_cublas)

    start = datetime.now()
    if cli.sweep_id is None:
        if "xlarge" in config.model_name_or_path:
            sweep_config = SWEEP_CONFIG_XLARGE.copy()
        elif "large" in config.model_name_or_path:
            sweep_config = SWEEP_CONFIG_LARGE.copy()
        elif "base" in config.model_name_or_path:
            sweep_config = SWEEP_CONFIG_BASE.copy()
        else:
            raise NotImplemented(
                f"Not supported model name: {config.model_name_or_path}"
            )
        sweep_id = wandb.sweep(sweep_config, project=WandbProjects.WANDB_DEBERTA_SWEEPS)
    else:
        sweep_id = cli.sweep_id

    wandb.agent(
        sweep_id=sweep_id,
        function=lambda: cless_ensemble_sweep(
            input_config=config, keep_best_models=cli.keep_best_models
        ),
        count=cli.count,
        project=WandbProjects.WANDB_DEBERTA_SWEEPS,
    )
    pprint(f"Script timer: {datetime.now() - start}")
