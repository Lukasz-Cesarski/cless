from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Optional

import wandb
from cless import (DEPRECATED_KEYS, Config, WandbProjects,
                   cless_ensamble_train, general_setup)
from transformers import HfArgumentParser


@dataclass
class CommandLine:
    run_id: Optional[str] = field(
        default="b6dmqiv3",  #  0.5065796375274658 large+add2+pseudo
        metadata={"help": "Wandb Sweep run ID to get parameters"},
    )
    free_cublas: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Suppress CUBLAS lock (this may affect experiments reproduction!)"
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser((CommandLine, Config))
    (cli, config) = parser.parse_args_into_dataclasses()
    general_setup(free_cublas=cli.free_cublas)

    start = datetime.now()
    run_id = cli.run_id
    api = wandb.Api()
    run = api.run(f"{WandbProjects.WANDB_DEBERTA_SWEEPS}/{run_id}")
    wandb_settings = {k: v for k, v in run.config.items()}
    wandb_settings["report_to"] = "wandb"
    for k, v in wandb_settings.items():
        if k not in DEPRECATED_KEYS:
            if getattr(Config, k) == getattr(config, k):
                setattr(config, k, v)

    pprint(config)
    fold_results, fold_results_log, new_dump_dir = cless_ensamble_train(config=config)
    pprint(f"Script timer: {datetime.now() - start}")
