from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Optional

import wandb
from cless import Config, WandbProjects, general_setup, cless_ensamble_train
from transformers import HfArgumentParser


@dataclass
class CommandLine:
    run_id: Optional[str] = field(
        default="wfx04u5o",
        metadata={"help": "Wandb Sweep run ID to get parameters"},
    )
    free_cublas: Optional[bool] = field(
        default=False,
        metadata={"help": "Suppress CUBLAS lock (this may affect experiments reproduction!)"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((CommandLine,))
    (cl,) = parser.parse_args_into_dataclasses()
    general_setup(free_cublas=cl.free_cublas)

    start = datetime.now()
    run_id = cl.run_id
    api = wandb.Api()
    run = api.run(f"{WandbProjects.WANDB_DEBERTA_SWEEPS}/{run_id}")
    config_settings = {k: v for k, v in run.config.items()}
    config_settings["report_to"] = "wandb"
    config = Config(
        **{
            k: v
            for k, v in config_settings.items()
            if k not in ["data_dir", "num_proc"]  # deprecated keys, remove this part
        }
    )
    pprint(config)
    fold_results, fold_results_log, new_dump_dir = cless_ensamble_train(config=config)
    pprint(f"Script timer: {datetime.now() - start}")
