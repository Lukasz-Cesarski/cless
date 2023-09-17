import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Optional

import torch
import wandb
from cless import Config, WandbProjects, cless_model_ensamble_train
from transformers import HfArgumentParser


@dataclass
class CommandLine:
    run_id: Optional[str] = field(
        default="bxu2kp2l",
        metadata={"help": "Wandb Sweep run ID"},
    )
    free_cublas: Optional[bool] = field(
        default=False,
        metadata={"help": "Suppress CUBLAS lock (this may affect experiments reproduction!)"},
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
    run_id = cl.run_id
    api = wandb.Api()
    run = api.run(f"{WandbProjects.WANDB_DEBERTA_SWEEPS}/{run_id}")
    config_settings = {k: v for k, v in run.config.items()}
    config_settings["report_to"] = "wandb"
    config = Config(
        **{
            k: v
            for k, v in config_settings.items()
            if k not in ["data_dir", "num_proc"]
        }
    )
    pprint(config)
    fold_results = cless_model_ensamble_train(config=config)
    pprint(fold_results)
    pprint(f"Script timer: {datetime.now() - start}")