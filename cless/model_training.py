import warnings
import logging
import os
from dataclasses import field, dataclass
from typing import Optional

import pandas as pd
import numpy as np
import wandb
from datetime import datetime

import torch
from datasets import disable_progress_bar
from datasets import Dataset
from cless import read_cless_data, Config, ClessModel, ID2FOLD, WandbProjects, cless_model_ensamble_predict, cless_model_ensamble_train, cless_model_fold_predict, TARGET_LABELS, compute_mcrmse_for_trainer, MODEL_DUMPS_DIR

from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, HfArgumentParser, Trainer,
                          TrainingArguments, set_seed)

from pprint import pprint

@dataclass
class CommandLine:
    run_id: Optional[str] = field(
        default="bxu2kp2l",
        metadata={"help": "Wandb Sweep run ID"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((CommandLine, ))
    cl, = parser.parse_args_into_dataclasses()

    warnings.simplefilter("ignore")
    try:
        wandb.login()
    except:
        print("Could not log in to WandB")
        exit(1)
    run_id = cl.run_id
    api = wandb.Api()
    run = api.run(f"{WandbProjects.WANDB_DEBERTA_SWEEPS}/{run_id}")
    config_settings = {k: v for k, v in run.config.items()}
    config_settings["report_to"] = "wandb"
    config = Config(**{k: v for k, v in config_settings.items() if k not in ["data_dir", "num_proc"]})
    pprint(config)
    fold_results = cless_model_ensamble_train(config=config)

