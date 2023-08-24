import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, HfArgumentParser, Trainer,
                          TrainingArguments, set_seed)

DATA_HOME_ENV = "DATA_HOME_DIR"
DATA_HOME_DIR_DEFAULT = "/kaggle/input"
DATA_TMP_ENV = "TMP_DIR"
COMPETITION_SUBDIR = "commonlit-evaluate-student-summaries"
PRO_TRAIN_FILE = "prompts_train.csv"
PRO_TEST_FILE = "prompts_test.csv"
SUM_TRAIN_FILE = "summaries_train.csv"
SUM_TEST_FILE = "summaries_test.csv"


def read_data(
    input_data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if input_data_dir is None:
        input_data_dir = os.environ.get(DATA_HOME_ENV)
    if input_data_dir is None:
        input_data_dir = DATA_HOME_DIR_DEFAULT
    data_dir = Path(input_data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found! {data_dir}")

    train_pro = pd.read_csv(data_dir / COMPETITION_SUBDIR / PRO_TRAIN_FILE)
    test_pro = pd.read_csv(data_dir / COMPETITION_SUBDIR / PRO_TEST_FILE)
    train_sum = pd.read_csv(data_dir / COMPETITION_SUBDIR / SUM_TRAIN_FILE)
    test_sum = pd.read_csv(data_dir / COMPETITION_SUBDIR / SUM_TEST_FILE)

    return train_pro, test_pro, train_sum, test_sum


@dataclass
class Config:
    model_name_or_path: Optional[str] = field(
        default="microsoft/deberta-v3-base",
        metadata={"help": "Model name or path"},
    )

    data_dir: Optional[str] = field(
        default="/kaggle/input/commonlit-evaluate-student-summaries",
        metadata={"help": "Data directory"},
    )

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max sequence length"},
    )

    add_prompt_question: Optional[bool] = field(
        default=False,
        metadata={"help": "Add prompt question into input"},
    )

    add_prompt_text: Optional[bool] = field(
        default=False,
        metadata={"help": "Add prompt text into input"},
    )

    fold: Optional[int] = field(
        default=0,
        metadata={"help": "Fold"},
    )

    num_proc: Optional[int] = field(
        default=4,
        metadata={"help": "Number of processes"},
    )

    dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Amount of dropout to apply"},
    )


def tokenize(example, tokenizer, config, labelled=True):
    sep = tokenizer.sep_token

    cols = []

    if config.add_prompt_question:
        cols.append("prompt_question")
    elif config.add_prompt_text:
        cols.append("prompt_text")
    cols.append("text")

    if labelled:
        labels = [example["content"], example["wording"]]

    tokenized = tokenizer(
        sep.join([example[c] for c in cols]),
        padding=False,
        truncation=True,
        max_length=config.max_seq_length,
    )

    result = {**tokenized}  # inplace copy
    if labelled:
        result["labels"] = labels

    return result


def compute_mcrmse(eval_pred):
    """
    Calculates mean columnwise root mean squared error
    https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
    """
    preds, labels = eval_pred

    col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    mcrmse = np.mean(col_rmse)

    return {
        "content_rmse": col_rmse[0],
        "wording_rmse": col_rmse[1],
        "mcrmse": mcrmse,
    }


def training_loop(args=None):
    if args is None:
        args = ["--output_dir", "tmp"]

    parser = HfArgumentParser((Config, TrainingArguments))
    config, training_args = parser.parse_args_into_dataclasses(args)

    set_seed(training_args.seed)

    if "wandb" in training_args.report_to:
        try:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            key = user_secrets.get_secret("wandb_api")
            wandb.login(key=key)
        except:
            print("Could not log in to WandB")

    run = wandb.init(reinit=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model_config = AutoConfig.from_pretrained(config.model_name_or_path)

    model_config.update(
        {
            "hidden_dropout_prob": config.dropout,
            "attention_probs_dropout_prob": config.dropout,
            "num_labels": 2,
            "problem_type": "regression",
            "cfg": config.__dict__,
        }
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path, config=model_config, ignore_mismatched_sizes=True
    )

    training_args.greater_is_better = False
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "mcrmse"
    # When load_best_model_at_end is set to True, the parameters save_strategy needs to be the same as evaluation_strategy,
    # and in the case it is “steps”, save_steps must be a round multiple of eval_steps.
    STEPS_VAL = 100
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = STEPS_VAL
    training_args.save_total_limit = 2
    training_args.num_train_epochs = 3
    training_args.save_strategy = "steps"
    training_args.save_steps = STEPS_VAL

    # read data
    train_pro, test_pro, train_sum, test_sum = read_data()
    df = train_pro.merge(train_sum, on="prompt_id")
    # 4 prompt ids, 4 folds
    id2fold = {
        "814d6b": 0,
        "39c16e": 1,
        "3b9047": 2,
        "ebad26": 3,
    }
    assert set(id2fold.keys()) == set(df["prompt_id"].unique())
    df["fold"] = df["prompt_id"].map(id2fold)

    train_ds = Dataset.from_pandas(df[df["fold"] != config.fold])
    val_ds = Dataset.from_pandas(df[df["fold"] == config.fold])

    train_ds = train_ds.map(
        tokenize,
        batched=False,
        num_proc=config.num_proc,
        fn_kwargs={"tokenizer": tokenizer, "config": config},
    )

    val_ds = val_ds.map(
        tokenize,
        batched=False,
        num_proc=config.num_proc,
        fn_kwargs={"tokenizer": tokenizer, "config": config},
    )

    if os.environ.get("CLESS_DRY_RUN") == "1":
        dry_run_ds = train_ds.train_test_split(None, 30)
        train_ds = dry_run_ds["train"]

        dry_run_ds = val_ds.train_test_split(None, 10)
        val_ds = dry_run_ds["train"]

    # print("Training set number of rows:", train_ds.num_rows)
    # print("Validation set number of rows:", val_ds.num_rows)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=16 if training_args.fp16 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_mcrmse,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    eval_res = trainer.evaluate()

    model.config.best_metric = trainer.state.best_metric
    model.config.save_pretrained(training_args.output_dir)

    dump_dir = os.path.join(
        "model_dumps",
        model_config.name_or_path.replace("/", "-") + f"_fold-{config.fold}",
    )
    os.makedirs(dump_dir, exist_ok=True)
    trainer.save_model(output_dir=dump_dir)
    shutil.rmtree(training_args.output_dir)
    run.finish()

    return eval_res

