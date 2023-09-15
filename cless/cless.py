"""
CommonLit - Evaluate Student Summaries (CLESS)
https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview

This file stores essential parts of code to
- provide version control in GIT
- upload it to remote machines
- paste it in kaggle notebook (use magic `%%writefile cless.py` command)
- easy code development (code refactor, black, import sort)
"""

import os
import re
import shutil
from dataclasses import dataclass
from pprint import pprint
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from autocorrect import Speller
from datasets import disable_progress_bar
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from spellchecker import SpellChecker
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed)
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
tqdm.pandas()  # for processor
disable_progress_bar()

class Files:
    PRO_TRAIN_FILE = "prompts_train.csv"
    PRO_TEST_FILE = "prompts_test.csv"
    SUM_TRAIN_FILE = "summaries_train.csv"
    SUM_TEST_FILE = "summaries_test.csv"


class WandbProjects:
    WANDB_DEBERTA_FOLDS = "cless-deberta-folds"
    WANDB_DEBERTA_ENSAMBLE = "cless-deberta-ensamble"
    WANDB_DEBERTA_SWEEPS = "cless-deberta-sweeps"

    WANDB_LGBM_FOLDS = "cless-lgbm-folds"
    WANDB_LGBM_ENSAMBLE = "cless-lgbm-ensamble"
    WANDB_LGBM_SWEEPS = "cless-lgbm-sweeps"

ID2FOLD = {
    "814d6b": 0,
    "39c16e": 1,
    "3b9047": 2,
    "ebad26": 3,
}

CLESS_DATA_ENV = "CLESS_DATA_ENV"
CLESS_DRY_RUN = "CLESS_DRY_RUN"
CLESS_DATA_ENV_DEFAULT = "/kaggle/input/"
TMP_DIR = "tmp"
MODEL_DUMPS_DIR = "model_dumps"
TARGET_LABELS = ["content", "wording"]


def read_cless_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_home = os.environ.get(CLESS_DATA_ENV)
    if data_home is None:
        # default kaggle location
        data_home = CLESS_DATA_ENV_DEFAULT
    data_home = Path(data_home)
    if not data_home.is_dir():
        raise FileNotFoundError(f"Directory not found! {data_home}")

    data_home_sub = data_home / "commonlit-evaluate-student-summaries"

    train_pro = pd.read_csv(data_home_sub / Files.PRO_TRAIN_FILE)
    test_pro = pd.read_csv(data_home_sub / Files.PRO_TEST_FILE)
    train_sum = pd.read_csv(data_home_sub / Files.SUM_TRAIN_FILE)
    test_sum = pd.read_csv(data_home_sub / Files.SUM_TEST_FILE)

    return train_pro, test_pro, train_sum, test_sum


@dataclass
class Config:
    model_name_or_path: str = "microsoft/deberta-v3-base"
    max_seq_length: int = 512
    add_prompt_question: bool = False
    add_prompt_text: bool = False
    hidden_dropout_prob: float = 0.05
    attention_probs_dropout_prob: float = 0.05
    learning_rate: float = 5e-05
    weight_decay: float = 0.0
    batch_size: int = 8
    num_train_epochs: int = 3
    seed: int = 42
    report_to: str = "wandb"
    eval_every: int = 50
    patience = 10  # early stopping


def tokenize(example, tokenizer, config, labelled=True):
    sep = tokenizer.sep_token

    cols = []

    if config.add_prompt_question:
        cols.append("prompt_question")
    elif config.add_prompt_text:
        cols.append("prompt_text")
    cols.append("text")

    tokenized = tokenizer(
        sep.join([example[c] for c in cols]),
        padding=False,
        truncation=True,
        max_length=config.max_seq_length,
    )

    result = {**tokenized}  # inplace copy

    if labelled:
        labels = [example[t] for t in TARGET_LABELS]
        result["labels"] = labels

    return result


def compute_mcrmse_for_trainer(eval_pred):
    """
    Calculates mean columnwise root mean squared error
    https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
    function adjusted for huggingface trainer
    """
    preds, labels = eval_pred

    col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    mcrmse = np.mean(col_rmse)

    result = {f"{t}_rmse": col_rmse[idx] for idx, t in enumerate(TARGET_LABELS)}
    result["mcrmse"] = mcrmse

    return result


class ClessModel:
    def __init__(
        self,
        model_name_or_path: str,
        tmp_dir: str = TMP_DIR ,
        dump_dir: str = MODEL_DUMPS_DIR,
    ):
        self.model_name_or_path = model_name_or_path
        self.tmp_dir = tmp_dir
        self.dump_dir = dump_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(self.model_name_or_path)

        self.model_config.update(
            {
                "num_labels": 2,
                "problem_type": "regression",
            }
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_single_fold(
        self,
        config: Config,
        fold: int,
    ):
        if config.report_to == "wandb":
            run = wandb.init(
                # reinit=True,
                project=WandbProjects.WANDB_DEBERTA_FOLDS,
                config=config.__dict__,
            )
        set_seed(config.seed)
        self.model_config.update(
            {
                "hidden_dropout_prob": config.hidden_dropout_prob,
                "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
                "cfg": config.__dict__,
            }
        )

        train_pro, test_pro, train_sum, test_sum = read_cless_data()
        df = train_pro.merge(train_sum, on="prompt_id")
        assert set(ID2FOLD.keys()) == set(df["prompt_id"].unique())
        df["fold"] = df["prompt_id"].map(ID2FOLD)
        train_ds = Dataset.from_pandas(df[df["fold"] != fold])
        val_ds = Dataset.from_pandas(df[df["fold"] == fold])

        train_ds = train_ds.map(
            tokenize,
            batched=False,
            fn_kwargs={"tokenizer": self.tokenizer, "config": config},
        )
        val_ds = val_ds.map(
            tokenize,
            batched=False,
            fn_kwargs={"tokenizer": self.tokenizer, "config": config},
        )

        # dry run
        if os.environ.get(CLESS_DRY_RUN) == "1":
            dry_run_ds = train_ds.train_test_split(None, 20)
            train_ds = dry_run_ds["train"]

            dry_run_ds = val_ds.train_test_split(None, 10)
            val_ds = dry_run_ds["train"]

        training_args = TrainingArguments(
            output_dir=self.tmp_dir,
            load_best_model_at_end=True,  # select best model
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_train_epochs,
            weight_decay=config.weight_decay,
            report_to=config.report_to,
            greater_is_better=False,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=config.eval_every,
            save_steps=config.eval_every,
            metric_for_best_model="mcrmse",
            save_total_limit=1,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, config=self.model_config
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_mcrmse_for_trainer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
        )
        trainer.train()
        eval_res = trainer.predict(test_dataset=val_ds)

        model_fold_dir = os.path.join(
            self.dump_dir,
            self.model_name_or_path.replace("/", "_"),
            str(fold),
        )
        os.makedirs(model_fold_dir, exist_ok=True)
        trainer.save_model(output_dir=model_fold_dir)
        if config.report_to == "wandb":
            run.finish()
        shutil.rmtree(self.tmp_dir)

        return eval_res

    def predict_single_fold(self, df):
        config = Config(**self.model_config.cfg)
        test_ds = Dataset.from_pandas(df)
        test_ds = test_ds.map(
            tokenize,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "config": config,
                "labelled": False,
            },
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            config=self.model_config,
            # ignore_mismatched_sizes=True,
        )
        model.eval()

        test_args = TrainingArguments(
            output_dir=self.tmp_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=4,
            dataloader_drop_last=False,
        )
        infer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=test_args,
        )
        predictions = infer.predict(test_dataset=test_ds)

        return predictions


def cless_model_ensamble_train(config):
    fold_results = {}

    for fold in range(4):
        print(f">>> fold {fold}:")
        cless_deberta = ClessModel(
            model_name_or_path=config.model_name_or_path,
        )
        eval_res = cless_deberta.train_single_fold(
            fold=fold,
            config=config,
        )
        fold_results[fold] = eval_res

    # evaluation
    fold_results_log = {f'f{k}': v.metrics for k, v in fold_results.items()}
    mean_metrics = {}
    for metric in tuple(fold_results_log.values())[0].keys():
        metric_values = []
        for fold in fold_results_log:
            metric_values.append(fold_results_log[fold][metric])
        mean_metrics[metric] = np.mean(metric_values)

    fold_results_log["macro"] = mean_metrics

    flat_p = np.vstack([v.predictions for v in fold_results.values()])
    flat_l = np.vstack([v.label_ids for v in fold_results.values()])

    fold_results_log["micro"] = {f"test_{k}": v for k, v in compute_mcrmse_for_trainer((flat_p, flat_l)).items()}

    pprint(fold_results_log)
    if config.report_to == "wandb":
        run = wandb.init(
            # Set the project where this run will be logged
            project=WandbProjects.WANDB_DEBERTA_ENSAMBLE,
            # Track hyperparameters and run metadata
            config=config.__dict__)
        run.log(data=fold_results_log)
        run.finish()

    return fold_results, fold_results_log


def cless_model_fold_predict(fold_subdir, df):
    """For cross-validation / LGBM training"""
    assert os.path.isdir(fold_subdir)
    cless_deberta = ClessModel(model_name_or_path=fold_subdir)
    fold_prediction = cless_deberta.predict_single_fold(df)
    partial_prediction = pd.concat(
        [
            df["student_id"],
            pd.DataFrame(
                fold_prediction.predictions, columns=TARGET_LABELS, index=df.index,
            ),
        ],
        axis=1,
    )
    return partial_prediction


def cless_model_ensamble_predict(folds_dir, df):
    """For submission"""
    predictions = {}
    for fold in range(4):
        fold_subdir = os.path.join(folds_dir, str(fold))
        print(fold_subdir)
        partial_prediction = cless_model_fold_predict(fold_subdir, df)
        predictions[fold] = partial_prediction
    return predictions


class Preprocessor:
    def __init__(self) -> None:
        self.twd = TreebankWordDetokenizer()
        self.STOP_WORDS = set(stopwords.words("english"))

        self.speller = Speller(lang="en")
        self.spellchecker = SpellChecker()

    def word_overlap_count(self, row):
        """intersection(prompt_text, text)"""

        def check_is_stop_word(word):
            return word in self.STOP_WORDS

        prompt_words = row["prompt_tokens"]
        summary_words = row["summary_tokens"]
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))

    def ngrams(self, token, n):
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int) -> int:
        # Tokenize the original text and summary into words
        original_tokens = row["prompt_tokens"]
        summary_tokens = row["summary_tokens"]

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)
        return len(common_ngrams)

    def quotes_count(self, row):
        summary = row["text"]
        text = row["prompt_text"]
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary) > 0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text):

        wordlist = text.split()
        amount_miss = len(list(self.spellchecker.unknown(wordlist)))

        return amount_miss

    def add_spelling_dictionary(self, tokens: List[str]) -> List[str]:
        """dictionary update for pyspell checker and autocorrect"""
        self.spellchecker.word_frequency.load_words(tokens)
        self.speller.nlp_data.update({token: 1000 for token in tokens})

    def run(
        self,
        prompts: pd.DataFrame,
        summaries: pd.DataFrame,
    ) -> pd.DataFrame:

        # before merge preprocess
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: word_tokenize(x)
        )
        prompts["prompt_length"] = prompts["prompt_tokens"].apply(len)
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: word_tokenize(x)
        )
        summaries["summary_length"] = summaries["summary_tokens"].apply(len)

        # Add prompt tokens into spelling checker dictionary
        prompts["prompt_tokens"].apply(lambda x: self.add_spelling_dictionary(x))

        #         from IPython.core.debugger import Pdb; Pdb().set_trace()
        # fix misspelling
        summaries["fixed_summary_text"] = summaries["text"].progress_apply(
            lambda x: self.speller(x)
        )

        # count misspelling
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)

        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # after merge preprocess
        input_df["length_ratio"] = (
            input_df["summary_length"] / input_df["prompt_length"]
        )

        input_df["word_overlap_count"] = input_df.progress_apply(
            self.word_overlap_count, axis=1
        )
        input_df["bigram_overlap_count"] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(2,), axis=1
        )
        input_df["bigram_overlap_ratio"] = input_df["bigram_overlap_count"] / (
            input_df["summary_length"] - 1
        )

        input_df["trigram_overlap_count"] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(3,), axis=1
        )
        input_df["trigram_overlap_ratio"] = input_df["trigram_overlap_count"] / (
            input_df["summary_length"] - 2
        )

        input_df["quotes_count"] = input_df.progress_apply(self.quotes_count, axis=1)

        input_df["fold"] = input_df["prompt_id"].map(ID2FOLD)

        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])


def get_preprocessed_dataset(prompts, summaries, save_path, preprocessor=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        return torch.load(save_path)

    if preprocessor is None:
        preprocessor = Preprocessor()
    input_df = preprocessor.run(
        prompts=prompts,
        summaries=summaries,
    )
    torch.save(input_df, save_path)

    return input_df


def choose_fold(row, target, ignore_fold):
    if ignore_fold:
        clean_targets = [f"{target}_{idx}" for idx in range(4)]
    else:
        fold = row["fold"]
        clean_targets = [f"{target}_{fold}"]
    mean_value = row[clean_targets].mean()
    return mean_value


def merge_features(preprocessed_df, deberta_features, ignore_fold=False):
    # mean of non-leak predictions
    merged_df = pd.merge(preprocessed_df, deberta_features, on="student_id", how="left")
    merged_df["deberta_content"] = merged_df.apply(lambda x: choose_fold(x, "content", ignore_fold), axis=1)
    merged_df["deberta_wording"] = merged_df.apply(lambda x: choose_fold(x, "wording", ignore_fold), axis=1)

    merged_df = merged_df.drop(columns=[f"content_{f}" for f in range(4)] + [f"wording_{f}" for f in range(4)])

    return merged_df


def train_lgbm(train, targets, drop_columns):
    # https://colab.research.google.com/drive/181GCGp36_75C2zm7WLxr9U2QjMXXoibt#scrollTo=aIhxl7glaJ5k
    # defaults
    default_params = {
        'boosting_type': 'gbdt',
        'random_state': 42,
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'boosting': "gbdt",
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'min_data_in_bin': 3,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
    }
    # Initialize a new wandb run
    wandb.init(
        config=default_params,
        reinit=True,
        project=WandbProjects.WANDB_LGBM_FOLDS,
    )

    TRAINING_COLUMNS = None
    model_dict = {}

    for target in targets:
        models = []

        for fold in range(4):
            print(f"target={target}, fold={fold}")

            X_train_cv = train[train["fold"] != fold].drop(columns=drop_columns)
            y_train_cv = train[train["fold"] != fold][target]
            TRAINING_COLUMNS = X_train_cv.columns  # to check with prediction

            X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)
            y_eval_cv = train[train["fold"] == fold][target]

            dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
            dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

            evaluation_results = {}
            # deepcopy may cause MaxRecursionException
            hyperparameters = {k:v for k, v in wandb.config.items()}

            model = lgb.train(hyperparameters,
                              num_boost_round=10000,
                              # categorical_feature = categorical_features,
                              valid_names=['train_single_fold', 'valid'],
                              train_set=dtrain,
                              valid_sets=dval,
                              callbacks=[
                                  lgb.early_stopping(stopping_rounds=30, verbose=True),
                                  lgb.log_evaluation(100),
                                  lgb.callback.record_evaluation(evaluation_results),
                                  # wandb_callback(),
                              ],
                              )
            # log_summary(model, save_model_checkpoint=True)
            models.append(model)

        model_dict[target] = models

    # cv
    metrics = {}

    for target in targets:
        models = model_dict[target]

        preds = []
        trues = []

        for fold, model in enumerate(models):
            X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)
            assert (X_eval_cv.columns == TRAINING_COLUMNS).all()
            y_eval_cv = train[train["fold"] == fold][target]

            pred = model.predict_single_fold(X_eval_cv)

            trues.extend(y_eval_cv)
            preds.extend(pred)

        rmse = np.sqrt(mean_squared_error(trues, preds))
        metrics[target] = rmse

    mcrmse = sum(metrics.values()) / len(metrics)
    metrics["mcrmse"] = mcrmse
    wandb.log(metrics)

    return model_dict, metrics
