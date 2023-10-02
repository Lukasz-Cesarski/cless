"""
CommonLit - Evaluate Student Summaries (CLESS)
https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview

This file stores essential parts of code to
- provide version control in GIT
- upload it to remote machines
- paste it in kaggle notebook (use magic `%%writefile cless.py` command)
- easy code development (code refactor, black, import sort)
"""
import gc
import json
import os
import re
import shutil
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import spacy
import textstat
import torch
import wandb
from datasets import Dataset, disable_progress_bar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed
)
from torch import nn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


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
PREDICTION_LABELS = [f"pred_{t}" for t in TARGET_LABELS]
KEEP_BEST_MODELS = 3


def get_wandb_tags():
    if any(k.startswith("KAGGLE") for k in os.environ.keys()):
        tags = ["kaggle_env"]
    else:
        tags = ["local_env"]
    return tags


def general_setup(free_cublas=False, internet_connection=True):
    warnings.simplefilter("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tqdm.pandas()  # for processor
    disable_progress_bar()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not free_cublas:
        # deberta processing long texts is very sensitive for randomness
        # https://pytorch.org/docs/stable/notes/randomness#avoiding-nondeterministic-algorithms
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    try:
        if any(k.startswith("KAGGLE") for k in os.environ.keys()):
            if internet_connection:
                # in kaggle environment import key from secrets
                from kaggle_secrets import UserSecretsClient

                user_secrets = UserSecretsClient()
                key = user_secrets.get_secret("wandb_api")
                wandb.login(key=key)
        else:
            if internet_connection:
                # locally use env variable to pass key "export WANDB_API_KEY=...."
                wandb.login()
            # locally we need to specify data location
            os.environ[CLESS_DATA_ENV] = "input/"
    except:
        print("Could not log in to WandB")
        exit(1)


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
    hidden_dropout_prob: float = 0.00
    attention_probs_dropout_prob: float = 0.00
    learning_rate: float = 5e-05
    weight_decay: float = 0.0
    batch_size: int = 6
    num_train_epochs: int = 3
    seed: int = 42
    report_to: str = "wandb"
    eval_every: int = 50
    patience: int = 15  # early stopping
    warmup: int = 0
    fp16: Optional[bool] = None

    def __post_init__(self):
        if self.fp16 is None:
            if "large" in self.model_name_or_path:
                self.fp16 = True
            else:
                self.fp16 = False

def tokenize(example, tokenizer, config, labelled=True):
    sep = f" {tokenizer.sep_token} "

    cols = []

    if config.add_prompt_text:
        cols.append("prompt_text")
    if config.add_prompt_text:
        cols.append("prompt_text_short")
    if config.add_prompt_question:
        cols.append("prompt_question")
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
        tmp_dir: str = TMP_DIR,
        dump_dir: str = MODEL_DUMPS_DIR,
    ):
        self.model_name_or_path = model_name_or_path
        self.tmp_dir = tmp_dir
        self.dump_dir = dump_dir

    def train_single_fold(
        self,
        config: Config,
        fold: int,
    ):
        model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        data_collator = DataCollatorWithPadding(tokenizer)

        if config.report_to == "wandb":
            run = wandb.init(
                # reinit=True,
                project=WandbProjects.WANDB_DEBERTA_FOLDS,
                config=asdict(config),
                tags=get_wandb_tags() + [f"fold_{fold}"],
            )
        set_seed(config.seed)
        model_config.update(
            {
                "hidden_dropout_prob": config.hidden_dropout_prob,
                "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
                "cfg": asdict(config),
            }
        )

        ### PRETRAINING ON PSEUDOLABELS ###
        fbp3_path = os.environ.get(CLESS_DATA_ENV, CLESS_DATA_ENV_DEFAULT)
        fbp3_path = Path(fbp3_path) / "feedback-prize-english-language-learning" / "train.csv"
        assert fbp3_path.exists()
        fbp3_data = pd.read_csv(fbp3_path)

        ps_lab_path = "sandbox/datasets/fbp3_pseudolabelling/pseudolabelling.csv"
        ps_lab_path = Path(ps_lab_path)
        assert ps_lab_path.exists()
        ps_lab_data = pd.read_csv(ps_lab_path)

        pretraining_df = pd.merge(fbp3_data, ps_lab_data, left_on="text_id", right_on="student_id")
        pretraining_df = pretraining_df.rename(columns={"full_text": "text"})

        fold_placeholder = -1000
        n_fold = 10
        pretraining_df["fold"] = fold_placeholder
        multifold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=config.seed)
        for n, (train_index, val_index) in enumerate(multifold.split(pretraining_df, pretraining_df[TARGET_LABELS])):
            pretraining_df.loc[val_index, 'fold'] = int(n)
        assert not (pretraining_df["fold"] == fold_placeholder).any()

        pseudo_train_ds = Dataset.from_pandas(pretraining_df[pretraining_df["fold"] != fold])
        pseudo_val_ds = Dataset.from_pandas(pretraining_df[pretraining_df["fold"] == fold])
        pseudo_train_ds = pseudo_train_ds.map(
            tokenize,
            batched=False,
            fn_kwargs={"tokenizer": tokenizer, "config": config},
        )
        pseudo_val_ds = pseudo_val_ds.map(
            tokenize,
            batched=False,
            fn_kwargs={"tokenizer": tokenizer, "config": config},
        )
        training_args = TrainingArguments(
            output_dir=TMP_DIR,
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
            logging_steps=config.eval_every,
            seed=config.seed,
            fp16=config.fp16,
            warmup_steps=200,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path, config=model_config
        )

        pseudo_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=pseudo_train_ds,
            eval_dataset=pseudo_val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_mcrmse_for_trainer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        pseudo_trainer.train()

        ### TRAINING ON COMPETITION DATA ###

        train_pro, test_pro, train_sum, test_sum = read_cless_data()
        df = train_pro.merge(train_sum, on="prompt_id")
        assert set(ID2FOLD.keys()) == set(df["prompt_id"].unique())
        df["fold"] = df["prompt_id"].map(ID2FOLD)
        train_ds = Dataset.from_pandas(df[df["fold"] != fold])
        val_ds = Dataset.from_pandas(df[df["fold"] == fold])

        train_ds = train_ds.map(
            tokenize,
            batched=False,
            fn_kwargs={"tokenizer": tokenizer, "config": config},
        )
        val_ds = val_ds.map(
            tokenize,
            batched=False,
            fn_kwargs={"tokenizer": tokenizer, "config": config},
        )

        # dry run
        if os.environ.get(CLESS_DRY_RUN) == "1":
            dry_run_ds = train_ds.train_test_split(None, 20)
            train_ds = dry_run_ds["train"]

            dry_run_ds = val_ds.train_test_split(None, 10)
            val_ds = dry_run_ds["train"]

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_mcrmse_for_trainer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
        )
        trainer.train()
        eval_res = trainer.predict(test_dataset=val_ds)

        model_fold_dir = os.path.join(self.dump_dir, str(fold))
        os.makedirs(model_fold_dir, exist_ok=True)
        trainer.save_model(output_dir=model_fold_dir)
        if config.report_to == "wandb":
            run.finish()
        shutil.rmtree(self.tmp_dir)

        return eval_res

    def predict_single_fold(self, df):
        model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        data_collator = DataCollatorWithPadding(tokenizer)

        config_kwargs = {
            k: v for k, v in model_config.cfg.items() if k not in [
                # legacy keywords
                "add_prompt_text_short",
            ]
        }

        config = Config(**config_kwargs)
        test_ds = Dataset.from_pandas(df)
        test_ds = test_ds.map(
            tokenize,
            fn_kwargs={
                "tokenizer": tokenizer,
                "config": config,
                "labelled": False,
            },
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            config=model_config,
        )
        model.eval()

        test_args = TrainingArguments(
            output_dir=self.tmp_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=config.batch_size,
            dataloader_drop_last=False,
        )
        infer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=test_args,
        )
        predictions = infer.predict(test_dataset=test_ds)

        return predictions


def cless_ensamble_train(config: Config):
    fold_results = {}
    ensamble_start = datetime.now().isoformat()[:-7]
    dump_dir = os.path.join(
        MODEL_DUMPS_DIR,
        config.model_name_or_path.replace("/", "_"),
        ensamble_start,
    )

    for fold in range(4):
        print(f">>> Training fold {fold}:")
        cless_model = ClessModel(
            model_name_or_path=config.model_name_or_path,
            dump_dir=dump_dir,
        )
        eval_res = cless_model.train_single_fold(
            fold=fold,
            config=config,
        )
        torch.cuda.empty_cache()
        gc.collect()

        fold_results[fold] = eval_res

    # evaluation
    fold_results_log = {f"f{k}": v.metrics for k, v in fold_results.items()}
    mean_metrics = {}
    for metric in tuple(fold_results_log.values())[0].keys():
        metric_values = []
        for fold in fold_results_log:
            metric_values.append(fold_results_log[fold][metric])
        mean_metrics[metric] = np.mean(metric_values)

    fold_results_log["macro"] = mean_metrics

    flat_p = np.vstack([v.predictions for v in fold_results.values()])
    flat_l = np.vstack([v.label_ids for v in fold_results.values()])

    fold_results_log["micro"] = {
        f"test_{k}": v for k, v in compute_mcrmse_for_trainer((flat_p, flat_l)).items()
    }
    pprint(fold_results_log)
    if config.report_to == "wandb":
        run = wandb.init(
            # Set the project where this run will be logged
            project=WandbProjects.WANDB_DEBERTA_ENSAMBLE,
            # Track hyperparameters and run metadata
            config=asdict(config),
            tags=get_wandb_tags(),
        )
        run.log(data=fold_results_log)
        run.finish()

    new_dump_dir = os.path.join(
        os.path.dirname(dump_dir),
        str(fold_results_log["micro"]["test_mcrmse"])
        + "__"
        + os.path.basename(dump_dir),
    )

    os.rename(dump_dir, new_dump_dir)

    return fold_results, fold_results_log, new_dump_dir


def cless_ensamble_sweep(cli_config: Config):
    cli_config.report_to = "none"
    default_params = asdict(cli_config)
    wandb.init(
        config=default_params,
        reinit=True,
        project=WandbProjects.WANDB_DEBERTA_SWEEPS,
        tags=get_wandb_tags(),
    )
    config = Config(**wandb.config)

    fold_results, fold_results_log, new_dump_dir = cless_ensamble_train(
        config=config
    )

    wandb.log(fold_results_log)

    # remove additional directories (worst runs)
    models_home = os.path.dirname(new_dump_dir)
    models_registry = {path: path.split("__") for path in os.listdir(models_home)}
    models_registry_fold_remove = sorted(
        models_registry.items(), key=lambda x: float(x[1][0]), reverse=False
    )[KEEP_BEST_MODELS:]

    for dir_name, _ in models_registry_fold_remove:
        full_dir_name = os.path.join(models_home, dir_name)
        shutil.rmtree(full_dir_name)


def cless_single_fold_predict(fold_subdir, df):
    """For cross-validation / LGBM training"""
    assert os.path.isdir(fold_subdir)
    cless_deberta = ClessModel(
        model_name_or_path=fold_subdir,
    )
    fold_prediction = cless_deberta.predict_single_fold(df)
    partial_prediction = pd.concat(
        [
            df["student_id"],
            pd.DataFrame(
                fold_prediction.predictions,
                columns=TARGET_LABELS,
                index=df.index,
            ),
        ],
        axis=1,
    )
    return partial_prediction


def cless_ensamble_predict_train(models_home):
    train_pro, test_pro, train_sum, test_sum = read_cless_data()
    df = train_pro.merge(train_sum, on="prompt_id")
    assert set(ID2FOLD.keys()) == set(df["prompt_id"].unique())
    df["fold"] = df["prompt_id"].map(ID2FOLD)

    fold_predictions = []
    for fold in range(4):
        fold_subdir = os.path.join(models_home, str(fold))
        df_fold = df[df["fold"] == fold]
        fold_prediction = cless_single_fold_predict(fold_subdir, df_fold)
        fold_predictions.append(fold_prediction)

    df_prediction = pd.concat(fold_predictions, axis=0).rename(
        columns={t: p for t, p in zip(TARGET_LABELS, PREDICTION_LABELS)}
    )

    df_metrics = pd.merge(df, df_prediction, on="student_id", how="inner")
    if len(df_metrics) != len(df):
        raise ValueError("Input vs prediction data missmatch")

    metrics = compute_mcrmse_for_trainer(
        (df_metrics[PREDICTION_LABELS].values, df_metrics[TARGET_LABELS].values)
    )

    return df_metrics, metrics


def cless_ensamble_predict_test(models_home):
    train_pro, test_pro, train_sum, test_sum = read_cless_data()
    df = test_pro.merge(test_sum, on="prompt_id")

    # run models
    fold_predictions = []
    for fold in range(4):
        fold_subdir = os.path.join(models_home, str(fold))
        fold_prediction = cless_single_fold_predict(fold_subdir, df)
        fold_predictions.append(fold_prediction)

    # calculate mean of folds
    submission_targets = []
    for target in TARGET_LABELS:
        stacked_df = pd.concat([p[target] for p in fold_predictions], axis=1)
        mean_df = stacked_df.mean(axis=1)
        mean_df.name = target
        submission_targets.append(mean_df)
    submission_targets.append(fold_predictions[0]["student_id"])
    submission_df = pd.concat(submission_targets, axis=1)[
        ["student_id"] + TARGET_LABELS
    ]

    return submission_df


class CPMPSpellchecker:
    """
    All credits goes for:
    https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/433451
    https://www.kaggle.com/code/atamazian/spell-checker-using-word2vec-updated-work-by-cpmp/notebook
    https://www.kaggle.com/code/cpmpml/spell-checker-using-word2vec
    """

    def __init__(self, words_path=None):
        if words_path is None:
            data_home = os.environ.get(CLESS_DATA_ENV)
            if data_home is None:
                # default kaggle location
                data_home = CLESS_DATA_ENV_DEFAULT
            data_home = Path(data_home)
            if not data_home.is_dir():
                raise FileNotFoundError(f"Directory not found! {data_home}")

            word_probs_path = data_home / "word-probs" / "word_probs.json"
        else:
            word_probs_path = words_path

        assert word_probs_path.exists()

        with open(word_probs_path) as fr:
            word_probs = json.load(fr)
        self.word_probs = word_probs

    @staticmethod
    def words(text):
        return re.findall(r"\w+", text)

    def P(self, word):
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return -self.word_probs.get(word, 0)

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (
            self.known([word])
            or self.known(self.edits1(word))
            or self.known(self.edits2(word))
            or [word]
        )

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.word_probs)

    @staticmethod
    def edits1(word):
        "All edits that are one edit away from `word`."
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def __call__(self, text):
        words = self.words(text)
        known_words = self.known(words)
        swap_dict = {w: self.correction(w) for w in words if w not in known_words}

        correct_text = text
        for incor_w, corr_w in swap_dict.items():
            correct_text = re.sub(
                r"\b{}\b".format(re.escape(incor_w)), corr_w, correct_text
            )
        return correct_text


class Preprocessor:
    def __init__(self) -> None:
        self.STOP_WORDS = set(stopwords.words("english"))
        self.cpmpspellchecker = CPMPSpellchecker()
        self.nlp = spacy.load("en_core_web_lg")

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
        # character-level ngrams
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

    def spelling(self, text: str) -> int:
        words = self.cpmpspellchecker.words(text)
        know_words = self.cpmpspellchecker.known(words)
        unknown = [w for w in words if w not in know_words]
        return len(unknown)

    def add_spelling_dictionary(self, tokens: List[str]):
        """dictionary update for spellchecker"""
        for t in tokens:
            if t not in self.cpmpspellchecker.word_probs:
                self.cpmpspellchecker.word_probs[t] = 0  # put them on the very top

    def get_lemm_tokens(self, row, column_name, remove_stop=False, remove_punct=False):
        spacy_obj = row[column_name]
        result = []
        for t in spacy_obj:
            if remove_stop and t.is_stop:
                continue
            if remove_punct and t.is_punct:
                continue
            result.append(t.lemma_)
        return result

    def lemma_overlap_count(self, row):
        prompt_lemma = self.get_lemm_tokens(
            row, "prompt_text_spacy", remove_stop=True, remove_punct=True
        )
        summary_lemma = self.get_lemm_tokens(
            row, "fixed_summary_text_spacy", remove_stop=True, remove_punct=True
        )

        return len(set(prompt_lemma).intersection(set(summary_lemma)))

    def shingles_overlap(self, row, remove_stop):
        prompt_tokens = self.get_lemm_tokens(
            row, "prompt_text_spacy", remove_stop=remove_stop, remove_punct=True
        )
        prompt_shingles = set(zip(prompt_tokens, prompt_tokens[1:]))

        summary_tokens = self.get_lemm_tokens(
            row, "fixed_summary_text_spacy", remove_stop=remove_stop, remove_punct=True
        )
        summary_shingles = set(zip(summary_tokens, summary_tokens[1:]))
        res = prompt_shingles.intersection(summary_shingles)
        return len(res)

    def get_NER(self, doc):
        BANNED_ENTITIES = {"ORDINAL", "CARDINAL", "PERCENT", "DATE", "TIME"}
        entities = {
            ent.text.strip() for ent in doc.ents if ent.label_ not in BANNED_ENTITIES
        }
        return entities

    def ner_co_occurence(self, row):
        prompt_ner = row["prompt_ner"]
        summaries_ner = row["summaries_ner"]
        return len(prompt_ner.intersection(summaries_ner))

    def run(
        self,
        prompts: pd.DataFrame,
        summaries: pd.DataFrame,
    ) -> pd.DataFrame:

        ### PROMPTS ###

        # before merge preprocess
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: word_tokenize(x)
        )
        prompts["prompt_length"] = prompts["prompt_tokens"].apply(len)
        # Add prompt tokens into spelling checker dictionary
        prompts["prompt_tokens"].apply(lambda x: self.add_spelling_dictionary(x))
        prompts["prompt_text_spacy"] = prompts["prompt_text"].apply(
            lambda x: self.nlp(x)
        )
        prompts["prompt_ner"] = prompts["prompt_text_spacy"].apply(
            lambda x: self.get_NER(x)
        )

        ### SUMMARIES ###

        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: word_tokenize(x)
        )
        summaries["summary_length"] = summaries["summary_tokens"].apply(len)
        # count misspelling
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)

        #         from IPython.core.debugger import Pdb; Pdb().set_trace()
        # fix misspelling
        summaries["fixed_summary_text"] = summaries["text"].progress_apply(
            lambda x: self.cpmpspellchecker(x)
        )
        summaries["fixed_summary_text_spacy"] = summaries[
            "fixed_summary_text"
        ].progress_apply(lambda x: self.nlp(x))
        summaries["summaries_ner"] = summaries["fixed_summary_text_spacy"].apply(
            lambda x: self.get_NER(x)
        )

        ### MERGED ###

        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # after merge preprocess
        input_df["length_ratio"] = (
            input_df["summary_length"] / input_df["prompt_length"]
        )
        input_df["word_overlap_count"] = input_df.progress_apply(
            self.word_overlap_count, axis=1
        )
        input_df["lemma_overlap_count"] = input_df.progress_apply(
            self.lemma_overlap_count, axis=1
        )
        # ngrams (char level)
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
        # shingle (word level)
        input_df["shingle_overlap_stop_on"] = input_df.progress_apply(
            lambda x: self.shingles_overlap(x, remove_stop=False), axis=1
        )
        input_df["shingle_overlap_stop_on_ratio"] = input_df[
            "shingle_overlap_stop_on"
        ] / (input_df["summary_length"] - 1)
        input_df["shingle_overlap_stop_off"] = input_df.progress_apply(
            lambda x: self.shingles_overlap(x, remove_stop=True), axis=1
        )
        input_df["shingle_overlap_stop_off_ratio"] = input_df[
            "shingle_overlap_stop_off"
        ] / (input_df["summary_length"] - 1)
        input_df["ner_co_occurence"] = input_df.apply(self.ner_co_occurence, axis=1)

        textstat_functions = [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "smog_index",
            "coleman_liau_index",
            "automated_readability_index",
            "dale_chall_readability_score",
            "difficult_words",
            "linsear_write_formula",
            "gunning_fog",
            # "text_standard", # output as float needed
            "fernandez_huerta",
            "szigriszt_pazos",
            "gutierrez_polini",
            "crawford",
            "gulpease_index",
            "osman",
        ]
        for txt_func in textstat_functions:
            input_df[txt_func] = input_df["text"].apply(getattr(textstat, txt_func))
        input_df["text_standard"] = input_df["text"].apply(
            lambda x: textstat.text_standard(x, float_output=True)
        )

        input_df["quotes_count"] = input_df.progress_apply(self.quotes_count, axis=1)

        input_df["fold"] = input_df["prompt_id"].map(ID2FOLD)

        drop_columns = ["summary_tokens", "prompt_tokens"]

        return input_df.drop(columns=drop_columns)


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


def train_lgbm(train, targets, drop_columns):
    # https://colab.research.google.com/drive/181GCGp36_75C2zm7WLxr9U2QjMXXoibt#scrollTo=aIhxl7glaJ5k
    # defaults
    default_params = {
        "random_state": 42,
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,  # ---sweep
        "boosting": "gbdt",
        "num_leaves": 31,  # ---sweep
        "max_depth": -1,  # ---sweep
        "min_data_in_leaf": 10,  # ---sweep
        "min_data_in_bin": 5,  # like in OLD
        "lambda_l1": 0.0,  # default
        "lambda_l2": 0.0,  # default
        "max_bin": 127,  # takie mid
        "feature_fraction": 1.0,  # check
        "bagging_fraction": 1.0,  # check
    }
    # Initialize a new wandb run
    wandb.init(
        config=default_params,
        reinit=True,
        project=WandbProjects.WANDB_LGBM_FOLDS,
        tags=get_wandb_tags(),
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
            hyperparameters = {k: v for k, v in wandb.config.items()}

            model = lgb.train(
                hyperparameters,
                num_boost_round=10000,
                # categorical_feature = categorical_features,
                valid_names=["train_single_fold", "valid"],
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

            pred = model.predict(X_eval_cv)

            trues.extend(y_eval_cv)
            preds.extend(pred)

        rmse = np.sqrt(mean_squared_error(trues, preds))
        metrics[target] = rmse

    mcrmse = sum(metrics.values()) / len(metrics)
    metrics["mcrmse"] = mcrmse
    wandb.log(metrics)

    return model_dict, metrics
