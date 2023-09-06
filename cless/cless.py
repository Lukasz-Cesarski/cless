import os
import re
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from autocorrect import Speller
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
from wandb.lightgbm import wandb_callback, log_summary


tqdm.pandas()

PRO_TRAIN_FILE = "prompts_train.csv"
PRO_TEST_FILE = "prompts_test.csv"
SUM_TRAIN_FILE = "summaries_train.csv"
SUM_TEST_FILE = "summaries_test.csv"

WANDB_FOLDS_PROJECT = "cless-folds"
WANDB_ENSAMBLE_PROJECT = "cless-ensamble"
WANDB_LGBM_FOLDS_PROJECT = "cless-lgbm-folds"
WANDB_LGBM_ENSAMBLE_PROJECT = "cless-lgbm-ensamble"

ID2FOLD = {
    "814d6b": 0,
    "39c16e": 1,
    "3b9047": 2,
    "ebad26": 3,
}


def read_data(
    input_data_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    data_dir = Path(input_data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found! {data_dir}")

    train_pro = pd.read_csv(data_dir / PRO_TRAIN_FILE)
    test_pro = pd.read_csv(data_dir / PRO_TEST_FILE)
    train_sum = pd.read_csv(data_dir / SUM_TRAIN_FILE)
    test_sum = pd.read_csv(data_dir / SUM_TEST_FILE)

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
    num_proc: Optional[int] = field(
        default=4,
        metadata={"help": "Number of processes to tokenize dataset"},
    )
    hidden_dropout_prob: Optional[float] = field(
        default=0.05,
        metadata={"help": "Amount of dropout to apply (hidden)"},
    )
    attention_probs_dropout_prob: Optional[float] = field(
        default=0.05,
        metadata={"help": "Amount of dropout to apply (attention)"},
    )
    learning_rate: Optional[float] = field(
        default=5e-05,
        metadata={"help": "Learning rate"},
    )
    weight_decay: Optional[float] = field(
        default=0.0,
        metadata={"help": "Weight decay"},
    )
    batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Size of the batch"},
    )
    num_train_epochs: Optional[int] = field(
        default=3,
        metadata={"help": "Number of training epochs"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"},
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={"help": "Where to log results"},
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


class ClessDeberta:
    def __init__(
        self,
        model_name_or_path: str,
        tmp_dir: str,
        dump_dir: str,
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

    @staticmethod
    def tokenize(
        example,
        tokenizer,
        config,
        labelled=True,
    ):
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

    def train(
        self,
        config: Config,
        fold: int,
    ):
        run = wandb.init(
            reinit=True,
            project=WANDB_FOLDS_PROJECT,
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

        train_pro, test_pro, train_sum, test_sum = read_data(config.data_dir)
        df = train_pro.merge(train_sum, on="prompt_id")
        assert set(ID2FOLD.keys()) == set(df["prompt_id"].unique())
        df["fold"] = df["prompt_id"].map(ID2FOLD)
        train_ds = Dataset.from_pandas(df[df["fold"] != fold])
        val_ds = Dataset.from_pandas(df[df["fold"] == fold])

        train_ds = train_ds.map(
            self.tokenize,
            batched=False,
            num_proc=config.num_proc,
            fn_kwargs={"tokenizer": self.tokenizer, "config": config},
        )
        val_ds = val_ds.map(
            self.tokenize,
            batched=False,
            num_proc=config.num_proc,
            fn_kwargs={"tokenizer": self.tokenizer, "config": config},
        )

        if os.environ.get("CLESS_DRY_RUN") == "1":
            dry_run_ds = train_ds.train_test_split(None, 20)
            train_ds = dry_run_ds["train"]

            dry_run_ds = val_ds.train_test_split(None, 10)
            val_ds = dry_run_ds["train"]

        STEPS_VAL = 50
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
            eval_steps=STEPS_VAL,
            save_steps=STEPS_VAL,
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
            compute_metrics=compute_mcrmse,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )
        trainer.train()
        eval_res = trainer.evaluate()

        model_fold_dir = os.path.join(
            self.dump_dir,
            self.model_name_or_path.replace("/", "_"),
            str(fold),
        )
        os.makedirs(model_fold_dir, exist_ok=True)
        trainer.save_model(output_dir=model_fold_dir)
        run.finish()
        shutil.rmtree(self.tmp_dir)

        return eval_res

    def predict(self, df):
        config = Config(**self.model_config.cfg)
        test_ds = Dataset.from_pandas(df)
        test_ds = test_ds.map(
            tokenize,
            batched=False,
            num_proc=config.num_proc,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "config": config,
                "labelled": False,
            },
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            config=self.model_config,
            ignore_mismatched_sizes=True,
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


def cless_predict(folds_dir, df):
    predictions = {}
    for fold in range(4):
        fold_path = os.path.join(folds_dir, str(fold))
        print(fold_path)
        assert os.path.isdir(fold_path)
        cless_deberta = ClessDeberta(
            model_name_or_path=fold_path, tmp_dir="tmp/", dump_dir="model_dumps/"
        )
        fold_prediction = cless_deberta.predict(df)
        partial_submission = pd.concat(
            [
                df["student_id"],
                pd.DataFrame(
                    fold_prediction.predictions, columns=["content", "wording"]
                ),
            ],
            axis=1,
        )
        predictions[fold] = partial_submission
    return predictions


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
        project=WANDB_LGBM_FOLDS_PROJECT,
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
                              valid_names=['train', 'valid'],
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
