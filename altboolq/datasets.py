import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import BatchEncoding

try:
    import datasets as hf_datasets
except ImportError:
    pass


def get_target(labels, overflow_to_sample):
    labels = torch.tensor(labels, dtype=torch.long)
    return labels[overflow_to_sample]


class BoolQDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, stride):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df.loc[idx, "question"]
        passage = self.df.loc[idx, "passage"]
        label = self.df.loc[idx, "label"]
        return question, passage, label

    def get_collate_fn(self):
        def collate_fn(examples):
            questions, passages, labels = [list(a) for a in zip(*examples)]
            inputs = self.tokenizer(
                passages,
                questions,
                add_special_tokens=True,
                padding=True,
                truncation="only_first",
                return_overflowing_tokens=True,
                max_length=self.max_len,
                stride=self.stride,
                return_tensors="pt",
            )
            targets = get_target(labels, inputs.overflow_to_sample_mapping)
            targets = targets.unsqueeze(-1)
            return len(examples), inputs, targets, labels

        return collate_fn


def split_batch(x, size):
    items = []
    for k, v in x.items():
        v_split = torch.split(v, size)
        items.append(list(zip([k for _ in range(len(v_split))], v_split)))
    items = [BatchEncoding({k: v for k, v in batch}) for batch in zip(*items)]
    return items


def to_device(example, device):
    batch_size, x, y, labels = example
    return batch_size, x.to(device), y.to(device), labels


def score(preds_batch, labels_batch):
    confusion = np.zeros((2, 2))
    for pred, label in zip(preds_batch, labels_batch):
        confusion[pred, label] += 1

    return confusion


def get_open_df(df):
    df = df.copy(deep=True)
    df["passage"] = df["passage"].apply(lambda _: "")
    return df


def get_dfs(path=None):
    boolq = hf_datasets.load_dataset("super_glue", "boolq")
    dfs = {
        "boolq_train": boolq["train"].to_pandas(),
        "boolq_valid": boolq["validation"].to_pandas(),
        "boolq_test": boolq["test"].to_pandas(),
    }
    dfs["open_boolq_train"] = get_open_df(dfs["boolq_train"])
    dfs["open_boolq_valid"] = get_open_df(dfs["boolq_valid"])
    dfs["open_boolq_test"] = get_open_df(dfs["boolq_test"])

    if path is not None:
        path = Path(path)
        for sub_path in path.iterdir():
            data_path = sub_path / "data.csv"
            if not data_path.exists():
                continue
            name = sub_path.name
            dfs[name] = pd.read_csv(data_path)

    return dfs


def write_df(config, df):
    path = Path(config["name"])
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    df.to_csv(path / "data.csv", index=False)
