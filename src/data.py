import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    dataset_csv, numeric_features, data_processed, data_raw, label_map,
    max_seq_len, max_vocab_size, glove_dim, glove_file,
)


def download_dataset():
    if os.path.exists(dataset_csv):
        return

    print("[*] Dataset not found, downloading from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install datasets: pip install datasets")

    os.makedirs(data_raw, exist_ok=True)
    ds = load_dataset("airt-ml/twitter-human-bots", split="train")
    ds.to_csv(dataset_csv)
    print(f"[+] Downloaded {len(ds):,} rows to {dataset_csv}")


def load_and_preprocess():
    print("[*] Loading dataset...")
    download_dataset()

    if not os.path.exists(dataset_csv):
        raise FileNotFoundError("Dataset not found and download failed.")

    df = pd.read_csv(dataset_csv)
    print(f"    Rows: {len(df):,}")

    df["label"] = df["account_type"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    bot_count = df["label"].sum()
    print(f"    Bots: {bot_count:,}  |  Humans: {len(df) - bot_count:,}")

    df["followers_count"] = df["followers_count"].fillna(0).astype(float)
    df["friends_count"] = df["friends_count"].fillna(0).astype(float)
    df["statuses_count"] = df["statuses_count"].fillna(0).astype(float)
    df["favourites_count"] = df["favourites_count"].fillna(0).astype(float)
    df["account_age_days"] = df["account_age_days"].fillna(365).astype(float).clip(lower=1)
    df["average_tweets_per_day"] = df["average_tweets_per_day"].fillna(0).astype(float)
    df["verified"] = df["verified"].fillna(False).astype(int)
    df["default_profile"] = df["default_profile"].fillna(False).astype(int)
    df["default_profile_image"] = df["default_profile_image"].fillna(False).astype(int)
    df["description"] = df["description"].fillna("")
    df["screen_name"] = df["screen_name"].fillna("")
    df["location"] = df["location"].fillna("")

    df["followers_to_friends_ratio"] = df["followers_count"] / df["friends_count"].clip(lower=1)
    df["favourites_to_statuses_ratio"] = df["favourites_count"] / df["statuses_count"].clip(lower=1)
    df["friends_to_followers_ratio"] = df["friends_count"] / df["followers_count"].clip(lower=1)
    df["statuses_to_followers_ratio"] = df["statuses_count"] / df["followers_count"].clip(lower=1)

    df["has_description"] = (df["description"].str.len() > 0).astype(int)
    df["has_location"] = (df["location"].str.len() > 0).astype(int)
    df["description_length"] = df["description"].str.len()
    df["screen_name_length"] = df["screen_name"].str.len()
    df["profile_completeness"] = (
        df["has_description"] + df["has_location"]
        + (1 - df["default_profile"]) + (1 - df["default_profile_image"])
        + df["verified"]
    )

    df["screen_name_digits"] = df["screen_name"].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df["screen_name_digit_ratio"] = df["screen_name_digits"] / df["screen_name_length"].clip(lower=1)
    df["screen_name_has_underscore"] = df["screen_name"].str.contains("_", na=False).astype(int)

    df["tweets_per_follower"] = df["statuses_count"] / df["followers_count"].clip(lower=1)
    df["tweets_per_day_per_follower"] = df["average_tweets_per_day"] / df["followers_count"].clip(lower=1)

    df["bio_url_count"] = df["description"].str.count(r"http|www\.|\.com|\.net")
    df["bio_hashtag_count"] = df["description"].str.count(r"#")
    df["bio_mention_count"] = df["description"].str.count(r"@")
    df["bio_word_count"] = df["description"].str.split().str.len().fillna(0).astype(int)

    news_pattern = r"\b(?:news|breaking|daily|magazine|journal|times|herald|tribune|gazette|broadcast|media|press|reporter|journalist|editor|anchor|correspondent|coverage|headlines|report)\b"
    org_pattern = r"\b(?:official|corp|inc\.?|llc|ltd|company|brand|store|shop|support|customer|service|team|foundation|organisation|organization|ngo|charity)\b"
    df["bio_has_news_keywords"] = df["description"].str.lower().str.contains(news_pattern, regex=True, na=False).astype(int)
    df["bio_has_org_keywords"] = df["description"].str.lower().str.contains(org_pattern, regex=True, na=False).astype(int)
    df["bio_likely_organisation"] = (
        (df["bio_has_news_keywords"] | df["bio_has_org_keywords"])
        & (df["followers_count"] > 1000)
        & (df["account_age_days"] > 365)
    ).astype(int)
    df["is_established_account"] = (
        (df["verified"] == 1)
        & (df["followers_count"] > 10000)
        & (df["account_age_days"] > 365)
    ).astype(int)

    df["log_followers_count"] = np.log1p(df["followers_count"])
    df["log_friends_count"] = np.log1p(df["friends_count"])
    df["log_statuses_count"] = np.log1p(df["statuses_count"])
    df["log_favourites_count"] = np.log1p(df["favourites_count"])
    df["log_tweets_per_follower"] = np.log1p(df["tweets_per_follower"])
    df["log_followers_to_friends_ratio"] = np.log1p(df["followers_to_friends_ratio"])

    print(f"    Engineered {len(numeric_features)} features")

    texts = []
    for _, row in df.iterrows():
        desc = str(row.get("description", "") or "")
        desc = re.sub(r"http\S+", "<URL>", desc)
        desc = re.sub(r"\s+", " ", desc).strip()
        texts.append(desc if desc else "<EMPTY>")

    n = len(df)
    indices = np.random.RandomState(42).permutation(n)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    user_ids = [str(i) for i in range(n)]
    splits = {
        "train": [user_ids[i] for i in indices[:train_end]],
        "val":   [user_ids[i] for i in indices[train_end:val_end]],
        "test":  [user_ids[i] for i in indices[val_end:]],
    }

    print(f"    Split: train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")
    print(f"[+] Preprocessed {n:,} users")

    return df, texts, df["label"].values, splits, user_ids


def save_processed(df, texts, labels, splits, user_ids):
    os.makedirs(data_processed, exist_ok=True)

    numeric_values = df[numeric_features].values.astype(np.float32)

    train_indices = [int(uid) for uid in splits["train"]]
    train_numeric = numeric_values[train_indices]
    mean = train_numeric.mean(axis=0)
    std = train_numeric.std(axis=0)
    std[std == 0] = 1.0

    numeric_normalised = (numeric_values - mean) / std
    print(f"    Normalised {len(numeric_features)} features (fitted on train split)")

    torch.save({
        "texts": texts,
        "labels": torch.tensor(labels, dtype=torch.long),
        "user_ids": user_ids,
        "splits": splits,
        "numeric_features": torch.tensor(numeric_normalised, dtype=torch.float32),
        "numeric_mean": mean,
        "numeric_std": std,
    }, os.path.join(data_processed, "processed.pt"))

    print(f"[+] Saved to {data_processed}/")


def load_processed():
    path = os.path.join(data_processed, "processed.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run: python src/data.py  (no data at {path})")
    return torch.load(path, weights_only=False)


class TwiBotDataset(Dataset):
    def __init__(self, texts, numeric_features, labels, user_ids, split_ids=None):
        if split_ids is not None:
            id_set = set(split_ids)
            indices = [i for i, uid in enumerate(user_ids) if uid in id_set]
        else:
            indices = list(range(len(user_ids)))

        self.texts = [texts[i] for i in indices]
        self.numeric = numeric_features[indices]
        self.labels = labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "numeric": self.numeric[idx], "label": self.labels[idx]}


def create_datasets(data, splits):
    texts = data["texts"]
    numeric = data["numeric_features"]
    labels = data["labels"]
    user_ids = data["user_ids"]

    train = TwiBotDataset(texts, numeric, labels, user_ids, splits.get("train"))
    val   = TwiBotDataset(texts, numeric, labels, user_ids, splits.get("val", splits.get("valid")))
    test  = TwiBotDataset(texts, numeric, labels, user_ids, splits.get("test"))

    print(f"[*] Datasets: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def _tokenize_text(text):
    text = text.lower()
    text = re.sub(r"<URL>", " url ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class GloveVocab:
    def __init__(self, word2idx=None):
        self.word2idx = word2idx or {"<PAD>": 0, "<UNK>": 1}

    @property
    def vocab_size(self):
        return len(self.word2idx)

    @classmethod
    def build_from_corpus(cls, texts, max_vocab=max_vocab_size):
        counter = Counter()
        for text in texts:
            counter.update(_tokenize_text(text))
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in counter.most_common(max_vocab - 2):
            word2idx[word] = len(word2idx)
        vocab = cls(word2idx)
        print(f"[+] Vocabulary: {vocab.vocab_size:,} words")
        return vocab

    def tokenize_batch(self, texts, max_len=max_seq_len):
        batch = []
        for text in texts:
            tokens = _tokenize_text(text)
            ids = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
            ids += [0] * (max_len - len(ids))
            batch.append(ids)
        return torch.tensor(batch, dtype=torch.long)

    def load_glove_embeddings(self, path=glove_file):
        print(f"[*] Loading GloVe from {os.path.basename(path)}...")
        glove = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if parts[0] in self.word2idx:
                    glove[parts[0]] = np.array(parts[1:], dtype=np.float32)

        matrix = np.random.normal(scale=0.6, size=(self.vocab_size, glove_dim)).astype(np.float32)
        matrix[0] = 0.0
        found = sum(1 for w in self.word2idx if w in glove)
        print(f"[+] GloVe coverage: {found:,}/{self.vocab_size:,} ({found/self.vocab_size*100:.1f}%)")
        for w, i in self.word2idx.items():
            if w in glove:
                matrix[i] = glove[w]
        return matrix

    def random_embeddings(self):
        matrix = np.random.normal(scale=0.6, size=(self.vocab_size, glove_dim)).astype(np.float32)
        matrix[0] = 0.0
        print(f"[*] Using random embeddings ({self.vocab_size:,} x {glove_dim})")
        return matrix

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))


if __name__ == "__main__":
    df, texts, labels, splits, user_ids = load_and_preprocess()
    save_processed(df, texts, labels, splits, user_ids)
