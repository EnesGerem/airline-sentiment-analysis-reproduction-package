"""
Shared data loading utilities for all model training scripts.
Provides CSV reading, vocabulary building, and PyTorch Dataset classes.
"""

import csv
import os
import random
from collections import Counter

import numpy as np

from src.utils.config import (
    SPLITS_DIR, LABEL_MAP, SEED,
    MAX_VOCAB_SIZE, MAX_SEQ_LENGTH_TWEET, MAX_SEQ_LENGTH_REVIEW,
)

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_split(source_config: str, split: str) -> tuple[list[str], list[int]]:
    """
    Load a split CSV and return (texts, labels).

    Args:
        source_config: e.g. "twitter", "skytrax+airlinequality", "twitter+skytrax+airlinequality"
        split: "train", "val", or "test"

    Returns:
        (texts, labels) where labels are integer-encoded via LABEL_MAP
    """
    path = os.path.join(SPLITS_DIR, f"{source_config}_{split}.csv")
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(LABEL_MAP[row["label"]])
    return texts, labels


def get_max_seq_length(source_config: str) -> int:
    """Return appropriate max sequence length based on whether config includes only tweets."""
    if source_config == "twitter":
        return MAX_SEQ_LENGTH_TWEET
    return MAX_SEQ_LENGTH_REVIEW


def has_twitter(source_config: str) -> bool:
    return "twitter" in source_config


# ---------------------------------------------------------------------------
# Vocabulary (for DL models: CNN, LSTM, GRU)
# ---------------------------------------------------------------------------

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def build_vocab(texts: list[str], max_size: int = MAX_VOCAB_SIZE) -> dict[str, int]:
    """
    Build word-to-index mapping from training texts.
    Reserves index 0 for PAD, 1 for UNK.
    """
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(max_size - 2)  # reserve PAD, UNK
    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)
    return word2idx


def texts_to_sequences(texts: list[str], word2idx: dict[str, int],
                       max_len: int) -> np.ndarray:
    """Convert texts to padded integer sequences."""
    sequences = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = text.split()[:max_len]
        for j, token in enumerate(tokens):
            sequences[i, j] = word2idx.get(token, UNK_IDX)
    return sequences


def load_glove_embeddings(glove_path: str, word2idx: dict[str, int],
                          embed_dim: int = 300) -> np.ndarray:
    """
    Load pre-trained GloVe embeddings for vocab words.
    Returns embedding matrix of shape (vocab_size, embed_dim).
    Words not in GloVe get random initialization.
    """
    vocab_size = len(word2idx)
    rng = np.random.RandomState(SEED)
    embeddings = rng.normal(0, 0.1, (vocab_size, embed_dim)).astype(np.float32)
    embeddings[PAD_IDX] = 0.0  # PAD stays zero

    found = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word2idx:
                idx = word2idx[word]
                embeddings[idx] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"[INFO] GloVe: {found}/{vocab_size} words found ({found/vocab_size*100:.1f}%)")
    return embeddings


# ---------------------------------------------------------------------------
# Compute class weights for imbalanced data
# ---------------------------------------------------------------------------

def compute_class_weights(labels: list[int], num_classes: int = 3) -> list[float]:
    """Compute inverse-frequency class weights."""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        w = total / (num_classes * counts[c]) if counts[c] > 0 else 1.0
        weights.append(w)
    return weights


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
