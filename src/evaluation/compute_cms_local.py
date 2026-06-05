"""
Compute in-domain confusion matrices for TF-IDF and DL models.

Runs locally on CPU (transformers go to Colab via notebooks/06_transformer_cms.ipynb).
Outputs one .npy file (3x3 int array) per (model, source_config) under
results/confusion_matrices/{model}__{source_config}.npy

Class order: 0=negative, 1=neutral, 2=positive.

Resumable: skips configs whose CM file already exists.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from src.utils.config import MODELS_DIR
from src.utils.data_utils import (
    load_split, build_vocab, texts_to_sequences, get_max_seq_length,
)
from src.models.train_dl import create_model
from src.utils.config import EMBEDDING_DIM, DL_HIDDEN_DIM, DL_DROPOUT

OUT_DIR = ROOT / "results" / "confusion_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CONFIGS = [
    "twitter", "skytrax", "airlinequality",
    "twitter+skytrax", "twitter+airlinequality", "skytrax+airlinequality",
    "twitter+skytrax+airlinequality",
]
TFIDF_MODELS = ["tfidf_lr", "tfidf_svc"]
DL_MODELS = ["cnn", "lstm", "gru"]


def cm_path(model: str, source_config: str) -> Path:
    return OUT_DIR / f"{model}__{source_config}.npy"


def predict_tfidf(model_key: str, source_config: str) -> np.ndarray:
    model_dir = Path(MODELS_DIR) / model_key / source_config
    with open(model_dir / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(model_dir / "model.pkl", "rb") as f:
        clf = pickle.load(f)
    test_texts, _ = load_split(source_config, "test")
    X = tfidf.transform(test_texts)
    return clf.predict(X)


def predict_dl(model_key: str, source_config: str) -> np.ndarray:
    """Reload DL model with saved vocab and run inference on CPU."""
    model_dir = Path(MODELS_DIR) / model_key / source_config
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)

    word2idx = ckpt["word2idx"]
    max_len = ckpt["max_len"]
    vocab_size = ckpt["vocab_size"]

    model = create_model(model_key, vocab_size, EMBEDDING_DIM, 3,
                         DL_HIDDEN_DIM, DL_DROPOUT, pretrained_embeddings=None)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_texts, _ = load_split(source_config, "test")
    X = texts_to_sequences(test_texts, word2idx, max_len)
    X = torch.from_numpy(X)

    preds = []
    batch = 128
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits = model(X[i:i + batch])
            preds.append(logits.argmax(1).numpy())
    return np.concatenate(preds)


def run_one(model_key: str, source_config: str, *, overwrite: bool) -> str:
    out = cm_path(model_key, source_config)
    if out.exists() and not overwrite:
        return "SKIP"
    if model_key.startswith("tfidf_"):
        y_pred = predict_tfidf(model_key, source_config)
    elif model_key in DL_MODELS:
        y_pred = predict_dl(model_key, source_config)
    else:
        raise ValueError(f"Unsupported model: {model_key}")
    _, y_true = load_split(source_config, "test")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    np.save(out, cm)
    return "OK"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute TF-IDF + DL confusion matrices.")
    parser.add_argument("--families", nargs="+", default=["tfidf", "dl"],
                        choices=["tfidf", "dl"],
                        help="Which families to run (default: both).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if CM file exists.")
    args = parser.parse_args()

    targets: list[str] = []
    if "tfidf" in args.families:
        targets += TFIDF_MODELS
    if "dl" in args.families:
        targets += DL_MODELS

    total = len(targets) * len(ALL_CONFIGS)
    print(f"Total: {total} (model, source) pairs. Output dir: {OUT_DIR.relative_to(ROOT)}")
    done = 0
    for m in targets:
        for cfg in ALL_CONFIGS:
            done += 1
            try:
                status = run_one(m, cfg, overwrite=args.overwrite)
                print(f"  [{done:2d}/{total}] {status:4s}  {m} | {cfg}")
            except FileNotFoundError as e:
                print(f"  [{done:2d}/{total}] MISS  {m} | {cfg}  ({e})")
            except Exception as e:
                print(f"  [{done:2d}/{total}] FAIL  {m} | {cfg}  ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
