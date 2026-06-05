"""
Shared evaluation utilities. All models use these for consistent metric reporting.
"""

import csv
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)

from src.utils.config import RESULTS_DIR, LABEL_NAMES


def compute_metrics(y_true, y_pred) -> dict:
    """Compute all evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def compute_per_class_metrics(y_true, y_pred) -> dict:
    """Compute per-class precision, recall, F1."""
    result = {}
    for avg_type in ["macro", "weighted", None]:
        p = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
        r = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
        f = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
        if avg_type is None:
            for i, name in enumerate(LABEL_NAMES):
                result[f"{name}_precision"] = p[i]
                result[f"{name}_recall"] = r[i]
                result[f"{name}_f1"] = f[i]
        else:
            result[f"{avg_type}_precision"] = p
            result[f"{avg_type}_recall"] = r
            result[f"{avg_type}_f1"] = f
    return result


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def save_results(model_name: str, source_config: str, metrics: dict,
                 extra: dict | None = None):
    """Append results to a central CSV log."""
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "tables", "experiment_log.csv")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "source_config": source_config,
        **metrics,
    }
    if extra:
        row.update(extra)

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[INFO] Results saved to {log_path}")


def print_metrics(metrics: dict, model_name: str = "", source: str = ""):
    """Pretty-print metrics."""
    header = f"{model_name} | {source}" if model_name else "Results"
    print(f"\n{'='*50}")
    print(f"  {header}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
    print(f"{'='*50}\n")
