"""
TF-IDF + Classical ML baseline training.

Usage:
    python src/models/train_baseline.py --model lr --source twitter
    python src/models/train_baseline.py --model svc --source twitter+skytrax
    python src/models/train_baseline.py --model all --source all
"""

import argparse
import os
import pickle
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    MODELS_DIR, SEED,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    SINGLE_SOURCE_CONFIGS, DUAL_SOURCE_CONFIGS, TRIPLE_SOURCE_CONFIG,
)
from src.utils.data_utils import load_split, set_seed
from src.utils.eval_utils import (
    compute_metrics, compute_per_class_metrics, get_confusion_matrix,
    save_results, print_metrics,
)

MODEL_REGISTRY = {
    "lr": ("LogisticRegression",
           lambda: LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=SEED, solver="lbfgs")),
    "svc": ("LinearSVC",
            lambda: LinearSVC(max_iter=5000, class_weight="balanced",
                              random_state=SEED)),
}


def get_all_source_configs() -> list[str]:
    configs = list(SINGLE_SOURCE_CONFIGS)
    configs += ["+".join(pair) for pair in DUAL_SOURCE_CONFIGS]
    configs.append("+".join(TRIPLE_SOURCE_CONFIG))
    return configs


def train_and_evaluate(model_key: str, source_config: str):
    """Train one TF-IDF + ML model on one source config."""
    model_name, model_fn = MODEL_REGISTRY[model_key]
    print(f"\n[TRAIN] {model_name} | {source_config}")

    train_texts, train_labels = load_split(source_config, "train")
    val_texts, val_labels = load_split(source_config, "val")
    test_texts, test_labels = load_split(source_config, "test")

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)

    # Train
    clf = model_fn()
    clf.fit(X_train, train_labels)

    # Evaluate on test
    y_pred = clf.predict(X_test)
    metrics = compute_metrics(test_labels, y_pred)
    per_class = compute_per_class_metrics(test_labels, y_pred)
    cm = get_confusion_matrix(test_labels, y_pred)

    print_metrics(metrics, f"TF-IDF+{model_name}", source_config)
    print("Confusion Matrix:")
    print(cm)

    # Save results
    all_metrics = {**metrics, **per_class}
    save_results(f"tfidf_{model_key}", source_config, all_metrics)

    # Save model + vectorizer
    save_dir = os.path.join(MODELS_DIR, f"tfidf_{model_key}", source_config)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(save_dir, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    print(f"[INFO] Model saved to {save_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + ML baselines")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY) + ["all"])
    parser.add_argument("--source", required=True,
                        help="Source config: twitter, skytrax+airlinequality, all, etc.")
    args = parser.parse_args()

    set_seed()

    models = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    sources = get_all_source_configs() if args.source == "all" else [args.source]

    for model_key in models:
        for source in sources:
            train_and_evaluate(model_key, source)


if __name__ == "__main__":
    main()
