"""
Create stratified train/val/test splits (70/15/15) for each source.

Reads harmonized CSVs from data/processed/ and outputs splits to data/splits/.
Splits are done per-source, then combined files are created for multi-source configs.

Usage:
    python src/preprocessing/create_splits.py
"""

import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    PROCESSED_DIR, SPLITS_DIR, SOURCES, SEED,
    TRAIN_RATIO, VAL_RATIO,
    DUAL_SOURCE_CONFIGS, TRIPLE_SOURCE_CONFIG,
)

FIELDNAMES = ["text", "label", "airline", "source"]


def read_harmonized(source: str) -> list[dict]:
    """Read all rows from a harmonized CSV."""
    path = os.path.join(PROCESSED_DIR, f"{source}_harmonized.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found — run harmonize_labels.py first")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def stratified_split(rows: list[dict], seed: int = SEED) -> tuple[list, list, list]:
    """Split rows into train/val/test with stratified sampling by label."""
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train, val, test = [], [], []
    rng = random.Random(seed)

    for label in sorted(by_label.keys()):
        items = by_label[label]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_split(rows: list[dict], path: str):
    """Write rows to CSV."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def combine_splits(sources: list[str], name: str, all_splits: dict):
    """Combine pre-computed splits from multiple sources into one."""
    for split_name in ["train", "val", "test"]:
        combined = []
        for src in sources:
            combined.extend(all_splits[src][split_name])
        rng = random.Random(SEED)
        rng.shuffle(combined)
        path = os.path.join(SPLITS_DIR, f"{name}_{split_name}.csv")
        write_split(combined, path)
    total = sum(len(all_splits[s][sp]) for s in sources for sp in ["train", "val", "test"])
    print(f"  {name}: {total} samples")


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    all_splits = {}

    # Per-source splits
    print("[INFO] Creating per-source splits...\n")
    for source in SOURCES:
        rows = read_harmonized(source)
        if not rows:
            continue
        train, val, test = stratified_split(rows)

        for split_name, split_rows in [("train", train), ("val", val), ("test", test)]:
            path = os.path.join(SPLITS_DIR, f"{source}_{split_name}.csv")
            write_split(split_rows, path)

        all_splits[source] = {"train": train, "val": val, "test": test}

        print(f"  {source}: train={len(train)}, val={len(val)}, test={len(test)}, total={len(rows)}")
        # Show class distribution per split
        for split_name, split_rows in [("train", train), ("val", val), ("test", test)]:
            counts = defaultdict(int)
            for r in split_rows:
                counts[r["label"]] += 1
            dist = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"    {split_name}: {dist}")

    # Dual-source combinations
    print("\n[INFO] Creating dual-source splits...")
    for src_a, src_b in DUAL_SOURCE_CONFIGS:
        if src_a not in all_splits or src_b not in all_splits:
            continue
        name = f"{src_a}+{src_b}"
        combine_splits([src_a, src_b], name, all_splits)

    # Triple-source combination
    print("\n[INFO] Creating triple-source split...")
    sources_present = [s for s in TRIPLE_SOURCE_CONFIG if s in all_splits]
    if len(sources_present) == 3:
        combine_splits(sources_present, "twitter+skytrax+airlinequality", all_splits)

    print("\n[DONE] All splits saved to:", SPLITS_DIR)


if __name__ == "__main__":
    main()
