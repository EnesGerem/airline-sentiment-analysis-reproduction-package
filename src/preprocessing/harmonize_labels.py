"""
Map ratings to unified 3-class sentiment labels across all sources.

Reads cleaned CSVs from data/processed/ and adds a unified 'label' column.
- Twitter: already has labels (positive/neutral/negative) — passed through
- Skytrax: 1-4→negative, 5-6→neutral, 7-10→positive
- AirlineQuality: 1-4→negative, 5-6→neutral, 7-9→positive (range is 1-9)

Usage:
    python src/preprocessing/harmonize_labels.py
"""

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import PROCESSED_DIR, SKYTRAX_LABEL_MAP, AIRLINEQUALITY_LABEL_MAP, SOURCES


def harmonize_source(source: str) -> dict:
    """Harmonize labels for one source. Returns class distribution."""
    cleaned_path = os.path.join(PROCESSED_DIR, f"{source}_cleaned.csv")
    harmonized_path = os.path.join(PROCESSED_DIR, f"{source}_harmonized.csv")

    if not os.path.exists(cleaned_path):
        print(f"  [SKIP] {cleaned_path} not found — run clean_data.py first")
        return {}

    label_counts = Counter()
    rows_written = 0

    with open(cleaned_path, encoding="utf-8") as fin, \
         open(harmonized_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["text", "label", "airline", "source"])
        writer.writeheader()

        for row in reader:
            if source == "twitter":
                label = row["label"]
            else:
                rating = int(float(row["rating"]))
                label_map = SKYTRAX_LABEL_MAP if source == "skytrax" else AIRLINEQUALITY_LABEL_MAP
                label = label_map.get(rating)
                if label is None:
                    continue

            label_counts[label] += 1
            writer.writerow({
                "text": row["text"],
                "label": label,
                "airline": row["airline"],
                "source": row["source"],
            })
            rows_written += 1

    print(f"  Total: {rows_written}")
    total = sum(label_counts.values())
    for lbl in ["negative", "neutral", "positive"]:
        cnt = label_counts.get(lbl, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"    {lbl}: {cnt} ({pct:.1f}%)")
    print(f"  Saved to: {harmonized_path}")
    return dict(label_counts)


def main():
    print("[INFO] Harmonizing labels for all sources...\n")
    all_counts = Counter()

    for source in SOURCES:
        print(f"[{source.upper()}]")
        counts = harmonize_source(source)
        all_counts.update(counts)
        print()

    print("[COMBINED]")
    total = sum(all_counts.values())
    print(f"  Total: {total}")
    for lbl in ["negative", "neutral", "positive"]:
        cnt = all_counts.get(lbl, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"    {lbl}: {cnt} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
