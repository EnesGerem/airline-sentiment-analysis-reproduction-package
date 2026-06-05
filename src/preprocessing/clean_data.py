"""
Clean raw dataset text for each source.

Usage:
    python src/preprocessing/clean_data.py --source twitter
    python src/preprocessing/clean_data.py --source skytrax
    python src/preprocessing/clean_data.py --source airlinequality
    python src/preprocessing/clean_data.py --source all
"""

import argparse
import csv
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import RAW_DIR, PROCESSED_DIR, SOURCES


def clean_text(text: str) -> str:
    """Apply text cleaning pipeline."""
    if not text or not isinstance(text, str):
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # Remove @mentions (Twitter)
    text = re.sub(r"@\w+", "", text)
    # Keep hashtag text, remove # symbol
    text = re.sub(r"#(\w+)", r"\1", text)
    # Remove HTML entities like &amp;
    text = re.sub(r"&\w+;", " ", text)
    # Lowercase
    text = text.lower()
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_twitter(raw_path: str, out_path: str) -> dict:
    """Clean Twitter dataset. Returns stats dict."""
    rows_in, rows_out, duplicates = 0, 0, 0
    seen_texts = set()

    with open(raw_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["text", "label", "airline", "source"])
        writer.writeheader()

        for row in reader:
            rows_in += 1
            text = clean_text(row["text"])
            label = row["airline_sentiment"].strip().lower()

            if label not in ("positive", "neutral", "negative"):
                continue
            if len(text.split()) < 3:
                continue
            if text in seen_texts:
                duplicates += 1
                continue
            seen_texts.add(text)

            writer.writerow({
                "text": text,
                "label": label,
                "airline": row.get("airline", ""),
                "source": "twitter",
            })
            rows_out += 1

    return {"source": "twitter", "rows_in": rows_in, "rows_out": rows_out, "duplicates": duplicates}


def clean_skytrax(raw_path: str, out_path: str) -> dict:
    """Clean Skytrax dataset. Returns stats dict."""
    rows_in, rows_out, duplicates, no_rating = 0, 0, 0, 0
    seen_texts = set()

    with open(raw_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["text", "rating", "airline", "source"])
        writer.writeheader()

        for row in reader:
            rows_in += 1
            rating_str = row.get("overall_rating", "").strip()
            if not rating_str:
                no_rating += 1
                continue
            try:
                rating = float(rating_str)
            except ValueError:
                no_rating += 1
                continue
            if rating < 1 or rating > 10:
                no_rating += 1
                continue

            text = clean_text(row.get("content", ""))
            if len(text.split()) < 3:
                continue
            if text in seen_texts:
                duplicates += 1
                continue
            seen_texts.add(text)

            writer.writerow({
                "text": text,
                "rating": int(rating),
                "airline": row.get("airline_name", ""),
                "source": "skytrax",
            })
            rows_out += 1

    return {"source": "skytrax", "rows_in": rows_in, "rows_out": rows_out,
            "duplicates": duplicates, "no_rating": no_rating}


def clean_airlinequality(raw_path: str, out_path: str) -> dict:
    """Clean AirlineQuality dataset. Returns stats dict."""
    rows_in, rows_out, duplicates, no_rating = 0, 0, 0, 0
    seen_texts = set()

    with open(raw_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["text", "rating", "airline", "source"])
        writer.writeheader()

        for row in reader:
            rows_in += 1
            rating_str = row.get("Overall_Rating", "").strip()
            if not rating_str:
                no_rating += 1
                continue
            try:
                rating = float(rating_str)
            except ValueError:
                no_rating += 1
                continue
            if rating < 1 or rating > 10:
                no_rating += 1
                continue

            text = clean_text(row.get("Review", ""))
            if len(text.split()) < 3:
                continue
            if text in seen_texts:
                duplicates += 1
                continue
            seen_texts.add(text)

            writer.writerow({
                "text": text,
                "rating": int(rating),
                "airline": row.get("Airline Name", ""),
                "source": "airlinequality",
            })
            rows_out += 1

    return {"source": "airlinequality", "rows_in": rows_in, "rows_out": rows_out,
            "duplicates": duplicates, "no_rating": no_rating}


RAW_FILES = {
    "twitter": "twitter_airline_sentiment.csv",
    "skytrax": "skytrax_reviews.csv",
    "airlinequality": "airlinequality_reviews.csv",
}

CLEANERS = {
    "twitter": clean_twitter,
    "skytrax": clean_skytrax,
    "airlinequality": clean_airlinequality,
}


def main():
    parser = argparse.ArgumentParser(description="Clean raw dataset text")
    parser.add_argument("--source", required=True, choices=SOURCES + ["all"],
                        help="Which dataset to clean")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    sources = SOURCES if args.source == "all" else [args.source]

    for source in sources:
        raw_path = os.path.join(RAW_DIR, RAW_FILES[source])
        out_path = os.path.join(PROCESSED_DIR, f"{source}_cleaned.csv")
        print(f"\n[INFO] Cleaning {source}...")
        stats = CLEANERS[source](raw_path, out_path)
        print(f"  Input rows:  {stats['rows_in']}")
        print(f"  Output rows: {stats['rows_out']}")
        print(f"  Duplicates:  {stats['duplicates']}")
        if "no_rating" in stats:
            print(f"  No rating:   {stats['no_rating']}")
        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
