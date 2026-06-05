"""
Transformer fine-tuning: DistilBERT, BERT, RoBERTa.

Usage:
    python src/models/train_transformer.py --model distilbert --source twitter
    python src/models/train_transformer.py --model bert --source skytrax+airlinequality
    python src/models/train_transformer.py --model roberta --source all
    python src/models/train_transformer.py --model all --source all
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    MODELS_DIR, SEED,
    TRANSFORMER_MODELS, TRANSFORMER_EPOCHS, TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_LEARNING_RATE, TRANSFORMER_WEIGHT_DECAY,
    TRANSFORMER_MAX_LENGTH_TWEET, TRANSFORMER_MAX_LENGTH_REVIEW,
    SINGLE_SOURCE_CONFIGS, DUAL_SOURCE_CONFIGS, TRIPLE_SOURCE_CONFIG,
)
from src.utils.data_utils import load_split, compute_class_weights, set_seed, has_twitter
from src.utils.eval_utils import (
    compute_metrics, compute_per_class_metrics, get_confusion_matrix,
    save_results, print_metrics,
)


class TextDataset(Dataset):
    """Dataset that tokenizes on-the-fly."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_max_length(source_config: str) -> int:
    if source_config == "twitter":
        return TRANSFORMER_MAX_LENGTH_TWEET
    return TRANSFORMER_MAX_LENGTH_REVIEW


def get_all_source_configs() -> list[str]:
    configs = list(SINGLE_SOURCE_CONFIGS)
    configs += ["+".join(pair) for pair in DUAL_SOURCE_CONFIGS]
    configs.append("+".join(TRIPLE_SOURCE_CONFIG))
    return configs


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * len(labels)
            all_preds.extend(outputs.logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(all_labels)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def train_and_evaluate(model_key: str, source_config: str,
                       device_override: str | None = None):
    """Fine-tune and evaluate one transformer on one source config."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup
    from src.utils.config import get_device

    device = torch.device(device_override) if device_override else get_device()
    model_name = TRANSFORMER_MODELS[model_key]
    max_length = get_max_length(source_config)

    print(f"\n{'='*60}")
    print(f"[TRAIN] {model_key.upper()} ({model_name}) | {source_config} | device={device}")
    print(f"  max_length={max_length}, epochs={TRANSFORMER_EPOCHS}, "
          f"batch_size={TRANSFORMER_BATCH_SIZE}")
    print(f"{'='*60}")

    # Load data
    train_texts, train_labels = load_split(source_config, "train")
    val_texts, val_labels = load_split(source_config, "val")
    test_texts, test_labels = load_split(source_config, "test")

    # Tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, max_length)
    test_ds = TextDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRANSFORMER_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=TRANSFORMER_BATCH_SIZE)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRANSFORMER_LEARNING_RATE,
                                  weight_decay=TRANSFORMER_WEIGHT_DECAY)
    total_steps = len(train_loader) * TRANSFORMER_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0
    best_state = None

    for epoch in range(1, TRANSFORMER_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_preds, val_labels_arr = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels_arr, val_preds)

        print(f"  Epoch {epoch}/{TRANSFORMER_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model and evaluate on test
    model.load_state_dict(best_state)
    model = model.to(device)
    _, test_preds, test_labels_arr = evaluate(model, test_loader, device)

    metrics = compute_metrics(test_labels_arr, test_preds)
    per_class = compute_per_class_metrics(test_labels_arr, test_preds)
    cm = get_confusion_matrix(test_labels_arr, test_preds)

    print_metrics(metrics, model_key.upper(), source_config)
    print("Confusion Matrix:")
    print(cm)

    # Save
    all_metrics = {**metrics, **per_class}
    save_results(model_key, source_config, all_metrics)

    save_dir = os.path.join(MODELS_DIR, model_key, source_config)
    os.makedirs(save_dir, exist_ok=True)
    model.cpu().save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] Model saved to {save_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer models")
    parser.add_argument("--model", required=True,
                        choices=list(TRANSFORMER_MODELS) + ["all"])
    parser.add_argument("--source", required=True,
                        help="Source config: twitter, skytrax+airlinequality, all, etc.")
    parser.add_argument("--device", default=None, help="Force device: cuda or cpu")
    args = parser.parse_args()

    set_seed()

    models = list(TRANSFORMER_MODELS) if args.model == "all" else [args.model]
    sources = get_all_source_configs() if args.source == "all" else [args.source]

    for model_key in models:
        for source in sources:
            train_and_evaluate(model_key, source, args.device)


if __name__ == "__main__":
    main()
