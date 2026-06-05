"""
Deep Learning model training: CNN, LSTM, GRU text classifiers.

Usage:
    python src/models/train_dl.py --model cnn --source twitter
    python src/models/train_dl.py --model lstm --source skytrax
    python src/models/train_dl.py --model gru --source all
    python src/models/train_dl.py --model all --source all
    python src/models/train_dl.py --model cnn --source twitter --glove path/to/glove.6B.300d.txt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    MODELS_DIR, SEED,
    EMBEDDING_DIM, DL_HIDDEN_DIM, DL_DROPOUT, DL_BATCH_SIZE,
    DL_MAX_EPOCHS, DL_PATIENCE, DL_LEARNING_RATE,
    SINGLE_SOURCE_CONFIGS, DUAL_SOURCE_CONFIGS, TRIPLE_SOURCE_CONFIG,
)
from src.utils.data_utils import (
    load_split, build_vocab, texts_to_sequences, load_glove_embeddings,
    compute_class_weights, get_max_seq_length, set_seed,
)
from src.utils.eval_utils import (
    compute_metrics, compute_per_class_metrics, get_confusion_matrix,
    save_results, print_metrics,
)


# ===== Model Definitions =====

class TextCNN(nn.Module):
    """Multi-kernel CNN text classifier."""

    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=128,
                 kernel_sizes=(3, 4, 5), dropout=0.5, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True  # fine-tune

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)         # (batch, embed_dim, seq_len)
        conv_outs = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        x = torch.cat(conv_outs, dim=1) # (batch, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        return self.fc(x)


class TextLSTM(nn.Module):
    """Bidirectional LSTM text classifier."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        # Concat forward and backward hidden states
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class TextGRU(nn.Module):
    """Bidirectional GRU text classifier."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True

        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


MODEL_REGISTRY = {
    "cnn": TextCNN,
    "lstm": TextLSTM,
    "gru": TextGRU,
}


# ===== Training Loop =====

def create_model(model_key, vocab_size, embed_dim, num_classes, hidden_dim,
                 dropout, pretrained_embeddings=None):
    """Instantiate model by key."""
    if model_key == "cnn":
        return TextCNN(vocab_size, embed_dim, num_classes,
                       num_filters=hidden_dim, kernel_sizes=(3, 4, 5),
                       dropout=0.5, pretrained_embeddings=pretrained_embeddings)
    else:
        cls = MODEL_REGISTRY[model_key]
        return cls(vocab_size, embed_dim, hidden_dim, num_classes,
                   dropout=dropout, pretrained_embeddings=pretrained_embeddings)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(all_labels)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def get_all_source_configs() -> list[str]:
    configs = list(SINGLE_SOURCE_CONFIGS)
    configs += ["+".join(pair) for pair in DUAL_SOURCE_CONFIGS]
    configs.append("+".join(TRIPLE_SOURCE_CONFIG))
    return configs


def train_and_evaluate(model_key: str, source_config: str, glove_path: str | None = None,
                       device_override: str | None = None):
    """Train and evaluate one DL model on one source config."""
    from src.utils.config import get_device
    device = torch.device(device_override) if device_override else get_device()

    print(f"\n{'='*60}")
    print(f"[TRAIN] {model_key.upper()} | {source_config} | device={device}")
    print(f"{'='*60}")

    # Load data
    train_texts, train_labels = load_split(source_config, "train")
    val_texts, val_labels = load_split(source_config, "val")
    test_texts, test_labels = load_split(source_config, "test")

    # Build vocab and convert to sequences
    max_len = get_max_seq_length(source_config)
    word2idx = build_vocab(train_texts)
    vocab_size = len(word2idx)

    X_train = texts_to_sequences(train_texts, word2idx, max_len)
    X_val = texts_to_sequences(val_texts, word2idx, max_len)
    X_test = texts_to_sequences(test_texts, word2idx, max_len)

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    y_test = np.array(test_labels)

    # Optional GloVe embeddings
    pretrained = None
    if glove_path and os.path.exists(glove_path):
        pretrained = load_glove_embeddings(glove_path, word2idx, EMBEDDING_DIM)

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=DL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=DL_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=DL_BATCH_SIZE)

    # Model
    model = create_model(model_key, vocab_size, EMBEDDING_DIM, 3,
                         DL_HIDDEN_DIM, DL_DROPOUT, pretrained)
    model = model.to(device)

    # Class-weighted loss
    weights = compute_class_weights(train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LEARNING_RATE)

    # Training with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, DL_MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels_arr = evaluate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels_arr, val_preds)

        print(f"  Epoch {epoch:2d}/{DL_MAX_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DL_PATIENCE:
                print(f"  [EARLY STOP] No improvement for {DL_PATIENCE} epochs.")
                break

    # Restore best model and evaluate on test
    model.load_state_dict(best_state)
    model = model.to(device)
    _, test_preds, test_labels_arr = evaluate(model, test_loader, criterion, device)

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
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "word2idx": word2idx,
        "max_len": max_len,
        "model_key": model_key,
        "vocab_size": vocab_size,
    }, os.path.join(save_dir, "model.pt"))
    print(f"[INFO] Model saved to {save_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train DL text classifiers")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY) + ["all"])
    parser.add_argument("--source", required=True,
                        help="Source config: twitter, skytrax+airlinequality, all, etc.")
    parser.add_argument("--glove", default=None,
                        help="Path to GloVe embeddings file (e.g. glove.6B.300d.txt)")
    parser.add_argument("--device", default=None, help="Force device: cuda or cpu")
    args = parser.parse_args()

    set_seed()

    models = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    sources = get_all_source_configs() if args.source == "all" else [args.source]

    for model_key in models:
        for source in sources:
            train_and_evaluate(model_key, source, args.glove, args.device)


if __name__ == "__main__":
    main()
