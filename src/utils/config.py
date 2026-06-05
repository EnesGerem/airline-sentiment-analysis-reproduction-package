"""
Shared configuration for all experiments.
All experiments must use these constants for reproducibility.
"""

import os

# Random seed for reproducibility
SEED = 42

# Data paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Label mapping
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = ["negative", "neutral", "positive"]

# Skytrax rating to sentiment mapping (1-10 scale)
SKYTRAX_LABEL_MAP = {
    1: "negative", 2: "negative", 3: "negative", 4: "negative",
    5: "neutral", 6: "neutral",
    7: "positive", 8: "positive", 9: "positive", 10: "positive"
}

# AirlineQuality rating to sentiment mapping (1-9 scale, same thresholds)
AIRLINEQUALITY_LABEL_MAP = {
    1: "negative", 2: "negative", 3: "negative", 4: "negative",
    5: "neutral", 6: "neutral",
    7: "positive", 8: "positive", 9: "positive",
}

# Source names
SOURCES = ["twitter", "skytrax", "airlinequality"]

# Experiment configurations
SINGLE_SOURCE_CONFIGS = ["twitter", "skytrax", "airlinequality"]
DUAL_SOURCE_CONFIGS = [
    ("twitter", "skytrax"),
    ("twitter", "airlinequality"),
    ("skytrax", "airlinequality"),
]
TRIPLE_SOURCE_CONFIG = ("twitter", "skytrax", "airlinequality")
CROSS_DOMAIN_CONFIGS = [
    ("twitter", "skytrax"),
    ("twitter", "airlinequality"),
    ("skytrax", "twitter"),
    ("skytrax", "airlinequality"),
    ("airlinequality", "twitter"),
    ("airlinequality", "skytrax"),
]

# TF-IDF settings
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)

# Deep Learning settings
EMBEDDING_DIM = 300
DL_HIDDEN_DIM = 128
DL_DROPOUT = 0.3
DL_BATCH_SIZE = 64
DL_MAX_EPOCHS = 20
DL_PATIENCE = 3  # early stopping patience
DL_LEARNING_RATE = 1e-3
MAX_VOCAB_SIZE = 50000
MAX_SEQ_LENGTH_TWEET = 64
MAX_SEQ_LENGTH_REVIEW = 256

# Transformer settings
TRANSFORMER_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}
TRANSFORMER_MAX_LENGTH_TWEET = 128
TRANSFORMER_MAX_LENGTH_REVIEW = 512
TRANSFORMER_EPOCHS = 4
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_STEPS = 0

# Evaluation
METRICS = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1"]
PRIMARY_METRIC = "macro_f1"

# XAI settings
SHAP_NUM_SAMPLES = 500
LIME_NUM_SAMPLES = 100
LIME_NUM_FEATURES = 20


def get_device():
    """
    Auto-detect best available device.
    Priority: CUDA/ROCm → DirectML → CPU.
    """
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using GPU (CUDA/ROCm): {device_name}")
        return torch.device("cuda")
    try:
        import torch_directml
        dml = torch_directml.device()
        device_name = torch_directml.device_name(0)
        print(f"[INFO] Using GPU (DirectML): {device_name}")
        return dml
    except ImportError:
        pass
    print("[INFO] No GPU detected. Using CPU.")
    return torch.device("cpu")
