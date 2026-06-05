# Multi-Source NLP for Airline Passenger Sentiment Analysis

Reproduction package for the M.Sc. thesis **"Multi-Source Natural Language
Processing for Airline Passenger Sentiment and Satisfaction Analysis"**
(Enes Gerem, Department of Computer Engineering, Hacettepe University).

The thesis builds an explainable, multi-source sentiment-classification framework
over Twitter, Skytrax, and AirlineQuality airline feedback. It compares classical
ML, deep-learning, and transformer models across single-, dual-, and
triple-source configurations, with cross-domain (out-of-distribution) evaluation
and post-hoc explainability (SHAP / LIME).

- **Fine-tuned model checkpoints:** https://huggingface.co/enesgerem/airline-sentiment-models
- **License:** code MIT (see [LICENSE](LICENSE)); data under source-specific terms (see below).

---

## Research questions

- **RQ1** — How effectively do transformer models capture airline sentiment across data sources?
- **RQ2** — To what extent do models generalize across domains (cross-source evaluation)?
- **RQ3** — How does XAI enhance interpretability without compromising performance?
- **RQ4** — What trade-offs exist between performance, robustness, and interpretability?

## Datasets

| Source | ~Samples (post-clean) | Avg. words | Origin |
|--------|----------------------|------------|--------|
| Twitter | 14,018 | 17 | Crowdflower *US Airline Sentiment* (CC BY-NC-SA) |
| Skytrax | 36,834 | 115 | Skytrax airline reviews (Kaggle) |
| AirlineQuality | 22,297 | 131 | airlinequality.com reviews (Kaggle) |

**Label harmonization (3-class).** Twitter uses its native positive / neutral /
negative labels. Review ratings are mapped: 1–4 → negative, 5–6 → neutral,
7–10 → positive. Integer encoding used throughout:
`{"negative": 0, "neutral": 1, "positive": 2}`.

### Data access

The raw datasets are **not redistributed in this repository**. Obtain them from
the original sources and place the CSV files under `data/raw/`:

- **Twitter** — Crowdflower *US Airline Sentiment*
  (https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment),
  redistributed under CC BY-NC-SA.
- **Skytrax** — Skytrax airline reviews
  (https://www.kaggle.com/datasets/austinpeck/skytrax-reviews-dataset-august-2nd-2015).
- **AirlineQuality** — airlinequality.com passenger reviews
  (https://www.kaggle.com/datasets/juhibhojani/airline-reviews).

The review corpora are point-in-time scrapes used solely for non-commercial
academic research under each platform's terms of service. All personally
identifiable information (handles, profile metadata) is removed during
preprocessing. See [docs/DATA_INVENTORY.md](docs/DATA_INVENTORY.md) for the
expected file layout and column schema.

## Models

| Family | Members |
|--------|---------|
| Classical ML | TF-IDF + Logistic Regression, TF-IDF + Linear SVC |
| Deep learning | CNN, BiLSTM, BiGRU (300-d embeddings) |
| Transformers | DistilBERT, BERT-base, RoBERTa-base (fine-tuned) |

Eight model families × seven source configurations = **56 in-domain** runs, plus
**48 cross-domain** transfer evaluations (8 models × 6 ordered source pairs).

---

## Setup

Python 3.10+ (the thesis used Python 3.12, PyTorch 2.10, Transformers 5.0).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

All randomness is seeded with **42** (Python, NumPy, PyTorch), and
`PYTHONHASHSEED` is set before any randomized operation. The 70/15/15
stratified split is generated once and reused across every experiment.

## Reproduction pipeline

```bash
# 1. Clean each raw source
python src/preprocessing/clean_data.py --source twitter
python src/preprocessing/clean_data.py --source skytrax
python src/preprocessing/clean_data.py --source airlinequality

# 2. Harmonize labels to 3 classes and build the 70/15/15 stratified splits
python src/preprocessing/harmonize_labels.py
python src/preprocessing/create_splits.py

# 3. Train (examples — see --help for all model/source options)
python src/models/train_baseline.py    --model lr        --source twitter
python src/models/train_dl.py          --model lstm      --source twitter+skytrax
python src/models/train_transformer.py --model roberta   --source twitter+skytrax+airlinequality

# 4. Cross-domain evaluation and confusion matrices
python src/evaluation/compute_cms_local.py

# 5. Master results table, degradation summary, and result tables/figures
python src/evaluation/build_chapter5_artifacts.py
```

The transformer experiments were run on Google Colab (T4 / A100); the
self-contained notebooks under [`notebooks/`](notebooks/) reproduce every run
end to end. Use the pre-trained checkpoints on the
[Hugging Face Hub](https://huggingface.co/enesgerem/airline-sentiment-models)
to skip training:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo, sub = "enesgerem/airline-sentiment-models", "roberta/twitter"
tok = AutoTokenizer.from_pretrained(repo, subfolder=sub)
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=sub)
```

## Key results (Macro-F1, test split)

| Model | Twitter | Skytrax | AirlineQuality | Triple-source |
|-------|:-------:|:-------:|:-----------:|:-------------:|
| RoBERTa | **0.806** | **0.721** | 0.655 | **0.751** |
| BERT | 0.799 | 0.715 | **0.662** | 0.746 |

Cross-domain transfer is strongest between the two review platforms
(RoBERTa Skytrax→AirlineQuality 0.678) and weakest across the tweet/review divide,
where every model loses 20–30 Macro-F1 points. The neutral class collapses in a
direction that mirrors each source's label skew. Full numbers are in
[`results/tables/`](results/tables/).

## Repository layout

```
src/preprocessing/   cleaning, label harmonization, split creation
src/models/          classical / DL / transformer training scripts
src/evaluation/      confusion matrices, master result table & figure builder
src/utils/           config (seeds, paths, label map), data and eval helpers
notebooks/           self-contained Colab notebooks for all experiments
results/tables/      experiment logs and cross-domain results (CSV)
docs/                data inventory (expected raw layout & schema)
```

## Citation

If you use this code or the models, please cite the thesis (see
[CITATION.cff](CITATION.cff)). A related conference paper (ICADA) will be added
once published.

## License

Source code is released under the MIT License ([LICENSE](LICENSE)). The datasets
are **not** covered by this license and remain subject to their original terms
(Crowdflower CC BY-NC-SA; Skytrax and AirlineQuality platform terms of service,
non-commercial academic use only).
