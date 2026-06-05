"""
Build Chapter 5 artifacts: master experiment log, figures, and LaTeX-ready tables.

Run from project root:
    python src/evaluation/build_chapter5_artifacts.py

Outputs:
    results/tables/experiment_log.csv         (master, 56 rows)
    results/tables/experiment_log.prev.csv    (backup of previous master)
    results/tables/degradation_summary.csv
    results/tables/results_single_source.tex
    results/tables/results_dual_source.tex
    results/tables/results_triple_source.tex
    results/figures/class_distribution.png
    results/figures/text_length_distribution.png
    results/figures/cross_domain_heatmap.png
    results/figures/in_vs_cross_domain.png
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["twitter", "skytrax", "airlinequality"]
SINGLE = SOURCES[:]
DUAL = ["twitter+skytrax", "twitter+airlinequality", "skytrax+airlinequality"]
TRIPLE = ["twitter+skytrax+airlinequality"]
ALL_CONFIGS = SINGLE + DUAL + TRIPLE

MODEL_ORDER = ["tfidf_lr", "tfidf_svc", "cnn", "lstm", "gru", "distilbert", "bert", "roberta"]
MODEL_LABEL = {
    "tfidf_lr": "TF-IDF + LR", "tfidf_svc": "TF-IDF + SVC",
    "cnn": "CNN", "lstm": "BiLSTM", "gru": "BiGRU",
    "distilbert": "DistilBERT", "bert": "BERT", "roberta": "RoBERTa",
}
SOURCE_LABEL = {"twitter": "Twitter", "skytrax": "Skytrax", "airlinequality": "AirlineQuality"}


# ---------------------------------------------------------------------------
# 1. Master log
# ---------------------------------------------------------------------------
def build_master_log() -> pd.DataFrame:
    master = TABLES_DIR / "experiment_log.csv"
    if master.exists():
        shutil.copy(master, TABLES_DIR / "experiment_log.prev.csv")

    parts = []
    for stem in ("experiment_log_tfidf", "experiment_log_dl", "experiment_log_transformers"):
        parts.append(pd.read_csv(TABLES_DIR / f"{stem}.csv"))
    merged = pd.concat(parts, ignore_index=True)
    merged = (merged.sort_values("timestamp")
                    .drop_duplicates(["model", "source_config"], keep="last"))

    cat_m = pd.Categorical(merged["model"], categories=MODEL_ORDER, ordered=True)
    cat_c = pd.Categorical(merged["source_config"], categories=ALL_CONFIGS, ordered=True)
    merged = (merged.assign(_m=cat_m, _c=cat_c)
                    .sort_values(["_m", "_c"])
                    .drop(columns=["_m", "_c"]))

    merged.to_csv(master, index=False)
    print(f"[OK] master log: {len(merged)} rows -> results/tables/experiment_log.csv")
    assert len(merged) == 56, f"Expected 56 rows in master log, got {len(merged)}"
    return merged


# ---------------------------------------------------------------------------
# 2. Dataset figures
# ---------------------------------------------------------------------------
def _load_full_source(src: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(SPLITS_DIR / f"{src}_{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    )


def figure_class_distribution() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, src in zip(axes, SOURCES):
        df = _load_full_source(src)
        counts = df["label"].value_counts().reindex(["negative", "neutral", "positive"])
        bars = ax.bar(counts.index, counts.values,
                      color=["#d62728", "#9e9e9e", "#2ca02c"])
        ax.set_title(f"{SOURCE_LABEL[src]} (N = {len(df):,})")
        ax.set_ylabel("Count")
        for bar, v in zip(bars, counts.values):
            pct = 100 * v / len(df)
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:,}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Class distribution per source (3-class sentiment, full dataset)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


def figure_text_length() -> None:
    series = {src: _load_full_source(src)["text"].str.split().str.len() for src in SOURCES}

    fig, ax = plt.subplots(figsize=(8.5, 5))
    parts = ax.violinplot([series[s].values for s in SOURCES],
                          showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.65)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([
        f"{SOURCE_LABEL[s]}\n(median={int(series[s].median())},  "
        f"mean={series[s].mean():.0f})" for s in SOURCES
    ])
    ax.set_ylabel("Word count")
    ax.set_title("Text length distribution per source (clipped at 99th percentile)")
    p99 = max(s.quantile(0.99) for s in series.values())
    ax.set_ylim(0, p99)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "text_length_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 3. Cross-domain figures
# ---------------------------------------------------------------------------
def figure_cross_domain_heatmap() -> None:
    df = pd.read_csv(TABLES_DIR / "cross_domain.csv")
    df["pair"] = df["train_source"] + " -> " + df["test_source"]

    pair_order = [f"{tr} -> {te}" for tr in SOURCES for te in SOURCES if tr != te]
    pivot = df.pivot_table(index="model", columns="pair",
                           values="macro_f1", aggfunc="first")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=pair_order)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis",
                vmin=0.2, vmax=0.75,
                cbar_kws={"label": "Macro-F1"},
                yticklabels=[MODEL_LABEL[m] for m in pivot.index], ax=ax)
    ax.set_title("Cross-domain Macro-F1 (8 models x 6 source pairs)")
    ax.set_xlabel("Train -> Test")
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = FIGURES_DIR / "cross_domain_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


def figure_degradation() -> None:
    master = pd.read_csv(TABLES_DIR / "experiment_log.csv")
    cross = pd.read_csv(TABLES_DIR / "cross_domain.csv")

    rows = []
    for m in MODEL_ORDER:
        in_dom = master[(master["model"] == m) &
                        (master["source_config"].isin(SOURCES))]["macro_f1"].mean()
        x_dom = cross[cross["model"] == m]["macro_f1"].mean()
        rows.append({
            "model": m,
            "in_domain_macro_f1": in_dom,
            "cross_domain_macro_f1": x_dom,
            "drop_absolute": in_dom - x_dom,
            "drop_relative_pct": 100 * (in_dom - x_dom) / in_dom if in_dom else float("nan"),
        })
    deg = pd.DataFrame(rows)
    deg.to_csv(TABLES_DIR / "degradation_summary.csv", index=False)
    print(f"[OK] results/tables/degradation_summary.csv")

    x = np.arange(len(deg))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5))
    b1 = ax.bar(x - w / 2, deg["in_domain_macro_f1"], w,
                label="In-domain (avg over 3 single-source)", color="#2ca02c")
    b2 = ax.bar(x + w / 2, deg["cross_domain_macro_f1"], w,
                label="Cross-domain (avg over 6 pairs)", color="#d62728")
    for bar, v in zip(b1, deg["in_domain_macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=8)
    for bar, v in zip(b2, deg["cross_domain_macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in deg["model"]], rotation=20, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("In-domain vs cross-domain performance per model")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(deg["in_domain_macro_f1"].max() * 1.15, 0.85))
    plt.tight_layout()
    out = FIGURES_DIR / "in_vs_cross_domain.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 4. LaTeX tables
# ---------------------------------------------------------------------------
def _emit_results_table(name: str, configs: list[str], caption: str, label: str) -> None:
    master = pd.read_csv(TABLES_DIR / "experiment_log.csv")
    sub = master[master["source_config"].isin(configs)]
    acc = sub.pivot(index="model", columns="source_config", values="accuracy") \
             .reindex(index=MODEL_ORDER, columns=configs)
    f1 = sub.pivot(index="model", columns="source_config", values="macro_f1") \
            .reindex(index=MODEL_ORDER, columns=configs)

    # Per-column best (winning model per config)
    best_per_cfg = {cfg: f1[cfg].idxmax() for cfg in configs}

    col_spec = "l" + "cc" * len(configs)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    header_top = "Model"
    for cfg in configs:
        header_top += f" & \\multicolumn{{2}}{{c}}{{{cfg}}}"
    lines.append(header_top + " \\\\")

    cmid_parts = [f"\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(len(configs))]
    lines.append(" ".join(cmid_parts))

    header_sub = ""
    for _ in configs:
        header_sub += " & Acc & F1"
    lines.append(header_sub + " \\\\")
    lines.append("\\midrule")

    for m in MODEL_ORDER:
        row = MODEL_LABEL[m]
        for cfg in configs:
            a = acc.loc[m, cfg]
            f = f1.loc[m, cfg]
            if pd.isna(a):
                row += " & -- & --"
            else:
                f_cell = f"\\textbf{{{f:.3f}}}" if best_per_cfg[cfg] == m else f"{f:.3f}"
                row += f" & {a:.3f} & {f_cell}"
        lines.append(row + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = TABLES_DIR / f"{name}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {out.relative_to(ROOT)}")


def latex_tables() -> None:
    _emit_results_table(
        "results_single_source", SINGLE,
        "Single-source classification results (Accuracy and Macro-F1; "
        "bold marks the best Macro-F1 per source).",
        "tab:results-single",
    )
    _emit_results_table(
        "results_dual_source", DUAL,
        "Dual-source classification results (Accuracy and Macro-F1; "
        "bold marks the best Macro-F1 per source combination).",
        "tab:results-dual",
    )
    _emit_results_table(
        "results_triple_source", TRIPLE,
        "Triple-source classification results "
        "(Twitter + Skytrax + AirlineQuality; bold marks the best Macro-F1).",
        "tab:results-triple",
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Building Chapter 5 artifacts")
    print("=" * 60)
    build_master_log()
    figure_class_distribution()
    figure_text_length()
    figure_cross_domain_heatmap()
    figure_degradation()
    latex_tables()
    print("\nAll artifacts ready under results/{tables,figures}/")
