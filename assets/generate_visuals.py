#!/usr/bin/env python3
"""
Smart Outcome Predictor - evaluation visual generator
=====================================================

Creates the chart images used in the README from YOUR real data, results
tables and exported models, so every figure reflects actual numbers.

Run from the project root, after the notebook has exported the models and the
two results CSV files (see "Training and Exporting Models" in the README):

    python generate_visuals.py

Outputs
-------
assets/figures/data_overview.png
assets/figures/classification_comparison.png
assets/figures/regression_comparison.png
assets/figures/confusion_matrix.png
assets/figures/roc_curve.png                (only if the best model exposes scores)
assets/figures/regression_diagnostics.png
assets/figures/feature_importance.png
results/results_tables.md                   (Markdown tables to paste into the README)

IMPORTANT: TEST_SIZE and RANDOM_STATE must match the values used in your
notebook's train-test split, otherwise the confusion matrix, ROC curve,
diagnostics and feature importance would be computed on different test rows.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration - edit to match your notebook
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_data.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "assets" / "figures"

CLASS_TARGET = "completion_status"
REG_TARGET = "final_score"
TEST_SIZE = 0.2        # must match the notebook
RANDOM_STATE = 42      # must match the notebook
DPI = 200

# --------------------------------------------------------------------------- #
# Theme (matches the dark SVG diagrams in assets/)
# --------------------------------------------------------------------------- #
BG, CARD, GRID, TEXT, MUTED = "#0f172a", "#1e293b", "#334155", "#e2e8f0", "#94a3b8"
INDIGO, TEAL, AMBER, ROSE, SLATE = "#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#64748b"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": CARD,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "legend.facecolor": CARD,
        "legend.edgecolor": GRID,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path.relative_to(ROOT)}")


def suptitle(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.suptitle(title, fontsize=17, fontweight="bold", color=TEXT, y=1.08)
    if subtitle:
        fig.text(0.5, 1.01, subtitle, ha="center", fontsize=10.5, color=MUTED)


def require(path: Path, hint: str) -> None:
    if not path.exists():
        sys.exit(f"[missing] {path.relative_to(ROOT)}\n          {hint}")


def load_metadata() -> dict:
    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[CLASS_TARGET, REG_TARGET])
    y_class, y_reg = df[CLASS_TARGET], df[REG_TARGET]

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
    )
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return (Xc_te, yc_te), (Xr_te, yr_te)


def md_table(df: pd.DataFrame, best_row: int) -> str:
    cols = list(df.columns)
    out = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|",
    ]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c == "Model":
                cells.append(f"**{v}**" if idx == best_row else str(v))
            elif pd.isna(v):
                cells.append("N/A")
            else:
                cells.append(f"{float(v):.4f}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_data_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    # 1) class balance
    counts = df[CLASS_TARGET].value_counts().sort_index()
    labels = ["Not Completed (0)" if k == 0 else "Completed (1)" for k in counts.index]
    bars = axes[0].bar(labels, counts.values, color=[ROSE, TEAL][: len(counts)], width=0.55)
    for b, v in zip(bars, counts.values):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:,}\n({v / counts.sum():.1%})",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=10.5,
        )
    axes[0].set_title("Completion status balance")
    axes[0].set_ylabel("Students")
    axes[0].set_ylim(0, counts.max() * 1.25)
    axes[0].grid(axis="x", visible=False)

    # 2) final score distribution
    scores = df[REG_TARGET].dropna()
    axes[1].hist(scores, bins=24, color=INDIGO, edgecolor=BG, alpha=0.95)
    axes[1].axvline(scores.mean(), color=AMBER, lw=2, ls="--", label=f"Mean {scores.mean():.1f}")
    axes[1].axvline(scores.median(), color=TEAL, lw=2, ls=":", label=f"Median {scores.median():.1f}")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_title("Final score distribution")
    axes[1].set_xlabel("Final score")
    axes[1].set_ylabel("Students")
    axes[1].legend()

    # 3) missing values
    feats = df.drop(columns=[CLASS_TARGET, REG_TARGET])
    miss = (feats.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(10)
    if miss.empty:
        axes[2].text(
            0.5, 0.5, "No missing values\nin the input features",
            ha="center", va="center", fontsize=13, color=MUTED, transform=axes[2].transAxes,
        )
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].grid(False)
    else:
        axes[2].barh(miss.index[::-1], miss.values[::-1], color=AMBER)
        axes[2].set_xlabel("Missing (%)")
        axes[2].grid(axis="y", visible=False)
    axes[2].set_title("Missing values by feature")

    suptitle(
        fig,
        "Dataset Overview",
        f"{len(df):,} students  |  {feats.shape[1]} input features  |  2 targets",
    )
    fig.tight_layout()
    save(fig, "data_overview.png")


def plot_classification_comparison(cls_df: pd.DataFrame) -> None:
    metrics = [m for m in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"] if m in cls_df]
    df = cls_df.sort_values("F1 Score", ascending=True).reset_index(drop=True)
    palette = [INDIGO, TEAL, AMBER, ROSE, "#a855f7"]

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.85 * len(df) + 1.5)))
    h = 0.8 / len(metrics)
    y = np.arange(len(df))
    for i, m in enumerate(metrics):
        vals = df[m].astype(float).fillna(0).values
        ax.barh(y + i * h - 0.4 + h / 2, vals, height=h * 0.92, label=m, color=palette[i % len(palette)])
    ax.set_yticks(y)
    ax.set_yticklabels(df["Model"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score (higher is better)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=len(metrics), loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    best = df.iloc[-1]["Model"]
    suptitle(fig, "Classification - Model Comparison", f"Sorted by F1 Score  |  best: {best}")
    fig.tight_layout()
    save(fig, "classification_comparison.png")


def plot_regression_comparison(reg_df: pd.DataFrame) -> None:
    specs = [("MAE", True), ("RMSE", True), ("R² Score", False)]  # (metric, lower_is_better)
    specs = [(m, lo) for m, lo in specs if m in reg_df]
    fig, axes = plt.subplots(1, len(specs), figsize=(5.6 * len(specs), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (m, lower) in zip(axes, specs):
        d = reg_df.sort_values(m, ascending=not lower).reset_index(drop=True)
        best_val = d[m].min() if lower else d[m].max()
        colors = [TEAL if v == best_val else SLATE for v in d[m]]
        ax.barh(d["Model"][::-1], d[m][::-1], color=colors[::-1])
        for yy, v in enumerate(d[m][::-1]):
            ax.text(v, yy, f" {v:.3f}", va="center", ha="left" if v >= 0 else "right", color=TEXT, fontsize=10)
        ax.set_title(f"{m}  ({'lower' if lower else 'higher'} is better)")
        ax.grid(axis="y", visible=False)
        lo, hi = min(0, d[m].min()), d[m].max()
        ax.set_xlim(lo * 1.15 if lo < 0 else 0, hi * 1.18 if hi > 0 else 0.05)
    best = reg_df.sort_values("RMSE").iloc[0]["Model"]
    suptitle(fig, "Regression - Model Comparison", f"Teal = best on that metric  |  selected by RMSE: {best}")
    fig.tight_layout()
    save(fig, "regression_comparison.png")


def plot_confusion(model, X_te, y_te, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    for ax, norm, title, fmt in (
        (axes[0], None, "Counts", "d"),
        (axes[1], "true", "Normalised by actual class", ".1%"),
    ):
        ConfusionMatrixDisplay.from_estimator(
            model, X_te, y_te, normalize=norm, ax=ax, cmap="viridis", colorbar=False,
            display_labels=["Not Completed", "Completed"], values_format=fmt,
        )
        ax.set_title(title)
        ax.grid(False)
    suptitle(fig, "Confusion Matrix - Best Classifier", f"{name}  |  evaluated on held-out test data")
    fig.tight_layout()
    save(fig, "confusion_matrix.png")


def plot_roc(model, X_te, y_te, name: str) -> None:
    try:
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax, color=TEAL, lw=3, name=name)
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.5, label="Chance")
        ax.legend(loc="lower right")
        ax.set_title("ROC Curve - Best Classifier")
        save(fig, "roc_curve.png")
    except Exception as exc:  # e.g. hard voting exposes no scores
        plt.close("all")
        print(f"  skipped roc_curve.png ({type(exc).__name__}: model exposes no scores)")


def plot_regression_diagnostics(model, X_te, y_te, name: str) -> None:
    pred = model.predict(X_te)
    resid = y_te.values - pred
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae = float(mean_absolute_error(y_te, pred))
    r2 = float(r2_score(y_te, pred))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    axes[0].scatter(y_te, pred, s=22, alpha=0.65, color=INDIGO, edgecolor="none")
    lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    axes[0].plot(lims, lims, color=AMBER, lw=2, ls="--", label="Perfect prediction")
    axes[0].set_xlabel("Actual final score")
    axes[0].set_ylabel("Predicted final score")
    axes[0].set_title("Actual vs predicted")
    axes[0].legend()

    axes[1].hist(resid, bins=28, color=TEAL, edgecolor=BG)
    axes[1].axvline(0, color=AMBER, lw=2, ls="--")
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_ylabel("Students")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_title("Residual distribution")

    axes[2].scatter(pred, resid, s=22, alpha=0.65, color=ROSE, edgecolor="none")
    axes[2].axhline(0, color=AMBER, lw=2, ls="--")
    axes[2].set_xlabel("Predicted final score")
    axes[2].set_ylabel("Residual")
    axes[2].set_title("Residuals vs predicted")

    suptitle(
        fig,
        "Regression Diagnostics - Best Regressor",
        f"{name}  |  MAE {mae:.2f}  |  RMSE {rmse:.2f}  |  R² {r2:.3f}",
    )
    fig.tight_layout()
    save(fig, "regression_diagnostics.png")


def plot_feature_importance(clf, reg, clf_te, reg_te, clf_name: str, reg_name: str) -> None:
    jobs = [
        (clf, clf_te, "f1", INDIGO, f"Classifier: {clf_name}", "Drop in F1 Score when shuffled"),
        (reg, reg_te, "neg_root_mean_squared_error", TEAL, f"Regressor: {reg_name}", "Increase in RMSE when shuffled"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    drew = False
    for ax, (model, (X_te, y_te), scoring, color, title, xlabel) in zip(axes, jobs):
        try:
            res = permutation_importance(
                model, X_te, y_te, scoring=scoring, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
            )
        except Exception as exc:
            ax.text(0.5, 0.5, f"Unavailable\n({type(exc).__name__})", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
            ax.set_title(title)
            continue
        order = np.argsort(res.importances_mean)[-10:]
        ax.barh(
            np.array(X_te.columns)[order], res.importances_mean[order],
            xerr=res.importances_std[order], color=color, ecolor=MUTED, capsize=3,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        drew = True
    if drew:
        suptitle(fig, "Permutation Feature Importance", "Top 10 original input features, computed on held-out test data")
        fig.tight_layout()
        save(fig, "feature_importance.png")
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("Smart Outcome Predictor - generating visuals\n")

    require(DATA_PATH, "Place your dataset at data/student_data.csv (or edit DATA_PATH).")
    require(MODELS_DIR / "best_classification_model.joblib", "Run the notebook's export cell first.")
    require(MODELS_DIR / "best_regression_model.joblib", "Run the notebook's export cell first.")
    require(RESULTS_DIR / "classification_results.csv", "Export classification_results from the notebook.")
    require(RESULTS_DIR / "regression_results.csv", "Export regression_results from the notebook.")

    df = pd.read_csv(DATA_PATH)
    cls_results = pd.read_csv(RESULTS_DIR / "classification_results.csv")
    reg_results = pd.read_csv(RESULTS_DIR / "regression_results.csv")
    clf = joblib.load(MODELS_DIR / "best_classification_model.joblib")
    reg = joblib.load(MODELS_DIR / "best_regression_model.joblib")

    meta = load_metadata()
    clf_name = meta.get("classification_model") or cls_results.sort_values("F1 Score").iloc[-1]["Model"]
    reg_name = meta.get("regression_model") or reg_results.sort_values("RMSE").iloc[0]["Model"]

    (clf_te, reg_te) = split_data(df)

    print("Figures")
    plot_data_overview(df)
    plot_classification_comparison(cls_results)
    plot_regression_comparison(reg_results)
    plot_confusion(clf, *clf_te, clf_name)
    plot_roc(clf, *clf_te, clf_name)
    plot_regression_diagnostics(reg, *reg_te, reg_name)
    plot_feature_importance(clf, reg, clf_te, reg_te, clf_name, reg_name)

    # README-ready tables
    RESULTS_DIR.mkdir(exist_ok=True)
    cls_sorted = cls_results.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    reg_sorted = reg_results.sort_values("RMSE", ascending=True).reset_index(drop=True)
    md = (
        "### Classification results (sorted by F1 Score)\n\n"
        + md_table(cls_sorted, 0)
        + "\n\n### Regression results (sorted by RMSE)\n\n"
        + md_table(reg_sorted, 0)
        + "\n"
    )
    (RESULTS_DIR / "results_tables.md").write_text(md, encoding="utf-8")
    print("\nTables\n  saved  results/results_tables.md  (paste into README section 'Results')")
    print("\nDone. Commit assets/figures/ so the README images render on GitHub.")


if __name__ == "__main__":
    main()
