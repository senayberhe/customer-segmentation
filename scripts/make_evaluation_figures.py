"""Regenerate the evaluation charts in docs/images/ and print the numbers
quoted in the README.

Run from the project root:  python -m scripts.make_evaluation_figures
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.customer_segmentation import (
    BrandChoiceModel,
    PurchasePropensityModel,
    PurchaseQuantityModel,
    SegmentationPipeline,
    load_purchase_data,
    load_segmentation_data,
)
from src.customer_segmentation.data_loader import SEGMENTATION_FEATURES
from src.customer_segmentation.evaluation import cross_validate_by_customer

OUT = Path(__file__).parents[1] / "docs" / "images"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MODEL, BASELINE = "#2a78d6", "#8a8983"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def cluster_selection_chart() -> None:
    X = load_segmentation_data()[SEGMENTATION_FEATURES]
    pipeline = SegmentationPipeline(n_components=3, n_clusters=4)
    diag = pipeline.cluster_diagnostics(X, range(2, 11))
    stability = pipeline.stability(X, n_boot=30)
    print("\n== Cluster diagnostics ==")
    print(diag.round(3))
    print(f"Bootstrap stability at k=4 (ARI): mean {stability.mean():.2f}, min {stability.min():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    panels = [
        ("inertia", "Inertia (lower = tighter)", "Elbow: gains flatten after k=4"),
        ("silhouette", "Silhouette (higher = better separated)", "Silhouette: weak throughout"),
    ]
    for ax, (col, ylabel, title) in zip(axes, panels):
        ax.plot(diag.index, diag[col], color=MODEL, linewidth=2)
        ax.scatter(diag.index, diag[col], color=MODEL, s=28, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([4], [diag.loc[4, col]], color=MODEL, s=90, zorder=4, edgecolor=SURFACE, linewidth=2)
        ax.annotate("k=4 (chosen)", (4, diag.loc[4, col]), textcoords="offset points",
                    xytext=(10, 10), color=INK, fontsize=9)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(list(diag.index))
    axes[1].axhline(0.5, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)))
    axes[1].text(10, 0.505, "~0.5+ is often read as clear structure", ha="right", va="bottom",
                 color=MUTED, fontsize=8)
    axes[1].set_ylim(0, 0.6)
    fig.tight_layout()
    fig.savefig(OUT / "cluster_selection.png", dpi=160)
    plt.close(fig)


def heldout_chart() -> None:
    df = load_purchase_data()
    specs = [
        ("Purchase propensity\nROC AUC (higher = better)", PurchasePropensityModel, False, "roc_auc", 0, 1),
        ("Brand choice\naccuracy (higher = better)", BrandChoiceModel, True, "accuracy", 0, 0.6),
        ("Purchase quantity\nR² (higher = better)", PurchaseQuantityModel, True, "r2", -0.05, 0.1),
    ]
    print("\n== Held-out, customer-grouped 5-fold CV (mean ± std across folds) ==")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (title, factory, occasions, metric, lo, hi) in zip(axes, specs):
        cv = cross_validate_by_customer(factory, df, n_splits=5, occasions_only=occasions)
        vals = [cv[f"baseline_{metric}"].mean(), cv[metric].mean()]
        errs = [cv[f"baseline_{metric}"].std(), cv[metric].std()]
        print(f"{title.splitlines()[0]:<22} {metric:<9} model {vals[1]:.3f}±{errs[1]:.3f}  "
              f"baseline {vals[0]:.3f}±{errs[0]:.3f}")
        bars = ax.bar(["Baseline", "Model"], vals, yerr=errs, color=[BASELINE, MODEL],
                      width=0.55, capsize=4, error_kw={"ecolor": INK, "linewidth": 1})
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, max(v + e, 0) + (hi - lo) * 0.02, f"{v:.3f}",
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_ylim(lo, hi)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "heldout_performance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    cluster_selection_chart()
    heldout_chart()
