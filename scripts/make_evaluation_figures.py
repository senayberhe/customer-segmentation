"""Regenerate the evaluation charts in docs/images/ and print the numbers
quoted in the README.

Run from the project root:  python -m scripts.make_evaluation_figures
"""

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from customer_segmentation import (
    BrandChoiceModel,
    PurchasePropensityModel,
    PurchaseQuantityModel,
    SegmentationPipeline,
    load_purchase_data,
    load_segmentation_data,
)
from customer_segmentation.data_loader import SEGMENTATION_FEATURES
from customer_segmentation.evaluation import cross_validate_by_customer
from customer_segmentation.history import add_purchase_history
from customer_segmentation.segment_behavior import (
    assign_segments,
    brand_shares_by_segment,
    segment_price_response,
)
from customer_segmentation.segmentation import SEGMENT_NAMES

OUT = Path(__file__).parents[1] / "docs" / "images"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MODEL, BASELINE = "#2a78d6", "#8a8983"
BRAND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5, fixed order

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


def segment_charts() -> None:
    df = load_purchase_data()
    df["Segment"] = assign_segments(df, SegmentationPipeline.load())
    label = lambda seg: f"{SEGMENT_NAMES[seg]}\n(segment {seg})"

    # --- brand share by segment: 100% stacked bars, one row per segment
    shares = brand_shares_by_segment(df)
    print("\n== Brand share by segment ==")
    print(shares.round(2))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    order = list(shares.index)[::-1]
    left = {seg: 0.0 for seg in order}
    for j, brand in enumerate(shares.columns):
        for seg in order:
            share = shares.loc[seg, brand]
            ax.barh(label(seg), share, left=left[seg], color=BRAND_COLORS[j], height=0.62,
                    edgecolor=SURFACE, linewidth=2, label=f"Brand {brand}" if seg == order[0] else None)
            if share >= 0.10:
                ax.text(left[seg] + share / 2, label(seg), f"{share:.0%}", ha="center", va="center",
                        color="#0b0b0b" if j in (3, 4, 2) else "#ffffff", fontsize=9)
            left[seg] += share
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of the segment's purchases")
    ax.set_title("Each segment has a different favourite brand", loc="left", color=INK, fontsize=11)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=9,
              handlelength=1, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "segment_brand_shares.png", dpi=160)
    plt.close(fig)

    # --- price elasticity by segment: point + 95% CI, two panels (different units)
    response = segment_price_response(df, n_boot=200, random_state=0)
    summary = response.summary
    print("\n== Elasticity by segment ==")
    print(summary.filter(like="elasticity").round(2))
    for metric, a, b in [("propensity_elasticity", 3, 1), ("propensity_elasticity", 3, 0),
                         ("quantity_elasticity", 1, 0), ("quantity_elasticity", 3, 1)]:
        d, lo, hi = response.compare(metric, a, b)
        print(f"  {metric}: segment {a} - segment {b} = {d:.2f} [{lo:.2f}, {hi:.2f}]")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    order = summary["propensity_elasticity"].sort_values(ascending=False).index[::-1]  # steepest on top
    for ax, (metric, title) in zip(axes, [
        ("propensity_elasticity", "Purchase probability\n(does a trip end in a purchase?)"),
        ("quantity_elasticity", "Purchase quantity\n(how many units, given a purchase?)"),
    ]):
        for y, seg in enumerate(order):
            r = summary.loc[seg]
            ax.plot([r[f"{metric}_ci_low"], r[f"{metric}_ci_high"]], [y, y], color=MODEL, linewidth=2,
                    solid_capstyle="round")
            ax.scatter([r[metric]], [y], color=MODEL, s=60, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.text(r[metric], y + 0.18, f"{r[metric]:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([label(s) for s in order])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Price elasticity (95% CI)   ← more price-sensitive")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "segment_elasticity.png", dpi=160)
    plt.close(fig)


def feature_ladder_chart() -> None:
    """Held-out performance as features are added: price -> segment -> history."""
    df = load_purchase_data()
    df["Segment"] = assign_segments(df, SegmentationPipeline.load())
    df = add_purchase_history(df)
    steps = [
        ("Price\nonly", {}),
        ("+ Segment", {"use_segment": True}),
        ("+ History", {"use_history": True}),
        ("+ Both", {"use_segment": True, "use_history": True}),
    ]
    panels = [
        ("Does this trip end in a purchase?\nROC AUC (higher = better)", PurchasePropensityModel, False,
         "roc_auc", 0.5, 0.0, 0.85),
        ("Which brand is bought?\nAccuracy (higher = better)", BrandChoiceModel, True,
         "accuracy", None, 0.0, 0.85),
    ]
    print("\n== Feature ladder (customer-grouped 5-fold CV) ==")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (title, cls, occasions, metric, floor, lo, hi) in zip(axes, panels):
        vals, errs = [], []
        for name, kwargs in steps:
            cv = cross_validate_by_customer(lambda: cls(**kwargs), df, occasions_only=occasions)
            vals.append(cv[metric].mean())
            errs.append(cv[metric].std())
            print(f"{title.splitlines()[0]:<34} {name.replace(chr(10), ' '):<12} {vals[-1]:.3f}±{errs[-1]:.3f}")
        baseline = cv[f"baseline_{metric}"].mean()
        x = range(len(steps))
        bars = ax.bar(x, vals, yerr=errs, color=[MODEL] * len(steps), width=0.6, capsize=4,
                      error_kw={"ecolor": INK, "linewidth": 1})
        ax.axhline(baseline, color=BASELINE, linewidth=1.5, linestyle=(0, (4, 3)))
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + (hi - lo) * 0.015, f"{v:.3f}",
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels([n for n, _ in steps])
        ax.set_ylim(lo, hi)
        ax.set_title(f"{title}\ndashed line = no-skill baseline ({baseline:.2f})", loc="left",
                     color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "feature_ladder.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    cluster_selection_chart()
    heldout_chart()
    segment_charts()
    feature_ladder_chart()
