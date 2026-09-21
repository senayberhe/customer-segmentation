"""How each customer segment responds to price.

Joins the two pipelines: the fitted ``SegmentationPipeline`` assigns every
shopper in the purchase data to a segment from their demographics, then
price sensitivity is estimated separately per segment.

Uncertainty comes from a *customer-level* bootstrap: whole shoppers are
resampled, not individual trips. Trips from the same person are strongly
correlated, so resampling trips would understate the uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from .data_loader import SEGMENTATION_FEATURES
from .purchase_behavior import PRICE_COLUMNS, PurchaseQuantityModel
from .segmentation import SegmentationPipeline

METRICS = ("propensity_elasticity", "quantity_elasticity")


def assign_segments(df: pd.DataFrame, pipeline: SegmentationPipeline) -> pd.Series:
    """Segment label for every row of purchase data *df*, from each
    shopper's demographics (constant within a shopper) via *pipeline*."""
    customers = df.groupby("ID")[SEGMENTATION_FEATURES].first()
    labels = pd.Series(pipeline.predict(customers), index=customers.index, name="Segment")
    return df["ID"].map(labels).rename("Segment")


@dataclass(frozen=True)
class SegmentResponse:
    """Per-segment price response with bootstrap uncertainty.

    ``summary`` has one row per segment. ``draws`` maps each metric to a
    (n_boot, n_segments) frame of bootstrap estimates, so any two segments
    can be compared with :meth:`compare`.
    """

    summary: pd.DataFrame
    draws: dict[str, pd.DataFrame]
    ci: float

    def compare(self, metric: str, segment_a: int, segment_b: int) -> tuple[float, float, float]:
        """Difference (a - b) in *metric*, with its bootstrap CI. If the CI
        crosses zero the two segments can't be told apart on this metric."""
        d = self.draws[metric]
        diff = d[segment_a] - d[segment_b]
        alpha = (1 - self.ci) / 2 * 100
        lo, hi = np.percentile(diff, [alpha, 100 - alpha])
        point = self.summary.loc[segment_a, metric] - self.summary.loc[segment_b, metric]
        return float(point), float(lo), float(hi)


class _SegmentData:
    """Arrays for one segment, pre-split by customer for fast resampling."""

    def __init__(self, seg_df: pd.DataFrame) -> None:
        self.customers = seg_df["ID"].unique()
        self.rows = {i: np.asarray(v) for i, v in seg_df.groupby("ID").indices.items()}

        self.mean_price = seg_df[PRICE_COLUMNS].mean(axis=1).to_numpy()
        self.incidence = seg_df["Incidence"].to_numpy()

        occasions = seg_df[seg_df["Incidence"] == 1]
        self.occ_features = PurchaseQuantityModel._features(occasions).to_numpy()
        self.occ_quantity = occasions["Quantity"].to_numpy()
        # Positions of each customer's purchases within the occasions arrays.
        occ_positions = pd.Series(np.arange(len(occasions)), index=occasions["ID"].to_numpy())
        self.occ_rows = {i: v.to_numpy() for i, v in occ_positions.groupby(level=0)}

    def resample(self, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray]:
        """Row positions (all trips, purchases only) for a customer resample
        — or for the original sample when *rng* is None."""
        ids = self.customers if rng is None else rng.choice(self.customers, len(self.customers))
        trips = np.concatenate([self.rows[i] for i in ids])
        purchases = [self.occ_rows[i] for i in ids if i in self.occ_rows]
        purchases = np.concatenate(purchases) if purchases else np.array([], dtype=int)
        return trips, purchases

    def elasticities(self, trips: np.ndarray, purchases: np.ndarray, ref_price: float) -> dict[str, float]:
        """Propensity and quantity elasticity at *ref_price*."""
        logit = LogisticRegression(solver="lbfgs", max_iter=1000)
        logit.fit(self.mean_price[trips, None], self.incidence[trips])
        beta = logit.coef_[0, 0]
        p_buy = logit.predict_proba([[ref_price]])[0, 1]

        lin = LinearRegression().fit(self.occ_features[purchases], self.occ_quantity[purchases])
        qty = lin.intercept_ + lin.coef_[0] * ref_price  # Promotion_Incidence = 0
        return {
            "propensity_elasticity": beta * ref_price * (1 - p_buy),
            "quantity_elasticity": lin.coef_[0] * ref_price / qty,
        }


def segment_price_response(
    df: pd.DataFrame,
    n_boot: int = 200,
    ci: float = 0.95,
    random_state: int | None = 0,
) -> SegmentResponse:
    """Estimate price elasticity of purchase probability and of quantity for
    each segment, with customer-level bootstrap intervals.

    Parameters
    ----------
    df:
        Purchase data with ``ID``, ``Segment`` (see :func:`assign_segments`),
        prices, ``Incidence``, ``Brand``, ``Quantity`` and promotions.
    n_boot:
        Number of customer-level bootstrap resamples per segment.

    Elasticities are evaluated at one shared reference price (the overall
    mean of the five brands' average price) so segments are comparable, and
    that price sits inside the range actually observed.

    Returns
    -------
    SegmentResponse
    """
    rng = np.random.default_rng(random_state)
    ref_price = float(df[PRICE_COLUMNS].mean(axis=1).mean())
    alpha = (1 - ci) / 2 * 100

    rows, draws = {}, {m: {} for m in METRICS}
    for segment, seg_df in df.groupby("Segment"):
        data = _SegmentData(seg_df.reset_index(drop=True))
        point = data.elasticities(*data.resample(None), ref_price)
        boot = pd.DataFrame(
            [data.elasticities(*data.resample(rng), ref_price) for _ in range(n_boot)]
        )

        occasions = seg_df[seg_df["Incidence"] == 1]
        shares = occasions["Brand"].value_counts(normalize=True)
        row = {
            "customers": seg_df["ID"].nunique(),
            "trips": len(seg_df),
            "purchase_rate": seg_df["Incidence"].mean(),
            "mean_quantity": occasions["Quantity"].mean(),
            "top_brand": int(shares.idxmax()),
            "top_brand_share": shares.max(),
        }
        for metric in METRICS:
            lo, hi = np.percentile(boot[metric], [alpha, 100 - alpha])
            row[metric] = point[metric]
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = lo, hi
            draws[metric][segment] = boot[metric].to_numpy()
        rows[segment] = row

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "Segment"
    return SegmentResponse(
        summary=summary,
        draws={m: pd.DataFrame(d) for m, d in draws.items()},
        ci=ci,
    )


def brand_shares_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Share of purchases going to each brand, per segment (rows sum to 1)."""
    occasions = df[df["Incidence"] == 1]
    return pd.crosstab(occasions["Segment"], occasions["Brand"], normalize="index")
