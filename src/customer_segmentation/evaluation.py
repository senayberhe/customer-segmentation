"""Held-out evaluation utilities for the purchase-behavior models.

Purchase data has many rows per customer, so a plain random row split
leaks a customer's habits from train into test and flatters every metric.
Everything here splits by customer ID instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclass(frozen=True)
class Evaluation:
    """Held-out metrics for a model next to a naive baseline.

    ``baseline`` is what you get by ignoring price entirely (the training
    base rate, brand shares, or mean quantity). A model that can't beat it
    has learned nothing usable for prediction.
    """

    model: dict[str, float]
    baseline: dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"model": self.model, "baseline": self.baseline})


def split_by_customer(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into train/test so no customer appears in both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["ID"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def cross_validate_by_customer(
    make_model: Callable[[], object],
    df: pd.DataFrame,
    n_splits: int = 5,
    occasions_only: bool = False,
    random_state: int = 0,
) -> pd.DataFrame:
    """Customer-grouped k-fold evaluation of a purchase-behavior model.

    Parameters
    ----------
    make_model:
        Zero-argument factory returning a fresh unfitted model that has
        ``fit(df)`` and ``evaluate(df)`` (all three purchase models do).
    df:
        Purchase data with an ``ID`` column.
    occasions_only:
        Restrict train and test folds to rows where a purchase occurred
        (``Incidence == 1``), as the brand-choice and quantity models need.
    random_state:
        Seed for shuffling customers into folds. Shuffling matters: without
        it folds depend on ID order, and results shift with library version.

    Returns
    -------
    pd.DataFrame
        One row per fold; columns are ``<metric>`` for the model and
        ``baseline_<metric>`` for the naive baseline.
    """
    rows = []
    folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in folds.split(df, groups=df["ID"]):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        if occasions_only:
            train, test = train[train["Incidence"] == 1], test[test["Incidence"] == 1]
        result = make_model().fit(train).evaluate(test)
        rows.append({
            **result.model,
            **{f"baseline_{k}": v for k, v in result.baseline.items()},
        })
    return pd.DataFrame(rows)
