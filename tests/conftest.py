"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest

from src.customer_segmentation.purchase_behavior import PRICE_COLUMNS, PROMOTION_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def purchase_df() -> pd.DataFrame:
    """Synthetic purchase-occasion data with a deliberate price -> behavior
    relationship, so the fitted models have a non-degenerate direction to
    test against (higher price => lower purchase probability / quantity).
    """
    rng = np.random.default_rng(0)
    n = 2000

    prices = rng.uniform(1.0, 3.0, size=(n, 5))
    promotions = rng.integers(0, 2, size=(n, 5))
    mean_price = prices.mean(axis=1)

    purchase_prob = 1 / (1 + np.exp(2 * (mean_price - 2.0)))
    incidence = rng.binomial(1, purchase_prob)

    # Cheaper brands are chosen more often, softmax over negative price.
    brand_logits = -prices * 3
    brand_probs = np.exp(brand_logits) / np.exp(brand_logits).sum(axis=1, keepdims=True)
    brand = np.array([
        rng.choice([1, 2, 3, 4, 5], p=brand_probs[i]) for i in range(n)
    ])
    chosen_price = prices[np.arange(n), brand - 1]
    chosen_promotion = promotions[np.arange(n), brand - 1]

    quantity = np.clip(
        np.round(6 - 1.5 * chosen_price + 0.5 * chosen_promotion + rng.normal(0, 0.3, n)),
        1,
        None,
    )

    df = pd.DataFrame(prices, columns=PRICE_COLUMNS)
    df.insert(0, "ID", np.arange(n) // 10)  # 200 customers, 10 trips each
    for i, col in enumerate(PROMOTION_COLUMNS):
        df[col] = promotions[:, i]
    df["Incidence"] = incidence
    df["Brand"] = np.where(incidence == 1, brand, 0)
    df["Quantity"] = np.where(incidence == 1, quantity, 0).astype(int)
    return df


@pytest.fixture
def purchase_occasions(purchase_df) -> pd.DataFrame:
    """Rows where a purchase actually happened (Incidence == 1)."""
    return purchase_df[purchase_df["Incidence"] == 1].reset_index(drop=True)
