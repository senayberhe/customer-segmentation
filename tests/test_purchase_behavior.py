"""Tests for the purchase-behavior models (propensity, brand choice, quantity)."""

import numpy as np
import pandas as pd
import pytest

from src.customer_segmentation.purchase_behavior import (
    PRICE_COLUMNS,
    PROMOTION_COLUMNS,
    BrandChoiceModel,
    PurchasePropensityModel,
    PurchaseQuantityModel,
)


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


# ---------------------------------------------------------------------------
# PurchasePropensityModel
# ---------------------------------------------------------------------------

class TestPurchasePropensityModel:
    def test_predict_proba_raises_before_fit(self):
        model = PurchasePropensityModel()
        with pytest.raises(RuntimeError):
            model.predict_proba(2.0)

    def test_price_elasticity_raises_before_fit(self):
        model = PurchasePropensityModel()
        with pytest.raises(RuntimeError):
            model.price_elasticity([1.0, 2.0])

    def test_fit_returns_self(self, purchase_df):
        model = PurchasePropensityModel()
        assert model.fit(purchase_df) is model

    def test_predict_proba_shape_and_bounds(self, purchase_df):
        model = PurchasePropensityModel().fit(purchase_df)
        proba = model.predict_proba([1.0, 2.0, 3.0])
        assert proba.shape == (3,)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_higher_price_lowers_purchase_probability(self, purchase_df):
        model = PurchasePropensityModel().fit(purchase_df)
        low, high = model.predict_proba([1.0, 3.0])
        assert low > high

    def test_price_elasticity_shape(self, purchase_df):
        model = PurchasePropensityModel().fit(purchase_df)
        elasticity = model.price_elasticity(np.arange(1.0, 3.0, 0.5))
        assert elasticity.shape == (4,)

    def test_with_promotion_feature(self, purchase_df):
        model = PurchasePropensityModel(use_promotion=True).fit(purchase_df)
        proba = model.predict_proba([2.0], promotion=1.0)
        assert proba.shape == (1,)

    def test_save_load_roundtrip(self, purchase_df, tmp_path):
        model = PurchasePropensityModel().fit(purchase_df)
        original = model.predict_proba([1.5, 2.5])
        path = model.save(tmp_path)
        loaded = PurchasePropensityModel.load(path)
        np.testing.assert_array_almost_equal(loaded.predict_proba([1.5, 2.5]), original)


# ---------------------------------------------------------------------------
# BrandChoiceModel
# ---------------------------------------------------------------------------

class TestBrandChoiceModel:
    def test_predict_proba_raises_before_fit(self, purchase_occasions):
        model = BrandChoiceModel()
        with pytest.raises(RuntimeError):
            model.predict_proba(purchase_occasions)

    def test_own_price_elasticity_raises_before_fit(self):
        model = BrandChoiceModel()
        with pytest.raises(RuntimeError):
            model.own_price_elasticity(1, [1.0, 2.0])

    def test_fit_returns_self(self, purchase_occasions):
        model = BrandChoiceModel()
        assert model.fit(purchase_occasions) is model

    def test_classes_match_observed_brands(self, purchase_occasions):
        model = BrandChoiceModel().fit(purchase_occasions)
        assert set(model.classes_) == set(purchase_occasions["Brand"].unique())

    def test_predict_proba_shape_sums_to_one(self, purchase_occasions):
        model = BrandChoiceModel().fit(purchase_occasions)
        proba = model.predict_proba(purchase_occasions.head(10))
        assert proba.shape == (10, len(model.classes_))
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(10))

    def test_own_price_elasticity_shape(self, purchase_occasions):
        model = BrandChoiceModel().fit(purchase_occasions)
        elasticity = model.own_price_elasticity(1, np.arange(1.0, 3.0, 0.5))
        assert elasticity.shape == (4,)

    def test_save_load_roundtrip(self, purchase_occasions, tmp_path):
        model = BrandChoiceModel().fit(purchase_occasions)
        original = model.predict_proba(purchase_occasions.head(5))
        path = model.save(tmp_path)
        loaded = BrandChoiceModel.load(path)
        np.testing.assert_array_almost_equal(
            loaded.predict_proba(purchase_occasions.head(5)), original
        )


# ---------------------------------------------------------------------------
# PurchaseQuantityModel
# ---------------------------------------------------------------------------

class TestPurchaseQuantityModel:
    def test_predict_raises_before_fit(self):
        model = PurchaseQuantityModel()
        with pytest.raises(RuntimeError):
            model.predict(2.0)

    def test_price_elasticity_raises_before_fit(self):
        model = PurchaseQuantityModel()
        with pytest.raises(RuntimeError):
            model.price_elasticity([1.0, 2.0])

    def test_fit_returns_self(self, purchase_occasions):
        model = PurchaseQuantityModel()
        assert model.fit(purchase_occasions) is model

    def test_predict_shape(self, purchase_occasions):
        model = PurchaseQuantityModel().fit(purchase_occasions)
        predicted = model.predict([1.0, 2.0, 3.0])
        assert predicted.shape == (3,)

    def test_higher_price_lowers_quantity(self, purchase_occasions):
        model = PurchaseQuantityModel().fit(purchase_occasions)
        low, high = model.predict([1.0, 3.0])
        assert low > high

    def test_price_elasticity_shape(self, purchase_occasions):
        model = PurchaseQuantityModel().fit(purchase_occasions)
        elasticity = model.price_elasticity(np.arange(1.0, 3.0, 0.5))
        assert elasticity.shape == (4,)

    def test_save_load_roundtrip(self, purchase_occasions, tmp_path):
        model = PurchaseQuantityModel().fit(purchase_occasions)
        original = model.predict([1.5, 2.5])
        path = model.save(tmp_path)
        loaded = PurchaseQuantityModel.load(path)
        np.testing.assert_array_almost_equal(loaded.predict([1.5, 2.5]), original)
