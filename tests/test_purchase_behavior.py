"""Tests for the purchase-behavior models (propensity, brand choice, quantity)."""

import numpy as np
import pytest

from customer_segmentation.purchase_behavior import (
    BrandChoiceModel,
    CoefficientEstimate,
    PurchasePropensityModel,
    PurchaseQuantityModel,
)


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

    def test_bootstrap_significance_raises_before_fit(self, purchase_occasions):
        model = BrandChoiceModel()
        with pytest.raises(RuntimeError):
            model.bootstrap_own_price_significance(1, purchase_occasions, n_boot=5)

    def test_bootstrap_significance_ci_is_well_formed(self, purchase_occasions):
        model = BrandChoiceModel().fit(purchase_occasions)
        est = model.bootstrap_own_price_significance(
            1, purchase_occasions, n_boot=20, random_state=0
        )
        assert isinstance(est, CoefficientEstimate)
        assert est.brand == 1
        assert est.ci_low <= est.mean <= est.ci_high

    def test_bootstrap_significance_detects_strong_effect(self, purchase_occasions):
        # Brand 1 has the steepest, most consistent price effect in the
        # synthetic fixture — its CI should not cross zero.
        model = BrandChoiceModel().fit(purchase_occasions)
        est = model.bootstrap_own_price_significance(
            1, purchase_occasions, n_boot=30, random_state=0
        )
        assert est.is_significant


class TestCoefficientEstimate:
    def test_significant_when_ci_excludes_zero(self):
        est = CoefficientEstimate(brand=1, mean=-2.0, ci_low=-2.5, ci_high=-1.5)
        assert est.is_significant is True

    def test_not_significant_when_ci_crosses_zero(self):
        est = CoefficientEstimate(brand=3, mean=0.4, ci_low=-0.3, ci_high=1.3)
        assert est.is_significant is False


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
