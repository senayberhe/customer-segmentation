"""Tests for held-out evaluation: customer-grouped splits and model metrics."""

import numpy as np
import pandas as pd
import pytest

from src.customer_segmentation.evaluation import (
    Evaluation,
    cross_validate_by_customer,
    split_by_customer,
)
from src.customer_segmentation.purchase_behavior import (
    BrandChoiceModel,
    PurchasePropensityModel,
    PurchaseQuantityModel,
)
from src.customer_segmentation.segmentation import SegmentationPipeline


# ---------------------------------------------------------------------------
# Customer-grouped splitting
# ---------------------------------------------------------------------------

class TestSplitByCustomer:
    def test_no_customer_in_both_sets(self, purchase_df):
        train, test = split_by_customer(purchase_df)
        assert set(train["ID"]).isdisjoint(test["ID"])

    def test_covers_every_row_once(self, purchase_df):
        train, test = split_by_customer(purchase_df)
        assert len(train) + len(test) == len(purchase_df)

    def test_test_size_is_respected_by_customers(self, purchase_df):
        _, test = split_by_customer(purchase_df, test_size=0.25)
        assert test["ID"].nunique() == pytest.approx(0.25 * purchase_df["ID"].nunique(), abs=1)

    def test_reproducible_with_same_seed(self, purchase_df):
        a, _ = split_by_customer(purchase_df, random_state=7)
        b, _ = split_by_customer(purchase_df, random_state=7)
        pd.testing.assert_frame_equal(a, b)


class _SpyModel:
    """Records which customers it is fit and evaluated on."""

    fit_ids: list[set] = []
    test_ids: list[set] = []
    fit_incidence: list[set] = []

    def fit(self, df):
        _SpyModel.fit_ids.append(set(df["ID"]))
        _SpyModel.fit_incidence.append(set(df["Incidence"]))
        return self

    def evaluate(self, df):
        _SpyModel.test_ids.append(set(df["ID"]))
        return Evaluation(model={"score": 1.0}, baseline={"score": 0.0})


class TestCrossValidateByCustomer:
    @pytest.fixture(autouse=True)
    def _reset_spy(self):
        _SpyModel.fit_ids, _SpyModel.test_ids, _SpyModel.fit_incidence = [], [], []

    def test_one_row_per_fold(self, purchase_df):
        result = cross_validate_by_customer(_SpyModel, purchase_df, n_splits=4)
        assert len(result) == 4

    def test_columns_hold_model_and_baseline(self, purchase_df):
        result = cross_validate_by_customer(_SpyModel, purchase_df, n_splits=3)
        assert set(result.columns) == {"score", "baseline_score"}

    def test_train_and_test_customers_never_overlap(self, purchase_df):
        cross_validate_by_customer(_SpyModel, purchase_df, n_splits=5)
        for train_ids, test_ids in zip(_SpyModel.fit_ids, _SpyModel.test_ids):
            assert train_ids.isdisjoint(test_ids)

    def test_every_customer_is_tested_exactly_once(self, purchase_df):
        cross_validate_by_customer(_SpyModel, purchase_df, n_splits=5)
        tested = [i for ids in _SpyModel.test_ids for i in ids]
        assert sorted(tested) == sorted(purchase_df["ID"].unique())

    def test_occasions_only_drops_non_purchases(self, purchase_df):
        cross_validate_by_customer(_SpyModel, purchase_df, n_splits=3, occasions_only=True)
        assert all(seen == {1} for seen in _SpyModel.fit_incidence)


# ---------------------------------------------------------------------------
# Model evaluate()
# ---------------------------------------------------------------------------

class TestEvaluationFrame:
    def test_to_frame_has_model_and_baseline_columns(self):
        frame = Evaluation(model={"a": 1.0}, baseline={"a": 2.0}).to_frame()
        assert list(frame.columns) == ["model", "baseline"]
        assert frame.loc["a", "baseline"] == 2.0


class TestPropensityEvaluate:
    def test_raises_before_fit(self, purchase_df):
        with pytest.raises(RuntimeError, match="not fitted"):
            PurchasePropensityModel().evaluate(purchase_df)

    def test_beats_baseline_when_price_matters(self, purchase_df):
        train, test = split_by_customer(purchase_df)
        result = PurchasePropensityModel().fit(train).evaluate(test)
        assert result.model["log_loss"] < result.baseline["log_loss"]
        assert result.model["brier"] < result.baseline["brier"]
        assert result.model["roc_auc"] > 0.5

    def test_baseline_uses_training_rate_not_test_rate(self, purchase_df):
        train, test = split_by_customer(purchase_df)
        model = PurchasePropensityModel().fit(train)
        assert model.base_rate_ == pytest.approx(train["Incidence"].mean())

    def test_with_promotion_feature(self, purchase_df):
        train, test = split_by_customer(purchase_df)
        result = PurchasePropensityModel(use_promotion=True).fit(train).evaluate(test)
        assert np.isfinite(result.model["log_loss"])


class TestBrandChoiceEvaluate:
    def test_raises_before_fit(self, purchase_occasions):
        with pytest.raises(RuntimeError, match="not fitted"):
            BrandChoiceModel().evaluate(purchase_occasions)

    def test_beats_baseline_when_price_matters(self, purchase_occasions):
        train, test = split_by_customer(purchase_occasions)
        result = BrandChoiceModel().fit(train).evaluate(test)
        assert result.model["log_loss"] < result.baseline["log_loss"]
        assert result.model["accuracy"] > result.baseline["accuracy"]

    def test_brand_shares_sum_to_one(self, purchase_occasions):
        model = BrandChoiceModel().fit(purchase_occasions)
        assert model.brand_shares_.sum() == pytest.approx(1.0)
        assert len(model.brand_shares_) == len(model.classes_)


class TestQuantityEvaluate:
    def test_raises_before_fit(self, purchase_occasions):
        with pytest.raises(RuntimeError, match="not fitted"):
            PurchaseQuantityModel().evaluate(purchase_occasions)

    def test_beats_baseline_when_price_matters(self, purchase_occasions):
        train, test = split_by_customer(purchase_occasions)
        result = PurchaseQuantityModel().fit(train).evaluate(test)
        assert result.model["r2"] > result.baseline["r2"]
        assert result.model["mae"] < result.baseline["mae"]

    def test_shuffled_target_does_not_beat_baseline(self, purchase_occasions):
        """If quantity is shuffled, no model should look better than the mean."""
        shuffled = purchase_occasions.assign(
            Quantity=np.random.default_rng(1).permutation(purchase_occasions["Quantity"].to_numpy())
        )
        train, test = split_by_customer(shuffled)
        result = PurchaseQuantityModel().fit(train).evaluate(test)
        assert result.model["r2"] < 0.02


# ---------------------------------------------------------------------------
# Segmentation diagnostics
# ---------------------------------------------------------------------------

@pytest.fixture
def three_blobs() -> pd.DataFrame:
    """Three tight, far-apart clusters in 7 dimensions."""
    rng = np.random.default_rng(0)
    centres = np.array([[0] * 7, [10] * 7, [-10, 10, -10, 10, -10, 10, -10]])
    X = np.vstack([c + rng.normal(0, 0.5, (60, 7)) for c in centres])
    return pd.DataFrame(X, columns=list("abcdefg"))


class TestClusterDiagnostics:
    def test_has_one_row_per_k(self, three_blobs):
        diag = SegmentationPipeline().cluster_diagnostics(three_blobs, k_range=range(2, 6))
        assert list(diag.index) == [2, 3, 4, 5]
        assert set(diag.columns) == {"inertia", "silhouette", "davies_bouldin"}

    def test_inertia_decreases_with_k(self, three_blobs):
        diag = SegmentationPipeline().cluster_diagnostics(three_blobs, k_range=range(2, 7))
        assert diag["inertia"].is_monotonic_decreasing

    def test_silhouette_recovers_true_cluster_count(self, three_blobs):
        diag = SegmentationPipeline().cluster_diagnostics(three_blobs, k_range=range(2, 8))
        assert diag["silhouette"].idxmax() == 3
        assert diag["davies_bouldin"].idxmin() == 3

    def test_does_not_fit_the_pipeline(self, three_blobs):
        pipeline = SegmentationPipeline()
        pipeline.cluster_diagnostics(three_blobs, k_range=range(2, 4))
        assert not pipeline._is_fitted


class TestStability:
    def test_returns_one_score_per_bootstrap(self, three_blobs):
        scores = SegmentationPipeline(n_clusters=3).stability(three_blobs, n_boot=5)
        assert scores.shape == (5,)

    def test_well_separated_clusters_are_stable(self, three_blobs):
        scores = SegmentationPipeline(n_clusters=3).stability(three_blobs, n_boot=10)
        assert scores.min() > 0.95

    def test_structureless_data_is_less_stable(self):
        noise = pd.DataFrame(np.random.default_rng(0).normal(size=(300, 7)))
        scores = SegmentationPipeline(n_clusters=4).stability(noise, n_boot=10)
        assert scores.mean() < 0.9

    def test_does_not_fit_the_pipeline(self, three_blobs):
        pipeline = SegmentationPipeline(n_clusters=3)
        pipeline.stability(three_blobs, n_boot=2)
        assert not pipeline._is_fitted
