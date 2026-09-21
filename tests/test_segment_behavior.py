"""Tests for per-segment price response and segment-aware purchase models."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from customer_segmentation.data_loader import SEGMENTATION_FEATURES
from customer_segmentation.evaluation import cross_validate_by_customer
from customer_segmentation.purchase_behavior import (
    PRICE_COLUMNS,
    PROMOTION_COLUMNS,
    BrandChoiceModel,
    PurchasePropensityModel,
)
from customer_segmentation.segment_behavior import (
    METRICS,
    assign_segments,
    brand_shares_by_segment,
    segment_price_response,
)
from customer_segmentation.segmentation import SegmentationPipeline

# Price slope of purchase probability per segment: 0 steep ... 3 flat.
SLOPES = {0: -12.0, 1: -6.0, 2: -3.0, 3: -0.5}
FAVOURITE_BRAND = {0: 1, 1: 2, 2: 3, 3: 4}


@pytest.fixture(scope="module")
def shoppers() -> pd.DataFrame:
    """Purchase data for 400 shoppers in 4 segments whose demographics are
    well separated, whose price sensitivity differs, and who each favour a
    different brand."""
    rng = np.random.default_rng(3)
    centres = rng.normal(0, 6, size=(4, len(SEGMENTATION_FEATURES)))
    # Shuffled so the segment isn't a function of ID: GroupKFold deals whole
    # customers to folds in ID order, which would put one segment per fold.
    segment_of = rng.permutation(np.arange(400) % 4)
    frames = []
    for cust in range(400):
        seg = int(segment_of[cust])
        demo = centres[seg] + rng.normal(0, 0.5, len(SEGMENTATION_FEATURES))
        n = 40
        prices = rng.uniform(1.8, 2.2, size=(n, 5))
        mean_price = prices.mean(axis=1)
        p_buy = 1 / (1 + np.exp(-SLOPES[seg] * (mean_price - 2.0) + 1.5 - seg))
        incidence = rng.binomial(1, p_buy)
        brand_p = np.full(5, 0.1)
        brand_p[FAVOURITE_BRAND[seg] - 1] = 0.6
        brand = rng.choice([1, 2, 3, 4, 5], size=n, p=brand_p)
        promotions = rng.integers(0, 2, size=(n, 5))
        chosen_price = prices[np.arange(n), brand - 1]
        quantity = np.clip(np.round(6 - (1 + seg) * 0.5 * chosen_price + rng.normal(0, 0.3, n)), 1, None)

        df = pd.DataFrame(prices, columns=PRICE_COLUMNS)
        for i, col in enumerate(PROMOTION_COLUMNS):
            df[col] = promotions[:, i]
        df["ID"] = cust
        df["Incidence"] = incidence
        df["Brand"] = np.where(incidence == 1, brand, 0)
        df["Quantity"] = np.where(incidence == 1, quantity, 0).astype(int)
        for feat, value in zip(SEGMENTATION_FEATURES, demo):
            df[feat] = value
        df["true_segment"] = seg
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def pipeline(shoppers) -> SegmentationPipeline:
    customers = shoppers.groupby("ID")[SEGMENTATION_FEATURES].first()
    return SegmentationPipeline(n_components=3, n_clusters=4, random_state=0).fit(customers)


@pytest.fixture(scope="module")
def segmented(shoppers, pipeline) -> pd.DataFrame:
    return shoppers.assign(Segment=assign_segments(shoppers, pipeline))


@pytest.fixture(scope="module")
def response(segmented):
    return segment_price_response(segmented, n_boot=40, random_state=0)


def _segment_of(segmented, true_segment: int) -> int:
    """The pipeline's (arbitrary) label for a true segment."""
    return int(segmented.loc[segmented["true_segment"] == true_segment, "Segment"].mode()[0])


# ---------------------------------------------------------------------------
# assign_segments
# ---------------------------------------------------------------------------

class TestAssignSegments:
    def test_one_label_per_row_and_named_segment(self, shoppers, segmented):
        assert len(segmented["Segment"]) == len(shoppers)
        assert segmented["Segment"].name == "Segment"

    def test_constant_within_a_shopper(self, segmented):
        assert (segmented.groupby("ID")["Segment"].nunique() == 1).all()

    def test_recovers_the_true_segments(self, segmented):
        ari = adjusted_rand_score(segmented["true_segment"], segmented["Segment"])
        assert ari > 0.95


# ---------------------------------------------------------------------------
# segment_price_response
# ---------------------------------------------------------------------------

class TestSegmentPriceResponse:
    def test_one_row_per_segment(self, response):
        assert len(response.summary) == 4
        assert response.summary["customers"].sum() == 400

    def test_reports_estimate_and_ci_for_each_metric(self, response):
        for metric in METRICS:
            assert {metric, f"{metric}_ci_low", f"{metric}_ci_high"} <= set(response.summary.columns)

    def test_ci_brackets_the_point_estimate(self, response):
        s = response.summary
        for metric in METRICS:
            assert (s[f"{metric}_ci_low"] <= s[metric]).all()
            assert (s[metric] <= s[f"{metric}_ci_high"]).all()

    def test_draws_shape(self, response):
        for metric in METRICS:
            assert response.draws[metric].shape == (40, 4)

    def test_all_elasticities_negative_when_price_hurts(self, response):
        for metric in METRICS:
            assert (response.summary[metric] < 0).all()

    def test_steepest_and_flattest_segments_are_recovered(self, segmented, response):
        steep, flat = _segment_of(segmented, 0), _segment_of(segmented, 3)
        e = response.summary["propensity_elasticity"]
        assert e[steep] < e[flat]

    def test_compare_detects_a_real_difference(self, segmented, response):
        steep, flat = _segment_of(segmented, 0), _segment_of(segmented, 3)
        diff, lo, hi = response.compare("propensity_elasticity", steep, flat)
        assert diff < 0 and hi < 0  # CI excludes zero

    def test_compare_segment_with_itself_is_exactly_zero(self, response):
        assert response.compare("quantity_elasticity", 1, 1) == (0.0, 0.0, 0.0)

    def test_reproducible_with_same_seed(self, segmented):
        a = segment_price_response(segmented, n_boot=10, random_state=5).summary
        b = segment_price_response(segmented, n_boot=10, random_state=5).summary
        pd.testing.assert_frame_equal(a, b)

    def test_ci_has_positive_width(self, response):
        s = response.summary
        for metric in METRICS:
            assert (s[f"{metric}_ci_high"] > s[f"{metric}_ci_low"]).all()


class TestBrandSharesBySegment:
    def test_rows_sum_to_one(self, segmented):
        shares = brand_shares_by_segment(segmented)
        np.testing.assert_allclose(shares.sum(axis=1), 1.0)

    def test_each_segment_favours_its_designed_brand(self, segmented):
        shares = brand_shares_by_segment(segmented)
        for true_seg, brand in FAVOURITE_BRAND.items():
            assert shares.loc[_segment_of(segmented, true_seg)].idxmax() == brand


# ---------------------------------------------------------------------------
# use_segment models
# ---------------------------------------------------------------------------

class TestSegmentAwareBrandChoice:
    def test_segment_lifts_held_out_accuracy_and_log_loss(self, segmented):
        base = cross_validate_by_customer(BrandChoiceModel, segmented, n_splits=4, occasions_only=True)
        with_seg = cross_validate_by_customer(
            lambda: BrandChoiceModel(use_segment=True), segmented, n_splits=4, occasions_only=True
        )
        assert with_seg["accuracy"].mean() > base["accuracy"].mean() + 0.1
        assert with_seg["log_loss"].mean() < base["log_loss"].mean()

    def test_predict_proba_rows_sum_to_one(self, segmented):
        occ = segmented[segmented["Incidence"] == 1]
        model = BrandChoiceModel(use_segment=True).fit(occ)
        np.testing.assert_allclose(model.predict_proba(occ.head(20)).sum(axis=1), 1.0)

    def test_needs_segment_column_to_fit(self, segmented):
        occ = segmented[segmented["Incidence"] == 1].drop(columns="Segment")
        with pytest.raises(KeyError):
            BrandChoiceModel(use_segment=True).fit(occ)

    def test_own_price_elasticity_requires_segment(self, segmented):
        model = BrandChoiceModel(use_segment=True).fit(segmented[segmented["Incidence"] == 1])
        with pytest.raises(ValueError, match="segment"):
            model.own_price_elasticity(1, [1.9, 2.0])

    def test_own_price_elasticity_with_segment(self, segmented):
        model = BrandChoiceModel(use_segment=True).fit(segmented[segmented["Incidence"] == 1])
        assert model.own_price_elasticity(1, [1.9, 2.0], segment=0).shape == (2,)

    def test_bootstrap_not_supported_with_segment(self, segmented):
        occ = segmented[segmented["Incidence"] == 1]
        model = BrandChoiceModel(use_segment=True).fit(occ)
        with pytest.raises(NotImplementedError):
            model.bootstrap_own_price_significance(1, occ, n_boot=2)

    def test_default_model_unchanged(self, segmented):
        occ = segmented[segmented["Incidence"] == 1].drop(columns="Segment")
        assert BrandChoiceModel().fit(occ).model.coef_.shape[1] == 5


class TestSegmentAwarePropensity:
    def test_segment_improves_held_out_log_loss(self, segmented):
        base = cross_validate_by_customer(PurchasePropensityModel, segmented, n_splits=4)
        with_seg = cross_validate_by_customer(
            lambda: PurchasePropensityModel(use_segment=True), segmented, n_splits=4
        )
        assert with_seg["log_loss"].mean() < base["log_loss"].mean()

    def test_predict_proba_requires_segment(self, segmented):
        model = PurchasePropensityModel(use_segment=True).fit(segmented)
        with pytest.raises(ValueError, match="segment"):
            model.predict_proba([2.0])

    def test_segment_changes_prediction(self, segmented):
        model = PurchasePropensityModel(use_segment=True).fit(segmented)
        a = model.predict_proba([2.0], segment=0)
        b = model.predict_proba([2.0], segment=3)
        assert a != pytest.approx(b)

    def test_price_elasticity_with_segment(self, segmented):
        model = PurchasePropensityModel(use_segment=True).fit(segmented)
        assert model.price_elasticity([1.9, 2.1], segment=1).shape == (2,)

    def test_save_load_roundtrip_keeps_segments(self, segmented, tmp_path):
        model = PurchasePropensityModel(use_segment=True).fit(segmented)
        path = model.save(tmp_path)
        loaded = PurchasePropensityModel.load(path)
        np.testing.assert_allclose(
            loaded.predict_proba([2.0], segment=2), model.predict_proba([2.0], segment=2)
        )
