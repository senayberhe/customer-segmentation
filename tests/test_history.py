"""Tests for purchase-history features and the history-aware models."""

import numpy as np
import pandas as pd
import pytest

from customer_segmentation.evaluation import cross_validate_by_customer
from customer_segmentation.history import HISTORY_COLUMNS, add_purchase_history
from customer_segmentation.purchase_behavior import (
    PRICE_COLUMNS,
    PROMOTION_COLUMNS,
    BrandChoiceModel,
    PurchasePropensityModel,
)


def trips(rows) -> pd.DataFrame:
    """Build trip data from (ID, Day, Incidence, Brand) tuples."""
    return pd.DataFrame(rows, columns=["ID", "Day", "Incidence", "Brand"])


# ---------------------------------------------------------------------------
# add_purchase_history
# ---------------------------------------------------------------------------

class TestAddPurchaseHistory:
    @pytest.fixture
    def one_shopper(self):
        return trips([
            (1, 1, 0, 0),
            (1, 5, 0, 0),
            (1, 10, 1, 3),   # first purchase
            (1, 12, 0, 0),
            (1, 20, 1, 4),   # second purchase
            (1, 21, 0, 0),
        ])

    def test_days_since_purchase(self, one_shopper):
        out = add_purchase_history(one_shopper)
        # Before any purchase: days since the first observed trip (day 1).
        assert out["Days_Since_Purchase"].tolist() == [0, 4, 9, 2, 10, 1]

    def test_no_prior_purchase_flag(self, one_shopper):
        out = add_purchase_history(one_shopper)
        # The first purchase trip itself still has no *earlier* purchase.
        assert out["No_Prior_Purchase"].tolist() == [1, 1, 1, 0, 0, 0]

    def test_last_brand_is_the_previous_purchase(self, one_shopper):
        out = add_purchase_history(one_shopper)
        assert out["Last_Brand"].tolist() == [0, 0, 0, 3, 3, 4]

    def test_adds_exactly_the_history_columns(self, one_shopper):
        out = add_purchase_history(one_shopper)
        assert list(out.columns) == list(one_shopper.columns) + HISTORY_COLUMNS

    def test_does_not_mutate_input(self, one_shopper):
        before = one_shopper.copy()
        add_purchase_history(one_shopper)
        pd.testing.assert_frame_equal(one_shopper, before)

    def test_preserves_row_order_and_index_when_input_is_shuffled(self, one_shopper):
        shuffled = one_shopper.sample(frac=1, random_state=0)
        out = add_purchase_history(shuffled)
        assert out.index.tolist() == shuffled.index.tolist()
        expected = add_purchase_history(one_shopper).loc[out.index]
        pd.testing.assert_frame_equal(out, expected)

    def test_shoppers_are_independent(self, one_shopper):
        other = trips([(2, 3, 1, 5), (2, 9, 0, 0)])
        together = add_purchase_history(pd.concat([one_shopper, other], ignore_index=True))
        alone = add_purchase_history(one_shopper)
        pd.testing.assert_frame_equal(
            together.iloc[: len(one_shopper)].reset_index(drop=True), alone.reset_index(drop=True)
        )
        assert together.iloc[-1]["Last_Brand"] == 5

    def test_a_trip_never_sees_itself(self, one_shopper):
        """Flipping a trip's own purchase/brand must not change its features."""
        changed = one_shopper.copy()
        changed.loc[4, ["Incidence", "Brand"]] = [0, 0]
        a, b = add_purchase_history(one_shopper), add_purchase_history(changed)
        pd.testing.assert_frame_equal(a.loc[[4], HISTORY_COLUMNS], b.loc[[4], HISTORY_COLUMNS])

    def test_later_trips_never_leak_backwards(self, one_shopper):
        """Changing what happens later must not change earlier trips' features."""
        changed = one_shopper.copy()
        changed.loc[4, ["Incidence", "Brand"]] = [0, 0]
        changed.loc[5, ["Incidence", "Brand"]] = [1, 2]
        a, b = add_purchase_history(one_shopper), add_purchase_history(changed)
        pd.testing.assert_frame_equal(a.loc[:4, HISTORY_COLUMNS], b.loc[:4, HISTORY_COLUMNS])

    def test_shopper_who_never_buys(self):
        out = add_purchase_history(trips([(1, 1, 0, 0), (1, 8, 0, 0)]))
        assert out["No_Prior_Purchase"].tolist() == [1, 1]
        assert out["Last_Brand"].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# History-aware models
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def habitual() -> pd.DataFrame:
    """Shoppers who buy often shortly after a purchase and rarely after a
    long gap (recency), and who repeat their previous brand 80% of the
    time; price has no effect at all."""
    rng = np.random.default_rng(11)
    frames = []
    for cust in range(150):
        n = 60
        day = np.cumsum(rng.integers(1, 6, n))
        incidence = np.zeros(n, dtype=int)
        brand = np.zeros(n, dtype=int)
        last_buy_day, last_brand = -100, 0
        for t in range(n):
            gap = day[t] - last_buy_day
            if rng.random() < (0.5 if gap < 8 else 0.05):
                incidence[t] = 1
                if last_brand and rng.random() < 0.8:
                    brand[t] = last_brand
                else:
                    brand[t] = rng.integers(1, 6)
                last_buy_day, last_brand = day[t], brand[t]
        df = pd.DataFrame(rng.uniform(1.8, 2.2, (n, 5)), columns=PRICE_COLUMNS)
        for col in PROMOTION_COLUMNS:
            df[col] = rng.integers(0, 2, n)
        df["ID"], df["Day"], df["Incidence"], df["Brand"] = cust, day, incidence, brand
        df["Quantity"] = np.where(incidence == 1, 2, 0)
        frames.append(df)
    return add_purchase_history(pd.concat(frames, ignore_index=True))


@pytest.fixture(scope="module")
def habitual_occasions(habitual) -> pd.DataFrame:
    return habitual[habitual["Incidence"] == 1]


class TestHistoryAwarePropensity:
    def test_history_lifts_held_out_auc_and_log_loss(self, habitual):
        base = cross_validate_by_customer(PurchasePropensityModel, habitual, n_splits=4)
        hist = cross_validate_by_customer(
            lambda: PurchasePropensityModel(use_history=True), habitual, n_splits=4
        )
        assert hist["roc_auc"].mean() > base["roc_auc"].mean() + 0.1
        assert hist["log_loss"].mean() < base["log_loss"].mean()

    def test_learns_the_direction_of_the_recency_effect(self, habitual):
        model = PurchasePropensityModel(use_history=True).fit(habitual)
        recent = model.predict_proba([2.0], days_since=2)
        lapsed = model.predict_proba([2.0], days_since=60)
        assert recent[0] > lapsed[0]

    def test_predict_proba_requires_days_since(self, habitual):
        model = PurchasePropensityModel(use_history=True).fit(habitual)
        with pytest.raises(ValueError, match="days_since"):
            model.predict_proba([2.0])

    def test_price_elasticity_with_history(self, habitual):
        model = PurchasePropensityModel(use_history=True).fit(habitual)
        assert model.price_elasticity([1.9, 2.1], days_since=10).shape == (2,)

    def test_needs_history_columns_to_fit(self, habitual):
        with pytest.raises(KeyError):
            PurchasePropensityModel(use_history=True).fit(habitual.drop(columns=HISTORY_COLUMNS))

    def test_combines_with_segment(self, habitual):
        df = habitual.assign(Segment=habitual["ID"] % 3)
        model = PurchasePropensityModel(use_segment=True, use_history=True).fit(df)
        assert model.predict_proba([2.0], segment=1, days_since=5).shape == (1,)

    def test_save_load_roundtrip(self, habitual, tmp_path):
        model = PurchasePropensityModel(use_history=True).fit(habitual)
        loaded = PurchasePropensityModel.load(model.save(tmp_path))
        np.testing.assert_allclose(
            loaded.predict_proba([2.0], days_since=7), model.predict_proba([2.0], days_since=7)
        )


class TestHistoryAwareBrandChoice:
    def test_history_lifts_held_out_accuracy_and_log_loss(self, habitual):
        base = cross_validate_by_customer(BrandChoiceModel, habitual, n_splits=4, occasions_only=True)
        hist = cross_validate_by_customer(
            lambda: BrandChoiceModel(use_history=True), habitual, n_splits=4, occasions_only=True
        )
        assert hist["accuracy"].mean() > base["accuracy"].mean() + 0.2
        assert hist["log_loss"].mean() < base["log_loss"].mean()

    def test_predicts_repeat_of_last_brand(self, habitual_occasions):
        model = BrandChoiceModel(use_history=True).fit(habitual_occasions)
        repeaters = habitual_occasions[habitual_occasions["Last_Brand"] > 0]
        agreement = (model.predict(repeaters) == repeaters["Last_Brand"]).mean()
        assert agreement > 0.9

    def test_predict_proba_rows_sum_to_one(self, habitual_occasions):
        model = BrandChoiceModel(use_history=True).fit(habitual_occasions)
        proba = model.predict_proba(habitual_occasions.head(25))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_own_price_elasticity_requires_last_brand(self, habitual_occasions):
        model = BrandChoiceModel(use_history=True).fit(habitual_occasions)
        with pytest.raises(ValueError, match="last_brand"):
            model.own_price_elasticity(1, [1.9, 2.0])

    def test_own_price_elasticity_with_last_brand(self, habitual_occasions):
        model = BrandChoiceModel(use_history=True).fit(habitual_occasions)
        assert model.own_price_elasticity(1, [1.9, 2.0], last_brand=0).shape == (2,)

    def test_bootstrap_not_supported_with_history(self, habitual_occasions):
        model = BrandChoiceModel(use_history=True).fit(habitual_occasions)
        with pytest.raises(NotImplementedError):
            model.bootstrap_own_price_significance(1, habitual_occasions, n_boot=2)

    def test_combines_with_segment(self, habitual_occasions):
        df = habitual_occasions.assign(Segment=habitual_occasions["ID"] % 3)
        model = BrandChoiceModel(use_segment=True, use_history=True).fit(df)
        assert model.predict(df.head(5)).shape == (5,)
