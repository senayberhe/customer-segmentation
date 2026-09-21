"""Tests for the customer segmentation pipeline."""

import numpy as np
import pandas as pd
import pytest

from customer_segmentation.data_loader import (
    SEGMENTATION_FEATURES,
    load_segmentation_data,
    load_purchase_data,
)
from customer_segmentation.segmentation import SegmentationPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic customer dataset matching real feature schema."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "Sex":             rng.integers(0, 2, n),
        "Marital status":  rng.integers(0, 2, n),
        "Age":             rng.integers(18, 70, n),
        "Education":       rng.integers(0, 4, n),
        "Income":          rng.integers(20_000, 200_000, n),
        "Occupation":      rng.integers(0, 3, n),
        "Settlement size": rng.integers(0, 3, n),
    })


@pytest.fixture
def fitted_pipeline(sample_df) -> SegmentationPipeline:
    pipeline = SegmentationPipeline(n_components=3, n_clusters=4, random_state=0)
    pipeline.fit(sample_df)
    return pipeline


# ---------------------------------------------------------------------------
# data_loader
# ---------------------------------------------------------------------------

class TestLoadSegmentationData:
    def test_returns_dataframe(self):
        df = load_segmentation_data()
        assert isinstance(df, pd.DataFrame)

    def test_id_column_dropped_by_default(self):
        df = load_segmentation_data()
        assert "ID" not in df.columns

    def test_id_column_kept_when_requested(self):
        df = load_segmentation_data(drop_id=False)
        assert "ID" in df.columns

    def test_has_expected_features(self):
        df = load_segmentation_data()
        assert set(SEGMENTATION_FEATURES).issubset(df.columns)

    def test_no_missing_values(self):
        df = load_segmentation_data()
        assert df.isnull().sum().sum() == 0


class TestLoadPurchaseData:
    def test_returns_dataframe(self):
        df = load_purchase_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_rows(self):
        df = load_purchase_data()
        assert len(df) > 0

    def test_has_id_column(self):
        df = load_purchase_data()
        assert "ID" in df.columns


# ---------------------------------------------------------------------------
# SegmentationPipeline — unfitted guards
# ---------------------------------------------------------------------------

class TestUnfittedPipeline:
    def test_predict_raises_before_fit(self, sample_df):
        pipeline = SegmentationPipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.predict(sample_df)

    def test_transform_raises_before_fit(self, sample_df):
        pipeline = SegmentationPipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.transform(sample_df)

    def test_explained_variance_raises_before_fit(self):
        pipeline = SegmentationPipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.explained_variance()

    def test_save_raises_before_fit(self, tmp_path):
        pipeline = SegmentationPipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.save(tmp_path)


# ---------------------------------------------------------------------------
# SegmentationPipeline — fit / predict
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_fit_returns_self(self, sample_df):
        pipeline = SegmentationPipeline(n_components=2, n_clusters=3)
        result = pipeline.fit(sample_df)
        assert result is pipeline

    def test_is_fitted_after_fit(self, sample_df):
        pipeline = SegmentationPipeline()
        pipeline.fit(sample_df)
        assert pipeline._is_fitted is True

    def test_predict_returns_array(self, fitted_pipeline, sample_df):
        labels = fitted_pipeline.predict(sample_df)
        assert isinstance(labels, np.ndarray)

    def test_predict_length_matches_input(self, fitted_pipeline, sample_df):
        labels = fitted_pipeline.predict(sample_df)
        assert len(labels) == len(sample_df)

    def test_labels_within_cluster_range(self, fitted_pipeline, sample_df):
        labels = fitted_pipeline.predict(sample_df)
        assert set(labels).issubset(set(range(fitted_pipeline.n_clusters)))

    def test_fit_predict_equals_fit_then_predict(self, sample_df):
        p1 = SegmentationPipeline(n_components=2, n_clusters=3, random_state=1)
        p2 = SegmentationPipeline(n_components=2, n_clusters=3, random_state=1)
        labels1 = p1.fit_predict(sample_df)
        p2.fit(sample_df)
        labels2 = p2.predict(sample_df)
        np.testing.assert_array_equal(labels1, labels2)

    def test_all_clusters_used(self, fitted_pipeline, sample_df):
        """With 100 samples and 4 clusters each cluster should be populated."""
        labels = fitted_pipeline.predict(sample_df)
        assert len(set(labels)) == fitted_pipeline.n_clusters


# ---------------------------------------------------------------------------
# SegmentationPipeline — transform / PCA
# ---------------------------------------------------------------------------

class TestTransform:
    def test_transform_shape(self, fitted_pipeline, sample_df):
        X_pca = fitted_pipeline.transform(sample_df)
        assert X_pca.shape == (len(sample_df), fitted_pipeline.n_components)

    def test_explained_variance_length(self, fitted_pipeline):
        ev = fitted_pipeline.explained_variance()
        assert len(ev) == fitted_pipeline.n_components

    def test_explained_variance_sums_to_at_most_one(self, fitted_pipeline):
        assert fitted_pipeline.explained_variance().sum() <= 1.0 + 1e-9

    def test_explained_variance_all_positive(self, fitted_pipeline):
        assert (fitted_pipeline.explained_variance() > 0).all()


# ---------------------------------------------------------------------------
# SegmentationPipeline — segment_summary
# ---------------------------------------------------------------------------

class TestSegmentSummary:
    def test_summary_is_dataframe(self, fitted_pipeline, sample_df):
        summary = fitted_pipeline.segment_summary(sample_df)
        assert isinstance(summary, pd.DataFrame)

    def test_summary_has_correct_number_of_rows(self, fitted_pipeline, sample_df):
        summary = fitted_pipeline.segment_summary(sample_df)
        assert len(summary) == fitted_pipeline.n_clusters

    def test_summary_mean_income_positive(self, fitted_pipeline, sample_df):
        summary = fitted_pipeline.segment_summary(sample_df)
        assert (summary[("Income", "mean")] > 0).all()


# ---------------------------------------------------------------------------
# SegmentationPipeline — persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_three_files(self, fitted_pipeline, tmp_path):
        fitted_pipeline.save(tmp_path)
        expected = {"scaler_model.pkl", "pca_model.pkl", "kmeans_pca_model.pkl"}
        assert {f.name for f in tmp_path.iterdir()} == expected

    def test_load_roundtrip_predictions(self, fitted_pipeline, sample_df, tmp_path):
        original_labels = fitted_pipeline.predict(sample_df)
        fitted_pipeline.save(tmp_path)
        loaded = SegmentationPipeline.load(tmp_path)
        loaded_labels = loaded.predict(sample_df)
        np.testing.assert_array_equal(original_labels, loaded_labels)

    def test_loaded_pipeline_is_fitted(self, fitted_pipeline, tmp_path):
        fitted_pipeline.save(tmp_path)
        loaded = SegmentationPipeline.load(tmp_path)
        assert loaded._is_fitted is True

    def test_loaded_n_clusters_matches(self, fitted_pipeline, tmp_path):
        fitted_pipeline.save(tmp_path)
        loaded = SegmentationPipeline.load(tmp_path)
        assert loaded.n_clusters == fitted_pipeline.n_clusters
