"""Customer segmentation pipeline: scaling → PCA → KMeans."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_DEFAULT_MODELS_DIR = Path(__file__).parents[2] / "models"


class SegmentationPipeline:
    """End-to-end pipeline for customer segmentation via PCA + KMeans.

    Parameters
    ----------
    n_components:
        Number of PCA components to retain.
    n_clusters:
        Number of KMeans clusters (customer segments).
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_clusters: int = 4,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")

        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "SegmentationPipeline":
        """Fit scaler, PCA, and KMeans on *X*.

        Parameters
        ----------
        X:
            DataFrame of numeric customer features (no target column).

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.kmeans.fit(X_pca)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Scale and project *X* into PCA space.

        Parameters
        ----------
        X:
            DataFrame of numeric customer features.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assign each customer in *X* to a segment label.

        Parameters
        ----------
        X:
            DataFrame of numeric customer features.

        Returns
        -------
        np.ndarray of integer cluster labels.
        """
        self._check_fitted()
        X_pca = self.transform(X)
        return self.kmeans.predict(X_pca)

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the pipeline and return segment labels for *X*."""
        return self.fit(X).predict(X)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def segment_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return per-segment mean feature values and cluster sizes.

        Parameters
        ----------
        X:
            Original (unscaled) feature DataFrame used for fitting.

        Returns
        -------
        pd.DataFrame indexed by segment label.
        """
        labels = self.predict(X)
        summary = (
            X.copy()
            .assign(Segment=labels)
            .groupby("Segment")
            .agg(["mean", "count"])
        )
        return summary

    def explained_variance(self) -> np.ndarray:
        """Return the explained variance ratio for each PCA component."""
        self._check_fitted()
        return self.pca.explained_variance_ratio_

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path | None = None) -> Path:
        """Persist the fitted pipeline to *directory*.

        Saves three files: ``scaler_model.pkl``, ``pca_model.pkl``,
        ``kmeans_pca_model.pkl``.

        Returns
        -------
        Path
            The directory the models were saved to.
        """
        self._check_fitted()
        directory = Path(directory) if directory else _DEFAULT_MODELS_DIR
        directory.mkdir(parents=True, exist_ok=True)

        for name, obj in [
            ("scaler_model.pkl", self.scaler),
            ("pca_model.pkl", self.pca),
            ("kmeans_pca_model.pkl", self.kmeans),
        ]:
            with open(directory / name, "wb") as f:
                pickle.dump(obj, f)

        return directory

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "SegmentationPipeline":
        """Load a previously saved pipeline from *directory*.

        Returns
        -------
        SegmentationPipeline
            A fitted instance ready for ``predict`` / ``transform``.
        """
        directory = Path(directory) if directory else _DEFAULT_MODELS_DIR

        pipeline = cls.__new__(cls)
        with open(directory / "scaler_model.pkl", "rb") as f:
            pipeline.scaler = pickle.load(f)
        with open(directory / "pca_model.pkl", "rb") as f:
            pipeline.pca = pickle.load(f)
        with open(directory / "kmeans_pca_model.pkl", "rb") as f:
            pipeline.kmeans = pickle.load(f)

        pipeline.n_components = pipeline.pca.n_components_
        pipeline.n_clusters = pipeline.kmeans.n_clusters
        pipeline.random_state = None
        pipeline._is_fitted = True
        return pipeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline is not fitted yet. Call .fit() before using this method."
            )

    def __repr__(self) -> str:
        return (
            f"SegmentationPipeline("
            f"n_components={self.n_components}, "
            f"n_clusters={self.n_clusters}, "
            f"fitted={self._is_fitted})"
        )
