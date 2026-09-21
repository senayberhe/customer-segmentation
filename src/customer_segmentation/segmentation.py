"""Customer segmentation pipeline: scaling → PCA → KMeans."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

_DEFAULT_MODELS_DIR = Path(__file__).parents[2] / "models"

# Names for the segments the default pipeline (k=4, random_state=42) finds,
# read off each segment's demographic profile. KMeans label numbers are
# arbitrary, so these only apply to the saved pipeline / that configuration.
SEGMENT_NAMES = {
    0: "standard",
    1: "fewer-opportunities",
    2: "career-focused",
    3: "well-off",
}


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

    def cluster_diagnostics(
        self,
        X: pd.DataFrame,
        k_range: range | list[int] = range(2, 11),
    ) -> pd.DataFrame:
        """Score candidate cluster counts on *X*, to justify ``n_clusters``.

        Reuses this pipeline's scaler/PCA settings and refits KMeans for each
        k. Higher silhouette and lower Davies-Bouldin mean better-separated
        clusters; inertia always falls with k, so look for an elbow instead
        of a minimum.

        Returns
        -------
        pd.DataFrame indexed by k with ``inertia``, ``silhouette`` and
        ``davies_bouldin`` columns.
        """
        Z = self._pca_space(X)
        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto").fit(Z)
            rows.append({
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(Z, km.labels_),
                "davies_bouldin": davies_bouldin_score(Z, km.labels_),
            })
        return pd.DataFrame(rows).set_index("k")

    def stability(self, X: pd.DataFrame, n_boot: int = 30, random_state: int = 0) -> np.ndarray:
        """Adjusted Rand index between the full-data segments and segments
        refit on bootstrap resamples of *X*.

        1.0 means the customers always land in the same groups; values near
        0 mean the segments are an artifact of the particular sample.
        """
        reference_pipeline = SegmentationPipeline(
            self.n_components, self.n_clusters, self.random_state
        ).fit(X)
        reference = reference_pipeline.predict(X)
        rng = np.random.default_rng(random_state)
        scores = []
        for i in range(n_boot):
            sample = X.iloc[rng.integers(0, len(X), len(X))]
            boot = SegmentationPipeline(self.n_components, self.n_clusters, random_state=i)
            scores.append(adjusted_rand_score(reference, boot.fit(sample).predict(X)))
        return np.array(scores)

    def _pca_space(self, X: pd.DataFrame) -> np.ndarray:
        """Fit a fresh scaler and PCA on *X* and return the projection.

        Independent of this pipeline's fitted state, so diagnostics can run
        on any data without disturbing a fitted model.
        """
        scaled = StandardScaler().fit_transform(X)
        return PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(scaled)

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
