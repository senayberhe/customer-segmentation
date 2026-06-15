"""Entry point for the customer segmentation pipeline."""

from src.customer_segmentation import (
    SegmentationPipeline,
    load_segmentation_data,
)
from src.customer_segmentation.data_loader import SEGMENTATION_FEATURES


def main() -> None:
    print("Loading data...")
    df = load_segmentation_data()

    X = df[SEGMENTATION_FEATURES]

    print("Fitting segmentation pipeline (PCA + KMeans)...")
    pipeline = SegmentationPipeline(n_components=3, n_clusters=4)
    labels = pipeline.fit_predict(X)

    print(f"\nAssigned {pipeline.n_clusters} segments to {len(labels)} customers.")
    print(f"Explained variance by PCA components: {pipeline.explained_variance().round(3)}")

    print("\nSegment summary (feature means):")
    summary = pipeline.segment_summary(X)
    print(summary["mean"].to_string())

    print("\nSaving models...")
    saved_to = pipeline.save()
    print(f"Models saved to: {saved_to}")


if __name__ == "__main__":
    main()

