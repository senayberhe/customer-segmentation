"""Entry point for the customer segmentation and purchase-behavior pipelines."""

import numpy as np

from src.customer_segmentation import (
    BrandChoiceModel,
    PurchasePropensityModel,
    PurchaseQuantityModel,
    SegmentationPipeline,
    load_purchase_data,
    load_segmentation_data,
)
from src.customer_segmentation.data_loader import SEGMENTATION_FEATURES
from src.customer_segmentation.evaluation import cross_validate_by_customer


def run_segmentation() -> None:
    print("Loading data...")
    df = load_segmentation_data()

    X = df[SEGMENTATION_FEATURES]

    print("Fitting segmentation pipeline (PCA + KMeans)...")
    pipeline = SegmentationPipeline(n_components=3, n_clusters=4)
    labels = pipeline.fit_predict(X)

    print(f"\nAssigned {pipeline.n_clusters} segments to {len(labels)} customers.")
    print(f"Explained variance by PCA components: {pipeline.explained_variance().round(3)}")

    print("\nCluster-count diagnostics (why k=4?):")
    print(pipeline.cluster_diagnostics(X).round(3).to_string())
    stability = pipeline.stability(X, n_boot=30)
    print(f"Bootstrap stability at k=4 (adjusted Rand index): mean {stability.mean():.2f}, min {stability.min():.2f}")

    print("\nSegment summary (feature means):")
    summary = pipeline.segment_summary(X)
    print(summary.xs("mean", axis=1, level=1).to_string())

    print("\nSaving models...")
    saved_to = pipeline.save()
    print(f"Models saved to: {saved_to}")


def run_purchase_behavior() -> None:
    print("\nLoading purchase-occasion data...")
    df = load_purchase_data()
    occasions = df[df["Incidence"] == 1]

    print("Fitting purchase propensity model (P(purchase) vs. price)...")
    propensity = PurchasePropensityModel().fit(df)
    sample_prices = np.array([1.0, 1.5, 2.0, 2.5])
    proba = propensity.predict_proba(sample_prices)
    elasticity = propensity.price_elasticity(sample_prices)
    print("  Price   P(purchase)   Elasticity")
    for p, pr, el in zip(sample_prices, proba, elasticity):
        print(f"  {p:>5.2f}   {pr:>11.3f}   {el:>10.3f}")

    print("\nFitting brand choice model (which brand is chosen)...")
    brand_choice = BrandChoiceModel().fit(occasions)
    print(f"  Brands: {list(brand_choice.classes_)}")
    print(f"  Mean historical prices: {brand_choice.mean_prices_}")

    print("\n  Checking whether each brand's own-price coefficient is statistically")
    print("  significant (bootstrapped 95% CI) rather than trusting point estimates...")
    for brand in brand_choice.classes_:
        est = brand_choice.bootstrap_own_price_significance(
            int(brand), occasions, n_boot=40, random_state=42
        )
        flag = "significant" if est.is_significant else "NOT significant — don't trust the sign"
        print(f"    Brand {brand}: coef≈{est.mean:>6.3f}  CI=[{est.ci_low:>6.3f}, {est.ci_high:>6.3f}]  {flag}")

    print("\nFitting purchase quantity model (units bought vs. price)...")
    quantity = PurchaseQuantityModel().fit(occasions)
    predicted_qty = quantity.predict(sample_prices)
    print("  Price   Predicted quantity")
    for p, q in zip(sample_prices, predicted_qty):
        print(f"  {p:>5.2f}   {q:>17.2f}")

    print("\nHeld-out evaluation (5-fold CV, folds split by customer, mean over folds):")
    for name, factory, occasions_only in [
        ("Propensity", PurchasePropensityModel, False),
        ("Brand choice", BrandChoiceModel, True),
        ("Quantity", PurchaseQuantityModel, True),
    ]:
        cv = cross_validate_by_customer(factory, df, occasions_only=occasions_only).mean()
        metrics = [m for m in cv.index if not m.startswith("baseline_")]
        line = "  ".join(f"{m}={cv[m]:.3f} (baseline {cv[f'baseline_{m}']:.3f})" for m in metrics)
        print(f"  {name:<13} {line}")

    print("\nSaving models...")
    for model in (propensity, brand_choice, quantity):
        saved_to = model.save()
        print(f"  {model.__class__.__name__} saved to: {saved_to}")


def main() -> None:
    run_segmentation()
    run_purchase_behavior()


if __name__ == "__main__":
    main()

