"""Purchase-behavior models: how customers react to price and promotions.

Packages the price-elasticity analysis from
notebooks/customer_analytics_predictive_analysis.ipynb into tested,
reusable models: purchase probability, brand choice, and purchase
quantity, each as a function of price (and promotion).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

_DEFAULT_MODELS_DIR = Path(__file__).parents[2] / "models"

PRICE_COLUMNS = [f"Price_{i}" for i in range(1, 6)]
PROMOTION_COLUMNS = [f"Promotion_{i}" for i in range(1, 6)]


class CoefficientEstimate(NamedTuple):
    """A bootstrapped own-price coefficient estimate for one brand."""

    brand: int
    mean: float
    ci_low: float
    ci_high: float

    @property
    def is_significant(self) -> bool:
        """False if the 95% CI crosses zero — the sign/magnitude can't be
        trusted, regardless of what the point estimate looks like."""
        return not (self.ci_low < 0 < self.ci_high)


class PurchasePropensityModel:
    """Predicts P(purchase) from average price, and optionally promotion.

    Parameters
    ----------
    use_promotion:
        If True, also fits on average promotion activity across the 5
        brands (the "Purchase Probability with Promotion Feature"
        analysis in the notebook).
    """

    def __init__(self, use_promotion: bool = False) -> None:
        self.use_promotion = use_promotion
        self.model = LogisticRegression(solver="sag", max_iter=5000)
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "PurchasePropensityModel":
        """Fit on purchase-occasion data (one row per shopping trip)."""
        X = pd.DataFrame({"Mean_Price": df[PRICE_COLUMNS].mean(axis=1)})
        if self.use_promotion:
            X["Mean_Promotion"] = df[PROMOTION_COLUMNS].mean(axis=1)
        self.model.fit(X, df["Incidence"])
        self._is_fitted = True
        return self

    def predict_proba(self, price, promotion: float = 1.0) -> np.ndarray:
        """P(purchase) at the given average price point(s)."""
        self._check_fitted()
        price = np.atleast_1d(np.asarray(price, dtype=float))
        X = pd.DataFrame({"Mean_Price": price})
        if self.use_promotion:
            X["Mean_Promotion"] = promotion
        return self.model.predict_proba(X)[:, 1]

    def price_elasticity(self, price_range, promotion: float = 1.0) -> np.ndarray:
        """Own-price elasticity of purchase probability at each price point.

        elasticity(p) = beta_price * p * (1 - P(purchase | p))
        """
        self._check_fitted()
        price_range = np.atleast_1d(np.asarray(price_range, dtype=float))
        proba = self.predict_proba(price_range, promotion=promotion)
        beta_price = self.model.coef_[0][0]
        return beta_price * price_range * (1 - proba)

    def save(self, directory: str | Path | None = None) -> Path:
        self._check_fitted()
        directory = Path(directory) if directory else _DEFAULT_MODELS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "purchase_propensity_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PurchasePropensityModel":
        path = Path(path) if path else _DEFAULT_MODELS_DIR / "purchase_propensity_model.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() before using this method.")

    def __repr__(self) -> str:
        return f"PurchasePropensityModel(use_promotion={self.use_promotion}, fitted={self._is_fitted})"


class BrandChoiceModel:
    """Multinomial logistic regression predicting which brand a customer
    chooses on a purchase occasion, from the five brands' prices.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(solver="sag", max_iter=5000)
        self._is_fitted = False
        self.classes_: np.ndarray | None = None
        self.mean_prices_: dict[int, float] = {}

    def fit(self, purchase_occasions: pd.DataFrame) -> "BrandChoiceModel":
        """Fit on rows where a purchase occurred (``Incidence == 1``)."""
        X = purchase_occasions[PRICE_COLUMNS]
        self.model.fit(X, purchase_occasions["Brand"])
        self.classes_ = self.model.classes_
        self.mean_prices_ = {i: X[f"Price_{i}"].mean() for i in range(1, 6)}
        self._is_fitted = True
        return self

    def predict_proba(self, prices: pd.DataFrame) -> np.ndarray:
        """P(brand) for each row of `prices` (must have the 5 price columns)."""
        self._check_fitted()
        return self.model.predict_proba(prices[PRICE_COLUMNS])

    def predict(self, prices: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(prices[PRICE_COLUMNS])

    def own_price_elasticity(
        self,
        brand: int,
        price_range,
        other_prices: dict[int, float] | None = None,
    ) -> np.ndarray:
        """Own-price elasticity of `brand`'s choice probability as its price
        varies over `price_range`, holding every other brand's price fixed
        (default: at each brand's mean price in the training data).
        """
        self._check_fitted()
        price_range = np.atleast_1d(np.asarray(price_range, dtype=float))
        fixed_prices = {**self.mean_prices_, **(other_prices or {})}

        X = pd.DataFrame({
            f"Price_{i}": np.full(price_range.shape, fixed_prices[i])
            for i in range(1, 6)
        })
        X[f"Price_{brand}"] = price_range

        proba = self.model.predict_proba(X[PRICE_COLUMNS])
        class_idx = list(self.classes_).index(brand)
        own_proba = proba[:, class_idx]
        beta_own = self.model.coef_[class_idx, brand - 1]
        return beta_own * price_range * (1 - own_proba)

    def bootstrap_own_price_significance(
        self,
        brand: int,
        purchase_occasions: pd.DataFrame,
        n_boot: int = 300,
        ci: float = 0.95,
        random_state: int | None = None,
    ) -> CoefficientEstimate:
        """Bootstrap a confidence interval for `brand`'s own-price coefficient.

        A single point estimate from `own_price_elasticity` can look like a
        real effect even when the data can't actually support it — this is
        especially likely for a low-share brand with correlated competitor
        prices. Resamples `purchase_occasions` with replacement, refits on
        each resample, and reports whether the resulting CI excludes zero.
        If it doesn't, the coefficient's sign shouldn't be trusted.

        `purchase_occasions` should be the same data (or a fresh sample of
        the same population) used for `fit`.
        """
        self._check_fitted()
        rng = np.random.default_rng(random_state)
        n = len(purchase_occasions)

        coefs = []
        for _ in range(n_boot):
            sample = purchase_occasions.iloc[rng.integers(0, n, n)]
            if sample["Brand"].nunique() < len(self.classes_):
                continue
            model = LogisticRegression(solver="lbfgs", max_iter=5000)
            model.fit(sample[PRICE_COLUMNS], sample["Brand"])
            if brand not in model.classes_:
                continue
            resample_idx = list(model.classes_).index(brand)
            coefs.append(model.coef_[resample_idx, brand - 1])

        coefs = np.array(coefs)
        alpha = (1 - ci) / 2 * 100
        lo, hi = np.percentile(coefs, [alpha, 100 - alpha])
        return CoefficientEstimate(brand=brand, mean=coefs.mean(), ci_low=lo, ci_high=hi)

    def save(self, directory: str | Path | None = None) -> Path:
        self._check_fitted()
        directory = Path(directory) if directory else _DEFAULT_MODELS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "brand_choice_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BrandChoiceModel":
        path = Path(path) if path else _DEFAULT_MODELS_DIR / "brand_choice_model.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() before using this method.")

    def __repr__(self) -> str:
        return f"BrandChoiceModel(fitted={self._is_fitted})"


class PurchaseQuantityModel:
    """Linear regression predicting units purchased, from the price paid
    for the chosen brand and whether that brand was on promotion.
    """

    def __init__(self) -> None:
        self.model = LinearRegression()
        self._is_fitted = False

    def fit(self, purchase_occasions: pd.DataFrame) -> "PurchaseQuantityModel":
        """Fit on rows where a purchase occurred (``Incidence == 1``)."""
        brand_dummies = pd.get_dummies(purchase_occasions["Brand"], prefix="Brand")
        price_incidence = sum(
            brand_dummies.get(f"Brand_{i}", 0) * purchase_occasions[f"Price_{i}"]
            for i in range(1, 6)
        )
        promotion_incidence = sum(
            brand_dummies.get(f"Brand_{i}", 0) * purchase_occasions[f"Promotion_{i}"]
            for i in range(1, 6)
        )
        X = pd.DataFrame({
            "Price_Incidence": price_incidence,
            "Promotion_Incidence": promotion_incidence,
        })
        self.model.fit(X, purchase_occasions["Quantity"])
        self._is_fitted = True
        return self

    def predict(self, price_paid, promotion: float = 0.0) -> np.ndarray:
        """Predicted quantity at the given price(s) paid for the chosen brand."""
        self._check_fitted()
        price_paid = np.atleast_1d(np.asarray(price_paid, dtype=float))
        X = pd.DataFrame({
            "Price_Incidence": price_paid,
            "Promotion_Incidence": promotion,
        })
        return self.model.predict(X)

    def price_elasticity(self, price_range, promotion: float = 0.0) -> np.ndarray:
        """Price elasticity of purchase quantity at each price point.

        elasticity(p) = beta_price * p / predicted_quantity(p)
        """
        self._check_fitted()
        price_range = np.atleast_1d(np.asarray(price_range, dtype=float))
        predicted_qty = self.predict(price_range, promotion=promotion)
        beta_price = self.model.coef_[0]
        return beta_price * price_range / predicted_qty

    def save(self, directory: str | Path | None = None) -> Path:
        self._check_fitted()
        directory = Path(directory) if directory else _DEFAULT_MODELS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "purchase_quantity_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PurchaseQuantityModel":
        path = Path(path) if path else _DEFAULT_MODELS_DIR / "purchase_quantity_model.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() before using this method.")

    def __repr__(self) -> str:
        return f"PurchaseQuantityModel(fitted={self._is_fitted})"
