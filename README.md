# Customer Segmentation — Data Science Portfolio Project

**By Senay Berhe** · [tsionberhe@gmail.com](mailto:tsionberhe@gmail.com)

> An end-to-end machine learning project demonstrating skills in exploratory data analysis, unsupervised clustering, supervised predictive modeling, dimensionality reduction, software engineering best practices, and production-ready code design.

---

## Project Overview

This project builds a complete customer analytics system from raw retail data, combining two complementary ML pipelines:

1. **Segmentation** — unsupervised PCA + KMeans on demographic data, discovering four distinct customer segments.
2. **Purchase behavior prediction** — supervised models that predict *how customers react* to price and promotions: the probability they purchase at all, which of five brands they choose, and how many units they buy.

Together these answer both "who are my customers?" and "how will they respond if I change price?" — insight a business could directly use to personalise marketing, run promotions, and set prices.

The project goes beyond a notebook prototype: it is structured as a proper Python package with a clean API, model persistence, and a full test suite.

---

## Skills Demonstrated

| Area | What I did |
|---|---|
| **Unsupervised ML** | Applied PCA for dimensionality reduction and KMeans clustering to discover meaningful customer groups |
| **Supervised predictive modeling** | Logistic regression for purchase propensity and multinomial brand choice; linear regression for purchase quantity — each with a price-elasticity method |
| **Feature engineering** | Selected and scaled 7 demographic features; reasoned about ordinal vs continuous encoding; engineered price/promotion incidence features for the purchase models |
| **Statistical thinking** | Used explained variance analysis to justify retaining 3 PCA components (80.8% variance captured); when one brand's price coefficient came back counter-intuitively positive, bootstrapped a confidence interval instead of trusting the point estimate — confirmed it wasn't statistically distinguishable from zero and diagnosed why (small sample, low own-price variance, correlated competitor prices) |
| **Software engineering** | Structured code as a reusable Python package (`src/` layout) with clear separation of concerns across two pipelines |
| **API design** | Designed `SegmentationPipeline`, `PurchasePropensityModel`, `BrandChoiceModel`, and `PurchaseQuantityModel` classes with a consistent sklearn-style interface (`fit`, `predict`/`predict_proba`) |
| **Model persistence** | Implemented `save()` / `load()` on every model with proper serialisation so models can be deployed without retraining |
| **Model evaluation** | Scored every supervised model on held-out customers (5-fold CV grouped by customer ID, so no shopper appears in both train and test) against a naive baseline; validated the choice of k with silhouette, Davies-Bouldin and bootstrap stability — and reported plainly where the models are weak |
| **Testing** | Wrote 85 pytest tests covering correctness, edge cases, guard rails, economic sanity checks (higher price ⇒ lower demand), evaluation leakage checks, and round-trip persistence |
| **Exploratory analysis** | Three Jupyter notebooks documenting EDA, predictive analysis, and purchase behaviour deep-dives |

---

## Results

| Metric | Value |
|---|---|
| Customers segmented | 2,000 |
| Purchase transactions analysed | 58,693 |
| PCA components retained | 3 |
| Cumulative variance explained | **80.8%** |
| Customer segments discovered | **4** |

**Segment sizes**

| Segment | Customers | Share |
|---|---|---|
| 0 | 602 | 30.1% |
| 1 | 610 | 30.5% |
| 2 | 526 | 26.3% |
| 3 | 262 | 13.1% |

**Purchase behavior**

| Model | What it predicts | Example finding |
|---|---|---|
| `PurchasePropensityModel` | P(purchase) from average price | Steep curve, but **extrapolated**: observed average prices only span $1.87–$2.10 (see caveat below) |
| `BrandChoiceModel` | Which of 5 brands is chosen | Beats the brand-share baseline on held-out customers (38.6% vs 33.8% accuracy) |
| `PurchaseQuantityModel` | Units purchased | Small but real price effect; explains ~3% of quantity variance on held-out customers |

---

## Visualizations

**Customer segments in PCA space** — the 4 clusters KMeans discovers, projected onto the top 2 principal components:

![Customer segments plotted in PCA space, colored by segment: well-off, career-focused, fewer-opportunities, standard](docs/images/segments_pca_scatter.png)

**Why 3 PCA components** — cumulative explained variance flattens out after the 3rd component (80.8%), which is why the pipeline retains 3:

![Cumulative explained variance by number of PCA components, showing 80.8% at 3 components](docs/images/pca_explained_variance.png)

**How customers react to price** — purchase probability and predicted quantity both fall as price rises, the direction economic theory predicts (but read the caveat below the chart):

![Purchase probability and predicted quantity both declining as price increases](docs/images/purchase_probability_quantity.png)

> **Caveat: this chart extrapolates.** The x-axis runs $1.00–$2.50, but the average price across the five brands only ever ranged from **$1.87 to $2.10** in the data (99% of trips fall inside $1.88–$2.09). The fitted curve is a logistic function pushed far beyond what was observed, so the headline numbers at the ends (78% purchase probability at $1.00, 9% at $2.50) describe the model's shape, not evidence that real shoppers would behave that way. Inside the observed range the effect is small — see the evaluation below.

**Brand choice under price competition** — each brand's probability of being chosen as its own price rises, holding competitors' prices at their historical average. Brand 3 is dashed because its price effect isn't statistically distinguishable from noise (see below):

![Brand choice probability declining with own price for four of five brands; Brand 3 shown dashed as not statistically significant](docs/images/brand_choice_elasticity.png)

> **On Brand 3's flat/rising curve — quantified, not just eyeballed.** The raw fitted coefficient for Brand 3 is `+0.43`, the only positive one among the five brands. Rather than trust or dismiss a single point estimate, `BrandChoiceModel.bootstrap_own_price_significance()` refits the model on 150 bootstrap resamples of the data and reports a confidence interval: Brand 3's is **[-0.29, +1.33]** — it crosses zero, so the sign can't be trusted. Every other brand's interval is comfortably negative and significant (e.g. Brand 1: [-4.27, -3.37]). The likely cause: Brand 3 has the smallest market share (5.7% of purchases) and the least own-price variation of any brand (std $0.046, a $1.87–$2.14 range), so there's little signal to separate its price effect from its correlation with Brand 4's and Brand 5's prices (r = 0.42 and 0.20). The chart reflects this honestly instead of hiding it or forcing the number to look "correct."

---

## Model Evaluation

Fitting a model and plotting its curve doesn't show it predicts anything. Every model here is scored on customers it never saw: purchase data has ~117 rows per shopper, so a random row split would leak each person's habits into the test set. Folds are grouped by customer ID instead, and each model is compared with a naive baseline that ignores price (the training purchase rate, brand shares, or mean quantity).

### Do the supervised models predict?

![Held-out performance of the three purchase models against their baselines](docs/images/heldout_performance.png)

*5-fold customer-grouped cross-validation; bars are means, whiskers are ±1 std across folds.*

| Model | Metric | Model | Baseline | Verdict |
|---|---|---|---|---|
| Propensity | ROC AUC | 0.537 | 0.500 | Barely better than chance; log loss (0.561 vs 0.562) and Brier score are indistinguishable from baseline |
| Brand choice | Accuracy | 0.386 | 0.338 | Real but modest lift; log loss 1.409 vs 1.445 |
| Quantity | R² | 0.033 | −0.010 | Tiny; the gain is within fold-to-fold noise (±0.036) |

**What this means.** Price is a genuine driver of *which brand* a shopper picks, but average price alone says almost nothing about *whether a given trip ends in a purchase* or *how many units*. The elasticity estimates are useful for describing direction and relative sensitivity between brands; they are not good enough to forecast individual purchases. Better predictors would need customer history (days since last purchase, previous brand and quantity, promotion exposure), which the data contains and the current models don't use — that's the obvious next step.

### Is k=4 the right number of segments?

![Elbow and silhouette curves for k=2 to 10, with k=4 marked](docs/images/cluster_selection.png)

| k | Inertia | Silhouette | Davies-Bouldin |
|---|---|---|---|
| 3 | 6,218 | 0.304 | 1.237 |
| **4** | **4,655** | **0.337** | **1.028** |
| 5 | 3,865 | 0.349 | 0.981 |
| 8 | 2,602 | 0.356 | 0.981 |
| 10 | 2,096 | 0.401 | 0.886 |

Being upfront about this: **the data doesn't single out k=4.** The elbow bends around 4 to 5, and silhouette scores (~0.30–0.40 everywhere, none near 0.5+) say the clusters overlap — demographic attributes like age, income and education form a continuum, not tight islands. Silhouette even keeps creeping up at k=9–10, but those extra groups would be too small to act on. I chose k=4 as the smallest count past the elbow that still gives segments a marketer can name and target.

Segment stability backs that up only partly: refitting on 30 bootstrap resamples and comparing with the full-data segments gives an adjusted Rand index of **0.64 on average (worst case 0.50)**. The segments are recognisably the same groups most of the time, but individual customers near a boundary do move between them, so treat segment membership as a soft label.

Reproduce everything above with `python main.py` (prints the tables) and `python -m scripts.make_evaluation_figures` (regenerates the charts).

---

## ML Pipeline

```
Raw demographic + purchase data
            │
            ▼
    StandardScaler          eliminates scale bias (age vs income)
            │
            ▼
    PCA (3 components)      reduces 7 features → 3, retains 80.8% variance
            │
            ▼
    KMeans (k=4)            discovers 4 distinct customer segments
            │
            ▼
  Segment labels + business-ready summary report
```

```
Purchase-occasion data (price, promotion, brand, quantity)
            │
            ├──▶ PurchasePropensityModel   (logistic regression)   → P(purchase | price)
            │
            ├──▶ BrandChoiceModel          (multinomial logistic)  → P(brand | 5 prices)
            │
            └──▶ PurchaseQuantityModel     (linear regression)     → predicted units
                        │
                        ▼
        Price elasticities: how demand reacts to a price change
```

---

## Project Structure

```
customer_segmentation/
│
├── docs/images/                    # Charts embedded in this README
│
├── data/                          # Raw input datasets
│   ├── segmentation+data.csv      # 2,000 customers × 8 demographic features
│   ├── purchase_data.csv          # 58,693 purchase transactions
│   └── brand_choice.csv           # Brand preference records
│
├── models/                        # Serialised, reusable model artefacts
│   ├── scaler_model.pkl
│   ├── pca_model.pkl
│   ├── kmeans_pca_model.pkl
│   ├── purchase_propensity_model.pkl
│   ├── brand_choice_model.pkl
│   └── purchase_quantity_model.pkl
│
├── notebooks/                     # Step-by-step analytical notebooks
│   ├── customer_analytics.ipynb
│   ├── customer_analytics_predictive_analysis.ipynb
│   └── purchase_analysis_descrip.ipynb
│
├── src/customer_segmentation/     # Reusable Python package
│   ├── data_loader.py             # Clean data ingestion with defaults
│   ├── segmentation.py            # SegmentationPipeline class
│   ├── purchase_behavior.py       # Purchase propensity, brand choice, quantity models
│   └── evaluation.py              # Customer-grouped splits and cross-validation
│
├── tests/
│   ├── conftest.py                # Shared synthetic purchase-data fixtures
│   ├── test_segmentation.py       # 30 tests
│   ├── test_purchase_behavior.py  # 27 tests
│   └── test_evaluation.py         # 28 tests — 85 total, 100% passing
│
├── scripts/
│   └── make_evaluation_figures.py # Regenerates the evaluation charts
│
├── main.py                        # Runnable pipeline entry point
└── pyproject.toml                 # Dependency management (uv)
```

---

## How to Run

```bash
# 1. Install dependencies (requires Python ≥ 3.10)
pip install uv && uv sync

# 2. Run both pipelines (segmentation + purchase-behavior prediction)
python main.py

# 3. Run the test suite
pytest tests/ -v
```

---

## Code Highlights

**Clean pipeline API — sklearn-style interface**
```python
from src.customer_segmentation import SegmentationPipeline, load_segmentation_data
from src.customer_segmentation.data_loader import SEGMENTATION_FEATURES

df = load_segmentation_data()
X = df[SEGMENTATION_FEATURES]

pipeline = SegmentationPipeline(n_components=3, n_clusters=4)
labels = pipeline.fit_predict(X)

print(pipeline.segment_summary(X).xs("mean", axis=1, level=1))   # per-segment feature means
print(pipeline.explained_variance())          # [0.357, 0.263, 0.188]
```

**Model persistence — deploy without retraining**
```python
pipeline.save()                          # serialises to models/

pipeline = SegmentationPipeline.load()   # reload anywhere
labels = pipeline.predict(new_customers)
```

**Predicting how customers react to price**
```python
from src.customer_segmentation import PurchasePropensityModel, load_purchase_data

df = load_purchase_data()
model = PurchasePropensityModel().fit(df)

model.predict_proba([1.0, 2.5])     # [0.776, 0.093]  — P(purchase) at each price
model.price_elasticity([1.0, 2.5])  # [-0.53, -5.33]  — demand grows more elastic as price rises
```

**Not trusting a coefficient just because it fit — quantifying uncertainty**
```python
from src.customer_segmentation import BrandChoiceModel, load_purchase_data

df = load_purchase_data()
occasions = df[df["Incidence"] == 1]
brand_model = BrandChoiceModel().fit(occasions)

est = brand_model.bootstrap_own_price_significance(brand=3, purchase_occasions=occasions)
est.mean, est.ci_low, est.ci_high, est.is_significant
# (0.43, -0.29, 1.33, False)  — the CI crosses zero, so this brand's
# apparent positive price coefficient can't actually be trusted
```

---

## Testing Philosophy

I treat tests as a first-class concern, not an afterthought. The test suite validates:

- **Data contracts** — schema, missing values, column presence
- **Guard rails** — clear errors if any model is used before fitting
- **Correctness** — label counts, array shapes, explained variance bounds, probabilities summing to 1
- **Economic sanity** — purchase probability and quantity both fall as price rises (the direction a real elasticity should have)
- **Statistical honesty** — a bootstrapped confidence interval, not just a point estimate, is available before trusting a coefficient's sign
- **Evaluation integrity** — train and test folds never share a customer; models are compared against a baseline; a model fit on shuffled targets can't beat the mean
- **Reproducibility** — `fit_predict` and `fit` → `predict` give identical results
- **Persistence** — saved and loaded models produce identical predictions

```bash
pytest tests/ -v
# 85 passed
```

---

## Technologies

`Python 3.10+` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `JupyterLab` · `pytest` · `uv`

---

*Thanks for taking the time to look at my work. I'd love to discuss the decisions I made and how I'd extend this further.*
