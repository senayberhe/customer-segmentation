# Customer Segmentation — Data Science Portfolio Project

**By Senay Berhe** · [tsionberhe@gmail.com](mailto:tsionberhe@gmail.com)

> An end-to-end unsupervised machine learning project demonstrating skills in exploratory data analysis, dimensionality reduction, clustering, software engineering best practices, and production-ready code design.

---

## Project Overview

This project builds a complete customer segmentation system from raw retail data. Starting from demographic and purchase transaction records, I designed and implemented a full ML pipeline that identifies four distinct customer segments — insight that a business could directly use to personalise marketing campaigns, optimise pricing, and improve retention.

The project goes beyond a notebook prototype: it is structured as a proper Python package with a clean API, model persistence, and a full test suite.

---

## Skills Demonstrated

| Area | What I did |
|---|---|
| **Unsupervised ML** | Applied PCA for dimensionality reduction and KMeans clustering to discover meaningful customer groups |
| **Feature engineering** | Selected and scaled 7 demographic features; reasoned about ordinal vs continuous encoding |
| **Statistical thinking** | Used explained variance analysis to justify retaining 3 PCA components (80.8% variance captured) |
| **Software engineering** | Structured code as a reusable Python package (`src/` layout) with clear separation of concerns |
| **API design** | Designed a `SegmentationPipeline` class with a consistent sklearn-style interface (`fit`, `predict`, `transform`) |
| **Model persistence** | Implemented `save()` / `load()` with proper serialisation so models can be deployed without retraining |
| **Testing** | Wrote 30 pytest tests covering correctness, edge cases, guard rails, and round-trip persistence |
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

---

## Project Structure

```
customer_segmentation/
│
├── data/                          # Raw input datasets
│   ├── segmentation+data.csv      # 2,000 customers × 8 demographic features
│   ├── purchase_data.csv          # 58,693 purchase transactions
│   └── brand_choice.csv           # Brand preference records
│
├── models/                        # Serialised, reusable model artefacts
│   ├── scaler_model.pkl
│   ├── pca_model.pkl
│   └── kmeans_pca_model.pkl
│
├── notebooks/                     # Step-by-step analytical notebooks
│   ├── customer_analytics.ipynb
│   ├── customer_analytics_predictive_analysis.ipynb
│   └── purchase_analysis_descrip.ipynb
│
├── src/customer_segmentation/     # Reusable Python package
│   ├── data_loader.py             # Clean data ingestion with defaults
│   └── segmentation.py            # SegmentationPipeline class
│
├── tests/
│   └── test_segmentation.py       # 30 tests — 100% passing
│
├── main.py                        # Runnable pipeline entry point
└── pyproject.toml                 # Dependency management (uv)
```

---

## How to Run

```bash
# 1. Install dependencies (requires Python ≥ 3.10)
pip install uv && uv sync

# 2. Run the full pipeline
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

print(pipeline.segment_summary(X)["mean"])   # per-segment feature means
print(pipeline.explained_variance())          # [0.357, 0.263, 0.188]
```

**Model persistence — deploy without retraining**
```python
pipeline.save()                          # serialises to models/

pipeline = SegmentationPipeline.load()   # reload anywhere
labels = pipeline.predict(new_customers)
```

---

## Testing Philosophy

I treat tests as a first-class concern, not an afterthought. The test suite validates:

- **Data contracts** — schema, missing values, column presence
- **Guard rails** — clear errors if the pipeline is used before fitting
- **Correctness** — label counts, array shapes, explained variance bounds
- **Reproducibility** — `fit_predict` and `fit` → `predict` give identical results
- **Persistence** — saved and loaded models produce identical predictions

```bash
pytest tests/ -v
# 30 passed in 1.05s
```

---

## Technologies

`Python 3.10+` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `JupyterLab` · `pytest` · `uv`

---

*Thanks for taking the time to look at my work. I'd love to discuss the decisions I made and how I'd extend this further.*
# customer-segemantation
