"""Data loading utilities for the customer segmentation project."""

from pathlib import Path

import pandas as pd

# Features used for segmentation modelling
SEGMENTATION_FEATURES = [
    "Sex",
    "Marital status",
    "Age",
    "Education",
    "Income",
    "Occupation",
    "Settlement size",
]

# Default data directory relative to project root
_DEFAULT_DATA_DIR = Path(__file__).parents[2] / "data"


def load_segmentation_data(
    path: str | Path | None = None,
    drop_id: bool = True,
) -> pd.DataFrame:
    """Load the raw customer segmentation dataset.

    Parameters
    ----------
    path:
        Path to the CSV file. Defaults to ``data/segmentation+data.csv``
        relative to the project root.
    drop_id:
        Whether to drop the ``ID`` column (default ``True``).

    Returns
    -------
    pd.DataFrame
        DataFrame with demographic features ready for modelling.
    """
    path = Path(path) if path else _DEFAULT_DATA_DIR / "segmentation+data.csv"
    df = pd.read_csv(path)
    if drop_id and "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


def load_purchase_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the customer purchase / transaction dataset.

    Parameters
    ----------
    path:
        Path to the CSV file. Defaults to ``data/purchase_data.csv``.

    Returns
    -------
    pd.DataFrame
        Raw purchase DataFrame.
    """
    path = Path(path) if path else _DEFAULT_DATA_DIR / "purchase_data.csv"
    return pd.read_csv(path)
