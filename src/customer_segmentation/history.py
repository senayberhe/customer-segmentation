"""Purchase-history features: what each shopper had done *before* a trip.

Every feature for a trip is computed from that shopper's earlier trips
only. Nothing from the trip itself or from later trips leaks in, so the
features are available at the moment the prediction would be made.
"""

from __future__ import annotations

import pandas as pd

HISTORY_COLUMNS = ["Days_Since_Purchase", "No_Prior_Purchase", "Last_Brand"]


def add_purchase_history(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with per-shopper purchase-history columns.

    Added columns
    -------------
    Days_Since_Purchase:
        Days since the shopper's previous purchase. Before their first
        purchase there is no such day, so it counts days since their first
        observed trip instead (a lower bound — see ``No_Prior_Purchase``).
    No_Prior_Purchase:
        1.0 if the shopper hasn't bought anything before this trip.
    Last_Brand:
        Brand of the shopper's most recent earlier purchase (0 if none).

    Compute this on the *full* trip data (not just the ``Incidence == 1``
    rows), then filter. Row order and index of *df* are preserved.
    """
    ordered = df.sort_values(["ID", "Day"], kind="stable")
    by_id = ordered["ID"]
    bought = ordered["Incidence"] == 1

    # Value at the shopper's most recent *earlier* purchase: forward-fill
    # over purchase rows, then shift by one trip so a trip never sees itself.
    def previous_purchase(values: pd.Series) -> pd.Series:
        return values.where(bought).groupby(by_id).transform(lambda s: s.ffill().shift())

    last_day = previous_purchase(ordered["Day"])
    first_day = ordered.groupby("ID")["Day"].transform("min")
    no_prior = last_day.isna()

    history = pd.DataFrame(
        {
            "Days_Since_Purchase": ordered["Day"] - last_day.fillna(first_day),
            "No_Prior_Purchase": no_prior.astype(float),
            "Last_Brand": previous_purchase(ordered["Brand"]).fillna(0).astype(int),
        },
        index=ordered.index,
    )
    return df.join(history)
