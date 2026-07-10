"""Neural-network data preparation for the Walmart forecasting project.

The tree models (XGBoost / LightGBM) treat every row as an independent example.
Neural time-series models (DLinear, N-BEATS, PatchTST) instead need the data in
the "long" panel format that Nixtla's `neuralforecast` expects:

    unique_id | ds (date) | y (target) | ...exogenous columns...

Each `unique_id` is one Store+Dept series. We reuse the teammate's
`merge_and_enrich` so cleaning/feature logic stays identical to the tree side.

Typical use in a notebook:

    from src.features.nn_preprocessing import build_long_df, build_static_df, \
        temporal_split, FREQ, FUTR_EXOG, HIST_EXOG

    Y_df = build_long_df(train_raw, stores, features)
    static_df = build_static_df(stores)
    train_df, valid_df, horizon = temporal_split(Y_df, "2012-01-01")
"""
import numpy as np
import pandas as pd

from src.features.preprocessing import merge_and_enrich

# Walmart weekly data is stamped on Fridays -> weekly frequency anchored to Friday.
FREQ = "W-FRI"

# Future exogenous: deterministic calendar values we also know for future dates.
FUTR_EXOG = ["IsHoliday", "Year", "Month", "Week", "Day",
             "Month_Sin", "Month_Cos", "Week_Sin", "Week_Cos"]

# Historical exogenous: only known for past dates (measured after the fact).
HIST_EXOG = ["Temperature", "Fuel_Price", "CPI", "Unemployment",
             "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
             "Has_Markdown"]

# Static exogenous: constant per store.
STAT_EXOG = ["Size", "Type_code"]

_TYPE_MAP = {"A": 0, "B": 1, "C": 2}


def _recompute_calendar(df):
    """Rebuild deterministic calendar features from ds (needed for filled rows)."""
    df["Year"] = df["ds"].dt.year
    df["Month"] = df["ds"].dt.month
    df["Week"] = df["ds"].dt.isocalendar().week.astype(int)
    df["Day"] = df["ds"].dt.day
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Week_Sin"] = np.sin(2 * np.pi * df["Week"] / 52)
    df["Week_Cos"] = np.cos(2 * np.pi * df["Week"] / 52)
    return df


def _fill_series_gaps(df):
    """Give every series a continuous weekly index (no missing Fridays).

    Missing weeks get y = 0 (no sales recorded). Calendar features are
    recomputed from the date; historical exogenous are forward/back filled
    within each series.
    """
    filled = []
    for uid, g in df.groupby("unique_id", sort=False):
        g = g.set_index("ds").sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq=FREQ)
        g = g.reindex(full_idx)
        g.index.name = "ds"
        g["unique_id"] = uid
        g["y"] = g["y"].fillna(0.0)
        # historical exogenous: carry last known value across the gap
        g[HIST_EXOG] = g[HIST_EXOG].ffill().bfill()
        g["IsHoliday"] = g["IsHoliday"].fillna(0).astype(int)
        filled.append(g.reset_index())
    out = pd.concat(filled, ignore_index=True)
    out = _recompute_calendar(out)
    return out


def build_long_df(train_raw, stores, features, fill_gaps=True):
    """Return the neuralforecast-ready long dataframe (unique_id, ds, y, exog)."""
    df = merge_and_enrich(train_raw, stores, features)

    df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)
    df = df.rename(columns={"Date": "ds", "Weekly_Sales": "y"})

    keep = ["unique_id", "ds", "y"] + FUTR_EXOG + HIST_EXOG
    df = df[keep].sort_values(["unique_id", "ds"]).reset_index(drop=True)

    if fill_gaps:
        df = _fill_series_gaps(df)

    # neuralforecast wants exactly this column order up front
    df = df[["unique_id", "ds", "y"] + FUTR_EXOG + HIST_EXOG]
    return df


def build_static_df(stores):
    """Return one row per series with its static (constant) features.

    NOTE: static features are per Store, so every Dept in a store shares them.
    The unique_id list is built later from the long df; this helper returns a
    per-Store table you join on. See `attach_static` for the convenience wrapper.
    """
    s = stores.copy()
    s["Type_code"] = s["Type"].map(_TYPE_MAP).astype(int)
    return s[["Store", "Type_code", "Size"]]


def attach_static(Y_df, stores):
    """Build the static_df aligned to the unique_ids present in Y_df."""
    ids = Y_df["unique_id"].drop_duplicates().to_frame()
    ids["Store"] = ids["unique_id"].str.split("_").str[0].astype(int)
    static_lookup = build_static_df(stores)
    static_df = ids.merge(static_lookup, on="Store", how="left")
    return static_df[["unique_id", "Type_code", "Size"]]


def temporal_split(Y_df, split_date="2012-01-01"):
    """Split into train / validation by date (same cutoff as the tree models).

    Returns (train_df, valid_df, horizon) where horizon = number of future
    weeks in the validation period (used as the model's forecast horizon `h`).
    """
    split = pd.Timestamp(split_date)
    train_df = Y_df[Y_df["ds"] < split].reset_index(drop=True)
    valid_df = Y_df[Y_df["ds"] >= split].reset_index(drop=True)
    horizon = int(valid_df["ds"].nunique())
    return train_df, valid_df, horizon


def get_real_validation(train_raw, stores, features, split_date="2012-01-01"):
    """Return the REAL observed validation rows (no gap-fill zeros).

    These are the same rows the tree models score on (from merge_and_enrich,
    Date >= split_date), so WMAE computed here is directly comparable to the
    tree/tabular models. Columns: unique_id, ds, y, IsHoliday.
    """
    df = merge_and_enrich(train_raw, stores, features)
    df = df[df["Date"] >= pd.Timestamp(split_date)].copy()
    df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)
    df = df.rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    return df[["unique_id", "ds", "y", "IsHoliday"]].reset_index(drop=True)
