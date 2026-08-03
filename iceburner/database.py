"""Loading the Gorgon MOBO simulation database and filtering bad rows."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

DEFAULT_OUTLIER_INSPECT_COLUMNS = [
    "index_0",
    "Y_TDT",
    "Y_rhoRDT",
    "Y_minus_peak_IFAR",
    "current",
]


def load_state_dataframe(db_path, table="state"):
    """Load the full simulation table from a Gorgon MOBO sqlite database."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def find_outliers(df, tdt_min=1.0, rhordt_max=60.0):
    """Flag simulations with non-physical or non-finite objective values.

    Returns a boolean mask, ``True`` for rows considered outliers:
    ``Y_TDT`` below ``tdt_min``, ``Y_rhoRDT`` above ``rhordt_max``, or
    either objective non-finite.
    """
    return (
        (df["Y_TDT"] < tdt_min)
        | (df["Y_rhoRDT"] > rhordt_max)
        | ~np.isfinite(df["Y_TDT"])
        | ~np.isfinite(df["Y_rhoRDT"])
    )


def filter_outliers(df, tdt_min=1.0, rhordt_max=60.0, inspect_columns=None):
    """Remove outlier simulations from ``df``.

    Returns
    -------
    (clean_df, outliers_df) : tuple of pandas.DataFrame
        ``clean_df`` has outlier rows removed and its index reset.
        ``outliers_df`` holds the removed rows (``inspect_columns`` only,
        default :data:`DEFAULT_OUTLIER_INSPECT_COLUMNS`) for inspection.
    """
    mask = find_outliers(df, tdt_min=tdt_min, rhordt_max=rhordt_max)

    columns = inspect_columns or [
        c for c in DEFAULT_OUTLIER_INSPECT_COLUMNS if c in df.columns
    ]
    outliers_df = df.loc[mask, columns]

    clean_df = df.loc[~mask].copy().reset_index(drop=True)
    return clean_df, outliers_df
