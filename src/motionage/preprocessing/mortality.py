"""Utilities for constructing fixed-horizon mortality labels."""

from __future__ import annotations

import pandas as pd


DEFAULT_FOLLOWUP_MONTH_COLUMN = "permth_int"
DEFAULT_FOLLOWUP_MONTHS = 60


def normalize_id_column(df: pd.DataFrame, *, id_column: str) -> pd.DataFrame:
    """Drop missing IDs and normalize the join key to int64."""
    normalized = df.dropna(subset=[id_column]).copy()
    normalized[id_column] = pd.to_numeric(normalized[id_column], errors="coerce")
    if normalized[id_column].isna().any():
        raise ValueError(f"Found non-numeric participant IDs in '{id_column}'.")
    normalized[id_column] = normalized[id_column].astype("int64")
    return normalized


def fixed_horizon_target_definition(
    *,
    mortality_column: str,
    followup_month_column: str,
    followup_months: int,
) -> str:
    """Return a concise textual description of the derived mortality target."""
    return f"I({mortality_column} = 1 and {followup_month_column} <= {followup_months})"


def build_fixed_horizon_mortality_table(
    mortality_raw: pd.DataFrame,
    *,
    id_column: str,
    mortality_column: str,
    followup_month_column: str = DEFAULT_FOLLOWUP_MONTH_COLUMN,
    followup_months: int = DEFAULT_FOLLOWUP_MONTHS,
    passthrough_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return one row per participant with a fixed-horizon binary mortality label."""
    passthrough = list(passthrough_columns or [])
    required = [id_column, mortality_column, followup_month_column, *passthrough]
    missing = [column for column in required if column not in mortality_raw.columns]
    if missing:
        raise KeyError(f"Mortality source is missing required columns: {missing}")

    table = normalize_id_column(mortality_raw[required], id_column=id_column)
    table[mortality_column] = pd.to_numeric(table[mortality_column], errors="coerce")
    table[followup_month_column] = pd.to_numeric(table[followup_month_column], errors="coerce")
    table = table[table[mortality_column].isin([0, 1])].copy()
    if table.empty:
        raise ValueError(f"No rows found with {mortality_column} in {{0, 1}}.")

    # A death without follow-up months cannot be placed relative to the requested horizon.
    invalid_deaths = table[mortality_column].eq(1) & table[followup_month_column].isna()
    table = table.loc[~invalid_deaths].copy()
    if table.empty:
        raise ValueError(
            "No rows remain after dropping deaths with missing "
            f"'{followup_month_column}' values."
        )

    for column in [mortality_column, followup_month_column, *passthrough]:
        duplicate_counts = table.groupby(id_column)[column].nunique(dropna=False)
        conflicting_ids = duplicate_counts[duplicate_counts > 1]
        if not conflicting_ids.empty:
            raise ValueError(
                f"Found participants with conflicting {column} values: {conflicting_ids.index.tolist()[:10]}"
            )

    table = table.drop_duplicates(subset=[id_column]).copy()
    table[mortality_column] = (
        table[mortality_column].eq(1) & table[followup_month_column].le(float(followup_months))
    ).astype("int8")
    return table.sort_values(id_column).reset_index(drop=True)
