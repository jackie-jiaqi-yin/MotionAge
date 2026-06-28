"""Table helpers for publication-facing MotionAge reports."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd


def summarize_fold_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str] | None = None,
    fold_column: str = "fold",
    ddof: int = 1,
) -> pd.DataFrame:
    """Summarize fold-level metrics as mean and fold standard deviation."""
    group_columns = list(group_columns)
    _require_columns(frame, [*group_columns, fold_column])

    if metric_columns is None:
        excluded = set(group_columns)
        excluded.add(fold_column)
        metric_columns = [
            column
            for column in frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        ]
    else:
        metric_columns = list(metric_columns)

    if not metric_columns:
        raise ValueError("At least one metric column is required.")
    _require_columns(frame, metric_columns)

    working = frame[[*group_columns, fold_column, *metric_columns]].copy()
    for metric in metric_columns:
        working[metric] = pd.to_numeric(working[metric], errors="coerce")

    rows: list[dict[str, object]] = []
    for group_key, group in working.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, object] = dict(zip(group_columns, group_key, strict=True))
        row["fold_count"] = int(group[fold_column].nunique(dropna=True))
        for metric in metric_columns:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else math.nan
            row[f"{metric}_sd"] = float(values.std(ddof=ddof)) if len(values) > ddof else math.nan
        rows.append(row)

    return pd.DataFrame(rows)


def format_mean_sd(mean: float, sd: float, *, digits: int = 3, missing: str = "") -> str:
    """Format a mean and fold standard deviation as `mean +/- SD`."""
    if not _is_finite(mean) or not _is_finite(sd):
        return missing
    return f"{float(mean):.{digits}f} +/- {float(sd):.{digits}f}"


def build_lower_triangle_ci_matrix(
    frame: pd.DataFrame,
    *,
    labels: Sequence[str],
    left_label_column: str = "left_label",
    right_label_column: str = "right_label",
    delta_column: str = "observed_auc_delta",
    lower_column: str = "ci95_lower",
    upper_column: str = "ci95_upper",
    digits: int = 4,
    diagonal: str = "-",
    missing: str = "",
    include_significance_star: bool = True,
) -> pd.DataFrame:
    """Build a lower-triangle matrix of paired deltas and confidence intervals."""
    labels = list(labels)
    _require_columns(frame, [left_label_column, right_label_column, delta_column, lower_column, upper_column])
    lookup = _paired_ci_lookup(
        frame,
        left_label_column=left_label_column,
        right_label_column=right_label_column,
        delta_column=delta_column,
        lower_column=lower_column,
        upper_column=upper_column,
    )

    values: list[list[str]] = []
    for row_index, row_label in enumerate(labels):
        row_values: list[str] = []
        for col_index, col_label in enumerate(labels):
            if row_index == col_index:
                row_values.append(diagonal)
            elif row_index < col_index:
                row_values.append(missing)
            else:
                cell = _lookup_or_reverse(lookup, row_label, col_label)
                row_values.append(
                    missing
                    if cell is None
                    else _format_delta_ci(
                        cell["delta"],
                        cell["lower"],
                        cell["upper"],
                        digits=digits,
                        include_significance_star=include_significance_star,
                    )
                )
        values.append(row_values)

    return pd.DataFrame(values, index=labels, columns=labels)


def render_markdown_table(matrix: pd.DataFrame, *, index_label: str = "") -> str:
    """Render a DataFrame as a padded GitHub-flavored Markdown table."""
    index_values = [str(value) for value in matrix.index]
    column_labels = [str(column) for column in matrix.columns]
    body = [[str(value) for value in row] for row in matrix.to_numpy()]

    index_width = max([len(index_label), *[len(value) for value in index_values]], default=0)
    widths = [
        max([len(column), *[len(row[index]) for row in body]], default=0)
        for index, column in enumerate(column_labels)
    ]

    rows: list[list[str]] = [[index_label, *column_labels]]
    rows.append(["-" * index_width, *["-" * width for width in widths]])
    rows.extend([[index_value, *body_row] for index_value, body_row in zip(index_values, body, strict=True)])

    padded_rows = []
    for row in rows:
        padded = [f"{row[0]:<{index_width}}"]
        padded.extend(f"{value:<{width}}" for value, width in zip(row[1:], widths, strict=True))
        padded_rows.append("| " + " | ".join(padded) + " |")
    return "\n".join(padded_rows)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _paired_ci_lookup(
    frame: pd.DataFrame,
    *,
    left_label_column: str,
    right_label_column: str,
    delta_column: str,
    lower_column: str,
    upper_column: str,
) -> dict[tuple[str, str], dict[str, float]]:
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in frame.to_dict("records"):
        key = (str(row[left_label_column]), str(row[right_label_column]))
        if key in lookup:
            raise ValueError(f"Duplicate paired comparison row for {key[0]} vs {key[1]}.")
        lookup[key] = {
            "delta": float(row[delta_column]),
            "lower": float(row[lower_column]),
            "upper": float(row[upper_column]),
        }
    return lookup


def _lookup_or_reverse(
    lookup: dict[tuple[str, str], dict[str, float]],
    left_label: str,
    right_label: str,
) -> dict[str, float] | None:
    direct = lookup.get((left_label, right_label))
    if direct is not None:
        return direct

    reverse = lookup.get((right_label, left_label))
    if reverse is None:
        return None
    return {
        "delta": -reverse["delta"],
        "lower": -reverse["upper"],
        "upper": -reverse["lower"],
    }


def _format_delta_ci(
    delta: float,
    lower: float,
    upper: float,
    *,
    digits: int,
    include_significance_star: bool,
) -> str:
    star = "*" if include_significance_star and (lower > 0.0 or upper < 0.0) else ""
    return f"{delta:.{digits}f} ({lower:.{digits}f}, {upper:.{digits}f}){star}"
