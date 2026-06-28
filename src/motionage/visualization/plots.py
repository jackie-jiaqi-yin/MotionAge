"""Reusable plotting helpers for MotionAge activity summaries."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_ID_COLUMN = "SEQN"
DEFAULT_DAY_COLUMN = "PAXDAY"
DEFAULT_HOUR_COLUMN = "PAXHOUR"
DEFAULT_AGE_COLUMN = "RIDAGEYR"
DEFAULT_INTENSITY_COLUMN = "intensity_mean"


def plot_metric_interval_forest(
    data: pd.DataFrame,
    *,
    label_col: str = "model_label",
    point_col: str = "estimate",
    lower_col: str = "ci_lower",
    upper_col: str = "ci_upper",
    xlabel: str = "Metric",
    title: Optional[str] = "Metric Estimate with Confidence Interval",
    reference_value: Optional[float] = None,
    marker_color: str = "#1f77b4",
    interval_color: str = "#1f77b4",
    reference_color: str = "0.45",
    marker_size: float = 36.0,
    interval_linewidth: float = 1.6,
    capsize: float = 3.0,
    figsize: Optional[tuple[int | float, int | float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot aggregate metric estimates with confidence intervals."""
    required = [label_col, point_col, lower_col, upper_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Metric interval plot input is missing required columns: {missing}")
    if data.empty:
        raise ValueError("Metric interval plot input is empty.")
    if marker_size <= 0:
        raise ValueError("marker_size must be positive")
    if interval_linewidth <= 0:
        raise ValueError("interval_linewidth must be positive")
    if capsize < 0:
        raise ValueError("capsize must be non-negative")

    plot_data = data[[label_col, point_col, lower_col, upper_col]].copy()
    for column in (point_col, lower_col, upper_col):
        plot_data[column] = pd.to_numeric(plot_data[column], errors="raise")
    if (plot_data[lower_col] > plot_data[point_col]).any() or (
        plot_data[point_col] > plot_data[upper_col]
    ).any():
        raise ValueError("Confidence interval bounds must satisfy lower <= point <= upper.")

    if ax is None:
        default_height = max(2.0, 0.45 * len(plot_data) + 0.8)
        figsize = figsize if figsize is not None else (6.5, default_height)
        _, ax = plt.subplots(figsize=figsize)

    y_positions = list(range(len(plot_data)))
    cap_half_height = 0.06 * capsize
    for y_position, row in zip(y_positions, plot_data.to_dict(orient="records"), strict=True):
        lower = float(row[lower_col])
        upper = float(row[upper_col])
        ax.plot(
            [lower, upper],
            [y_position, y_position],
            color=interval_color,
            linewidth=interval_linewidth,
            zorder=1,
        )
        if capsize:
            ax.plot(
                [lower, lower],
                [y_position - cap_half_height, y_position + cap_half_height],
                color=interval_color,
                linewidth=interval_linewidth,
                zorder=1,
            )
            ax.plot(
                [upper, upper],
                [y_position - cap_half_height, y_position + cap_half_height],
                color=interval_color,
                linewidth=interval_linewidth,
                zorder=1,
            )
    ax.scatter(
        plot_data[point_col],
        y_positions,
        s=marker_size,
        color=marker_color,
        zorder=2,
    )
    if reference_value is not None:
        ax.axvline(
            reference_value,
            color=reference_color,
            linestyle="--",
            linewidth=1.0,
            zorder=0,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(label) for label in plot_data[label_col]])
    ax.set_xlabel(xlabel)
    ax.set_title(title if title is not None else "", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    return ax


def plot_intensity_timeseries_by_hour(
    data: pd.DataFrame,
    intensity_col: str = DEFAULT_INTENSITY_COLUMN,
    hour_col: str = DEFAULT_HOUR_COLUMN,
    day_col: str = DEFAULT_DAY_COLUMN,
    age_col: str = DEFAULT_AGE_COLUMN,
    id_col: str = DEFAULT_ID_COLUMN,
    n_age_groups: int = 4,
    age_bins: Optional[Sequence[float]] = None,
    age_labels: Optional[Sequence[str]] = None,
    ids: Optional[list] = None,
    figsize: Optional[tuple[int | float, int | float]] = None,
    title: Optional[str] = "Mean Intensity Time Series by Hour and Age Group",
    xlabel: str = "Time of Week",
    ylabel: str = "Mean Intensity",
    ax: Optional[plt.Axes] = None,
    show_std: bool = True,
    show_stats: bool = True,
    legend_title: Optional[str] = None,
    legend_ncol: int = 1,
    legend_loc: str = "best",
    legend_bbox_to_anchor: Optional[tuple[float, float]] = None,
    legend_frameon: bool = True,
    axis_label_fontsize: Optional[float] = None,
    tick_label_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = None,
    legend_title_fontsize: Optional[float] = None,
    line_width: float = 2.0,
    y_headroom: float = 0.0,
    week_start_day: int = 1,
    tick_interval_hours: int = 24,
    palette: Optional[Any] = None,
    theme_params: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot one-week hourly mean intensity trajectories by age group."""
    required = [intensity_col, hour_col, day_col, age_col, id_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Intensity plot input is missing required columns: {missing}")
    if y_headroom < 0:
        raise ValueError("y_headroom must be non-negative")
    if line_width <= 0:
        raise ValueError("line_width must be positive")

    theme = {"grid_alpha": 0.3, "stats_box_alpha": 0.5}
    if theme_params:
        theme.update(theme_params)
    figsize = figsize if figsize is not None else plt.rcParams.get("figure.figsize", (14, 6))

    plot_data = data.copy()
    if ids is not None:
        plot_data = plot_data[plot_data[id_col].isin(ids)].copy()
    if plot_data.empty:
        raise ValueError("Intensity plot input is empty after filtering.")

    plot_data["hour_index"] = ((plot_data[day_col] - week_start_day) % 7) * 24 + plot_data[hour_col]

    hourly_stats = (
        plot_data.groupby([id_col, "hour_index"])
        .agg({intensity_col: "mean", age_col: "first"})
        .reset_index()
    )
    hourly_stats["age_group"], group_labels, computed_age_bins = _build_age_groups(
        hourly_stats,
        age_col=age_col,
        n_age_groups=n_age_groups,
        age_bins=age_bins,
        age_labels=age_labels,
    )
    hourly_age_summary = (
        hourly_stats.groupby(["hour_index", "age_group"], observed=False)[intensity_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    colors = _resolve_palette(palette, len(group_labels))
    for idx, age_group in enumerate(group_labels):
        group_data = hourly_age_summary[
            hourly_age_summary["age_group"] == age_group
        ].sort_values("hour_index")
        if computed_age_bins is None:
            label = str(age_group)
        else:
            age_idx = group_labels.index(age_group)
            age_min = computed_age_bins[age_idx]
            age_max = computed_age_bins[age_idx + 1]
            label = f"{age_group}: {age_min:.0f}-{age_max:.0f} years"

        ax.plot(
            group_data["hour_index"],
            group_data["mean"],
            linewidth=line_width,
            color=colors[idx],
            label=label,
            **kwargs,
        )
        if show_std:
            ax.fill_between(
                group_data["hour_index"],
                group_data["mean"] - group_data["std"],
                group_data["mean"] + group_data["std"],
                color=colors[idx],
                alpha=0.2,
            )

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_title(title if title is not None else "", fontweight="bold")
    for day in range(1, 7):
        ax.axvline(x=day * 24, color="gray", linestyle="--", alpha=0.25, linewidth=0.8)

    ax.set_xlim(0, 7 * 24)
    tick_positions, tick_labels = _build_week_ticks(
        week_start_day=week_start_day,
        tick_interval_hours=tick_interval_hours,
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        rotation=45 if tick_interval_hours < 24 else 0,
        ha="right" if tick_interval_hours < 24 else "center",
    )
    if tick_label_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

    if y_headroom:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + (y_max - y_min) * y_headroom)

    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        frameon=legend_frameon,
        framealpha=0.9 if legend_frameon else None,
        title=legend_title,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    ax.grid(True, alpha=theme["grid_alpha"], linestyle="--", axis="y")

    if show_stats:
        stats_text = (
            f"N = {hourly_stats[id_col].nunique()} participants\n"
            f"Observations = {len(hourly_stats)}\n"
            f"Overall mean = {hourly_stats[intensity_col].mean():.2f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=theme["stats_box_alpha"]),
            fontsize=10,
        )

    plt.tight_layout()
    return ax


def _build_age_groups(
    data: pd.DataFrame,
    age_col: str,
    n_age_groups: int,
    age_bins: Optional[Sequence[float]] = None,
    age_labels: Optional[Sequence[str]] = None,
) -> tuple[pd.Series, list[str], Optional[list[float]]]:
    if age_bins is not None:
        if age_labels is None:
            raise ValueError("age_labels must be provided when age_bins is set")
        if len(age_bins) < 2:
            raise ValueError("age_bins must contain at least two edges")
        if len(age_labels) != len(age_bins) - 1:
            raise ValueError("age_labels length must match len(age_bins) - 1")
        if any(right <= left for left, right in zip(age_bins, age_bins[1:])):
            raise ValueError("age_bins must be strictly increasing")
        groups = pd.cut(
            data[age_col],
            bins=list(age_bins),
            labels=list(age_labels),
            right=False,
            include_lowest=True,
        )
        return groups, list(age_labels), None

    group_labels = [f"Age {idx + 1}" for idx in range(n_age_groups)]
    groups, quantile_bins = pd.qcut(
        data[age_col],
        q=n_age_groups,
        labels=group_labels,
        retbins=True,
        duplicates="drop",
    )
    return groups, list(groups.cat.categories), list(quantile_bins)


def _resolve_palette(palette: Optional[Any], n_colors: int) -> list:
    if palette is None:
        return list(sns.color_palette("husl", n_colors))
    if isinstance(palette, str):
        return list(sns.color_palette(palette, n_colors))

    colors = list(palette)
    if len(colors) < n_colors:
        raise ValueError("palette must provide at least as many colors as age groups")
    return colors[:n_colors]


def _build_week_ticks(week_start_day: int, tick_interval_hours: int) -> tuple[list[int], list[str]]:
    if not 1 <= week_start_day <= 7:
        raise ValueError("week_start_day must be in [1, 7]")
    if tick_interval_hours <= 0 or 24 % tick_interval_hours != 0:
        raise ValueError("tick_interval_hours must be a positive divisor of 24")

    weekday_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    ordered_days = [((week_start_day - 1 + offset) % 7) + 1 for offset in range(7)]

    ticks: list[int] = []
    labels: list[str] = []
    for day_offset, day_code in enumerate(ordered_days):
        day_name = weekday_names[day_code - 1]
        for hour in range(0, 24, tick_interval_hours):
            ticks.append(day_offset * 24 + hour)
            labels.append(f"{day_name} {hour:02d}:00")
    return ticks, labels
