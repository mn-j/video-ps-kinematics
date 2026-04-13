"""Plotting helpers for kinematic feature analysis.

Extracted from the orchestrator function in ``scripts/analyze.py`` (lines
4142-4560) so that visualisation logic can be tested and reused independently
of the monolithic analysis script.

All functions receive their dependencies (palette, stat_tests, etc.) as
explicit parameters rather than relying on closure variables.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Discrete (integer-valued) feature columns that should use bubble plots
# instead of standard boxplots.
BUBBLE_FEATURES: frozenset[str] = frozenset(
    {
        "Num Hesitations",
        "Num Arrests",
        "Num Interruptions (2x)",
    }
)


# ---------------------------------------------------------------------------
# Significance brackets
# ---------------------------------------------------------------------------


def add_significance_brackets(
    ax: plt.Axes,
    data: pd.DataFrame,
    col: str,
    stat_tests_map: dict[str, Any],
    y_pad_frac: float = 0.05,
) -> None:
    """Draw Bonferroni-corrected pairwise significance brackets on a boxplot.

    Reads pre-computed post-hoc results from *stat_tests_map* and annotates
    score-0 vs each other score bracket on *ax*.  Only draws brackets for
    comparisons that are statistically significant (i.e. not ``"ns"``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object containing the boxplot.
    data : pd.DataFrame
        Data used for the plot; must contain a ``"score_clean"`` column.
    col : str
        Feature column name used as the key into *stat_tests_map*.
    stat_tests_map : dict
        Mapping of feature names to statistical test result dicts.  Each
        value is expected to contain an optional ``"posthoc_bonferroni"`` key
        whose value maps pair labels (e.g. ``"0v2"``) to dicts with at least
        a ``"sig"`` field.
    y_pad_frac : float, optional
        Fraction of the y-axis range used as initial padding above the data
        for the first bracket.  Default ``0.05``.
    """
    if col not in stat_tests_map:
        return

    _posthoc = stat_tests_map[col].get("posthoc_bonferroni", {})
    if not _posthoc:
        return

    _score_vals = sorted(data["score_clean"].unique())
    _score_to_x: dict[int, int] = {s: i for i, s in enumerate(_score_vals)}

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    _ref = _score_vals[0]
    _bracket_top = y_max + y_range * y_pad_frac

    for _k, _info in _posthoc.items():
        _sig = _info.get("sig", "ns")
        if _sig == "ns":
            continue
        try:
            _s2 = int(_k.split("v")[1])
        except (IndexError, ValueError):
            continue
        if _ref not in _score_to_x or _s2 not in _score_to_x:
            continue

        x1, x2 = _score_to_x[_ref], _score_to_x[_s2]
        _bh = _bracket_top + y_range * 0.02
        ax.plot(
            [x1, x1, x2, x2],
            [_bracket_top, _bh, _bh, _bracket_top],
            lw=1.2,
            c="black",
        )
        ax.text(
            (x1 + x2) / 2,
            _bh + y_range * 0.01,
            _sig,
            ha="center",
            va="bottom",
            fontsize=10,
        )
        _bracket_top += y_range * 0.10

    ax.set_ylim(y_min, _bracket_top + y_range * 0.05)


# ---------------------------------------------------------------------------
# Single boxplot
# ---------------------------------------------------------------------------


def create_single_boxplot(
    data: pd.DataFrame,
    col: str,
    ylabel: str,
    description: str,
    score_col_label: str,
    palette: Sequence[Any],
    stat_tests: dict[str, Any],
    figsize: tuple[int, int] = (12, 8),
    title_extra: str = "",
) -> plt.Figure:
    """Create a single boxplot with stripplot overlay for a kinematic feature.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot; must contain ``"score_clean"`` and *col*.
    col : str
        Column name of the kinematic feature.
    ylabel : str
        Human-readable label for the y-axis.
    description : str
        Short description included in the plot title.
    score_col_label : str
        Label for the MDS-UPDRS score variant (e.g. ``"3.4"``).
    palette : sequence
        Colour palette (one colour per score value).
    stat_tests : dict
        Pre-computed statistical tests passed through to
        :func:`add_significance_brackets`.
    figsize : tuple[int, int], optional
        Figure size in inches.  Default ``(12, 8)``.
    title_extra : str, optional
        Additional text appended below the title (e.g. outlier info).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.boxplot(
            data=data,
            x="score_clean",
            y=col,
            hue="score_clean",
            palette=palette,
            legend=False,
            ax=ax,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markerfacecolor": "gray",
                "markersize": 4,
                "alpha": 0.5,
            },
        )

    sns.stripplot(
        data=data,
        x="score_clean",
        y=col,
        color="black",
        alpha=0.4,
        size=4,
        ax=ax,
        jitter=True,
    )

    ax.set_xlabel(f"MDS-UPDRS {score_col_label} Score", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    title = f"{ylabel} by MDS-UPDRS {score_col_label} Score\n({description})"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=14)

    for i, sv in enumerate(sorted(data["score_clean"].unique())):
        n = int((data["score_clean"] == sv).sum())
        y_pos = ax.get_ylim()[1]
        ax.annotate(
            f"n={n}",
            xy=(i, y_pos),
            ha="center",
            va="bottom",
            fontsize=9,
            color="gray",
        )

    add_significance_brackets(ax, data, col, stat_tests)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bubble plot (for discrete integer-valued features)
# ---------------------------------------------------------------------------


def create_bubble_plot(
    data: pd.DataFrame,
    col: str,
    ylabel: str,
    description: str,
    score_col_label: str,
    palette: Sequence[Any],
    stat_tests: dict[str, Any],
    figsize: tuple[int, int] = (12, 8),
    title_extra: str = "",
) -> plt.Figure:
    """Bubble plot for discrete integer-valued features.

    Each bubble corresponds to a (score, count) pair; bubble size encodes the
    proportion of participants in that score group with that value.  Matches
    the style of Zarrat Ehsan et al. (2024) Figure 3 subplot (l).

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot; must contain ``"score_clean"`` and *col*.
    col : str
        Column name of the discrete feature (e.g. ``"Num Hesitations"``).
    ylabel : str
        Human-readable label for the y-axis.
    description : str
        Short description included in the plot title.
    score_col_label : str
        Label for the MDS-UPDRS score variant.
    palette : sequence
        Colour palette (one colour per score value).
    stat_tests : dict
        Pre-computed statistical tests passed through to
        :func:`add_significance_brackets`.
    figsize : tuple[int, int], optional
        Figure size in inches.  Default ``(12, 8)``.
    title_extra : str, optional
        Additional text appended below the title.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    _score_vals = sorted(data["score_clean"].unique())
    _pal_map = {s: palette[i] for i, s in enumerate(_score_vals)}
    _all_counts = sorted(data[col].dropna().unique())
    _max_size = 1200.0

    for _sv in _score_vals:
        _grp = data.loc[data["score_clean"] == _sv, col].dropna()
        if _grp.empty:
            continue
        _n_grp = len(_grp)
        for _cnt in _all_counts:
            _prop = float((_grp == _cnt).sum()) / _n_grp
            if _prop == 0.0:
                continue
            ax.scatter(
                _sv,
                _cnt,
                s=_prop * _max_size,
                c=[_pal_map[_sv]],
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

    add_significance_brackets(ax, data, col, stat_tests, y_pad_frac=0.08)

    ax.set_xlabel(f"MDS-UPDRS {score_col_label} Score", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    title = f"{ylabel} by MDS-UPDRS {score_col_label} Score\n({description})"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=14)

    ax.set_xticks(_score_vals)

    for i, sv in enumerate(_score_vals):
        n = int((data["score_clean"] == sv).sum())
        y_pos = ax.get_ylim()[1]
        ax.annotate(
            f"n={n}",
            xy=(sv, y_pos),
            ha="center",
            va="bottom",
            fontsize=9,
            color="gray",
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helper: sanitise column name for use in file paths
# ---------------------------------------------------------------------------


def _safe_col_name(col: str) -> str:
    """Return a filesystem-safe version of a column name."""
    return col.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")


# ---------------------------------------------------------------------------
# Per-feature boxplot loop
# ---------------------------------------------------------------------------


def plot_feature_boxplots(
    merged: pd.DataFrame,
    features: list[tuple[str, str, str, bool]],
    score_col_label: str,
    palette: Sequence[Any],
    stat_tests: dict[str, Any],
    output_dir: str,
    detect_extreme_outliers_fn: Callable[..., tuple[bool, pd.Series, float, float]],
    extreme_iqr_multiplier: float,
    figsize: tuple[int, int],
    save_plots: bool,
    show_plots: bool,
) -> dict[str, pd.DataFrame | None]:
    """Plot individual and filtered boxplots for all features.

    For each feature in *features* a boxplot (or bubble plot for discrete
    features listed in :data:`BUBBLE_FEATURES`) is generated.  When extreme
    outliers are detected a second "filtered" plot is produced.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged dataset containing ``"score_clean"`` and all feature columns.
    features : list[tuple[str, str, str, bool]]
        Iterable of ``(col, ylabel, description, can_normalise)`` tuples.
    score_col_label : str
        Label for the MDS-UPDRS score variant.
    palette : sequence
        Colour palette.
    stat_tests : dict
        Pre-computed statistical tests.
    output_dir : str
        Directory to write PNG files into.
    detect_extreme_outliers_fn : callable
        Function with signature
        ``(series, *, iqr_multiplier) -> (bool, mask, lo, hi)``.
    extreme_iqr_multiplier : float
        Multiplier passed to *detect_extreme_outliers_fn*.
    figsize : tuple[int, int]
        Figure size in inches.
    save_plots : bool
        Whether to save figures to *output_dir*.
    show_plots : bool
        Whether to display figures interactively.

    Returns
    -------
    dict[str, pd.DataFrame | None]
        Mapping from feature column name to the filtered DataFrame (without
        extreme outliers) or ``None`` if no extremes were found.
    """
    _feat_filtered: dict[str, pd.DataFrame | None] = {}

    for col, ylabel, description, can_norm in features:
        valid_data = merged[col].dropna()
        if len(valid_data) == 0:
            _feat_filtered[col] = None
            continue

        # Use bubble plot for discrete integer-valued features
        if col in BUBBLE_FEATURES:
            fig = create_bubble_plot(
                merged, col, ylabel, description, score_col_label, palette, stat_tests, figsize
            )
        else:
            fig = create_single_boxplot(
                merged, col, ylabel, description, score_col_label, palette, stat_tests, figsize
            )

        if save_plots:
            safe_col = _safe_col_name(col)
            plot_path = os.path.join(output_dir, f"boxplot_{safe_col}_by_{score_col_label}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Detect and optionally re-plot without extreme outliers
        has_extremes, mask_no_extremes, lower_bound, upper_bound = detect_extreme_outliers_fn(
            merged[col], iqr_multiplier=extreme_iqr_multiplier
        )

        if has_extremes:
            n_extremes = int((~mask_no_extremes).sum())
            print(f"    {n_extremes} extreme outliers in {col}")
            filtered_data = merged[mask_no_extremes].copy()
            _feat_filtered[col] = filtered_data if len(filtered_data) > 0 else None

            if len(filtered_data) > 0:
                title_extra = (
                    f"[Excluding {n_extremes} extreme values outside "
                    f"[{lower_bound:.1f}, {upper_bound:.1f}]]"
                )
                fig = create_single_boxplot(
                    filtered_data,
                    col,
                    ylabel,
                    description,
                    score_col_label,
                    palette,
                    stat_tests,
                    figsize,
                    title_extra,
                )
                if save_plots:
                    safe_col = _safe_col_name(col)
                    plot_path = os.path.join(
                        output_dir, f"boxplot_{safe_col}_by_{score_col_label}_no_extremes.png"
                    )
                    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                if show_plots:
                    plt.show()
                else:
                    plt.close()
        else:
            _feat_filtered[col] = None

    return _feat_filtered


# ---------------------------------------------------------------------------
# Combined multi-panel figure
# ---------------------------------------------------------------------------


def plot_combined_figure(
    merged: pd.DataFrame,
    features: list[tuple[str, str, str, bool]],
    feature_filtered_dfs: dict[str, pd.DataFrame | None],
    score_col_label: str,
    palette: Sequence[Any],
    output_dir: str,
    save_plots: bool,
    show_plots: bool,
) -> None:
    """Create a combined multi-panel figure of all features.

    Each panel is a boxplot for one feature.  If a filtered DataFrame (with
    extreme outliers removed) exists for a feature it is used; otherwise the
    full *merged* dataset is plotted.

    Parameters
    ----------
    merged : pd.DataFrame
        Full merged dataset.
    features : list[tuple[str, str, str, bool]]
        Iterable of ``(col, ylabel, description, can_normalise)`` tuples.
    feature_filtered_dfs : dict[str, pd.DataFrame | None]
        Mapping produced by :func:`plot_feature_boxplots`.
    score_col_label : str
        Label for the MDS-UPDRS score variant.
    palette : sequence
        Colour palette.
    output_dir : str
        Directory to write PNG files into.
    save_plots : bool
        Whether to save the figure to *output_dir*.
    show_plots : bool
        Whether to display the figure interactively.
    """
    n_features = len(features)
    if n_features <= 1:
        return

    n_cols = 2
    n_rows = (n_features + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = (
        axes.flatten()
        if n_features > 2
        else ([axes] if n_features == 1 else axes.flatten())
    )

    for idx, (col, ylabel, description, can_norm) in enumerate(features):
        ax = axes[idx]
        data_to_plot = (
            feature_filtered_dfs.get(col)
            if feature_filtered_dfs.get(col) is not None
            else merged
        )
        if len(data_to_plot[col].dropna()) == 0:
            ax.set_visible(False)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sns.boxplot(
                data=data_to_plot,
                x="score_clean",
                y=col,
                hue="score_clean",
                palette=palette,
                legend=False,
                ax=ax,
                showfliers=True,
            )

        ax.set_xlabel(f"MDS-UPDRS {score_col_label} Score", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        title_c = f"{col}"
        if feature_filtered_dfs.get(col) is not None:
            title_c += " (extremes excluded)"
        ax.set_title(title_c, fontsize=11, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Kinematic Features by MDS-UPDRS {score_col_label} Score",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_plots:
        combined_path = os.path.join(
            output_dir, f"boxplots_all_features_by_{score_col_label}.png"
        )
        plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Age / gender adjusted boxplots
# ---------------------------------------------------------------------------


def plot_age_gender_adjusted(
    merged: pd.DataFrame,
    features: list[tuple[str, str, str, bool]],
    adj_feature_map: dict[str, str],
    score_col_label: str,
    palette: Sequence[Any],
    stat_tests: dict[str, Any],
    detect_extreme_outliers_fn: Callable[..., tuple[bool, pd.Series, float, float]],
    extreme_iqr_multiplier: float,
    output_dir: str,
    figsize: tuple[int, int],
    save_plots: bool,
    show_plots: bool,
) -> None:
    """Plot age/gender adjusted boxplots for all features.

    Creates an ``"age_gender_adjusted"`` subdirectory under *output_dir* and
    generates individual boxplots (with optional extreme-outlier-filtered
    variants) as well as a combined multi-panel figure.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged dataset containing ``"score_clean"``, raw feature columns, and
        their adjusted counterparts.
    features : list[tuple[str, str, str, bool]]
        Iterable of ``(col, ylabel, description, can_normalise)`` tuples for
        the *raw* (unadjusted) features.
    adj_feature_map : dict[str, str]
        Mapping from raw column name to adjusted column name.
    score_col_label : str
        Label for the MDS-UPDRS score variant.
    palette : sequence
        Colour palette.
    stat_tests : dict
        Pre-computed statistical tests.
    detect_extreme_outliers_fn : callable
        Function with signature
        ``(series, *, iqr_multiplier) -> (bool, mask, lo, hi)``.
    extreme_iqr_multiplier : float
        Multiplier passed to *detect_extreme_outliers_fn*.
    output_dir : str
        Parent directory; an ``"age_gender_adjusted"`` subdirectory is
        created inside it.
    figsize : tuple[int, int]
        Figure size in inches.
    save_plots : bool
        Whether to save figures.
    show_plots : bool
        Whether to display figures interactively.
    """
    if not adj_feature_map:
        return

    adj_dir = os.path.join(output_dir, "age_gender_adjusted")
    os.makedirs(adj_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("AGE / GENDER ADJUSTED BOXPLOTS")
    print(f"{'=' * 60}")

    adj_feature_filtered_dfs: dict[str, pd.DataFrame | None] = {}

    for col, label, desc, can_norm in features:
        adj_col = adj_feature_map.get(col)
        if adj_col is None or adj_col not in merged.columns:
            continue

        adj_data = merged.dropna(subset=[adj_col])
        if len(adj_data) == 0:
            adj_feature_filtered_dfs[adj_col] = None
            continue

        adj_ylabel = f"{label} (age/gender adjusted)"
        adj_desc = f"{desc} \u2014 adjusted for age and gender"

        fig = create_single_boxplot(
            adj_data, adj_col, adj_ylabel, adj_desc, score_col_label, palette, stat_tests, figsize
        )

        if save_plots:
            safe_col = _safe_col_name(col)
            plot_path = os.path.join(
                adj_dir, f"boxplot_{safe_col}_by_{score_col_label}_age_gender_adjusted.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close()

        has_ext, mask_no_ext, lo, hi = detect_extreme_outliers_fn(
            adj_data[adj_col], iqr_multiplier=extreme_iqr_multiplier
        )

        if has_ext:
            n_ext = int((~mask_no_ext).sum())
            filtered_adj = adj_data[mask_no_ext].copy()
            adj_feature_filtered_dfs[adj_col] = filtered_adj if len(filtered_adj) > 0 else None

            if len(filtered_adj) > 0:
                title_extra = (
                    f"[Excluding {n_ext} extreme values outside [{lo:.1f}, {hi:.1f}]]"
                )
                fig = create_single_boxplot(
                    filtered_adj,
                    adj_col,
                    adj_ylabel,
                    adj_desc,
                    score_col_label,
                    palette,
                    stat_tests,
                    figsize,
                    title_extra,
                )
                if save_plots:
                    safe_col = _safe_col_name(col)
                    plot_path = os.path.join(
                        adj_dir,
                        f"boxplot_{safe_col}_by_{score_col_label}_age_gender_adjusted_no_extremes.png",
                    )
                    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                if show_plots:
                    plt.show()
                else:
                    plt.close()
        else:
            adj_feature_filtered_dfs[adj_col] = None

    # --- Combined adjusted figure ---
    adj_cols_for_plot = [
        (adj_feature_map[col], label, desc, can_norm)
        for col, label, desc, can_norm in features
        if col in adj_feature_map
    ]
    n_adj_features = len(adj_cols_for_plot)
    if n_adj_features <= 1:
        return

    n_cols_fig = 2
    n_rows_fig = (n_adj_features + 1) // 2
    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(14, 5 * n_rows_fig))
    axes = (
        axes.flatten()
        if n_adj_features > 2
        else ([axes] if n_adj_features == 1 else axes.flatten())
    )

    for idx, (adj_col, label, desc, can_norm) in enumerate(adj_cols_for_plot):
        ax = axes[idx]
        data_to_plot = (
            adj_feature_filtered_dfs.get(adj_col)
            if adj_feature_filtered_dfs.get(adj_col) is not None
            else merged.dropna(subset=[adj_col])
        )
        if len(data_to_plot[adj_col].dropna()) == 0:
            ax.set_visible(False)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sns.boxplot(
                data=data_to_plot,
                x="score_clean",
                y=adj_col,
                hue="score_clean",
                palette=palette,
                legend=False,
                ax=ax,
                showfliers=True,
            )

        orig_col = adj_col[:-4] if adj_col.endswith("_adj") else adj_col
        ax.set_xlabel(f"MDS-UPDRS {score_col_label} Score", fontsize=10)
        ax.set_ylabel(f"{label} (adj.)", fontsize=10)
        title_adj = f"{orig_col} (age/gender adj.)"
        if adj_feature_filtered_dfs.get(adj_col) is not None:
            title_adj += " (extremes excl.)"
        ax.set_title(title_adj, fontsize=11, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_adj_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Kinematic Features by MDS-UPDRS {score_col_label} Score\n"
        "(Adjusted for Age and Gender)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_plots:
        combined_adj_path = os.path.join(
            adj_dir, f"boxplots_all_features_by_{score_col_label}_age_gender_adjusted.png"
        )
        plt.savefig(combined_adj_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
