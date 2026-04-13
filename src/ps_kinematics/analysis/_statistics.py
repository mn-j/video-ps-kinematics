"""Statistical computation logic for kinematic feature analysis.

Provides functions for descriptive statistics, hypothesis testing,
cycle-count correlation checks, and effect-size computation.

Data-only outputs: each function returns a dict of results and prints
human-readable summaries to stdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._constants import EXPECTED_DIRECTIONS


# ── helpers ────────────────────────────────────────────────────────────────

def _significance_label(p: float) -> str:
    """Return a significance-level string for a *p*-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _effect_label(d: float) -> str:
    """Return a Cohen-style magnitude label for an effect size."""
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "LARGE"


# ── 1. Descriptive statistics ──────────────────────────────────────────────

def compute_summary_statistics(
    merged: pd.DataFrame,
    available_features: list[tuple[str, str, str, bool]],
) -> dict:
    """Compute per-score-group descriptive statistics for each feature.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged kinematic + clinical DataFrame with a ``score_clean`` column.
    available_features : list[tuple[str, str, str, bool]]
        Each element is ``(column_name, ylabel, description, can_normalize)``.

    Returns
    -------
    dict
        ``{column_name: DataFrame}`` where each value is a pandas DataFrame
        with rows = score groups and columns =
        ``count, mean, std, median, min, max``.
    """
    summary_stats: dict[str, pd.DataFrame] = {}
    for col, ylabel, _, can_norm in available_features:
        if col not in merged.columns:
            continue
        print(f"\n{col}:")
        stats = merged.groupby("score_clean")[col].agg(
            ["count", "mean", "std", "median", "min", "max"]
        )
        print(stats.to_string())
        summary_stats[col] = stats
    return summary_stats


# ── 2. Statistical tests ──────────────────────────────────────────────────

def compute_statistical_tests(
    merged: pd.DataFrame,
    available_features: list[tuple[str, str, str, bool]],
) -> dict:
    """Run Spearman, Kruskal-Wallis, ANCOVA, and Bonferroni post-hoc tests.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged kinematic + clinical DataFrame with a ``score_clean`` column
        and optional ``age`` / ``sex`` columns used as ANCOVA covariates.
    available_features : list[tuple[str, str, str, bool]]
        Each element is ``(column_name, ylabel, description, can_normalize)``.

    Returns
    -------
    dict
        ``{column_name: {...test results...}}`` with keys such as
        ``spearman_rho``, ``spearman_p``, ``kruskal_H``, ``kruskal_p``,
        ``ancova_F``, ``ancova_p``, ``posthoc_bonferroni``, and ``n``.
    """
    # ------------------------------------------------------------------
    # Optional dependency checks
    # ------------------------------------------------------------------
    try:
        from scipy.stats import (
            kruskal,
            mannwhitneyu,
            shapiro,
            spearmanr,
            ttest_ind,
        )
        _HAS_SCIPY_STATS = True
    except ImportError:
        _HAS_SCIPY_STATS = False

    try:
        import statsmodels.formula.api as smf
        import statsmodels.stats.anova as sma
        _HAS_STATSMODELS = True
    except ImportError:
        _HAS_STATSMODELS = False

    stat_tests: dict[str, dict] = {}

    for col, ylabel, _, can_norm in available_features:
        if col not in merged.columns:
            continue
        if not _HAS_SCIPY_STATS:
            continue

        valid = merged[[col, "score_clean"]].dropna()
        if len(valid) < 5:
            continue

        # ── Spearman correlation ──────────────────────────────────────
        rho, p_spearman = spearmanr(valid["score_clean"], valid[col])

        # ── Kruskal-Wallis ────────────────────────────────────────────
        groups = [
            g[col].dropna().values
            for _, g in valid.groupby("score_clean")
            if len(g[col].dropna()) >= 2
        ]
        if len(groups) >= 2:
            h_stat, p_kruskal = kruskal(*groups)
        else:
            h_stat, p_kruskal = float("nan"), float("nan")

        stat_tests[col] = {
            "spearman_rho": rho,
            "spearman_p": p_spearman,
            "kruskal_H": h_stat,
            "kruskal_p": p_kruskal,
            "n": len(valid),
        }

        sig_sp = _significance_label(p_spearman)
        sig_kw = _significance_label(p_kruskal)
        print(f"  Spearman rho={rho:+.4f}, p={p_spearman:.2e} ({sig_sp})")
        print(f"  Kruskal-Wallis H={h_stat:.2f}, p={p_kruskal:.2e} ({sig_kw})")

        # ── ANCOVA (optional) ─────────────────────────────────────────
        _ancova_p = float("nan")
        _ancova_F = float("nan")
        if _HAS_STATSMODELS:
            _has_age = "age" in merged.columns
            _has_sex = "sex" in merged.columns
            _ancova_cols = [col, "score_clean"]
            if _has_age:
                _ancova_cols.append("age")
            if _has_sex:
                _ancova_cols.append("sex")
            _anc_df = merged[_ancova_cols].dropna().copy()
            _anc_df["score_clean"] = _anc_df["score_clean"].astype("category")

            _safe_col = (
                col.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("%", "pct")
                .replace("-", "_")
                .replace("/", "_")
            )
            _anc_df = _anc_df.rename(columns={col: _safe_col})

            _covariates: list[str] = []
            if _has_age and "age" in _anc_df.columns:
                _covariates.append("age")
            if _has_sex and "sex" in _anc_df.columns:
                _anc_df["sex"] = _anc_df["sex"].astype("category")
                _covariates.append("C(sex)")

            _formula = _safe_col + " ~ C(score_clean)"
            if _covariates:
                _formula += " + " + " + ".join(_covariates)

            try:
                _anc_model = smf.ols(_formula, data=_anc_df).fit()
                _anc_table = sma.anova_lm(_anc_model, typ=2)
                _anc_row = (
                    _anc_table.loc["C(score_clean)"]
                    if "C(score_clean)" in _anc_table.index
                    else None
                )
                if _anc_row is not None:
                    _ancova_F = float(_anc_row.get("F", float("nan")))
                    _ancova_p = float(_anc_row.get("PR(>F)", float("nan")))
            except Exception:
                pass

            sig_anc = _significance_label(_ancova_p)
            _adj_note = " [age+sex adjusted]" if _covariates else ""
            print(
                f"  ANCOVA F={_ancova_F:.2f}, p={_ancova_p:.2e} ({sig_anc})"
                + _adj_note
            )
            stat_tests[col]["ancova_F"] = _ancova_F
            stat_tests[col]["ancova_p"] = _ancova_p

        # ── Bonferroni-corrected post-hoc ─────────────────────────────
        _scores_sorted = sorted(valid["score_clean"].unique())
        _ref_score = _scores_sorted[0]
        _other_scores = [s for s in _scores_sorted if s != _ref_score]
        _n_comparisons = len(_other_scores)

        if _n_comparisons >= 1:
            _ref_vals = valid.loc[
                valid["score_clean"] == _ref_score, col
            ].values
            _posthoc: dict[str, dict] = {}

            for _s in _other_scores:
                _cmp_vals = valid.loc[
                    valid["score_clean"] == _s, col
                ].values
                if len(_ref_vals) < 3 or len(_cmp_vals) < 3:
                    continue

                _sw_ref_p = (
                    shapiro(_ref_vals)[1] if len(_ref_vals) >= 3 else 0.0
                )
                _sw_cmp_p = (
                    shapiro(_cmp_vals)[1] if len(_cmp_vals) >= 3 else 0.0
                )
                _both_normal = _sw_ref_p > 0.05 and _sw_cmp_p > 0.05

                if _both_normal:
                    _, _raw_p = ttest_ind(_ref_vals, _cmp_vals)
                    _test_name = "t-test"
                else:
                    _, _raw_p = mannwhitneyu(
                        _ref_vals, _cmp_vals, alternative="two-sided"
                    )
                    _test_name = "MWU"

                _bonf_p = min(1.0, float(_raw_p) * _n_comparisons)
                _sig = _significance_label(_bonf_p)
                _posthoc[f"{_ref_score}v{_s}"] = {
                    "p_raw": float(_raw_p),
                    "p_bonferroni": _bonf_p,
                    "test": _test_name,
                    "sig": _sig,
                }
                print(
                    f"  Post-hoc {_ref_score} vs {_s} ({_test_name}): "
                    f"p_raw={_raw_p:.2e}, p_bonf={_bonf_p:.2e} ({_sig})"
                )

            stat_tests[col]["posthoc_bonferroni"] = _posthoc

    return stat_tests


# ── 3. CV-cycle correlation check ─────────────────────────────────────────

def check_cv_cycle_correlation(merged: pd.DataFrame) -> None:
    """Warn if Amplitude CV is inflated in videos with low cycle counts.

    Prints a diagnostic summary. If a significant negative Spearman
    correlation (rho < -0.15, *p* < 0.05) is found, a warning is issued.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged kinematic DataFrame with optional ``Amplitude CV`` and
        ``Total Cycles`` columns.
    """
    if "Amplitude CV" not in merged.columns or "Total Cycles" not in merged.columns:
        return

    from scipy.stats import spearmanr

    _cv_cyc = merged[["Amplitude CV", "Total Cycles"]].dropna()
    if len(_cv_cyc) < 10:
        return

    _rho_cv, _p_cv = spearmanr(_cv_cyc["Total Cycles"], _cv_cyc["Amplitude CV"])
    print(f"\n{'─' * 50}")
    print("CV vs Cycle Count check:")
    print(f"  Spearman rho = {_rho_cv:+.3f}, p = {_p_cv:.2e}")
    if _p_cv < 0.05 and _rho_cv < -0.15:
        print("  WARNING: Significant negative correlation detected!")
        print("    Amplitude CV is inflated in low-cycle videos.")
    else:
        print("  No problematic correlation detected.")


# ── 4. Effect sizes ───────────────────────────────────────────────────────

def compute_effect_sizes(
    merged: pd.DataFrame,
    available_features: list[tuple[str, str, str, bool]],
    expected_directions: dict[str, int] | None = None,
) -> dict:
    """Compute Cohen's d, Glass's delta, and direction checks per feature.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged kinematic + clinical DataFrame with a ``score_clean`` column.
    available_features : list[tuple[str, str, str, bool]]
        Each element is ``(column_name, ylabel, description, can_normalize)``.
    expected_directions : dict[str, int] | None
        Mapping of feature name to expected sign of change (+1 / -1) with
        increasing clinical severity.  Falls back to
        ``_constants.EXPECTED_DIRECTIONS`` when *None*.

    Returns
    -------
    dict
        ``{column_name: {comparison_key: cohen_d, ...}}`` with additional
        keys ``glass_delta`` where applicable.
    """
    if expected_directions is None:
        expected_directions = EXPECTED_DIRECTIONS

    effect_sizes: dict[str, dict] = {}

    for col, ylabel, _, can_norm in available_features:
        if col not in merged.columns:
            continue

        valid = merged[[col, "score_clean"]].dropna()
        scores_present = sorted(valid["score_clean"].unique())
        if len(scores_present) < 2:
            continue

        print(f"\n{col}:")
        col_effects: dict[str, float] = {}

        # ── Adjacent-score Cohen's d ──────────────────────────────────
        for i in range(len(scores_present) - 1):
            s_lo, s_hi = scores_present[i], scores_present[i + 1]
            g_lo = valid.loc[valid["score_clean"] == s_lo, col].values
            g_hi = valid.loc[valid["score_clean"] == s_hi, col].values
            if len(g_lo) >= 2 and len(g_hi) >= 2:
                pooled_std = np.sqrt(
                    (
                        (len(g_lo) - 1) * np.var(g_lo, ddof=1)
                        + (len(g_hi) - 1) * np.var(g_hi, ddof=1)
                    )
                    / (len(g_lo) + len(g_hi) - 2)
                )
                d = (
                    (np.mean(g_hi) - np.mean(g_lo)) / pooled_std
                    if pooled_std > 1e-9
                    else float("nan")
                )
                label = _effect_label(d)
                col_effects[f"{s_lo}v{s_hi}"] = d
                print(
                    f"  Score {s_lo} vs {s_hi}: d = {d:+.3f} ({label}), "
                    f"n = {len(g_lo)}+{len(g_hi)}"
                )

        # ── Extreme contrast: score 0 vs highest ─────────────────────
        if 0 in scores_present and scores_present[-1] > 0:
            g0 = valid.loc[valid["score_clean"] == 0, col].values
            g_max = valid.loc[
                valid["score_clean"] == scores_present[-1], col
            ].values
            if len(g0) >= 2 and len(g_max) >= 2:
                pooled_std = np.sqrt(
                    (
                        (len(g0) - 1) * np.var(g0, ddof=1)
                        + (len(g_max) - 1) * np.var(g_max, ddof=1)
                    )
                    / (len(g0) + len(g_max) - 2)
                )
                d_extreme = (
                    (np.mean(g_max) - np.mean(g0)) / pooled_std
                    if pooled_std > 1e-9
                    else float("nan")
                )
                label = _effect_label(d_extreme)
                col_effects[f"0v{scores_present[-1]}"] = d_extreme
                print(
                    f"  Score 0 vs {scores_present[-1]} (extreme): "
                    f"d = {d_extreme:+.3f} ({label})"
                )

                # Glass's delta (reference = score 0)
                sd0 = np.std(g0, ddof=1)
                if sd0 > 1e-9:
                    glass = (np.mean(g_max) - np.mean(g0)) / sd0
                    label = _effect_label(glass)
                    print(f"  Glass's delta (ref=score 0): {glass:+.3f} ({label})")
                    col_effects["glass_delta"] = glass

        # ── Direction check ───────────────────────────────────────────
        expected = expected_directions.get(col, None)
        if expected is not None and f"0v{scores_present[-1]}" in col_effects:
            d_val = col_effects[f"0v{scores_present[-1]}"]
            actual_dir = np.sign(d_val) if not np.isnan(d_val) else 0
            if actual_dir != 0 and actual_dir != expected:
                dir_word = "increase" if expected > 0 else "decrease"
                obs_word = "increase" if actual_dir > 0 else "decrease"
                print(
                    f"  WRONG DIRECTION: expected {dir_word} with severity, "
                    f"observed {obs_word}"
                )
            elif actual_dir == expected:
                dir_word = "increase" if expected > 0 else "decrease"
                print(f"  Correct direction ({dir_word} with severity)")

        effect_sizes[col] = col_effects

    return effect_sizes
