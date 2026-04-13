"""Subgroup analysis helpers: age-group binning, gender normalisation, statistical tests."""

import numpy as np
import pandas as pd

from ._constants import AGE_BINS, AGE_LABELS


def age_group_labels(age_series: "pd.Series") -> "pd.Series":
    """Bin numeric ages into three clinical groups: <60, 60-69, 70+."""
    return pd.cut(
        pd.to_numeric(age_series, errors="coerce"),
        bins=AGE_BINS,
        right=False,
        labels=AGE_LABELS,
    ).astype(object)


def safe_group_label(gval) -> str:
    """Return a filesystem-safe string for a subgroup label."""
    return (
        str(gval)
        .replace(" ", "_")
        .replace("<", "lt")
        .replace("\u2265", "ge")
        .replace(">=", "ge")
        .replace("+", "plus")
        .replace("-", "_")
    )


def normalize_gender_series(s: "pd.Series") -> "pd.Series":
    """Normalise heterogeneous Gender values to 'Male' / 'Female' / NaN.

    Handles:
    - Numeric codes stored by ``load_age_gender``: 1 / 1.0 -> Male, 2 / 2.0 -> Female
    - Common string variants: M/F, Male/Female, Man/Woman
    """
    mapping = {
        "m": "Male",
        "male": "Male",
        "man": "Male",
        "1": "Male",
        "f": "Female",
        "female": "Female",
        "woman": "Female",
        "2": "Female",
    }
    normalised = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\.0$", "", regex=True)
    )
    return normalised.map(mapping)


def subgroup_delta_stats(deltas: "pd.Series", groups: "pd.Series", prefix: str) -> dict:
    """Return per-subgroup delta statistics as a flat dict.

    Output keys follow the pattern ``<stat>_<prefix>_<safe_group>``.
    Subgroups with zero valid observations are still reported (as NaN).
    """
    out: dict = {}
    valid_groups = groups.dropna()
    for gval in sorted(valid_groups.unique(), key=str):
        safe = safe_group_label(gval)
        key = f"{prefix}_{safe}"
        sub = deltas[groups == gval].dropna()
        n = len(sub)
        out[f"n_{key}"] = int(n)
        out[f"delta_mean_{key}"] = float(sub.mean()) if n else np.nan
        out[f"delta_median_{key}"] = float(sub.median()) if n else np.nan
        out[f"delta_std_{key}"] = float(sub.std(ddof=1)) if n >= 2 else np.nan
        if n >= 2:
            _std = float(sub.std(ddof=1))
            out[f"effect_size_{key}"] = float(sub.mean() / _std) if _std > 0 else np.nan
        else:
            out[f"effect_size_{key}"] = np.nan
    return out


def effect_size_from_deltas(deltas: "pd.Series") -> float:
    """Effect size: mean(delta) / std(delta, ddof=1), or NaN if < 2 valid pairs."""
    vals = pd.to_numeric(deltas, errors="coerce").dropna()
    if len(vals) < 2:
        return np.nan
    std = float(vals.std(ddof=1))
    return float(vals.mean() / std) if std > 0 else np.nan


def wilcoxon_p(deltas: "pd.Series") -> float:
    """Two-sided Wilcoxon signed-rank p-value against zero, or NaN if < 4 valid pairs."""
    vals = pd.to_numeric(deltas, errors="coerce").dropna()
    if len(vals) < 4:
        return np.nan
    try:
        from scipy.stats import wilcoxon as _wilcoxon

        _, p = _wilcoxon(vals, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


def ttest_rel_p(a: "pd.Series", b: "pd.Series") -> float:
    """Two-sided paired t-test p-value, or NaN if < 4 valid pairs."""
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    valid = a_num.notna() & b_num.notna()
    if valid.sum() < 4:
        return np.nan
    try:
        from scipy.stats import ttest_rel as _ttest_rel

        _, p = _ttest_rel(a_num[valid], b_num[valid])
        return float(p)
    except Exception:
        return np.nan
