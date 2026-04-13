"""Feature computation for kinematic analysis.

Extracted from the orchestrator in ``scripts/analyze.py`` (lines 3593-3715,
3998-4056).  Each public function takes a merged DataFrame, adds derived
columns in-place, and returns the same DataFrame so calls can be chained.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ._constants import COMPOSITE_COMPONENTS, COMPOSITE_V2_COMPONENTS, KINEMATIC_FEATURES


# ---------------------------------------------------------------------------
# 1. Composite Score (equal-weight Z-score)
# ---------------------------------------------------------------------------

def compute_composite_score(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute the *Composite Score* column via equal-weight Z-scoring.

    For each component in :pydata:`COMPOSITE_COMPONENTS` the Z-score is
    computed relative to the ``score_clean == 0`` (healthy-control)
    population.  The composite is the mean of the (sign-flipped) Z-scores.

    Parameters
    ----------
    merged:
        DataFrame that must contain a ``score_clean`` column and the
        feature columns listed in ``COMPOSITE_COMPONENTS``.

    Returns
    -------
    pd.DataFrame
        *merged* with a ``"Composite Score"`` column added (when all
        components are available).
    """
    z_tmp_cols: list[str] = []
    composite_ok = True
    score0_mask = merged["score_clean"] == 0

    for feat, sign in COMPOSITE_COMPONENTS:
        if feat not in merged.columns:
            print(f"  Composite Score: missing column '{feat}', skipping composite.")
            composite_ok = False
            break

        mu0 = float(merged.loc[score0_mask, feat].mean())
        sd0 = float(merged.loc[score0_mask, feat].std(ddof=1))

        if not np.isfinite(sd0) or sd0 == 0.0:
            print(
                f"  Composite Score: zero/NaN std for '{feat}' in score-0 group, skipping."
            )
            composite_ok = False
            break

        ztmp = f"__z_{feat}__"
        merged[ztmp] = sign * (merged[feat] - mu0) / sd0
        z_tmp_cols.append(ztmp)

    if composite_ok and z_tmp_cols:
        merged["Composite Score"] = merged[z_tmp_cols].mean(axis=1)
        n_comp = int(merged["Composite Score"].notna().sum())
        print(f"  Composite Score computed for {n_comp} rows")
    elif z_tmp_cols:
        print("  Composite Score skipped (missing components).")

    merged.drop(
        columns=[c for c in z_tmp_cols if c in merged.columns],
        inplace=True,
        errors="ignore",
    )

    return merged


# ---------------------------------------------------------------------------
# 2. Interaction features
# ---------------------------------------------------------------------------

def compute_interaction_features(merged: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Compute interaction / derived features.

    Features added (when their inputs are present):

    * **Movement Vigor** -- ``Mean Amplitude * Peak Velocity / 100``
    * **Amp-Vel Product** -- ``Mean Amplitude * Mean Velocity / 100``
    * **Fatigue Index** -- clipped sum of amplitude and velocity decrement
    * **Irregularity Burden** -- amplitude CV weighted by pause and
      hesitation burden

    Parameters
    ----------
    merged:
        DataFrame with the required source columns.

    Returns
    -------
    tuple[pd.DataFrame, int]
        A tuple of (*merged* with new columns, count of features added).
    """
    n_interaction = 0

    if "Mean Amplitude" in merged.columns and "Peak Velocity" in merged.columns:
        merged["Movement Vigor"] = (
            merged["Mean Amplitude"] * merged["Peak Velocity"] / 100.0
        )
        n_interaction += 1

    if "Mean Amplitude" in merged.columns and "Mean Velocity" in merged.columns:
        merged["Amp-Vel Product"] = (
            merged["Mean Amplitude"] * merged["Mean Velocity"] / 100.0
        )
        n_interaction += 1

    if "Amp Decrement %" in merged.columns and "Velocity Decrement %" in merged.columns:
        amp_dec = merged["Amp Decrement %"].clip(lower=0).fillna(0)
        vel_dec = merged["Velocity Decrement %"].clip(lower=0).fillna(0)
        merged["Fatigue Index"] = amp_dec + vel_dec
        n_interaction += 1

    if (
        "Amplitude CV" in merged.columns
        and "Pause Time Ratio" in merged.columns
        and "Total Cycles" in merged.columns
    ):
        hes = merged.get("Num Hesitations", pd.Series(0, index=merged.index))
        tc = merged["Total Cycles"].replace(0, np.nan)
        merged["Irregularity Burden"] = (
            merged["Amplitude CV"]
            * (1.0 + merged["Pause Time Ratio"])
            * (1.0 + hes / tc)
        )
        n_interaction += 1

    return merged, n_interaction


# ---------------------------------------------------------------------------
# 3. Clinical Composite v2 (weighted Z-score)
# ---------------------------------------------------------------------------

def compute_clinical_composite_v2(merged: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Compute *Clinical Composite v2* using a weighted Z-score approach.

    Components and weights are defined in
    :pydata:`COMPOSITE_V2_COMPONENTS`.  Z-scores are referenced to the
    ``score_clean == 0`` group (minimum 3 subjects required).

    Parameters
    ----------
    merged:
        DataFrame that must contain a ``score_clean`` column.

    Returns
    -------
    tuple[pd.DataFrame, int]
        A tuple of (*merged* with ``"Clinical Composite v2"`` column
        added when at least 3 components are available, count of
        composite features added -- 0 or 1).
    """
    z_v2_cols: list[str] = []
    z_v2_weights: list[float] = []
    n_added = 0

    score0_mask = merged["score_clean"] == 0
    n_s0 = int(score0_mask.sum())

    if n_s0 >= 3:
        for feat, sign, weight in COMPOSITE_V2_COMPONENTS:
            if feat not in merged.columns:
                continue

            vals = merged.loc[score0_mask, feat].dropna()
            if len(vals) < 3:
                continue

            mu0 = float(vals.mean())
            sd0 = float(vals.std(ddof=1))
            if not np.isfinite(sd0) or sd0 < 1e-9:
                continue

            ztmp = f"__zv2_{feat}__"
            merged[ztmp] = sign * (merged[feat] - mu0) / sd0
            z_v2_cols.append(ztmp)
            z_v2_weights.append(weight)

        if len(z_v2_cols) >= 3:
            w_arr = np.array(z_v2_weights, dtype=float)
            w_arr = w_arr / w_arr.sum()
            merged["Clinical Composite v2"] = sum(
                merged[c] * w for c, w in zip(z_v2_cols, w_arr)
            )
            n_cv2 = int(merged["Clinical Composite v2"].notna().sum())
            print(
                f"  Clinical Composite v2 computed for {n_cv2} rows "
                f"({len(z_v2_cols)} components)"
            )
            n_added = 1

    merged.drop(
        columns=[c for c in z_v2_cols if c in merged.columns],
        inplace=True,
        errors="ignore",
    )

    return merged, n_added


# ---------------------------------------------------------------------------
# 4. Feature availability filter
# ---------------------------------------------------------------------------

def filter_available_features(
    merged: pd.DataFrame,
) -> List[Tuple[str, str, str, bool]]:
    """Return the subset of ``KINEMATIC_FEATURES`` whose column exists in *merged*.

    Parameters
    ----------
    merged:
        DataFrame to check column presence against.

    Returns
    -------
    list[tuple[str, str, str, bool]]
        Each element is ``(column_name, label, description, can_normalize)``
        for features present in *merged*.
    """
    available_features: list[tuple[str, str, str, bool]] = [
        (col, label, desc, can_norm)
        for col, label, desc, can_norm in KINEMATIC_FEATURES
        if col in merged.columns
    ]
    return available_features
