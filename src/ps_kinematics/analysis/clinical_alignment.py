"""Clinical alignment scoring for automated optimization."""

import numpy as np
import pandas as pd

from ps_kinematics.io import (
    coerce_int_score,
    load_videoid_to_patientid_map,
    normalize_hand,
    normalize_med_state,
    normalize_visit_to_int,
    parse_hand_from_path,
    parse_ids_and_visit,
    parse_medication_state_from_path,
)
from ._constants import EXPECTED_DIRECTIONS


def _within_score_dispersion_ratio(valid, feature_col, score_col="score_clean"):
    groups = []
    for _, g in valid.groupby(score_col):
        arr = pd.to_numeric(g[feature_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(arr) >= 2:
            groups.append(arr)
    if len(groups) < 2:
        return None
    n_total = int(sum(len(g) for g in groups))
    k = int(len(groups))
    if n_total <= k:
        return None
    ss_within = float(sum(np.sum((g - np.mean(g)) ** 2) for g in groups))
    pooled_within_var = ss_within / float(n_total - k)
    all_vals = np.concatenate(groups)
    total_var = float(np.var(all_vals, ddof=1))
    if (not np.isfinite(total_var)) or total_var <= 1e-12:
        return None
    ratio = float(np.sqrt(max(0.0, pooled_within_var) / total_var))
    if not np.isfinite(ratio):
        return None
    return float(np.clip(ratio, 0.0, 2.0))


def _group_median_monotonicity(valid, feature_col, direction, score_col="score_clean"):
    """Kendall \u03c4 between score group ordinals and group medians, signed by expected direction.

    Requires \u22653 groups each with \u22652 observations.
    Returns (signed_tau, raw_tau, tau_p, n_groups).
    Returns (None, None, None, n_groups) when data are insufficient.
    """
    try:
        from scipy.stats import kendalltau
    except ImportError:
        return None, None, None, 0

    group_scores: list = []
    group_medians: list = []
    for score_val, g in valid.groupby(score_col):
        arr = pd.to_numeric(g[feature_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(arr) >= 2:
            group_scores.append(float(score_val))
            group_medians.append(float(np.median(arr)))
    n_groups = len(group_scores)
    if n_groups < 3:
        return None, None, None, n_groups
    tau, p_val = kendalltau(group_scores, group_medians)
    if tau is None or not np.isfinite(tau):
        return None, None, float(p_val) if p_val is not None else None, n_groups
    signed_tau = float(tau) * float(direction)
    return float(signed_tau), float(tau), float(p_val), n_groups


def _within_to_between_spread_ratio(valid, feature_col, score_col="score_clean"):
    """Ratio of pooled within-group std to between-group std (std of group medians).

    Lower is better: within-group noise is small relative to between-group signal.
    Returns None when insufficient data, otherwise a float clipped to [0, 5].
    """
    groups: list = []
    group_medians: list = []
    for _, g in valid.groupby(score_col):
        arr = pd.to_numeric(g[feature_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(arr) >= 2:
            groups.append(arr)
            group_medians.append(float(np.median(arr)))
    if len(groups) < 2:
        return None
    n_total = int(sum(len(g) for g in groups))
    k = int(len(groups))
    if n_total <= k:
        return None
    ss_within = float(sum(np.sum((g - np.mean(g)) ** 2) for g in groups))
    pooled_within_std = float(np.sqrt(ss_within / float(n_total - k)))
    between_std = float(np.std(group_medians, ddof=1)) if len(group_medians) >= 2 else 0.0
    if not np.isfinite(between_std) or between_std < 1e-12:
        return None
    ratio = pooled_within_std / between_std
    if not np.isfinite(ratio):
        return None
    return float(np.clip(ratio, 0.0, 5.0))


def _empty_alignment_result(reason: str):
    return {
        "composite_score": -999.0,
        "signed_rho_sum": 0.0,
        "dispersion_penalty_sum": 0.0,
        "dispersion_penalty_weight": 0.0,
        "per_feature": {
            f: {
                "rho": None,
                "p": None,
                "n": 0,
                "direction": d,
                "dispersion_ratio": None,
                "dispersion_penalty": 0.0,
                "contribution": 0.0,
            }
            for f, d in EXPECTED_DIRECTIONS.items()
        },
        "n_matched": 0,
        "n_with_features": 0,
        "n_contributing_features": 0,
        "error": reason,
    }


def compute_clinical_alignment_score(
    kinematics_csv_path: str,
    score_csv_path: str,
    id2vid_csv_path: str,
    score_column: str = "ProS",
    dispersion_penalty_weight: float = 0.20,
    monotonicity_weight: float = 0.30,
    within_group_cv_penalty_weight: float = 0.20,
    verbose: bool = False,
):
    """Compute how well extracted features align with clinical expectations.

    Per-feature contribution:
        signed_rho (Spearman, direction-adjusted)
      + monotonicity_weight x signed_tau  (Kendall tau on group medians)
      - dispersion_penalty_weight x dispersion_ratio (pooled within/total variance)
      - within_group_cv_penalty_weight x spread_ratio (within-group std / between-group std)
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        raise RuntimeError("scipy is required for compute_clinical_alignment_score")

    kin = pd.read_csv(kinematics_csv_path)
    if "visit" not in kin.columns:
        kin["visit"] = pd.NA
    kin["visit"] = kin["visit"].apply(normalize_visit_to_int)
    try:
        kin["visit"] = kin["visit"].astype("Int64")
    except Exception:
        pass

    if "ids" not in kin.columns:
        if "video_path" in kin.columns:
            kin["ids"] = [parse_ids_and_visit(vp)[0] for vp in kin["video_path"].tolist()]
        else:
            return _empty_alignment_result("No 'ids' or 'video_path' column")

    if "medication_state" not in kin.columns:
        if "log_medication_state" in kin.columns:
            kin["medication_state"] = kin["log_medication_state"]
        else:
            kin["medication_state"] = [
                parse_medication_state_from_path(vp)
                for vp in kin.get("video_path", [None] * len(kin))
            ]
    kin["medication_state"] = kin["medication_state"].apply(normalize_med_state)

    if "hand" not in kin.columns:
        if "log_hand" in kin.columns:
            kin["hand"] = kin["log_hand"]
        else:
            kin["hand"] = [
                parse_hand_from_path(vp) for vp in kin.get("video_path", [None] * len(kin))
            ]
    kin["hand"] = kin["hand"].apply(normalize_hand)

    video_to_patient = load_videoid_to_patientid_map(id2vid_csv_path)
    kin["video_ids"] = kin["ids"].astype(str)
    kin["ids"] = kin["video_ids"].map(video_to_patient)
    kin = kin.dropna(subset=["ids"]).copy()
    if kin.empty:
        return _empty_alignment_result("No rows after mapping")

    scores_df = pd.read_csv(score_csv_path, sep=None, engine="python")
    if "visit" in scores_df.columns:
        scores_df["visit"] = scores_df["visit"].apply(normalize_visit_to_int)
        try:
            scores_df["visit"] = scores_df["visit"].astype("Int64")
        except Exception:
            pass
    scores_df["hand"] = scores_df["hand"].apply(normalize_hand)
    scores_df["medication_state"] = scores_df["medication_state"].apply(normalize_med_state)

    KEY_COLS = ["ids", "visit", "medication_state", "hand"]
    missing_keys = [c for c in KEY_COLS if c not in kin.columns or c not in scores_df.columns]
    if missing_keys:
        return _empty_alignment_result(f"Missing key columns: {missing_keys}")

    merged = kin.merge(scores_df[KEY_COLS + [score_column]], on=KEY_COLS, how="inner")
    if merged.empty:
        return _empty_alignment_result("Merge produced 0 rows")

    merged["score_clean"] = merged[score_column].apply(coerce_int_score)
    merged = merged.dropna(subset=["score_clean"]).copy()
    merged["score_clean"] = merged["score_clean"].astype(int)

    if verbose:
        print(f"  Alignment eval \u2013 merged rows: {len(merged)}")

    per_feature = {}
    composite = 0.0
    signed_rho_sum = 0.0
    dispersion_penalty_sum = 0.0
    monotonicity_sum = 0.0
    spread_penalty_sum = 0.0
    n_contributing = 0

    for feat, direction in EXPECTED_DIRECTIONS.items():
        if feat not in merged.columns:
            per_feature[feat] = {
                "rho": None,
                "p": None,
                "n": 0,
                "direction": direction,
                "contribution": 0.0,
            }
            continue
        valid = merged[[feat, "score_clean"]].dropna()
        n = len(valid)
        if n < 5:
            per_feature[feat] = {
                "rho": None,
                "p": None,
                "n": n,
                "direction": direction,
                "dispersion_ratio": None,
                "dispersion_penalty": 0.0,
                "tau": None,
                "tau_p": None,
                "n_groups": 0,
                "spread_ratio": None,
                "spread_penalty": 0.0,
                "contribution": 0.0,
            }
            continue
        rho, p_val = spearmanr(valid["score_clean"], valid[feat])
        if rho is None or not np.isfinite(rho):
            per_feature[feat] = {
                "rho": None,
                "p": float(p_val) if p_val is not None else None,
                "n": n,
                "direction": direction,
                "dispersion_ratio": None,
                "dispersion_penalty": 0.0,
                "tau": None,
                "tau_p": None,
                "n_groups": 0,
                "spread_ratio": None,
                "spread_penalty": 0.0,
                "contribution": 0.0,
            }
            continue
        signed_rho = float(rho) * float(direction)
        dispersion_ratio = _within_score_dispersion_ratio(valid, feat, "score_clean")
        dispersion_penalty = (
            float(dispersion_penalty_weight) * float(dispersion_ratio)
            if (dispersion_ratio is not None and dispersion_penalty_weight > 0)
            else 0.0
        )
        signed_tau, raw_tau, tau_p, n_groups = _group_median_monotonicity(
            valid, feat, direction, "score_clean"
        )
        tau_contribution = (
            float(monotonicity_weight) * float(signed_tau)
            if (signed_tau is not None and monotonicity_weight > 0)
            else 0.0
        )
        spread_ratio = _within_to_between_spread_ratio(valid, feat, "score_clean")
        spread_penalty = (
            float(within_group_cv_penalty_weight) * float(spread_ratio)
            if (spread_ratio is not None and within_group_cv_penalty_weight > 0)
            else 0.0
        )
        contribution = signed_rho + tau_contribution - dispersion_penalty - spread_penalty
        composite += contribution
        signed_rho_sum += signed_rho
        dispersion_penalty_sum += dispersion_penalty
        monotonicity_sum += tau_contribution
        spread_penalty_sum += spread_penalty
        n_contributing += 1
        per_feature[feat] = {
            "rho": float(rho),
            "p": float(p_val),
            "n": n,
            "direction": direction,
            "dispersion_ratio": float(dispersion_ratio) if dispersion_ratio is not None else None,
            "dispersion_penalty": float(dispersion_penalty),
            "tau": float(raw_tau) if raw_tau is not None else None,
            "tau_p": float(tau_p) if tau_p is not None else None,
            "n_groups": n_groups,
            "spread_ratio": float(spread_ratio) if spread_ratio is not None else None,
            "spread_penalty": float(spread_penalty),
            "contribution": contribution,
        }
        if verbose:
            tau_str = f"{signed_tau:+.4f}" if signed_tau is not None else "N/A"
            spread_str = f"{spread_ratio:.3f}" if spread_ratio is not None else "N/A"
            print(
                f"  {feat}: signed_rho={signed_rho:+.4f}  "
                f"signed_tau={tau_str}  "
                f"spread_ratio={spread_str}  contrib={contribution:+.4f}"
            )

    feat_cols = [f for f in EXPECTED_DIRECTIONS if f in merged.columns]
    n_with_features = int(merged[feat_cols].notna().any(axis=1).sum()) if feat_cols else 0

    return {
        "composite_score": composite,
        "signed_rho_sum": signed_rho_sum,
        "dispersion_penalty_sum": dispersion_penalty_sum,
        "monotonicity_sum": monotonicity_sum,
        "spread_penalty_sum": spread_penalty_sum,
        "dispersion_penalty_weight": float(dispersion_penalty_weight),
        "monotonicity_weight": float(monotonicity_weight),
        "within_group_cv_penalty_weight": float(within_group_cv_penalty_weight),
        "per_feature": per_feature,
        "n_matched": len(merged),
        "n_with_features": n_with_features,
        "n_contributing_features": n_contributing,
    }
