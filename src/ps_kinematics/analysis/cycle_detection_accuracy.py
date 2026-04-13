"""Cycle detection accuracy: bijective tolerance-based matching vs manual annotations."""

import json

import numpy as np
import pandas as pd

from ps_kinematics.io import (
    normalize_video_path_series_for_matching,
    read_xlsx_stdlib,
)
from ._filtering import (
    apply_video_quality_filter,
    load_recording_angle_labels,
    load_video_quality_labels,
    normalize_recording_angle_filter,
    segment_inclusion_mask,
)


def _parse_json_array(s) -> list[float]:
    """Parse a JSON-serialised array from a CSV cell into a list of floats."""
    if pd.isna(s) or s == "":
        return []
    try:
        arr = json.loads(s)
        return [float(x) for x in arr]
    except (json.JSONDecodeError, TypeError):
        return []


def _nn_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each element in *a*, return the distance to the nearest element in *b*."""
    return np.abs(a[:, None] - b[None, :]).min(axis=1)


def _bijective_match(
    det: np.ndarray, man: np.ndarray, tol: float
) -> tuple[int, int, int, float]:
    """Greedy bijective matching: closest pairs within *tol* assigned first.

    Returns (tp, fp, fn, mae_s) where mae_s is mean timing error of matched
    pairs (NaN when tp=0).
    """
    if len(det) == 0:
        return 0, 0, len(man), float("nan")
    if len(man) == 0:
        return 0, len(det), 0, float("nan")

    # Build all candidate pairs within tolerance, sort by distance.
    det_idx, man_idx = np.where(np.abs(det[:, None] - man[None, :]) <= tol)
    dists = np.abs(det[det_idx] - man[man_idx])
    order = np.argsort(dists, kind="stable")
    det_idx, man_idx, dists = det_idx[order], man_idx[order], dists[order]

    used_det: set[int] = set()
    used_man: set[int] = set()
    matched_errors: list[float] = []
    for di, mi, d in zip(det_idx, man_idx, dists):
        di, mi = int(di), int(mi)
        if di not in used_det and mi not in used_man:
            used_det.add(di)
            used_man.add(mi)
            matched_errors.append(float(d))

    tp = len(matched_errors)
    fp = len(det) - tp
    fn = len(man) - tp
    mae_s = float(np.mean(matched_errors)) if matched_errors else float("nan")
    return tp, fp, fn, mae_s


def _merge_extrema_lists(series: pd.Series) -> list[float]:
    """Combine multiple extrema lists into one sorted, deduplicated list."""
    combined: list[float] = []
    for lst in series:
        combined.extend(lst)
    return sorted(set(round(t, 6) for t in combined))


def compute_cycle_detection_accuracy(
    tracking_csv_path: str,
    validation_csv_path: str,
    tolerance_s: float = 0.16,
    recording_angle_csv_path: str = None,
    selected_recording_angles: list[str] | None = None,
    video_quality_labels_csv_path: str = None,
    video_quality_threshold: int = 3,
    segmented_only: bool = False,
    non_segmented: bool = False,
) -> dict:
    """Measure accuracy of automatically detected extrema against manual annotations.

    Uses **bijective tolerance-based matching** as the primary metric.

    A detected event is a True Positive (TP) if it falls within ``tolerance_s``
    of an unmatched manual event (greedy closest-pair assignment).  Each manual
    event and each detected event can be used in at most one match, so duplicates
    near the same ground-truth event are counted as False Positives.

    Primary metrics (bijective matching)
    -------------------------------------
    **Precision** = TP / (TP + FP)
    **Recall**    = TP / (TP + FN)
    **F1**        = 2 * P * R / (P + R)
    **MAE (matched)** mean |det_time - man_time| for TP pairs (seconds)

    Supplementary set-distance metrics (diagnostic only)
    -----------------------------------------------------
    Wasserstein-1 distance, symmetric nearest-neighbour distance, Hausdorff
    distance, count ratio and count difference.

    Parameters
    ----------
    tracking_csv_path : str
        Path to pipeline output CSV with ``cycle_peak_times_s`` and
        ``cycle_trough_times_s`` columns (JSON arrays, seconds).
    validation_csv_path : str
        Path to manual annotation file (.csv or .xlsx) with a ``video_path``
        join key and extrema times.
    tolerance_s : float
        Maximum timing error (seconds) for a detection to count as a TP.
    recording_angle_csv_path : str, optional
        Path to angle annotations CSV.
    selected_recording_angles : list[str] | None
        Optional allow-list of recording-angle labels to include.
    video_quality_labels_csv_path : str, optional
        Path to manual video-quality labels CSV.
    video_quality_threshold : int
        Keep only videos with manual ``quality_label`` <= this value.
    segmented_only : bool
        If True, include only videos whose name/path contains "segmented".
    non_segmented : bool
        If True, include only videos whose name/path does NOT contain "segmented".

    Returns
    -------
    dict
        Aggregate + per-video scores.
    """
    from scipy.stats import wasserstein_distance

    tracking = pd.read_csv(tracking_csv_path)
    selected_angles_norm = normalize_recording_angle_filter(selected_recording_angles)
    if selected_angles_norm is not None and recording_angle_csv_path is None:
        raise ValueError(
            "selected_recording_angles was provided but recording_angle_csv_path is None"
        )

    _vpath = str(validation_csv_path)
    if _vpath.lower().endswith((".xlsx", ".xls")):
        validation = read_xlsx_stdlib(_vpath)
    else:
        validation = pd.read_csv(_vpath)

    if "video_path" not in tracking.columns:
        print("WARNING: Tracking CSV has no 'video_path' column.")
        return {}
    if "video_path" not in validation.columns:
        print("WARNING: Validation file has no 'video_path' column.")
        return {}

    if segmented_only or non_segmented:
        n_tracking_before = len(tracking)
        n_validation_before = len(validation)
        tracking_mask = segment_inclusion_mask(
            tracking["video_path"],
            segmented_only=segmented_only,
            non_segmented=non_segmented,
        )
        validation_mask = segment_inclusion_mask(
            validation["video_path"],
            segmented_only=segmented_only,
            non_segmented=non_segmented,
        )
        tracking = tracking[tracking_mask].copy()
        validation = validation[validation_mask].copy()
        print(
            "  Segmentation-name filter "
            f"(segmented_only={segmented_only}, non_segmented={non_segmented}): "
            f"tracking {len(tracking)}/{n_tracking_before}, "
            f"validation {len(validation)}/{n_validation_before}"
        )

    if video_quality_threshold not in (1, 2, 3):
        raise ValueError("video_quality_threshold must be one of: 1, 2, 3")
    if video_quality_labels_csv_path is None and video_quality_threshold < 3:
        raise ValueError("video_quality_threshold < 3 requires video_quality_labels_csv_path")
    if video_quality_labels_csv_path is not None:
        print(f"  Loading manual video quality labels from: {video_quality_labels_csv_path}")
        quality_df = load_video_quality_labels(video_quality_labels_csv_path)
        print(f"  Video quality labels loaded: {len(quality_df)} unique videos")
        tracking = apply_video_quality_filter(
            tracking,
            quality_df,
            video_quality_threshold,
            video_path_column="video_path",
            stage_name="cycle validation tracking",
        )
        validation = apply_video_quality_filter(
            validation,
            quality_df,
            video_quality_threshold,
            video_path_column="video_path",
            stage_name="cycle validation labels",
        )

    tracking["_join_key"] = normalize_video_path_series_for_matching(tracking["video_path"])
    validation["_join_key"] = normalize_video_path_series_for_matching(validation["video_path"])

    if selected_angles_norm is not None:
        angle_df = load_recording_angle_labels(recording_angle_csv_path)
        allowed_angle_keys = set(
            angle_df.loc[
                angle_df["recording_angle"].isin(selected_angles_norm),
                "_angle_key",
            ].tolist()
        )
        n_tracking_before = len(tracking)
        n_validation_before = len(validation)
        tracking = tracking[tracking["_join_key"].isin(allowed_angle_keys)].copy()
        validation = validation[validation["_join_key"].isin(allowed_angle_keys)].copy()
        print(
            "  Recording-angle filter "
            f"{sorted(selected_angles_norm)} applied pre-processing: "
            f"tracking {len(tracking)}/{n_tracking_before}, "
            f"validation {len(validation)}/{n_validation_before}"
        )

    _key_counts = tracking["_join_key"].value_counts(dropna=False)
    _n_collision_keys = int((_key_counts > 1).sum())
    if _n_collision_keys > 0:
        _extra_rows = int((_key_counts[_key_counts > 1] - 1).sum())
        print(
            "  WARNING: join-key collisions detected in tracking data: "
            f"{_n_collision_keys} keys, {_extra_rows} extra rows before collapsing"
        )

    if "cycle_peak_times_s" not in tracking.columns:
        print("WARNING: No 'cycle_peak_times_s' column in tracking CSV.")
        return {}

    tracking["_det_peaks"] = tracking["cycle_peak_times_s"].apply(_parse_json_array)
    if "cycle_trough_times_s" in tracking.columns:
        tracking["_det_troughs"] = tracking["cycle_trough_times_s"].apply(_parse_json_array)
    else:
        tracking["_det_troughs"] = [[] for _ in range(len(tracking))]

    tracking["_det_extrema"] = [
        sorted(p + t) for p, t in zip(tracking["_det_peaks"], tracking["_det_troughs"])
    ]

    n_tracking_rows = len(tracking)
    _ps_trim_cols = [
        c for c in ["ps_start_frame", "ps_end_frame", "fps", "ps_trimmed"] if c in tracking.columns
    ]
    if _ps_trim_cols:
        _ps_info = (
            tracking[["_join_key"] + _ps_trim_cols]
            .drop_duplicates("_join_key")
            .set_index("_join_key")
        )
    else:
        _ps_info = None

    tracking = tracking.groupby("_join_key", as_index=False).agg(
        {"_det_extrema": _merge_extrema_lists, "video_path": "first"}
    )

    if _ps_info is not None:
        for _c in _ps_trim_cols:
            tracking[_c] = tracking["_join_key"].map(_ps_info[_c])

    n_collapsed = len(tracking)
    print(
        f"\n  Tracking rows: {n_tracking_rows} -> {n_collapsed} unique video_path entries "
        f"(peaks + troughs combined per video)"
    )

    if "peak_times_s" in validation.columns:
        validation["_man_peaks"] = validation["peak_times_s"].apply(_parse_json_array)
    elif "extrema_times_s" in validation.columns:
        validation["_man_peaks"] = validation["extrema_times_s"].apply(_parse_json_array)
    else:
        peak_cols = sorted(
            [c for c in validation.columns if c.startswith("peak_") or c.startswith("extrema_")]
        )
        if peak_cols:
            validation["_man_peaks"] = validation[peak_cols].apply(
                lambda row: sorted([float(v) for v in row if pd.notna(v)]), axis=1
            )
        else:
            print(
                "WARNING: Validation file has no 'peak_times_s'/'extrema_times_s' "
                "or 'peak_N'/'extrema_N' columns."
            )
            return {}

    merged = tracking.merge(
        validation[["_join_key", "_man_peaks", "video_path"]].rename(
            columns={"video_path": "_val_video_path"}
        ),
        on="_join_key",
        how="inner",
    )

    print(
        f"\nCycle-Detection Accuracy ({len(merged)} matched videos, "
        f"tolerance={tolerance_s:.3f}s, bijective matching)"
    )
    print("-" * 60)

    results: list[dict] = []
    for _, row in merged.iterrows():
        det = np.array(sorted(row["_det_extrema"]), dtype=float)
        man = np.array(sorted(row["_man_peaks"]), dtype=float)
        if len(man) == 0:
            continue

        nan = float("nan")

        # Exclude manual points outside the trimmed PS-active window
        n_man_excluded = 0
        _ps_flag = row.get("ps_trimmed") if "ps_trimmed" in row.index else None
        if _ps_flag and str(_ps_flag).lower() not in ("false", "nan", "none", "", "0"):
            _sf = row.get("ps_start_frame") if "ps_start_frame" in row.index else None
            _ef = row.get("ps_end_frame") if "ps_end_frame" in row.index else None
            _fps_row = (
                float(row.get("fps", 25.0))
                if "fps" in row.index and pd.notna(row.get("fps"))
                else 25.0
            )
            if _sf is not None and _ef is not None and pd.notna(_sf) and pd.notna(_ef):
                ps_start_s_row = float(_sf) / _fps_row
                ps_end_s_row = float(_ef) / _fps_row
                n_man_before = len(man)
                man = man[(man >= ps_start_s_row) & (man <= ps_end_s_row)]
                n_man_excluded = n_man_before - len(man)
        if len(man) == 0:
            continue

        # Primary: bijective tolerance-based matching
        tp, fp, fn, mae_s = _bijective_match(det, man, tolerance_s)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Supplementary: set-distance metrics (diagnostic)
        if len(det) > 0:
            w_dist = float(wasserstein_distance(det, man))
            d2m = _nn_distances(det, man)
            m2d = _nn_distances(man, det)
            nn_det_to_man_s = float(d2m.mean())
            nn_man_to_det_s = float(m2d.mean())
            sym_nn_s = (nn_det_to_man_s + nn_man_to_det_s) / 2.0
            hausdorff_s = float(max(d2m.max(), m2d.max()))
        else:
            w_dist = float(np.mean(man))
            nn_det_to_man_s = nn_man_to_det_s = sym_nn_s = hausdorff_s = nan

        results.append(
            {
                "video_path": row["video_path"],
                "n_detected": int(len(det)),
                "n_manual": int(len(man)),
                "n_manual_excluded": int(n_man_excluded),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mae_s": round(mae_s, 4) if not np.isnan(mae_s) else nan,
                "count_ratio": round(len(det) / len(man), 4) if len(man) else nan,
                "count_diff": int(len(det)) - int(len(man)),
                "wasserstein_s": round(w_dist, 4),
                "nn_det_to_man_s": (
                    round(nn_det_to_man_s, 4) if not np.isnan(nn_det_to_man_s) else nan
                ),
                "nn_man_to_det_s": (
                    round(nn_man_to_det_s, 4) if not np.isnan(nn_man_to_det_s) else nan
                ),
                "sym_nn_s": round(sym_nn_s, 4) if not np.isnan(sym_nn_s) else nan,
                "hausdorff_s": round(hausdorff_s, 4) if not np.isnan(hausdorff_s) else nan,
            }
        )

    if not results:
        print("  No videos with manual annotations matched.")
        return {}

    res_df = pd.DataFrame(results)

    def _nanmean(col: str) -> float:
        return float(res_df[col].dropna().mean()) if res_df[col].notna().any() else float("nan")

    agg = {
        "n_videos": len(res_df),
        "tolerance_s": tolerance_s,
        # Primary (bijective)
        "mean_precision": _nanmean("precision"),
        "mean_recall": _nanmean("recall"),
        "mean_f1": _nanmean("f1"),
        "mean_mae_s": _nanmean("mae_s"),
        "mean_mae_frames": _nanmean("mae_s") * 25.0,
        # Supplementary (set-distance)
        "mean_wasserstein_s": _nanmean("wasserstein_s"),
        "mean_sym_nn_s": _nanmean("sym_nn_s"),
        "mean_nn_det_to_man_s": _nanmean("nn_det_to_man_s"),
        "mean_nn_man_to_det_s": _nanmean("nn_man_to_det_s"),
        "mean_hausdorff_s": _nanmean("hausdorff_s"),
        "mean_count_ratio": _nanmean("count_ratio"),
        "mean_count_diff": _nanmean("count_diff"),
    }

    print(f"  Videos evaluated:            {agg['n_videos']}  (tolerance={tolerance_s:.3f}s)")
    print(f"  Mean F1:                     {agg['mean_f1']:.4f}  (1.0 = perfect)")
    print(f"  Mean Precision:              {agg['mean_precision']:.4f}")
    print(f"  Mean Recall:                 {agg['mean_recall']:.4f}")
    if not np.isnan(agg["mean_mae_s"]):
        print(
            f"  Mean MAE (matched pairs):    {agg['mean_mae_s']:.4f} s  "
            f"({agg['mean_mae_frames']:.2f} frames at 25fps)"
        )
    print("  -- Supplementary set-distance metrics --")
    print(f"  Mean Wasserstein distance:   {agg['mean_wasserstein_s']:.4f} s")
    print(f"  Mean symmetric NN distance:  {agg['mean_sym_nn_s']:.4f} s")
    print(f"  Mean Hausdorff distance:     {agg['mean_hausdorff_s']:.4f} s")
    print(f"  Mean count ratio:            {agg['mean_count_ratio']:.3f}  (1.0 = exact count)")
    print(f"  Mean count diff:             {agg['mean_count_diff']:+.1f}  (0 = exact count)")

    worst = res_df.nsmallest(min(5, len(res_df)), "f1")
    if len(worst) > 0:
        print("\n  Worst F1 videos:")
        for _, r in worst.iterrows():
            print(f"    {r['video_path']}")
            print(
                f"      F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
                f"MAE={r['mae_s']:.3f}s  "
                f"({r['n_detected']} det / {r['n_manual']} manual, "
                f"TP={r['tp']} FP={r['fp']} FN={r['fn']})"
            )

    agg["per_video"] = results
    return agg
