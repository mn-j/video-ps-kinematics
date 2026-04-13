"""Orchestrator: create_kinematic_boxplots_by_score.

Thin entry-point that delegates to extracted sub-modules for data loading,
filtering, feature computation, plotting, and statistical analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ps_kinematics.io import (
    coerce_int_score,
    detect_extreme_outliers,
    load_age_gender,
    load_videoid_to_patientid_map,
    normalize_hand,
    normalize_med_state,
    normalize_video_path_series_for_matching,
    normalize_visit_to_int,
    parse_hand_from_path,
    parse_ids_and_visit,
    parse_medication_state_from_path,
    residualize_for_age_gender,
)
from ._constants import EXPECTED_DIRECTIONS, KINEMATIC_FEATURES
from ._features import (
    compute_clinical_composite_v2,
    compute_composite_score,
    compute_interaction_features,
    filter_available_features,
)
from ._filtering import (
    apply_video_quality_filter,
    load_recording_angle_labels,
    load_video_quality_labels,
    normalize_recording_angle_filter,
    segment_inclusion_mask,
)
from ._merge import load_updrs_total_wide
from ._plotting import (
    plot_age_gender_adjusted,
    plot_combined_figure,
    plot_feature_boxplots,
)
from ._statistics import (
    check_cv_cycle_correlation,
    compute_effect_sizes,
    compute_statistical_tests,
    compute_summary_statistics,
)
from .longitudinal import ensure_visit_from_pom_token, run_longitudinal_progression_report
from .medication import run_medicine_effect_report as _run_medicine_effect_report
from .pca_varimax import run_pca_varimax_analysis
from .recording_angle import run_recording_angle_analysis as _run_recording_angle_analysis
from .signal_diagnostics import (
    plot_amplitude_vs_cycle_time_scatter,
    plot_confidence_video_averages,
    plot_handedness_confidence,
    plot_signal_quality_distribution,
)
from .video_quality_factors import plot_video_quality_factors


def create_kinematic_boxplots_by_score(
    kinematics_csv_path: str,
    score_csv_path: str,
    id2vid_csv_path: str,
    score_column: str = "ProS",
    output_dir: str = None,
    figsize: tuple = (12, 8),
    save_plots: bool = True,
    show_plots: bool = True,
    normalize_by_cycles: bool = False,
    extreme_iqr_multiplier: float = 3.0,
    age_gender_csv_path: str = None,
    signal_quality_threshold: float = 0.0,
    signal_quality_sub_thresholds: dict = None,
    min_cycles: int = 0,
    min_quality_cycles: int = 0,
    min_inter_mcp_span_px: float = 0.0,
    min_detection_rate: float = 0.0,
    updrs_total_csv_path: str = None,
    recording_angle_csv_path: str = None,
    selected_recording_angles: list[str] | None = None,
    video_quality_labels_csv_path: str = None,
    video_quality_threshold: int = 3,
    run_per_video_diagnostic_plots: bool = True,
    run_video_quality_analysis: bool = True,
    run_recording_angle_analysis: bool = True,
    run_longitudinal_report: bool = True,
    run_medicine_effect_report: bool = True,
    run_age_gender_adjustment: bool = True,
    segmented_only: bool = False,
    non_segmented: bool = False,
):
    """Create box plots for each kinematic feature grouped by MDS-UPDRS score.

    Parameters
    ----------
    kinematics_csv_path : str
        Path to the CSV output from batch processing containing kinematic features.
    score_csv_path : str
        Path to final_scores_summary.csv containing MDS-UPDRS scores.
    id2vid_csv_path : str
        Path to id2vid.csv mapping patient_id -> video_ids.
    score_column : str
        Column name for the MDS-UPDRS score to use (default: "ProS").
    output_dir : str
        Directory to save output plots. If None, uses directory of kinematics_csv.
    figsize : tuple
        Figure size for individual plots.
    save_plots : bool
        Whether to save plots to files.
    show_plots : bool
        Whether to display plots interactively.
    normalize_by_cycles : bool
        Deprecated -- kept for backward compatibility.
    extreme_iqr_multiplier : float
        Multiplier for IQR to detect extreme outliers (default: 3.0).
    age_gender_csv_path : str, optional
        Path to age_gender.csv for age/gender covariate adjustment.
    signal_quality_threshold : float
        Minimum per-video Signal Quality score (0-1).
    signal_quality_sub_thresholds : dict, optional
        Per-sub-score minimum thresholds.
    min_cycles : int
        Minimum number of detected total cycles.
    min_quality_cycles : int
        Minimum number of detected quality cycles.
    min_inter_mcp_span_px : float
        Minimum median index-to-pinky MCP pixel distance.
    min_detection_rate : float
        Minimum keypoint detection rate (0-1).
    updrs_total_csv_path : str, optional
        Path to wide-format UPDRS Part-III total scores CSV.
    recording_angle_csv_path : str, optional
        Path to recording angle annotations CSV.
    selected_recording_angles : list[str] | None
        Optional allow-list of recording-angle labels.
    video_quality_labels_csv_path : str, optional
        Path to manual video-quality labels CSV.
    video_quality_threshold : int
        Keep only videos with quality_label <= this value.
    run_per_video_diagnostic_plots : bool
        Whether to generate per-video diagnostic plots.
    run_video_quality_analysis : bool
        Whether to run video-quality-factor analysis.
    run_recording_angle_analysis : bool
        Whether to run recording-angle effect analysis.
    run_longitudinal_report : bool
        Whether to run longitudinal progression report.
    run_medicine_effect_report : bool
        Whether to run medication-effect analysis.
    run_age_gender_adjustment : bool
        Whether to compute age/gender-adjusted features.
    segmented_only : bool
        If True, include only videos whose name/path contains "segmented".
    non_segmented : bool
        If True, include only videos whose name/path does NOT contain "segmented".

    Returns
    -------
    dict : Summary statistics and merged dataframe.
    """
    if output_dir is None:
        output_dir = os.path.dirname(kinematics_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    if not show_plots:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    # ── Load kinematics data ─────────────────────────────────────────────
    print(f"Loading kinematics from: {kinematics_csv_path}")
    kin = pd.read_csv(kinematics_csv_path)
    print(f"  Total kinematic records: {len(kin)}")

    # Keep only VIDEO records
    if "record_type" in kin.columns:
        n_before = len(kin)
        kin = kin[kin["record_type"] == "VIDEO"].copy()
        n_dropped = n_before - len(kin)
        if n_dropped:
            print(
                f"  record_type filter: kept {len(kin)}/{n_before} VIDEO records "
                f"({n_dropped} non-VIDEO rows dropped)"
            )

    # Detection rate filter
    if min_detection_rate > 0.0:
        if "VQ_detection_rate" not in kin.columns:
            print(
                "  WARNING: min_detection_rate > 0 but 'VQ_detection_rate' column "
                "not found in CSV; skipping detection rate filter."
            )
        else:
            det = pd.to_numeric(kin["VQ_detection_rate"], errors="coerce").fillna(0.0)
            n_before = len(kin)
            kin = kin[det >= min_detection_rate].copy()
            print(
                f"  Detection rate filter (>= {min_detection_rate:.2f}): "
                f"kept {len(kin)}/{n_before} ({n_before - len(kin)} dropped)"
            )

    # Segmented/non-segmented filter
    if segmented_only or non_segmented:
        _name_col = None
        for _candidate in ["video_path", "video_ids", "ids"]:
            if _candidate in kin.columns:
                _name_col = _candidate
                break
        if _name_col is None:
            print("  WARNING: segmented filter requested but no video naming column found; skipping.")
        else:
            n_before = len(kin)
            seg_mask = segment_inclusion_mask(
                kin[_name_col], segmented_only=segmented_only, non_segmented=non_segmented
            )
            kin = kin[seg_mask].copy()
            print(
                f"  Segmentation-name filter ({_name_col}): "
                f"kept {len(kin)}/{n_before} ({n_before - len(kin)} dropped)"
            )

    # Video quality threshold validation
    if video_quality_threshold not in (1, 2, 3):
        raise ValueError("video_quality_threshold must be one of: 1, 2, 3")
    if video_quality_labels_csv_path is None and video_quality_threshold < 3:
        raise ValueError("video_quality_threshold < 3 requires video_quality_labels_csv_path")

    quality_df = None
    if video_quality_labels_csv_path is not None:
        print(f"\nLoading manual video quality labels from: {video_quality_labels_csv_path}")
        quality_df = load_video_quality_labels(video_quality_labels_csv_path)
        print(f"  Video quality labels loaded: {len(quality_df)} unique videos")

    # ── Merge recording angle labels ─────────────────────────────────────
    selected_angles_norm = normalize_recording_angle_filter(selected_recording_angles)
    if selected_angles_norm is not None and recording_angle_csv_path is None:
        raise ValueError(
            "selected_recording_angles was provided but recording_angle_csv_path is None"
        )

    if recording_angle_csv_path is not None:
        print(f"\nLoading recording angle labels from: {recording_angle_csv_path}")
        angle_df = load_recording_angle_labels(recording_angle_csv_path)
        print(f"  Recording angle labels loaded: {len(angle_df)} rows")

        if "video_path" in kin.columns:
            kin["_angle_key"] = normalize_video_path_series_for_matching(kin["video_path"])
            _before = len(kin)
            kin = kin.merge(
                angle_df[["_angle_key", "recording_angle"]].drop_duplicates(subset="_angle_key"),
                on="_angle_key",
                how="left",
            )
            kin.drop(columns=["_angle_key"], inplace=True)
            n_matched = int(kin["recording_angle"].notna().sum())
            print(f"  Recording angle matched: {n_matched}/{_before} videos")
            if n_matched > 0:
                for _aval in sorted(kin["recording_angle"].dropna().unique()):
                    print(f"    {_aval}: {int((kin['recording_angle'] == _aval).sum())}")
        elif "ids" in kin.columns:
            kin["_angle_key"] = kin["ids"].astype(str).str.strip().str.lower()
            kin = kin.merge(
                angle_df[["_angle_key", "recording_angle"]].drop_duplicates(subset="_angle_key"),
                on="_angle_key",
                how="left",
            )
            if "_angle_key" in kin.columns:
                kin.drop(columns=["_angle_key"], inplace=True)
            n_matched = int(kin["recording_angle"].notna().sum())
            print(f"  Recording angle matched (by video_id): {n_matched}/{len(kin)} videos")
        else:
            print("  WARNING: no 'video_path' or 'ids' column for angle merge")

    if selected_angles_norm is not None:
        if "recording_angle" not in kin.columns:
            raise RuntimeError(
                "selected_recording_angles requires recording angle labels merged into data"
            )
        n_before = len(kin)
        kin = kin[kin["recording_angle"].isin(selected_angles_norm)].copy()
        print(
            f"  Recording-angle filter {sorted(selected_angles_norm)}: "
            f"kept {len(kin)}/{n_before} ({n_before - len(kin)} dropped)"
        )

    if quality_df is not None:
        kin = apply_video_quality_filter(
            kin, quality_df, video_quality_threshold,
            video_path_column="video_path", stage_name="main analysis",
        )

    # Inter-MCP span filter
    if min_inter_mcp_span_px > 0.0:
        if "VQ_inter_mcp_span_px" not in kin.columns:
            print("  WARNING: 'VQ_inter_mcp_span_px' column not found; skipping span filter")
        else:
            span_vals = pd.to_numeric(kin["VQ_inter_mcp_span_px"], errors="coerce").fillna(0.0)
            n_before = len(kin)
            kin = kin[span_vals >= min_inter_mcp_span_px].copy()
            print(
                f"  Inter-MCP span filter (>= {min_inter_mcp_span_px:.1f} px): "
                f"kept {len(kin)}/{n_before} ({n_before - len(kin)} dropped)"
            )

    # ── Signal Quality distribution + filtering ──────────────────────────
    plot_signal_quality_distribution(
        kin, output_dir, signal_quality_threshold,
        save_plots=save_plots, show_plots=show_plots,
    )
    if signal_quality_threshold > 0.0 and "Signal Quality" in kin.columns:
        sq = pd.to_numeric(kin["Signal Quality"], errors="coerce").fillna(0.0)
        n_before = len(kin)
        kin = kin[sq >= signal_quality_threshold].copy()
        print(
            f"  Signal Quality threshold {signal_quality_threshold:.2f}: "
            f"kept {len(kin)}/{n_before} ({n_before - len(kin)} dropped)"
        )

    # Per-sub-score signal quality filters
    if signal_quality_sub_thresholds:
        for sub_key, sub_thresh in signal_quality_sub_thresholds.items():
            col = sub_key if sub_key.startswith("SQ_") else f"SQ_{sub_key}"
            if col not in kin.columns:
                print(f"  WARNING: sub-score column '{col}' not found; skipping")
                continue
            sub_vals = pd.to_numeric(kin[col], errors="coerce").fillna(0.0)
            n_before = len(kin)
            kin = kin[sub_vals >= float(sub_thresh)].copy()
            print(
                f"  SQ sub-score '{col}' threshold {sub_thresh:.2f}: "
                f"kept {len(kin)}/{n_before} ({n_before - len(kin)} dropped)"
            )

    # Quality cycles filter
    if min_quality_cycles > 0:
        if "Quality Cycles" not in kin.columns:
            print("  WARNING: 'Quality Cycles' column not found; skipping quality-cycle filter")
        else:
            qcounts = pd.to_numeric(kin["Quality Cycles"], errors="coerce").fillna(0.0)
            n_before = len(kin)
            kin = kin[qcounts >= float(min_quality_cycles)].copy()
            print(
                f"  Quality cycles filter (>= {min_quality_cycles}): "
                f"kept {len(kin)}/{n_before} ({n_before - len(kin)} dropped)"
            )

    # Total cycles filter
    if min_cycles > 0:
        if "Total Cycles" not in kin.columns:
            print("  WARNING: 'Total Cycles' column not found; skipping total-cycle filter")
        else:
            tcounts = pd.to_numeric(kin["Total Cycles"], errors="coerce").fillna(0.0)
            n_before = len(kin)
            kin = kin[tcounts >= float(min_cycles)].copy()
            print(
                f"  Total cycles filter (>= {min_cycles}): "
                f"kept {len(kin)}/{n_before} ({n_before - len(kin)} dropped)"
            )

    # ── ID parsing and visit extraction ──────────────────────────────────
    if "ids" not in kin.columns:
        if "video_path" not in kin.columns:
            raise RuntimeError("Kinematics CSV is missing both 'ids' and 'video_path'.")
        kin["ids"] = [parse_ids_and_visit(vp)[0] for vp in kin["video_path"].tolist()]

    kin = ensure_visit_from_pom_token(kin)
    if kin["visit"].isna().all():
        raise RuntimeError(
            "Could not determine any visit values. "
            "Expected 'visit' column or POMx token in ids/video_path."
        )

    # ── Load scores ──────────────────────────────────────────────────────
    print(f"Loading scores from: {score_csv_path}")
    scores_df = pd.read_csv(score_csv_path, sep=None, engine="python")
    print(f"  Total score records: {len(scores_df)}")
    if "visit" in scores_df.columns:
        scores_df["visit"] = scores_df["visit"].apply(normalize_visit_to_int)
        try:
            scores_df["visit"] = scores_df["visit"].astype("Int64")
        except Exception:
            pass

    # ── Load video-to-patient mapping ────────────────────────────────────
    print(f"Loading id2vid mapping from: {id2vid_csv_path}")
    video_to_patient = load_videoid_to_patientid_map(id2vid_csv_path)
    print(f"  Total video->patient mappings: {len(video_to_patient)}")

    # Ensure medication_state column
    if "medication_state" not in kin.columns:
        if "log_medication_state" in kin.columns:
            kin["medication_state"] = kin["log_medication_state"]
        else:
            kin["medication_state"] = [
                parse_medication_state_from_path(vp)
                for vp in kin.get("video_path", [None] * len(kin))
            ]
    kin["medication_state"] = kin["medication_state"].apply(normalize_med_state)

    # Ensure hand column
    if "hand" not in kin.columns:
        if "log_hand" in kin.columns:
            kin["hand"] = kin["log_hand"]
        else:
            kin["hand"] = [
                parse_hand_from_path(vp) for vp in kin.get("video_path", [None] * len(kin))
            ]
    kin["hand"] = kin["hand"].apply(normalize_hand)

    # Map video IDs to patient IDs
    kin["video_ids"] = kin["ids"].astype(str)
    kin["ids"] = kin["video_ids"].map(video_to_patient)
    mapped_ok = kin["ids"].notna().mean()
    n_mapping_dropped = int(kin["ids"].isna().sum())
    print(f"\nVideo->Patient mapping success: {mapped_ok * 100:.1f}%")
    print(f"  Dropped (unmapped): {n_mapping_dropped} rows")
    if 0 < n_mapping_dropped < 20:
        print("  Unmapped video_ids:")
        print(kin.loc[kin["ids"].isna(), "video_ids"].value_counts().head(10))
    kin = kin.dropna(subset=["ids"]).copy()

    # ── Merge with scores ────────────────────────────────────────────────
    scores = scores_df.copy()
    scores["hand"] = scores["hand"].apply(normalize_hand)
    scores["medication_state"] = scores["medication_state"].apply(normalize_med_state)

    KEY_COLS = ["ids", "visit", "medication_state", "hand"]
    missing_keys = [c for c in KEY_COLS if c not in kin.columns or c not in scores.columns]
    if missing_keys:
        raise RuntimeError(f"Missing key columns for merge: {missing_keys}")

    merged = kin.merge(scores[KEY_COLS + [score_column]], on=KEY_COLS, how="inner")
    print(f"\nMerge results:")
    print(f"  Kinematic rows after mapping: {len(kin)}")
    print(f"  Matched with scores: {len(merged)}")
    print(f"  Unmatched: {len(kin) - len(merged)}")

    if merged.empty:
        print("\nWARNING: Merge produced 0 rows!")
        return {"merged_df": merged, "n_matched": 0}

    merged["score_clean"] = merged[score_column].apply(coerce_int_score)
    merged = merged.dropna(subset=["score_clean"]).copy()
    merged["score_clean"] = merged["score_clean"].astype(int)
    print(f"  After score cleaning: {len(merged)} rows")

    # Video-level snapshot before aggregation
    merged_video_level = merged.copy()

    # ── Per-video diagnostic plots ───────────────────────────────────────
    if run_per_video_diagnostic_plots:
        print("\nPlotting per-video MCP landmark confidence summaries...")
        plot_confidence_video_averages(
            merged_video_level, output_dir=output_dir,
            save_plots=save_plots, show_plots=show_plots,
        )
        print("\nPlotting per-video handedness confidence...")
        plot_handedness_confidence(
            merged_video_level, output_dir=output_dir,
            save_plots=save_plots, show_plots=show_plots,
        )

    print("\nPlotting amplitude vs cycle duration scatter...")
    plot_amplitude_vs_cycle_time_scatter(
        merged_video_level, output_dir=output_dir, score_column=score_column,
        save_plots=save_plots, show_plots=show_plots,
    )

    # Video quality factor analysis
    _vq_cols_exist = any(c.startswith("VQ_") for c in merged_video_level.columns)
    _has_angle_col = "recording_angle" in merged_video_level.columns
    if run_video_quality_analysis and (_vq_cols_exist or _has_angle_col):
        print("\nAnalysing video quality factors...")
        plot_video_quality_factors(
            merged_video_level, output_dir=output_dir,
            save_plots=save_plots, show_plots=show_plots,
        )

    # ── Age/gender adjustment ────────────────────────────────────────────
    has_age_gender = False
    if run_age_gender_adjustment and age_gender_csv_path is not None:
        print(f"\nLoading age/gender data from: {age_gender_csv_path}")
        ag = load_age_gender(age_gender_csv_path)
        print(f"  Age/gender records loaded: {len(ag)}")
        _TMP = "__patient_id_ag__"
        merged[_TMP] = merged["video_ids"].map(video_to_patient)
        merged = merged.merge(ag.rename(columns={"ids": _TMP}), on=_TMP, how="left").drop(
            columns=[_TMP]
        )
        n_with_ag = int(merged[["age", "Gender"]].notna().all(axis=1).sum())
        print(
            f"  Rows with complete age+gender: {n_with_ag}/{len(merged)} "
            f"({100 * n_with_ag / len(merged):.1f}%)"
        )
        has_age_gender = n_with_ag >= 5

    # ── Same-condition median aggregation ────────────────────────────────
    _AGG_KEYS = ["ids", "visit", "medication_state", "hand"]
    _numeric_cols = merged.select_dtypes(include="number").columns.tolist()
    _n_before = len(merged)
    merged = (
        merged.groupby(_AGG_KEYS, dropna=False)
        .agg(
            {c: "median" if c in _numeric_cols else "first" for c in merged.columns if c not in _AGG_KEYS}
        )
        .reset_index()
    )
    print(f"\n  Same-condition median aggregation: {_n_before} -> {len(merged)} rows")

    # ── Composite scores and interaction features ────────────────────────
    merged = compute_composite_score(merged)

    print("\n  Computing interaction features...")
    merged, _n_interaction = compute_interaction_features(merged)
    merged, _n_cv2 = compute_clinical_composite_v2(merged)
    _n_interaction += _n_cv2
    print(f"  {_n_interaction} interaction features computed.")

    # ── Filter available features ────────────────────────────────────────
    available_features = filter_available_features(merged)
    if not available_features:
        print("\nWARNING: No kinematic feature columns found!")
        return {"merged_df": merged, "n_matched": len(merged)}

    print(f"\nCreating box plots for {len(available_features)} features:")
    for col, label, _, _can_norm in available_features:
        print(f"  - {col}")

    # ── Sub-analyses ─────────────────────────────────────────────────────
    if run_recording_angle_analysis and "recording_angle" in merged.columns:
        _run_recording_angle_analysis(
            merged=merged, available_features=available_features,
            output_dir=output_dir, save_plots=save_plots, show_plots=show_plots,
        )

    longitudinal_report = {}
    if run_longitudinal_report:
        longitudinal_report = run_longitudinal_progression_report(
            merged=merged, available_features=available_features,
            output_dir=output_dir, save_plots=save_plots, show_plots=show_plots,
        )

    medicine_effect_report = {}
    if run_medicine_effect_report:
        medicine_effect_report = _run_medicine_effect_report(
            merged=merged, available_features=available_features,
            output_dir=output_dir, save_plots=save_plots, show_plots=show_plots,
        )

    # ── Age/gender adjusted columns ──────────────────────────────────────
    adj_feature_map: dict = {}
    if has_age_gender:
        print("\nComputing age/gender-adjusted features...")
        for col, label, desc, can_norm in available_features:
            if col not in merged.columns:
                continue
            adj_col = f"{col}_adj"
            merged[adj_col] = residualize_for_age_gender(merged, col)
            adj_feature_map[col] = adj_col
            n_adj = int(merged[adj_col].notna().sum())
            print(f"  {col} -> {adj_col}  ({n_adj} valid rows)")

    # ── UPDRS-total secondary merge ──────────────────────────────────────
    merged_updrs = None
    _updrs_score_col = "UPDRS_Total"
    if updrs_total_csv_path is not None:
        print(f"\nLoading UPDRS-total scores from: {updrs_total_csv_path}")
        _scores_updrs = load_updrs_total_wide(updrs_total_csv_path, score_col_name=_updrs_score_col)
        if not _scores_updrs.empty:
            _scores_updrs["visit"] = _scores_updrs["visit"].apply(normalize_visit_to_int)
            try:
                _scores_updrs["visit"] = _scores_updrs["visit"].astype("Int64")
            except Exception:
                pass
            _KEY_U = ["ids", "visit", "medication_state"]
            _missing_u = [c for c in _KEY_U if c not in kin.columns or c not in _scores_updrs.columns]
            if _missing_u:
                print(f"  WARNING: UPDRS-total merge skipped: missing columns {_missing_u}")
            else:
                merged_updrs = kin.merge(
                    _scores_updrs[_KEY_U + [_updrs_score_col]], on=_KEY_U, how="inner",
                )
                print(f"\nUPDRS-total merge results:")
                print(f"  Kinematic rows: {len(kin)}")
                print(f"  Matched with UPDRS-total scores: {len(merged_updrs)}")
                merged_updrs["score_clean"] = merged_updrs[_updrs_score_col].apply(coerce_int_score)
                merged_updrs = merged_updrs.dropna(subset=["score_clean"]).copy()
                merged_updrs["score_clean"] = merged_updrs["score_clean"].astype(int)
                print(f"  After score cleaning: {len(merged_updrs)} rows")
                # Same-condition median aggregation
                _kin_num_u = merged_updrs.select_dtypes(include="number").columns.tolist()
                _n_bef_u = len(merged_updrs)
                _agg_keys_u = ["ids", "visit", "medication_state", "hand"]
                merged_updrs = (
                    merged_updrs.groupby(_agg_keys_u, dropna=False)
                    .agg(
                        {c: "median" if c in _kin_num_u else "first" for c in merged_updrs.columns if c not in _agg_keys_u}
                    )
                    .reset_index()
                )
                print(f"  Same-condition median aggregation: {_n_bef_u} -> {len(merged_updrs)} rows")
                # Composite score for UPDRS merge
                merged_updrs = compute_composite_score(merged_updrs)
                if merged_updrs.empty:
                    merged_updrs = None

    # ── Build plot jobs ──────────────────────────────────────────────────
    _updrs_plot_dir = os.path.join(output_dir, "updrs_total")
    _plot_jobs: list = [(merged, score_column, output_dir)]
    if merged_updrs is not None:
        os.makedirs(_updrs_plot_dir, exist_ok=True)
        _plot_jobs.append((merged_updrs, _updrs_score_col, _updrs_plot_dir))

    # ── Create plots ─────────────────────────────────────────────────────
    sns.set_style("whitegrid")
    feature_filtered_dfs: dict = {}
    stat_tests: dict = {}

    for _pmerged, _pcol, _pout_dir in _plot_jobs:
        _pfeatures = [
            (col, label, desc, can_norm)
            for col, label, desc, can_norm in KINEMATIC_FEATURES
            if col in _pmerged.columns and _pmerged[col].notna().any()
        ]
        if not _pfeatures:
            print(f"\nWARNING: No kinematic feature columns found for {_pcol} plots.")
            continue

        _score_values = sorted(_pmerged["score_clean"].unique())
        _palette = sns.color_palette("Blues_d", n_colors=len(_score_values))

        print(f"\nMDS-UPDRS {_pcol} score distribution:")
        for _sv in _score_values:
            print(f"  Score {_sv}: {int((_pmerged['score_clean'] == _sv).sum())} samples")

        # Individual and filtered boxplots
        _feat_filtered = plot_feature_boxplots(
            merged=_pmerged,
            features=_pfeatures,
            score_col_label=_pcol,
            palette=_palette,
            stat_tests=stat_tests,
            output_dir=_pout_dir,
            detect_extreme_outliers_fn=detect_extreme_outliers,
            extreme_iqr_multiplier=extreme_iqr_multiplier,
            figsize=figsize,
            save_plots=save_plots,
            show_plots=show_plots,
        )

        # Combined figure
        plot_combined_figure(
            merged=_pmerged,
            features=_pfeatures,
            feature_filtered_dfs=_feat_filtered,
            score_col_label=_pcol,
            palette=_palette,
            output_dir=_pout_dir,
            save_plots=save_plots,
            show_plots=show_plots,
        )

        # Age/gender adjusted boxplots (primary merge only)
        if has_age_gender and adj_feature_map and (_pmerged is merged):
            plot_age_gender_adjusted(
                merged=_pmerged,
                features=_pfeatures,
                adj_feature_map=adj_feature_map,
                score_col_label=_pcol,
                palette=_palette,
                stat_tests=stat_tests,
                detect_extreme_outliers_fn=detect_extreme_outliers,
                extreme_iqr_multiplier=extreme_iqr_multiplier,
                output_dir=_pout_dir,
                figsize=figsize,
                save_plots=save_plots,
                show_plots=show_plots,
            )

        # Preserve primary feature_filtered_dfs for the stats section
        if _pmerged is merged:
            feature_filtered_dfs = _feat_filtered

    # ── Summary statistics ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS BY SCORE")
    print("=" * 60)
    summary_stats = compute_summary_statistics(merged, available_features)

    stat_tests = compute_statistical_tests(merged, available_features)

    check_cv_cycle_correlation(merged)

    # ── Effect sizes ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("EFFECT SIZES (Cohen's d)")
    print("=" * 60)
    print("Interpretation: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >=0.8 large")
    effect_sizes = compute_effect_sizes(merged, available_features, EXPECTED_DIRECTIONS)

    # ── PCA-Varimax ──────────────────────────────────────────────────────
    _pca_feat_cols = [
        col for col, _, _, _ in available_features
        if col in merged.columns and merged[col].notna().sum() >= 10
    ]
    pca_results = {}
    if len(_pca_feat_cols) >= 2:
        try:
            pca_results = run_pca_varimax_analysis(
                merged_df=merged, feature_cols=_pca_feat_cols,
                output_dir=output_dir, save_plots=save_plots, show_plots=show_plots,
            )
        except Exception as _pca_err:
            print(f"  WARNING: PCA-varimax analysis failed: {_pca_err}")

    return {
        "merged_df": merged,
        "n_matched": len(merged),
        "score_distribution": merged["score_clean"].value_counts().sort_index().to_dict(),
        "summary_stats": summary_stats,
        "available_features": [col for col, _, _, _ in available_features],
        "longitudinal_report": longitudinal_report,
        "medicine_effect_report": medicine_effect_report,
        "normalize_by_cycles": normalize_by_cycles,
        "signal_quality_threshold": signal_quality_threshold,
        "signal_quality_sub_thresholds": signal_quality_sub_thresholds or {},
        "min_cycles": int(min_cycles),
        "min_quality_cycles": int(min_quality_cycles),
        "age_gender_adjusted": has_age_gender,
        "adj_feature_map": adj_feature_map,
        "stat_tests": stat_tests,
        "effect_sizes": effect_sizes,
        "pca_varimax": pca_results,
    }
