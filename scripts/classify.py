"""
scripts/classify.py — ML prediction of PD motor severity or direct MDS-UPDRS score.

  * Three severity classes: Mild (MDS-UPDRS 0-1), Moderate (2), Severe (3-4).
    * Optional direct regression of MDS-UPDRS score (0-4).
  * Three classifiers: LightGBM, Logistic Regression, Random Forest.
    * Three regressors: LightGBM, Ridge Regression, Random Forest Regressor.
  * Two label modes: Multi-class and Ordinal (treated as multi-class with ordered labels).
  * Leave-one-subject-out cross-validation.
    * Metrics (severity mode): Accuracy, Balanced Accuracy, Macro Precision, Macro F1.
    * Metrics (score mode): MAE, RMSE, R2, Spearman.
    * Confusion matrix visualisation.
  * Feature importance ranking (LightGBM).

Usage:
  python scripts/classify.py \\
      --kinematics tracking_logs.csv \\
      --scores scores.csv \\
      --id2vid id2vid.csv \\
            --prediction-target score \
        --output results/ \\
        --optuna-trials 50 \\
                --optuna-metric rmse

Dependencies:
  lightgbm, scikit-learn, optuna (optional, for hyperparameter tuning)
"""

import argparse
import json
import os
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd

# Use a non-GUI backend for reliable CLI execution (avoids Tkinter teardown errors).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ps_kinematics.io import (
    canonicalize_video_id,
    coerce_int_score,
    load_videoid_to_patientid_map,
    normalize_hand,
    normalize_med_state,
    normalize_visit_to_int,
    parse_hand_from_path,
    parse_ids_and_visit,
    parse_medication_state_from_path,
)

# Feature columns to use for classification (+ PS extensions)
# These columns must be present in the tracking_logs CSV.  Missing columns
# are silently excluded.  Composite / interaction features are omitted to
# avoid data leakage from score-referenced Z-scores.
CLASSIFICATION_FEATURES = [
    # Hypokinesia
    "Mean Amplitude",
    # Bradykinesia
    "Mean Frequency",
    "Avg Cycle Duration",
    # Speed
    "Peak Velocity",
    "Mean Velocity",
    "Global Velocity",
    # Sequence Effect
    "Norm Decrement Slope",
    "Raw Amp Slope",
    "Amp Decrement Onset",
    "Amp Decrement %",
    "Norm TI Slope",
    "Raw Cycle Duration Slope",
    "Norm Velocity Decrement Slope",
    "Raw Velocity Slope",
    "Raw Speed Slope",
    "Velocity Decrement Onset",
    "Velocity Decrement %",
    # Hesitation-Halts
    "Amplitude CV",
    "Rhythm (CV %)",
    "Cycle Duration CV",
    "Peak Velocity CV",
    "Mean Velocity CV",
    "Num Hesitations",
    "Num Arrests",
    "Num Interruptions (2x)",
    "Max Pause Duration (s)",
    "Pause Time Ratio",
    "Sample Entropy",
    "Amp-Vel Coupling",
    # Task-Specific (PS)
    "Arm Swing Index",
    # Metadata (encoded)
    "medication_state_enc",  # 1 = on-medication, 0 = off-medication
    "hand_enc",  # 1 = right, 0 = left
]

# Three-class severity grouping (Zarrat Ehsan et al. 2024)
SEVERITY_MAP = {0: "Mild", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Severe"}
CLASS_NAMES = ["Mild", "Moderate", "Severe"]
CLASS_LABEL_MAP = {"Mild": 0, "Moderate": 1, "Severe": 2}


def _map_severity(score):
    """Map MDS-UPDRS 3.6 score (0-4) to 3-class label."""
    return SEVERITY_MAP.get(int(score), None)


def _normalize_video_path_for_matching(path_value) -> str:
    """Build a canonical filename-level key for cross-file video matching."""
    if pd.isna(path_value):
        return ""

    out = str(path_value).strip().replace("\\", "/").lower()
    out = re.sub(r"_cl_sh_sr4x(?:_gimmvfi)?(?=\.[^./\\]+$|$)", "", out)
    name = os.path.basename(out)

    if "segmented_video_" in name:
        tail = name.split("segmented_video_", 1)[1]
        tail = re.sub(r"_cropped(?=\.[^./\\]+$|$)", "", tail)
        return f"seg::{tail}"

    visit_match = re.search(r"visit\s*(\d+)", out)
    pom_match = re.search(r"(pom\d+vd\d+)", out)
    hand_tail_match = re.search(r"((?:on|off)_4[lr].*)", name)
    if pom_match and hand_tail_match:
        visit_part = visit_match.group(1) if visit_match else "na"
        hand_tail = re.sub(r"\s+", " ", hand_tail_match.group(1)).strip()
        hand_tail = re.sub(r"_cropped(?=\.[^./\\]+$|$)", "", hand_tail)
        return f"clip::v{visit_part}_{pom_match.group(1)}_{hand_tail}"

    name = re.sub(r"^enhanced_videos\d*_", "", name)
    name = re.sub(r"enhanced_videos\d*", "cropped_videos_output", name)
    name = re.sub(r"^clahe_sharpen_esrgan_videos\d*_", "", name)
    name = re.sub(r"clahe_sharpen_esrgan_videos\d*", "cropped_videos_output", name)
    if name.startswith("visit "):
        name = f"cropped_videos_output_{name}"
    return re.sub(r"\s+", " ", name).strip()


def _normalize_video_path_series_for_matching(series: "pd.Series") -> "pd.Series":
    if series.empty:
        return series.astype(str)
    return series.fillna("").astype(str).map(_normalize_video_path_for_matching)


def _load_video_quality_labels(video_quality_csv_path: str) -> "pd.DataFrame":
    """Load manual quality labels and attach a normalized join key."""
    quality_df = pd.read_csv(video_quality_csv_path)
    required_cols = {"video_path", "quality_label"}
    missing = required_cols.difference(quality_df.columns)
    if missing:
        raise RuntimeError(
            "video quality label file is missing required column(s): " + ", ".join(sorted(missing))
        )

    quality_df = quality_df.copy()
    quality_df["quality_label"] = pd.to_numeric(quality_df["quality_label"], errors="coerce")
    quality_df = quality_df.dropna(subset=["video_path", "quality_label"]).copy()
    quality_df["quality_label"] = quality_df["quality_label"].astype(int)

    valid_mask = quality_df["quality_label"].between(1, 3)
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        print(
            "  WARNING: dropped "
            f"{n_invalid} rows from video quality labels with quality_label outside [1, 3]"
        )
        quality_df = quality_df[valid_mask].copy()

    quality_df["_quality_key"] = _normalize_video_path_series_for_matching(quality_df["video_path"])
    quality_df = quality_df.dropna(subset=["_quality_key"]).copy()

    dup_mask = quality_df.duplicated(subset=["_quality_key"], keep=False)
    n_dup_keys = int(quality_df.loc[dup_mask, "_quality_key"].nunique())
    if n_dup_keys > 0:
        print(
            "  WARNING: duplicate video_path keys in video quality labels for "
            f"{n_dup_keys} videos; keeping the best (lowest) quality_label per key"
        )
        quality_df = (
            quality_df.sort_values(["_quality_key", "quality_label"])
            .drop_duplicates(subset=["_quality_key"], keep="first")
            .copy()
        )

    return quality_df[["_quality_key", "quality_label"]]


def _apply_video_quality_filter(
    kin: "pd.DataFrame",
    quality_df: "pd.DataFrame",
    quality_threshold: int,
) -> "pd.DataFrame":
    """Keep only rows with manual quality_label <= threshold."""
    if "video_path" not in kin.columns:
        raise RuntimeError(
            "video_path column is required in kinematics CSV to apply video quality filter"
        )

    kin = kin.copy()
    kin["_quality_key"] = _normalize_video_path_series_for_matching(kin["video_path"])
    quality_map = quality_df.set_index("_quality_key")["quality_label"]
    matched_labels = kin["_quality_key"].map(quality_map)

    n_before = len(kin)
    n_matched = int(matched_labels.notna().sum())
    keep_mask = matched_labels.notna() & (matched_labels <= int(quality_threshold))
    kin = kin[keep_mask].copy()
    if not kin.empty:
        kin["quality_label"] = matched_labels[keep_mask].astype(int).to_numpy()
    kin = kin.drop(columns=["_quality_key"], errors="ignore")

    print(
        f"  Manual video quality filter (threshold <= {quality_threshold}): "
        f"matched {n_matched}/{n_before}, kept {len(kin)}/{n_before} "
        f"({n_before - len(kin)} dropped)"
    )
    return kin


def _load_and_merge(
    kinematics_csv_path: str,
    score_csv_path: str,
    id2vid_csv_path: str,
    score_column: str = "score_clean",
    signal_quality_threshold: float = 0.0,
    signal_quality_sub_thresholds: dict | None = None,
    min_cycles: int = 0,
    min_quality_cycles: int = 0,
    min_inter_mcp_span_px: float = 0.0,
    min_detection_rate: float = 0.0,
    recording_angle_csv_path: str | None = None,
    selected_recording_angles: list | None = None,
    video_quality_labels_csv_path: str | None = None,
    video_quality_threshold: int = 3,
) -> pd.DataFrame:
    """Load and merge kinematic features with clinical scores.

    Returns a DataFrame with columns: feature columns, 'subject_id',
    'severity_label', 'severity_int'.

    Filters applied (in order, matching analyze.py):
    1. record_type == "VIDEO"
    2. min_detection_rate (VQ_detection_rate)
    3. recording angle merge + selected_recording_angles filter
    4. manual video quality filter (quality_label <= threshold)
    5. min_inter_mcp_span_px (VQ_inter_mcp_span_px)
    6. signal_quality_threshold (Signal Quality)
    7. signal_quality_sub_thresholds (SQ_* columns)
    8. min_quality_cycles (Quality Cycles)
    9. min_cycles (Total Cycles)
    """
    kin = pd.read_csv(kinematics_csv_path)

    if video_quality_threshold not in (1, 2, 3):
        raise ValueError("video_quality_threshold must be one of: 1, 2, 3")
    if video_quality_labels_csv_path is None and video_quality_threshold < 3:
        raise ValueError("video_quality_threshold < 3 requires video_quality_labels_csv_path")

    quality_df = None
    if video_quality_labels_csv_path is not None:
        print(f"  Loading manual video quality labels from: {video_quality_labels_csv_path}")
        quality_df = _load_video_quality_labels(video_quality_labels_csv_path)
        print(f"  Video quality labels loaded: {len(quality_df)} unique videos")

    # --- 1. record_type filter: keep only fully processed VIDEO rows ---
    if "record_type" in kin.columns:
        n_before = len(kin)
        kin = kin[kin["record_type"] == "VIDEO"].copy()
        n_dropped = n_before - len(kin)
        if n_dropped:
            print(
                f"  record_type filter: kept {len(kin)}/{n_before} VIDEO records "
                f"({n_dropped} non-VIDEO rows dropped)"
            )

    # --- 2. Detection rate filter ---
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
            n_dropped = n_before - len(kin)
            print(
                f"  Detection rate filter (VQ_detection_rate >= {min_detection_rate:.2f}): "
                f"kept {len(kin)}/{n_before} ({n_dropped} dropped)"
            )

    # --- 3. Recording angle: merge labels + optional allow-list filter ---
    if recording_angle_csv_path is not None:
        _ext = os.path.splitext(recording_angle_csv_path)[1].lower()
        if _ext in (".xlsx", ".xls"):
            try:
                _angle_df = pd.read_excel(recording_angle_csv_path)
            except Exception:
                _angle_df = pd.read_csv(recording_angle_csv_path)
        else:
            _angle_df = pd.read_csv(recording_angle_csv_path)

        if "recording_angle" not in _angle_df.columns:
            print("  WARNING: recording angle file has no 'recording_angle' column; skipping.")
        else:
            # Normalize angle labels (common aliases)
            _ANGLE_ALIASES = {
                "frontal": "front",
                "front view": "front",
                "angled view": "angled",
                "lateral view": "lateral",
                "side": "lateral",
            }

            def _norm_angle(v):
                if pd.isna(v):
                    return None
                s = str(v).strip().lower()
                return _ANGLE_ALIASES.get(s, s) if s else None

            _angle_df["recording_angle"] = _angle_df["recording_angle"].apply(_norm_angle)
            _angle_df = _angle_df.dropna(subset=["recording_angle"]).copy()

            if "video_path" in _angle_df.columns and "video_path" in kin.columns:
                _angle_df["_angle_key"] = _angle_df["video_path"].apply(
                    lambda p: os.path.basename(str(p)).lower()
                )
                kin["_angle_key"] = kin["video_path"].apply(
                    lambda p: os.path.basename(str(p)).lower()
                )
                kin = kin.merge(
                    _angle_df[["_angle_key", "recording_angle"]].drop_duplicates("_angle_key"),
                    on="_angle_key",
                    how="left",
                )
                kin.drop(columns=["_angle_key"], inplace=True)
                n_matched = int(kin["recording_angle"].notna().sum())
                print(f"  Recording angle matched: {n_matched}/{len(kin)} videos")
            elif "video_id" in _angle_df.columns and "ids" in kin.columns:
                _angle_df["_angle_key"] = _angle_df["video_id"].astype(str).str.strip().str.lower()
                kin["_angle_key"] = kin["ids"].astype(str).str.strip().str.lower()
                kin = kin.merge(
                    _angle_df[["_angle_key", "recording_angle"]].drop_duplicates("_angle_key"),
                    on="_angle_key",
                    how="left",
                )
                kin.drop(columns=["_angle_key"], inplace=True)
                n_matched = int(kin["recording_angle"].notna().sum())
                print(f"  Recording angle matched (by video_id): {n_matched}/{len(kin)} videos")
            else:
                print(
                    "  WARNING: recording angle CSV has no 'video_path' or 'video_id' column; skipping merge."
                )

    if selected_recording_angles is not None:
        if "recording_angle" not in kin.columns:
            raise RuntimeError(
                "selected_recording_angles requires recording angle labels to be merged; "
                "provide recording_angle_csv_path."
            )
        _ANGLE_ALIASES = {
            "frontal": "front",
            "front view": "front",
            "angled view": "angled",
            "lateral view": "lateral",
            "side": "lateral",
        }
        _sel = {
            _ANGLE_ALIASES.get(str(a).strip().lower(), str(a).strip().lower())
            for a in selected_recording_angles
        }
        n_before = len(kin)
        kin = kin[kin["recording_angle"].isin(_sel)].copy()
        print(
            f"  Recording-angle filter {sorted(_sel)}: kept {len(kin)}/{n_before} "
            f"({n_before - len(kin)} dropped)"
        )

    # --- 4. Manual video-quality threshold ---
    if quality_df is not None:
        kin = _apply_video_quality_filter(
            kin,
            quality_df,
            video_quality_threshold,
        )

    # --- 5. Inter-MCP span filter ---
    if min_inter_mcp_span_px > 0.0:
        if "VQ_inter_mcp_span_px" not in kin.columns:
            print(
                "  WARNING: min_inter_mcp_span_px > 0 but 'VQ_inter_mcp_span_px' column "
                "not found in CSV; skipping inter-MCP span filter."
            )
        else:
            span = pd.to_numeric(kin["VQ_inter_mcp_span_px"], errors="coerce").fillna(0.0)
            n_before = len(kin)
            kin = kin[span >= min_inter_mcp_span_px].copy()
            n_dropped = n_before - len(kin)
            print(
                f"  Inter-MCP span filter (VQ_inter_mcp_span_px >= {min_inter_mcp_span_px:.1f} px): "
                f"kept {len(kin)}/{n_before} ({n_dropped} dropped)"
            )

    # --- 6. Signal Quality threshold ---
    if signal_quality_threshold > 0.0 and "Signal Quality" in kin.columns:
        sq = pd.to_numeric(kin["Signal Quality"], errors="coerce").fillna(0.0)
        n_before = len(kin)
        kin = kin[sq >= signal_quality_threshold].copy()
        n_dropped = n_before - len(kin)
        print(
            f"  Signal Quality threshold {signal_quality_threshold:.2f}: "
            f"kept {len(kin)}/{n_before} ({n_dropped} dropped)"
        )

    # --- 7. Per-sub-score signal quality filters ---
    if signal_quality_sub_thresholds:
        for sub_key, sub_thresh in signal_quality_sub_thresholds.items():
            col = sub_key if sub_key.startswith("SQ_") else f"SQ_{sub_key}"
            if col not in kin.columns:
                print(
                    f"  WARNING: sub-score column '{col}' not found in CSV; skipping sub-threshold filter."
                )
                continue
            sub_vals = pd.to_numeric(kin[col], errors="coerce").fillna(0.0)
            n_before = len(kin)
            kin = kin[sub_vals >= float(sub_thresh)].copy()
            n_dropped = n_before - len(kin)
            print(
                f"  SQ sub-score '{col}' threshold {sub_thresh:.2f}: "
                f"kept {len(kin)}/{n_before} ({n_dropped} dropped)"
            )

    # --- 8. Minimum quality cycles filter ---
    if min_quality_cycles > 0:
        if "Quality Cycles" not in kin.columns:
            print(
                "  WARNING: min_quality_cycles > 0 but 'Quality Cycles' column not found; "
                "skipping quality-cycle filter."
            )
        else:
            qcounts = pd.to_numeric(kin["Quality Cycles"], errors="coerce").fillna(0.0)
            n_before = len(kin)
            kin = kin[qcounts >= float(min_quality_cycles)].copy()
            n_dropped = n_before - len(kin)
            print(
                f"  Minimum quality cycles filter (Quality Cycles >= {min_quality_cycles}): "
                f"kept {len(kin)}/{n_before} ({n_dropped} dropped)"
            )

    # --- 9. Minimum total cycles filter ---
    if min_cycles > 0:
        if "Total Cycles" not in kin.columns:
            print(
                "  WARNING: min_cycles > 0 but 'Total Cycles' column not found; "
                "skipping total-cycle filter."
            )
        else:
            tcounts = pd.to_numeric(kin["Total Cycles"], errors="coerce").fillna(0.0)
            n_before = len(kin)
            kin = kin[tcounts >= float(min_cycles)].copy()
            n_dropped = n_before - len(kin)
            print(
                f"  Minimum cycles filter (Total Cycles >= {min_cycles}): "
                f"kept {len(kin)}/{n_before} ({n_dropped} dropped)"
            )

    # Parse IDs from video path (fill missing values only)
    if "video_path" in kin.columns:
        _parsed = kin["video_path"].apply(lambda p: parse_ids_and_visit(str(p)))
        _ids_from_path = _parsed.apply(
            lambda x: (
                x[0] if isinstance(x, tuple) else (x.get("ids") if hasattr(x, "get") else None)
            )
        )
        _visit_from_path = _parsed.apply(
            lambda x: (
                x[1] if isinstance(x, tuple) else (x.get("visit") if hasattr(x, "get") else None)
            )
        )
        _hand_from_path = kin["video_path"].apply(
            lambda p: normalize_hand(parse_hand_from_path(str(p)))
        )
        _med_from_path = kin["video_path"].apply(
            lambda p: normalize_med_state(parse_medication_state_from_path(str(p)))
        )

        if "ids" in kin.columns:
            kin["ids"] = kin["ids"].where(kin["ids"].notna(), _ids_from_path)
        else:
            kin["ids"] = _ids_from_path

        if "visit" in kin.columns:
            kin["visit"] = kin["visit"].where(kin["visit"].notna(), _visit_from_path)
        else:
            kin["visit"] = _visit_from_path

        if "hand" in kin.columns:
            kin["hand"] = kin["hand"].where(kin["hand"].notna(), _hand_from_path)
        else:
            kin["hand"] = _hand_from_path

        if "medication_state" in kin.columns:
            kin["medication_state"] = kin["medication_state"].where(
                kin["medication_state"].notna(), _med_from_path
            )
        else:
            kin["medication_state"] = _med_from_path

    if "log_medication_state" in kin.columns:
        _med_from_log = kin["log_medication_state"].apply(normalize_med_state)
        if "medication_state" not in kin.columns:
            kin["medication_state"] = _med_from_log
        else:
            kin["medication_state"] = kin["medication_state"].where(
                kin["medication_state"].notna(), _med_from_log
            )

    if "log_hand" in kin.columns:
        _hand_from_log = kin["log_hand"].apply(normalize_hand)
        if "hand" not in kin.columns:
            kin["hand"] = _hand_from_log
        else:
            kin["hand"] = kin["hand"].where(kin["hand"].notna(), _hand_from_log)

    if "ids" in kin.columns:
        kin["ids"] = kin["ids"].apply(canonicalize_video_id)
    if "visit" in kin.columns:
        kin["visit"] = kin["visit"].apply(normalize_visit_to_int)
        try:
            kin["visit"] = kin["visit"].astype("Int64")
        except Exception:
            pass
    if "hand" in kin.columns:
        kin["hand"] = kin["hand"].apply(normalize_hand)
    if "medication_state" in kin.columns:
        kin["medication_state"] = kin["medication_state"].apply(normalize_med_state)

    # Map video IDs to patient IDs
    if id2vid_csv_path and os.path.isfile(id2vid_csv_path):
        _id_map = load_videoid_to_patientid_map(id2vid_csv_path)
        if "ids" in kin.columns:
            kin["ids"] = kin["ids"].map(lambda x: _id_map.get(canonicalize_video_id(x), x))

    # Load scores
    scores = pd.read_csv(score_csv_path)
    if "ids" in scores.columns:
        scores["ids"] = scores["ids"].astype(str).str.strip()
    if "visit" in scores.columns:
        scores["visit"] = scores["visit"].apply(normalize_visit_to_int)
        try:
            scores["visit"] = scores["visit"].astype("Int64")
        except Exception:
            pass
    if "hand" in scores.columns:
        scores["hand"] = scores["hand"].apply(normalize_hand)
    if "medication_state" in scores.columns:
        scores["medication_state"] = scores["medication_state"].apply(normalize_med_state)

    _score_col = score_column
    if _score_col not in scores.columns:
        _priority = ["score_clean", "score", "ProS", "pros"]
        _score_col = next((c for c in _priority if c in scores.columns), None)
        if _score_col is None:
            raise ValueError(
                f"Could not find score column '{score_column}' in scores file. "
                f"Available columns: {list(scores.columns)}."
            )
        print(f"  ⚠ Score column '{score_column}' not found; using '{_score_col}' instead.")

    # Merge
    _expected_keys = ["ids", "visit", "hand", "medication_state"]
    _key_cols = [c for c in _expected_keys if c in kin.columns and c in scores.columns]
    _missing_kin = [c for c in _expected_keys if c not in kin.columns]
    _missing_scores = [c for c in _expected_keys if c not in scores.columns]
    if _missing_kin:
        print(f"  ⚠ Key columns absent from kinematics (excluded from merge key): {_missing_kin}")
    if _missing_scores:
        print(f"  ⚠ Key columns absent from scores (excluded from merge key): {_missing_scores}")
        if "visit" in _missing_scores:
            print(
                "    WARNING: 'visit' missing from scores — all visits per patient will be "
                "joined to the same score, potentially inflating the dataset."
            )
    if not _key_cols:
        raise ValueError(
            f"No common key columns between kinematics and scores. "
            f"Kin: {list(kin.columns[:10])}, Scores: {list(scores.columns[:10])}"
        )

    _n_kin_before_merge = len(kin)
    merged = kin.merge(scores[_key_cols + [_score_col]], on=_key_cols, how="inner")
    print(
        f"  Merge on {_key_cols}: {len(merged)}/{_n_kin_before_merge} kinematic rows matched "
        f"({_n_kin_before_merge - len(merged)} unmatched)."
    )
    if len(merged) == 0:
        print(
            "  ⚠ Zero rows after merge — check that ids/visit/hand/medication_state values "
            "are consistent between kinematics CSV and scores CSV."
        )
    elif len(merged) < _n_kin_before_merge * 0.5:
        print(
            "  ⚠ Less than 50% of kinematic rows matched scores. "
            "Check normalization of key columns."
        )
    merged["score_clean"] = merged[_score_col].apply(coerce_int_score)
    merged = merged.dropna(subset=["score_clean"]).copy()
    merged["score_clean"] = merged["score_clean"].astype(int)
    merged["severity_label"] = merged["score_clean"].apply(_map_severity)
    merged = merged.dropna(subset=["severity_label"])
    merged["severity_int"] = merged["severity_label"].map(CLASS_LABEL_MAP)
    merged["subject_id"] = merged["ids"] if "ids" in merged.columns else merged.index.astype(str)

    return merged


def _build_classifiers():
    """Return dict of classifier instances."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    clfs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        clfs["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    except ImportError:
        print("  ⚠ LightGBM not installed — skipping LightGBM classifier.")
    return clfs


def _build_regressors():
    """Return dict of regressor instances for direct score prediction."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    regs = {
        "Ridge Regression": Ridge(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }
    try:
        from lightgbm import LGBMRegressor

        regs["LightGBM Regressor"] = LGBMRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    except ImportError:
        print("  ⚠ LightGBM not installed — skipping LightGBM regressor.")
    return regs


def _build_classifier_from_params(clf_name: str, params: dict, random_state: int = 42):
    """Build a classifier instance from a model name and parameter dictionary."""
    if clf_name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression

        _solver = params.get("solver", "lbfgs")
        _penalty = "l1" if _solver == "liblinear" else "l2"
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver=_solver,
            penalty=_penalty,
            max_iter=int(params.get("max_iter", 1000)),
            class_weight="balanced",
            random_state=random_state,
        )

    if clf_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    if clf_name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            max_depth=int(params.get("max_depth", -1)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(f"Unsupported classifier for Optuna tuning: {clf_name}")


def _build_regressor_from_params(reg_name: str, params: dict, random_state: int = 42):
    """Build a regressor instance from a model name and parameter dictionary."""
    if reg_name == "Ridge Regression":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=random_state)

    if reg_name == "Random Forest Regressor":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth", None),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=random_state,
            n_jobs=-1,
        )

    if reg_name == "LightGBM Regressor":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            max_depth=int(params.get("max_depth", -1)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(f"Unsupported regressor for Optuna tuning: {reg_name}")


def _suggest_hyperparameters(trial, clf_name: str) -> dict:
    """Define Optuna search spaces for each supported classifier."""
    if clf_name == "Logistic Regression":
        # Keep multiclass-compatible solvers only to avoid invalid trials.
        _solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        return {
            "solver": _solver,
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "max_iter": trial.suggest_int("max_iter", 300, 2000),
        }

    if clf_name == "Random Forest":
        _max_depth = trial.suggest_int("max_depth", 3, 20)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": _max_depth,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    if clf_name == "LightGBM":
        _max_depth = trial.suggest_int("max_depth", 3, 12)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": _max_depth,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    raise ValueError(f"Unsupported classifier for Optuna tuning: {clf_name}")


def _suggest_regression_hyperparameters(trial, reg_name: str) -> dict:
    """Define Optuna search spaces for each supported regressor."""
    if reg_name == "Ridge Regression":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True),
        }

    if reg_name == "Random Forest Regressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    if reg_name == "LightGBM Regressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    raise ValueError(f"Unsupported regressor for Optuna tuning: {reg_name}")


def _tune_classifier_with_optuna(
    clf_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int,
    optimize_metric: str,
    random_state: int = 42,
    sample_weights=None,
):
    """Tune classifier hyperparameters with Optuna and LOSO CV."""
    try:
        import optuna
    except ImportError:
        print("  ⚠ Optuna not installed — using default hyperparameters.")
        return None, None

    if n_trials <= 0:
        return None, None

    _valid_metrics = {"accuracy", "balanced_accuracy", "macro_precision", "macro_f1"}
    if optimize_metric not in _valid_metrics:
        raise ValueError(
            f"Unsupported optuna metric '{optimize_metric}'. "
            f"Choose one of {sorted(_valid_metrics)}."
        )

    _sampler = optuna.samplers.TPESampler(seed=random_state)
    _study = optuna.create_study(direction="maximize", sampler=_sampler)

    def _objective(trial):
        _params = _suggest_hyperparameters(trial, clf_name)
        _clf = _build_classifier_from_params(clf_name, _params, random_state=random_state)
        try:
            _cv_res = _loso_cv(X, y, groups, _clf, sample_weights=sample_weights)
        except ValueError as _err:
            trial.set_user_attr("fit_error", str(_err))
            return 0.0
        if len(_cv_res["y_true"]) == 0:
            return 0.0
        _metrics = _compute_metrics(_cv_res["y_true"], _cv_res["y_pred"])
        return float(_metrics[optimize_metric])

    _study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    _best_params = dict(_study.best_params)
    _best_clf = _build_classifier_from_params(clf_name, _best_params, random_state=random_state)

    _summary = {
        "best_value": float(_study.best_value),
        "best_params": _best_params,
        "n_trials": int(n_trials),
        "metric": optimize_metric,
    }
    return _best_clf, _summary


def _tune_regressor_with_optuna(
    reg_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int,
    optimize_metric: str,
    random_state: int = 42,
    sample_weights=None,
):
    """Tune regressor hyperparameters with Optuna and LOSO CV."""
    try:
        import optuna
    except ImportError:
        print("  ⚠ Optuna not installed — using default hyperparameters.")
        return None, None

    if n_trials <= 0:
        return None, None

    _valid_metrics = {"mae", "rmse", "r2", "spearman"}
    if optimize_metric not in _valid_metrics:
        raise ValueError(
            f"Unsupported optuna metric '{optimize_metric}'. "
            f"Choose one of {sorted(_valid_metrics)}."
        )

    _direction = "minimize" if optimize_metric in {"mae", "rmse"} else "maximize"
    _sampler = optuna.samplers.TPESampler(seed=random_state)
    _study = optuna.create_study(direction=_direction, sampler=_sampler)

    def _objective(trial):
        _params = _suggest_regression_hyperparameters(trial, reg_name)
        _reg = _build_regressor_from_params(reg_name, _params, random_state=random_state)
        try:
            _cv_res = _loso_cv_regression(X, y, groups, _reg, sample_weights=sample_weights)
        except ValueError as _err:
            trial.set_user_attr("fit_error", str(_err))
            return 1e9 if _direction == "minimize" else -1e9
        if len(_cv_res["y_true"]) == 0:
            return 1e9 if _direction == "minimize" else -1e9
        _metrics = _compute_regression_metrics(_cv_res["y_true"], _cv_res["y_pred"])
        return float(_metrics[optimize_metric])

    _study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    _best_params = dict(_study.best_params)
    _best_reg = _build_regressor_from_params(reg_name, _best_params, random_state=random_state)

    _summary = {
        "best_value": float(_study.best_value),
        "best_params": _best_params,
        "n_trials": int(n_trials),
        "metric": optimize_metric,
    }
    return _best_reg, _summary


def _loso_cv(X, y, groups, clf, ordinal=False, sample_weights=None):
    """Leave-one-subject-out cross-validation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) — integer class labels
    groups : np.ndarray, shape (n_samples,) — subject IDs for LOSO split
    clf : sklearn-compatible classifier
    ordinal : bool
        If True, class labels are treated as ordinal-valued (same as multi-class
        here; reserved for future ordinal-regression extensions).
    sample_weights : np.ndarray or None
        Per-sample weights passed to clf.fit (e.g. SQ_cycle_yield).
        None means uniform weights.

    Returns
    -------
    dict with 'y_true', 'y_pred', 'subjects'
    """
    from sklearn.base import clone
    from sklearn.preprocessing import StandardScaler

    unique_subjects = np.unique(groups)
    y_true_all, y_pred_all, subj_all = [], [], []

    for subj in unique_subjects:
        _test_mask = groups == subj
        _train_mask = ~_test_mask
        if _train_mask.sum() < 5 or _test_mask.sum() < 1:
            continue

        X_train, X_test = X[_train_mask], X[_test_mask]
        y_train = y[_train_mask]

        # Scale features per fold
        _scaler = StandardScaler()
        X_train = _scaler.fit_transform(X_train)
        X_test = _scaler.transform(X_test)

        _clf = clone(clf)
        _fit_kwargs = {}
        if sample_weights is not None:
            _fit_kwargs["sample_weight"] = sample_weights[_train_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _clf.fit(X_train, y_train, **_fit_kwargs)

        _preds = _clf.predict(X_test)
        y_true_all.extend(y[_test_mask].tolist())
        y_pred_all.extend(_preds.tolist())
        subj_all.extend([subj] * int(_test_mask.sum()))

    return {
        "y_true": np.array(y_true_all),
        "y_pred": np.array(y_pred_all),
        "subjects": subj_all,
    }


def _loso_cv_regression(X, y, groups, reg, sample_weights=None):
    """Leave-one-subject-out CV for direct score regression."""
    from sklearn.base import clone
    from sklearn.preprocessing import StandardScaler

    unique_subjects = np.unique(groups)
    y_true_all, y_pred_all, subj_all = [], [], []

    for subj in unique_subjects:
        _test_mask = groups == subj
        _train_mask = ~_test_mask
        if _train_mask.sum() < 5 or _test_mask.sum() < 1:
            continue

        X_train, X_test = X[_train_mask], X[_test_mask]
        y_train = y[_train_mask]

        _scaler = StandardScaler()
        X_train = _scaler.fit_transform(X_train)
        X_test = _scaler.transform(X_test)

        _reg = clone(reg)
        _fit_kwargs = {}
        if sample_weights is not None:
            _fit_kwargs["sample_weight"] = sample_weights[_train_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _reg.fit(X_train, y_train, **_fit_kwargs)

        _preds = _reg.predict(X_test)
        y_true_all.extend(y[_test_mask].tolist())
        y_pred_all.extend(np.asarray(_preds, dtype=float).tolist())
        subj_all.extend([subj] * int(_test_mask.sum()))

    return {
        "y_true": np.array(y_true_all, dtype=float),
        "y_pred": np.array(y_pred_all, dtype=float),
        "subjects": subj_all,
    }


def _loso_cv_mean_baseline(y, groups):
    """LOSO baseline: predict each test fold with the train-fold mean score."""
    unique_subjects = np.unique(groups)
    y_true_all, y_pred_all, subj_all = [], [], []

    for subj in unique_subjects:
        _test_mask = groups == subj
        _train_mask = ~_test_mask
        if _train_mask.sum() < 5 or _test_mask.sum() < 1:
            continue
        _pred_val = float(np.mean(y[_train_mask]))
        _n_test = int(_test_mask.sum())

        y_true_all.extend(y[_test_mask].tolist())
        y_pred_all.extend([_pred_val] * _n_test)
        subj_all.extend([subj] * _n_test)

    return {
        "y_true": np.array(y_true_all, dtype=float),
        "y_pred": np.array(y_pred_all, dtype=float),
        "subjects": subj_all,
    }


def _compute_metrics(y_true, y_pred):
    """Compute accuracy, balanced accuracy, macro precision, macro F1."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)) * 100.0,
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) * 100.0,
            "macro_precision": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            )
            * 100.0,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0,
        }


def _compute_regression_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R2, and Spearman rank correlation."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    _s_true = pd.Series(y_true)
    _s_pred = pd.Series(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _spearman = _s_true.corr(_s_pred, method="spearman")
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": float(_spearman) if pd.notna(_spearman) else float("nan"),
        }


def _plot_confusion_matrix(y_true, y_pred, clf_name, output_dir, save_plots, show_plots):
    """Save a seaborn confusion matrix plot (percentage normalised by true class)."""
    from sklearn.metrics import confusion_matrix

    _cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    _cm_pct = _cm.astype(float) / _cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        _cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        vmin=0,
        vmax=100,
        annot_kws={"size": 12},
    )
    for _t in ax.texts:
        _t.set_text(_t.get_text() + "%")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(clf_name, fontsize=13)
    plt.tight_layout()

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        _safe = clf_name.replace(" ", "_")
        _path = os.path.join(output_dir, f"confusion_matrix_{_safe}.png")
        fig.savefig(_path, dpi=150, bbox_inches="tight")
        print(f"    Saved confusion matrix → {_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _plot_feature_importance(clf, feature_names, clf_name, output_dir, save_plots, show_plots):
    """Plot LightGBM feature importances (gain-based)."""
    try:
        _imp = clf.feature_importances_
    except AttributeError:
        return
    _idx = np.argsort(_imp)[::-1][:20]
    _names = [feature_names[i] for i in _idx]
    _vals = _imp[_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(_names) * 0.35)))
    ax.barh(range(len(_names)), _vals[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(_names)))
    ax.set_yticklabels(_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"{clf_name} — Top Feature Importances")
    plt.tight_layout()

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        _safe = clf_name.replace(" ", "_")
        _path = os.path.join(output_dir, f"feature_importance_{_safe}.png")
        fig.savefig(_path, dpi=150, bbox_inches="tight")
        print(f"    Saved feature importance → {_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _plot_regression_scatter(y_true, y_pred, model_name, output_dir, save_plots, show_plots):
    """Plot true vs predicted scores for direct score prediction mode."""
    if len(y_true) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.75, color="teal", edgecolors="none")
    _lo = float(min(np.min(y_true), np.min(y_pred)))
    _hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([_lo, _hi], [_lo, _hi], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("True MDS-UPDRS score")
    ax.set_ylabel("Predicted score")
    ax.set_title(model_name)
    plt.tight_layout()

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        _safe = model_name.replace(" ", "_")
        _path = os.path.join(output_dir, f"score_scatter_{_safe}.png")
        fig.savefig(_path, dpi=150, bbox_inches="tight")
        print(f"    Saved regression scatter → {_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def run_classification(
    kinematics_csv_path: str,
    score_csv_path: str,
    id2vid_csv_path: str,
    output_dir: str = ".",
    score_column: str = "score_clean",
    signal_quality_threshold: float = 0.0,
    signal_quality_sub_thresholds: dict | None = None,
    min_cycles: int = 0,
    min_quality_cycles: int = 0,
    min_inter_mcp_span_px: float = 0.0,
    min_detection_rate: float = 0.0,
    recording_angle_csv_path: str | None = None,
    selected_recording_angles: list | None = None,
    video_quality_labels_csv_path: str | None = None,
    video_quality_threshold: int = 3,
    optuna_trials: int = 0,
    optuna_metric: str = "macro_f1",
    prediction_target: str = "severity",
    label_modes: list[str] | None = None,
    save_plots: bool = True,
    show_plots: bool = False,
    use_sample_weights: bool = True,
):
    """Run the full ML classification pipeline.

    Parameters
    ----------
    kinematics_csv_path : str
        Path to tracking_logs.csv (pipeline output).
    score_csv_path : str
        Path to clinical scores CSV.
    id2vid_csv_path : str
        Path to id2vid.csv for patient ID mapping.
    output_dir : str
        Directory for saving results and plots.
    score_column : str
        Name of the score column in the scores CSV.
    signal_quality_threshold : float
        Minimum Signal Quality score (0–1) for inclusion (0 = no filter).
    signal_quality_sub_thresholds : dict, optional
        Per-sub-score minimum thresholds, e.g. {"SQ_cycle_yield": 0.4}.
        Keys accept the SQ_ prefix or without it.
    min_cycles : int
        Minimum Total Cycles required (0 = no filter).
    min_quality_cycles : int
        Minimum Quality Cycles required (0 = no filter).
    min_inter_mcp_span_px : float
        Minimum VQ_inter_mcp_span_px (0 = no filter).
    min_detection_rate : float
        Minimum VQ_detection_rate (0–1) required (0 = no filter).
    recording_angle_csv_path : str, optional
        Path to recording-angle labels CSV/Excel.
    selected_recording_angles : list, optional
        Allow-list of angle labels to keep, e.g. ["front", "angled"].
        None keeps all angles.
    video_quality_labels_csv_path : str, optional
        Path to manual video-quality labels CSV with columns
        ``video_path`` and ``quality_label`` (1=best, 3=worst).
        When provided, rows are filtered by ``video_quality_threshold``.
    video_quality_threshold : int
        Keep only videos with manual ``quality_label`` <= this value.
        Must be one of ``{1, 2, 3}``.
    optuna_trials : int
        Number of Optuna trials per classifier (0 disables tuning).
    optuna_metric : str
        Severity mode: accuracy, balanced_accuracy, macro_precision, macro_f1.
        Score mode: mae, rmse, r2, spearman.
    prediction_target : str
        Prediction target: "severity" (3-class classification) or
        "score" (direct regression of MDS-UPDRS score).
    label_modes : list[str] | None
        Which label modes to run for severity classification.
        Supported values: "Multi", "Ordinal".  Default (None) runs both.
        Pass ``["Multi"]`` during hyperparameter optimisation to skip
        Ordinal and cut classification time by ~40%.
    save_plots : bool
        Whether to save figures to disk.
    show_plots : bool
        Whether to display figures interactively.
    use_sample_weights : bool
        If True (default), weight each training sample by its SQ_cycle_yield
        score so that low-quality recordings contribute less to classifier
        fitting.  Set to False for uniform weights (original behaviour).

    Returns
    -------
    dict
        Results: 'metrics_table' (pd.DataFrame), 'per_clf' (dict of per-classifier results).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ML PREDICTION PIPELINE")
    print("=" * 60)

    print("\nLoading and merging data...")
    merged = _load_and_merge(
        kinematics_csv_path,
        score_csv_path,
        id2vid_csv_path,
        score_column=score_column,
        signal_quality_threshold=signal_quality_threshold,
        signal_quality_sub_thresholds=signal_quality_sub_thresholds,
        min_cycles=min_cycles,
        min_quality_cycles=min_quality_cycles,
        min_inter_mcp_span_px=min_inter_mcp_span_px,
        min_detection_rate=min_detection_rate,
        recording_angle_csv_path=recording_angle_csv_path,
        selected_recording_angles=selected_recording_angles,
        video_quality_labels_csv_path=video_quality_labels_csv_path,
        video_quality_threshold=video_quality_threshold,
    )

    # Encode categorical metadata features for classifier input
    if "medication_state" in merged.columns:
        merged["medication_state_enc"] = (
            merged["medication_state"].apply(normalize_med_state) == "on"
        ).astype(float)
    if "hand" in merged.columns:
        merged["hand_enc"] = (merged["hand"].apply(normalize_hand) == "right").astype(float)

    print(f"  Total samples after merge: {len(merged)}")
    print("  Severity distribution:")
    for _lbl in CLASS_NAMES:
        _n = int((merged["severity_label"] == _lbl).sum())
        print(f"    {_lbl}: {_n}")

    _feat_cols = [c for c in CLASSIFICATION_FEATURES if c in merged.columns]
    _missing = [c for c in CLASSIFICATION_FEATURES if c not in merged.columns]
    if _missing:
        print(f"\n  ⚠ Missing feature columns (excluded): {_missing}")
        print("    Fix: regenerate tracking logs with all kinematic features enabled, ")
        print("         or rename legacy columns in the CSV to match expected names.")
    print(f"  Using {len(_feat_cols)} feature columns.")

    _target_col = "severity_int" if prediction_target == "severity" else "score_clean"
    _valid = merged.dropna(subset=_feat_cols + [_target_col, "subject_id"]).copy()
    print(f"  Samples with complete features: {len(_valid)}")

    if len(_valid) < 10:
        print("  ⚠ Too few complete samples for classification. Exiting.")
        return {}

    X = _valid[_feat_cols].values.astype(float)
    if prediction_target == "severity":
        y = _valid["severity_int"].values.astype(int)
    elif prediction_target == "score":
        y = _valid["score_clean"].values.astype(float)
    else:
        raise ValueError("prediction_target must be either 'severity' or 'score'.")
    groups = _valid["subject_id"].values

    # Build per-sample weights from SQ_cycle_yield (0–1 quality score).
    # Non-finite or missing values are treated as zero weight.
    sample_weights = None
    if use_sample_weights:
        if "SQ_cycle_yield" in _valid.columns:
            _sq = _valid["SQ_cycle_yield"].values.astype(float)
            _sq = np.where(np.isfinite(_sq), _sq, 0.0)
            sample_weights = np.clip(_sq, 0.0, None)
            print(
                f"  Sample weights (SQ_cycle_yield): "
                f"min={sample_weights.min():.3f}  "
                f"mean={sample_weights.mean():.3f}  "
                f"max={sample_weights.max():.3f}"
            )
        else:
            print(
                "  ⚠ use_sample_weights=True but 'SQ_cycle_yield' not in data — using uniform weights."
            )

    if prediction_target == "severity":
        models = _build_classifiers()
    else:
        models = _build_regressors()

    _severity_metrics = {"accuracy", "balanced_accuracy", "macro_precision", "macro_f1"}
    _score_metrics = {"mae", "rmse", "r2", "spearman"}
    if prediction_target == "severity" and optuna_metric not in _severity_metrics:
        print(
            f"  ⚠ optuna_metric='{optuna_metric}' is not valid for severity mode; "
            "falling back to 'balanced_accuracy'."
        )
        optuna_metric = "balanced_accuracy"
    if prediction_target == "score" and optuna_metric not in _score_metrics:
        print(
            f"  ⚠ optuna_metric='{optuna_metric}' is not valid for score mode; "
            "falling back to 'rmse'."
        )
        optuna_metric = "rmse"

    all_results = []
    per_model = {}
    tuned_models = {}

    for model_name, model in models.items():
        _eval_model = model
        _tuning_summary = None

        if optuna_trials > 0:
            print(
                f"\n  [{model_name}] Optuna tuning: "
                f"trials={optuna_trials}, metric={optuna_metric}"
            )
            try:
                if prediction_target == "severity":
                    _tuned, _summary = _tune_classifier_with_optuna(
                        clf_name=model_name,
                        X=X,
                        y=y,
                        groups=groups,
                        n_trials=optuna_trials,
                        optimize_metric=optuna_metric,
                        sample_weights=sample_weights,
                    )
                else:
                    _tuned, _summary = _tune_regressor_with_optuna(
                        reg_name=model_name,
                        X=X,
                        y=y,
                        groups=groups,
                        n_trials=optuna_trials,
                        optimize_metric=optuna_metric,
                        sample_weights=sample_weights,
                    )
                if _tuned is not None and _summary is not None:
                    _eval_model = _tuned
                    _tuning_summary = _summary
                    tuned_models[model_name] = _tuned
                    print(f"    Best {optuna_metric}: {_summary['best_value']:.2f}")
                    print(f"    Best params: {_summary['best_params']}")
                else:
                    print("    Using default hyperparameters.")
            except Exception as _tune_err:
                print(f"    ⚠ Optuna tuning failed ({_tune_err}); using defaults.")

        per_model[model_name] = {"tuning": _tuning_summary}

        if prediction_target == "severity":
            _active_modes = label_modes if label_modes is not None else ["Multi", "Ordinal"]
            for label_mode in _active_modes:
                print(f"\n  [{model_name}] label_mode={label_mode}")
                _cv_res = _loso_cv(
                    X,
                    y,
                    groups,
                    _eval_model,
                    ordinal=(label_mode == "Ordinal"),
                    sample_weights=sample_weights,
                )

                if len(_cv_res["y_true"]) == 0:
                    print("    No predictions — skipping.")
                    continue

                _metrics = _compute_metrics(_cv_res["y_true"], _cv_res["y_pred"])
                print(f"    Accuracy:          {_metrics['accuracy']:.2f}%")
                print(f"    Balanced Accuracy: {_metrics['balanced_accuracy']:.2f}%")
                print(f"    Macro Precision:   {_metrics['macro_precision']:.2f}%")
                print(f"    Macro F1:          {_metrics['macro_f1']:.2f}%")

                all_results.append(
                    {
                        "Model": model_name,
                        "Mode": label_mode,
                        "Accuracy": round(_metrics["accuracy"], 2),
                        "Balanced Acc.": round(_metrics["balanced_accuracy"], 2),
                        "Macro Precision": round(_metrics["macro_precision"], 2),
                        "Macro F1": round(_metrics["macro_f1"], 2),
                        "n": len(_cv_res["y_true"]),
                        "Tuned": "Yes" if _tuning_summary is not None else "No",
                    }
                )
                per_model[model_name][label_mode] = {**_metrics, "cv_results": _cv_res}

                if label_mode == "Multi":
                    _plot_confusion_matrix(
                        _cv_res["y_true"],
                        _cv_res["y_pred"],
                        f"{label_mode} {model_name}",
                        output_dir,
                        save_plots,
                        show_plots,
                    )
        else:
            print(f"\n  [{model_name}] direct score regression")
            _cv_res = _loso_cv_regression(
                X,
                y,
                groups,
                _eval_model,
                sample_weights=sample_weights,
            )
            if len(_cv_res["y_true"]) == 0:
                print("    No predictions — skipping.")
                continue

            _metrics = _compute_regression_metrics(_cv_res["y_true"], _cv_res["y_pred"])
            print(f"    MAE:      {_metrics['mae']:.3f}")
            print(f"    RMSE:     {_metrics['rmse']:.3f}")
            print(f"    R2:       {_metrics['r2']:.3f}")
            print(f"    Spearman: {_metrics['spearman']:.3f}")

            all_results.append(
                {
                    "Model": model_name,
                    "Mode": "Score",
                    "MAE": round(_metrics["mae"], 4),
                    "RMSE": round(_metrics["rmse"], 4),
                    "R2": round(_metrics["r2"], 4),
                    "Spearman": round(_metrics["spearman"], 4),
                    "n": len(_cv_res["y_true"]),
                    "Tuned": "Yes" if _tuning_summary is not None else "No",
                }
            )
            per_model[model_name]["Score"] = {**_metrics, "cv_results": _cv_res}
            _plot_regression_scatter(
                _cv_res["y_true"],
                _cv_res["y_pred"],
                model_name,
                output_dir,
                save_plots,
                show_plots,
            )

    if prediction_target == "severity":
        for _bname, _bmet in [
            (
                "Random Guess",
                {
                    "accuracy": 100.0 / 3,
                    "balanced_accuracy": 100.0 / 3,
                    "macro_precision": 100.0 / 3,
                    "macro_f1": 100.0 / 3,
                },
            ),
            (
                "Majority Class",
                {
                    "accuracy": float(np.max(np.bincount(y)) / len(y) * 100),
                    "balanced_accuracy": 100.0 / 3,
                    "macro_precision": float(np.max(np.bincount(y)) / len(y) * 100 / 3),
                    "macro_f1": float(np.max(np.bincount(y)) / len(y) * 100 / 3),
                },
            ),
        ]:
            all_results.append(
                {
                    "Model": _bname,
                    "Mode": "—",
                    "Accuracy": round(_bmet["accuracy"], 2),
                    "Balanced Acc.": round(_bmet["balanced_accuracy"], 2),
                    "Macro Precision": round(_bmet["macro_precision"], 2),
                    "Macro F1": round(_bmet["macro_f1"], 2),
                    "n": len(y),
                    "Tuned": "—",
                }
            )
    else:
        _base_res = _loso_cv_mean_baseline(y, groups)
        if len(_base_res["y_true"]) > 0:
            _base_metrics = _compute_regression_metrics(_base_res["y_true"], _base_res["y_pred"])
            all_results.append(
                {
                    "Model": "Train-Fold Mean Baseline",
                    "Mode": "Score",
                    "MAE": round(_base_metrics["mae"], 4),
                    "RMSE": round(_base_metrics["rmse"], 4),
                    "R2": round(_base_metrics["r2"], 4),
                    "Spearman": round(_base_metrics["spearman"], 4),
                    "n": len(_base_res["y_true"]),
                    "Tuned": "—",
                }
            )

    metrics_df = pd.DataFrame(all_results)
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(metrics_df.to_string(index=False))

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        _csv_path = os.path.join(output_dir, "classification_results.csv")
        metrics_df.to_csv(_csv_path, index=False)
        print(f"\n  Results saved → {_csv_path}")

        _tuning_rows = []
        for _model_name, _model_data in per_model.items():
            _summary = _model_data.get("tuning")
            if _summary is None:
                continue
            _tuning_rows.append(
                {
                    "Model": _model_name,
                    "Metric": _summary["metric"],
                    "Best Value": round(float(_summary["best_value"]), 4),
                    "Trials": int(_summary["n_trials"]),
                    "Best Params": json.dumps(_summary["best_params"], sort_keys=True),
                }
            )
        if _tuning_rows:
            _tuning_path = os.path.join(output_dir, "optuna_best_params.csv")
            pd.DataFrame(_tuning_rows).to_csv(_tuning_path, index=False)
            print(f"  Optuna tuning summary saved → {_tuning_path}")

    if prediction_target == "severity" and "LightGBM" in models:
        print("\n  Computing LightGBM feature importances (full dataset)...")
        from sklearn.preprocessing import StandardScaler

        _scaler_all = StandardScaler()
        X_all = _scaler_all.fit_transform(X)
        _lgbm = tuned_models.get("LightGBM", models["LightGBM"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _lgbm.fit(X_all, y)
        _plot_feature_importance(_lgbm, _feat_cols, "LightGBM", output_dir, save_plots, show_plots)
    elif prediction_target == "score" and "LightGBM Regressor" in models:
        print("\n  Computing LightGBM regressor feature importances (full dataset)...")
        from sklearn.preprocessing import StandardScaler

        _scaler_all = StandardScaler()
        X_all = _scaler_all.fit_transform(X)
        _lgbm_reg = tuned_models.get("LightGBM Regressor", models["LightGBM Regressor"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _lgbm_reg.fit(X_all, y)
        _plot_feature_importance(
            _lgbm_reg,
            _feat_cols,
            "LightGBM_Regr",
            output_dir,
            save_plots,
            show_plots,
        )

    return {"metrics_table": metrics_df, "per_model": per_model}


# CLI entry point


def _parse_args():
    p = argparse.ArgumentParser(
        description="ML prediction of PD motor severity or direct MDS-UPDRS score."
    )
    p.add_argument("--kinematics", required=True, help="Path to tracking_logs.csv")
    p.add_argument("--scores", required=True, help="Path to clinical scores CSV")
    p.add_argument("--id2vid", default="", help="Path to id2vid.csv (optional)")
    p.add_argument("--output", default="results/classification", help="Output directory")
    p.add_argument("--score-column", default="ProS", help="Score column name")
    p.add_argument(
        "--prediction-target",
        default="severity",
        choices=["severity", "score"],
        help="Target to predict: severity (3-class) or score (direct MDS-UPDRS regression)",
    )
    # Filters (mirror analyze.py)
    p.add_argument(
        "--signal-quality",
        type=float,
        default=0.0,
        help="Minimum Signal Quality threshold (0 = no filter)",
    )
    p.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.0,
        help="Minimum VQ_detection_rate (0 = no filter)",
    )
    p.add_argument(
        "--min-inter-mcp-span-px",
        type=float,
        default=0.0,
        help="Minimum VQ_inter_mcp_span_px in pixels (0 = no filter)",
    )
    p.add_argument(
        "--min-cycles", type=int, default=0, help="Minimum Total Cycles required (0 = no filter)"
    )
    p.add_argument(
        "--min-quality-cycles",
        type=int,
        default=0,
        help="Minimum Quality Cycles required (0 = no filter)",
    )
    p.add_argument(
        "--recording-angle-csv",
        default="",
        help="Path to recording-angle labels CSV/Excel (optional)",
    )
    p.add_argument(
        "--selected-recording-angles",
        nargs="*",
        default=None,
        help="Allow-list of recording angles to keep, e.g. front angled",
    )
    p.add_argument(
        "--video-quality-labels-csv",
        default="",
        help="Path to manual video-quality labels CSV with video_path and quality_label",
    )
    p.add_argument(
        "--video-quality-threshold",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Keep videos with quality_label <= threshold (1=best, 3=worst)",
    )
    # Optuna
    p.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials per classifier (0 = disabled)",
    )
    p.add_argument(
        "--optuna-metric",
        default="balanced_accuracy",
        choices=[
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_f1",
            "mae",
            "rmse",
            "r2",
            "spearman",
        ],
        help="Optuna metric (depends on prediction target)",
    )
    p.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    p.add_argument(
        "--no-sample-weights",
        action="store_true",
        help="Disable SQ_cycle_yield sample weighting (use uniform weights)",
    )
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    run_classification(
        kinematics_csv_path=_args.kinematics,
        score_csv_path=_args.scores,
        id2vid_csv_path=_args.id2vid,
        output_dir=_args.output,
        score_column=_args.score_column,
        prediction_target=_args.prediction_target,
        signal_quality_threshold=_args.signal_quality,
        min_detection_rate=_args.min_detection_rate,
        min_inter_mcp_span_px=_args.min_inter_mcp_span_px,
        min_cycles=_args.min_cycles,
        min_quality_cycles=_args.min_quality_cycles,
        recording_angle_csv_path=_args.recording_angle_csv or None,
        selected_recording_angles=_args.selected_recording_angles,
        video_quality_labels_csv_path=_args.video_quality_labels_csv or None,
        video_quality_threshold=_args.video_quality_threshold,
        optuna_trials=_args.optuna_trials,
        optuna_metric=_args.optuna_metric,
        save_plots=True,
        show_plots=_args.show_plots,
        use_sample_weights=not _args.no_sample_weights,
    )
