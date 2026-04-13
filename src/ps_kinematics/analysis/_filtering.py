"""Recording angle, video quality, and segment inclusion filters."""

import os

import pandas as pd
from pandas.errors import EmptyDataError

from ps_kinematics.io import (
    normalize_video_path_series_for_matching,
    read_xlsx_stdlib,
)


def segment_inclusion_mask(
    names: "pd.Series",
    segmented_only: bool,
    non_segmented: bool,
) -> "pd.Series":
    """Return a row mask based on segmented/non-segmented inclusion toggles.

    Rules:
      - both False: keep all rows
      - segmented_only=True: keep rows whose name contains "segmented"
      - non_segmented=True: keep rows whose name does NOT contain "segmented"
      - both True: keep none (empty result)
    """
    if segmented_only and non_segmented:
        return pd.Series(False, index=names.index)

    lower_names = names.fillna("").astype(str).str.lower()
    has_segmented = lower_names.str.contains("segmented", na=False)

    if segmented_only:
        return has_segmented
    if non_segmented:
        return ~has_segmented
    return pd.Series(True, index=names.index)


def canonicalize_recording_angle(angle_value) -> str | None:
    """Normalize recording-angle labels to a stable lowercase token."""
    if pd.isna(angle_value):
        return None
    angle = str(angle_value).strip().lower()
    if not angle:
        return None
    aliases = {
        "frontal": "front",
        "front view": "front",
        "angled view": "angled",
        "lateral view": "lateral",
        "side": "lateral",
    }
    return aliases.get(angle, angle)


def normalize_recording_angle_filter(
    selected_recording_angles: list[str] | None,
) -> set[str] | None:
    """Convert user angle filter list to a canonical set; None means no filter."""
    if selected_recording_angles is None:
        return None
    normalized = set()
    for v in selected_recording_angles:
        norm_v = canonicalize_recording_angle(v)
        if norm_v is not None:
            normalized.add(norm_v)
    return normalized


def load_recording_angle_labels(recording_angle_csv_path: str) -> "pd.DataFrame":
    """Load recording-angle labels and attach a normalized video-path join key."""
    _ext = os.path.splitext(recording_angle_csv_path)[1].lower()
    if _ext in (".xlsx", ".xls"):
        try:
            angle_df = read_xlsx_stdlib(recording_angle_csv_path)
        except Exception:
            angle_df = pd.read_excel(recording_angle_csv_path)
    else:
        angle_df = pd.read_csv(recording_angle_csv_path)

    if "recording_angle" not in angle_df.columns:
        raise RuntimeError("recording angle file must contain a 'recording_angle' column")

    if "video_path" in angle_df.columns:
        angle_df = angle_df.copy()
        angle_df["_angle_key"] = normalize_video_path_series_for_matching(angle_df["video_path"])
    elif "video_id" in angle_df.columns:
        angle_df = angle_df.copy()
        angle_df["_angle_key"] = angle_df["video_id"].astype(str).str.strip().str.lower()
    else:
        raise RuntimeError("recording angle file must contain either 'video_path' or 'video_id'")

    angle_df["recording_angle"] = angle_df["recording_angle"].apply(canonicalize_recording_angle)
    angle_df = angle_df.dropna(subset=["recording_angle", "_angle_key"]).copy()
    return angle_df


def load_video_quality_labels(video_quality_csv_path: str) -> "pd.DataFrame":
    """Load manual video-quality labels and attach a normalized join key.

    Expected columns:
      - ``video_path``
      - ``quality_label`` with values in {1, 2, 3} (1=best, 3=worst)
    """
    if not os.path.exists(video_quality_csv_path):
        raise FileNotFoundError(f"Video quality labels CSV not found: {video_quality_csv_path}")
    if os.path.getsize(video_quality_csv_path) == 0:
        raise ValueError(
            "Video quality labels CSV is empty (0 bytes): "
            f"{video_quality_csv_path}. "
            "Provide a video quality labels CSV (see config.example.yaml)."
        )

    try:
        quality_df = pd.read_csv(video_quality_csv_path)
    except EmptyDataError as exc:
        raise ValueError(
            "Video quality labels CSV has no parseable columns: "
            f"{video_quality_csv_path}. "
            "Expected columns: video_path, quality_label"
        ) from exc

    if quality_df.empty:
        raise ValueError(
            "Video quality labels CSV contains no rows: "
            f"{video_quality_csv_path}. Add at least one labeled video."
        )

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

    quality_df["_quality_key"] = normalize_video_path_series_for_matching(quality_df["video_path"])
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


def apply_video_quality_filter(
    df: "pd.DataFrame",
    quality_df: "pd.DataFrame",
    quality_threshold: int,
    *,
    video_path_column: str,
    stage_name: str,
) -> "pd.DataFrame":
    """Keep only rows with manual quality_label <= threshold.

    Rows without a matching quality label are excluded when this filter is active.
    """
    if quality_threshold not in (1, 2, 3):
        raise ValueError("video_quality_threshold must be one of: 1, 2, 3")
    if video_path_column not in df.columns:
        raise RuntimeError(
            f"{stage_name}: required column '{video_path_column}' is missing; "
            "cannot apply video quality threshold"
        )

    if df.empty:
        return df

    df = df.copy()
    df["_quality_key"] = normalize_video_path_series_for_matching(df[video_path_column])
    quality_map = quality_df.set_index("_quality_key")["quality_label"]
    matched_labels = df["_quality_key"].map(quality_map)

    n_before = len(df)
    n_matched = int(matched_labels.notna().sum())
    keep_mask = matched_labels.notna() & (matched_labels <= int(quality_threshold))
    df = df[keep_mask].copy()
    if not df.empty:
        df["quality_label"] = matched_labels[keep_mask].astype(int).to_numpy()
    df = df.drop(columns=["_quality_key"], errors="ignore")

    print(
        f"  Manual video quality filter ({stage_name}, threshold <= {quality_threshold}): "
        f"matched {n_matched}/{n_before}, kept {len(df)}/{n_before} "
        f"({n_before - len(df)} dropped)"
    )
    return df
