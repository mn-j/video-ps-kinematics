#!/usr/bin/env python3
"""
prepare_yolo_pseudolabels.py — Generate a YOLO-Pose training dataset from
MediaPipe pseudo-labels on high-quality PD pronation-supination videos.

Workflow
--------
1. Load ``tracking_logs.csv`` produced by the main pipeline.
2. Compute a Pseudo-Label Eligibility (PLE) score per video.
3. Select top-quality videos via an adaptive elbow threshold.
4. Re-run MediaPipe on selected videos to extract per-frame (21, 4) landmarks.
5. Apply per-frame quality filters (MCP confidence, boundary, bbox size).
6. Save filtered frames as JPEG + YOLO-pose label files.
7. Video-stratified train/val split.
8. Write YOLO dataset YAML config.

Usage
-----
    python scripts/prepare_yolo_pseudolabels.py \\
        --tracking-logs output/tracking_logs.csv \\
        --output-dir datasets/yolo_pd_hand \\
        [--min-frames 5000] [--max-frames 25000] \\
        [--val-frac 0.2] [--seed 42]

Requires: mediapipe, opencv-python, pandas, numpy, matplotlib (for plot)
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root used only for discovering the hand_landmarker.task model file.
_PROJECT_ROOT_FOR_MODEL_LOOKUP = str(Path(__file__).resolve().parent.parent)

from ps_kinematics.core import _compute_mcp_confidence_proxy
from ps_kinematics.utils import (
    YOLO_PD_BBOX_LABEL_PADDING,
    YOLO_PD_FRAME_BOUNDARY_MARGIN,
    YOLO_PD_MIN_BBOX_FRAC,
    YOLO_PD_MIN_MCP_CONF,
)

logger = logging.getLogger(__name__)

# MCP indices used for confidence proxy (same as kinematic pipeline)
_MCP_INDICES = [5, 9, 13, 17]  # INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP

# Kinematically-relevant keypoints: the only 5 landmarks the angle pipeline uses.
# Training YOLO on all 21 MediaPipe landmarks dilutes the loss signal and
# trains on noisier fingertip/DIP/PIP pseudo-labels that are irrelevant to
# the pronation-supination angle computation.
_LABEL_KPT_INDICES = [0, 5, 9, 13, 17]  # Wrist, Index MCP, Middle MCP, Ring MCP, Pinky MCP

# ---------------------------------------------------------------------------
# 1. PLE Score
# ---------------------------------------------------------------------------

# PLE component weights (total = 11.5, normalised to [0, 1])
_PLE_WEIGHTS = {
    "signal_quality": 3.0,
    "detection_rate": 2.0,
    "mcp_confidence": 2.0,
    "sharpness_norm": 1.5,
    "gap_complement": 1.5,
    "bbox_area_norm": 1.0,
    "luminance_quality": 0.5,
}
_PLE_TOTAL_WEIGHT = sum(_PLE_WEIGHTS.values())


def _safe_float(val, default=0.0):
    """Convert *val* to float, returning *default* on failure or NaN."""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _parse_json_series(val):
    """Parse a JSON-serialised numeric array from a CSV cell."""
    if isinstance(val, (list, np.ndarray)):
        return np.asarray(val, dtype=np.float64)
    if not isinstance(val, str) or not val.strip():
        return np.array([], dtype=np.float64)
    try:
        arr = json.loads(val)
        return np.asarray(arr, dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError):
        return np.array([], dtype=np.float64)


def compute_ple_scores(df: pd.DataFrame) -> pd.Series:
    """Compute per-video Pseudo-Label Eligibility scores.

    Parameters
    ----------
    df : pd.DataFrame
        The tracking_logs DataFrame (one row per video).

    Returns
    -------
    pd.Series
        PLE scores in [0, 1], indexed like *df*.
    """
    w = _PLE_WEIGHTS

    sig_q = df["Signal Quality"].apply(lambda v: _safe_float(v, 0.0))
    det_rate = df["VQ_detection_rate"].apply(lambda v: _safe_float(v, 0.0))

    # MCP confidence: median of per-frame min-across-keypoints confidence.
    # Stored as JSON array in conf_mcp_min_series column.
    def _mcp_median(val):
        arr = _parse_json_series(val)
        arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if len(arr) > 0 else 0.0

    mcp_conf = df["conf_mcp_min_series"].apply(_mcp_median)

    # Sharpness: min-max normalise across the dataset
    sharpness_raw = df["VQ_sharpness_median"].apply(lambda v: _safe_float(v, 0.0))
    s_min, s_max = sharpness_raw.min(), sharpness_raw.max()
    sharpness_norm = (sharpness_raw - s_min) / max(s_max - s_min, 1e-8)

    # Gap complement: 1 - gap_fraction
    gap_frac = df["VQ_gap_fraction"].apply(lambda v: _safe_float(v, 1.0))
    gap_complement = 1.0 - gap_frac

    # Bbox area: min-max normalise
    bbox_raw = df["VQ_hand_bbox_area_median_px"].apply(lambda v: _safe_float(v, 0.0))
    b_min, b_max = bbox_raw.min(), bbox_raw.max()
    bbox_norm = (bbox_raw - b_min) / max(b_max - b_min, 1e-8)

    # Luminance quality: 1 - |median - 128| / 128
    lum_raw = df["VQ_luminance_median"].apply(lambda v: _safe_float(v, 128.0))
    lum_quality = 1.0 - (lum_raw - 128.0).abs() / 128.0
    lum_quality = lum_quality.clip(0.0, 1.0)

    ple = (
        w["signal_quality"] * sig_q
        + w["detection_rate"] * det_rate
        + w["mcp_confidence"] * mcp_conf
        + w["sharpness_norm"] * sharpness_norm
        + w["gap_complement"] * gap_complement
        + w["bbox_area_norm"] * bbox_norm
        + w["luminance_quality"] * lum_quality
    ) / _PLE_TOTAL_WEIGHT

    return ple.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Threshold Selection (adaptive elbow)
# ---------------------------------------------------------------------------


def find_selection_threshold(
    ple_scores: pd.Series,
    min_frames: int = 5000,
    max_frames: int = 25000,
    min_videos: int = 50,
    total_frames_col: pd.Series = None,
) -> float:
    """Find the PLE threshold via an adaptive elbow method.

    Parameters
    ----------
    ple_scores : pd.Series
        Per-video PLE scores (same index as tracking_logs).
    min_frames, max_frames : int
        Hard constraints on total extracted frames.
    min_videos : int
        Minimum number of videos to select regardless of elbow position.
        Prevents the elbow from landing in the extreme right tail when the
        PLE distribution is unimodal (no natural knee).  100 is a good
        default — still top ~7% of a 1400-video dataset, with enough
        patient diversity for robust generalisation.
    total_frames_col : pd.Series
        Per-video total frame count (for cumulative frame budget).

    Returns
    -------
    float
        PLE threshold — videos with PLE >= threshold are selected.
    """
    if total_frames_col is None:
        # Assume ~250 frames per video if frame counts unavailable
        total_frames_col = pd.Series(250, index=ple_scores.index)

    # Sort descending by PLE
    order = ple_scores.sort_values(ascending=False)
    cum_frames = total_frames_col.loc[order.index].cumsum()

    # Find the elbow: where d(PLE)/d(cum_frames) steepens beyond 2x median
    ple_vals = order.values
    cf_vals = cum_frames.values

    if len(ple_vals) < 3:
        return float(ple_vals[-1]) if len(ple_vals) > 0 else 0.0

    gradients = np.abs(np.diff(ple_vals)) / np.maximum(np.diff(cf_vals), 1.0)
    median_grad = float(np.median(gradients[gradients > 0])) if np.any(gradients > 0) else 1e-8

    elbow_idx = None
    for i, g in enumerate(gradients):
        if g > 2.0 * median_grad and cf_vals[i + 1] >= min_frames:
            elbow_idx = i + 1
            break

    # Apply frame constraints
    if elbow_idx is None:
        # No clear elbow — use max_frames as budget
        candidates = np.where(cf_vals <= max_frames)[0]
        elbow_idx = candidates[-1] if len(candidates) > 0 else len(ple_vals) - 1

    # Ensure minimum frame count
    if cf_vals[elbow_idx] < min_frames:
        candidates = np.where(cf_vals >= min_frames)[0]
        if len(candidates) > 0:
            elbow_idx = candidates[0]
        else:
            elbow_idx = len(ple_vals) - 1  # use all videos

    # Ensure maximum frame count
    if cf_vals[elbow_idx] > max_frames:
        candidates = np.where(cf_vals <= max_frames)[0]
        if len(candidates) > 0:
            elbow_idx = candidates[-1]

    # Ensure minimum video count — prevents elbow from landing in the extreme
    # right tail when the PLE distribution is unimodal with no natural knee.
    min_idx = min(min_videos - 1, len(ple_vals) - 1)
    if elbow_idx < min_idx:
        elbow_idx = min_idx

    threshold = float(ple_vals[elbow_idx])
    return threshold


def save_ple_diagnostic_plot(ple_scores, threshold, output_path):
    """Save a PLE distribution histogram with the threshold marked."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping PLE diagnostic plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ple_scores.values, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.3f}"
    )
    n_selected = int((ple_scores >= threshold).sum())
    ax.set_title(f"PLE Distribution — {n_selected} videos selected " f"(of {len(ple_scores)})")
    ax.set_xlabel("PLE Score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("PLE diagnostic plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# 3. MediaPipe Landmark Extraction
# ---------------------------------------------------------------------------


def _extract_landmarks_mediapipe(video_path, hand_to_track, hand_model_path=None):
    """Run MediaPipe on a video and return per-frame landmarks.

    Uses IMAGE mode per-frame (independent detections, no temporal smoothing)
    to get honest per-frame landmark quality for pseudo-label generation.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    hand_to_track : str
        "Left" or "Right".
    hand_model_path : str or None
        Explicit path to hand_landmarker.task.  If None, falls back to env var
        and well-known locations.

    Returns
    -------
    frames_dict : dict
        ``{frame_idx: np.ndarray (21, 4)}`` for detected frames.
    total_frames : int
    frame_hw : tuple
        (height, width) in pixels.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    # Locate the MediaPipe hand landmarker model
    model_path = _find_mediapipe_model(explicit_path=hand_model_path)
    if model_path is None:
        logger.error("MediaPipe hand_landmarker.task model not found.")
        return {}, 0, (0, 0)

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_dict = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return {}, 0, (0, 0)

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = 0

    expected_label = hand_to_track.capitalize()

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx = total_frames
            total_frames += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                continue

            # Find the detection matching the expected handedness
            best_lm = None
            for hand_i, lm_list in enumerate(result.hand_landmarks):
                if hand_i < len(result.handedness):
                    cats = result.handedness[hand_i]
                    if cats and cats[0].category_name == expected_label:
                        lm_arr = np.array(
                            [
                                [
                                    lm.x,
                                    lm.y,
                                    lm.z,
                                    float(lm.visibility) if lm.visibility is not None else 1.0,
                                ]
                                for lm in lm_list
                            ],
                            dtype=np.float32,
                        )
                        best_lm = lm_arr
                        break

            if best_lm is not None:
                frames_dict[frame_idx] = best_lm

    cap.release()
    return frames_dict, total_frames, (frame_h, frame_w)


def _find_mediapipe_model(explicit_path=None):
    """Locate the MediaPipe hand_landmarker.task model file.

    Parameters
    ----------
    explicit_path : str or None
        Caller-supplied path (e.g. from CONFIG["hand_path"]).  Checked first.
    """
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    # Fall back to the env var used by the main pipeline
    _env_path = os.environ.get("PD_PS_HAND_PATH", "")
    if _env_path:
        candidates.append(_env_path)
    candidates += [
        os.path.join(_PROJECT_ROOT_FOR_MODEL_LOOKUP, "models", "hand_landmarker.task"),
        os.path.join(_PROJECT_ROOT_FOR_MODEL_LOOKUP, "hand_landmarker.task"),
    ]
    # Also check common mediapipe cache locations
    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, ".cache", "mediapipe", "hand_landmarker.task"))

    for p in candidates:
        if os.path.exists(p):
            return p

    # Try to find it via glob
    import glob

    matches = glob.glob(
        os.path.join(_PROJECT_ROOT_FOR_MODEL_LOOKUP, "**", "hand_landmarker.task"), recursive=True
    )
    if matches:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# 4. Per-Frame Quality Filtering
# ---------------------------------------------------------------------------


def _filter_frames(
    frames_dict,
    total_frames,
    frame_h,
    frame_w,
    min_mcp_conf=YOLO_PD_MIN_MCP_CONF,
    boundary_margin=YOLO_PD_FRAME_BOUNDARY_MARGIN,
    min_bbox_frac=YOLO_PD_MIN_BBOX_FRAC,
):
    """Apply per-frame quality filters and return accepted frame indices.

    Filters:
    1. MCP confidence proxy > min_mcp_conf
    2. No keypoint within boundary_margin of frame edge
    3. Temporal consistency: within 2-sigma of 5-frame rolling median
    4. Hand bbox area > min_bbox_frac of frame area

    Returns
    -------
    accepted : set of int
        Frame indices that pass all filters.
    """
    if not frames_dict:
        return set()

    # 1. MCP confidence proxy
    _, conf_min, _ = _compute_mcp_confidence_proxy(
        frames_dict,
        total_frames,
        _MCP_INDICES,
        window=5,
    )
    conf_ok = set()
    for fi in frames_dict:
        if fi < len(conf_min) and np.isfinite(conf_min[fi]) and conf_min[fi] >= min_mcp_conf:
            conf_ok.add(fi)

    accepted = set(frames_dict.keys()) & conf_ok

    # 2. Boundary margin: no keypoint within margin of edge
    margin = boundary_margin
    to_remove = set()
    for fi in accepted:
        lm = frames_dict[fi]  # (21, 4)
        xs, ys = lm[:, 0], lm[:, 1]
        if (
            np.any(xs < margin)
            or np.any(xs > 1.0 - margin)
            or np.any(ys < margin)
            or np.any(ys > 1.0 - margin)
        ):
            to_remove.add(fi)
    accepted -= to_remove

    # 3. Temporal consistency: per-keypoint, within 2-sigma of 5-frame rolling median
    if len(accepted) > 10:
        sorted_frames = sorted(accepted)
        half_w = 2
        to_remove = set()
        for fi in sorted_frames:
            lm = frames_dict[fi]  # (21, 4)
            # Get neighbours within ±2 frames
            neighbours = [
                f for f in sorted_frames if abs(f - fi) <= half_w and f != fi and f in accepted
            ]
            if len(neighbours) < 2:
                continue  # not enough context to filter
            nb_lms = np.stack([frames_dict[f][:, :2] for f in neighbours])  # (N, 21, 2)
            medians = np.median(nb_lms, axis=0)  # (21, 2)
            stds = np.std(nb_lms, axis=0)  # (21, 2)
            stds = np.maximum(stds, 0.005)  # floor to prevent zero-std
            diffs = np.abs(lm[:, :2] - medians)
            if np.any(diffs > 2.0 * stds):
                to_remove.add(fi)
        accepted -= to_remove

    # 4. Minimum hand bbox area
    frame_area = frame_h * frame_w
    to_remove = set()
    for fi in accepted:
        lm = frames_dict[fi]
        xs = lm[:, 0] * frame_w
        ys = lm[:, 1] * frame_h
        bbox_w = float(np.max(xs) - np.min(xs))
        bbox_h = float(np.max(ys) - np.min(ys))
        if (bbox_w * bbox_h) < min_bbox_frac * frame_area:
            to_remove.add(fi)
    accepted -= to_remove

    return accepted


# ---------------------------------------------------------------------------
# 5. YOLO Label Generation
# ---------------------------------------------------------------------------


def landmarks_to_yolo_label(lm_arr, frame_h, frame_w, bbox_padding=YOLO_PD_BBOX_LABEL_PADDING):
    """Convert (21, 4) normalised landmarks to a YOLO-pose label line.

    Format per line (single class 0 = hand):
        class_id  cx cy w h  x1 y1 v1  x2 y2 v2  ...  x21 y21 v21

    All coordinates are normalised [0, 1].  Visibility flags: 2 = visible.

    Parameters
    ----------
    lm_arr : np.ndarray (21, 4)
        Normalised landmarks (x, y, z, visibility).
    frame_h, frame_w : int
        Frame pixel dimensions.
    bbox_padding : float
        Fractional padding for the bounding box.

    Returns
    -------
    str
        Single-line YOLO label string.
    """
    xs = lm_arr[:, 0]  # normalised [0, 1]
    ys = lm_arr[:, 1]

    # Bounding box from landmarks (normalised coords)
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    bw = x_max - x_min
    bh = y_max - y_min
    pad_x = bw * bbox_padding
    pad_y = bh * bbox_padding

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = bw + 2 * pad_x
    h = bh + 2 * pad_y

    # Clip to [0, 1]
    cx = np.clip(cx, 0.0, 1.0)
    cy = np.clip(cy, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0)
    h = np.clip(h, 0.0, 1.0)

    parts = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]

    for i in _LABEL_KPT_INDICES:
        kx = float(np.clip(lm_arr[i, 0], 0.0, 1.0))
        ky = float(np.clip(lm_arr[i, 1], 0.0, 1.0))
        parts.append(f"{kx:.6f} {ky:.6f} 2")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# 6. Main Dataset Extraction
# ---------------------------------------------------------------------------


def extract_dataset(
    tracking_logs_csv,
    output_dir,
    min_frames=5000,
    max_frames=25000,
    min_videos=150,
    val_frac=0.2,
    seed=42,
    hand_model_path=None,
    front_view_only=True,
):
    """End-to-end pseudo-label dataset extraction.

    Parameters
    ----------
    tracking_logs_csv : str
        Path to the pipeline's tracking_logs.csv.
    output_dir : str
        Root directory for the YOLO dataset.
    min_frames, max_frames : int
        Frame budget constraints.
    min_videos : int
        Minimum number of videos to select regardless of elbow position.
    val_frac : float
        Fraction of videos held out for validation.
    seed : int
        Random seed for reproducible train/val split.
    front_view_only : bool
        When True (default) and a ``recording_angle`` column is present in the
        tracking logs, restrict training to front-view videos only.
        Angled/lateral views produce noisier MCP pseudo-labels (worse hand
        visibility mid-rotation) and hurt label quality more than they help
        with pose diversity.  At inference time YOLO still runs on all frames
        — the per-keypoint confidence threshold provides a natural fallback to
        MediaPipe for non-frontal frames.

    Returns
    -------
    str
        Path to the generated dataset YAML file.
    """
    import cv2

    output_dir = os.path.abspath(output_dir)

    logger.info("Loading tracking logs from %s", tracking_logs_csv)
    df = pd.read_csv(tracking_logs_csv)

    # Filter to successfully processed videos only
    if "record_type" in df.columns:
        df = df[df["record_type"] == "VIDEO"].copy()
    required_cols = [
        "Signal Quality",
        "VQ_detection_rate",
        "VQ_sharpness_median",
        "VQ_gap_fraction",
        "VQ_hand_bbox_area_median_px",
        "VQ_luminance_median",
        "conf_mcp_min_series",
        "video_path",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in tracking_logs.csv: {missing}")

    # Drop rows with no video path
    df = df.dropna(subset=["video_path"]).copy()
    df = df[df["video_path"].apply(lambda p: os.path.isfile(str(p)))].copy()
    logger.info("Found %d videos with valid paths.", len(df))

    if len(df) == 0:
        raise RuntimeError("No valid videos found in tracking_logs.csv.")

    # ── Front-view filtering ─────────────────────────────────────────
    if front_view_only and "recording_angle" in df.columns:
        n_before = len(df)
        df = df[df["recording_angle"].str.lower() == "front"].copy()
        n_after = len(df)
        logger.info(
            "front_view_only=True: retained %d/%d videos with recording_angle='front'.",
            n_after,
            n_before,
        )
        if n_after == 0:
            raise RuntimeError(
                "No front-view videos found after recording_angle filter. "
                "Provide recording angle annotations, or "
                "pass front_view_only=False to disable this filter."
            )
    elif front_view_only and "recording_angle" not in df.columns:
        logger.warning(
            "front_view_only=True but 'recording_angle' column not found in "
            "tracking_logs.csv — skipping angle filter. Run "
            "provide recording angle annotations and merge the output to enable it."
        )

    # ── PLE scoring ──────────────────────────────────────────────────
    df["ple_score"] = compute_ple_scores(df)

    total_frames_col = df["total_frames"].apply(lambda v: _safe_float(v, 250))
    threshold = find_selection_threshold(
        df["ple_score"],
        min_frames=min_frames,
        max_frames=max_frames,
        min_videos=min_videos,
        total_frames_col=total_frames_col,
    )
    logger.info("PLE threshold: %.4f", threshold)

    selected = df[df["ple_score"] >= threshold].copy()
    logger.info(
        "Selected %d videos (PLE >= %.4f), est. %d total frames.",
        len(selected),
        threshold,
        int(total_frames_col.loc[selected.index].sum()),
    )

    # Save diagnostic plot
    os.makedirs(output_dir, exist_ok=True)
    save_ple_diagnostic_plot(
        df["ple_score"],
        threshold,
        os.path.join(output_dir, "ple_distribution.png"),
    )

    # ── Video-stratified train/val split ─────────────────────────────
    rng = random.Random(seed)
    video_indices = list(selected.index)
    rng.shuffle(video_indices)
    n_val = max(1, int(len(video_indices) * val_frac))
    val_indices = set(video_indices[:n_val])
    train_indices = set(video_indices[n_val:])
    logger.info("Split: %d train videos, %d val videos.", len(train_indices), len(val_indices))

    # ── Clear stale frame/label files from any previous run ──────────
    # Must wipe before writing so that old train/val assignments don't
    # survive a re-run with a different split (data leakage).

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            stale_dir = os.path.join(output_dir, sub, split)
            if os.path.isdir(stale_dir):
                shutil.rmtree(stale_dir)
                logger.info("Cleared stale directory: %s", stale_dir)

    # ── Create output directories ────────────────────────────────────
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # ── Extract frames and labels ────────────────────────────────────
    total_saved = {"train": 0, "val": 0}

    for row_idx, row in selected.iterrows():
        video_path = str(row["video_path"])
        hand = str(row.get("hand", row.get("log_hand", "Right")))
        video_stem = Path(video_path).stem
        split = "val" if row_idx in val_indices else "train"

        logger.info("Processing [%s] %s (PLE=%.3f) ...", split, video_stem, row["ple_score"])

        frames_dict, total_frames, (fh, fw) = _extract_landmarks_mediapipe(
            video_path,
            hand,
            hand_model_path=hand_model_path,
        )
        if not frames_dict:
            logger.warning("  No landmarks extracted from %s, skipping.", video_stem)
            continue

        accepted = _filter_frames(frames_dict, total_frames, fh, fw)
        if not accepted:
            logger.warning("  No frames passed quality filters for %s.", video_stem)
            continue

        # Re-open video to extract frame images
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        saved_this_video = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx in accepted:
                img_name = f"{video_stem}_f{frame_idx:06d}.jpg"
                img_path = os.path.join(output_dir, "images", split, img_name)
                lbl_name = f"{video_stem}_f{frame_idx:06d}.txt"
                lbl_path = os.path.join(output_dir, "labels", split, lbl_name)

                cv2.imwrite(img_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                label_line = landmarks_to_yolo_label(
                    frames_dict[frame_idx],
                    fh,
                    fw,
                )
                with open(lbl_path, "w") as f:
                    f.write(label_line + "\n")
                saved_this_video += 1

            frame_idx += 1
        cap.release()

        total_saved[split] += saved_this_video
        logger.info("  Saved %d frames from %s.", saved_this_video, video_stem)

    logger.info("Total frames saved — train: %d, val: %d", total_saved["train"], total_saved["val"])

    if total_saved["train"] == 0:
        raise RuntimeError(
            "No training frames were extracted. Check video paths in "
            "tracking_logs.csv and MediaPipe model availability."
        )

    # ── Write dataset YAML ───────────────────────────────────────────
    yaml_path = create_dataset_yaml(output_dir)
    logger.info("Dataset YAML written to %s", yaml_path)
    return yaml_path


def create_dataset_yaml(output_dir):
    """Write the YOLO dataset configuration YAML.

    Parameters
    ----------
    output_dir : str
        Root directory of the dataset (contains images/ and labels/).

    Returns
    -------
    str
        Path to the written YAML file.
    """
    output_dir = os.path.abspath(output_dir)
    yaml_path = os.path.join(output_dir, "dataset.yaml")

    # Use YAML-safe format (no PyYAML dependency required)
    content = (
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"# Single class: hand\n"
        f"names:\n"
        f"  0: hand\n"
        f"\n"
        f"# 5 kinematic keypoints (wrist + 4 MCPs), each with (x, y, visibility)\n"
        f"# Order: wrist(0), index_mcp(5), middle_mcp(9), ring_mcp(13), pinky_mcp(17)\n"
        f"kpt_shape: [5, 3]\n"
    )

    with open(yaml_path, "w") as f:
        f.write(content)

    return yaml_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate YOLO-Pose training dataset from MediaPipe pseudo-labels.",
    )
    parser.add_argument(
        "--tracking-logs",
        required=True,
        help="Path to tracking_logs.csv from the main pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/yolo_pd_hand",
        help="Root directory for the YOLO dataset output.",
    )
    parser.add_argument(
        "--min-videos",
        type=int,
        default=150,
        help="Minimum number of videos to select regardless of elbow position (default: 150).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5000,
        help="Minimum total frames to extract (default: 5000).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=35000,
        help="Maximum total frames to extract (default: 25000).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of videos for validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42).",
    )
    parser.add_argument(
        "--front-view-only",
        action="store_true",
        default=True,
        help="Restrict training to front-view videos (recording_angle='front'). "
        "Requires recording_angle column in tracking_logs.csv. Default: True.",
    )
    parser.add_argument(
        "--no-front-view-only",
        dest="front_view_only",
        action="store_false",
        help="Disable front-view restriction and include all recording angles.",
    )
    args = parser.parse_args()

    yaml_path = extract_dataset(
        tracking_logs_csv=args.tracking_logs,
        output_dir=args.output_dir,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        min_videos=args.min_videos,
        val_frac=args.val_frac,
        seed=args.seed,
        front_view_only=args.front_view_only,
    )
    print(f"\nDataset ready. YAML config: {yaml_path}")
    print("Next step: fine-tune with")
    print(
        f'  python -c "from ps_kinematics.refinement.yolo import train_yolo_pd_hand_model; '
        f"train_yolo_pd_hand_model('{yaml_path}', 'models/yolo_hand_pose.pt', "
        f"'models/yolo_pd_hand_pose.pt')\""
    )


if __name__ == "__main__":
    main()
