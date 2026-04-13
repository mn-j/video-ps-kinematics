"""
ps_kinematics.core — Shared tracking/landmark utilities and backward-compatible re-exports.

The large classes and worker functions have been split into dedicated modules:
  - tracker.py   → MultiHandOfflineTracker
  - processor.py → HandLandmarkProcessor
  - workers.py   → standalone multiprocessing worker functions

This module retains shared helper functions used by both the processor and
workers, and re-exports the main classes so that existing imports from
``ps_kinematics.core`` continue to work.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2

    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision.hand_landmarker import (  # noqa: F401
        HandLandmark,
        HandLandmarksConnections,
    )

    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

    class HandLandmark:
        WRIST = 0
        INDEX_FINGER_MCP = 5
        MIDDLE_FINGER_MCP = 9
        RING_FINGER_MCP = 13
        PINKY_MCP = 17


import pandas as pd

from . import utils as _u
from .kinematics import _proxy_angle_deg
from .utils import (
    LANDMARK_SMOOTH_WINDOW,
    MAX_PS_DURATION_S,
    MIN_PS_DURATION_S,
    ONEEURO_BETA,
    ONEEURO_D_CUTOFF,
    ONEEURO_MIN_CUTOFF,
    PS_ACTIVITY_PERCENTILE,
    PS_ACTIVITY_THRESHOLD_RATIO,
    PS_ACTIVITY_WINDOW_S,
    PS_MERGE_GAP_S,
    ROI_REDETECT_PADDING,
    USE_ONE_EURO,
    OneEuroFilter,
)


def _lm_vis(lm, default=1.0):
    """Extract visibility from a MediaPipe landmark, defaulting when absent.

    MediaPipe Hand Landmarker does *not* populate ``visibility`` on
    ``NormalizedLandmark``; the attribute exists but is ``None``.
    ``getattr(lm, 'visibility', 1.0)`` therefore returns ``None`` instead
    of the intended fallback, which numpy then converts to ``nan``.
    """
    v = getattr(lm, "visibility", None)
    return float(v) if v is not None else default


def _compute_mcp_confidence_proxy(frames_dict, total_frames, mcp_indices, window=5):
    """Per-MCP-keypoint per-frame temporal-stability confidence proxy.

    MediaPipe Hand Landmarker does not provide per-landmark confidence.
    This function derives a 0-1 proxy from positional jitter: for each
    MCP keypoint at each frame the Euclidean displacement from its
    rolling-median position (over *window* frames) is computed, normalised
    by the median inter-MCP span, and mapped through a logistic:

        conf = 1 / (1 + displacement / scale)

    where ``scale = 0.05 * median_inter_mcp_span``.  A value of 1.0 means
    the keypoint sits exactly on its local median; values near 0 indicate
    extreme jitter relative to hand size.

    Returns
    -------
    conf : dict
        ``{mcp_index: np.ndarray}`` of shape ``(total_frames,)`` per keypoint.
    conf_min : np.ndarray
        Per-frame minimum confidence across all MCP keypoints.
    conf_thresh : float
        Fixed confidence threshold (0.5) for the used-mask.
    """
    n_kp = len(mcp_indices)

    # --- Extract (x, y) time-series per keypoint ---
    xy = np.full((n_kp, total_frames, 2), np.nan, dtype=np.float64)
    for f_idx, lm_arr in frames_dict.items():
        fi = int(f_idx)
        if fi < 0 or fi >= total_frames:
            continue
        for ki, li in enumerate(mcp_indices):
            xy[ki, fi, 0] = float(lm_arr[li, 0])
            xy[ki, fi, 1] = float(lm_arr[li, 1])

    # --- Median inter-MCP span (index-to-pinky) for normalisation ---
    span_vals = []
    for f_idx, lm_arr in frames_dict.items():
        fi = int(f_idx)
        if fi < 0 or fi >= total_frames:
            continue
        p0 = np.array([float(lm_arr[mcp_indices[0], 0]), float(lm_arr[mcp_indices[0], 1])])
        p3 = np.array([float(lm_arr[mcp_indices[-1], 0]), float(lm_arr[mcp_indices[-1], 1])])
        span_vals.append(float(np.linalg.norm(p0 - p3)))

    median_span = float(np.median(span_vals)) if span_vals else 0.0
    scale = max(median_span * 0.05, 1e-8)

    # --- Per-keypoint rolling-median displacement → confidence ---
    half_w = window // 2
    conf_arrays = {}
    for ki, li in enumerate(mcp_indices):
        c = np.full(total_frames, np.nan, dtype=np.float64)
        x_s = xy[ki, :, 0]
        y_s = xy[ki, :, 1]
        detected = np.where(np.isfinite(x_s))[0]
        for fi in detected:
            lo = max(0, fi - half_w)
            hi = min(total_frames, fi + half_w + 1)
            wx = x_s[lo:hi]
            wy = y_s[lo:hi]
            valid = np.isfinite(wx)
            if valid.sum() < 2:
                c[fi] = 1.0
                continue
            med_x = float(np.median(wx[valid]))
            med_y = float(np.median(wy[valid]))
            disp = np.sqrt((x_s[fi] - med_x) ** 2 + (y_s[fi] - med_y) ** 2)
            c[fi] = 1.0 / (1.0 + disp / scale)
        conf_arrays[li] = c

    # --- Min across keypoints ---
    stacked = np.stack(list(conf_arrays.values()), axis=0)  # (n_kp, T)
    conf_min = np.full(total_frames, np.nan, dtype=np.float64)
    any_det = np.any(np.isfinite(stacked), axis=0)
    conf_min[any_det] = np.nanmin(stacked[:, any_det], axis=0)

    conf_thresh = 0.5
    return conf_arrays, conf_min, conf_thresh


def _format_clinical_score_overlay_line(score_column, clinical_score):
    """Return a compact overlay line for the configured MDS-UPDRS score."""
    if not score_column:
        return None
    if clinical_score is None or pd.isna(clinical_score):
        return f"MDS-UPDRS {score_column}: N/A"
    return f"MDS-UPDRS {score_column}: {int(clinical_score)}"


def _runtime_gpu_config():
    """Return GPU refinement config from utils at runtime.

    This avoids stale values when apply_tuning_overrides() updates module-level
    constants in ps_kinematics.utils after core.py has already imported them.
    """
    use_openpose = bool(_u.USE_OPENPOSE)
    use_rtmpose = bool(_u.USE_RTMPOSE) and not use_openpose
    use_yolo_hand = bool(_u.USE_YOLO_HAND)
    use_yolo_only = bool(_u.USE_YOLO_ONLY)
    if use_yolo_only:
        keypoint_backend = "yolo_only"
    elif bool(_u.USE_RTMPOSE) and use_openpose:
        keypoint_backend = "openpose"
    elif use_yolo_hand:
        keypoint_backend = "yolo"
    elif use_rtmpose:
        keypoint_backend = "rtmpose"
    else:
        keypoint_backend = "mediapipe"

    return {
        "use_superres": bool(_u.USE_SUPERRES),
        "use_rtmpose": use_rtmpose,
        "use_openpose": use_openpose,
        "use_yolo_hand": use_yolo_hand,
        "use_yolo_only": use_yolo_only,
        "keypoint_backend": keypoint_backend,
        "rtmpose_model_cfg": _u.RTMPOSE_MODEL_CFG,
        "rtmpose_checkpoint_path": _u.RTMPOSE_CHECKPOINT_PATH,
        "rtmpose_bbox_padding": float(_u.RTMPOSE_BBOX_PADDING),
        "openpose_proto_path": _u.OPENPOSE_PROTO_PATH,
        "openpose_weights_path": _u.OPENPOSE_WEIGHTS_PATH,
        "openpose_bbox_padding": float(_u.OPENPOSE_BBOX_PADDING),
        "openpose_conf_threshold": float(_u.OPENPOSE_CONF_THRESHOLD),
        "openpose_input_size": int(_u.OPENPOSE_INPUT_SIZE),
        "openpose_use_cuda": bool(_u.OPENPOSE_USE_CUDA),
        "yolo_hand_model_path": _u.YOLO_HAND_MODEL_PATH,
        "yolo_hand_bbox_padding": float(_u.YOLO_HAND_BBOX_PADDING),
        "yolo_hand_conf_threshold": float(_u.YOLO_HAND_CONF_THRESHOLD),
        "gpu_concurrency": max(1, int(_u.GPU_CONCURRENCY)),
        "enable_runtime_diagnostics": bool(_u.ENABLE_RUNTIME_DIAGNOSTICS),
    }


# ============================
# Per-landmark temporal outlier correction
# ============================


def reject_landmark_outliers(track, window=None, thresh=None, fps=None):
    """Replace per-landmark position outliers with rolling-median values."""
    from .utils import BASE_FPS, LM_OUTLIER_THRESH, LM_OUTLIER_WINDOW

    if track is None:
        return track
    if window is None:
        _scale = max(1.0, float(fps)) / BASE_FPS if fps is not None else 1.0
        window = max(3, round(LM_OUTLIER_WINDOW * _scale))
    if thresh is None:
        thresh = LM_OUTLIER_THRESH
    window = int(window)
    if window < 3:
        return track
    frames_dict = track.get("frames", {})
    if len(frames_dict) < window:
        return track

    sorted_frames = sorted(frames_dict.keys())
    n = len(sorted_frames)
    half = window // 2

    key_lms = [
        int(HandLandmark.WRIST),
        int(HandLandmark.INDEX_FINGER_MCP),
        int(HandLandmark.MIDDLE_FINGER_MCP),
        int(HandLandmark.RING_FINGER_MCP),
        int(HandLandmark.PINKY_MCP),
    ]

    pos_xy = np.zeros((n, len(key_lms), 2), dtype=np.float32)
    for fi, f in enumerate(sorted_frames):
        lm_arr = frames_dict[f]
        for li, lm_idx in enumerate(key_lms):
            pos_xy[fi, li, 0] = float(lm_arr[lm_idx, 0])
            pos_xy[fi, li, 1] = float(lm_arr[lm_idx, 1])

    new_frames = {}
    for fi, f in enumerate(sorted_frames):
        lo = max(0, fi - half)
        hi = min(n - 1, fi + half) + 1
        median_pos = np.median(pos_xy[lo:hi, :, :], axis=0)
        lm_arr = frames_dict[f].copy()
        changed = False
        for li, lm_idx in enumerate(key_lms):
            cur = pos_xy[fi, li, :]
            med = median_pos[li, :]
            if float(np.linalg.norm(cur - med)) > thresh:
                lm_arr[lm_idx, 0] = med[0]
                lm_arr[lm_idx, 1] = med[1]
                changed = True
        if changed:
            new_frames[f] = lm_arr

    frames_dict.update(new_frames)
    return track


# ============================
# Shared fill-pass utilities
# ============================


def _extract_conf(handedness, hand_to_track):
    """Return the confidence score most relevant to hand_to_track."""
    if not handedness:
        return 0.0
    if hand_to_track is not None:
        for c in handedness:
            if c.category_name == hand_to_track:
                return float(c.score)
    return float(handedness[0].score)


def _roi_zoom_detect(frame_bgr, frame_idx, main_track, landmarker_img, hand_to_track=None):
    """Re-detect a hand by zooming into the expected hand bounding-box ROI."""
    if not MEDIAPIPE_OK or not CV2_OK:
        return None
    frames = main_track.get("frames", {})
    if not frames:
        return None
    nearest_f = min(frames.keys(), key=lambda f: abs(f - frame_idx))
    nearest_lm = frames[nearest_f]

    h_img, w_img = frame_bgr.shape[:2]
    pad = ROI_REDETECT_PADDING
    lm_x = nearest_lm[:, 0]
    lm_y = nearest_lm[:, 1]
    x1 = max(0.0, float(np.min(lm_x)) - pad)
    y1 = max(0.0, float(np.min(lm_y)) - pad)
    x2 = min(1.0, float(np.max(lm_x)) + pad)
    y2 = min(1.0, float(np.max(lm_y)) + pad)

    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w < 0.05 or roi_h < 0.05:
        return None

    px1, py1 = int(x1 * w_img), int(y1 * h_img)
    px2, py2 = int(x2 * w_img), int(y2 * h_img)
    roi = frame_bgr[py1:py2, px1:px2]
    if roi.size == 0:
        return None

    roi_scaled = cv2.resize(roi, (w_img, h_img))
    roi_rgb = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
    result = landmarker_img.detect(mp_img)
    if not result.hand_landmarks:
        return None

    best_lm_list = None
    if hand_to_track is not None and result.handedness:
        for hi, hcat in enumerate(result.handedness):
            if hi < len(result.hand_landmarks) and hcat:
                if hcat[0].category_name == hand_to_track:
                    best_lm_list = result.hand_landmarks[hi]
                    break
    if best_lm_list is None:
        best_lm_list = result.hand_landmarks[0]

    lm_arr = np.array(
        [[x1 + lm.x * roi_w, y1 + lm.y * roi_h, float(lm.z), _lm_vis(lm)] for lm in best_lm_list],
        dtype=np.float32,
    )
    return lm_arr


# ============================
# Temporal landmark smoothing
# ============================


def smooth_track_landmarks(track, total_frames, window=None, fps=None):
    """Smooth landmark positions over time.

    When ``USE_ONE_EURO`` is True (default), applies a per-landmark,
    per-coordinate One-Euro adaptive low-pass filter.  This smooths
    heavily when the hand is stationary (killing jitter) and lightly
    when the hand is moving fast (preserving signal).

    When ``USE_ONE_EURO`` is False, falls back to a fixed-window
    centred moving-average.

    Parameters
    ----------
    track : dict or None
        Track dictionary with ``"frames"`` mapping frame_idx → (21,4) array.
    total_frames : int
        Total number of frames in the video.
    window : int or None
        Legacy moving-average window (only used when USE_ONE_EURO=False).
    fps : float or None
        Video frame rate.  Used by the One-Euro filter; if None the rate
        is estimated from the track's frame indices (falls back to 25).
    """
    if track is None:
        return track

    frames_dict = track.get("frames", {})
    if not frames_dict:
        return track

    if USE_ONE_EURO:
        return _smooth_track_one_euro(track, total_frames, fps)
    else:
        if window is None and fps is not None:
            from .utils import BASE_FPS, LANDMARK_SMOOTH_WINDOW

            _scale = max(1.0, float(fps)) / BASE_FPS
            window = max(1, round(LANDMARK_SMOOTH_WINDOW * _scale))
        return _smooth_track_moving_avg(track, total_frames, window)


def ensure_track_visibility_channel(track, default_visibility=1.0):
    """Ensure per-frame landmarks include a visibility channel.

    Some refinement backends return landmark arrays shaped (N,3) with
    only x/y/z. The kinematic pipeline can still process these frames,
    so we explicitly append a visibility column with ``default_visibility``
    to keep downstream exports and diagnostics consistent.
    """
    if track is None:
        return track

    frames_dict = track.get("frames", {})
    if not frames_dict:
        return track

    updated_frames = {}
    for frame_idx, lm_arr in frames_dict.items():
        if not isinstance(lm_arr, np.ndarray) or lm_arr.ndim != 2:
            updated_frames[frame_idx] = lm_arr
            continue
        if lm_arr.shape[1] >= 4:
            updated_frames[frame_idx] = lm_arr
            continue
        if lm_arr.shape[1] == 3:
            vis_col = np.full((lm_arr.shape[0], 1), float(default_visibility), dtype=lm_arr.dtype)
            updated_frames[frame_idx] = np.concatenate([lm_arr, vis_col], axis=1)
            continue
        updated_frames[frame_idx] = lm_arr

    track["frames"] = updated_frames
    return track


def _smooth_track_one_euro(track, total_frames, fps=None):
    """Apply per-landmark per-coordinate One-Euro adaptive filtering."""
    frames_dict = track.get("frames", {})
    sorted_indices = sorted(frames_dict.keys())
    n = len(sorted_indices)
    if n < 2:
        return track

    # Estimate fps from frame indices if not provided
    if fps is None or fps <= 0:
        fps = 25.0  # safe default for this dataset

    n_lm = frames_dict[sorted_indices[0]].shape[0]  # typically 21
    n_coords = min(frames_dict[sorted_indices[0]].shape[1], 3)  # x, y, z only

    # Build filters: one per landmark per coordinate
    filters = [
        [
            OneEuroFilter(
                fps=fps,
                min_cutoff=ONEEURO_MIN_CUTOFF,
                beta=ONEEURO_BETA,
                d_cutoff=ONEEURO_D_CUTOFF,
            )
            for _ in range(n_coords)
        ]
        for _ in range(n_lm)
    ]

    # Process frames in temporal order
    prev_frame_idx = None
    for f in sorted_indices:
        lm_arr = frames_dict[f]
        # If there's a gap > 1 frame, reset all filters (continuity broken)
        if prev_frame_idx is not None and (f - prev_frame_idx) > 1:
            for lm_filters in filters:
                for filt in lm_filters:
                    filt.reset()
        prev_frame_idx = f

        smoothed = lm_arr.copy()
        for lm_idx in range(n_lm):
            for c in range(n_coords):
                smoothed[lm_idx, c] = filters[lm_idx][c](float(lm_arr[lm_idx, c]))
        frames_dict[f] = smoothed

    return track


def _smooth_track_moving_avg(track, total_frames, window=None):
    """Legacy centred moving-average smoothing (original implementation)."""
    if window is None:
        window = LANDMARK_SMOOTH_WINDOW
    window = int(window)
    if window <= 1:
        return track
    if window % 2 == 0:
        window += 1

    half = window // 2
    frames_dict = track.get("frames", {})
    if not frames_dict:
        return track

    sorted_indices = sorted(frames_dict.keys())
    frame_set = set(sorted_indices)

    smoothed = {}
    for f in sorted_indices:
        accum = np.zeros_like(frames_dict[f], dtype=np.float64)
        count = 0
        for offset in range(-half, half + 1):
            neighbour = f + offset
            if neighbour in frame_set:
                accum += frames_dict[neighbour].astype(np.float64)
                count += 1
        smoothed[f] = (accum / count).astype(np.float32)

    track["frames"] = smoothed
    return track


# ============================
# PS-activity segmentation
# ============================


def trim_track_to_ps_segment(
    track,
    total_frames,
    fps,
    max_duration_s=None,
    min_duration_s=None,
    window_s=None,
    percentile=None,
    threshold_ratio=None,
    merge_gap_s=None,
):
    """Trim a track to the first ``max_duration_s`` seconds of active PS.

    Activity detection:
      1. Compute per-frame angular velocity and smooth it with a rolling window.
      2. Mark frames above ``threshold_ratio * peak_velocity`` as active.
      3. Merge active segments separated by less than ``merge_gap_s`` seconds.
      4. Discard segments shorter than ``min_duration_s`` seconds.
      5. Select the **first** qualifying segment (earliest PS onset).
      6. Retain at most the first ``max_duration_s`` seconds from that onset.
    """
    if max_duration_s is None:
        max_duration_s = MAX_PS_DURATION_S
    if min_duration_s is None:
        min_duration_s = MIN_PS_DURATION_S
    if window_s is None:
        window_s = PS_ACTIVITY_WINDOW_S
    if percentile is None:
        percentile = PS_ACTIVITY_PERCENTILE
    if threshold_ratio is None:
        threshold_ratio = PS_ACTIVITY_THRESHOLD_RATIO
    if merge_gap_s is None:
        merge_gap_s = PS_MERGE_GAP_S

    if track is None or not track.get("frames"):
        return track, None, None

    frames_dict = track["frames"]
    sorted_frames = sorted(frames_dict.keys())

    if len(sorted_frames) < 10:
        return track, None, None

    angles = np.full(total_frames, np.nan)
    for f in sorted_frames:
        angles[f] = _proxy_angle_deg(frames_dict[f])

    ang_vel = np.zeros(total_frames)
    for i in range(1, len(sorted_frames)):
        f_prev = sorted_frames[i - 1]
        f_curr = sorted_frames[i]
        dt = f_curr - f_prev
        if dt > 0 and not np.isnan(angles[f_prev]) and not np.isnan(angles[f_curr]):
            ang_vel[f_curr] = abs(angles[f_curr] - angles[f_prev]) / dt

    win_frames = max(1, int(window_s * fps))
    if win_frames % 2 == 0:
        win_frames += 1
    kernel = np.ones(win_frames) / win_frames
    smoothed_vel = np.convolve(ang_vel, kernel, mode="same")

    track_mask = np.zeros(total_frames, dtype=bool)
    for f in sorted_frames:
        track_mask[f] = True
    tracked_vel = smoothed_vel[track_mask]
    peak_vel = float(np.percentile(tracked_vel, percentile)) if tracked_vel.size > 0 else 0.0
    if peak_vel < 1e-6:
        return track, None, None
    threshold = threshold_ratio * peak_vel

    active = np.zeros(total_frames, dtype=bool)
    for f in sorted_frames:
        if smoothed_vel[f] >= threshold:
            active[f] = True

    segments = []
    seg_start = None
    for f in range(total_frames):
        if active[f]:
            if seg_start is None:
                seg_start = f
        else:
            if seg_start is not None:
                segments.append((seg_start, f - 1))
                seg_start = None
    if seg_start is not None:
        segments.append((seg_start, total_frames - 1))

    if not segments:
        return track, None, None

    merge_gap_frames = max(1, int(merge_gap_s * fps))
    merged = [segments[0]]
    for seg in segments[1:]:
        prev_end = merged[-1][1]
        curr_start = seg[0]
        if curr_start - prev_end <= merge_gap_frames:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    # Filter out segments that are shorter than the minimum required PS duration.
    min_frames = int(min_duration_s * fps)
    qualifying = [(s, e) for (s, e) in merged if (e - s + 1) >= min_frames]

    if not qualifying:
        # Fall back: if no segment meets the minimum, take the longest one anyway.
        qualifying = [max(merged, key=lambda se: se[1] - se[0])]

    # Take the FIRST qualifying segment (earliest PS onset).
    first_seg = min(qualifying, key=lambda se: se[0])
    seg_start, seg_end = first_seg

    # Retain at most the first max_duration_s seconds from onset.
    max_frames = int(max_duration_s * fps)
    seg_end = min(seg_end, seg_start + max_frames - 1)

    keep = set(f for f in sorted_frames if seg_start <= f <= seg_end)
    track["frames"] = {f: frames_dict[f] for f in keep}
    if "conf" in track:
        track["conf"] = {f: v for f, v in track["conf"].items() if f in keep}

    return track, int(seg_start), int(seg_end)


# ============================================================
# GPU-visible device parser (shared by processor and workers)
# ============================================================


def _parse_visible_gpu_ids():
    """Return the CUDA-visible GPU identifiers in process order.

    Examples
    --------
    CUDA_VISIBLE_DEVICES="0,1" -> ["0", "1"]
    unset/empty                -> ["0"]
    """
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw:
        return ["0"]
    ids = [tok.strip() for tok in raw.split(",") if tok.strip()]
    return ids if ids else ["0"]


# ============================================================
# Backward-compatible re-exports
# ============================================================
# Classes and worker functions have been split into dedicated modules.
# Re-export them here so existing ``from ps_kinematics.core import ...``
# statements continue to work without modification.

from .processor import HandLandmarkProcessor  # noqa: E402, F401
from .tracker import MultiHandOfflineTracker  # noqa: E402, F401
from .workers import (  # noqa: E402, F401
    _avg_confidence_from_track_standalone,
    _choose_detection_for_track_spatial_standalone,
    _collect_runtime_cuda_info,
    _compute_appearance_metrics_standalone,
    _dist_standalone,
    _draw_hand_from_array_standalone,
    _draw_top_right_text_standalone,
    _frame_timestamp_ms,
    _infer_tracks_offline_standalone,
    _init_worker_gpu,
    _log_runtime_diag,
    _nearest_reference_wrist_standalone,
    _process_single_video,
    _process_video_worker,
    _process_video_worker_inner,
    _render_video_with_plots_standalone,
    _robust_worker_entry,
    _wrist_xy_standalone,
)
