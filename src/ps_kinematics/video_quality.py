"""
ps_kinematics.video_quality — Per-video quality factor computation.

Computes observable video-level metrics that explain why some recordings
yield clean kinematic signals while others produce noise.  These factors
are logged alongside kinematic features in the tracking CSV.

Factors implemented
-------------------
1. Hand pixel size (bounding-box area from landmark extent)
2. Motion blur (Laplacian variance sharpness in the hand ROI)
3. Lighting (luminance mean, CV, and saturation fraction in hand ROI)
4. Occlusion (gap statistics derived from the detection mask)
5. Global (full-frame) sharpness, contrast, luminance, saturation
6. Temporal stability (frame-to-frame pixel differences)
7. Video bitrate (compression quality)
"""

import os

import numpy as np

try:
    import cv2

    _CV2_OK = True
except ImportError:
    cv2 = None
    _CV2_OK = False


# ============================================================
# Hand bounding-box helpers
# ============================================================


def _hand_bbox_from_landmarks(
    lm_arr: np.ndarray, frame_h: int, frame_w: int, padding: float = 0.15
) -> tuple[int, int, int, int]:
    """Compute a padded pixel bounding box from a (21, 3) landmark array.

    Parameters
    ----------
    lm_arr : np.ndarray, shape (21, 3)
        Normalised [0,1] landmark coordinates (x, y, z).
    frame_h, frame_w : int
        Frame dimensions in pixels.
    padding : float
        Fractional padding around the tight bbox.

    Returns
    -------
    (x1, y1, x2, y2) : tuple[int, int, int, int]
        Pixel coordinates (clipped to frame bounds).
    """
    xs = lm_arr[:, 0] * frame_w
    ys = lm_arr[:, 1] * frame_h
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    w = x_max - x_min
    h = y_max - y_min
    pad_x = w * padding
    pad_y = h * padding
    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(frame_w, int(x_max + pad_x))
    y2 = min(frame_h, int(y_max + pad_y))
    return x1, y1, x2, y2


# ============================================================
# Per-frame metric computation
# ============================================================


def compute_frame_sharpness(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Laplacian variance of the hand ROI — higher = sharper.

    Returns
    -------
    float or nan  — Laplacian variance (sharpness score).
    """
    if not _CV2_OK or frame_bgr is None:
        return float("nan")
    x1, y1, x2, y2 = bbox
    if x2 - x1 < 8 or y2 - y1 < 8:
        return float("nan")
    roi = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def compute_frame_luminance(
    frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]
) -> dict[str, float]:
    """Luminance statistics of the hand ROI.

    Returns
    -------
    dict with keys:
        lum_mean  — mean luminance (0-255)
        lum_std   — luminance std within the ROI
        sat_frac  — fraction of pixels near saturation (>245 or <10)
    """
    if not _CV2_OK or frame_bgr is None:
        return {"lum_mean": float("nan"), "lum_std": float("nan"), "sat_frac": float("nan")}
    x1, y1, x2, y2 = bbox
    if x2 - x1 < 4 or y2 - y1 < 4:
        return {"lum_mean": float("nan"), "lum_std": float("nan"), "sat_frac": float("nan")}
    roi = frame_bgr[y1:y2, x1:x2]
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    y_ch = ycrcb[:, :, 0].astype(float)
    n_px = y_ch.size
    sat_count = int(np.sum((y_ch > 245) | (y_ch < 10)))
    return {
        "lum_mean": float(np.mean(y_ch)),
        "lum_std": float(np.std(y_ch)),
        "sat_frac": float(sat_count / max(n_px, 1)),
    }


# ============================================================
# Per-video aggregation of frame-level metrics
# ============================================================


def compute_video_quality_metrics(
    video_path: str,
    frames_dict: dict[int, np.ndarray],
    total_frames: int,
    fps: float,
    ps_start_frame: int | None = None,
    ps_end_frame: int | None = None,
    frame_hw: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Compute per-video quality factor metrics.

    When *frame_hw* ``(height, width)`` is supplied by the caller (from the
    main processing pass), bbox-area and occlusion metrics are computed
    directly from landmarks — the video file is only re-opened for
    BGR-dependent metrics (sharpness, luminance).  This avoids a redundant
    full video read when the caller already knows the frame dimensions.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frames_dict : dict
        ``{frame_index: np.ndarray}`` of (21,3) landmark arrays.
    total_frames : int
        Total frame count.
    fps : float
        Frames per second.
    ps_start_frame, ps_end_frame : int or None
        PS-activity window bounds (inclusive).
    frame_hw : tuple[int, int] or None
        ``(frame_height, frame_width)`` in pixels.  When provided, bbox-area
        and occlusion stats are computed without opening the video.

    Returns
    -------
    dict with video quality metrics.
    """
    result = _empty_quality_metrics()

    if not frames_dict:
        return result

    # Determine analysis window
    if ps_start_frame is not None and ps_end_frame is not None:
        win_start = int(ps_start_frame)
        win_end = int(ps_end_frame)
    else:
        sorted_frames = sorted(frames_dict.keys())
        win_start = sorted_frames[0]
        win_end = sorted_frames[-1]

    # Collect frame indices to analyse (detected frames within the window)
    analyse_frames = set()
    for fi in frames_dict:
        if win_start <= fi <= win_end:
            analyse_frames.add(int(fi))

    if not analyse_frames:
        return result

    # --- Resolve frame dimensions ---
    # Prefer caller-supplied dimensions to avoid opening the video just for
    # CAP_PROP_FRAME_HEIGHT / CAP_PROP_FRAME_WIDTH.
    frame_h, frame_w = 0, 0
    if frame_hw is not None and frame_hw[0] > 0 and frame_hw[1] > 0:
        frame_h, frame_w = frame_hw
    elif _CV2_OK:
        cap_probe = cv2.VideoCapture(video_path)
        if cap_probe.isOpened():
            frame_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_probe.release()

    if frame_h <= 0 or frame_w <= 0:
        return result

    result["video_width_px"] = float(frame_w)
    result["video_height_px"] = float(frame_h)

    # --- Bbox area from landmarks (no video read needed) ---
    bbox_area_vals = []
    sorted_analyse = sorted(analyse_frames)
    for fidx in sorted_analyse:
        lm_arr = frames_dict[fidx]
        bbox = _hand_bbox_from_landmarks(lm_arr, frame_h, frame_w)
        x1, y1, x2, y2 = bbox
        bbox_area_vals.append(float((x2 - x1) * (y2 - y1)))

    if bbox_area_vals:
        arr = np.array(bbox_area_vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            result["hand_bbox_area_median_px"] = float(np.median(arr))
            result["hand_bbox_area_q25_px"] = float(np.percentile(arr, 25))

    # --- BGR-dependent metrics (sharpness, luminance) — requires video read ---
    if _CV2_OK:
        sharpness_vals = []
        lum_mean_vals = []
        lum_std_vals = []
        sat_frac_vals = []

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            max_frame_needed = sorted_analyse[-1]
            window_len = max_frame_needed - sorted_analyse[0] + 1
            use_seek = len(sorted_analyse) < 0.5 * window_len and window_len > 20

            if use_seek:
                for fidx in sorted_analyse:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ok, frame_bgr = cap.read()
                    if not ok:
                        continue
                    lm_arr = frames_dict[fidx]
                    bbox = _hand_bbox_from_landmarks(lm_arr, frame_h, frame_w)
                    sharpness_vals.append(compute_frame_sharpness(frame_bgr, bbox))
                    lum = compute_frame_luminance(frame_bgr, bbox)
                    lum_mean_vals.append(lum["lum_mean"])
                    lum_std_vals.append(lum["lum_std"])
                    sat_frac_vals.append(lum["sat_frac"])
            else:
                fidx = 0
                while fidx <= max_frame_needed:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    if fidx in analyse_frames:
                        lm_arr = frames_dict[fidx]
                        bbox = _hand_bbox_from_landmarks(lm_arr, frame_h, frame_w)
                        sharpness_vals.append(compute_frame_sharpness(frame_bgr, bbox))
                        lum = compute_frame_luminance(frame_bgr, bbox)
                        lum_mean_vals.append(lum["lum_mean"])
                        lum_std_vals.append(lum["lum_std"])
                        sat_frac_vals.append(lum["sat_frac"])
                    fidx += 1

            cap.release()

            _aggregate_sharpness(result, sharpness_vals)
            _aggregate_luminance(result, lum_mean_vals, lum_std_vals, sat_frac_vals)

    # --- Occlusion / gap statistics (no video read needed) ---
    result.update(
        _compute_occlusion_stats(
            frames_dict,
            total_frames,
            win_start,
            win_end,
            fps,
        )
    )

    # --- Global (full-frame) metrics — no landmark dependency ---
    global_metrics = compute_global_frame_metrics(
        video_path,
        total_frames,
        fps,
        ps_start_frame=ps_start_frame,
        ps_end_frame=ps_end_frame,
        frame_hw=(frame_h, frame_w) if frame_h > 0 and frame_w > 0 else None,
    )
    result.update(global_metrics)

    return result


def _aggregate_sharpness(result: dict, sharpness_vals: list[float]) -> None:
    """Aggregate per-frame sharpness values into result dict."""
    if not sharpness_vals:
        return
    arr = np.array(sharpness_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) > 0:
        result["sharpness_median"] = float(np.median(arr))
        result["sharpness_q10"] = float(np.percentile(arr, 10))
        result["sharpness_q25"] = float(np.percentile(arr, 25))


def _aggregate_luminance(
    result: dict, lum_mean_vals: list[float], lum_std_vals: list[float], sat_frac_vals: list[float]
) -> None:
    """Aggregate per-frame luminance values into result dict."""
    if lum_mean_vals:
        arr = np.array(lum_mean_vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            result["luminance_median"] = float(np.median(arr))
            mean_lum = float(np.mean(arr))
            std_lum = float(np.std(arr))
            result["luminance_mean"] = mean_lum
            result["luminance_cv"] = float(std_lum / max(mean_lum, 1e-6))

    if lum_std_vals:
        arr = np.array(lum_std_vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            result["luminance_uniformity_median"] = float(np.median(arr))

    if sat_frac_vals:
        arr = np.array(sat_frac_vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            result["saturation_frac_median"] = float(np.median(arr))


def _compute_occlusion_stats(frames_dict, total_frames, win_start, win_end, fps):
    """Compute detection gap statistics within the analysis window.

    Returns
    -------
    dict with occlusion metrics.
    """
    result = {}
    win_len = win_end - win_start + 1
    if win_len <= 0:
        return {
            "detection_rate": float("nan"),
            "n_gaps": 0,
            "longest_gap_frames": 0,
            "longest_gap_s": float("nan"),
            "mean_gap_frames": float("nan"),
            "total_gap_frames": 0,
            "gap_fraction": float("nan"),
        }

    detected = sorted(fi for fi in frames_dict if win_start <= fi <= win_end)
    detection_rate = float(len(detected) / win_len) if win_len > 0 else 0.0
    result["detection_rate"] = detection_rate

    # Build a boolean mask for the window
    mask = np.zeros(win_len, dtype=bool)
    for fi in detected:
        mask[fi - win_start] = True

    # Find gaps (runs of False)
    gaps = []
    in_gap = False
    gap_start = 0
    for i in range(win_len):
        if not mask[i]:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gaps.append(i - gap_start)
                in_gap = False
    if in_gap:
        gaps.append(win_len - gap_start)

    result["n_gaps"] = len(gaps)
    if gaps:
        result["longest_gap_frames"] = int(max(gaps))
        result["longest_gap_s"] = float(max(gaps) / max(fps, 1.0))
        result["mean_gap_frames"] = float(np.mean(gaps))
        result["total_gap_frames"] = int(sum(gaps))
        result["gap_fraction"] = float(sum(gaps) / win_len)
    else:
        result["longest_gap_frames"] = 0
        result["longest_gap_s"] = 0.0
        result["mean_gap_frames"] = 0.0
        result["total_gap_frames"] = 0
        result["gap_fraction"] = 0.0

    return result


def _empty_quality_metrics():
    """Return a dict with all quality metric keys set to NaN / defaults."""
    return {
        "video_width_px": float("nan"),
        "video_height_px": float("nan"),
        "hand_bbox_area_median_px": float("nan"),
        "hand_bbox_area_q25_px": float("nan"),
        "sharpness_median": float("nan"),
        "sharpness_q10": float("nan"),
        "sharpness_q25": float("nan"),
        "luminance_median": float("nan"),
        "luminance_mean": float("nan"),
        "luminance_cv": float("nan"),
        "luminance_uniformity_median": float("nan"),
        "saturation_frac_median": float("nan"),
        "detection_rate": float("nan"),
        "n_gaps": 0,
        "longest_gap_frames": 0,
        "longest_gap_s": float("nan"),
        "mean_gap_frames": float("nan"),
        "total_gap_frames": 0,
        "gap_fraction": float("nan"),
        # Global (full-frame) metrics — no landmark dependency
        "global_sharpness_median": float("nan"),
        "global_sharpness_q10": float("nan"),
        "global_contrast_median": float("nan"),
        "global_luminance_median": float("nan"),
        "global_luminance_cv": float("nan"),
        "global_saturation_frac_median": float("nan"),
        "temporal_diff_median": float("nan"),
        "temporal_diff_cv": float("nan"),
        "bitrate_kbps": float("nan"),
        "resolution_area_mpx": float("nan"),
    }


# ============================================================
# Global (full-frame) metrics — no landmark dependency
# ============================================================


def compute_global_frame_metrics(
    video_path: str,
    total_frames: int,
    fps: float,
    ps_start_frame: int | None = None,
    ps_end_frame: int | None = None,
    frame_hw: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Compute full-frame quality metrics without any landmark dependency.

    Processes every frame in the PS window (win_start..win_end inclusive),
    computing sharpness, contrast, luminance, saturation, and temporal
    stability on the full frame — no landmark positions required.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    total_frames : int
        Total frame count.
    fps : float
        Frames per second.
    ps_start_frame, ps_end_frame : int or None
        PS-activity window bounds (inclusive).
    frame_hw : tuple[int, int] or None
        ``(frame_height, frame_width)`` — used only for resolution_area_mpx
        when the video cannot be opened.

    Returns
    -------
    dict with global quality metrics.
    """
    result = {
        "global_sharpness_median": float("nan"),
        "global_sharpness_q10": float("nan"),
        "global_contrast_median": float("nan"),
        "global_luminance_median": float("nan"),
        "global_luminance_cv": float("nan"),
        "global_saturation_frac_median": float("nan"),
        "temporal_diff_median": float("nan"),
        "temporal_diff_cv": float("nan"),
        "bitrate_kbps": float("nan"),
        "resolution_area_mpx": float("nan"),
    }

    # Bitrate from file size
    duration_s = total_frames / max(fps, 1.0)
    if duration_s > 0:
        try:
            file_size_bytes = os.path.getsize(video_path)
            result["bitrate_kbps"] = float(file_size_bytes * 8 / (duration_s * 1000))
        except OSError:
            pass

    # Resolution area from caller-supplied dimensions or video probe
    fh, fw = 0, 0
    if frame_hw is not None and frame_hw[0] > 0 and frame_hw[1] > 0:
        fh, fw = frame_hw
    if fh > 0 and fw > 0:
        result["resolution_area_mpx"] = float(fh * fw / 1e6)

    if not _CV2_OK:
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return result

    if fh <= 0 or fw <= 0:
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if fh > 0 and fw > 0:
            result["resolution_area_mpx"] = float(fh * fw / 1e6)

    # Determine PS window
    win_start = int(ps_start_frame) if ps_start_frame is not None else 0
    win_end = int(ps_end_frame) if ps_end_frame is not None else max(total_frames - 1, 0)
    win_len = win_end - win_start + 1
    if win_len <= 0:
        cap.release()
        return result

    # Sequential read of frames in the PS window.
    # When TARGET_PROCESSING_FPS is set and the video fps exceeds it,
    # subsample to keep quality computation time proportional.
    from .utils import TARGET_PROCESSING_FPS as _tpfps

    _qm_stride = 1
    if _tpfps and _tpfps > 0 and fps > _tpfps:
        _qm_stride = max(1, int(fps / _tpfps))

    sharpness_vals = []
    contrast_vals = []
    lum_mean_vals = []
    sat_frac_vals = []
    prev_gray = None
    temporal_diffs = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, win_start)
    fidx = win_start
    while fidx <= win_end:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if _qm_stride > 1 and (fidx - win_start) % _qm_stride != 0:
            fidx += 1
            continue
        fidx += 1

        # Convert to grayscale / YCrCb
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y_ch = ycrcb[:, :, 0].astype(float)

        # Sharpness: Laplacian variance of full frame
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_vals.append(float(np.var(lap)))

        # Contrast: RMS contrast of Y channel
        y_mean = float(np.mean(y_ch))
        y_std = float(np.std(y_ch))
        contrast_vals.append(y_std)

        # Luminance
        lum_mean_vals.append(y_mean)

        # Saturation fraction (over/under-exposed)
        n_px = y_ch.size
        sat_count = int(np.sum((y_ch > 245) | (y_ch < 10)))
        sat_frac_vals.append(float(sat_count / max(n_px, 1)))

        # Temporal difference with previous sampled frame
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, prev_gray)
            temporal_diffs.append(float(np.mean(diff)))
        prev_gray = gray

    cap.release()

    # Aggregate sharpness
    if sharpness_vals:
        arr = np.array(sharpness_vals, dtype=float)
        result["global_sharpness_median"] = float(np.median(arr))
        result["global_sharpness_q10"] = float(np.percentile(arr, 10))

    # Aggregate contrast
    if contrast_vals:
        arr = np.array(contrast_vals, dtype=float)
        result["global_contrast_median"] = float(np.median(arr))

    # Aggregate luminance
    if lum_mean_vals:
        arr = np.array(lum_mean_vals, dtype=float)
        median_lum = float(np.median(arr))
        mean_lum = float(np.mean(arr))
        std_lum = float(np.std(arr))
        result["global_luminance_median"] = median_lum
        result["global_luminance_cv"] = float(std_lum / max(mean_lum, 1e-6))

    # Aggregate saturation fraction
    if sat_frac_vals:
        arr = np.array(sat_frac_vals, dtype=float)
        result["global_saturation_frac_median"] = float(np.median(arr))

    # Aggregate temporal differences
    if temporal_diffs:
        arr = np.array(temporal_diffs, dtype=float)
        median_td = float(np.median(arr))
        mean_td = float(np.mean(arr))
        std_td = float(np.std(arr))
        result["temporal_diff_median"] = median_td
        result["temporal_diff_cv"] = float(std_td / max(mean_td, 1e-6))

    return result
