"""
ps_kinematics.yolo_tracker — Full YOLO-only hand tracking pipeline.

Replaces MediaPipe entirely: YOLO-Pose provides primary detection on every
frame; frame-to-frame tracking reuses MultiHandOfflineTracker unchanged;
gap-fill picks the YOLO detection whose wrist is closest to the last known
track position.

Returns the identical 6-tuple as _infer_tracks_offline_standalone():
    (main_track, total_frames, fps, fill_added, detected_before_fill, frame_hw)

Handedness (Left / Right) is inferred from the normalised wrist x-coordinate:
wrist_x > 0.5 → subject's left hand (mirrors camera convention in which the
subject's left hand appears on the right of the frame), wrist_x < 0.5 →
subject's right hand.  This matches MediaPipe's handedness convention.

All main-track selection logic (rotation scoring, choose_main_track) is
inherited unchanged from MultiHandOfflineTracker in tracker.py.
"""

import logging
import queue
import threading

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2

    CV2_OK = True
except ImportError:
    cv2 = None
    CV2_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe Category shim ─ lets MultiHandOfflineTracker run unchanged
# ──────────────────────────────────────────────────────────────────────────────


class _FakeCategory:
    """Minimal duck-type shim for a MediaPipe Classification category.

    MultiHandOfflineTracker accesses only ``.category_name`` and ``.score``;
    this shim satisfies both without importing mediapipe.
    """

    __slots__ = ("category_name", "score")

    def __init__(self, category_name: str, score: float) -> None:
        self.category_name = category_name
        self.score = float(score)


# ──────────────────────────────────────────────────────────────────────────────
# Handedness inference from wrist x-position
# ──────────────────────────────────────────────────────────────────────────────


def _infer_handedness_from_wrist(wrist_x_norm: float) -> str:
    """Infer hand label from normalised wrist x-coordinate.

    In a typical frontal recording the subject's *left* hand appears on the
    *right* side of the frame (x_norm > 0.5), matching MediaPipe's mirrored
    handedness convention.

    Parameters
    ----------
    wrist_x_norm : float   normalised [0, 1] wrist x-coordinate.

    Returns
    -------
    str   "Left" or "Right" (subject perspective, same as MediaPipe).
    """
    return "Left" if wrist_x_norm > 0.5 else "Right"


# ──────────────────────────────────────────────────────────────────────────────
# Per-result parsing (shared by single-frame and batched paths)
# ──────────────────────────────────────────────────────────────────────────────


def _parse_yolo_result(result):
    """Convert one ultralytics Result object to a MediaPipe-compatible detection list.

    Returns
    -------
    list of (lm_arr, handedness_list)
        lm_arr         – np.ndarray shape (21, 4): [x_norm, y_norm, z=0, kpt_conf]
        handedness_list – [_FakeCategory(label, box_conf)]
    """
    if result.keypoints is None:
        return []
    kpts = result.keypoints
    if kpts.xyn is None or kpts.xyn.shape[0] == 0:
        return []

    detections = []
    n_dets = int(kpts.xyn.shape[0])

    for i in range(n_dets):
        box_conf = 1.0
        if hasattr(result, "boxes") and result.boxes is not None and i < len(result.boxes):
            box_conf = float(result.boxes.conf[i])

        xyn_i = kpts.xyn[i].cpu().numpy()  # (K, 2) normalised [0,1]
        if kpts.conf is not None:
            kpt_conf_i = kpts.conf[i].cpu().numpy()  # (K,)
        else:
            kpt_conf_i = np.full(xyn_i.shape[0], box_conf, dtype=np.float32)

        if xyn_i.shape[0] < 21:
            continue

        lm_arr = np.zeros((21, 4), dtype=np.float32)
        lm_arr[:, 0] = xyn_i[:, 0]
        lm_arr[:, 1] = xyn_i[:, 1]
        lm_arr[:, 2] = 0.0
        lm_arr[:, 3] = kpt_conf_i

        wrist_x = float(lm_arr[0, 0])
        label = _infer_handedness_from_wrist(wrist_x)
        handedness = [_FakeCategory(label, box_conf)]

        detections.append((lm_arr, handedness))

    return detections


# ──────────────────────────────────────────────────────────────────────────────
# Full-frame YOLO detection — single frame (retained for gap-fill compat)
# ──────────────────────────────────────────────────────────────────────────────


def _yolo_full_frame_detections(model, frame_bgr, min_det_conf: float = 0.15):
    """Run YOLO-Pose on the full frame; return MediaPipe-compatible detection list.

    Parameters
    ----------
    model : ultralytics.YOLO  (already loaded and warmed up)
    frame_bgr : np.ndarray   shape (H, W, 3), BGR colour order
    min_det_conf : float   minimum box confidence for a detection to be kept

    Returns
    -------
    list of (lm_arr, handedness_list)
        lm_arr         – np.ndarray shape (21, 4): [x_norm, y_norm, z=0, kpt_conf]
        handedness_list – [_FakeCategory(label, box_conf)]

        Compatible with MultiHandOfflineTracker.associate_frame().
    """
    try:
        results = model.predict(
            frame_bgr,
            device="cuda:0",
            verbose=False,
            conf=min_det_conf,
        )
    except Exception:
        return []

    if not results:
        return []

    return _parse_yolo_result(results[0])


# ──────────────────────────────────────────────────────────────────────────────
# Full-frame YOLO detection — batched (multiple frames in one predict call)
# ──────────────────────────────────────────────────────────────────────────────


def _yolo_full_frame_detections_batch(model, frames_bgr, min_det_conf: float = 0.15):
    """Run YOLO-Pose on a list of frames in a single predict() call.

    Ultralytics natively accepts a list of np.ndarray as input and returns one
    Result per frame.  Batching amortises kernel-launch overhead and fills more
    CUDA SMs per call, which raises GPU utilisation and power draw closer to
    the thermal limit.

    Parameters
    ----------
    model : ultralytics.YOLO
    frames_bgr : list of np.ndarray   each shape (H, W, 3), BGR
    min_det_conf : float

    Returns
    -------
    list of detection_lists — one element per input frame, same format as
    _yolo_full_frame_detections().
    """
    if not frames_bgr:
        return []
    try:
        results = model.predict(
            frames_bgr,
            device="cuda:0",
            verbose=False,
            conf=min_det_conf,
        )
    except Exception:
        return [[] for _ in frames_bgr]

    return [_parse_yolo_result(r) for r in results]


# ──────────────────────────────────────────────────────────────────────────────
# Gap-fill helper
# ──────────────────────────────────────────────────────────────────────────────


def _nearest_reference_wrist_yolo(main_track, frame_idx):
    """Return the wrist (x, y) from the closest detected frame to *frame_idx*.

    Used during the gap-fill pass to anchor the spatial search.
    Returns None if the track has no detected frames.
    """
    frames = main_track.get("frames", {})
    if not frames:
        return None
    best_f = min(frames.keys(), key=lambda f: abs(f - frame_idx))
    lm = frames[best_f]
    return float(lm[0, 0]), float(lm[0, 1])


# ──────────────────────────────────────────────────────────────────────────────
# Background frame reader (producer thread)
# ──────────────────────────────────────────────────────────────────────────────

_SENTINEL = object()  # signals end-of-video to the consumer


def _frame_reader_thread(cap, frame_queue):
    """Read frames from *cap* and push them onto *frame_queue*.

    Runs in a daemon thread so the GPU-inference loop never stalls waiting for
    OpenCV to decode the next frame.  Pushes ``_SENTINEL`` when the video ends.
    """
    while True:
        ok, frame = cap.read()
        frame_queue.put(frame if ok else _SENTINEL)
        if not ok:
            break


# ──────────────────────────────────────────────────────────────────────────────
# Primary standalone tracker
# ──────────────────────────────────────────────────────────────────────────────


def _infer_tracks_yolo_standalone(
    video_path,
    hand_to_track,
    model_path=None,
    conf_thresh: float = 0.25,
    fill_max_dist: float = None,
    fill_iterations: int = None,
):
    """YOLO-only hand tracking; returns the same 6-tuple as the MediaPipe version.

    Parameters
    ----------
    video_path : str
    hand_to_track : str or None   e.g. ``"Left"`` or ``"Right"``
    model_path : str or None
        Path to the YOLO-Pose .pt checkpoint.  Defaults to
        ``utils.YOLO_HAND_MODEL_PATH`` if None.
    conf_thresh : float
        Minimum per-keypoint confidence during the gap-fill pass.
    fill_max_dist : float or None
        Maximum normalised wrist distance for gap-fill acceptance.
        Defaults to ``utils.FILL_MAX_DIST`` (fps-scaled at runtime).
    fill_iterations : int or None
        Number of gap-fill re-scan passes.
        Defaults to ``utils.FILL_ITERATIONS``.

    Returns
    -------
    tuple
        ``(main_track, total_frames, fps, fill_added, detected_before_fill,
        frame_hw)`` — identical format to ``_infer_tracks_offline_standalone()``.
    """
    if not CV2_OK:
        return None, 0, 25.0, 0, 0, (0, 0)

    # Live module reference so that apply_tuning_overrides() values are
    # respected in worker processes (a bare ``from ... import`` is frozen at
    # import time and cannot be updated by overrides).
    from . import utils as _utils
    from .refinement.yolo import _get_yolo_model
    from .tracker import MultiHandOfflineTracker

    if fill_max_dist is None:
        fill_max_dist = float(_utils.FILL_MAX_DIST)
    if fill_iterations is None:
        fill_iterations = max(1, int(_utils.FILL_ITERATIONS))

    batch_size = max(1, int(_utils.YOLO_BATCH_SIZE))

    model = _get_yolo_model(model_path)
    if model is None:
        logger.error("[YOLO-Only] Failed to load YOLO model from %s", model_path)
        return None, 0, 25.0, 0, 0, (0, 0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    tracker = MultiHandOfflineTracker(
        expected_label=hand_to_track,
        match_thresh=_utils.TRACK_MATCH_THRESH,
        max_gap=_utils.MAX_GAP,
        max_jump_per_frame=_utils.MAX_JUMP_PER_FRAME,
        fps=fps,
    )

    # ── Primary pass: batched YOLO inference with prefetch thread ─────────────
    # A background thread fills frame_q while the main thread drains it in
    # batches of `batch_size` and sends each batch through YOLO in one call.
    frame_q = queue.Queue(maxsize=batch_size * 3)
    reader = threading.Thread(target=_frame_reader_thread, args=(cap, frame_q), daemon=True)
    reader.start()

    frame_idx = 0
    total_frames = 0
    batch_frames = []
    batch_idxs = []

    while True:
        frame = frame_q.get()

        if frame is _SENTINEL:
            # Flush any remaining frames in the partial batch.
            if batch_frames:
                for fidx, dets in zip(
                    batch_idxs,
                    _yolo_full_frame_detections_batch(model, batch_frames),
                ):
                    tracker.associate_frame(fidx, dets)
            break

        batch_frames.append(frame)
        batch_idxs.append(frame_idx)
        frame_idx += 1
        total_frames += 1

        if len(batch_frames) == batch_size:
            for fidx, dets in zip(
                batch_idxs,
                _yolo_full_frame_detections_batch(model, batch_frames),
            ):
                tracker.associate_frame(fidx, dets)
            batch_frames = []
            batch_idxs = []

    reader.join()
    cap.release()

    main_track = tracker.choose_main_track()
    if main_track is None:
        return None, total_frames, fps, 0, 0, (frame_h, frame_w_px)

    detected_before_fill = len(main_track.get("frames", {}))
    missing = [f for f in range(total_frames) if f not in main_track["frames"]]
    if not missing:
        return main_track, total_frames, fps, 0, detected_before_fill, (frame_h, frame_w_px)

    # ── Gap-fill pass ─────────────────────────────────────────────────────────
    # Re-scan missing frames in batches; accept the YOLO detection whose wrist
    # is closest to the last known wrist position and within fill_max_dist.
    fill_added = 0
    _fps_scale = max(1.0, fps) / _utils.BASE_FPS
    _scaled_fill = fill_max_dist / _fps_scale

    for _fill_iter in range(fill_iterations):
        missing = [f for f in range(total_frames) if f not in main_track["frames"]]
        if not missing:
            break
        missing_set = set(missing)

        cap2 = cv2.VideoCapture(video_path)
        frame_q2 = queue.Queue(maxsize=batch_size * 3)
        reader2 = threading.Thread(target=_frame_reader_thread, args=(cap2, frame_q2), daemon=True)
        reader2.start()

        fidx = 0
        gap_batch_frames = []  # frames for YOLO (missing only)
        gap_batch_fidxs = []  # corresponding frame indices

        def _flush_gap_batch():
            """Run YOLO on the current gap batch and update main_track."""
            nonlocal fill_added
            for gfidx, dets in zip(
                gap_batch_fidxs,
                _yolo_full_frame_detections_batch(model, gap_batch_frames),
            ):
                if not dets:
                    continue
                ref_wrist = _nearest_reference_wrist_yolo(main_track, gfidx)
                best_det = None
                best_score = float("inf")
                for lm_arr, hdness in dets:
                    wx = float(lm_arr[0, 0])
                    wy = float(lm_arr[0, 1])
                    dist = (
                        float(((wx - ref_wrist[0]) ** 2 + (wy - ref_wrist[1]) ** 2) ** 0.5)
                        if ref_wrist is not None
                        else 0.0
                    )
                    if dist > _scaled_fill:
                        continue
                    score = dist
                    if (
                        hand_to_track is not None
                        and hdness
                        and hdness[0].category_name != hand_to_track
                    ):
                        score *= float(_utils.HANDEDNESS_PENALTY_MULT)
                    if score < best_score:
                        best_score = score
                        best_det = (lm_arr, hdness)

                if best_det is not None:
                    lm_arr, hdness = best_det
                    if gfidx not in main_track["frames"]:
                        fill_added += 1
                    main_track["frames"][gfidx] = lm_arr
                    kpt_confs = lm_arr[:, 3]
                    valid = kpt_confs[kpt_confs >= conf_thresh]
                    avg_conf = float(np.mean(valid)) if len(valid) > 0 else float(hdness[0].score)
                    main_track.setdefault("conf", {})[gfidx] = avg_conf

        while True:
            frame = frame_q2.get()

            if frame is _SENTINEL:
                if gap_batch_frames:
                    _flush_gap_batch()
                break

            if fidx in missing_set:
                gap_batch_frames.append(frame)
                gap_batch_fidxs.append(fidx)
                if len(gap_batch_frames) == batch_size:
                    _flush_gap_batch()
                    gap_batch_frames = []
                    gap_batch_fidxs = []

            fidx += 1

        reader2.join()
        cap2.release()

    return main_track, total_frames, fps, fill_added, detected_before_fill, (frame_h, frame_w_px)
