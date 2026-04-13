"""
ps_kinematics.workers — Standalone worker functions for multiprocessing.

Contains all per-video processing functions that run inside
``ProcessPoolExecutor`` / ``multiprocessing.Process`` workers dispatched
by ``HandLandmarkProcessor`` in ``core.py``.  These are kept in a
separate module so the import graph is acyclic and so that the
pickling requirements for Windows ``spawn`` workers are satisfied
without pulling in the full ``core`` module.
"""

import logging
import os
import re
import signal

import numpy as np
import pandas as pd

try:
    import cv2

    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarksConnections

    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

    class HandLandmark:
        WRIST = 0
        INDEX_FINGER_MCP = 5
        MIDDLE_FINGER_MCP = 9
        RING_FINGER_MCP = 13
        PINKY_MCP = 17


from .core import (
    _compute_mcp_confidence_proxy,
    _extract_conf,
    _format_clinical_score_overlay_line,
    _lm_vis,
    _roi_zoom_detect,
    _runtime_gpu_config,
    ensure_track_visibility_channel,
    reject_landmark_outliers,
    smooth_track_landmarks,
    trim_track_to_ps_segment,
)
from .io import (
    canonicalize_video_id,
    load_clinical_scores_table,
    load_videoid_to_patientid_map,
    parse_hand_from_path,
    parse_ids_and_visit,
    parse_medication_state_from_path,
    resolve_video_clinical_score,
)
from .kinematics import (
    KinematicAnalyzer,
    _build_pca_angle_deg,
    _build_unwrapped_angle_deg,
    _build_wrist_z_signal,
    _compute_arm_swing_index,
    _compute_inter_mcp_span,
    _compute_inter_mcp_span_px,
    _knuckle_line_angle_rad_standalone,
)
from .plotting import render_two_plot_panel
from .tracker import MultiHandOfflineTracker
from .utils import (
    EXPORT_PLOT_VIDEO,
    FILL_ITERATIONS,
    MAX_PS_DURATION_S,
    MIN_PS_DURATION_S,
    PLOT_WIDTH_RATIO,
    ROI_REDETECT_PADDING,
    ROT_AVG_THRESH,
    SUPERRES_HALF,
    SUPERRES_MODEL_NAME,
    SUPERRES_MODEL_PATH,
    SUPERRES_SCALE,
    TARGET_PROCESSING_FPS,
    TRIM_TO_PS_ACTIVITY,
    USE_CLAHE_ON_FILL,
    USE_PCA_ANGLE,
    _clahe_enhance,
    apply_tuning_overrides,
    series_to_json,
)
from .video_quality import compute_video_quality_metrics

logger = logging.getLogger(__name__)


# ============================================================
# Standalone worker function for multiprocessing
# ============================================================


def _init_worker_gpu(gpu_sem):
    """Worker initializer: store the shared GPU semaphore for this process."""
    from .gpu_manager import init_gpu_semaphore

    init_gpu_semaphore(gpu_sem)


def _robust_worker_entry(
    video_path,
    hand_to_track,
    config,
    save_dir,
    gpu_sem,
    assigned_gpu_id,
    result_queue,
    report_queue,
):
    """Entry point for each multiprocessing.Process worker.

    Parameters
    ----------
    result_queue : multiprocessing.Queue
        Receives exactly one item: the final result dict (or error dict).
    report_queue : multiprocessing.Queue
        Receives GPU-semaphore sentinel dicts ({"_gpu_acquired": True, "pid": ...}
        and {"_gpu_released": True, "pid": ...}) so the parent can track
        semaphore ownership and recover it if this process is killed.
    """
    # Dump C/CUDA stack trace to stderr on SIGSEGV so the crash site in
    # YOLO / cuDNN is visible in the SLURM output rather than a silent death.
    import faulthandler as _fh
    import sys as _sys

    _fh.enable(file=_sys.stderr, all_threads=True)

    # Pin this worker to one GPU so cuda:0 maps to the assigned device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu_id)

    # Mirror the training subprocess policy: cuDNN disabled by default.
    # Certain cuDNN versions trigger a SIGSEGV in F.conv2d on specific GPU
    # compute capabilities.  --enable-cudnn (run_pipeline.py) sets
    # ENABLE_CUDNN=1 in the environment before workers are spawned, which
    # re-enables cuDNN on GPUs where it is known to be stable.
    _cudnn_enabled = os.environ.get("ENABLE_CUDNN", "0") == "1"
    try:
        import torch

        torch.backends.cudnn.enabled = _cudnn_enabled
    except Exception:
        pass

    from .gpu_manager import init_gpu_semaphore

    # Pass report_queue so acquire_gpu/release_gpu send sentinels to parent.
    init_gpu_semaphore(gpu_sem, report_queue=report_queue)
    _log_runtime_diag("worker_gpu_pin", assigned_gpu_id=assigned_gpu_id)
    try:
        result = _process_video_worker(video_path, hand_to_track, config, save_dir)
        result_queue.put(result)
    except Exception as exc:
        result_queue.put(
            {
                "record_type": "ERROR",
                "video_path": video_path,
                "hand": hand_to_track,
                "error": str(exc),
            }
        )


def _collect_runtime_cuda_info():
    info = {
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info["torch_version"] = getattr(torch, "__version__", None)
        info["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda_available
        if cuda_available:
            count = int(torch.cuda.device_count())
            info["cuda_device_count"] = count
            info["cuda_device_0_name"] = torch.cuda.get_device_name(0) if count > 0 else None
        else:
            info["cuda_device_count"] = 0
            info["cuda_device_0_name"] = None
    except Exception as e:
        info["torch_error"] = str(e)
    return info


def _log_runtime_diag(stage, **extra):
    if not _runtime_gpu_config()["enable_runtime_diagnostics"]:
        return
    info = _collect_runtime_cuda_info()
    parts = [f"[RuntimeDiag][{stage}]"]
    for k, v in info.items():
        parts.append(f"{k}={v}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    logger.debug(" ".join(parts))


# ============================================================
# Intra-video parallel chunk inference
# ============================================================


class _SimpleCategory:
    """Picklable stand-in for mediapipe Category.

    MediaPipe's ``Category`` may not survive cross-process pickling
    (pybind11 wrapper).  This plain-Python class exposes the same
    ``.category_name`` / ``.score`` interface used by the tracker and
    ``_extract_conf``, so downstream code works unchanged.
    """

    __slots__ = ("category_name", "score")

    def __init__(self, category_name: str, score: float) -> None:
        self.category_name = category_name
        self.score = score

    def __repr__(self) -> str:
        return f"_SimpleCategory({self.category_name!r}, {self.score:.3f})"


def _infer_detections_chunk(
    video_path: str,
    start_frame: int,
    end_frame: int,
    hand_path: str,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
    native_fps: float,
    stride: int,
) -> dict[int, list]:
    """Run MediaPipe VIDEO-mode inference on frames [start_frame, end_frame).

    This is a **top-level function** so it can be pickled and dispatched to
    a child process via ``multiprocessing.Pool``.

    Returns
    -------
    dict[int, list[tuple[np.ndarray, list[_SimpleCategory]]]]
        ``{frame_idx: [(lm_arr, handedness), ...]}`` for every processed
        frame in the chunk.
    """
    import mediapipe as _mp

    _BaseOptions = _mp.tasks.BaseOptions
    _HLOptions = _mp.tasks.vision.HandLandmarkerOptions
    _RunMode = _mp.tasks.vision.RunningMode

    # Try GPU delegates first, fall back to CPU — mirrors main worker logic.
    _base = None
    for _delegate_kw in (
        {"delegate": _mp.tasks.BaseOptions.Delegate.GPU},
        {"delegate": _mp.tasks.BaseOptions.Delegate.TFLITE_GPU},
        {"delegate": "GPU"},
    ):
        try:
            _base = _BaseOptions(model_asset_path=hand_path, **_delegate_kw)
            break
        except Exception:
            pass
    if _base is None:
        _base = _BaseOptions(model_asset_path=hand_path)

    _opts = _HLOptions(
        running_mode=_RunMode.VIDEO,
        base_options=_base,
        num_hands=2,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detections_map: dict[int, list] = {}
    with _mp.tasks.vision.HandLandmarker.create_from_options(_opts) as landmarker:
        for frame_idx in range(start_frame, end_frame):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if stride > 1 and frame_idx % stride != 0:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ts_ms = int(frame_idx * 1000.0 / native_fps)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            chunk_dets: list[tuple[np.ndarray, list[_SimpleCategory]]] = []
            for hand_i, lm_list in enumerate(result.hand_landmarks):
                lm_arr = np.array(
                    [[lm.x, lm.y, lm.z, _lm_vis(lm)] for lm in lm_list],
                    dtype=np.float32,
                )
                raw_h = (
                    result.handedness[hand_i] if hand_i < len(result.handedness) else []
                )
                simple_h = [_SimpleCategory(c.category_name, c.score) for c in raw_h]
                chunk_dets.append((lm_arr, simple_h))

            if chunk_dets:
                detections_map[frame_idx] = chunk_dets

    cap.release()
    return detections_map


# Per-video SIGALRM timeout used inside worker processes (Linux only).
# Kills a worker that hangs in cap.read() on a stale NFS handle or in
# MediaPipe inference.  Must be slightly less than PER_VIDEO_DEADLINE_S
# in the main process so the worker self-terminates before the parent
# evicts its future.  Value should exceed the longest legitimate video.
_WORKER_SIGALRM_S = 840  # 14 min (< 15 min main-process deadline)


class _WorkerTimeout(RuntimeError):
    """Raised inside a worker when the per-video SIGALRM fires."""


def _worker_alarm_handler(signum, frame):
    raise _WorkerTimeout(
        f"worker exceeded {_WORKER_SIGALRM_S}s wall-clock limit (cap.read or inference hung)"
    )


def _process_video_worker(video_path, hand_to_track, config, save_dir):
    """Standalone worker function that processes a single video."""
    # ── Per-video timeout (Linux only) ────────────────────────────────────
    # Cap each video at _WORKER_SIGALRM_S seconds so a stuck cap.read() on
    # an NFS stale handle (or a hung MediaPipe inference call) doesn't freeze
    # the worker permanently.  The SIGALRM fires only in THIS process and is
    # cancelled in the finally block regardless of outcome.
    # Disabled when config['_no_worker_timeout'] is True (retry-failed mode).
    _alarm_set = False
    if not config.get("_no_worker_timeout", False):
        try:
            signal.signal(signal.SIGALRM, _worker_alarm_handler)
            signal.alarm(_WORKER_SIGALRM_S)
            _alarm_set = True
        except (AttributeError, OSError):
            pass  # Windows / environments without SIGALRM — skip gracefully

    try:
        return _process_video_worker_inner(video_path, hand_to_track, config, save_dir)
    finally:
        if _alarm_set:
            try:
                signal.alarm(0)  # cancel any pending alarm
            except Exception:
                pass


def _process_video_worker_inner(video_path, hand_to_track, config, save_dir):
    """Inner implementation — called by _process_video_worker after alarm setup."""
    # Select tuning preset by video FPS: probe fps cheaply before opening fully.
    _fps_probe = None
    try:
        _cap_probe = cv2.VideoCapture(str(video_path))
        _fps_probe = _cap_probe.get(cv2.CAP_PROP_FPS) or None
        _cap_probe.release()
    except Exception:
        pass
    _tuning_50fps = config.get("tuning_overrides_50fps", {})
    _fps_threshold = config.get("tuning_fps_threshold", 40)
    if _tuning_50fps and _fps_probe is not None and _fps_probe >= _fps_threshold:
        tuning = _tuning_50fps
    else:
        tuning = config.get("tuning_overrides", {})
    if tuning:
        apply_tuning_overrides(tuning)
    gpu_cfg = _runtime_gpu_config()

    _log_runtime_diag(
        "worker_init",
        use_superres=gpu_cfg["use_superres"],
        use_rtmpose=gpu_cfg["use_rtmpose"],
        use_openpose=gpu_cfg["use_openpose"],
        use_yolo_hand=gpu_cfg["use_yolo_hand"],
        keypoint_backend=gpu_cfg["keypoint_backend"],
        export_plot_video=EXPORT_PLOT_VIDEO,
        rtmpose_cfg=gpu_cfg["rtmpose_model_cfg"],
        rtmpose_ckpt=gpu_cfg["rtmpose_checkpoint_path"],
        openpose_proto=gpu_cfg["openpose_proto_path"],
        openpose_weights=gpu_cfg["openpose_weights_path"],
        yolo_hand_model=gpu_cfg["yolo_hand_model_path"],
        superres_model=SUPERRES_MODEL_NAME,
        superres_path=SUPERRES_MODEL_PATH,
    )

    if gpu_cfg.get("use_yolo_only"):
        # YOLO-only mode: no MediaPipe model needed.
        options_video = None
        options_image = None
    else:
        hand_path = config["hand_path"]

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        base = None
        candidates = []
        try:
            candidates.append({"delegate": mp.tasks.BaseOptions.Delegate.GPU})
        except Exception:
            pass
        try:
            candidates.append({"delegate": mp.tasks.BaseOptions.Delegate.TFLITE_GPU})
        except Exception:
            pass
        candidates.append({"delegate": "GPU"})

        for cand in candidates:
            try:
                base = BaseOptions(model_asset_path=hand_path, **cand)
                break
            except Exception:
                pass

        if base is None:
            base = BaseOptions(model_asset_path=hand_path)

        common_kwargs = dict(
            base_options=base,
            num_hands=2,
            min_hand_detection_confidence=tuning.get("min_hand_detection_confidence", 0.35),
            min_hand_presence_confidence=tuning.get("min_hand_presence_confidence", 0.35),
            min_tracking_confidence=tuning.get("min_tracking_confidence", 0.35),
        )

        options_video = HandLandmarkerOptions(running_mode=VisionRunningMode.VIDEO, **common_kwargs)
        options_image = HandLandmarkerOptions(running_mode=VisionRunningMode.IMAGE, **common_kwargs)

    # Build a serializable dict of MP confidence thresholds so that
    # INTRA_VIDEO_WORKERS chunk sub-processes can create their own
    # HandLandmarker instances without needing to pickle options objects.
    _mp_confidence = None
    if not gpu_cfg.get("use_yolo_only"):
        _mp_confidence = {
            "detection": tuning.get("min_hand_detection_confidence", 0.35),
            "presence": tuning.get("min_hand_presence_confidence", 0.35),
            "tracking": tuning.get("min_tracking_confidence", 0.35),
        }

    visit = None
    try:
        vid_score_path = config.get("vid_score_path")
        if vid_score_path:
            vid_df = pd.read_csv(vid_score_path)
            if "video_path" in vid_df.columns and "visit" in vid_df.columns:
                sel = vid_df[vid_df["video_path"] == video_path]
                if sel.empty:
                    norm = os.path.normpath(video_path)
                    try:
                        sel = vid_df[
                            vid_df["video_path"].apply(lambda x: os.path.normpath(str(x)) == norm)
                        ]
                    except Exception:
                        sel = pd.DataFrame()
                if not sel.empty:
                    v = sel.iloc[0]["visit"]
                    if pd.notna(v):
                        visit = v
    except Exception:
        visit = None

    from .workers import _process_single_video

    id2vid_csv_path = config.get("id2vid_csv_path", config.get("id2vid"))
    score_csv_path = config.get("score_csv_path")
    score_column = config.get("score_column", "ProS")
    video_to_patient = load_videoid_to_patientid_map(id2vid_csv_path) if id2vid_csv_path else {}
    scores_df = load_clinical_scores_table(score_csv_path, score_column)
    return _process_single_video(
        video_path=video_path,
        hand_to_track=hand_to_track,
        save_dir=save_dir,
        options_video=options_video,
        options_image=options_image,
        visit=visit,
        cutoff_hz=tuning.get("cutoff_hz", 2.5),
        highpass_hz=tuning.get("highpass_hz", 0.1),
        prominence_deg=tuning.get("prominence_deg", 10.0),
        filter_order=int(tuning.get("filter_order", 4)),
        max_movement_hz=tuning.get("max_movement_hz", 3.0),
        adaptive_prom_frac=tuning.get("adaptive_prom_frac", 0.20),
        trim_to_ps=tuning.get("trim_to_ps_activity", TRIM_TO_PS_ACTIVITY),
        max_ps_duration_s=tuning.get("max_ps_duration_s", MAX_PS_DURATION_S),
        min_ps_duration_s=tuning.get("min_ps_duration_s", MIN_PS_DURATION_S),
        video_to_patient=video_to_patient,
        scores_df=scores_df,
        score_column=score_column,
        use_parabolic_interp=tuning.get("use_parabolic_interp", False),
        hand_path=config.get("hand_path"),
        mp_confidence=_mp_confidence,
    )


# ============================================================
# Standalone helper functions for multiprocessing workers
# ============================================================


def _frame_timestamp_ms(frame_idx, fps):
    return int(frame_idx * 1000.0 / fps)


def _wrist_xy_standalone(lm_arr):
    return float(lm_arr[HandLandmark.WRIST, 0]), float(lm_arr[HandLandmark.WRIST, 1])


def _dist_standalone(a, b):
    return float((((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5))


def _nearest_reference_wrist_standalone(track, frame_idx):
    if track is None or not track.get("frames"):
        return None
    frames = sorted(track["frames"].keys())
    best_f = min(frames, key=lambda f: abs(f - frame_idx))
    return _wrist_xy_standalone(track["frames"][best_f])


def _choose_detection_for_track_spatial_standalone(
    detections, ref_wrist, hand_to_track, fill_max_dist=None
):
    if not detections:
        return None
    if ref_wrist is None:
        if hand_to_track is not None:
            for lm_arr, handedness in detections:
                if handedness and handedness[0].category_name == hand_to_track:
                    return (lm_arr, handedness)
        return None

    from . import utils as _utils

    _fill_max_dist = fill_max_dist if fill_max_dist is not None else _utils.FILL_MAX_DIST

    best = None
    best_score = float("inf")
    for lm_arr, handedness in detections:
        w = _wrist_xy_standalone(lm_arr)
        d = _dist_standalone(w, ref_wrist)
        if hand_to_track is not None and handedness:
            if handedness[0].category_name != hand_to_track:
                d *= _utils.HANDEDNESS_PENALTY_MULT
        if d < best_score:
            best_score = d
            best = (lm_arr, handedness, d)

    if best is None:
        return None
    lm_arr, handedness, d = best
    if d > _fill_max_dist:
        return None
    return (lm_arr, handedness)


def _compute_appearance_metrics_standalone(track, total_frames):
    if track is None or not track.get("frames"):
        return None, None, None, None, 0, 0, 0
    frames = sorted(track["frames"].keys())
    first, last = frames[0], frames[-1]
    window_len = last - first + 1
    detected_in_window = sum(1 for f in range(first, last + 1) if f in track["frames"])
    detected_total = len(track["frames"])
    adjusted = 100.0 * detected_in_window / window_len if window_len > 0 else None
    non_adjusted = 100.0 * detected_total / total_frames if total_frames > 0 else None
    return adjusted, non_adjusted, first, last, detected_in_window, window_len, detected_total


def _avg_confidence_from_track_standalone(track, first, last):
    if track is None or first is None or last is None:
        return None, 0.0, 0
    vals = []
    for f in range(first, last + 1):
        if f in track.get("conf", {}):
            vals.append(track["conf"][f])
    if not vals:
        return None, 0.0, 0
    s = float(np.sum(vals))
    c = len(vals)
    return (s / c), s, c


def _draw_hand_from_array_standalone(image_bgr, lm_arr):
    if not CV2_OK:
        return
    h, w = image_bgr.shape[:2]
    CONNECTION_COLOR = (0, 255, 0)
    KINEMATIC_LANDMARK_COLOR = (0, 165, 255)
    DEFAULT_LANDMARK_COLOR = (0, 255, 0)

    if MEDIAPIPE_OK:
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            x1, y1 = lm_arr[conn.start, 0], lm_arr[conn.start, 1]
            x2, y2 = lm_arr[conn.end, 0], lm_arr[conn.end, 1]
            p1 = (int(x1 * w), int(y1 * h))
            p2 = (int(x2 * w), int(y2 * h))
            cv2.line(image_bgr, p1, p2, CONNECTION_COLOR, 1)

    kinematic_lms = {
        HandLandmark.WRIST,
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.MIDDLE_FINGER_MCP,
        HandLandmark.RING_FINGER_MCP,
        HandLandmark.PINKY_MCP,
    }
    for i in range(lm_arr.shape[0]):
        x, y = lm_arr[i, 0], lm_arr[i, 1]
        p = (int(x * w), int(y * h))
        if i in kinematic_lms:
            color, radius = KINEMATIC_LANDMARK_COLOR, 3
        else:
            color, radius = DEFAULT_LANDMARK_COLOR, 2
        cv2.circle(image_bgr, p, radius, color, -1)


def _draw_top_right_text_standalone(image_bgr, lines):
    if not CV2_OK:
        return
    MARGIN = 10
    FONT_SIZE = 0.7
    FONT_THICKNESS = 2

    if isinstance(lines, str):
        lines = [lines]
    h, w = image_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, FONT_SIZE, FONT_THICKNESS)[0] for t in lines]
    max_tw = max((tw for tw, th in sizes), default=0)
    x = max(MARGIN, w - max_tw - MARGIN)
    y = MARGIN
    for i, text in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, font, FONT_SIZE, FONT_THICKNESS)
        yy = y + (i + 1) * (th + 6)
        cv2.putText(
            image_bgr, text, (x, yy), font, FONT_SIZE, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA
        )
        cv2.putText(
            image_bgr, text, (x, yy), font, FONT_SIZE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA
        )


def _render_video_with_plots_standalone(
    video_path,
    save_vid_path,
    track,
    overlay_lines,
    time_s,
    filtered_deg,
    metrics,
    plot_width_ratio=PLOT_WIDTH_RATIO,
    start_frame=None,
    end_frame=None,
):
    """Render tracking overlay + plot panel to an output video.

    When start_frame/end_frame are provided (PS activity window), only those
    frames are rendered.  This avoids re-reading the full video for long
    recordings where only a short PS segment was detected.
    """
    if not CV2_OK:
        return
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Seek to the first frame we need so we don't decode the entire video.
    _start = int(start_frame) if start_frame is not None else 0
    _end = int(end_frame) if end_frame is not None else None
    if _start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, _start)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Unable to read video: {video_path}")

    h, w = first_frame.shape[:2]
    plot_w = int(w * float(plot_width_ratio))
    out_w = w + plot_w
    out_h = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_vid_path, fourcc, fps, (out_w, out_h))

    # Re-seek so the first frame is included in the output.
    cap.set(cv2.CAP_PROP_POS_FRAMES, _start)
    frame_idx = _start

    while True:
        if _end is not None and frame_idx > _end:
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break

        annotated = frame_bgr.copy()

        if track is not None and frame_idx in track.get("frames", {}):
            _draw_hand_from_array_standalone(annotated, track["frames"][frame_idx])

        if overlay_lines:
            _draw_top_right_text_standalone(annotated, overlay_lines)

        plot_panel = render_two_plot_panel(
            time_s=time_s,
            filtered_deg=filtered_deg,
            metrics=metrics,
            current_frame_idx=frame_idx,
            fps=fps,
            panel_h=h,
            panel_w=plot_w,
        )

        combined = cv2.hconcat([annotated, plot_panel])
        out.write(combined)

        frame_idx += 1

    cap.release()
    out.release()


def _compute_frame_stride(native_fps: float) -> int:
    """Return the frame stride for the given native fps.

    When ``TARGET_PROCESSING_FPS`` is set (> 0) and *native_fps* exceeds
    it, we skip frames so the effective rate stays near the target.
    Returns 1 (process every frame) when no downsampling is needed.
    """
    from . import utils as _utils

    target = _utils.TARGET_PROCESSING_FPS
    if not target or target <= 0 or native_fps <= target:
        return 1
    return max(1, int(native_fps / target))


def _infer_tracks_offline_standalone(
    video_path,
    hand_to_track,
    options_video,
    options_image,
    hand_path=None,
    mp_confidence=None,
):
    """Standalone version of track inference for multiprocessing.

    Parameters
    ----------
    hand_path : str or None
        Path to the MediaPipe hand landmarker ``.task`` file.  Required
        when ``INTRA_VIDEO_WORKERS > 1`` so that chunk sub-processes can
        build their own ``HandLandmarker`` instances.
    mp_confidence : dict or None
        ``{"detection": float, "presence": float, "tracking": float}``
        MediaPipe confidence thresholds.  Required when
        ``INTRA_VIDEO_WORKERS > 1``.

    Returns
    -------
    tuple : (main_track, total_frames, fps, fill_added, detected_before_fill, frame_hw)
        frame_hw is (height, width) in pixels, or (0, 0) on error.
    """
    if not MEDIAPIPE_OK or not CV2_OK:
        return None, 0, 25.0, 0, 0, (0, 0)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Use live module references for overridable constants so that
    # apply_tuning_overrides() values are respected in worker processes.
    # A bare `from ... import` binding is frozen at import time.
    from . import utils as _utils

    stride = _compute_frame_stride(fps)
    if stride > 1:
        logger.info(
            "[Stride] %s: native %.0f fps → stride %d (effective %.1f fps)",
            os.path.basename(video_path),
            fps,
            stride,
            fps / stride,
        )

    intra_workers = max(1, int(_utils.INTRA_VIDEO_WORKERS))

    tracker = MultiHandOfflineTracker(
        expected_label=hand_to_track,
        match_thresh=_utils.TRACK_MATCH_THRESH,
        max_gap=_utils.MAX_GAP,
        max_jump_per_frame=_utils.MAX_JUMP_PER_FRAME,
        fps=fps,
    )

    # ── Pass 1: MediaPipe VIDEO-mode inference ────────────────────────────
    if intra_workers > 1 and hand_path and mp_confidence and total_frames > 0:
        # Parallel chunk inference.
        cap.release()  # chunks open their own captures

        # If FRAME_COUNT was unreliable (0), fall back to serial.
        chunk_size = max(1, total_frames // intra_workers)
        chunks = []
        for i in range(intra_workers):
            s = i * chunk_size
            e = min((i + 1) * chunk_size, total_frames) if i < intra_workers - 1 else total_frames
            if s < total_frames:
                chunks.append((s, e))

        logger.info(
            "[IntraVideo] %s: %d chunks × ~%d frames across %d sub-processes",
            os.path.basename(video_path),
            len(chunks),
            chunk_size,
            intra_workers,
        )

        from multiprocessing import get_context as _mp_ctx

        ctx = _mp_ctx("spawn")
        chunk_args = [
            (
                video_path,
                s,
                e,
                hand_path,
                mp_confidence["detection"],
                mp_confidence["presence"],
                mp_confidence["tracking"],
                fps,
                stride,
            )
            for s, e in chunks
        ]
        try:
            with ctx.Pool(processes=len(chunks)) as pool:
                chunk_results = pool.starmap(_infer_detections_chunk, chunk_args)
        except Exception as exc:
            logger.warning(
                "[IntraVideo] Pool failed (%s); falling back to serial inference.",
                exc,
            )
            chunk_results = None

        if chunk_results is not None:
            # Feed merged detections into the tracker in frame order.
            all_detections: dict[int, list] = {}
            for cr in chunk_results:
                all_detections.update(cr)
            for frame_idx in sorted(all_detections):
                tracker.associate_frame(frame_idx, all_detections[frame_idx])
        else:
            # Serial fallback — re-open video and process the old way.
            intra_workers = 1  # flag for below
            cap = cv2.VideoCapture(video_path)
            total_frames = _serial_inference_pass(
                cap, fps, stride, tracker, options_video
            )
            cap.release()
    else:
        # Serial inference (original path).
        total_frames = _serial_inference_pass(
            cap, fps, stride, tracker, options_video
        )
        cap.release()

    main_track = tracker.choose_main_track()

    if main_track is None:
        return None, total_frames, fps, 0, 0, (frame_h, frame_w)

    detected_before_fill = len(main_track.get("frames", {}))

    # Build the set of frames that are eligible for fill (only strided frames).
    # Non-strided frames are intentionally skipped — filling them would defeat
    # the purpose of stride-based downsampling.
    _eligible = set(range(0, total_frames, stride)) if stride > 1 else set(range(total_frames))
    missing = sorted(f for f in _eligible if f not in main_track["frames"])
    if not missing:
        return main_track, total_frames, fps, 0, detected_before_fill, (frame_h, frame_w)

    fill_added = 0
    with mp.tasks.vision.HandLandmarker.create_from_options(options_image) as landmarker_img:
        for _fill_iter in range(max(1, int(FILL_ITERATIONS))):
            missing = sorted(f for f in _eligible if f not in main_track["frames"])
            if not missing:
                break
            missing_set = set(missing)
            cap2 = cv2.VideoCapture(video_path)
            frame_idx = 0
            while True:
                ok, frame_bgr = cap2.read()
                if not ok:
                    break
                if frame_idx in missing_set:
                    if USE_CLAHE_ON_FILL:
                        frame_bgr = _clahe_enhance(frame_bgr)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    result = landmarker_img.detect(mp_image)

                    detections = []
                    for hand_i, lm_list in enumerate(result.hand_landmarks):
                        lm_arr = np.array(
                            [[lm.x, lm.y, lm.z, _lm_vis(lm)] for lm in lm_list], dtype=np.float32
                        )
                        handedness = (
                            result.handedness[hand_i] if hand_i < len(result.handedness) else []
                        )
                        detections.append((lm_arr, handedness))

                    if detections:
                        ref_wrist = _nearest_reference_wrist_standalone(main_track, frame_idx)
                        from . import utils as _utils

                        _fps_scale = max(1.0, fps) / _utils.BASE_FPS
                        _scaled_fill = _utils.FILL_MAX_DIST / _fps_scale
                        chosen = _choose_detection_for_track_spatial_standalone(
                            detections, ref_wrist, hand_to_track, fill_max_dist=_scaled_fill
                        )
                        if chosen is not None:
                            lm_arr, handedness = chosen
                            if frame_idx not in main_track["frames"]:
                                fill_added += 1
                            main_track["frames"][frame_idx] = lm_arr
                            main_track.setdefault("conf", {})[frame_idx] = _extract_conf(
                                handedness, hand_to_track
                            )

                    if frame_idx in missing_set and frame_idx not in main_track["frames"]:
                        zoomed = _roi_zoom_detect(
                            frame_bgr, frame_idx, main_track, landmarker_img, hand_to_track
                        )
                        if zoomed is not None:
                            main_track["frames"][frame_idx] = zoomed
                            main_track.setdefault("conf", {})[frame_idx] = 0.3
                            fill_added += 1

                frame_idx += 1
            cap2.release()

    return main_track, total_frames, fps, fill_added, detected_before_fill, (frame_h, frame_w)


def _serial_inference_pass(cap, fps, stride, tracker, options_video):
    """Run MediaPipe VIDEO-mode inference serially (original logic).

    Returns the total frame count.
    """
    frame_idx = 0
    total_frames = 0
    with mp.tasks.vision.HandLandmarker.create_from_options(options_video) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            total_frames += 1
            if stride > 1 and frame_idx % stride != 0:
                frame_idx += 1
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ts_ms = _frame_timestamp_ms(frame_idx, fps)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            detections = []
            for hand_i, lm_list in enumerate(result.hand_landmarks):
                lm_arr = np.array(
                    [[lm.x, lm.y, lm.z, _lm_vis(lm)] for lm in lm_list], dtype=np.float32
                )
                handedness = result.handedness[hand_i] if hand_i < len(result.handedness) else []
                detections.append((lm_arr, handedness))

            tracker.associate_frame(frame_idx, detections)
            frame_idx += 1

    return total_frames


def _process_single_video(
    video_path,
    hand_to_track,
    save_dir,
    options_video,
    options_image,
    visit=None,
    cutoff_hz=2.5,
    highpass_hz=0.1,
    prominence_deg=10.0,
    filter_order=4,
    max_movement_hz=3.0,
    adaptive_prom_frac=0.20,
    trim_to_ps=False,
    max_ps_duration_s=10.0,
    min_ps_duration_s=None,
    video_to_patient=None,
    scores_df=None,
    score_column="ProS",
    use_parabolic_interp=False,
    hand_path=None,
    mp_confidence=None,
):
    """Process a single video file — standalone worker implementation."""
    normalized_path = os.path.normpath(video_path)
    base_name = os.path.basename(normalized_path)
    subfolder_name = os.path.basename(os.path.dirname(normalized_path))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(normalized_path)))
    if re.match(r"(?i)subject_\d+", grandparent):
        new_filename = f"{grandparent}_{subfolder_name}_{base_name}"
    else:
        new_filename = f"{subfolder_name}_{base_name}"
    save_vid_path = os.path.join(save_dir, new_filename)

    gpu_cfg = _runtime_gpu_config()

    # ── Primary tracking: YOLO-only or MediaPipe ─────────────────────────────
    if gpu_cfg.get("use_yolo_only"):
        # Full YOLO-only path — GPU semaphore acquired here for the entire
        # tracking pass (no separate refinement pass needed).
        from .gpu_manager import acquire_gpu, cleanup_gpu, release_gpu

        _yolo_gpu_acquired = acquire_gpu()
        if not _yolo_gpu_acquired:
            logger.warning(
                "[YOLO-Only] acquire_gpu() timed out for %s; skipping video.",
                os.path.basename(video_path),
            )
            return {
                "record_type": "TIMEOUT",
                "video_path": video_path,
                "hand": hand_to_track,
                "error": "YOLO-Only GPU semaphore acquire timeout",
            }
        try:
            from .yolo_tracker import _infer_tracks_yolo_standalone

            main_track, total_frames, fps, fill_added, detected_before_fill, frame_hw = (
                _infer_tracks_yolo_standalone(
                    video_path,
                    hand_to_track,
                    model_path=gpu_cfg["yolo_hand_model_path"],
                    conf_thresh=gpu_cfg["yolo_hand_conf_threshold"],
                )
            )
        finally:
            cleanup_gpu()
            release_gpu()
    else:
        # MediaPipe primary tracking
        main_track, total_frames, fps, fill_added, detected_before_fill, frame_hw = (
            _infer_tracks_offline_standalone(
                video_path,
                hand_to_track,
                options_video,
                options_image,
                hand_path=hand_path,
                mp_confidence=mp_confidence,
            )
        )

    _log_runtime_diag(
        "video_start",
        video=os.path.basename(video_path),
        main_track_found=bool(main_track is not None),
        detected_before_fill=detected_before_fill,
        fill_added=fill_added,
        total_frames=total_frames,
        fps=fps,
    )

    # ── Landmark refinement passes (SuperRes / RTMPose / OpenPose / YOLO) ──
    # A shared GPU semaphore is used only for GPU-backed refiners.
    # Refinement passes are merged into a single video read.
    # Skipped entirely in YOLO-only mode (tracking already used GPU above).
    superres_refined = 0
    rtmpose_refined = 0
    openpose_refined = 0
    yolo_refined = 0
    need_refinement = (
        not gpu_cfg.get("use_yolo_only")
        and main_track is not None
        and (
            gpu_cfg["use_superres"]
            or gpu_cfg["use_rtmpose"]
            or gpu_cfg["use_openpose"]
            or gpu_cfg["use_yolo_hand"]
        )
    )
    if need_refinement:
        need_gpu_lock = bool(
            gpu_cfg["use_superres"] or gpu_cfg["use_rtmpose"] or gpu_cfg["use_yolo_hand"]
        )
        gpu_acquired = False
        if need_gpu_lock:
            from .gpu_manager import acquire_gpu, cleanup_gpu, release_gpu

            gpu_acquired = acquire_gpu()  # returns False on timeout instead of hanging forever
            if not gpu_acquired:
                need_refinement = False
                logger.warning(
                    "[GPU] acquire_gpu() timed out; skipping SuperRes/RTMPose refinement for this video "
                    "to avoid deadlock."
                )
        else:
            gpu_acquired = True

    if need_refinement and gpu_acquired:
        try:
            frame_set = set(main_track.get("frames", {}).keys())
            sr_landmarker = None
            if gpu_cfg["use_superres"]:
                try:
                    from .refinement.superres import superres_refine_landmarks

                    sr_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                        options_image
                    )
                except Exception as e:
                    logger.error("[SuperRes] Could not create landmarker: %s", e)
            rtm_inferencer_ok = False
            if gpu_cfg["use_rtmpose"]:
                try:
                    from .refinement.rtmpose import _get_inferencer, refine_landmarks_rtmpose

                    if (
                        _get_inferencer(
                            gpu_cfg["rtmpose_model_cfg"],
                            checkpoint_path=gpu_cfg["rtmpose_checkpoint_path"],
                        )
                        is not None
                    ):
                        rtm_inferencer_ok = True
                except Exception as e:
                    logger.error("[RTMPose] Could not initialise inferencer: %s", e)

            openpose_ready = False
            if gpu_cfg["use_openpose"]:
                try:
                    from .refinement.openpose import _get_openpose_net, refine_landmarks_openpose

                    openpose_ready = (
                        _get_openpose_net(
                            gpu_cfg["openpose_proto_path"],
                            gpu_cfg["openpose_weights_path"],
                            use_cuda=gpu_cfg["openpose_use_cuda"],
                        )
                        is not None
                    )
                except Exception as e:
                    logger.error("[OpenPose] Could not initialise network: %s", e)

            yolo_ready = False
            if gpu_cfg["use_yolo_hand"]:
                try:
                    from .refinement.yolo import _get_yolo_model, refine_landmarks_yolo

                    if _get_yolo_model(gpu_cfg["yolo_hand_model_path"]) is not None:
                        yolo_ready = True
                except Exception as e:
                    logger.error("[YOLO-Hand] Could not initialise model: %s", e)

            cap_gpu = cv2.VideoCapture(video_path)
            fidx = 0
            while True:
                ok, frame_bgr = cap_gpu.read()
                if not ok:
                    break
                if fidx in frame_set:
                    if sr_landmarker is not None:
                        try:
                            refined_sr = superres_refine_landmarks(
                                frame_bgr,
                                main_track,
                                fidx,
                                sr_landmarker,
                                hand_to_track=hand_to_track,
                                scale=SUPERRES_SCALE,
                                model_name=SUPERRES_MODEL_NAME,
                                model_path=SUPERRES_MODEL_PATH,
                                half=SUPERRES_HALF,
                                padding=ROI_REDETECT_PADDING,
                            )
                            if refined_sr is not None:
                                main_track["frames"][fidx] = refined_sr
                                superres_refined += 1
                        except Exception:
                            pass
                    if rtm_inferencer_ok:
                        try:
                            lm_current = main_track["frames"][fidx]
                            refined_rtm = refine_landmarks_rtmpose(
                                frame_bgr,
                                lm_current,
                                model_cfg=gpu_cfg["rtmpose_model_cfg"],
                                checkpoint_path=gpu_cfg["rtmpose_checkpoint_path"],
                                padding=gpu_cfg["rtmpose_bbox_padding"],
                            )
                            if refined_rtm is not None:
                                main_track["frames"][fidx] = refined_rtm
                                rtmpose_refined += 1
                        except Exception:
                            pass
                    if openpose_ready:
                        try:
                            lm_current = main_track["frames"][fidx]
                            refined_openpose = refine_landmarks_openpose(
                                frame_bgr,
                                lm_current,
                                proto_path=gpu_cfg["openpose_proto_path"],
                                weights_path=gpu_cfg["openpose_weights_path"],
                                padding=gpu_cfg["openpose_bbox_padding"],
                                conf_thresh=gpu_cfg["openpose_conf_threshold"],
                                input_size=gpu_cfg["openpose_input_size"],
                                use_cuda=gpu_cfg["openpose_use_cuda"],
                            )
                            if refined_openpose is not None:
                                main_track["frames"][fidx] = refined_openpose
                                openpose_refined += 1
                        except Exception:
                            pass
                    if yolo_ready:
                        try:
                            lm_current = main_track["frames"][fidx]
                            refined_yolo = refine_landmarks_yolo(
                                frame_bgr,
                                lm_current,
                                model_path=gpu_cfg["yolo_hand_model_path"],
                                padding=gpu_cfg["yolo_hand_bbox_padding"],
                                conf_thresh=gpu_cfg["yolo_hand_conf_threshold"],
                            )
                            if refined_yolo is not None:
                                main_track["frames"][fidx] = refined_yolo
                                yolo_refined += 1
                        except Exception:
                            pass
                fidx += 1
            cap_gpu.release()

            if sr_landmarker is not None:
                try:
                    sr_landmarker.close()
                except Exception:
                    pass
            if gpu_cfg["use_openpose"]:
                try:
                    from .refinement.openpose import cleanup_openpose

                    cleanup_openpose()
                except Exception:
                    pass
            if gpu_cfg["use_yolo_hand"]:
                try:
                    from .refinement.yolo import cleanup_yolo

                    cleanup_yolo()
                except Exception:
                    pass
        finally:
            if need_gpu_lock:
                cleanup_gpu()
                release_gpu()

    _log_runtime_diag(
        "video_refine",
        video=os.path.basename(video_path),
        keypoint_backend=gpu_cfg["keypoint_backend"],
        superres_enabled=gpu_cfg["use_superres"],
        rtmpose_enabled=gpu_cfg["use_rtmpose"],
        openpose_enabled=gpu_cfg["use_openpose"],
        yolo_hand_enabled=gpu_cfg["use_yolo_hand"],
        superres_refined=superres_refined,
        rtmpose_refined=rtmpose_refined,
        openpose_refined=openpose_refined,
        yolo_refined=yolo_refined,
    )

    main_track = ensure_track_visibility_channel(main_track)
    main_track = smooth_track_landmarks(main_track, total_frames, fps=fps)
    main_track = reject_landmark_outliers(main_track, fps=fps)

    ps_trimmed = False
    ps_start_frame = None
    ps_end_frame = None
    ps_duration_s = None
    if trim_to_ps and main_track is not None:
        main_track, ps_start_frame, ps_end_frame = trim_track_to_ps_segment(
            main_track,
            total_frames,
            fps,
            max_duration_s=max_ps_duration_s,
            min_duration_s=min_ps_duration_s,
        )
        if ps_start_frame is not None:
            ps_trimmed = True
            ps_duration_s = (ps_end_frame - ps_start_frame + 1) / fps

    medication_state = parse_medication_state_from_path(video_path)
    hand = parse_hand_from_path(video_path)
    if hand is None and main_track is not None:
        hand = main_track.get("hand_label")

    ids_parsed, visit_parsed = parse_ids_and_visit(video_path)
    ids = canonicalize_video_id(ids_parsed)
    if visit is None:
        visit = visit_parsed

    clinical_score_info = resolve_video_clinical_score(
        video_path=video_path,
        video_to_patient=video_to_patient,
        scores_df=scores_df,
        score_column=score_column,
        visit=visit,
        medication_state=medication_state,
        hand=hand,
    )
    patient_id = clinical_score_info.get("patient_id")
    clinical_score = clinical_score_info.get("score_clean")
    clinical_score_raw = clinical_score_info.get("score_raw")
    score_line = None
    if scores_df is not None and not scores_df.empty:
        score_line = _format_clinical_score_overlay_line(score_column, clinical_score)

    adj_app, non_adj_app, first, last, det_win, win_len, det_total = (
        _compute_appearance_metrics_standalone(main_track, total_frames)
    )
    avg_conf, conf_sum, conf_count = _avg_confidence_from_track_standalone(main_track, first, last)

    time_s = np.arange(total_frames, dtype=float) / float(fps)

    metrics = None
    filtered_deg_full = np.full((total_frames,), np.nan, dtype=float)
    raw_deg_full = np.full((total_frames,), np.nan, dtype=float)
    signal_quality = 0.0
    sq_sub_scores = {}
    _mcp_indices = [
        int(HandLandmark.INDEX_FINGER_MCP),
        int(HandLandmark.MIDDLE_FINGER_MCP),
        int(HandLandmark.RING_FINGER_MCP),
        int(HandLandmark.PINKY_MCP),
    ]
    conf_idx_series = np.full((total_frames,), np.nan, dtype=float)
    conf_mid_series = np.full((total_frames,), np.nan, dtype=float)
    conf_ring_series = np.full((total_frames,), np.nan, dtype=float)
    conf_pinky_series = np.full((total_frames,), np.nan, dtype=float)
    conf_min_series = np.full((total_frames,), np.nan, dtype=float)
    conf_used_mask_series = np.zeros((total_frames,), dtype=float)
    conf_low_thresh = 0.5
    _frames_dict: dict = {}

    if main_track is not None:
        _frames_dict = main_track.get("frames", {})
        _conf_map, _conf_min, conf_low_thresh = _compute_mcp_confidence_proxy(
            _frames_dict,
            total_frames,
            _mcp_indices,
        )
        conf_idx_series = _conf_map[_mcp_indices[0]]
        conf_mid_series = _conf_map[_mcp_indices[1]]
        conf_ring_series = _conf_map[_mcp_indices[2]]
        conf_pinky_series = _conf_map[_mcp_indices[3]]
        conf_min_series = _conf_min
        conf_used_mask_series = np.where(
            np.isfinite(_conf_min) & (_conf_min >= conf_low_thresh),
            1.0,
            0.0,
        )

        if USE_PCA_ANGLE:
            raw_deg_full = _build_pca_angle_deg(_frames_dict, total_frames, fps=fps)
        else:
            raw_deg_full = _build_unwrapped_angle_deg(
                _frames_dict, total_frames, _knuckle_line_angle_rad_standalone, fps=fps
            )

        wrist_z = _build_wrist_z_signal(_frames_dict, total_frames)

        analyzer = KinematicAnalyzer(
            time_s,
            raw_deg_full,
            fps=fps,
            cutoff_hz=cutoff_hz,
            filter_order=filter_order,
            highpass_hz=highpass_hz,
            wrist_z=wrist_z,
            use_parabolic_interp=use_parabolic_interp,
        )
        # When PS-activity trimming is active, erase all filtered signal
        # outside the detected activity window so that filter bleed cannot
        # produce spurious peaks/cycles in non-task regions.
        if ps_trimmed and ps_start_frame is not None:
            if ps_start_frame > 0:
                analyzer.clean_signal[:ps_start_frame] = 0.0
            if ps_end_frame + 1 < total_frames:
                analyzer.clean_signal[ps_end_frame + 1 :] = 0.0
        # Keep a display copy with NaN outside the PS window (shows blank
        # in plots) while the analyzer retains 0.0 (no spurious peaks).
        filtered_deg_full = analyzer.clean_signal.copy()
        if ps_trimmed and ps_start_frame is not None:
            filtered_deg_full[:ps_start_frame] = np.nan
            filtered_deg_full[ps_end_frame + 1 :] = np.nan
        metrics = analyzer.extract_features(
            prominence_deg=prominence_deg,
            max_movement_hz=max_movement_hz,
            adaptive_prom_frac=adaptive_prom_frac,
        )

        # --- SSL signal export ---
        # Saves the normalised, fixed-length angle signal for offline SSL
        # pretraining.  One .npy file per video, stored under save_dir/signals/.
        # Controlled by config key 'ssl_save_signals' (default True).
        if save_dir is not None:
            try:
                _ssl_signal_dir = os.path.join(save_dir, "signals")
                os.makedirs(_ssl_signal_dir, exist_ok=True)
                _ssl_max_len = 1024
                _ssl_sig = analyzer.get_ssl_signal(max_len=_ssl_max_len)
                _ssl_stem = str(ids).replace("/", "_").replace("\\", "_") if ids else "unknown"
                np.save(os.path.join(_ssl_signal_dir, f"{_ssl_stem}.npy"), _ssl_sig)
            except Exception as _ssl_exc:
                import warnings as _w

                _w.warn(f"SSL signal export failed for {ids}: {_ssl_exc}")

        sq_result = analyzer.compute_signal_quality(
            metrics,
            ps_start_frame=ps_start_frame if ps_trimmed else None,
            ps_end_frame=ps_end_frame if ps_trimmed else None,
        )
        signal_quality = sq_result["signal_quality"]
        sq_sub_scores = sq_result["sq_sub_scores"]

        # Compute inter-MCP span for camera-distance normalisation
        inter_mcp_span = _compute_inter_mcp_span(_frames_dict, total_frames)
        arm_swing_index = _compute_arm_swing_index(
            _frames_dict,
            total_frames,
            ps_start_frame=ps_start_frame if ps_trimmed else None,
            ps_end_frame=ps_end_frame if ps_trimmed else None,
            inter_mcp_span=inter_mcp_span,
        )
        if metrics is not None:
            metrics["inter_mcp_span"] = inter_mcp_span
            metrics["arm_swing_index"] = arm_swing_index

    # ── Video quality factor metrics ──────────────────────────────────
    vq_metrics = {}
    if main_track is not None:
        _vq_frames = main_track.get("frames", {})
        vq_metrics = compute_video_quality_metrics(
            video_path,
            _vq_frames,
            total_frames,
            fps,
            ps_start_frame=ps_start_frame if ps_trimmed else None,
            ps_end_frame=ps_end_frame if ps_trimmed else None,
            frame_hw=frame_hw,
        )

    if main_track is None:
        overlay_lines = ["No main track", f"FPS: {fps:.1f} Frames: {total_frames}"]
    else:
        overlay_lines = [
            f"AdjApp: {adj_app:5.1f}%" if adj_app is not None else "AdjApp: N/A",
            f"NonAdjApp: {non_adj_app:5.1f}%" if non_adj_app is not None else "NonAdjApp: N/A",
            (
                f"FillAdded: {fill_added} Conf: {avg_conf:.2f}"
                if avg_conf is not None
                else f"FillAdded: {fill_added}"
            ),
        ]
        if score_line:
            overlay_lines.insert(0, score_line)
        if metrics is not None:
            overlay_lines.extend(
                [
                    f"MeanAmp: {metrics['avg_amp']:.1f} deg  AmpCV: {metrics['amp_cv']:.1f}%",
                    f"MeanFreq: {metrics['freq']:.2f} Hz",
                    f"NormDecSlope: {metrics['norm_decrement_slope']:.2f}%/s  RhythmCV: {metrics['cv']:.1f}%",
                    f"NormTISlope: {metrics['norm_ti_slope']:.2f}%/cyc  Arrests: {metrics['num_arrests']}",
                    f"SigQuality: {signal_quality:.3f}",
                ]
            )
        else:
            overlay_lines.append("Metrics: insufficient cycles")
        if ps_trimmed:
            overlay_lines.append(f"PS: {ps_duration_s:.1f}s [{ps_start_frame}-{ps_end_frame}]")
    if main_track is None and score_line:
        overlay_lines.insert(0, score_line)

    render_ok = False
    if EXPORT_PLOT_VIDEO:
        try:
            _render_video_with_plots_standalone(
                video_path=video_path,
                save_vid_path=save_vid_path,
                track=main_track,
                overlay_lines=overlay_lines,
                time_s=time_s,
                filtered_deg=filtered_deg_full,
                metrics=metrics,
                start_frame=ps_start_frame if ps_trimmed else None,
                end_frame=ps_end_frame if ps_trimmed else None,
            )
            render_ok = True
        except Exception:
            render_ok = False

    _log_runtime_diag(
        "video_done",
        video=os.path.basename(video_path),
        render_ok=render_ok,
        main_track_found=bool(main_track is not None),
        detected_frames=0 if main_track is None else len(main_track.get("frames", {})),
    )

    if main_track is None:
        rot_total = None
        rot_amp = None
        det_frames = 0
        avg_rot = None
        chosen_track_id = None
    else:
        rot_total = float(main_track.get("rot_total", 0.0))
        rot_amp = float(main_track.get("angle_max", 0.0) - main_track.get("angle_min", 0.0))
        det_frames = len(main_track.get("frames", {}))
        avg_rot = rot_total / max(1, det_frames)
        chosen_track_id = int(main_track.get("id", -1))

    from .utils import SCIPY_OK as _scipy_ok

    log_row = {
        "record_type": "VIDEO",
        "video_path": video_path,
        "output_video": save_vid_path,
        "patient_id": patient_id,
        "clinical_score_column": (
            score_column if scores_df is not None and not scores_df.empty else None
        ),
        "clinical_score": clinical_score,
        "clinical_score_raw": clinical_score_raw,
        "hand": hand,
        "fps": float(fps) if fps else None,
        "total_frames": int(total_frames),
        "main_track_found": bool(main_track is not None),
        "chosen_track_id": chosen_track_id,
        "rot_total_rad": rot_total,
        "rot_amp_rad": rot_amp,
        "detected_frames": int(det_frames),
        "avg_rot_rad_per_frame": avg_rot,
        "rot_avg_thresh": float(ROT_AVG_THRESH),
        "adjusted_appearance_pct": adj_app,
        "non_adjusted_appearance_pct": non_adj_app,
        "avg_conf": avg_conf,
        "conf_sum": conf_sum,
        "conf_count": int(conf_count),
        "detected_in_window": int(det_win),
        "window_len": int(win_len),
        "detected_total": int(det_total),
        "detected_before_fill": int(detected_before_fill),
        "fill_added_frames": int(fill_added),
        "keypoint_backend": gpu_cfg["keypoint_backend"],
        "superres_enabled": bool(gpu_cfg["use_superres"]),
        "rtmpose_enabled": bool(gpu_cfg["use_rtmpose"]),
        "openpose_enabled": bool(gpu_cfg["use_openpose"]),
        "superres_refined_frames": int(superres_refined),
        "rtmpose_refined_frames": int(rtmpose_refined),
        "openpose_refined_frames": int(openpose_refined),
        "render_ok": bool(render_ok),
        "scipy_ok": bool(_scipy_ok),
        "log_medication_state": medication_state,
        "log_hand": hand,
        "ids": ids,
        "visit": visit,
        "ps_trimmed": ps_trimmed,
        "ps_start_frame": ps_start_frame,
        "ps_end_frame": ps_end_frame,
        "ps_duration_s": ps_duration_s,
    }

    if metrics is not None:
        log_row.update(
            {
                "Mean Amplitude": float(metrics["avg_amp"]),
                "Amplitude CV": float(metrics["amp_cv"]),
                "Mean Frequency": float(metrics["freq"]),
                "Rhythm (CV %)": float(metrics["cv"]),
                "Norm Decrement Slope": float(metrics["norm_decrement_slope"]),
                "Amp Decrement Onset": float(metrics.get("amp_decrement_onset", float("nan"))),
                "Amp Decrement %": float(metrics.get("amp_decrement_pct", float("nan"))),
                "Norm TI Slope": float(metrics["norm_ti_slope"]),
                "Num Hesitations": int(metrics.get("num_hesitations", 0)),
                "Num Arrests": int(metrics["num_arrests"]),
                "Max Pause Duration (s)": float(metrics.get("max_pause_duration_s", float("nan"))),
                "Pause Time Ratio": float(metrics.get("pause_time_ratio", float("nan"))),
                "Peak Velocity": float(metrics.get("peak_velocity", float("nan"))),
                "Mean Velocity": float(metrics.get("mean_velocity", float("nan"))),
                "Peak Velocity CV": float(metrics.get("peak_velocity_cv", float("nan"))),
                "Mean Velocity CV": float(metrics.get("mean_velocity_cv", float("nan"))),
                "Norm Velocity Decrement Slope": float(
                    metrics.get("norm_velocity_decrement_slope", float("nan"))
                ),
                "Velocity Decrement Onset": float(
                    metrics.get("velocity_decrement_onset", float("nan"))
                ),
                "Velocity Decrement %": float(metrics.get("velocity_decrement_pct", float("nan"))),
                "Global Velocity": float(metrics.get("global_velocity", float("nan"))),
                "Arm Swing Index": (
                    float(metrics["arm_swing_index"])
                    if metrics.get("arm_swing_index") is not None
                    else float("nan")
                ),
                "Sample Entropy": float(metrics.get("sample_entropy", float("nan"))),
                "Amp-Vel Coupling": float(metrics.get("amp_vel_coupling", float("nan"))),
                "Hilbert Amplitude": float(metrics.get("hilbert_amplitude", float("nan"))),
                "Integral Amplitude": float(metrics.get("integral_amplitude", float("nan"))),
                "Inter-MCP Span": (
                    float(metrics["inter_mcp_span"])
                    if metrics.get("inter_mcp_span") is not None
                    else None
                ),
                "Total Cycles": int(metrics["total_cycles"]),
                "Quality Cycles": int(metrics["quality_cycles"]),
                "Signal Quality": float(signal_quality),
            }
        )
        for sq_key, sq_val in sq_sub_scores.items():
            log_row[f"SQ_{sq_key}"] = float(sq_val)
    else:
        log_row.update(
            {
                "Mean Amplitude": None,
                "Amplitude CV": None,
                "Mean Frequency": None,
                "Rhythm (CV %)": None,
                "Norm Decrement Slope": None,
                "Amp Decrement Onset": None,
                "Amp Decrement %": None,
                "Norm TI Slope": None,
                "Num Hesitations": None,
                "Num Arrests": None,
                "Max Pause Duration (s)": None,
                "Pause Time Ratio": None,
                "Peak Velocity": None,
                "Mean Velocity": None,
                "Peak Velocity CV": None,
                "Mean Velocity CV": None,
                "Norm Velocity Decrement Slope": None,
                "Velocity Decrement Onset": None,
                "Velocity Decrement %": None,
                "Global Velocity": None,
                "Arm Swing Index": None,
                "Sample Entropy": None,
                "Amp-Vel Coupling": None,
                "Hilbert Amplitude": None,
                "Integral Amplitude": None,
                "Inter-MCP Span": None,
                "Total Cycles": 0,
                "Quality Cycles": 0,
                "Signal Quality": 0.0,
            }
        )

    log_row.update(
        {
            "raw_rotation_series": series_to_json(raw_deg_full),
            "filtered_rotation_series": series_to_json(filtered_deg_full),
            "conf_index_mcp_series": series_to_json(conf_idx_series, decimals=4),
            "conf_middle_mcp_series": series_to_json(conf_mid_series, decimals=4),
            "conf_ring_mcp_series": series_to_json(conf_ring_series, decimals=4),
            "conf_pinky_mcp_series": series_to_json(conf_pinky_series, decimals=4),
            "conf_mcp_min_series": series_to_json(conf_min_series, decimals=4),
            "conf_mcp_used_mask_series": series_to_json(conf_used_mask_series, decimals=0),
            "conf_mcp_low_threshold": float(conf_low_thresh),
        }
    )
    if metrics is not None:
        log_row.update(
            {
                "cycle_peak_times_s": series_to_json(
                    metrics.get("detected_peak_times", metrics.get("peak_times", [])), decimals=4
                ),
                "cycle_trough_times_s": series_to_json(metrics.get("trough_times", []), decimals=4),
                "cycle_amplitudes_deg": series_to_json(metrics["amplitudes"], decimals=4),
                "cycle_trendline_deg": series_to_json(metrics["trend_line"], decimals=4),
            }
        )

    # ── Video quality factor columns ──────────────────────────────────
    log_row.update(
        {
            "VQ_video_width_px": vq_metrics.get("video_width_px"),
            "VQ_video_height_px": vq_metrics.get("video_height_px"),
            "VQ_inter_mcp_span_px": _compute_inter_mcp_span_px(
                _frames_dict,
                total_frames,
                vq_metrics.get("video_width_px"),
                vq_metrics.get("video_height_px"),
            ),
            "VQ_hand_bbox_area_median_px": vq_metrics.get("hand_bbox_area_median_px", float("nan")),
            "VQ_hand_bbox_area_q25_px": vq_metrics.get("hand_bbox_area_q25_px", float("nan")),
            "VQ_sharpness_median": vq_metrics.get("sharpness_median", float("nan")),
            "VQ_sharpness_q10": vq_metrics.get("sharpness_q10", float("nan")),
            "VQ_sharpness_q25": vq_metrics.get("sharpness_q25", float("nan")),
            "VQ_luminance_median": vq_metrics.get("luminance_median", float("nan")),
            "VQ_luminance_mean": vq_metrics.get("luminance_mean", float("nan")),
            "VQ_luminance_cv": vq_metrics.get("luminance_cv", float("nan")),
            "VQ_luminance_uniformity_median": vq_metrics.get(
                "luminance_uniformity_median", float("nan")
            ),
            "VQ_saturation_frac_median": vq_metrics.get("saturation_frac_median", float("nan")),
            "VQ_detection_rate": vq_metrics.get("detection_rate", float("nan")),
            "VQ_n_gaps": vq_metrics.get("n_gaps", 0),
            "VQ_longest_gap_frames": vq_metrics.get("longest_gap_frames", 0),
            "VQ_longest_gap_s": vq_metrics.get("longest_gap_s", float("nan")),
            "VQ_mean_gap_frames": vq_metrics.get("mean_gap_frames", float("nan")),
            "VQ_total_gap_frames": vq_metrics.get("total_gap_frames", 0),
            "VQ_gap_fraction": vq_metrics.get("gap_fraction", float("nan")),
        }
    )

    return log_row
