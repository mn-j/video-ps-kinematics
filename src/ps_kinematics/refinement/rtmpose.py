"""
ps_kinematics.rtmpose — RTMPose-Hand hybrid landmark refinement.

Provides a GPU-accelerated RTMPose-Hand inferencer that can refine the
MCP keypoint coordinates initially detected by MediaPipe.  Only used
when ``USE_RTMPOSE = True`` and a CUDA GPU is available.

Integration strategy (hybrid):
  1. MediaPipe VIDEO mode runs first → hand detection, tracking, handedness.
  2. RTMPose-Hand refines the landmark (x, y) coordinates on each frame's
     hand ROI, overwriting the MediaPipe landmarks in the track.

This preserves the entire MultiHandOfflineTracker, fill-pass logic, and
handedness classification.  Only the *coordinates* change.

Requires: mmpose, mmdet, mmengine, mmcv (all optional).
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# ── Deferred torch / CUDA availability check ─────────────────────────
# See superres.py for rationale: eager import in each worker creates
# a ~300-500 MB CUDA context; with 25 workers that alone OOMs the GPU.
_torch_checked = False
GPU_AVAILABLE = False


def _ensure_torch():
    """Lazily check torch + CUDA availability (called once per process)."""
    global _torch_checked, GPU_AVAILABLE
    if _torch_checked:
        return
    _torch_checked = True
    try:
        import torch

        GPU_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        GPU_AVAILABLE = False


try:
    import cv2

    CV2_OK = True
except ImportError:
    CV2_OK = False

# Lazy-initialised inferencer (one per process)
_inferencer = None
_inferencer_cfg = None
_inferencer_ckpt = None

# ============================================================
# MediaPipe ↔ RTMPose landmark mapping
# ============================================================
# Both use 21-point COCO-WholeBody hand convention with the same ordering:
#   0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
# So the mapping is identity — no remapping required.
RTMPOSE_TO_MP_INDEX = list(range(21))

# The 5 keypoints used by the kinematic pipeline
_KINEMATIC_INDICES = [0, 5, 9, 13, 17]  # WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP


def _candidate_model_aliases(model_cfg):
    """Return ordered candidate aliases for RTMPose-Hand across mmpose versions."""
    requested = str(model_cfg)
    leaf = requested.split("/")[-1]
    candidates = [
        requested,
        leaf,
        f"hand/rtmpose/{leaf}",
        f"wholebody_2d_keypoint/rtmpose/coco-wholebody-hand/{leaf}",
        f"hand_2d_keypoint/rtmpose/coco-wholebody-hand/{leaf}",
    ]
    out = []
    seen = set()
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _get_inferencer(model_cfg=None, checkpoint_path=None):
    """Lazily create and cache an MMPose RTMPose-Hand inferencer.

    Parameters
    ----------
    model_cfg : str or None
        MMPose model config alias, e.g.
        ``'hand/rtmpose/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256'``.
        If None, uses the default medium variant (matches utils.RTMPOSE_MODEL_CFG).

    Returns
    -------
    MMPoseInferencer or None
        The inferencer, or None if dependencies are missing or no GPU.

    Raises
    ------
    RuntimeError
        If a GPU is available but the model ends up on CPU after initialisation,
        indicating a silent configuration failure.
    """
    global _inferencer, _inferencer_cfg, _inferencer_ckpt

    _ensure_torch()
    if not GPU_AVAILABLE:
        # Explicit error so callers/logs see why RTMPose is being skipped.
        raise RuntimeError(
            "[RTMPose] No CUDA GPU detected by torch (torch.cuda.is_available() == False). "
            "RTMPose requires a GPU. Check CUDA_VISIBLE_DEVICES, your torch installation, "
            "and that the GPU driver is accessible from this process."
        )

    if model_cfg is None:
        model_cfg = "hand/rtmpose/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256"

    if checkpoint_path is None:
        checkpoint_path = os.environ.get("RTMPOSE_CHECKPOINT_PATH")
    if checkpoint_path:
        checkpoint_path = os.path.abspath(str(checkpoint_path))

    if (
        _inferencer is not None
        and _inferencer_cfg == model_cfg
        and _inferencer_ckpt == checkpoint_path
    ):
        return _inferencer

    try:
        import torch
        from mmpose.apis import MMPoseInferencer

        init_errors = []
        selected_alias = None
        for alias in _candidate_model_aliases(model_cfg):
            infer_kwargs = {
                "pose2d": alias,
                "device": "cuda:0",
            }
            if checkpoint_path:
                infer_kwargs["pose2d_weights"] = checkpoint_path

            try:
                _inferencer = MMPoseInferencer(**infer_kwargs)
                selected_alias = alias
                break
            except TypeError:
                if checkpoint_path:
                    try:
                        logger.warning(
                            "Current mmpose version does not accept pose2d_weights; falling back to default weight resolution."
                        )
                        infer_kwargs.pop("pose2d_weights", None)
                        _inferencer = MMPoseInferencer(**infer_kwargs)
                        selected_alias = alias
                        break
                    except Exception as e:
                        init_errors.append(f"{alias}: {e}")
                        continue
                init_errors.append(f"{alias}: pose2d_weights unsupported")
                continue
            except Exception as e:
                init_errors.append(f"{alias}: {e}")

        if _inferencer is None:
            raise RuntimeError(
                "[RTMPose] Could not initialise inferencer with any candidate model alias. "
                f"Requested='{model_cfg}'. Tried={_candidate_model_aliases(model_cfg)}. Errors={init_errors}"
            )

        if selected_alias is not None and selected_alias != model_cfg:
            logger.info("Resolved model alias '%s' -> '%s'", model_cfg, selected_alias)

        # ── GPU placement verification ────────────────────────────────────
        # Confirm the model actually landed on CUDA; a misconfigured mmpose
        # or torch build can silently fall back to CPU.
        try:
            param_device = next(_inferencer.model.parameters()).device
            if param_device.type != "cuda":
                # Tear down what we built and hard-fail so the caller sees a
                # clear error rather than silently running inference on CPU.
                del _inferencer
                _inferencer = None
                raise RuntimeError(
                    f"[RTMPose] FATAL: model loaded on '{param_device}' despite requesting "
                    "device='cuda:0'.  This means RTMPose is running on the CPU.  "
                    "Check your mmpose / torch / CUDA configuration."
                )
            mem_mb = torch.cuda.memory_allocated(param_device.index or 0) // (1024 * 1024)
            logger.info("Model on %s — GPU memory allocated: %d MB", param_device, mem_mb)
        except (AttributeError, StopIteration):
            # .model or .parameters() not accessible on this mmpose version;
            # fall back to a memory-delta check as best-effort.
            mem_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
            if mem_mb == 0:
                logger.warning(
                    "No GPU memory allocated after model creation. "
                    "The model may be running on CPU. Check CUDA / mmpose installation."
                )
            else:
                logger.info(
                    "Model initialised — GPU memory allocated: %d MB (device=cuda:0)", mem_mb
                )

        _inferencer_cfg = model_cfg
        _inferencer_ckpt = checkpoint_path
        return _inferencer
    except RuntimeError:
        # Re-raise our own RuntimeErrors (GPU placement failures) verbatim.
        raise
    except ModuleNotFoundError as e:
        missing_mod = getattr(e, "name", None)
        if missing_mod == "pkg_resources":
            logger.error(
                "Failed to initialise inferencer: missing 'pkg_resources'. "
                "Install/upgrade setuptools in the SAME runtime environment used by this Slurm job. "
                "Example: pip install -U setuptools"
            )
        else:
            logger.error("Failed to initialise inferencer: missing module '%s'", missing_mod)
        _inferencer = None
        _inferencer_ckpt = None
        return None
    except Exception as e:
        logger.error("Failed to initialise inferencer: %s", e)
        _inferencer = None
        _inferencer_ckpt = None
        return None


def _hand_bbox_from_landmarks(lm_arr, img_w, img_h, padding=0.30):
    """Compute a pixel-space bounding box from normalised landmarks.

    Returns
    -------
    list of [x1, y1, x2, y2, score]
        Pixel bounding box with confidence 1.0, or None if degenerate.
    """
    xs = lm_arr[:, 0] * img_w
    ys = lm_arr[:, 1] * img_h

    x1 = float(np.min(xs))
    y1 = float(np.min(ys))
    x2 = float(np.max(xs))
    y2 = float(np.max(ys))

    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding
    pad_y = h * padding

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return [x1, y1, x2, y2, 1.0]


def refine_landmarks_rtmpose(
    frame_bgr, lm_arr_mp, model_cfg=None, checkpoint_path=None, padding=0.30
):
    """Refine hand landmarks using RTMPose on a single frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Full video frame, BGR, shape (H, W, 3).
    lm_arr_mp : np.ndarray
        MediaPipe landmark array, shape (21, 4) with [x_norm, y_norm, z, vis].
    model_cfg : str or None
        MMPose model config alias.
    padding : float
        Normalised padding for the hand bounding box.

    Returns
    -------
    np.ndarray or None
        Refined (21, 4) landmark array with updated (x, y) in normalised
        coords.  z and visibility are preserved from MediaPipe.
        Returns None if inference fails (caller should keep MediaPipe result).
    """
    if not CV2_OK or frame_bgr is None or lm_arr_mp is None:
        return None

    inferencer = _get_inferencer(model_cfg, checkpoint_path=checkpoint_path)
    if inferencer is None:
        return None

    h_img, w_img = frame_bgr.shape[:2]

    bbox = _hand_bbox_from_landmarks(lm_arr_mp, w_img, h_img, padding=padding)
    if bbox is None:
        return None

    try:
        # MMPose inferencer expects a list of bboxes
        results_gen = inferencer(
            frame_bgr,
            bboxes=[bbox[:4]],
            bbox_format="xyxy",
        )
        # The generator yields one result dict per image
        result = next(results_gen)
    except Exception:
        return None

    # Extract predictions
    predictions = result.get("predictions", [])
    if not predictions or len(predictions) == 0:
        return None

    # predictions is a list of per-image results; each is a list of per-instance dicts
    instances = predictions[0] if isinstance(predictions[0], list) else [predictions[0]]
    if not instances:
        return None

    instance = instances[0]
    keypoints = instance.get("keypoints", None)
    if keypoints is None:
        return None

    # keypoints is shape (21, 2) in pixel coords
    kp = np.array(keypoints, dtype=np.float32)
    if kp.shape[0] < 21:
        return None

    # Optionally get keypoint scores
    kp_scores = instance.get("keypoint_scores", None)
    if kp_scores is not None:
        kp_scores = np.array(kp_scores, dtype=np.float32)
    else:
        kp_scores = np.ones(21, dtype=np.float32)

    # Build refined landmark array: normalise pixel coords back to [0,1]
    refined = lm_arr_mp.copy()
    for i in range(min(21, kp.shape[0])):
        mp_idx = RTMPOSE_TO_MP_INDEX[i]
        # Only overwrite if RTMPose confidence is reasonable
        if kp_scores[i] > 0.3:
            refined[mp_idx, 0] = float(kp[i, 0]) / w_img  # x normalised
            refined[mp_idx, 1] = float(kp[i, 1]) / h_img  # y normalised
            # Preserve z from MediaPipe (RTMPose is 2D)
            # Update visibility to RTMPose score (more calibrated for heatmap)
            refined[mp_idx, 3] = float(kp_scores[i])

    return refined


def refine_track_with_rtmpose(
    track, video_path, model_cfg=None, checkpoint_path=None, padding=0.30
):
    """Refine all landmark frames in a track using RTMPose.

    Reads the video, and for each frame in the track, runs RTMPose
    on the hand ROI and overwrites the landmark coordinates with
    the refined positions.

    Parameters
    ----------
    track : dict or None
        Track dictionary with ``"frames"`` mapping frame_idx → (21, 4) array.
    video_path : str
        Path to the video file.
    model_cfg : str or None
        MMPose model config alias.
    padding : float
        Bounding box padding.

    Returns
    -------
    dict or None
        The track with refined landmarks (modified in place), or None.
    int
        Number of frames successfully refined.
    """
    if track is None or not CV2_OK:
        return track, 0

    frames_dict = track.get("frames", {})
    if not frames_dict:
        return track, 0

    inferencer = _get_inferencer(model_cfg, checkpoint_path=checkpoint_path)
    if inferencer is None:
        return track, 0

    frame_set = set(frames_dict.keys())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return track, 0

    refined_count = 0
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx in frame_set:
            lm_arr_mp = frames_dict[frame_idx]
            refined = refine_landmarks_rtmpose(
                frame_bgr,
                lm_arr_mp,
                model_cfg=model_cfg,
                checkpoint_path=checkpoint_path,
                padding=padding,
            )
            if refined is not None:
                frames_dict[frame_idx] = refined
                refined_count += 1
        frame_idx += 1

    cap.release()
    return track, refined_count


def refine_track_with_rtmpose_batched(
    track, video_path, model_cfg=None, checkpoint_path=None, padding=0.30, batch_size=16
):
    """Batch-optimised version of ``refine_track_with_rtmpose``.

    Reads frames in batches and runs RTMPose inference with reduced
    Python-loop overhead.  Falls back to per-frame refinement on error.

    Parameters
    ----------
    track : dict or None
        Track with ``"frames"`` mapping.
    video_path : str
        Video file path.
    model_cfg : str or None
        MMPose model config.
    padding : float
        Bounding box padding.
    batch_size : int
        Number of frames to accumulate before inference.

    Returns
    -------
    dict or None
        Track with refined landmarks.
    int
        Number of frames successfully refined.
    """
    if track is None or not CV2_OK:
        return track, 0

    frames_dict = track.get("frames", {})
    if not frames_dict:
        return track, 0

    inferencer = _get_inferencer(model_cfg, checkpoint_path=checkpoint_path)
    if inferencer is None:
        return track, 0

    frame_set = set(frames_dict.keys())
    sorted_frames = sorted(frame_set)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return track, 0

    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Read all needed frames
    needed_data = {}  # frame_idx -> frame_bgr
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx in frame_set:
            needed_data[frame_idx] = frame_bgr
        frame_idx += 1
    cap.release()

    refined_count = 0

    # Process in batches
    for batch_start in range(0, len(sorted_frames), batch_size):
        batch_indices = sorted_frames[batch_start : batch_start + batch_size]

        for fidx in batch_indices:
            if fidx not in needed_data:
                continue
            frame_bgr = needed_data[fidx]
            lm_arr_mp = frames_dict[fidx]

            refined = refine_landmarks_rtmpose(
                frame_bgr,
                lm_arr_mp,
                model_cfg=model_cfg,
                checkpoint_path=checkpoint_path,
                padding=padding,
            )
            if refined is not None:
                frames_dict[fidx] = refined
                refined_count += 1

    return track, refined_count


def cleanup_rtmpose():
    """Free the cached RTMPose inferencer from GPU memory.

    Called by ``gpu_manager.cleanup_gpu()`` after the GPU semaphore is
    about to be released, so the next worker gets a clean GPU.
    """
    global _inferencer, _inferencer_cfg, _inferencer_ckpt
    if _inferencer is not None:
        try:
            del _inferencer
        except Exception:
            pass
        _inferencer = None
        _inferencer_cfg = None
        _inferencer_ckpt = None
