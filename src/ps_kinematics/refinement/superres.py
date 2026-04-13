"""
ps_kinematics.superres — GPU-accelerated hand-ROI super-resolution.

Uses Real-ESRGAN to upscale the hand region before landmark inference,
improving MCP keypoint resolution on small or distant hands.

Only loaded when ``USE_SUPERRES = True`` and a CUDA GPU is available.
Falls back gracefully (returns the original frame) when Real-ESRGAN is
not installed or GPU is unavailable.
"""

import os

import numpy as np


def _lm_vis(lm, default=1.0):
    """Extract visibility from a MediaPipe landmark, defaulting when absent."""
    v = getattr(lm, "visibility", None)
    return float(v) if v is not None else default


try:
    import cv2

    CV2_OK = True
except ImportError:
    CV2_OK = False

# ── Deferred torch / CUDA availability check ─────────────────────────
# Importing torch at module level in every ProcessPoolExecutor worker
# eagerly creates a CUDA context (~300-500 MB GPU RAM each).  With 25
# workers that alone can OOM the GPU.  Instead we check lazily, only
# when GPU work is actually requested (inside the GPU semaphore).
_torch_checked = False
TORCH_OK = False
GPU_AVAILABLE = False


def _ensure_torch():
    """Lazily check torch availability (called once per process)."""
    global _torch_checked, TORCH_OK, GPU_AVAILABLE
    if _torch_checked:
        return
    _torch_checked = True
    try:
        import torch

        TORCH_OK = True
        GPU_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        TORCH_OK = False
        GPU_AVAILABLE = False


# Lazy-initialised Real-ESRGAN upsampler (one per process)
_upsampler = None
_upsampler_scale = None


def _get_upsampler(scale=2, model_name="realesr-general-x4v3", model_path=None, half=True):
    """Lazily create and cache a RealESRGAN upsampler instance.

    Parameters
    ----------
    scale : int
        Output upscale factor (2 or 4).
    model_name : str
        Model variant name passed to ``RealESRGANer``.
    half : bool
        Use FP16 inference (faster on Ampere+ GPUs).

    Returns
    -------
    RealESRGANer or None
        The upsampler, or None if dependencies are missing or no GPU.
    """
    global _upsampler, _upsampler_scale

    _ensure_torch()
    if not GPU_AVAILABLE:
        return None

    if _upsampler is not None and _upsampler_scale == scale:
        return _upsampler

    if model_path is None:
        model_path = os.environ.get("SUPERRES_MODEL_PATH")

    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # Select architecture by model name
        if "x4v3" in model_name:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
            netscale = 4
        elif "x2plus" in model_name:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
            )
            netscale = 2
        else:
            # Default: x4v3
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
            netscale = 4

        _upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,  # use local weights when provided, else library default behaviour
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half and torch.cuda.is_available(),
            gpu_id=0,
        )
        _upsampler_scale = scale
        return _upsampler
    except Exception:
        _upsampler = None
        return None


def superres_upscale_roi(
    frame_bgr,
    hand_bbox_norm,
    scale=2,
    model_name="realesr-general-x4v3",
    model_path=None,
    half=True,
):
    """Upscale a hand ROI using Real-ESRGAN GPU inference.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Full video frame in BGR format, shape (H, W, 3).
    hand_bbox_norm : tuple of (x1, y1, x2, y2)
        Normalised [0,1] bounding box of the hand region.
    scale : int
        Target upscale factor (2 or 4).
    model_name : str
        Real-ESRGAN model variant.
    half : bool
        FP16 inference.

    Returns
    -------
    tuple of (upscaled_roi_bgr, roi_pixel_bounds)
        upscaled_roi_bgr : np.ndarray, shape (roi_h*scale, roi_w*scale, 3)
        roi_pixel_bounds : (px1, py1, px2, py2) in original frame pixels
        Returns (None, None) if upscaling fails.
    """
    _ensure_torch()
    if not CV2_OK or frame_bgr is None:
        return None, None

    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = hand_bbox_norm

    px1 = max(0, int(x1 * w_img))
    py1 = max(0, int(y1 * h_img))
    px2 = min(w_img, int(x2 * w_img))
    py2 = min(h_img, int(y2 * h_img))

    if px2 - px1 < 10 or py2 - py1 < 10:
        return None, None

    roi = frame_bgr[py1:py2, px1:px2].copy()
    if roi.size == 0:
        return None, None

    upsampler = _get_upsampler(scale=scale, model_name=model_name, model_path=model_path, half=half)
    if upsampler is None:
        # Fallback: bicubic upscale (CPU, still helpful)
        upscaled = cv2.resize(
            roi, (roi.shape[1] * scale, roi.shape[0] * scale), interpolation=cv2.INTER_CUBIC
        )
        return upscaled, (px1, py1, px2, py2)

    try:
        upscaled, _ = upsampler.enhance(roi, outscale=scale)
        return upscaled, (px1, py1, px2, py2)
    except Exception:
        # Fallback on any error
        upscaled = cv2.resize(
            roi, (roi.shape[1] * scale, roi.shape[0] * scale), interpolation=cv2.INTER_CUBIC
        )
        return upscaled, (px1, py1, px2, py2)


def compute_hand_bbox_from_track(track, frame_idx, padding=0.25):
    """Compute a normalised hand bounding box from the nearest tracked frame.

    Parameters
    ----------
    track : dict
        Track dictionary with ``"frames"`` mapping.
    frame_idx : int
        Current frame index.
    padding : float
        Normalised padding around the landmark extent.

    Returns
    -------
    tuple of (x1, y1, x2, y2) or None
        Normalised bounding box, or None if no reference is available.
    """
    if track is None:
        return None
    frames = track.get("frames", {})
    if not frames:
        return None

    # Find nearest frame with landmarks
    nearest_f = min(frames.keys(), key=lambda f: abs(f - frame_idx))
    lm_arr = frames[nearest_f]

    lm_x = lm_arr[:, 0]
    lm_y = lm_arr[:, 1]
    x1 = max(0.0, float(np.min(lm_x)) - padding)
    y1 = max(0.0, float(np.min(lm_y)) - padding)
    x2 = min(1.0, float(np.max(lm_x)) + padding)
    y2 = min(1.0, float(np.max(lm_y)) + padding)

    if x2 - x1 < 0.05 or y2 - y1 < 0.05:
        return None

    return (x1, y1, x2, y2)


def superres_refine_landmarks(
    frame_bgr,
    track,
    frame_idx,
    landmarker_img,
    hand_to_track=None,
    scale=2,
    model_name="realesr-general-x4v3",
    model_path=None,
    half=True,
    padding=0.25,
):
    """Run super-resolution on the hand ROI + re-detect landmarks.

    Upscales the hand ROI, runs MediaPipe IMAGE-mode detection on it,
    and maps landmarks back to original normalised coordinates.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Full frame in BGR.
    track : dict
        Current track with ``"frames"`` mapping.
    frame_idx : int
        Frame index to refine.
    landmarker_img : MediaPipe HandLandmarker (IMAGE mode)
        Landmarker instance for detection.
    hand_to_track : str or None
        Expected handedness label ("Left" / "Right").
    scale : int
        Super-resolution upscale factor.
    model_name : str
        Real-ESRGAN model name.
    half : bool
        FP16 inference.
    padding : float
        Normalised padding for hand bbox.

    Returns
    -------
    np.ndarray or None
        Refined (21, 4) landmark array in original normalised coords, or None.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None

    bbox = compute_hand_bbox_from_track(track, frame_idx, padding=padding)
    if bbox is None:
        return None

    upscaled, pixel_bounds = superres_upscale_roi(
        frame_bgr, bbox, scale=scale, model_name=model_name, model_path=model_path, half=half
    )
    if upscaled is None:
        return None

    # Run MediaPipe on the upscaled ROI
    upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=upscaled_rgb)

    try:
        result = landmarker_img.detect(mp_image)
    except Exception:
        return None

    if not result.hand_landmarks:
        return None

    # Select the best detection (prefer matching handedness)
    best_lm_list = None
    if hand_to_track is not None and result.handedness:
        for hi, hcat in enumerate(result.handedness):
            if hi < len(result.hand_landmarks) and hcat:
                if hcat[0].category_name == hand_to_track:
                    best_lm_list = result.hand_landmarks[hi]
                    break
    if best_lm_list is None:
        best_lm_list = result.hand_landmarks[0]

    # Map landmarks from upscaled-ROI coords back to original normalised coords
    x1, y1, x2, y2 = bbox
    roi_w = x2 - x1
    roi_h = y2 - y1

    lm_arr = np.array(
        [[x1 + lm.x * roi_w, y1 + lm.y * roi_h, float(lm.z), _lm_vis(lm)] for lm in best_lm_list],
        dtype=np.float32,
    )
    return lm_arr


def cleanup_superres():
    """Free the cached Real-ESRGAN upsampler from GPU memory.

    Called by ``gpu_manager.cleanup_gpu()`` after the GPU semaphore is
    about to be released, so the next worker gets a clean GPU.
    """
    global _upsampler, _upsampler_scale
    if _upsampler is not None:
        try:
            del _upsampler
        except Exception:
            pass
        _upsampler = None
        _upsampler_scale = None
