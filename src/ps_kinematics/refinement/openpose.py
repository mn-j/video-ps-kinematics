"""
ps_kinematics.openpose — OpenPose-Hand landmark refinement.

Provides a pre-trained OpenPose hand-keypoint inferencer (Caffe model)
that refines MediaPipe landmarks on each frame's hand ROI.

Integration strategy (hybrid):
  1. MediaPipe VIDEO mode runs first → hand detection/tracking/handedness.
  2. OpenPose refines hand landmark (x, y) coordinates on each tracked ROI,
     overwriting MediaPipe coordinates.

This keeps the existing tracker/fill-pass logic untouched.

Model files required (download once):
  - pose_deploy.prototxt
  - pose_iter_102000.caffemodel
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2

    CV2_OK = True
except ImportError:
    cv2 = None
    CV2_OK = False

_openpose_net = None
_openpose_proto = None
_openpose_weights = None
_warned_missing_files = False


OPENPOSE_TO_MP_INDEX = list(range(21))


def _as_abs(path_value):
    if path_value is None:
        return None
    return os.path.abspath(str(path_value))


def _get_openpose_net(proto_path, weights_path, use_cuda=False):
    """Lazily create and cache OpenPose Caffe network.

    Returns
    -------
    cv2.dnn_Net or None
        Loaded OpenPose net, or None when OpenCV DNN/model files are unavailable.
    """
    global _openpose_net, _openpose_proto, _openpose_weights, _warned_missing_files

    if not CV2_OK:
        return None

    proto = _as_abs(proto_path)
    weights = _as_abs(weights_path)

    # If this process has already loaded the exact same network, reuse it
    # directly. This avoids repeated filesystem checks on every frame call
    # (which can be noisy/flaky on network filesystems).
    if _openpose_net is not None and _openpose_proto == proto and _openpose_weights == weights:
        return _openpose_net

    if not proto or not weights:
        if not _warned_missing_files:
            logger.warning(
                "Missing model paths. Set OPENPOSE_PROTO_PATH and OPENPOSE_WEIGHTS_PATH."
            )
            _warned_missing_files = True
        return None

    if not os.path.exists(proto) or not os.path.exists(weights):
        if not _warned_missing_files:
            logger.warning(
                "Model file(s) not found. Expected:\n" "  prototxt: %s\n" "  caffemodel: %s",
                proto,
                weights,
            )
            _warned_missing_files = True
        return None

    try:
        net = cv2.dnn.readNetFromCaffe(proto, weights)
        if use_cuda:
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                pass
        _openpose_net = net
        _openpose_proto = proto
        _openpose_weights = weights
        return _openpose_net
    except Exception as exc:
        logger.error("Failed to load network: %s", exc)
        _openpose_net = None
        _openpose_proto = None
        _openpose_weights = None
        return None


def _hand_bbox_from_landmarks(lm_arr, img_w, img_h, padding=0.30):
    """Compute a pixel-space hand bbox from normalised landmarks."""
    xs = lm_arr[:, 0] * img_w
    ys = lm_arr[:, 1] * img_h

    x1 = float(np.min(xs))
    y1 = float(np.min(ys))
    x2 = float(np.max(xs))
    y2 = float(np.max(ys))

    w = x2 - x1
    h = y2 - y1
    pad_x = w * float(padding)
    pad_y = h * float(padding)

    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(img_w), x2 + pad_x)
    y2 = min(float(img_h), y2 + pad_y)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return int(x1), int(y1), int(x2), int(y2)


def refine_landmarks_openpose(
    frame_bgr,
    lm_arr_mp,
    proto_path=None,
    weights_path=None,
    padding=0.30,
    conf_thresh=0.10,
    input_size=368,
    use_cuda=False,
):
    """Refine hand landmarks with OpenPose.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Full frame in BGR format.
    lm_arr_mp : np.ndarray
        MediaPipe landmark array shape (21, 4): [x_norm, y_norm, z, visibility].
    proto_path : str
        OpenPose hand prototxt path.
    weights_path : str
        OpenPose hand caffemodel path.
    padding : float
        ROI padding around MediaPipe hand bbox.
    conf_thresh : float
        Minimum OpenPose heatmap confidence required to overwrite a keypoint.
    input_size : int
        OpenPose network input size for the hand ROI.

    Returns
    -------
    np.ndarray or None
        Refined landmark array (shape (21, 4)), or None on failure.
    """
    if not CV2_OK or frame_bgr is None or lm_arr_mp is None:
        return None

    net = _get_openpose_net(proto_path, weights_path, use_cuda=use_cuda)
    if net is None:
        return None

    h_img, w_img = frame_bgr.shape[:2]
    bbox = _hand_bbox_from_landmarks(lm_arr_mp, w_img, h_img, padding=padding)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    try:
        in_size = int(max(64, input_size))
        blob = cv2.dnn.blobFromImage(
            roi,
            scalefactor=1.0 / 255.0,
            size=(in_size, in_size),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )
        net.setInput(blob)
        out = net.forward()
    except Exception:
        return None

    if out is None or out.ndim != 4 or out.shape[1] < 21:
        return None

    out_h = out.shape[2]
    out_w = out.shape[3]
    roi_h, roi_w = roi.shape[:2]

    refined = lm_arr_mp.copy()
    for i in range(21):
        heatmap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        if float(conf) < float(conf_thresh):
            continue

        px = (float(point[0]) / max(1, out_w - 1)) * max(1, roi_w - 1)
        py = (float(point[1]) / max(1, out_h - 1)) * max(1, roi_h - 1)

        full_x = x1 + px
        full_y = y1 + py

        mp_idx = OPENPOSE_TO_MP_INDEX[i]
        refined[mp_idx, 0] = float(full_x) / max(1.0, float(w_img))
        refined[mp_idx, 1] = float(full_y) / max(1.0, float(h_img))
        refined[mp_idx, 3] = float(conf)

    return refined


def cleanup_openpose():
    """Release cached OpenPose network."""
    global _openpose_net, _openpose_proto, _openpose_weights
    _openpose_net = None
    _openpose_proto = None
    _openpose_weights = None
