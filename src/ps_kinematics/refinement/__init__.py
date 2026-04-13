"""Pluggable hand-landmark refinement backends.

Each submodule wraps a third-party vision library that can optionally refine
MediaPipe hand landmarks (or replace MediaPipe entirely, in the case of
``yolo`` when USE_YOLO_ONLY is set):

    yolo      — Ultralytics YOLO-Pose hand keypoint refinement
    rtmpose   — MMPose RTMPose-Hand whole-body hand keypoints
    openpose  — OpenPose-Hand OpenCV DNN backend
    superres  — Real-ESRGAN super-resolution preprocessing

All backends are disabled by default. Enable them via the USE_* flags in a
tuning profile (see configs/tuning_25fps.yaml).
"""

from . import openpose, rtmpose, superres, yolo

__all__ = ["openpose", "rtmpose", "superres", "yolo"]
