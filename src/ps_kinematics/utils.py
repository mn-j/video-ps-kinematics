"""
ps_kinematics.utils — Constants, tuning knobs, and small helper functions.

"""

import json
import logging
import warnings
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

# ============================
# Reference frame rate
# ============================
# All frame-count-based constants below are calibrated for BASE_FPS = 25.0.
# At runtime the pipeline reads the actual fps from each video and scales
# frame-count values proportionally (e.g. max_gap * actual_fps / BASE_FPS),
# so that the thresholds remain consistent in wall-clock time regardless of
# whether the input is the original 25 fps footage or 50 fps interpolated video.
BASE_FPS = 25.0

# Maximum processing frame rate.  When a video's native fps exceeds this
# value, the tracking loop processes only every Nth frame (stride =
# floor(native_fps / TARGET_PROCESSING_FPS)) so that the effective rate
# stays near the target.  Set to 0 or None to disable (process every frame).
# Default 0 = disabled, preserving existing behaviour for 25 fps datasets.
TARGET_PROCESSING_FPS: float = 0

# Intra-video parallelism: number of sub-processes to split a single
# video's MediaPipe inference across.  Default 1 = serial (original
# behaviour).  See PipelineConfig.INTRA_VIDEO_WORKERS for details.
INTRA_VIDEO_WORKERS: int = 1

# ============================
# TUNING KNOBS (tracking)
# ============================
TRACK_MATCH_THRESH = 0.08  # wrist match threshold (normalized)
MAX_GAP = 3  # max missing frames before a track becomes stale   [@ BASE_FPS]
MAX_JUMP_PER_FRAME = 0.04  # max allowed wrist movement per frame (normalized) [@ BASE_FPS]
FILL_MAX_DIST = 0.12  # max allowed distance to fill a missing frame (normalized)
HANDEDNESS_PENALTY_MULT = 1.25  # penalty multiplier if handedness mismatches preference
ROT_AVG_THRESH = 0.001  # rad/frame; minimum avg rotation for track eligibility [@ BASE_FPS]
LANDMARK_SMOOTH_WINDOW = 3  # temporal smoothing window (frames) for keypoints  [@ BASE_FPS]

# ============================
# Keypoint reliability & fill quality
# ============================
FILL_ITERATIONS = 2  # max IMAGE-mode fill passes; 1 = original single-pass behaviour
USE_CLAHE_ON_FILL = True  # CLAHE contrast enhancement before every IMAGE-mode detection frame
ROI_REDETECT_PADDING = 0.25  # normalised padding around hand bbox for ROI-zoom re-detection
VISIBILITY_THRESHOLD = (
    0.35  # per-landmark visibility below this → frame treated as angle-unreliable
)
CYCLE_NAN_THRESHOLD = 0.40  # if >40% of frames in a cycle were interpolated, exclude its amplitude
USE_PCA_ANGLE = True  # use per-video PCA on 2D (x,y) knuckle-line vectors.
#   3D knuckle vectors were tested but cause volatile amplitude estimates:
#   for frontal recordings the PS signal (σ1) and z-noise (σ2) have
#   similar magnitude, making the SVD eigenvectors ill-conditioned.
#   Restricting to x,y (directly observed) keeps σ1 >> σ2 reliably,
#   giving stable amplitudes while preserving view-invariance in the
#   image plane via PCA centering.
#   Disable to fall back to frame-by-frame knuckle-line angle unwrap.
LM_OUTLIER_WINDOW = (
    7  # rolling-median window (frames) for per-landmark outlier correction [@ BASE_FPS]
)
LM_OUTLIER_THRESH = (
    0.06  # normalised distance; landmarks beyond rolling median by this are replaced
)

# ============================
# Velocity feature computation
# ============================
PEAK_VELOCITY_PERCENTILE = 95  # percentile used for per-cycle peak velocity (CMS):
#   Zarrat Ehsan et al. (2024) use p95 to be robust to
#   momentary keypoint noise; set to 100 for true maximum.

# ============================
# Pre-filter angle detrending
# ============================
DETREND_POLY_ORDER = 3  # polynomial order for pre-filter drift removal:
#   3 (cubic) handles both linear ramp (accumulated
#   unwrap errors) and V/U/S-shaped drift.  The polynomial
#   fits only the slow trend — the faster PS oscillation
#   (0.25–3.5 Hz) is orthogonal and preserved.  Set to 0
#   to disable detrending entirely.

# ============================
# Wrist-Z cycle-detection confirmation
# ============================
USE_WRIST_Z_CONFIRMATION = True  # use wrist z oscillation to validate detected peaks/valleys
WRIST_Z_CONFIRM_WINDOW = (
    5  # +-frames around angle extremum to look for a wrist z extremum [@ BASE_FPS]
)
WRIST_Z_PROM_BOOST = 1.5  # unconfirmed peaks must exceed effective_prom * this factor
WRIST_Z_MIN_VALID_FRAC = 0.50  # skip wrist z validation if <50% of frames have valid z
WRIST_Z_MIN_SNR = 1.5  # skip wrist z validation if signal range / noise < this

# ============================
# Adaptive visibility threshold
# ============================
ADAPTIVE_VISIBILITY = True  # if True, per-video adaptive visibility threshold replaces fixed
ADAPTIVE_VIS_PERCENTILE = 20  # percentile of per-frame min-MCP-visibility used as lower bound
ADAPTIVE_VIS_FLOOR = 0.15  # hard minimum — never accept visibility below this

# ============================
# Super-resolution preprocessing (GPU only)
# ============================
USE_SUPERRES = False  # True → upscale hand ROI via Real-ESRGAN before landmark inference
SUPERRES_SCALE = 4  # upscale factor (2 or 4); 2 is fastest with good gains
SUPERRES_MODEL_NAME = "realesr-general-x4v3"  # Real-ESRGAN model variant
SUPERRES_MODEL_PATH = (
    None  # optional local .pth path; set via TUNING_OVERRIDES to avoid per-worker auto-download
)
SUPERRES_HALF = True  # use FP16 inference (faster on modern GPUs)

# ============================
# RTMPose-Hand hybrid refinement (GPU only)
# ============================
USE_RTMPOSE = False  # True → refine MCP landmarks via RTMPose-Hand after MediaPipe tracking
RTMPOSE_MODEL_CFG = "rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256"  # MMPose model config
RTMPOSE_CHECKPOINT_PATH = (
    None  # optional local RTMPose checkpoint path; set via TUNING_OVERRIDES to avoid auto-download
)
RTMPOSE_BBOX_PADDING = 0.30  # normalised padding around wrist-based hand bbox for RTMPose input

# ============================
# OpenPose-Hand hybrid refinement (OpenCV DNN), not currently working
# ============================
USE_OPENPOSE = False  # True → refine landmarks via OpenPose instead of RTMPose
OPENPOSE_PROTO_PATH = "models/openpose/hand/pose_deploy.prototxt"
OPENPOSE_WEIGHTS_PATH = "models/openpose/hand/pose_iter_102000.caffemodel"
OPENPOSE_BBOX_PADDING = 0.30  # normalised padding around hand bbox for OpenPose input
OPENPOSE_CONF_THRESHOLD = 0.10  # minimum heatmap confidence to accept an OpenPose keypoint
OPENPOSE_INPUT_SIZE = 368  # hand network input resolution (square)
OPENPOSE_USE_CUDA = False  # requires OpenCV built with CUDA DNN backend

# ============================
# YOLO-Pose hand keypoint refinement (GPU)
# ============================
USE_YOLO_HAND = False  # True → refine landmarks via YOLO-Pose hand model
USE_YOLO_ONLY = False  # True → skip MediaPipe entirely; YOLO is the primary detector
YOLO_HAND_MODEL_PATH = "models/yolo_hand_pose.pt"  # path to a trained .pt checkpoint
YOLO_HAND_BBOX_PADDING = 0.30  # normalised padding around hand bbox for YOLO input
YOLO_HAND_CONF_THRESHOLD = 0.25  # minimum per-keypoint confidence to overwrite a landmark

# ============================
# YOLO-Pose PD-specific fine-tuning (pseudo-label pipeline)
# ============================
YOLO_PD_MODEL_PATH = "models/yolo_pd_hand_pose.pt"  # output path for the fine-tuned model
YOLO_PD_MIN_MCP_CONF = 0.70  # per-frame MCP jitter-proxy confidence threshold for label inclusion
YOLO_PD_FRAME_BOUNDARY_MARGIN = 0.02  # fraction of frame dim; keypoints closer to edge are excluded
YOLO_PD_MIN_BBOX_FRAC = 0.01  # minimum hand bbox area as fraction of frame area
YOLO_PD_TRAIN_EPOCHS = 100  # fine-tuning epochs
YOLO_PD_TRAIN_IMGSZ = 640  # training image size (square)
YOLO_PD_TRAIN_BATCH = 64  # fixed batch; avoids AutoBatch profiling crash on certain GPU architectures
YOLO_PD_BBOX_LABEL_PADDING = 0.15  # fractional padding when deriving bbox from landmarks for labels

# ============================
# Adaptive landmark smoothing (One-Euro Filter)
# ============================
USE_ONE_EURO = True  # True → adaptive One-Euro filter; False → fixed moving-average fallback
ONEEURO_MIN_CUTOFF = 1.5  # Hz; stationary cutoff. alpha ≈ 0.27 @ 25fps → good jitter suppression.
ONEEURO_BETA = 50.0  # speed coefficient for normalised [0,1] coords @ 25fps:
#   at 0.3 norm/s cutoff ≈ 16 Hz (alpha≈0.80), 1.0 norm/s → alpha≈0.93.
#   beta=2 caused visible lag (α≈0.38 during motion).
ONEEURO_D_CUTOFF = 2.0  # Hz; derivative LP cutoff — higher means faster speed detection at
#   motion onset (was 1.0 Hz → too slow to open filter on PS start).

# ============================
# PS-activity trimming (remove non-task frames)
# ============================
TRIM_TO_PS_ACTIVITY = True  # if True, keep only frames during active pronation-supination
MAX_PS_DURATION_S = 10.0  # maximum duration (seconds) of the retained PS segment; the
#   first this-many seconds from PS onset are kept
MIN_PS_DURATION_S = 4.0  # minimum total active-PS duration (seconds) required for a
#   merged segment to be considered a valid PS bout; shorter
#   segments are discarded before onset detection
PS_ACTIVITY_WINDOW_S = 1  # rolling-window width (seconds) for smoothing angular velocity
PS_ACTIVITY_PERCENTILE = 75  # percentile of tracked velocity used as the reference baseline
#   for the activity threshold; replaces np.max to avoid a
#   single early transient (rotation ramp or PCA angle flip)
#   from setting an unrealistically high peak that causes the
#   rest of the recording to fall below the threshold
PS_ACTIVITY_THRESHOLD_RATIO = 0.15  # fraction of PS_ACTIVITY_PERCENTILE velocity used as activity
#   cutoff (was 0.20 × peak; now 0.15 × 75th-pctile)
PS_MERGE_GAP_S = 2.5  # merge active regions separated by less than this (seconds)
#   (was 2.0; increased to 2.5 to reduce over-splitting)

# ============================
# Zero-crossing cycle-detection tuning
# ============================
ZCR_DC_WINDOW_CYCLES = 2.0  # DC-removal window = this × estimated period (frames).
#   Lower values (1.5) adapt faster to amplitude changes but
#   risk attenuating the fundamental; 2.0 is a good balance.
STARTUP_TRANSIENT_FACTOR = 4.0  # If the first full cycle amplitude exceeds median(rest) × this
#   factor, treat it as a startup transient and exclude it from
#   feature statistics.  Set to 0 to disable.  Raised from 3.0
#   to 4.0 — legitimate large first cycles were being excluded
#   too aggressively; the Hampel filter now handles onset noise.
HAMPEL_WINDOW = 5  # Rolling window (frames, must be odd) for Hampel filter applied
#   to the detrended angle signal before peak detection; removes
#   single-frame noise spikes that generate spurious FP peaks
HAMPEL_SIGMA = 3.0  # MAD multiplier threshold for Hampel outlier detection;
#   samples with |x − median| > HAMPEL_SIGMA × 1.4826 × MAD
#   are replaced by the local median
AMP_DECREMENT_SPLIT_FRAC = 1.0 / 3  # Fraction of cycle count used as early/late window for
#   sequence-effect percent computation (early-to-late
#   amplitude/velocity decrement).  Default 1/3.
ARREST_MIN_DURATION_S = 0.8  # Absolute minimum half-cycle duration (seconds) to qualify
#   as an arrest.  The relative threshold (1.5 × median) is
#   taken as the maximum of itself and this floor, preventing
#   fast healthy performers from accumulating false-positive
#   arrests from ordinary timing jitter.

# ============================
# Plot-video settings
# ============================
EXPORT_PLOT_VIDEO = True
PLOT_WIDTH_RATIO = 0.95  # plot panel width = video_width * ratio

# ============================
# Runtime diagnostics (debug)
# ============================
ENABLE_RUNTIME_DIAGNOSTICS = True  # True -> print per-worker CUDA/runtime diagnostics

# ============================
# Parallelism settings
# ============================
NUM_WORKERS = None  # Will default to os.cpu_count()
CHUNK_SIZE = 1
YOLO_BATCH_SIZE = 16  # Frames per YOLO predict() call. Larger batches saturate GPU better.
# 1  — frame-by-frame (original behaviour, low GPU utilisation).
# 16 — good default; raises power draw ~2× vs batch=1.
# 32 — marginal extra gain, increases per-worker memory ~200 MB.
GPU_CONCURRENCY = 20  # Max workers doing GPU work (YOLO/SuperRes/RTMPose) simultaneously.
# 1  — safest for a small GPU (≤12 GB VRAM).
# 8  — conservative for GPUs with ≥48 GB VRAM when YOLO is enabled
#       (~270 MB YOLO model × 8 = ~6 GB, trivial).
# 20 — suitable for high-memory GPUs (≥80 GB) with MediaPipe-only inference:
#       each worker uses ~1.1 GB.
#       Lower to 8–12 if YOLO is also enabled per worker.

# ============================
# Optional SciPy (recommended)
# ============================
try:

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the pipeline, replacing global mutable state.

    All tuning constants are fields with defaults matching the module-level
    globals.  Construct with overrides::

        config = PipelineConfig(**{**defaults, **user_overrides})

    ``__post_init__`` validates field types and flags unknown keys at
    construction time, preventing silent-typo bugs.

    This dataclass is the preferred mechanism for passing configuration
    through the pipeline.  The ``apply_tuning_overrides()`` +
    module-level globals are retained for backward compatibility.
    """

    # Reference frame rate — frame-count constants are calibrated for this fps.
    # Overriding this is NOT necessary for normal use; the pipeline infers fps
    # from each video's metadata and scales automatically.  Set only if you want
    # to reinterpret all frame-count tuning values at a different base rate.
    BASE_FPS: float = 25.0

    # Frame-rate cap — see module-level TARGET_PROCESSING_FPS docstring.
    TARGET_PROCESSING_FPS: float = 0

    # Intra-video parallelism — split a single video's MediaPipe inference
    # across this many sub-processes.  Useful when you have more CPU cores
    # than videos.  Each sub-process handles a chunk of frames; results are
    # merged before tracking.  Set to 1 (default) for serial per-video
    # inference.  The total process count is roughly
    # --workers × INTRA_VIDEO_WORKERS, so size accordingly.
    INTRA_VIDEO_WORKERS: int = 1

    # Tracking
    TRACK_MATCH_THRESH: float = 0.08
    MAX_GAP: int = 3
    MAX_JUMP_PER_FRAME: float = 0.04
    FILL_MAX_DIST: float = 0.12
    HANDEDNESS_PENALTY_MULT: float = 1.25
    ROT_AVG_THRESH: float = 0.001
    LANDMARK_SMOOTH_WINDOW: int = 3

    # Keypoint reliability & fill quality
    FILL_ITERATIONS: int = 2
    USE_CLAHE_ON_FILL: bool = True
    ROI_REDETECT_PADDING: float = 0.25
    VISIBILITY_THRESHOLD: float = 0.35
    CYCLE_NAN_THRESHOLD: float = 0.40
    USE_PCA_ANGLE: bool = True
    LM_OUTLIER_WINDOW: int = 7
    LM_OUTLIER_THRESH: float = 0.06

    # Pre-filter angle detrending
    DETREND_POLY_ORDER: int = 3

    # Wrist-Z cycle-detection confirmation
    USE_WRIST_Z_CONFIRMATION: bool = True
    WRIST_Z_CONFIRM_WINDOW: int = 5
    WRIST_Z_PROM_BOOST: float = 1.5
    WRIST_Z_MIN_VALID_FRAC: float = 0.50
    WRIST_Z_MIN_SNR: float = 1.5

    # Adaptive visibility threshold
    ADAPTIVE_VISIBILITY: bool = True
    ADAPTIVE_VIS_PERCENTILE: int = 20
    ADAPTIVE_VIS_FLOOR: float = 0.15

    # Super-resolution
    USE_SUPERRES: bool = False
    SUPERRES_SCALE: int = 4
    SUPERRES_MODEL_NAME: str = "realesr-general-x4v3"
    SUPERRES_MODEL_PATH: Optional[str] = None
    SUPERRES_HALF: bool = True

    # RTMPose
    USE_RTMPOSE: bool = False
    RTMPOSE_MODEL_CFG: str = "rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256"
    RTMPOSE_CHECKPOINT_PATH: Optional[str] = None
    RTMPOSE_BBOX_PADDING: float = 0.30

    # OpenPose
    USE_OPENPOSE: bool = False
    OPENPOSE_PROTO_PATH: str = "models/openpose/hand/pose_deploy.prototxt"
    OPENPOSE_WEIGHTS_PATH: str = "models/openpose/hand/pose_iter_102000.caffemodel"
    OPENPOSE_BBOX_PADDING: float = 0.30
    OPENPOSE_CONF_THRESHOLD: float = 0.10
    OPENPOSE_INPUT_SIZE: int = 368
    OPENPOSE_USE_CUDA: bool = False

    # YOLO-Pose
    USE_YOLO_HAND: bool = False
    USE_YOLO_ONLY: bool = False
    YOLO_HAND_MODEL_PATH: str = "models/yolo_hand_pose.pt"
    YOLO_HAND_BBOX_PADDING: float = 0.30
    YOLO_HAND_CONF_THRESHOLD: float = 0.25

    # Smoothing
    USE_ONE_EURO: bool = True
    ONEEURO_MIN_CUTOFF: float = 1.5
    ONEEURO_BETA: float = 50.0
    ONEEURO_D_CUTOFF: float = 2.0

    # PS-activity trimming
    TRIM_TO_PS_ACTIVITY: bool = True
    MAX_PS_DURATION_S: float = 10.0
    MIN_PS_DURATION_S: float = 4.0
    PS_ACTIVITY_WINDOW_S: int = 1
    PS_ACTIVITY_PERCENTILE: int = 75
    PS_ACTIVITY_THRESHOLD_RATIO: float = 0.15
    PS_MERGE_GAP_S: float = 2.5

    # Zero-crossing cycle-detection
    ZCR_DC_WINDOW_CYCLES: float = 2.0
    STARTUP_TRANSIENT_FACTOR: float = 4.0
    HAMPEL_WINDOW: int = 5
    HAMPEL_SIGMA: float = 3.0
    AMP_DECREMENT_SPLIT_FRAC: float = 1.0 / 3
    ARREST_MIN_DURATION_S: float = 0.8

    # Plot-video settings
    EXPORT_PLOT_VIDEO: bool = True
    PLOT_WIDTH_RATIO: float = 0.95

    # Runtime diagnostics
    ENABLE_RUNTIME_DIAGNOSTICS: bool = True

    # Parallelism
    NUM_WORKERS: Optional[int] = None
    CHUNK_SIZE: int = 1
    GPU_CONCURRENCY: int = 1
    YOLO_BATCH_SIZE: int = 16

    @classmethod
    def from_overrides(cls, overrides: dict) -> "PipelineConfig":
        """Build a PipelineConfig from a dict of overrides.

        Unknown keys raise a ``ValueError`` immediately — no silent
        ignoring of typos.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(
                f"Unknown PipelineConfig keys: {sorted(unknown)}. " f"Valid keys: {sorted(known)}"
            )
        return cls(**overrides)

    def apply_to_globals(self):
        """Push this config's values into the module-level globals.

        Bridges the new PipelineConfig dataclass with code that still
        reads module-level constants.  Call once at worker initialisation.
        """
        g = globals()
        for f in fields(self):
            if f.name in g:
                g[f.name] = getattr(self, f.name)


def apply_tuning_overrides(overrides):
    """Apply tuning parameter overrides to module-level constants.

    Call at the start of each worker process or before processing.
    Accepts a dict mapping constant names to new values.
    Only known constants are updated; unknown keys are silently ignored
    (they may be KinematicAnalyzer or MediaPipe-confidence params
    handled elsewhere).
    """
    global BASE_FPS, TARGET_PROCESSING_FPS, INTRA_VIDEO_WORKERS
    global TRACK_MATCH_THRESH, MAX_GAP, MAX_JUMP_PER_FRAME, FILL_MAX_DIST
    global HANDEDNESS_PENALTY_MULT, ROT_AVG_THRESH, EXPORT_PLOT_VIDEO
    global LANDMARK_SMOOTH_WINDOW
    global TRIM_TO_PS_ACTIVITY, MAX_PS_DURATION_S, MIN_PS_DURATION_S
    global PS_ACTIVITY_WINDOW_S, PS_ACTIVITY_PERCENTILE, PS_ACTIVITY_THRESHOLD_RATIO, PS_MERGE_GAP_S
    global FILL_ITERATIONS, USE_CLAHE_ON_FILL, ROI_REDETECT_PADDING
    global VISIBILITY_THRESHOLD, CYCLE_NAN_THRESHOLD, USE_PCA_ANGLE
    global LM_OUTLIER_WINDOW, LM_OUTLIER_THRESH
    global USE_WRIST_Z_CONFIRMATION, WRIST_Z_CONFIRM_WINDOW, WRIST_Z_PROM_BOOST
    global WRIST_Z_MIN_VALID_FRAC, WRIST_Z_MIN_SNR
    global DETREND_POLY_ORDER
    global USE_ONE_EURO, ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF
    global ADAPTIVE_VISIBILITY, ADAPTIVE_VIS_PERCENTILE, ADAPTIVE_VIS_FLOOR
    global USE_SUPERRES, SUPERRES_SCALE, SUPERRES_MODEL_NAME, SUPERRES_MODEL_PATH, SUPERRES_HALF
    global USE_RTMPOSE, RTMPOSE_MODEL_CFG, RTMPOSE_CHECKPOINT_PATH, RTMPOSE_BBOX_PADDING
    global USE_OPENPOSE, OPENPOSE_PROTO_PATH, OPENPOSE_WEIGHTS_PATH
    global OPENPOSE_BBOX_PADDING, OPENPOSE_CONF_THRESHOLD, OPENPOSE_INPUT_SIZE, OPENPOSE_USE_CUDA
    global USE_YOLO_HAND, USE_YOLO_ONLY, YOLO_HAND_MODEL_PATH, YOLO_HAND_BBOX_PADDING, YOLO_HAND_CONF_THRESHOLD
    global ENABLE_RUNTIME_DIAGNOSTICS
    global GPU_CONCURRENCY, YOLO_BATCH_SIZE
    global ZCR_DC_WINDOW_CYCLES, STARTUP_TRANSIENT_FACTOR, HAMPEL_WINDOW, HAMPEL_SIGMA, AMP_DECREMENT_SPLIT_FRAC, ARREST_MIN_DURATION_S
    global PLOT_WIDTH_RATIO, NUM_WORKERS, CHUNK_SIZE
    if not overrides:
        return
    _MAP = {
        "BASE_FPS": "BASE_FPS",
        "TARGET_PROCESSING_FPS": "TARGET_PROCESSING_FPS",
        "INTRA_VIDEO_WORKERS": "INTRA_VIDEO_WORKERS",
        "TRACK_MATCH_THRESH": "TRACK_MATCH_THRESH",
        "MAX_GAP": "MAX_GAP",
        "MAX_JUMP_PER_FRAME": "MAX_JUMP_PER_FRAME",
        "FILL_MAX_DIST": "FILL_MAX_DIST",
        "HANDEDNESS_PENALTY_MULT": "HANDEDNESS_PENALTY_MULT",
        "ROT_AVG_THRESH": "ROT_AVG_THRESH",
        "EXPORT_PLOT_VIDEO": "EXPORT_PLOT_VIDEO",
        "LANDMARK_SMOOTH_WINDOW": "LANDMARK_SMOOTH_WINDOW",
        "TRIM_TO_PS_ACTIVITY": "TRIM_TO_PS_ACTIVITY",
        "MAX_PS_DURATION_S": "MAX_PS_DURATION_S",
        "MIN_PS_DURATION_S": "MIN_PS_DURATION_S",
        "PS_ACTIVITY_WINDOW_S": "PS_ACTIVITY_WINDOW_S",
        "PS_ACTIVITY_PERCENTILE": "PS_ACTIVITY_PERCENTILE",
        "PS_ACTIVITY_THRESHOLD_RATIO": "PS_ACTIVITY_THRESHOLD_RATIO",
        "PS_MERGE_GAP_S": "PS_MERGE_GAP_S",
        "FILL_ITERATIONS": "FILL_ITERATIONS",
        "USE_CLAHE_ON_FILL": "USE_CLAHE_ON_FILL",
        "ROI_REDETECT_PADDING": "ROI_REDETECT_PADDING",
        "VISIBILITY_THRESHOLD": "VISIBILITY_THRESHOLD",
        "CYCLE_NAN_THRESHOLD": "CYCLE_NAN_THRESHOLD",
        "USE_PCA_ANGLE": "USE_PCA_ANGLE",
        "LM_OUTLIER_WINDOW": "LM_OUTLIER_WINDOW",
        "LM_OUTLIER_THRESH": "LM_OUTLIER_THRESH",
        "USE_WRIST_Z_CONFIRMATION": "USE_WRIST_Z_CONFIRMATION",
        "WRIST_Z_CONFIRM_WINDOW": "WRIST_Z_CONFIRM_WINDOW",
        "WRIST_Z_PROM_BOOST": "WRIST_Z_PROM_BOOST",
        "WRIST_Z_MIN_VALID_FRAC": "WRIST_Z_MIN_VALID_FRAC",
        "WRIST_Z_MIN_SNR": "WRIST_Z_MIN_SNR",
        "DETREND_POLY_ORDER": "DETREND_POLY_ORDER",
        "USE_ONE_EURO": "USE_ONE_EURO",
        "ONEEURO_MIN_CUTOFF": "ONEEURO_MIN_CUTOFF",
        "ONEEURO_BETA": "ONEEURO_BETA",
        "ONEEURO_D_CUTOFF": "ONEEURO_D_CUTOFF",
        "ADAPTIVE_VISIBILITY": "ADAPTIVE_VISIBILITY",
        "ADAPTIVE_VIS_PERCENTILE": "ADAPTIVE_VIS_PERCENTILE",
        "ADAPTIVE_VIS_FLOOR": "ADAPTIVE_VIS_FLOOR",
        "USE_SUPERRES": "USE_SUPERRES",
        "SUPERRES_SCALE": "SUPERRES_SCALE",
        "SUPERRES_MODEL_NAME": "SUPERRES_MODEL_NAME",
        "SUPERRES_MODEL_PATH": "SUPERRES_MODEL_PATH",
        "SUPERRES_HALF": "SUPERRES_HALF",
        "USE_RTMPOSE": "USE_RTMPOSE",
        "RTMPOSE_MODEL_CFG": "RTMPOSE_MODEL_CFG",
        "RTMPOSE_CHECKPOINT_PATH": "RTMPOSE_CHECKPOINT_PATH",
        "RTMPOSE_BBOX_PADDING": "RTMPOSE_BBOX_PADDING",
        "USE_OPENPOSE": "USE_OPENPOSE",
        "OPENPOSE_PROTO_PATH": "OPENPOSE_PROTO_PATH",
        "OPENPOSE_WEIGHTS_PATH": "OPENPOSE_WEIGHTS_PATH",
        "OPENPOSE_BBOX_PADDING": "OPENPOSE_BBOX_PADDING",
        "OPENPOSE_CONF_THRESHOLD": "OPENPOSE_CONF_THRESHOLD",
        "OPENPOSE_INPUT_SIZE": "OPENPOSE_INPUT_SIZE",
        "OPENPOSE_USE_CUDA": "OPENPOSE_USE_CUDA",
        "USE_YOLO_HAND": "USE_YOLO_HAND",
        "USE_YOLO_ONLY": "USE_YOLO_ONLY",
        "YOLO_HAND_MODEL_PATH": "YOLO_HAND_MODEL_PATH",
        "YOLO_HAND_BBOX_PADDING": "YOLO_HAND_BBOX_PADDING",
        "YOLO_HAND_CONF_THRESHOLD": "YOLO_HAND_CONF_THRESHOLD",
        "ENABLE_RUNTIME_DIAGNOSTICS": "ENABLE_RUNTIME_DIAGNOSTICS",
        "GPU_CONCURRENCY": "GPU_CONCURRENCY",
        "YOLO_BATCH_SIZE": "YOLO_BATCH_SIZE",
        "ZCR_DC_WINDOW_CYCLES": "ZCR_DC_WINDOW_CYCLES",
        "STARTUP_TRANSIENT_FACTOR": "STARTUP_TRANSIENT_FACTOR",
        "HAMPEL_WINDOW": "HAMPEL_WINDOW",
        "HAMPEL_SIGMA": "HAMPEL_SIGMA",
        "AMP_DECREMENT_SPLIT_FRAC": "AMP_DECREMENT_SPLIT_FRAC",
        "ARREST_MIN_DURATION_S": "ARREST_MIN_DURATION_S",
        "PLOT_WIDTH_RATIO": "PLOT_WIDTH_RATIO",
        "NUM_WORKERS": "NUM_WORKERS",
        "CHUNK_SIZE": "CHUNK_SIZE",
    }
    for key, gname in _MAP.items():
        if key in overrides:
            globals()[gname] = overrides[key]

    # Warn about unrecognised keys (typos, renamed constants, etc.)
    # Keys handled elsewhere (KinematicAnalyzer kwargs, MediaPipe confidence)
    # are listed in _PASSTHROUGH_KEYS to suppress false positives.
    _PASSTHROUGH_KEYS = {
        "cutoff_hz",
        "highpass_hz",
        "prominence_deg",
        "filter_order",
        "max_movement_hz",
        "adaptive_prom_frac",
        "min_half_cycle_s",
        "min_hand_detection_confidence",
        "min_hand_presence_confidence",
        "min_tracking_confidence",
    }
    unknown = set(overrides) - set(_MAP) - _PASSTHROUGH_KEYS
    if unknown:
        warnings.warn(
            f"apply_tuning_overrides: unrecognised keys ignored: {sorted(unknown)}. "
            "Check for typos or add them to _MAP / _PASSTHROUGH_KEYS.",
            stacklevel=2,
        )


# ============================
# Hampel filter — spike suppression on angle signal
# ============================


def hampel_filter(
    signal: np.ndarray, window: int = HAMPEL_WINDOW, sigma: float = HAMPEL_SIGMA
) -> np.ndarray:
    """Replace outlier samples in *signal* with the local median.

    Uses a rolling window of *window* frames (must be odd).  A sample is
    considered an outlier if ``|x − median| > sigma × 1.4826 × MAD`` where
    MAD is the median absolute deviation of the window.  1.4826 is the
    consistency factor that makes the MAD a consistent estimator of σ for
    Gaussian noise.

    Pure-numpy implementation; no scipy dependency.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
    window : int
        Rolling window half-width = (window - 1) // 2 frames; must be odd.
    sigma : float
        Number of normalised MADs above which a sample is flagged.

    Returns
    -------
    np.ndarray
        Copy of *signal* with outliers replaced by the local median.
    """
    if window < 1:
        return signal.copy()
    if window % 2 == 0:
        window += 1  # enforce odd
    half = window // 2
    out = signal.copy().astype(float)
    n = len(signal)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        patch = signal[lo:hi]
        med = np.median(patch)
        mad = np.median(np.abs(patch - med))
        if np.abs(signal[i] - med) > sigma * 1.4826 * mad:
            out[i] = med
    return out


# ============================
# One-Euro Filter — adaptive low-pass for landmark smoothing
# ============================


class OneEuroFilter:
    """Adaptive low-pass filter that smooths heavily when still and lightly
    when moving fast.  Standard in AR/VR hand-tracking pipelines.

    Reference: Casiez, Roussel & Vogel, "1€ Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012.

    Parameters
    ----------
    fps : float
        Sampling rate (frames per second).
    min_cutoff : float
        Minimum cutoff frequency (Hz).  Lower → more smoothing when
        stationary (kills jitter).
    beta : float
        Speed coefficient.  Higher → less lag during fast movement.
    d_cutoff : float
        Cutoff for the derivative filter (Hz).  Rarely needs tuning.
    """

    def __init__(
        self, fps: float, min_cutoff: float = 0.05, beta: float = 0.5, d_cutoff: float = 1.0
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dt = 1.0 / max(float(fps), 1.0)
        # State (initialised on first call)
        self._x_prev = None
        self._dx_prev = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float) -> float:
        """Filter a single scalar sample and return the smoothed value."""
        if self._x_prev is None:
            # First sample — no filtering possible
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        # Derivative (speed) estimate
        a_d = self._alpha(self.dt, self.d_cutoff)
        dx = (x - self._x_prev) / self.dt
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff — increases with speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(self.dt, cutoff)

        # Filtered value
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self):
        """Clear internal state so the next sample is treated as the first."""
        self._x_prev = None
        self._dx_prev = None


def one_euro_filter_array(
    values: np.ndarray,
    fps: float,
    min_cutoff: float = 0.05,
    beta: float = 0.5,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    """Apply a One-Euro filter to each element of a 1-D array in order.

    NaN entries are skipped (and the filter state is reset at NaN gaps
    longer than 1 frame).

    Returns
    -------
    np.ndarray  — same shape as *values*, smoothed in-place.
    """
    out = values.copy()
    filt = OneEuroFilter(fps=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    gap = 0
    for i in range(len(out)):
        if np.isnan(out[i]):
            gap += 1
            continue
        # Reset filter state when resuming after a gap of 2+ NaN frames.
        # A single NaN frame is short enough to bridge without resetting.
        if gap > 1:
            filt.reset()
        gap = 0
        out[i] = filt(out[i])
    return out


# ============================
# Utilities: lightweight JSON series
# ============================


def series_to_json(x, decimals=4):
    """Convert numpy array/list into a compact JSON string (NaN -> null)."""
    arr = x.tolist() if hasattr(x, "tolist") else list(x)
    out = []
    for v in arr:
        if v is None:
            out.append(None)
        else:
            try:
                fv = float(v)
                if np.isnan(fv) or np.isinf(fv):
                    out.append(None)
                else:
                    out.append(round(fv, decimals))
            except Exception:
                out.append(None)
    return json.dumps(out, separators=(",", ":"))


def json_to_series(s):
    return np.array(json.loads(s), dtype=float)


# ============================
# Frame enhancement utility
# ============================


def clahe_enhance(frame_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """Apply CLAHE to the Y channel for better hand detection in dim / uneven light.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR image.
    clip_limit : float
        CLAHE clip limit (default 2.0).
    tile_grid : tuple[int, int]
        CLAHE tile grid size (default (8, 8)).

    Returns
    -------
    np.ndarray
        Enhanced BGR image.
    """
    if cv2 is None:
        return frame_bgr
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# Backward-compatible alias for internal callers
_clahe_enhance = clahe_enhance


# ============================
# Phase unwrapping
# ============================


def _unwrap_segments(raw_rad):
    """Gap-aware phase unwrapping for a 1-D array of wrapped angles (radians).

    Unlike plain ``np.unwrap``, which requires a gapless sequence and is
    therefore typically called on a linearly-interpolated bridge across
    NaN gaps, this function:

    1. Unwraps each contiguous run of non-NaN samples independently.
    2. Stitches adjacent runs with a single +/-2pi-multiple correction that
       minimises the absolute phase jump at each gap boundary.

    Parameters
    ----------
    raw_rad : np.ndarray, shape (N,)
        Wrapped angles in radians.  NaN marks missing / invalid frames.

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        Unwrapped angles in radians.  NaN positions are preserved exactly.
    """
    raw_rad = np.asarray(raw_rad, dtype=np.float64)
    result = np.full_like(raw_rad, np.nan)
    valid = ~np.isnan(raw_rad)
    if valid.sum() < 2:
        return raw_rad.copy()

    # Identify contiguous runs of valid (non-NaN) indices
    runs = []
    in_run = False
    run_start = 0
    for i in range(len(raw_rad)):
        if valid[i]:
            if not in_run:
                run_start = i
                in_run = True
        else:
            if in_run:
                runs.append((run_start, i))
                in_run = False
    if in_run:
        runs.append((run_start, len(raw_rad)))

    # Unwrap each run independently, stitch at gap boundaries
    prev_end_val = None
    for s, e in runs:
        segment = np.unwrap(raw_rad[s:e].copy())
        if prev_end_val is not None:
            delta = segment[0] - prev_end_val
            correction = -round(delta / (2.0 * np.pi)) * 2.0 * np.pi
            segment = segment + correction
        result[s:e] = segment
        prev_end_val = float(segment[-1])

    return result
