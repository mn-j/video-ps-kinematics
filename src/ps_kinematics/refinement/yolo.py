"""
ps_kinematics.yolo — YOLO-Pose hand landmark refinement.

Provides a GPU-accelerated YOLO-Pose hand keypoint inferencer that refines
the MCP keypoint coordinates initially detected by MediaPipe.
Only used when ``USE_YOLO_HAND = True``.

Integration strategy (hybrid, same as RTMPose/OpenPose):
  1. MediaPipe VIDEO mode runs first → hand detection, tracking, handedness.
  2. YOLO-Pose refines hand landmark (x, y) coordinates on each tracked ROI,
     overwriting MediaPipe coordinates.

This preserves the entire MultiHandOfflineTracker, fill-pass logic, and
handedness classification.  Only the *coordinates* change.

Model provisioning:
  Set ``YOLO_HAND_MODEL_PATH`` to a trained ``.pt`` checkpoint.
  Use ``--yolo-pd-finetune`` in run_pipeline.py to fine-tune a model on
  PD pseudo-labels extracted from your own dataset.

Keypoint convention:
  Both Ultralytics hand-keypoints and MediaPipe use the identical 21-point
  layout (wrist + 4 joints × 5 fingers), so the mapping is identity.

Requires: ultralytics  (``pip install ultralytics``)
"""

import logging
import os
import shutil

import numpy as np

logger = logging.getLogger(__name__)

# ── Deferred torch / CUDA availability check ─────────────────────────
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
    cv2 = None
    CV2_OK = False


# Lazy-initialised YOLO model (one per process)
_yolo_model = None
_yolo_model_path = None

# ============================================================
# Ultralytics hand-keypoints ↔ MediaPipe landmark mapping
# ============================================================
# The model is trained on 5 kinematically-relevant keypoints only
# (pronation-supination angle uses wrist + 4 MCPs; fingertip/DIP/PIP
# landmarks are excluded to concentrate loss signal and reduce label noise).
#
# YOLO output index → MediaPipe landmark index:
#   0 → 0   wrist
#   1 → 5   index MCP
#   2 → 9   middle MCP
#   3 → 13  ring MCP
#   4 → 17  pinky MCP
YOLO_TO_MP_INDEX = [0, 5, 9, 13, 17]

# The 5 keypoints used by the kinematic pipeline (same set as above)
_KINEMATIC_INDICES = [0, 5, 9, 13, 17]  # WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP


# ============================================================
# Model loading
# ============================================================


def _get_yolo_model(model_path=None):
    """Lazily load and cache a YOLO hand keypoint model.

    Parameters
    ----------
    model_path : str or None
        Path to a trained Ultralytics YOLO hand-keypoint model (``.pt``).
        If None, uses the default from utils.YOLO_HAND_MODEL_PATH.

    Returns
    -------
    ultralytics.YOLO or None
        The YOLO model, or None if dependencies/GPU are unavailable.

    Raises
    ------
    RuntimeError
        If no CUDA GPU is detected.
    FileNotFoundError
        If the model file does not exist.
    """
    global _yolo_model, _yolo_model_path

    _ensure_torch()
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "[YOLO-Hand] No CUDA GPU detected by torch (torch.cuda.is_available() == False). "
            "YOLO-Hand requires a GPU.  Check CUDA_VISIBLE_DEVICES, your torch "
            "installation, and that the GPU driver is accessible from this process."
        )

    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "yolo_hand_pose.pt",
        )

    model_path = os.path.abspath(str(model_path))

    # Return cached model if path hasn't changed
    if _yolo_model is not None and _yolo_model_path == model_path:
        return _yolo_model

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[YOLO-Hand] Model file not found: {model_path}\n"
            "Run --yolo-pd-finetune to fine-tune a model on PD pseudo-labels,\n"
            "or set YOLO_HAND_MODEL_PATH to an existing YOLO-Pose .pt checkpoint."
        )

    try:
        import torch
        from ultralytics import YOLO

        _yolo_model = YOLO(model_path, task="pose")
        # Warm up with a dummy inference to load weights onto GPU
        _yolo_model.predict(
            np.zeros((64, 64, 3), dtype=np.uint8),
            device="cuda:0",
            verbose=False,
        )

        mem_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
        logger.info("Model loaded from %s — GPU memory: %d MB", model_path, mem_mb)

        _yolo_model_path = model_path
        return _yolo_model

    except ImportError:
        logger.error("ultralytics package not installed. " "Install with: pip install ultralytics")
        _yolo_model = None
        _yolo_model_path = None
        return None
    except RuntimeError:
        raise
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        _yolo_model = None
        _yolo_model_path = None
        return None


# ============================================================
# Landmark refinement
# ============================================================


def _hand_bbox_from_landmarks(lm_arr, img_w, img_h, padding=0.30):
    """Compute a pixel-space hand bbox from normalised landmarks.

    Returns
    -------
    tuple of (x1, y1, x2, y2) or None
        Integer pixel bounding box, or None if degenerate.
    """
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

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(int(img_w), int(x2 + pad_x))
    y2 = min(int(img_h), int(y2 + pad_y))

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return x1, y1, x2, y2


def refine_landmarks_yolo(frame_bgr, lm_arr_mp, model_path=None, padding=0.30, conf_thresh=0.25):
    """Refine hand landmarks with YOLO-Pose on a single frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Full video frame in BGR format, shape (H, W, 3).
    lm_arr_mp : np.ndarray
        MediaPipe landmark array, shape (21, 4): [x_norm, y_norm, z, visibility].
    model_path : str or None
        Path to the YOLO hand keypoint model (``.pt``).
    padding : float
        Normalised padding around the MediaPipe hand bounding box.
    conf_thresh : float
        Minimum per-keypoint confidence required to overwrite a landmark.

    Returns
    -------
    np.ndarray or None
        Refined landmark array (shape (21, 4)) with updated (x, y) in
        normalised coords.  z is preserved from MediaPipe.  Visibility is
        updated to the YOLO keypoint confidence.
        Returns None if inference fails (caller should keep MediaPipe result).
    """
    if not CV2_OK or frame_bgr is None or lm_arr_mp is None:
        return None

    model = _get_yolo_model(model_path)
    if model is None:
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
        results = model.predict(
            roi,
            device="cuda:0",
            verbose=False,
            conf=0.15,  # detection confidence threshold (permissive;
            # per-keypoint filtering is done below)
        )
        if results is None or len(results) == 0:
            return None

        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        kpts = result.keypoints
        if kpts.xy is None or kpts.xy.shape[0] == 0:
            return None

        # Select the detection with the highest box confidence
        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            best_idx = int(result.boxes.conf.argmax())
        else:
            best_idx = 0

        xy = kpts.xy[best_idx].cpu().numpy()  # (K, 2) pixels in ROI
        if kpts.conf is not None:
            conf = kpts.conf[best_idx].cpu().numpy()  # (K,)
        else:
            conf = np.ones(xy.shape[0], dtype=np.float32)

        if xy.shape[0] < len(YOLO_TO_MP_INDEX):
            return None

        # Build refined landmark array — only overwrite the 5 kinematic keypoints
        refined = lm_arr_mp.copy()
        for i in range(len(YOLO_TO_MP_INDEX)):
            mp_idx = YOLO_TO_MP_INDEX[i]
            if float(conf[i]) < float(conf_thresh):
                continue

            # Map ROI pixel coords → full-frame normalised coords
            px = float(x1) + float(xy[i, 0])
            py = float(y1) + float(xy[i, 1])

            refined[mp_idx, 0] = px / max(1.0, float(w_img))
            refined[mp_idx, 1] = py / max(1.0, float(h_img))
            # Preserve z from MediaPipe (YOLO is 2D)
            refined[mp_idx, 3] = float(conf[i])

        return refined

    except Exception:
        return None


# ============================================================
# PD-specific fine-tuning from pseudo-labels
# ============================================================


def _yolo_training_worker(cand_path, train_kwargs, cudnn_enabled=False):
    """Top-level spawned-process worker that runs one YOLO training trial.

    Must be a *module-level* function (not a closure) so it is picklable by
    the 'spawn' multiprocessing start method.

    Sets ``CUDA_LAUNCH_BLOCKING=1`` before importing torch/ultralytics so
    that CUDA operations are executed synchronously.  This prevents the
    async CUDA kernel race conditions that produce SIGSEGV.

    Parameters
    ----------
    cand_path : str
        Path to the base model or ``last.pt`` checkpoint to resume from.
    train_kwargs : dict
        Keyword arguments forwarded to ``model.train()``.
    cudnn_enabled : bool
        Whether to enable cuDNN.  Default ``False`` — certain cuDNN versions
        have a known SIGSEGV bug on specific GPU compute capabilities.
        Set ``True`` via ``--enable-cudnn`` on GPUs where cuDNN is stable,
        for ~30% faster convolutions.
    """
    import faulthandler
    import os
    import sys

    import numpy as np

    # Dump full C/CUDA stack trace to stderr on SIGSEGV so the crash site is visible.
    faulthandler.enable(file=sys.stderr, all_threads=True)
    # Synchronous CUDA: prevents async kernel races that can SIGSEGV.
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # Reduce CUDA memory allocator fragmentation.  Without this, the allocator
    # can hold 3–4 GB of reserved-but-unallocated blocks after a large training
    # batch, leaving insufficient contiguous space for the subsequent validation
    # forward pass (observed as OOM on 12 GB GPUs despite apparent free VRAM).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    # Some consumer GPUs block NCCL P2P direct GPU memory access,
    # causing SIGSEGV in DDP child processes.  Disable P2P and IB so NCCL
    # falls back to socket-based transfers, which work in all environments.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    # cuDNN control: disabled by default to work around a SIGSEGV bug in
    # certain cuDNN versions on specific GPU compute capabilities.
    # Pass cudnn_enabled=True via --enable-cudnn when running on GPUs
    # where cuDNN is known to be stable.
    try:
        import torch

        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.enabled = cudnn_enabled
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False  # benchmark=True re-invokes cuDNN
    except Exception:
        pass

    # Ultralytics 8.4.23 hardcodes OKS_SIGMA as a 17-element array (COCO body
    # keypoints).  When training with kpt_shape=[5, 3], PoseLoss tries to
    # broadcast this (17,) sigma against (N, 5, 3) keypoint predictions.
    # Patch to 5 uniform values before Ultralytics builds the loss function.
    try:
        import ultralytics.utils.metrics as _ul_metrics

        if len(_ul_metrics.OKS_SIGMA) != 5:
            _ul_metrics.OKS_SIGMA = np.full(5, 0.1, dtype=np.float64)
    except Exception:
        pass

    # Motion blur augmentation — teaches the model to localise MCP keypoints
    # through the directional blur that occurs when less-severe PD patients
    # move their hands quickly (high angular velocity mid-cycle).
    # motion_blur_p is popped here so it is never forwarded to model.train(),
    # which would raise an UnknownArgumentError on unrecognised kwargs.
    _motion_blur_p = float(train_kwargs.pop("motion_blur_p", 0.0))
    if _motion_blur_p > 0:
        try:
            import albumentations as _A
            import ultralytics.data.augment as _uda

            _orig_albu_init = _uda.Albumentations.__init__

            def _patched_albu_init(self, p: float = 1.0) -> None:
                _orig_albu_init(self, p)
                if getattr(self, "transform", None) is not None:
                    self.transform = _A.Compose(
                        list(self.transform.transforms)
                        + [
                            # blur_limit=(3, 9): kernel sizes 3/5/7/9 px —
                            # covers hand-crop blur at 30-120 px MCP span.
                            _A.MotionBlur(blur_limit=(3, 9), p=_motion_blur_p),
                        ]
                    )

            _uda.Albumentations.__init__ = _patched_albu_init
            import sys as _sys

            print(
                f"[YOLO-PD] MotionBlur augmentation enabled (p={_motion_blur_p})",
                file=_sys.stderr,
            )
        except Exception as _e:
            import sys as _sys

            print(f"[YOLO-PD] MotionBlur patch failed (continuing without): {_e}", file=_sys.stderr)

    from ultralytics import YOLO

    model = YOLO(cand_path)
    model.train(**train_kwargs)


def train_yolo_pd_hand_model(
    dataset_yaml,
    base_model_path,
    output_path,
    epochs=100,
    imgsz=640,
    batch=4,
    device=0,
    amp=None,
    workers=8,
    cudnn_enabled=False,
    motion_blur_p=0.3,
):
    """Fine-tune a YOLO-Pose hand model on PD pseudo-labels.

    Fine-tunes a YOLO-Pose model on a custom dataset prepared by
    ``scripts/prepare_yolo_pseudolabels.py``.

    **Must be called from the main process BEFORE spawning workers.**

    Parameters
    ----------
    dataset_yaml : str
        Path to the YOLO dataset YAML produced by ``prepare_yolo_pseudolabels.py``.
    base_model_path : str
        Path to the starting checkpoint.  Typically the generic hand-pose model
        at ``YOLO_HAND_MODEL_PATH`` (already fine-tuned on 26k hand images).
    output_path : str
        Destination path for the fine-tuned model (``models/yolo_pd_hand_pose.pt``).
    epochs : int
        Number of fine-tuning epochs.
    imgsz : int
        Training image size (square).
    batch : int
        Batch size.
    device : int, str, list, or None
        Training device.  Same semantics as ``train_yolo_pd_hand_model()`` itself.
    amp : bool or None
        AMP flag — ``None`` forces ``False`` to avoid the AMP-validation
        SIGSEGV present on this system.
    workers : int
        Dataloader workers.  Default 4.  DataLoader workers only load images
        to CPU and never touch CUDA, so spawning them inside the training
        subprocess is safe.  Increase to 8 on nodes with many CPUs and fast
        storage if the GPU sits idle between batches.
    cudnn_enabled : bool
        Whether to enable cuDNN in the training subprocess.  Default ``False``
        to work around a SIGSEGV in certain cuDNN versions on specific GPU
        compute capabilities.  Pass ``True`` via ``--enable-cudnn`` on GPUs
        where cuDNN is stable, for ~30% faster convolutions.
    motion_blur_p : float
        Probability of applying MotionBlur augmentation per image (0–1).
        Default 0.3.  Teaches the model to localise MCP keypoints through
        directional blur, which occurs when less-severe PD patients rotate
        their hands quickly.  Set to 0.0 to disable.

    Returns
    -------
    str
        Absolute path to the fine-tuned model file.
    """
    output_path = os.path.abspath(str(output_path))
    dataset_yaml = os.path.abspath(str(dataset_yaml))
    base_model_path = os.path.abspath(str(base_model_path))

    if os.path.exists(output_path):
        logger.info("[YOLO-PD] Fine-tuned model already exists: %s", output_path)
        return output_path

    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        raise ImportError(
            "[YOLO-PD] The 'ultralytics' package is not installed.\n"
            "Install with:  pip install ultralytics"
        )

    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(
            f"[YOLO-PD] Dataset YAML not found: {dataset_yaml}\n"
            "Run scripts/prepare_yolo_pseudolabels.py first."
        )
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"[YOLO-PD] Base model not found: {base_model_path}\n"
            "Set base_model_path to an existing YOLO-Pose .pt checkpoint."
        )

    logger.info(
        "[YOLO-PD] Fine-tuning from %s on %s for %d epochs.", base_model_path, dataset_yaml, epochs
    )

    model_dir = os.path.dirname(output_path) or "."
    os.makedirs(model_dir, exist_ok=True)
    _out_dir = os.path.join(model_dir, "yolo_pd_hand_train")

    # Skip retraining when a previous run already completed all epochs.
    # We detect completion via results.csv (one row per epoch) rather than
    # the presence of best.pt alone, because best.pt also exists after an
    # interrupted mid-run.  If training was complete but the final copy to
    # output_path failed (e.g. NFS PermissionError), this reattempts the
    # copy only.  Delete the yolo_pd_hand_train directory to force a fresh run.
    _weights_best = os.path.join(_out_dir, "weights", "best.pt")
    _results_csv = os.path.join(_out_dir, "results.csv")
    if os.path.exists(_weights_best) and os.path.exists(_results_csv):
        try:
            with open(_results_csv) as _f:
                _done_epochs = sum(1 for _ in _f) - 1  # subtract header row
        except Exception:
            _done_epochs = 0
        if _done_epochs >= epochs:
            logger.info(
                "[YOLO-PD] Training already complete (%d/%d epochs in %s). "
                "Skipping retraining. Delete %s to force a fresh run.",
                _done_epochs,
                epochs,
                _out_dir,
                _out_dir,
            )
            if not os.path.exists(output_path):
                shutil.copy(_weights_best, output_path)
                logger.info("[YOLO-PD] Fine-tuned model copied to: %s", output_path)
            return output_path

    # Resume from last.pt if it exists (interrupted training), otherwise start
    # fresh.  When starting fresh, remove any stale best.pt from a prior run
    # so we never accidentally copy an old weight at the end of this run.
    _last_pt = os.path.join(_out_dir, "weights", "last.pt")
    _resuming = os.path.exists(_last_pt)
    if _resuming:
        logger.info("[YOLO-PD] Found existing checkpoint — resuming from: %s", _last_pt)
    else:
        _stale_best = os.path.join(_out_dir, "weights", "best.pt")
        if os.path.exists(_stale_best):
            try:
                os.remove(_stale_best)
                logger.info("[YOLO-PD] Removed stale checkpoint: %s", _stale_best)
            except Exception as _e:
                logger.warning("[YOLO-PD] Could not remove %s: %s", _stale_best, _e)

    _is_multi_gpu = (
        (isinstance(device, (list, tuple)) and len(device) > 1)
        or (isinstance(device, str) and "," in device)
        or device is None
    )
    if _is_multi_gpu:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        # AutoBatch (batch=-1) is unsupported for DDP — resolve to 4 per GPU.
        if isinstance(batch, int) and batch < 1:
            try:
                import torch

                if device is None:
                    n_gpus = torch.cuda.device_count()
                elif isinstance(device, (list, tuple)):
                    n_gpus = len(device)
                else:
                    n_gpus = len(str(device).split(","))
                n_gpus = max(1, n_gpus)
            except Exception:
                n_gpus = 1
            batch = 4 * n_gpus
            logger.info(
                "[YOLO-PD] AutoBatch disabled for multi-GPU DDP; " "using batch=%d (%d GPU(s) x 4)",
                batch,
                n_gpus,
            )
    if amp is None:
        amp = False

    train_kwargs = dict(
        data=dataset_yaml,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        project=model_dir,
        name="yolo_pd_hand_train",
        exist_ok=True,
        verbose=True,
        amp=amp,
        workers=int(workers),
        # PyTorch 2.1.x: deterministic=True + nearest-mode Upsample = SIGSEGV.
        # Keep False explicitly so Ultralytics doesn't re-enable it in its trainer.
        deterministic=False,
        # Fine-tuning augmentation: per-image transforms only.
        # mosaic/mixup remain disabled: Ultralytics 8.4.23 has a hardcoded
        # 17-keypoint assumption in its multi-sample label composition path.
        degrees=15.0,  # rotation ±15°
        scale=0.10,  # scale ±10%
        fliplr=0.5,  # horizontal flip (disabled at runtime — no flip_idx in yaml)
        hsv_h=0.015,  # hue jitter
        hsv_s=0.7,  # saturation jitter
        hsv_v=0.4,  # value/brightness jitter
        mosaic=0.0,  # disabled — 17-kpt assumption in label composition
        mixup=0.0,  # disabled — same reason
        # Passed through to the worker where it patches Albumentations;
        # popped before model.train() so Ultralytics never sees it.
        motion_blur_p=float(motion_blur_p),
    )
    if device is not None:
        train_kwargs["device"] = device

    import multiprocessing as _mp

    _spawn_ctx = _mp.get_context("spawn")

    def _remove_partial():
        for _w in ["best.pt", "last.pt"]:
            _p = os.path.join(_out_dir, "weights", _w)
            if os.path.exists(_p):
                try:
                    os.remove(_p)
                except Exception:
                    pass

    # When resuming, Ultralytics restores all hyper-parameters from the
    # checkpoint — only resume=True is needed.  When starting fresh, pass the
    # full train_kwargs built above.
    _cand = _last_pt if _resuming else base_model_path
    _effective_kwargs = {"resume": True} if _resuming else train_kwargs

    proc = _spawn_ctx.Process(
        target=_yolo_training_worker,
        args=(_cand, _effective_kwargs, cudnn_enabled),
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        _remove_partial()
        raise RuntimeError(
            f"[YOLO-PD] Fine-tuning process crashed (exit {proc.exitcode}). "
            "Check CUDA driver compatibility and available VRAM. "
            f"Base model: {base_model_path}  Dataset: {dataset_yaml}"
        )

    best = os.path.join(_out_dir, "weights", "best.pt")
    last = os.path.join(_out_dir, "weights", "last.pt")
    src = best if os.path.exists(best) else (last if os.path.exists(last) else None)
    if src is None:
        raise RuntimeError(
            f"[YOLO-PD] Training completed but no model found under "
            f"{_out_dir}/weights/. Check Ultralytics logs above."
        )

    # Use shutil.copy (not copy2) — copy2 tries to copy file timestamps which
    # raises PermissionError on NFS mounts that don't support utime().
    shutil.copy(src, output_path)
    logger.info("[YOLO-PD] Fine-tuned model saved to: %s", output_path)
    return output_path


# ============================================================
# Cleanup
# ============================================================


def cleanup_yolo():
    """Free the cached YOLO model from GPU memory.

    Called by ``gpu_manager.cleanup_gpu()`` after the GPU semaphore is
    released, so the next worker starts with a clean GPU.
    """
    global _yolo_model, _yolo_model_path
    if _yolo_model is not None:
        try:
            del _yolo_model
        except Exception:
            pass
        _yolo_model = None
        _yolo_model_path = None
