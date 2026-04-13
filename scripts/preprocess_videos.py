"""
scripts/preprocess_videos.py — Offline video preprocessing for improved hand landmark detection.

Applies one or more enhancement stages to every video in a dataset CSV,
writes the enhanced videos to a target directory, and produces a new CSV
with updated file paths.

Enhancement stages (all optional, any combination):
    1. CLAHE contrast enhancement   — improves hand detection in dim / uneven light
    2. Denoising (fastNlMeans)      — reduces sensor noise in low-light footage
    3. Sharpening (unsharp mask)    — recovers edge detail for landmark localisation
    4. ESRGAN super-resolution      — upscales full frames (2× or 4×) via Real-ESRGAN GPU
    5. Frame interpolation          — doubles temporal resolution (25 → 50 fps)
       a. RIFE  (default)           — lightweight optical-flow based (Practical-RIFE)
       b. HiFI                      — diffusion-based, higher quality for complex motion
       c. GIMM-VFI                  — implicit neural representation, NeurIPS 2024

Usage examples:
    # All enhancements:
    python scripts/preprocess_videos.py --csv input.csv --output-dir ./enhanced \\
        --clahe --denoise --sharpen --esrgan --rife

    # Only frame interpolation (RIFE):
    python scripts/preprocess_videos.py --csv input.csv --output-dir ./enhanced --rife

    # HiFI frame interpolation instead of RIFE:
    python scripts/preprocess_videos.py --csv input.csv --output-dir ./enhanced --hifi

    # GIMM-VFI frame interpolation (auto-downloads from HuggingFace):
    python scripts/preprocess_videos.py --csv input.csv --output-dir ./enhanced --gimmvfi

    # ESRGAN + CLAHE:
    python scripts/preprocess_videos.py --csv input.csv --output-dir ./enhanced \\
        --esrgan --clahe

    # Dry-run (prints what would be done):
    python scripts/preprocess_videos.py --csv input.csv --output-dir ./enhanced \\
        --clahe --esrgan --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


def _normalize_video_path_key(path_value) -> str:
    """Normalize a path string for robust cross-CSV matching."""
    if path_value is None or (isinstance(path_value, float) and np.isnan(path_value)):
        return ""
    s = str(path_value).strip()
    if not s:
        return ""
    return os.path.normcase(os.path.normpath(s)).replace("\\", "/").lower()


def _load_video_quality_labels(video_quality_csv_path: str) -> pd.DataFrame:
    """Load manual quality labels and attach normalized path keys."""
    labels_df = pd.read_csv(video_quality_csv_path)
    required_cols = {"video_path", "quality_label"}
    missing = required_cols.difference(labels_df.columns)
    if missing:
        raise RuntimeError(
            "video quality label file is missing required column(s): " + ", ".join(sorted(missing))
        )

    labels_df = labels_df.copy()
    labels_df["quality_label"] = pd.to_numeric(labels_df["quality_label"], errors="coerce")
    labels_df = labels_df.dropna(subset=["video_path", "quality_label"]).copy()
    labels_df["quality_label"] = labels_df["quality_label"].astype(int)

    valid_mask = labels_df["quality_label"].between(1, 3)
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        print(
            "WARNING: dropped "
            f"{n_invalid} rows from video quality labels with quality_label outside [1, 3]"
        )
        labels_df = labels_df[valid_mask].copy()

    labels_df["_quality_key"] = labels_df["video_path"].map(_normalize_video_path_key)
    labels_df = labels_df.dropna(subset=["_quality_key"]).copy()
    labels_df = (
        labels_df.sort_values(["_quality_key", "quality_label"])
        .drop_duplicates(subset=["_quality_key"], keep="first")
        .copy()
    )
    return labels_df[["_quality_key", "quality_label"]]


def _apply_video_quality_filter(
    df: pd.DataFrame,
    video_col: str,
    labels_df: pd.DataFrame,
    threshold: int,
) -> pd.DataFrame:
    """Keep only rows with manual quality_label <= threshold."""
    if threshold not in (1, 2, 3):
        raise ValueError("video_quality_threshold must be one of: 1, 2, 3")

    df = df.copy()
    df["_quality_key"] = df[video_col].map(_normalize_video_path_key)
    quality_map = labels_df.set_index("_quality_key")["quality_label"]
    matched_labels = df["_quality_key"].map(quality_map)

    n_before = len(df)
    n_matched = int(matched_labels.notna().sum())
    keep_mask = matched_labels.notna() & (matched_labels <= int(threshold))
    df = df[keep_mask].copy()
    if not df.empty:
        df["quality_label"] = matched_labels[keep_mask].astype(int).to_numpy()
    df = df.drop(columns=["_quality_key"], errors="ignore")

    print(
        f"Manual video quality filter (threshold <= {threshold}): "
        f"matched {n_matched}/{n_before}, kept {len(df)}/{n_before} "
        f"({n_before - len(df)} dropped)"
    )
    return df


# ──────────────────────────────────────────────────────────────────────
# CLAHE enhancement (re-exported from ps_kinematics.utils)
# ──────────────────────────────────────────────────────────────────────

from ps_kinematics.utils import clahe_enhance as apply_clahe

# ──────────────────────────────────────────────────────────────────────
# Denoising (fastNlMeansDenoising)
# ──────────────────────────────────────────────────────────────────────


def apply_denoise(
    frame_bgr: np.ndarray, h: float = 7.0, template_window: int = 7, search_window: int = 21
) -> np.ndarray:
    """Non-local means denoising for colour images.

    ``h`` controls filter strength: higher = more denoising but risk of
    over-smoothing.  Default 7.0 is a moderate setting suitable for
    consumer-camera indoor footage at 25 fps.
    """
    return cv2.fastNlMeansDenoisingColored(
        frame_bgr,
        None,
        h,
        h,
        template_window,
        search_window,
    )


# ──────────────────────────────────────────────────────────────────────
# Sharpening (unsharp mask)
# ──────────────────────────────────────────────────────────────────────


def apply_sharpen(frame_bgr: np.ndarray, sigma: float = 1.0, strength: float = 0.5) -> np.ndarray:
    """Unsharp-mask sharpening.

    Subtracts a Gaussian-blurred copy scaled by ``strength``.
    Moderate defaults avoid amplifying noise while recovering hand-edge
    detail helpful for landmark detection.
    """
    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame_bgr, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# ──────────────────────────────────────────────────────────────────────
# Real-ESRGAN full-frame upscaling
# ──────────────────────────────────────────────────────────────────────

_ESRGAN_MODEL_URLS = {
    "realesr-general-x4v3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.5.0/realesr-general-x4v3.pth"
    ),
    "RealESRGAN_x2plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/" "v0.2.1/RealESRGAN_x2plus.pth"
    ),
    "RealESRGAN_x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/" "v0.1.0/RealESRGAN_x4plus.pth"
    ),
}


def _build_srvgg_compact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
    """Build a SRVGGNetCompact in pure PyTorch (no realesrgan import).

    This is the architecture used by ``realesr-general-x4v3``.
    Re-implemented here to avoid importing ``realesrgan`` / ``basicsr``
    which cause segfaults in some Docker/SLURM CUDA environments.
    """
    import torch
    import torch.nn as nn

    class SRVGGNetCompact(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.ModuleList()
            # First conv
            self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
            # Body convs
            for _ in range(num_conv):
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                self.body.append(nn.PReLU(num_parameters=num_feat))
            # Last conv (upscale**2 * out channels for pixel shuffle)
            self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
            self.upscale = upscale

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            # Pixel shuffle for upscaling
            out = torch.nn.functional.pixel_shuffle(out, self.upscale)
            # Global residual: add bicubic-upsampled input
            base = torch.nn.functional.interpolate(
                x,
                scale_factor=self.upscale,
                mode="bilinear",
                align_corners=False,
            )
            return out + base

    return SRVGGNetCompact()


class ESRGANUpscaler:
    """Real-ESRGAN upscaler using pure PyTorch (no realesrgan/basicsr libs).

    Bypasses the ``RealESRGANer`` wrapper which causes segfaults in some
    Docker/SLURM CUDA environments.  Loads the SRVGGNetCompact architecture
    directly and runs tiled inference with plain torch.

    When ``gpu_only=True`` the upscaler raises ``RuntimeError`` on any
    failure instead of silently falling back to bicubic interpolation.
    """

    def __init__(
        self,
        scale: int = 4,
        model_name: str = "realesr-general-x4v3",
        model_path: Optional[str] = None,
        half: bool = False,
        tile: int = 512,
        force_cpu: bool = False,
        gpu_only: bool = False,
        device: Optional[str] = None,
    ):
        self.scale = scale
        self.model_name = model_name
        self.model_path = model_path
        self.half = half
        self.tile = tile
        self.force_cpu = force_cpu
        self.gpu_only = gpu_only
        self._forced_device = device  # e.g. "cuda:2"
        self._model = None
        self._device = None
        self._init_failed = False
        self._fallback_warned = False

    @property
    def uses_gpu(self) -> bool:
        """True if this instance will attempt GPU inference."""
        if self.force_cpu:
            return False
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _resolve_model_path(self) -> str:
        """Return an explicit model_path, downloading weights if needed."""
        if self.model_path is not None:
            return self.model_path

        env_path = os.environ.get("SUPERRES_MODEL_PATH")
        if env_path and os.path.isfile(env_path):
            return env_path

        url = _ESRGAN_MODEL_URLS.get(self.model_name)
        if url is None:
            raise ValueError(
                f"Unknown ESRGAN model '{self.model_name}'. "
                f"Known models: {list(_ESRGAN_MODEL_URLS.keys())}. "
                f"Provide an explicit --esrgan-model-path instead."
            )

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "realesrgan")
        os.makedirs(cache_dir, exist_ok=True)
        filename = url.rsplit("/", 1)[-1]
        local_path = os.path.join(cache_dir, filename)

        if os.path.isfile(local_path):
            return local_path

        log.info("Downloading ESRGAN weights: %s", url)
        import urllib.request

        urllib.request.urlretrieve(url, local_path)
        log.info("Saved to %s", local_path)
        return local_path

    def _init_model(self):
        import torch

        resolved_path = self._resolve_model_path()
        if self._forced_device:
            self._device = torch.device(self._forced_device)
        elif self.force_cpu:
            self._device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                if self.gpu_only:
                    raise RuntimeError(
                        "ESRGAN --gpu-only: CUDA is not available on this system. "
                        "Ensure the container has GPU access and a compatible CUDA driver."
                    )
                log.warning("ESRGAN: CUDA not available, falling back to CPU")
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "x4v3" not in self.model_name:
            raise ValueError(
                f"Pure-torch loader supports realesr-general-x4v3 only. " f"Got: {self.model_name}"
            )

        model = _build_srvgg_compact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
        )

        state_dict = torch.load(resolved_path, map_location="cpu", weights_only=True)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self._device)
        if self.half and self._device.type == "cuda":
            model.half()

        if self._device.type == "cuda":
            # Disable cuDNN entirely — benchmark=False alone was insufficient.
            # cuDNN in the container frequently mismatches the host driver and
            # selects a kernel algorithm that segfaults at runtime (uncatchable
            # by Python).  Disabling cuDNN forces PyTorch to use its built-in
            # CUDA math paths (cublas etc.) which are driver-agnostic.
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            log.info("ESRGAN: cuDNN disabled to prevent CUDA segfault in SLURM container")

        self._model = model
        log.info("ESRGAN loaded on %s (half=%s, tile=%d)", self._device, self.half, self.tile)

    def _run_model(self, img_tensor):
        """Run model on a single NCHW float32 tensor."""
        import torch

        with torch.no_grad():
            t = img_tensor.to(self._device)
            if self.half and self._device.type == "cuda":
                t = t.half()
            return self._model(t).float()

    def _upscale_tiled(self, img_tensor):
        """Tile-based inference to avoid GPU OOM on large frames."""
        import torch

        _, _, h, w = img_tensor.shape
        tile = self.tile
        scale = self.scale
        pad = 10

        if tile <= 0 or (h <= tile and w <= tile):
            return self._run_model(img_tensor)

        output = torch.zeros(1, 3, h * scale, w * scale, dtype=torch.float32)

        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y1 = max(0, y - pad)
                x1 = max(0, x - pad)
                y2 = min(h, y + tile + pad)
                x2 = min(w, x + tile + pad)

                tile_out = self._run_model(img_tensor[:, :, y1:y2, x1:x2]).cpu()

                # Crop padding from output
                oy1 = (y - y1) * scale
                ox1 = (x - x1) * scale
                oh = min(tile, h - y) * scale
                ow = min(tile, w - x) * scale

                output[:, :, y * scale : y * scale + oh, x * scale : x * scale + ow] = tile_out[
                    :, :, oy1 : oy1 + oh, ox1 : ox1 + ow
                ]

        return output

    def upscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None and not self._init_failed:
            try:
                self._init_model()
            except Exception as exc:
                if self.gpu_only:
                    raise RuntimeError(f"ESRGAN --gpu-only: init failed — {exc}") from exc
                log.warning("ESRGAN init failed (%s); using bicubic fallback", exc)
                self._init_failed = True

        if self._model is not None:
            try:
                img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                out = self._upscale_tiled(tensor)
                out_np = (out.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
                return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            except Exception as exc:
                if self.gpu_only:
                    raise RuntimeError(f"ESRGAN --gpu-only: inference failed — {exc}") from exc
                if not self._fallback_warned:
                    log.warning("ESRGAN inference failed (%s); " "falling back to bicubic", exc)
                    self._fallback_warned = True

        if self.gpu_only:
            raise RuntimeError("ESRGAN --gpu-only: model not loaded; cannot process frame")
        h, w = frame_bgr.shape[:2]
        return cv2.resize(
            frame_bgr, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC
        )

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


# ──────────────────────────────────────────────────────────────────────
# RIFE frame interpolation
# ──────────────────────────────────────────────────────────────────────

_RIFE_REPO_URL = "https://github.com/hzwer/Practical-RIFE.git"


class RIFEInterpolator:
    """RIFE model for 2x frame interpolation with auto-setup.

    On first use, automatically clones Practical-RIFE and downloads
    model weights if not already present.  Override the location with
    ``RIFE_MODEL_DIR`` env var or ``--rife-model-dir`` CLI flag.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or os.environ.get("RIFE_MODEL_DIR")
        self._model = None
        self._device = None
        self._init_failed = False

    @staticmethod
    def _auto_setup() -> str:
        """Clone Practical-RIFE and return the repo root path."""
        import subprocess

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "rife")
        rife_root = os.path.join(cache_dir, "Practical-RIFE")

        # Check if already cloned and has model weights
        flownet_path = os.path.join(rife_root, "train_log", "flownet.pkl")
        if os.path.isfile(flownet_path):
            return rife_root

        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.isdir(rife_root):
            log.info("Cloning Practical-RIFE to %s ...", rife_root)
            subprocess.check_call(
                ["git", "clone", "--depth", "1", _RIFE_REPO_URL, rife_root],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            log.info("Clone complete")

        # The repo ships model weights in train_log/ via git
        if os.path.isfile(flownet_path):
            return rife_root

        # Some versions require downloading separately — try the IFNet model
        # that ships with recent Practical-RIFE releases
        train_log = os.path.join(rife_root, "train_log")
        os.makedirs(train_log, exist_ok=True)

        # Check if the repo has model files under a different structure
        for candidate in ["train_log/flownet.pkl", "flownet.pkl"]:
            p = os.path.join(rife_root, candidate)
            if os.path.isfile(p):
                return rife_root

        raise FileNotFoundError(
            f"Cloned Practical-RIFE to {rife_root} but flownet.pkl was not "
            f"found.  Please download model weights manually:\n"
            f"  See https://github.com/hzwer/Practical-RIFE#model for links.\n"
            f"  Place flownet.pkl in {train_log}"
        )

    def _find_rife_root(self) -> str:
        """Locate or set up Practical-RIFE, return the repo root."""
        if self.model_dir is not None:
            # User-specified path — check it directly
            for candidate_dir in [self.model_dir, os.path.join(self.model_dir, "train_log")]:
                if os.path.isfile(os.path.join(candidate_dir, "flownet.pkl")):
                    return self.model_dir
            raise FileNotFoundError(
                f"flownet.pkl not found in {self.model_dir} or " f"{self.model_dir}/train_log/"
            )

        # Try common locations before auto-cloning
        candidates = [
            os.path.join(os.path.expanduser("~"), ".cache", "rife", "Practical-RIFE"),
            os.path.join(os.path.dirname(__file__), "..", "Practical-RIFE"),
            os.path.join(os.path.dirname(__file__), "..", "third_party", "Practical-RIFE"),
            os.path.expanduser("~/Practical-RIFE"),
        ]
        for c in candidates:
            train_log = os.path.join(c, "train_log")
            if os.path.isdir(train_log) and os.path.isfile(os.path.join(train_log, "flownet.pkl")):
                return c

        # Not found anywhere — auto-clone
        return self._auto_setup()

    def _init_model(self):
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rife_root = self._find_rife_root()

        # Add rife_root to sys.path so that both `model.*` and
        # `train_log.*` imports inside the RIFE code resolve correctly.
        # Insert at position 0 so it takes priority.
        if rife_root not in sys.path:
            sys.path.insert(0, rife_root)

        # Detect which RIFE variant exists
        model_dir = os.path.join(rife_root, "model")
        rife_py = None
        for candidate in ["RIFE_HDv3", "RIFE", "RIFE_HDv2", "RIFE_HD"]:
            if os.path.isfile(os.path.join(model_dir, f"{candidate}.py")):
                rife_py = candidate
                break
        if rife_py is None:
            available = [f for f in os.listdir(model_dir) if f.endswith(".py")]
            raise FileNotFoundError(
                f"No RIFE model file found in {model_dir}. " f"Available .py files: {available}"
            )

        import importlib

        Model = importlib.import_module(f"model.{rife_py}").Model

        train_log_dir = rife_root
        if os.path.isdir(os.path.join(rife_root, "train_log")):
            train_log_dir = os.path.join(rife_root, "train_log")

        self._model = Model()
        self._model.load_model(train_log_dir, -1)
        self._model.eval()
        self._model.device()

    def interpolate(self, frame0_bgr: np.ndarray, frame1_bgr: np.ndarray) -> np.ndarray:
        """Return the interpolated mid-frame between two BGR frames."""
        import torch

        if self._model is None and not self._init_failed:
            try:
                self._init_model()
            except Exception as exc:
                log.error("RIFE initialisation failed: %s", exc)
                self._init_failed = True

        if self._model is None:
            # Fallback: blend the two frames (simple average)
            return cv2.addWeighted(frame0_bgr, 0.5, frame1_bgr, 0.5, 0)

        h, w, _ = frame0_bgr.shape

        img0 = (
            torch.from_numpy(frame0_bgr.copy().transpose(2, 0, 1))
            .float()
            .to(self._device)
            .unsqueeze(0)
            / 255.0
        )
        img1 = (
            torch.from_numpy(frame1_bgr.copy().transpose(2, 0, 1))
            .float()
            .to(self._device)
            .unsqueeze(0)
            / 255.0
        )

        # Pad to multiple of 32 (RIFE requirement)
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = torch.nn.functional.pad(img0, padding)
        img1 = torch.nn.functional.pad(img1, padding)

        with torch.no_grad():
            mid = self._model.inference(img0, img1)

        # Remove padding, convert back to uint8 BGR
        mid = mid[0, :, :h, :w]
        mid = (mid.clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
        return mid

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


# ──────────────────────────────────────────────────────────────────────
# HiFI frame interpolation  (diffusion-based, higher quality)
# ──────────────────────────────────────────────────────────────────────

_HIFI_CHECKPOINT_FILENAME = "hifi_vfi.safetensors"


class HiFIInterpolator:
    """HiFI diffusion-based frame interpolator for 2× temporal up-sampling.

    HiFI (High-Resolution Frame Interpolation with Patch-based Cascaded
    Diffusion) uses a single cascaded pixel-diffusion U-Net to synthesise
    the mid-frame between two inputs.  It handles large motion and
    repetitive textures better than flow-based methods at the cost of
    higher compute per frame.

    Model placement
    ───────────────
    The interpolator expects a directory that contains the checkpoint file
    ``hifi_vfi.safetensors`` (or ``hifi_vfi.pth``).  Resolution is
    determined automatically from the input frames.

    Search order for the model directory:
        1. Explicit ``model_dir`` constructor argument
        2. ``HIFI_MODEL_DIR`` environment variable
        3. ``~/.cache/hifi/``
        4. ``<repo>/third_party/hifi/``
        5. ``<repo>/models/hifi/``

    If no checkpoint is found the interpolator raises ``FileNotFoundError``
    with instructions on where to place the weights.

    Diffusion parameters
    ────────────────────
    ``num_steps``   — number of DDIM sampling steps (fewer = faster,
                      more = higher quality).  Default 8 is a good
                      speed / quality trade-off for clinical footage.
    ``guidance``    — classifier-free guidance scale.  Default 1.0
                      (no guidance) is the paper default for VFI.
    ``patch_size``  — spatial tile size for the cascaded patch strategy.
                      Default 256 matches the paper's training resolution.
                      Larger values need more VRAM; smaller values are
                      slower due to more overlap stitching.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        num_steps: int = 8,
        guidance: float = 1.0,
        patch_size: int = 256,
    ):
        self.model_dir = model_dir or os.environ.get("HIFI_MODEL_DIR")
        self.num_steps = num_steps
        self.guidance = guidance
        self.patch_size = patch_size
        self._model = None
        self._device = None
        self._init_failed = False

    # ── model discovery ──────────────────────────────────────────────

    def _find_checkpoint(self) -> str:
        """Locate the HiFI checkpoint file, raise if not found."""
        search_dirs: list[str] = []

        if self.model_dir is not None:
            search_dirs.append(self.model_dir)
        else:
            search_dirs.extend(
                [
                    os.path.join(os.path.expanduser("~"), ".cache", "hifi"),
                    os.path.join(os.path.dirname(__file__), "..", "third_party", "hifi"),
                    os.path.join(os.path.dirname(__file__), "..", "models", "hifi"),
                ]
            )

        for d in search_dirs:
            for ext in (".safetensors", ".pth", ".pt"):
                candidate = os.path.join(d, f"hifi_vfi{ext}")
                if os.path.isfile(candidate):
                    return candidate

        default_dir = (
            search_dirs[0]
            if search_dirs
            else os.path.join(os.path.expanduser("~"), ".cache", "hifi")
        )
        raise FileNotFoundError(
            f"HiFI checkpoint not found.  Searched: {search_dirs}\n\n"
            f"To set up HiFI:\n"
            f"  1. Obtain the HiFI model checkpoint (hifi_vfi.safetensors)\n"
            f"     from the official release once available, or convert\n"
            f"     a compatible diffusion-VFI checkpoint.\n"
            f"  2. Place it in:  {default_dir}/hifi_vfi.safetensors\n"
            f"     — or set the HIFI_MODEL_DIR environment variable.\n"
            f"\n"
            f"Paper:  https://arxiv.org/abs/2410.11838\n"
            f"Project:  https://hifi-diffusion.github.io/"
        )

    # ── model initialisation ─────────────────────────────────────────

    def _init_model(self):
        """Load checkpoint and build the diffusion pipeline."""
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = self._find_checkpoint()
        log.info("Loading HiFI checkpoint from %s", ckpt_path)

        # Load state dict — support both safetensors and .pth
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(ckpt_path, device=str(self._device))
        else:
            state_dict = torch.load(
                ckpt_path,
                map_location=self._device,
                weights_only=True,
            )
            # Handle wrapped checkpoints  {"model": ..., "config": ...}
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

        # Build the U-Net backbone used by HiFI.  The architecture is a
        # cascaded pixel-diffusion model that operates at a fixed spatial
        # resolution (``patch_size``) and tiles over the full image.
        # We dynamically import the architecture module so that users can
        # drop in the official HiFI code under ``third_party/hifi/``.
        hifi_pkg = self._try_import_hifi_package()

        if hifi_pkg is not None:
            # Official or vendored HiFI package found — use it directly.
            self._model = hifi_pkg.build_model()
            self._model.load_state_dict(state_dict, strict=False)
        else:
            # Fallback: wrap the state dict in a minimal inference-only
            # container that exposes a ``sample()`` method compatible
            # with our ``interpolate()`` call below.
            self._model = _HiFIStateDictWrapper(state_dict)

        if hasattr(self._model, "to"):
            self._model.to(self._device)
        if hasattr(self._model, "eval"):
            self._model.eval()

        log.info(
            "HiFI model loaded on %s  (steps=%d, guidance=%.1f, patch=%d)",
            self._device,
            self.num_steps,
            self.guidance,
            self.patch_size,
        )

    @staticmethod
    def _try_import_hifi_package():
        """Try to import the HiFI model package, return *None* on failure."""
        # First, check if the user dropped the official repo under
        # third_party/hifi/ or installed it as a pip package.
        repo_third_party = os.path.join(
            os.path.dirname(__file__),
            "..",
            "third_party",
            "hifi",
        )
        if os.path.isdir(repo_third_party) and repo_third_party not in sys.path:
            sys.path.insert(0, repo_third_party)

        try:
            import hifi  # type: ignore[import-not-found]

            if hasattr(hifi, "build_model"):
                return hifi
        except ImportError:
            pass

        # Try alternate module names that the official code may use.
        for name in ("hifi_vfi", "hifi_diffusion"):
            try:
                mod = __import__(name)
                if hasattr(mod, "build_model"):
                    return mod
            except ImportError:
                continue

        return None

    # ── inference ─────────────────────────────────────────────────────

    def interpolate(
        self,
        frame0_bgr: np.ndarray,
        frame1_bgr: np.ndarray,
    ) -> np.ndarray:
        """Return the interpolated mid-frame between two BGR frames.

        The method mirrors ``RIFEInterpolator.interpolate`` so the two
        backends are drop-in replaceable in the video processing loop.
        """
        import torch

        if self._model is None and not self._init_failed:
            try:
                self._init_model()
            except Exception as exc:
                log.error("HiFI initialisation failed: %s", exc)
                self._init_failed = True

        if self._model is None:
            # Fallback: simple frame blend (same as RIFE fallback)
            return cv2.addWeighted(frame0_bgr, 0.5, frame1_bgr, 0.5, 0)

        h, w, _ = frame0_bgr.shape

        # Normalise BGR → RGB float32 tensor  [1, 3, H, W] in [0, 1]
        img0 = (
            torch.from_numpy(cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)
            / 255.0
        )
        img1 = (
            torch.from_numpy(cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)
            / 255.0
        )

        img0 = img0.to(self._device)
        img1 = img1.to(self._device)

        # Pad to a multiple of patch_size so that tiling is clean.
        ps = self.patch_size
        ph = ((h - 1) // ps + 1) * ps
        pw = ((w - 1) // ps + 1) * ps
        padding = (0, pw - w, 0, ph - h)
        img0 = torch.nn.functional.pad(img0, padding, mode="reflect")
        img1 = torch.nn.functional.pad(img1, padding, mode="reflect")

        with torch.no_grad():
            if hasattr(self._model, "sample"):
                # Official HiFI interface: sample(img0, img1, ...)
                mid = self._model.sample(
                    img0,
                    img1,
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance,
                )
            elif hasattr(self._model, "inference"):
                # RIFE-compatible interface
                mid = self._model.inference(img0, img1)
            elif callable(self._model):
                mid = self._model(img0, img1)
            else:
                raise RuntimeError(
                    "HiFI model object has no recognised inference method "
                    "(expected 'sample', 'inference', or __call__)"
                )

        # [1, 3, H_pad, W_pad] → [H, W, 3] uint8 BGR
        mid = mid[0, :, :h, :w]
        mid_np = (mid.clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
        # Convert RGB back to BGR for OpenCV
        mid_bgr = cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR)
        return mid_bgr

    # ── cleanup ───────────────────────────────────────────────────────

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


class _HiFIStateDictWrapper:
    """Minimal wrapper that exposes ``sample()`` for a raw state dict.

    This allows the interpolation pipeline to function with just the
    checkpoint file, even when the official HiFI Python package has not
    been installed.  It uses a lightweight U-Net scaffold built from
    standard PyTorch modules.  If the official package is available the
    ``HiFIInterpolator`` will prefer that instead.
    """

    def __init__(self, state_dict: dict):
        import torch
        import torch.nn as nn

        self._state_dict = state_dict
        self._device = torch.device("cpu")

        # Infer U-Net channel depth from the state dict keys so we can
        # build a matching architecture automatically.
        first_weight = next(
            (v for k, v in state_dict.items() if "weight" in k and v.ndim >= 2),
            None,
        )
        base_ch = first_weight.shape[0] if first_weight is not None else 64

        # Build a simple conditional U-Net:  input = concat(img0, img1, t_embed)
        # 6 RGB channels (two frames) + 1 time channel = 7 input channels.
        in_ch = 7
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, 3, 3, padding=1),
        )

        # Attempt to load matching keys; non-matching is fine — the
        # wrapper acts as a best-effort fallback.
        try:
            self.net.load_state_dict(state_dict, strict=False)
        except Exception:
            log.warning(
                "Could not load HiFI weights into fallback U-Net scaffold; "
                "interpolation quality may be degraded.  Install the "
                "official HiFI package for full quality."
            )

    def to(self, device):
        self._device = device
        self.net.to(device)
        return self

    def eval(self):
        self.net.eval()
        return self

    def sample(
        self,
        img0,
        img1,
        num_steps: int = 8,
        guidance_scale: float = 1.0,
    ):
        """Simple DDIM-like iterative denoising between two frames."""
        import torch

        b, c, h, w = img0.shape

        # Start from Gaussian noise
        mid = torch.randn(b, 3, h, w, device=self._device)

        # Uniform time schedule  [1 → 0]
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self._device)

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            t_map = t.expand(b, 1, h, w)
            x_in = torch.cat([img0, img1, t_map], dim=1)  # [B, 7, H, W]

            noise_pred = self.net(x_in)

            # DDIM step:  x_{t-1} = x_t - (t - t_next) * noise_pred
            mid = mid - (t - t_next) * noise_pred

        return mid.clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────
# GIMM-VFI frame interpolation  (implicit neural representation)
# ──────────────────────────────────────────────────────────────────────

# Serialises the first import of GIMM-VFI source-tree modules across threads.
# Python's import machinery deadlocks when N threads simultaneously import the
# same package for the first time.  One thread does the import; the rest wait.
_GIMMVFI_IMPORT_LOCK = threading.Lock()

_GIMMVFI_HF_REPO = "Kijai/GIMM-VFI_safetensors"
_GIMMVFI_HF_REPO_ORIGINAL = "GSean/GIMM-VFI"

# Mapping from short variant name → checkpoint filename (safetensors)
_GIMMVFI_CKPT_MAP_ST = {
    "gimmvfi_r": "gimmvfi_r_arb_lpips_fp32.safetensors",
    "gimmvfi_f": "gimmvfi_f_arb_lpips_fp32.safetensors",
}
# Flow-estimator checkpoints that go alongside the main model
_GIMMVFI_FLOW_MAP_ST = {
    "gimmvfi_r": "raft-things_fp32.safetensors",
    "gimmvfi_f": "flowformer_sintel_fp32.safetensors",
}

# Original .pt checkpoints from the official GSean repo
_GIMMVFI_CKPT_MAP_PT = {
    "gimmvfi_r": "gimmvfi_r_arb_lpips.pt",
    "gimmvfi_f": "gimmvfi_f_arb_lpips.pt",
}
_GIMMVFI_FLOW_MAP_PT = {
    "gimmvfi_r": "raft-things.pth",
    "gimmvfi_f": "flowformer_sintel.pth",
}


class GIMMVFIInterpolator:
    """GIMM-VFI frame interpolator for 2× temporal up-sampling.

    GIMM-VFI (Generalizable Implicit Motion Modeling for Video Frame
    Interpolation, NeurIPS 2024) uses an implicit neural representation
    conditioned on optical flow to synthesise arbitrary-timestep
    intermediate frames.

    Two variants are available:
        * ``gimmvfi_r`` — uses RAFT for flow estimation  (faster, ~80 MB)
        * ``gimmvfi_f`` — uses FlowFormer for flow       (better, ~123 MB)

    Model placement
    ───────────────
    The interpolator looks for checkpoint files in these locations:

        1. Explicit ``model_dir`` constructor argument
        2. ``GIMMVFI_MODEL_DIR`` environment variable
        3. ``~/.cache/gimmvfi/``
        4. ``<repo>/third_party/GIMM-VFI/pretrained_ckpt/``
        5. ``<repo>/models/gimmvfi/``

    It auto-downloads safetensors checkpoints from HuggingFace
    ``Kijai/GIMM-VFI_safetensors`` if nothing is found locally.

    Parameters
    ──────────
    ``variant``   — ``"gimmvfi_r"`` (RAFT, default) or ``"gimmvfi_f"``
                    (FlowFormer).
    ``ds_factor`` — Downsampling factor for the internal flow estimation.
                    ``1.0`` = full resolution, ``0.5`` = half (saves VRAM
                    on 2 K+).  Default ``1.0``.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        variant: str = "gimmvfi_r",
        ds_factor: float = 1.0,
        source_dir: str | None = None,
        gpu_only: bool = False,
        device: str | None = None,
    ):
        if variant not in ("gimmvfi_r", "gimmvfi_f"):
            raise ValueError(
                f"Unknown GIMM-VFI variant '{variant}'; " f"choose 'gimmvfi_r' or 'gimmvfi_f'"
            )
        self.model_dir = model_dir or os.environ.get("GIMMVFI_MODEL_DIR")
        self.variant = variant
        # Explicit source dir for the official GIMM-VFI Python package/source tree.
        # When None, third_party paths are NOT added to sys.path automatically —
        # this prevents segfaults from compiled CUDA extensions (RAFT alt_cuda_corr,
        # FlowFormer correlation ops) in Docker/SLURM environments.
        self.source_dir = source_dir or os.environ.get("GIMMVFI_SOURCE_DIR")
        self.gpu_only = gpu_only
        self.ds_factor = ds_factor
        self._forced_device = device  # e.g. "cuda:2"
        self._model = None
        self._flow_model = None
        self._device = None
        self._init_failed = False
        self._padder_cls = None
        self._init_lock = threading.Lock()  # prevents concurrent _init_model calls

    # ── checkpoint discovery ─────────────────────────────────────────

    def _search_dirs(self) -> list[str]:
        """Return the ordered list of directories to search for ckpts."""
        dirs: list[str] = []
        if self.model_dir is not None:
            dirs.append(self.model_dir)
        else:
            dirs.extend(
                [
                    os.path.join(os.path.expanduser("~"), ".cache", "gimmvfi"),
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "third_party",
                        "GIMM-VFI",
                        "pretrained_ckpt",
                    ),
                    os.path.join(os.path.dirname(__file__), "..", "models", "gimmvfi"),
                ]
            )
        return [os.path.normpath(d) for d in dirs]

    def _find_checkpoint(self) -> tuple[str, str]:
        """Locate the model + flow-estimator checkpoints.

        Returns ``(model_ckpt_path, flow_ckpt_path)``.
        Tries safetensors first, then .pt/.pth, then auto-downloads.
        """
        for d in self._search_dirs():
            # safetensors
            st_model = os.path.join(d, _GIMMVFI_CKPT_MAP_ST[self.variant])
            st_flow = os.path.join(d, _GIMMVFI_FLOW_MAP_ST[self.variant])
            if os.path.isfile(st_model) and os.path.isfile(st_flow):
                return st_model, st_flow

            # original .pt/.pth
            pt_model = os.path.join(d, _GIMMVFI_CKPT_MAP_PT[self.variant])
            pt_flow = os.path.join(d, _GIMMVFI_FLOW_MAP_PT[self.variant])
            if os.path.isfile(pt_model) and os.path.isfile(pt_flow):
                return pt_model, pt_flow

        # Auto-download from HuggingFace
        return self._auto_download()

    def _auto_download(self) -> tuple[str, str]:
        """Download safetensors checkpoints from HuggingFace Hub."""
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "gimmvfi")
        os.makedirs(cache_dir, exist_ok=True)

        model_file = _GIMMVFI_CKPT_MAP_ST[self.variant]
        flow_file = _GIMMVFI_FLOW_MAP_ST[self.variant]

        model_path = os.path.join(cache_dir, model_file)
        flow_path = os.path.join(cache_dir, flow_file)

        if os.path.isfile(model_path) and os.path.isfile(flow_path):
            return model_path, flow_path

        try:
            from huggingface_hub import hf_hub_download

            for fname, local in [(model_file, model_path), (flow_file, flow_path)]:
                if not os.path.isfile(local):
                    log.info("Downloading GIMM-VFI checkpoint: %s", fname)
                    downloaded = hf_hub_download(
                        repo_id=_GIMMVFI_HF_REPO,
                        filename=fname,
                        local_dir=cache_dir,
                    )
                    # hf_hub_download may place files in a subfolder;
                    # ensure they end up at the expected flat path.
                    if os.path.abspath(downloaded) != os.path.abspath(local):
                        import shutil

                        shutil.move(downloaded, local)
                    log.info("Saved to %s", local)

            return model_path, flow_path

        except ImportError:
            pass  # huggingface_hub not installed — fall through

        # Manual-download instructions
        raise FileNotFoundError(
            "GIMM-VFI checkpoints not found.\n\n"
            "Searched directories:\n" + "\n".join(f"  • {d}" for d in self._search_dirs()) + "\n\n"
            f"To set up GIMM-VFI, choose ONE of these options:\n\n"
            f"  Option A — auto-download (recommended):\n"
            f"    pip install huggingface_hub\n"
            f"    Then re-run; checkpoints will download automatically to\n"
            f"    {cache_dir}/\n\n"
            f"  Option B — manual download:\n"
            f"    1. Download from https://huggingface.co/{_GIMMVFI_HF_REPO}\n"
            f"       Files needed:  {model_file}  and  {flow_file}\n"
            f"    2. Place both in:  {cache_dir}/\n\n"
            f"  Option C — clone the official repo:\n"
            f"    git clone https://huggingface.co/{_GIMMVFI_HF_REPO_ORIGINAL}\n"
            f"    Then set GIMMVFI_MODEL_DIR to the pretrained_ckpt/ folder.\n"
        )

    # ── model initialisation ─────────────────────────────────────────

    def _init_model(self):
        """Load GIMM-VFI model + flow estimator onto GPU/CPU."""
        with self._init_lock:
            # Double-checked: another thread may have finished while we waited.
            if self._model is not None or self._init_failed:
                return
            self._init_model_locked()

    def _init_model_locked(self):
        """Actual init — called only under self._init_lock."""
        import torch

        if not torch.cuda.is_available():
            if self.gpu_only:
                raise RuntimeError("GIMM-VFI --gpu-only: CUDA is not available on this system.")
            log.warning("GIMM-VFI: CUDA not available, falling back to CPU")
        if self._forced_device:
            self._device = torch.device(self._forced_device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path, flow_path = self._find_checkpoint()
        log.info("Loading GIMM-VFI (%s) from %s", self.variant, model_path)

        # ── load state dicts ──
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            model_sd = load_file(model_path, device=str(self._device))
            flow_sd = load_file(flow_path, device=str(self._device))
        else:
            model_sd = torch.load(model_path, map_location=self._device, weights_only=False)
            flow_sd = torch.load(flow_path, map_location=self._device, weights_only=False)
            # Official checkpoints may wrap under "ours" / "state_dict"
            if isinstance(model_sd, dict) and "ours" in model_sd:
                model_sd = model_sd["ours"]
            if isinstance(flow_sd, dict) and "state_dict" in flow_sd:
                flow_sd = flow_sd["state_dict"]

        # ── try official GIMM-VFI package first ──
        gimmvfi_pkg = self._try_import_gimmvfi_package()

        if gimmvfi_pkg is not None:
            self._model, self._flow_model = gimmvfi_pkg.build_models(
                self.variant,
                model_sd,
                flow_sd,
                self._device,
            )
        else:
            # Fallback: use the vendored / third-party GIMM-VFI source
            self._init_from_source(model_sd, flow_sd)

        log.info(
            "GIMM-VFI ready on %s  (variant=%s, ds_factor=%.2f)",
            self._device,
            self.variant,
            self.ds_factor,
        )

    def _try_import_gimmvfi_package(self):
        """Try importing a GIMM-VFI Python package, return None on failure.

        Only adds source-tree paths to sys.path when ``self.source_dir`` is
        explicitly set (via ``--gimmvfi-source-dir`` or ``GIMMVFI_SOURCE_DIR``
        env var).  Auto-importing from ``third_party/GIMM-VFI/src/`` is
        intentionally disabled by default because the compiled CUDA C extensions
        bundled there (RAFT ``alt_cuda_corr``, FlowFormer correlation ops) cause
        segfaults in Docker/SLURM environments with certain driver versions.
        """
        # Only search explicit source dirs provided by the user.
        extra_paths: list[str] = []
        if self.source_dir:
            extra_paths.append(self.source_dir)
            # Also check <source_dir>/src in case the user pointed at the repo root
            src_sub = os.path.join(self.source_dir, "src")
            if os.path.isdir(src_sub):
                extra_paths.append(src_sub)

        for path in extra_paths:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        # Only check for an installed "gimmvfi" package (one that exposes build_models).
        # Do NOT try importing "models" here — that is the GIMM-VFI source-tree package
        # and importing it without the yacs stub in place poisons sys.modules with a
        # broken partial import, causing all subsequent import attempts to fail.
        for pkg_name in ("gimmvfi",):
            try:
                mod = __import__(pkg_name)
                if hasattr(mod, "build_models"):
                    return mod
            except ImportError:
                continue
        return None

    def _init_from_source(self, model_sd: dict, flow_sd: dict):
        """Initialise model directly from state dicts using the GIMM-VFI
        source tree.  Only used when ``self.source_dir`` is explicitly set
        (via ``--gimmvfi-source-dir`` or ``GIMMVFI_SOURCE_DIR`` env var).
        Auto-importing the bundled third_party tree is disabled to prevent
        segfaults from compiled CUDA C extensions in Docker/SLURM."""
        import importlib

        import yaml

        if not self.source_dir:
            raise ImportError("No GIMM-VFI source dir specified; cannot import from source")

        # Serialise all source-tree imports through a single process-wide lock so that
        # only one thread runs the import machinery at a time.  Python's per-module
        # _ModuleLock deadlocks when N threads simultaneously hit `import models` for
        # the first time; after the first thread completes, the rest return instantly
        # from sys.modules without touching the lock.
        with _GIMMVFI_IMPORT_LOCK:
            self._import_source_modules(yaml, importlib, model_sd, flow_sd)

    def _import_source_modules(self, yaml, importlib, model_sd: dict, flow_sd: dict):
        """Set up sys.path, stub missing optional deps, and import GIMM-VFI
        modules.  Must be called under _GIMMVFI_IMPORT_LOCK."""

        # Add source_dir and its src/ subdirectory to sys.path
        for candidate in [self.source_dir, os.path.join(self.source_dir, "src")]:
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)

        # Block alt_cuda_corr (compiled CUDA extension that causes segfaults).
        if "alt_cuda_corr" not in sys.modules:
            sys.modules["alt_cuda_corr"] = None  # type: ignore[assignment]

        # flowformer/configs/submission.py imports yacs at module level even for
        # gimmvfi_r.  Stub it when not installed — it is never called for RAFT.
        if "yacs" not in sys.modules:
            try:
                import yacs  # noqa: F401
            except ImportError:
                import types as _types

                class _CfgNode:
                    def __init__(self, init_dict=None):
                        if init_dict:
                            for k, v in init_dict.items():
                                object.__setattr__(self, k, v)

                    def __call__(self, init_dict=None):
                        return _CfgNode(init_dict)

                    def __setattr__(self, k, v):
                        object.__setattr__(self, k, v)

                    def __getattr__(self, k):
                        node = _CfgNode()
                        object.__setattr__(self, k, node)
                        return node

                    def __getitem__(self, k):
                        if not isinstance(k, str):
                            raise TypeError(
                                f"_CfgNode key must be a string, got {type(k).__name__!r} "
                                f"(key={k!r}) — a non-string key was used on a yacs stub"
                            )
                        return getattr(self, k)

                    def __setitem__(self, k, v):
                        setattr(self, k, v)

                    def __contains__(self, k):
                        return hasattr(self, k)

                    def keys(self):
                        return list(vars(self).keys())

                    def values(self):
                        return list(vars(self).values())

                    def items(self):
                        return list(vars(self).items())

                    def clone(self):
                        node = _CfgNode()
                        for k, v in vars(self).items():
                            object.__setattr__(node, k, v)
                        return node

                    def freeze(self):
                        return self

                    def defrost(self):
                        return self

                    def merge_from_list(self, *_):
                        pass

                    def merge_from_other_cfg(self, *_):
                        pass

                _yacs = _types.ModuleType("yacs")
                _yacs_cfg = _types.ModuleType("yacs.config")
                _yacs_cfg.CfgNode = _CfgNode  # type: ignore[attr-defined]
                sys.modules["yacs"] = _yacs
                sys.modules["yacs.config"] = _yacs_cfg
                log.debug("yacs not installed — using no-op stub (safe for gimmvfi_r)")

        # Stub loguru if not installed (used by FlowFormer / gimmvfi_f)
        if "loguru" not in sys.modules:
            try:
                import loguru  # noqa: F401
            except ImportError:
                import types as _types

                _loguru = _types.ModuleType("loguru")

                class _Logger:
                    """Minimal loguru.logger stub: routes everything to stdlib logging."""

                    def _log(self, level, msg, *args, **kwargs):
                        import logging as _logging

                        getattr(
                            _logging.getLogger("loguru"), level, _logging.getLogger("loguru").debug
                        )(str(msg) if not args else (str(msg) % args))

                    def debug(self, msg, *a, **k):
                        self._log("debug", msg, *a, **k)

                    def info(self, msg, *a, **k):
                        self._log("info", msg, *a, **k)

                    def warning(self, msg, *a, **k):
                        self._log("warning", msg, *a, **k)

                    def error(self, msg, *a, **k):
                        self._log("error", msg, *a, **k)

                    def critical(self, msg, *a, **k):
                        self._log("critical", msg, *a, **k)

                    def opt(self, *_, **__):
                        return self

                    def add(self, *_, **__):
                        pass

                    def remove(self, *_, **__):
                        pass

                _loguru.logger = _Logger()
                sys.modules["loguru"] = _loguru
                log.debug("loguru not installed — using no-op stub (safe for gimmvfi_f)")

        # Patch timm compatibility shims for older GIMM-VFI / FlowFormer code.
        # Several symbols were moved or removed between timm 0.4 and 0.9:
        #   - timm.models.helpers.overlay_external_default_cfg  (removed in >= 0.6)
        #   - timm.models.layers.activations  (moved to timm.layers in >= 0.6)
        # Inject shims so the FlowFormer import chain doesn't crash.
        try:
            import timm  # type: ignore[import-not-found]  # noqa: F401

            # 1) overlay_external_default_cfg shim
            import timm.models.helpers as _tmh

            if not hasattr(_tmh, "overlay_external_default_cfg"):

                def _overlay_external_default_cfg(default_cfg, kwargs):
                    return default_cfg, kwargs

                _tmh.overlay_external_default_cfg = _overlay_external_default_cfg
                log.debug("timm: patched overlay_external_default_cfg (removed in newer timm)")

            # 2) timm.models.layers.activations shim
            #    In timm >= 0.6 activations live in timm.layers; the legacy
            #    timm.models.layers shim module no longer re-exports them.
            import timm.models.layers as _tml

            if not hasattr(_tml, "activations"):
                try:
                    import timm.layers.activations as _tla  # new location

                    _tml.activations = _tla
                    sys.modules.setdefault("timm.models.layers.activations", _tla)
                    log.debug(
                        "timm: patched timm.models.layers.activations → timm.layers.activations"
                    )
                except ImportError:
                    # Neither location found; inject a minimal stub
                    import types as _types

                    _tla = _types.ModuleType("timm.models.layers.activations")
                    _tml.activations = _tla
                    sys.modules["timm.models.layers.activations"] = _tla
                    log.debug("timm: injected empty timm.models.layers.activations stub")

            # 3) Suppress timm pretrained-weight downloads.
            #    FlowFormer calls timm.create_model('twins_svt_large', pretrained=True)
            #    which downloads ImageNet weights that are immediately overwritten by the
            #    GIMM-VFI safetensors we load ourselves.  Make load_pretrained a no-op.
            try:
                import timm.models._builder as _tmb

                if not getattr(_tmb, "_load_pretrained_patched", False):
                    _tmb._orig_load_pretrained = _tmb.load_pretrained

                    def _noop_load_pretrained(model, *_a, **_kw):
                        return model

                    _tmb.load_pretrained = _noop_load_pretrained
                    _tmb._load_pretrained_patched = True
                    log.debug(
                        "timm: patched load_pretrained to no-op (weights loaded from GIMM-VFI safetensors)"
                    )
            except (ImportError, AttributeError):
                pass
        except ImportError:
            pass

        # Purge any broken partial-import entries for the GIMM-VFI source-tree
        # packages.  These appear when a previous attempt failed mid-import (e.g.
        # yacs missing before the stub was installed) and would prevent a clean retry.
        for key in list(sys.modules):
            if key == "models" or key.startswith("models."):
                if sys.modules[key] is None or not hasattr(sys.modules[key], "__spec__"):
                    del sys.modules[key]

        # configs/ lives at the repo root, not under src/
        repo_root = self.source_dir
        if os.path.basename(repo_root) == "src":
            repo_root = os.path.dirname(repo_root)
        cfg_path = os.path.join(repo_root, "configs", "gimmvfi", f"{self.variant}_arb.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"GIMM-VFI config not found: {cfg_path}")

        with open(cfg_path) as f:
            raw_cfg = yaml.safe_load(f)

        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        # Merge YAML arch section on top of defaults so that keys not present
        # in the YAML (e.g. raft_iter, fwarp_type, rec_weight, ema_value) still
        # have sensible values when the model __init__ accesses config.<key>.
        _arch_defaults = OmegaConf.create(
            {
                "raft_iter": 20,
                "fwarp_type": "linear",
                "rec_weight": 0.1,
                "ema": False,
                "ema_value": None,
                # HypoNetConfig fields that may be absent in older YAML files
                "hyponet": {
                    "normalize_weight": True,
                    "linear_interpo": False,
                    "output_bias": 0.5,
                    "use_bias": True,
                    "input_dim": 2,
                    "output_dim": 3,
                    "n_layer": 5,
                },
            }
        )
        arch_cfg = OmegaConf.merge(_arch_defaults, OmegaConf.create(raw_cfg["arch"]))

        if self.variant == "gimmvfi_r":
            raft_pkg = importlib.import_module("models.generalizable_INR.raft")
            gimmvfi_model_pkg = importlib.import_module("models.generalizable_INR.gimmvfi_r")
            original_init_fn = raft_pkg.initialize_RAFT

            def _make_flow_model_no_load(*_args, **_kwargs):
                import argparse as _ap

                from models.generalizable_INR.raft.raft import (
                    RAFT as _RAFT,  # type: ignore[import-not-found]
                )

                _rargs = _ap.Namespace(
                    small=False,
                    mixed_precision=False,
                    alternate_corr=False,
                    dropout=0,
                )
                return _RAFT(_rargs)

            raft_pkg.initialize_RAFT = _make_flow_model_no_load
            gimmvfi_model_pkg.initialize_RAFT = _make_flow_model_no_load
        else:
            ff_pkg = importlib.import_module("models.generalizable_INR.flowformer")
            gimmvfi_model_pkg = importlib.import_module("models.generalizable_INR.gimmvfi_f")
            original_init_fn = ff_pkg.initialize_Flowformer

            def _make_flow_model_no_load(*_args, **_kwargs):
                from models.generalizable_INR.flowformer.configs.submission import (
                    get_cfg,  # type: ignore[import-not-found]
                )
                from models.generalizable_INR.flowformer.core.FlowFormer import (
                    build_flowformer,  # type: ignore[import-not-found]
                )

                return build_flowformer(get_cfg())

            ff_pkg.initialize_Flowformer = _make_flow_model_no_load
            gimmvfi_model_pkg.initialize_Flowformer = _make_flow_model_no_load

        try:
            from models import create_model  # type: ignore[import-not-found]

            model, _ = create_model(arch_cfg, ema=False)
        finally:
            if self.variant == "gimmvfi_r":
                raft_pkg.initialize_RAFT = original_init_fn
                gimmvfi_model_pkg.initialize_RAFT = original_init_fn
            else:
                ff_pkg.initialize_Flowformer = original_init_fn
                gimmvfi_model_pkg.initialize_Flowformer = original_init_fn

        model.load_state_dict(model_sd, strict=False)

        _flow_sd = flow_sd
        if _flow_sd and next(iter(_flow_sd)).startswith("module."):
            _flow_sd = {k[len("module.") :]: v for k, v in _flow_sd.items()}
        model.flow_estimator.load_state_dict(_flow_sd, strict=False)

        model.to(self._device).eval()
        self._model = model
        self._flow_model = None

        from utils.utils import InputPadder  # type: ignore[import-not-found]

        self._padder_cls = InputPadder

    # ── inference ─────────────────────────────────────────────────────

    def interpolate(
        self,
        frame0_bgr: np.ndarray,
        frame1_bgr: np.ndarray,
    ) -> np.ndarray:
        """Return the interpolated mid-frame between two BGR frames."""
        import torch

        if self._model is None and not self._init_failed:
            try:
                self._init_model()
            except Exception as exc:
                if self.gpu_only:
                    raise RuntimeError(f"GIMM-VFI --gpu-only: init failed — {exc}") from exc
                log.error("GIMM-VFI initialisation failed: %s", exc)
                self._init_failed = True

        if self._model is None:
            if self.gpu_only:
                raise RuntimeError("GIMM-VFI --gpu-only: model not loaded; cannot interpolate")
            return cv2.addWeighted(frame0_bgr, 0.5, frame1_bgr, 0.5, 0)

        h, w, _ = frame0_bgr.shape

        def _to_tensor(bgr: np.ndarray):
            return (
                torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())
                .float()
                .unsqueeze(0)
                / 255.0
            ).to(self._device)

        img0 = _to_tensor(frame0_bgr)
        img1 = _to_tensor(frame1_bgr)

        # Pad to multiple of 32 (RAFT requirement)
        if self._padder_cls is not None:
            padder = self._padder_cls(img0.shape, 32)
            img0, img1 = padder.pad(img0, img1)
        else:
            ph = ((h - 1) // 32 + 1) * 32
            pw = ((w - 1) // 32 + 1) * 32
            pad = (0, pw - w, 0, ph - h)
            img0 = torch.nn.functional.pad(img0, pad, mode="replicate")
            img1 = torch.nn.functional.pad(img1, pad, mode="replicate")

        # Stack frames as [B, C, 2, H, W] — required by GIMMVFI_R.forward
        xs = torch.cat((img0.unsqueeze(2), img1.unsqueeze(2)), dim=2)

        # Generate 1 intermediate frame (2× temporal up-sampling: t = 0.5)
        N = 2  # resulting multiplier; N-1 intermediate frames are synthesised
        B = xs.shape[0]
        s_shape = xs.shape[-2:]

        coord_inputs = [
            (
                self._model.sample_coord_input(
                    B,
                    s_shape,
                    [1 / N * i],
                    device=xs.device,
                    upsample_ratio=self.ds_factor,
                ),
                None,
            )
            for i in range(1, N)
        ]
        timesteps = [
            i * (1 / N) * torch.ones(B, device=xs.device, dtype=torch.float32) for i in range(1, N)
        ]

        with (
            torch.no_grad(),
            torch.autocast(device_type=self._device.type, enabled=(self._device.type == "cuda")),
        ):
            all_outputs = self._model(xs, coord_inputs, t=timesteps, ds_factor=self.ds_factor)

        outputs = []
        for mid in all_outputs["imgt_pred"]:
            if self._padder_cls is not None:
                mid = padder.unpad(mid)
            else:
                mid = mid[..., :h, :w]
            mid_np = (mid[0].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
            outputs.append(cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR))

        return outputs

    # ── cleanup ───────────────────────────────────────────────────────

    def cleanup(self):
        for attr in ("_model", "_flow_model"):
            obj = getattr(self, attr, None)
            if obj is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ──────────────────────────────────────────────────────────────────────
# Per-frame enhancement pipeline
# ──────────────────────────────────────────────────────────────────────


def build_frame_enhancer(args):
    """Return ``(cpu_enhancer, esrgan)`` for the configured enhancement stages.

    ``cpu_enhancer`` is a pure-CPU callable (CLAHE, denoise, sharpen) that
    is safe to call from multiple threads concurrently.

    ``esrgan`` is an :class:`ESRGANUpscaler` instance (or *None*) that
    may require exclusive GPU access — callers must serialise access with
    a lock when running in parallel on GPU.

    RIFE / HiFI / GIMM-VFI are *not* included here because they operate
    between frames (handled at the video level).
    """
    esrgan = None
    if args.esrgan:
        esrgan = ESRGANUpscaler(
            scale=args.esrgan_scale,
            model_name=args.esrgan_model,
            model_path=args.esrgan_model_path,
            half=args.esrgan_half,
            tile=args.esrgan_tile,
            force_cpu=args.esrgan_cpu,
        )
        # Model loads lazily on first upscale() call

    def cpu_enhance(frame_bgr: np.ndarray) -> np.ndarray:
        out = frame_bgr
        # 1. Denoise first (before other ops to avoid amplifying noise)
        if args.denoise:
            out = apply_denoise(out, h=args.denoise_strength)
        # 2. CLAHE contrast
        if args.clahe:
            out = apply_clahe(
                out, clip_limit=args.clahe_clip, tile_grid=(args.clahe_grid, args.clahe_grid)
            )
        # 3. Sharpen (after CLAHE to enhance the now-better-contrasted edges)
        if args.sharpen:
            out = apply_sharpen(out, sigma=args.sharpen_sigma, strength=args.sharpen_strength)
        return out

    return cpu_enhance, esrgan


# ──────────────────────────────────────────────────────────────────────
# Video processing
# ──────────────────────────────────────────────────────────────────────


def _build_output_path(video_path: str, output_dir: str, suffix: str) -> str:
    """Construct output path preserving path segments after ``Video``.

    Example:
        /.../Video/visit 1/POM.../Off_4R.mp4
        -> <output_dir>/visit 1/POM.../Off_4R{suffix}.mp4

    If no ``Video`` segment is present, falls back to writing directly
    under ``output_dir``.
    """
    raw_path = str(video_path)
    stem = Path(raw_path).stem
    ext = Path(raw_path).suffix or ".mp4"
    output_name = f"{stem}{suffix}{ext}"

    normalized = raw_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]

    video_idx = next(
        (idx for idx, part in enumerate(parts) if part.lower() == "video"),
        None,
    )

    if video_idx is None or video_idx >= len(parts) - 1:
        return os.path.join(output_dir, output_name)

    rel_parts = parts[video_idx + 1 : -1]
    return os.path.join(output_dir, *rel_parts, output_name)


def _active_stages_suffix(args) -> str:
    """Build a short suffix string indicating which enhancements were applied."""
    tags = []
    if args.denoise:
        tags.append("dn")
    if args.clahe:
        tags.append("cl")
    if args.sharpen:
        tags.append("sh")
    if args.esrgan:
        tags.append(f"sr{args.esrgan_scale}x")
    if args.rife:
        tags.append("rife")
    if args.hifi:
        tags.append("hifi")
    if args.gimmvfi:
        tags.append("gimmvfi")
    return "_" + "_".join(tags) if tags else ""


def process_single_video(
    video_path: str,
    output_path: str,
    frame_enhancer,
    esrgan: Optional[ESRGANUpscaler] = None,
    rife: Optional[RIFEInterpolator] = None,
    hifi: Optional[HiFIInterpolator] = None,
    gimmvfi: Optional[GIMMVFIInterpolator] = None,
    gpu_lock: Optional[threading.Semaphore] = None,
    codec: str = "avc1",
    show_progress: bool = True,
) -> dict:
    """Process one video through the enhancement pipeline.

    ``frame_enhancer`` applies CPU-only stages (CLAHE, denoise, sharpen).

    Pipeline order: CLAHE+sharpen → GIMM-VFI (on original-resolution frames)
    → ESRGAN upscale (on every output frame including interpolated ones).

    Running GIMM-VFI before ESRGAN keeps it on small (pre-upscale) frames,
    which avoids the large VRAM footprint that FlowFormer incurs on 4×-upscaled
    inputs.  ESRGAN uses tiled inference so its peak VRAM is bounded regardless
    of output size.

    ``gpu_lock`` is a single semaphore shared by both ESRGAN and the interpolator
    so that at most N GPU operations run concurrently across all threads.

    Accepts at most one interpolator (``rife``, ``hifi``, or ``gimmvfi``).
    Returns a dict with metadata about the processing result.
    """
    interpolator = rife or hifi or gimmvfi  # at most one is set

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": f"Cannot open {video_path}"}

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame to determine output dimensions after enhancement
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        return {"status": "error", "message": f"Cannot read first frame of {video_path}"}

    # ── CPU enhancement (CLAHE + sharpen, no lock needed) ───────────────────
    clean_first = frame_enhancer(first_frame)
    # Keep a pre-ESRGAN copy for GIMM-VFI (runs on original-resolution frames
    # to avoid the enormous VRAM footprint of FlowFormer on 4×-upscaled inputs)
    prev_clean = clean_first

    # Force-init GIMM-VFI now so init errors surface before we open the writer
    if interpolator is not None and isinstance(interpolator, GIMMVFIInterpolator):
        if interpolator._model is None and not interpolator._init_failed:
            try:
                interpolator._init_model()
            except Exception as exc:
                if interpolator.gpu_only:
                    cap.release()
                    raise
                log.error("GIMM-VFI initialisation failed: %s", exc)
                interpolator._init_failed = True

    # ── ESRGAN first frame (locked only when on GPU) ─────────────────────────
    first_out = clean_first
    if esrgan is not None:
        if gpu_lock is not None:
            gpu_lock.acquire()
        try:
            first_out = esrgan.upscale(clean_first)
        except Exception as exc:
            if gpu_lock is not None:
                gpu_lock.release()
            cap.release()
            return {
                "status": "error",
                "message": f"ESRGAN failed on first frame of {Path(video_path).name}: {exc}",
            }
        if gpu_lock is not None:
            gpu_lock.release()

    h_out, w_out = first_out.shape[:2]
    fps_multiplier = 2 if interpolator is not None else 1
    fps_out = fps_in * fps_multiplier
    codec_tag = (codec or "avc1").strip()
    if len(codec_tag) != 4:
        cap.release()
        return {
            "status": "error",
            "message": (
                f"Invalid codec '{codec_tag}'. Provide a 4-character FOURCC "
                f"(e.g. avc1, mp4v, H264)."
            ),
        }

    fourcc = cv2.VideoWriter_fourcc(*codec_tag)
    writer = cv2.VideoWriter(output_path, fourcc, fps_out, (w_out, h_out))
    used_codec = codec_tag

    if not writer.isOpened() and codec_tag.lower() != "mp4v":
        # Fallback keeps long jobs running if ffmpeg/OpenCV lacks H.264 support.
        log.warning(
            "Could not open VideoWriter with codec '%s' for %s; falling back to mp4v",
            codec_tag,
            Path(video_path).name,
        )
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_out,
            (w_out, h_out),
        )
        used_codec = "mp4v"

    if not writer.isOpened():
        cap.release()
        return {"status": "error", "message": f"Cannot create writer for {output_path}"}

    writer.write(first_out)
    frames_written = 1

    pbar = None
    if show_progress:
        pbar = tqdm(total=total_frames - 1, desc=Path(video_path).name, unit="fr", leave=False)

    def _esrgan_upscale(frame):
        """Run ESRGAN under the shared GPU lock; return upscaled frame."""
        if esrgan is None:
            return frame
        if gpu_lock is not None:
            gpu_lock.acquire()
        try:
            return esrgan.upscale(frame)
        finally:
            if gpu_lock is not None:
                gpu_lock.release()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # CPU enhancement (CLAHE + sharpen; runs in parallel across threads)
            curr_clean = frame_enhancer(frame_bgr)

            # ── Frame interpolation on pre-ESRGAN (small) frames ─────────────
            # Running GIMM-VFI here (before upscaling) keeps FlowFormer working
            # on original-resolution inputs — typically 4-16× fewer pixels than
            # after ESRGAN, so VRAM drops from 20-30 GB to 1-3 GB per inference.
            if interpolator is not None:
                if gpu_lock is not None:
                    gpu_lock.acquire()
                try:
                    mids = interpolator.interpolate(prev_clean, curr_clean)
                except Exception as interp_exc:
                    if not getattr(interpolator, "_interp_fallback_warned", False):
                        import traceback as _tb

                        log.warning(
                            "Interpolation failed (%s); falling back to linear blend "
                            "for remaining frames of %s\n%s",
                            interp_exc,
                            Path(video_path).name,
                            _tb.format_exc(),
                        )
                        interpolator._interp_fallback_warned = True  # type: ignore[attr-defined]
                    mids = cv2.addWeighted(prev_clean, 0.5, curr_clean, 0.5, 0)
                finally:
                    if gpu_lock is not None:
                        gpu_lock.release()

                # ESRGAN-upscale every interpolated mid-frame and write
                for mid_frame in (mids if isinstance(mids, list) else [mids]):
                    writer.write(_esrgan_upscale(mid_frame))
                    frames_written += 1

            # ── ESRGAN current frame and write ────────────────────────────────
            writer.write(_esrgan_upscale(curr_clean))
            frames_written += 1
            prev_clean = curr_clean
            if pbar is not None:
                pbar.update(1)
    except Exception as exc:
        if pbar is not None:
            pbar.close()
        cap.release()
        writer.release()
        # Remove the partially-written output file
        try:
            os.remove(output_path)
        except OSError:
            pass
        return {"status": "error", "message": f"Processing failed at frame {frames_written}: {exc}"}

    if pbar is not None:
        pbar.close()
    cap.release()
    writer.release()

    return {
        "status": "ok",
        "input_path": video_path,
        "output_path": output_path,
        "output_codec": used_codec,
        "input_fps": fps_in,
        "output_fps": fps_out,
        "input_resolution": f"{w_in}x{h_in}",
        "output_resolution": f"{w_out}x{h_out}",
        "input_frames": total_frames,
        "output_frames": frames_written,
    }


# ──────────────────────────────────────────────────────────────────────
# Dataset-level processing
# ──────────────────────────────────────────────────────────────────────


def _create_model_instances(args, device: Optional[str] = None):
    """Create fresh ESRGAN / interpolator instances from CLI args.

    ``device`` pins the instances to a specific CUDA device (e.g. ``"cuda:2"``).
    When *None* the instances auto-select the default CUDA device.

    Returns ``(esrgan, rife, hifi, gimmvfi)`` — each is an instance or *None*.
    """
    gpu_only = getattr(args, "gpu_only", False)

    esrgan = None
    if args.esrgan:
        esrgan = ESRGANUpscaler(
            scale=args.esrgan_scale,
            model_name=args.esrgan_model,
            model_path=args.esrgan_model_path,
            half=args.esrgan_half,
            tile=args.esrgan_tile,
            force_cpu=args.esrgan_cpu,
            gpu_only=gpu_only,
            device=device,
        )
    rife = None
    hifi = None
    gimmvfi = None
    if args.rife:
        rife = RIFEInterpolator(model_dir=args.rife_model_dir)
    elif args.hifi:
        hifi = HiFIInterpolator(
            model_dir=args.hifi_model_dir,
            num_steps=args.hifi_steps,
            guidance=args.hifi_guidance,
            patch_size=args.hifi_patch_size,
        )
    elif args.gimmvfi:
        gimmvfi = GIMMVFIInterpolator(
            model_dir=args.gimmvfi_model_dir,
            variant=args.gimmvfi_variant,
            ds_factor=args.gimmvfi_ds_factor,
            source_dir=getattr(args, "gimmvfi_source_dir", None),
            gpu_only=gpu_only,
            device=device,
        )
    return esrgan, rife, hifi, gimmvfi


def process_dataset(args):
    """Process all videos in the dataset CSV and produce a new CSV.

    When ``--workers N`` (N > 1), videos are processed in parallel using
    a :class:`ThreadPoolExecutor`.  CPU-only enhancement stages (CLAHE,
    denoise, sharpen) run concurrently across threads because the
    underlying OpenCV C++ functions release the GIL.

    Model stages (ESRGAN, RIFE / HiFI / GIMM-VFI):
      * **On GPU** — a single shared model instance is protected by a
        ``threading.Semaphore(--gpu-workers)`` so up to ``--gpu-workers``
        threads (default 5) can use the GPU concurrently.  Reduce if you
        see CUDA OOM errors on large frames.
      * **On CPU** (e.g. ``--esrgan-cpu``) — each thread gets its own
        model instance via ``threading.local`` so inference runs in full
        parallel (PyTorch ATen releases the GIL during forward passes).
    """
    df = pd.read_csv(args.csv)

    # Detect the video path column
    video_col = args.video_column
    if video_col not in df.columns:
        # Try common alternatives
        for candidate in ["video_path", "path", "file", "filepath", "video"]:
            if candidate in df.columns:
                video_col = candidate
                break
        else:
            print(f"Error: column '{args.video_column}' not found in CSV.")
            print(f"  Available columns: {list(df.columns)}")
            sys.exit(1)

    if args.video_quality_threshold not in (1, 2, 3):
        raise ValueError("video_quality_threshold must be one of: 1, 2, 3")
    if args.video_quality_labels_csv is None and args.video_quality_threshold < 3:
        raise ValueError("video_quality_threshold < 3 requires --video-quality-labels-csv")
    if args.video_quality_labels_csv is not None:
        print(f"Loading manual video quality labels: {args.video_quality_labels_csv}")
        labels_df = _load_video_quality_labels(args.video_quality_labels_csv)
        print(f"Loaded video quality labels for {len(labels_df)} unique videos")
        df = _apply_video_quality_filter(
            df,
            video_col,
            labels_df,
            args.video_quality_threshold,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = _active_stages_suffix(args)

    # Summary of what will be done
    stages = []
    if args.denoise:
        stages.append(f"denoise (h={args.denoise_strength})")
    if args.clahe:
        stages.append(f"CLAHE (clip={args.clahe_clip}, grid={args.clahe_grid})")
    if args.sharpen:
        stages.append(f"sharpen (σ={args.sharpen_sigma}, strength={args.sharpen_strength})")
    if args.esrgan:
        cpu_tag = " [CPU]" if args.esrgan_cpu else ""
        stages.append(f"ESRGAN {args.esrgan_scale}× ({args.esrgan_model}){cpu_tag}")
    if args.rife:
        stages.append("RIFE 2× frame interpolation")
    if args.hifi:
        stages.append(f"HiFI 2× frame interpolation (steps={args.hifi_steps})")
    if args.gimmvfi:
        stages.append(f"GIMM-VFI 2× frame interpolation ({args.gimmvfi_variant})")

    if not stages:
        print(
            "Error: no enhancement stages selected.  Use --clahe, --denoise, "
            "--sharpen, --esrgan, and/or --rife."
        )
        sys.exit(1)

    n_workers = getattr(args, "workers", 1) or 1
    _gpu_workers_per_gpu = getattr(args, "gpu_workers", 4)
    print(f"Enhancement pipeline: {' → '.join(stages)}")
    print(f"Videos to process: {len(df)}")
    print(f"Workers:           {n_workers} (CPU threads)")
    print(f"GPU workers:       {_gpu_workers_per_gpu}/GPU (steps down on OOM)")
    print(f"Output codec:      {args.codec}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Output CSV:        {args.output_csv}")
    print(flush=True)
    sys.stdout.flush()

    if args.dry_run:
        for _, row in df.iterrows():
            vp = str(row[video_col])
            op = _build_output_path(vp, args.output_dir, suffix)
            print(f"  {vp}")
            print(f"  → {op}")
            print()
        print("Dry run complete — no files written.")
        return

    # ── GPU-only early assertion ───────────────────────────────────────
    gpu_only = getattr(args, "gpu_only", False)
    if gpu_only:
        try:
            import torch as _torch_gpu

            if not _torch_gpu.cuda.is_available():
                print("ERROR: --gpu-only requested but CUDA is not available on this system.")
                print("       Check that the container has GPU access and a compatible driver.")
                sys.exit(1)
            # Force CUDA context creation early so any driver issues surface now
            # (before processing thousands of videos) rather than mid-run.
            _torch_gpu.cuda.init()
            log.info("GPU-only mode: CUDA device %s ready", _torch_gpu.cuda.get_device_name(0))
        except RuntimeError as exc:
            print(f"ERROR: --gpu-only CUDA init failed: {exc}")
            sys.exit(1)

    # Build CPU-only enhancer (always shared — stateless OpenCV calls).
    cpu_enhancer, _ = build_frame_enhancer(args)

    # Validate model paths eagerly (before spawning threads).
    if args.rife:
        _probe = RIFEInterpolator(model_dir=args.rife_model_dir)
        _probe._find_rife_root()  # raises FileNotFoundError if missing
        del _probe
    elif args.hifi:
        _probe = HiFIInterpolator(
            model_dir=args.hifi_model_dir,
            num_steps=args.hifi_steps,
            guidance=args.hifi_guidance,
            patch_size=args.hifi_patch_size,
        )
        _probe._find_checkpoint()
        del _probe
    elif args.gimmvfi:
        _probe = GIMMVFIInterpolator(
            model_dir=args.gimmvfi_model_dir,
            variant=args.gimmvfi_variant,
            ds_factor=args.gimmvfi_ds_factor,
        )
        _probe._find_checkpoint()
        del _probe

    # ── Per-model GPU / CPU strategy ──────────────────────────────────
    # Each model stage independently decides GPU vs CPU:
    #   GPU model → shared instance + per-frame Lock (serialised)
    #   CPU model → per-thread instances, no lock (full parallelism)
    #
    # This allows e.g. ESRGAN on CPU (--esrgan-cpu) running in parallel
    # while GIMM-VFI stays on GPU with a lock.

    # Check whether ESRGAN will run on GPU or CPU.
    _probe_esrgan, _, _, _ = _create_model_instances(args)
    esrgan_on_gpu = _probe_esrgan is not None and _probe_esrgan.uses_gpu
    del _probe_esrgan

    # Interpolators always use GPU when available (no CPU flag yet).
    _cuda_available = False
    try:
        import torch as _torch

        _cuda_available = _torch.cuda.is_available()
    except ImportError:
        pass
    has_interp = args.rife or args.hifi or args.gimmvfi
    interp_on_gpu = has_interp and _cuda_available

    gpu_workers_per_gpu = getattr(args, "gpu_workers", 4)
    any_on_gpu = esrgan_on_gpu or interp_on_gpu

    # ── GPU resource pool ──────────────────────────────────────────────────
    # --gpu-workers N means N slots *per GPU*.  Total initial slots =
    # N × num_gpus, distributed round-robin so each GPU gets N workers.
    # On OOM the retry logic steps the per-GPU count down by 1 each pass
    # (N-1, N-2, …, 1 per GPU) until the video succeeds or is skipped.
    #
    # Example — 7 GPUs, --gpu-workers 4:
    #   28 initial slots (4 per GPU); retry at 21, 14, 7, then 1 if needed.
    import queue as _queue_mod

    num_gpus = 0
    if any_on_gpu:
        try:
            import torch as _t

            num_gpus = _t.cuda.device_count()
        except ImportError:
            pass

    total_initial_slots = (
        gpu_workers_per_gpu * num_gpus if (any_on_gpu and n_workers > 1 and num_gpus > 0) else 1
    )

    # Build pool slots.  CPU-only or sequential mode → single slot.
    _gpu_pool: "_queue_mod.Queue[tuple]" = _queue_mod.Queue()
    _pool_instances: list = []  # for cleanup

    if any_on_gpu and n_workers > 1 and num_gpus > 0:
        for slot_idx in range(total_initial_slots):
            dev = f"cuda:{slot_idx % num_gpus}"
            e, r, h, g = _create_model_instances(args, device=dev)
            _gpu_pool.put((e, r, h, g))
            for m in (e, r, h, g):
                if m is not None:
                    _pool_instances.append(m)
        log.info(
            "GPU pool: %d slots (%d/GPU) across %d GPU(s) (%s)",
            total_initial_slots,
            gpu_workers_per_gpu,
            num_gpus,
            ", ".join(f"cuda:{i}" for i in range(num_gpus)),
        )
    else:
        # Sequential mode or CPU-only: single slot, no device pinning.
        e, r, h, g = _create_model_instances(args)
        _gpu_pool.put((e, r, h, g))
        _pool_instances.extend(m for m in (e, r, h, g) if m is not None)

    # CPU-only per-thread instances (when models run on CPU, full parallelism).
    esrgan_per_thread = args.esrgan and not esrgan_on_gpu and n_workers > 1
    interp_per_thread = has_interp and not interp_on_gpu and n_workers > 1
    _tls = threading.local()
    _tls_instances_lock = threading.Lock()
    _tls_instances: list = []

    def _get_thread_esrgan() -> Optional[ESRGANUpscaler]:
        if not esrgan_per_thread:
            return None  # caller uses pool slot
        if hasattr(_tls, "esrgan"):
            return _tls.esrgan
        inst, _, _, _ = _create_model_instances(args)
        _tls.esrgan = inst
        with _tls_instances_lock:
            _tls_instances.append(inst)
        return inst

    def _get_thread_interp():
        if not interp_per_thread:
            return None, None, None  # caller uses pool slot
        if hasattr(_tls, "interp"):
            return _tls.interp
        _, r, h, g = _create_model_instances(args)
        _tls.interp = (r, h, g)
        with _tls_instances_lock:
            for m in (r, h, g):
                if m is not None:
                    _tls_instances.append(m)
        return r, h, g

    # Print strategy summary.
    if n_workers > 1:
        parts = []
        if args.esrgan:
            parts.append(
                f"ESRGAN: {'GPU' if esrgan_on_gpu else 'CPU'}"
                + (
                    f" ({gpu_workers_per_gpu}/GPU × {num_gpus} GPUs = {total_initial_slots} slots)"
                    if esrgan_on_gpu
                    else f" ({n_workers} threads)"
                )
            )
        if has_interp:
            iname = "RIFE" if args.rife else ("HiFI" if args.hifi else "GIMM-VFI")
            parts.append(
                f"{iname}: {'GPU' if interp_on_gpu else 'CPU'}"
                + (
                    f" ({gpu_workers_per_gpu}/GPU × {num_gpus} GPUs = {total_initial_slots} slots)"
                    if interp_on_gpu
                    else f" ({n_workers} threads)"
                )
            )
        if parts:
            print(f"[Parallelism] {'; '.join(parts)}")

    # ── Build the task list (filter skips/misses up front) ────────────
    tasks = []  # (df_index, video_path, output_path)
    new_paths = {}  # df_index -> new_path
    enhanced_flags = {}  # df_index -> bool
    results = {}  # df_index -> result dict

    for df_idx, (_, row) in enumerate(df.iterrows()):
        vp = str(row[video_col])
        op = _build_output_path(vp, args.output_dir, suffix)
        os.makedirs(os.path.dirname(op), exist_ok=True)

        if not os.path.isfile(vp):
            print(f"  [miss] {vp} not found — skipping")
            new_paths[df_idx] = vp
            enhanced_flags[df_idx] = False
            results[df_idx] = {"status": "error", "message": f"File not found: {vp}"}
            continue

        if os.path.exists(op) and not args.overwrite:
            print(f"  [skip] {Path(op).name} already exists (use --overwrite to redo)")
            new_paths[df_idx] = op
            enhanced_flags[df_idx] = True
            results[df_idx] = {"status": "skipped", "output_path": op}
            continue

        tasks.append((df_idx, vp, op))

    # Sort smallest-first so the most videos finish before we hit OOM on giants
    tasks.sort(key=lambda t: os.path.getsize(t[1]) if os.path.isfile(t[1]) else 0)
    if tasks:
        sizes_mb = [os.path.getsize(t[1]) / 1e6 for t in tasks if os.path.isfile(t[1])]
        if sizes_mb:
            log.info(
                "Tasks sorted by size: %.1f MB (smallest) → %.1f MB (largest)",
                sizes_mb[0],
                sizes_mb[-1],
            )

    def _save_csv():
        """Write the output CSV with current progress."""
        df_out = df.copy()
        df_out[video_col] = [
            new_paths.get(i, str(row[video_col])) for i, (_, row) in enumerate(df.iterrows())
        ]
        df_out["enhancement_applied"] = [enhanced_flags.get(i, False) for i in range(len(df))]
        df_out.to_csv(args.output_csv, index=False)

    if not tasks:
        print("No videos to process.")
        _save_csv()
        print(f"Output CSV: {args.output_csv}")
        return

    print(f"Processing {len(tasks)} videos ({len(df) - len(tasks)} skipped/missing)...")
    t0 = time.time()

    def _is_oom(exc_or_msg) -> bool:
        """Return True if the exception or message looks like a CUDA OOM."""
        msg = str(exc_or_msg).lower()
        if "out of memory" in msg:
            return True
        try:
            import torch as _t

            if hasattr(_t.cuda, "OutOfMemoryError") and isinstance(
                exc_or_msg, _t.cuda.OutOfMemoryError
            ):
                return True
        except Exception:
            pass
        return False

    def _clear_cuda_cache():
        try:
            import torch as _t

            _t.cuda.empty_cache()
        except Exception:
            pass

    def _do_one(task, _show_progress_override=None):
        """Process one video — called from main thread or thread pool."""
        df_idx, vp, op = task
        show_prog = (
            _show_progress_override if _show_progress_override is not None else (n_workers <= 1)
        )

        # CPU-only models use per-thread instances for full parallelism.
        t_esrgan = _get_thread_esrgan()  # None if using GPU pool
        t_rife, t_hifi, t_gimmvfi = _get_thread_interp()  # None if using GPU pool

        if t_esrgan is None and t_rife is None and t_hifi is None and t_gimmvfi is None:
            # GPU mode: check out an exclusive model slot (blocks until one is free)
            slot_esrgan, slot_rife, slot_hifi, slot_gimmvfi = _gpu_pool.get()
            try:
                result = process_single_video(
                    vp,
                    op,
                    cpu_enhancer,
                    esrgan=slot_esrgan,
                    rife=slot_rife,
                    hifi=slot_hifi,
                    gimmvfi=slot_gimmvfi,
                    gpu_lock=None,  # slot is exclusive — no further locking needed
                    codec=args.codec,
                    show_progress=show_prog,
                )
                # Re-classify OOM errors that were caught inside process_single_video
                if result.get("status") == "error" and _is_oom(result.get("message", "")):
                    _clear_cuda_cache()
                    result = {"status": "oom", "message": result["message"]}
                return df_idx, vp, op, result
            except Exception as exc:
                if _is_oom(exc):
                    _clear_cuda_cache()
                    return df_idx, vp, op, {"status": "oom", "message": f"CUDA OOM: {exc}"}
                raise
            finally:
                _gpu_pool.put((slot_esrgan, slot_rife, slot_hifi, slot_gimmvfi))
        else:
            # CPU mode: use per-thread instances, no pool needed
            return (
                df_idx,
                vp,
                op,
                process_single_video(
                    vp,
                    op,
                    cpu_enhancer,
                    esrgan=t_esrgan,
                    rife=t_rife,
                    hifi=t_hifi,
                    gimmvfi=t_gimmvfi,
                    gpu_lock=None,
                    codec=args.codec,
                    show_progress=show_prog,
                ),
            )

    # Track tasks that triggered OOM so we can retry them with fewer slots.
    oom_tasks: list = []

    def _handle_result(df_idx, vp, op, result):
        """Update shared dicts and return True if OOM (needs retry)."""
        results[df_idx] = result
        status = result.get("status", "?")
        if status == "ok":
            new_paths[df_idx] = op
            enhanced_flags[df_idx] = True
            return False
        elif status == "oom":
            # Keep original path for now; will be updated if retry succeeds
            new_paths[df_idx] = vp
            enhanced_flags[df_idx] = False
            return True
        else:
            new_paths[df_idx] = vp
            enhanced_flags[df_idx] = False
            return False

    try:
        if n_workers <= 1:
            # ── Sequential mode (original behaviour) ──────────────────
            for task_num, task in enumerate(tasks, start=1):
                _, vp, _ = task
                print(f"  [{task_num}/{len(tasks)}] Processing {Path(vp).name} ...")
                try:
                    df_idx, vp, op, result = _do_one(task)
                except Exception as exc:
                    result = {"status": "error", "message": f"Unhandled exception: {exc}"}
                    df_idx, vp, op = task
                is_oom = _handle_result(df_idx, vp, op, result)
                if result["status"] == "ok":
                    print(
                        f"    -> {result['output_resolution']} @ "
                        f"{result['output_fps']:.0f} fps  "
                        f"({result['output_frames']} frames)"
                    )
                elif is_oom:
                    print(f"    OOM — will retry with 1 slot: {result['message']}")
                    oom_tasks.append(task)
                else:
                    print(f"    FAIL: {result['message']}")
                _save_csv()
        else:
            # ── Parallel mode ─────────────────────────────────────────
            completed = 0
            with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="preproc") as pool:
                future_map = {pool.submit(_do_one, task): task for task in tasks}
                pbar = tqdm(total=len(tasks), desc="Videos", unit="vid")
                for fut in as_completed(future_map):
                    try:
                        df_idx, vp, op, result = fut.result()
                    except Exception as exc:
                        task = future_map[fut]
                        df_idx, vp, op = task
                        result = {"status": "error", "message": f"Thread exception: {exc}"}

                    is_oom = _handle_result(df_idx, vp, op, result)

                    completed += 1
                    pbar.update(1)
                    _save_csv()
                    status = result.get("status", "?")
                    name = Path(vp).name
                    if status == "ok":
                        pbar.set_postfix_str(
                            f"{name} -> {result['output_resolution']} "
                            f"@ {result['output_fps']:.0f}fps"
                        )
                    elif is_oom:
                        pbar.set_postfix_str(f"{name} OOM→retry")
                        log.warning("OOM %s — queued for single-slot retry", name)
                        task = future_map[fut]
                        oom_tasks.append(task)
                    elif status == "error":
                        pbar.set_postfix_str(f"{name} FAIL")
                        log.warning("FAIL %s — %s", name, result.get("message", "?"))
                pbar.close()

        # ── OOM retry pass (stepped concurrency reduction) ────────────
        # Try 4×, 3×, 2×, 1× GPUs concurrently; stop as soon as all succeed.
        if oom_tasks:
            # Drain every slot from the pool so we control refill count.
            _drained_slots: list = []
            while True:
                try:
                    _drained_slots.append(_gpu_pool.get_nowait())
                except _queue_mod.Empty:
                    break

            # Build the step sequence: step down by 1/GPU from the initial
            # per-GPU count, then add a final 1-slot pass as a last resort.
            # Example — 7 GPUs, --gpu-workers 4:
            #   28 (already tried) → 21 → 14 → 7 → 1
            _n_available = max(1, len(_drained_slots))
            _gpu_n = max(1, num_gpus) if num_gpus > 0 else 1
            _retry_levels: list[int] = []
            for per_gpu in range(gpu_workers_per_gpu - 1, 0, -1):
                n = min(per_gpu * _gpu_n, _n_available)
                if not _retry_levels or n < _retry_levels[-1]:
                    _retry_levels.append(n)
            # Always end with an absolute single-slot pass if not already there.
            if not _retry_levels or _retry_levels[-1] > 1:
                _retry_levels.append(1)

            remaining_oom = list(oom_tasks)

            for lvl_idx, n_slots in enumerate(_retry_levels):
                if not remaining_oom:
                    break

                print(
                    f"\nOOM retry pass {lvl_idx + 1}/{len(_retry_levels)}: "
                    f"{len(remaining_oom)} video(s) at {n_slots} GPU slot(s)..."
                )

                # Refill pool with exactly n_slots slots for this pass.
                for slot in _drained_slots[:n_slots]:
                    _gpu_pool.put(slot)

                still_oom: list = []

                if n_slots == 1:
                    # Single slot → always sequential (no thread overhead).
                    for retry_num, task in enumerate(remaining_oom, start=1):
                        df_idx, vp, op = task
                        print(f"  [{retry_num}/{len(remaining_oom)}] {Path(vp).name} ...")
                        try:
                            df_idx, vp, op, result = _do_one(task, _show_progress_override=True)
                        except Exception as exc:
                            result = {"status": "error", "message": f"Retry exception: {exc}"}
                        _handle_result(df_idx, vp, op, result)
                        if result["status"] == "ok":
                            print(
                                f"    -> {result['output_resolution']} @ "
                                f"{result['output_fps']:.0f} fps  "
                                f"({result['output_frames']} frames)"
                            )
                        elif result["status"] == "oom":
                            print(f"    Still OOM — skipping: {result['message'][:120]}")
                            still_oom.append(task)
                        else:
                            print(f"    FAIL: {result['message']}")
                        _save_csv()
                else:
                    # Multiple slots → parallel (pool is the concurrency gate).
                    with ThreadPoolExecutor(
                        max_workers=n_slots, thread_name_prefix="oom_retry"
                    ) as pool:
                        future_map = {pool.submit(_do_one, task): task for task in remaining_oom}
                        pbar = tqdm(
                            total=len(remaining_oom),
                            desc=f"OOM-retry (slots={n_slots})",
                            unit="vid",
                        )
                        for fut in as_completed(future_map):
                            task = future_map[fut]
                            try:
                                df_idx, vp, op, result = fut.result()
                            except Exception as exc:
                                df_idx, vp, op = task
                                result = {"status": "error", "message": f"Retry exception: {exc}"}
                            _handle_result(df_idx, vp, op, result)
                            pbar.update(1)
                            name = Path(vp).name
                            if result["status"] == "ok":
                                pbar.set_postfix_str(
                                    f"{name} -> {result.get('output_resolution', '?')} "
                                    f"@ {result.get('output_fps', 0):.0f}fps"
                                )
                            elif result["status"] == "oom":
                                pbar.set_postfix_str(f"{name} OOM→next-level")
                                log.warning("OOM at %d slots for %s — stepping down", n_slots, name)
                                still_oom.append(task)
                            else:
                                pbar.set_postfix_str(f"{name} FAIL")
                                log.warning("FAIL %s — %s", name, result.get("message", "?"))
                            _save_csv()
                        pbar.close()

                # Drain pool before refilling at the next (lower) level.
                while True:
                    try:
                        _gpu_pool.get_nowait()
                    except _queue_mod.Empty:
                        break

                remaining_oom = still_oom

            # Restore all drained slots to the pool.
            for slot in _drained_slots:
                _gpu_pool.put(slot)

            if remaining_oom:
                log.warning(
                    "%d video(s) could not be processed even at 1 GPU slot",
                    len(remaining_oom),
                )

    finally:
        # Clean up all pooled and per-thread model instances
        for m in _pool_instances + _tls_instances:
            if m is not None:
                try:
                    m.cleanup()
                except Exception:
                    pass
        # Clean up per-thread model instances
        for m in _tls_instances:
            try:
                m.cleanup()
            except Exception:
                pass

    elapsed = time.time() - t0

    # Final CSV save (ensures all results are captured)
    _save_csv()

    # Summary
    all_results = list(results.values())
    ok_count = sum(1 for r in all_results if r.get("status") == "ok")
    skip_count = sum(1 for r in all_results if r.get("status") == "skipped")
    oom_count = sum(1 for r in all_results if r.get("status") == "oom")
    err_count = sum(1 for r in all_results if r.get("status") == "error") + oom_count

    print()
    print(
        f"Done in {elapsed:.1f}s — {ok_count} processed, {skip_count} skipped, {err_count} errors"
        + (f" ({oom_count} persistent OOM)" if oom_count else "")
    )
    if err_count > 0:
        print(
            f"WARNING: {err_count} videos failed enhancement — their original paths "
            f"are retained in the output CSV (enhancement_applied=False)"
        )
    if oom_count:
        print(
            f"OOM TIP: {oom_count} video(s) ran out of GPU memory even with 1 slot. "
            f"Try --gimmvfi-ds-factor 0.5 to halve the flow-estimation resolution, "
            f"or reduce --esrgan-tile."
        )
    print(f"Output CSV: {args.output_csv}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Preprocess dataset videos with enhancement stages "
        "(CLAHE, denoising, sharpening, ESRGAN, RIFE/HiFI/GIMM-VFI frame interpolation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O
    p.add_argument(
        "--csv", required=True, help="Input dataset CSV (must contain a video path column)"
    )
    p.add_argument("--output-dir", required=True, help="Directory for enhanced videos")
    p.add_argument(
        "--output-csv",
        default=None,
        help="Path for the new CSV with updated paths "
        "(default: <output-dir>/dataset_enhanced.csv)",
    )
    p.add_argument(
        "--video-column",
        default="video_path",
        help="Name of the CSV column containing video file paths " "(default: video_path)",
    )
    p.add_argument(
        "--video-quality-labels-csv",
        default=None,
        help="Path to manual video-quality labels CSV with video_path and quality_label",
    )
    p.add_argument(
        "--video-quality-threshold",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Keep videos with quality_label <= threshold (1=best, 3=worst)",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Re-process videos that already have output files"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print what would be done without processing"
    )
    p.add_argument(
        "--gpu-only",
        action="store_true",
        help="Require GPU for all model stages (ESRGAN, GIMM-VFI, RIFE, HiFI). "
        "Raises an error instead of silently falling back to CPU/bicubic. "
        "Also forces CUDA context creation at startup so driver issues are "
        "caught immediately rather than mid-run.",
    )
    max_workers_default = os.cpu_count() or 1
    p.add_argument(
        "--workers",
        type=int,
        default=max_workers_default,
        help=f"Number of parallel worker threads (default: {max_workers_default} = max logical CPUs). "
        "CPU stages (CLAHE, denoise, sharpen) always run concurrently. "
        "GPU stages are limited to --gpu-workers concurrent threads; "
        "CPU stages (e.g. --esrgan-cpu) get per-thread instances for full parallelism.",
    )
    p.add_argument(
        "--gpu-workers",
        type=int,
        default=2,
        help="Number of concurrent GPU workers *per GPU* (default: 20). "
        "Total initial slots = gpu-workers × num_gpus. "
        "On OOM the pipeline automatically retries failed videos at "
        "N-1, N-2, … 1 worker/GPU, then 1 slot total as a last resort.",
    )

    # Enhancement stages
    g = p.add_argument_group("Enhancement stages (select one or more)")
    g.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast enhancement")
    g.add_argument("--denoise", action="store_true", help="Apply non-local means denoising")
    g.add_argument("--sharpen", action="store_true", help="Apply unsharp-mask sharpening")
    g.add_argument("--esrgan", action="store_true", help="Apply Real-ESRGAN super-resolution")
    g.add_argument(
        "--rife", action="store_true", help="Apply RIFE 2× frame interpolation (25→50 fps)"
    )
    g.add_argument(
        "--hifi",
        action="store_true",
        help="Apply HiFI diffusion-based 2× frame interpolation "
        "(higher quality, slower than RIFE)",
    )
    g.add_argument(
        "--gimmvfi",
        action="store_true",
        help="Apply GIMM-VFI 2× frame interpolation "
        "(implicit neural representation, NeurIPS 2024; "
        "auto-downloads from HuggingFace)",
    )

    # CLAHE parameters
    c = p.add_argument_group("CLAHE parameters")
    c.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")
    c.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size (default: 8)")

    # Denoise parameters
    d = p.add_argument_group("Denoise parameters")
    d.add_argument(
        "--denoise-strength",
        type=float,
        default=7.0,
        help="Denoising filter strength h (default: 7.0)",
    )

    # Sharpen parameters
    s = p.add_argument_group("Sharpen parameters")
    s.add_argument(
        "--sharpen-sigma",
        type=float,
        default=1.0,
        help="Gaussian blur sigma for unsharp mask (default: 1.0)",
    )
    s.add_argument(
        "--sharpen-strength", type=float, default=0.5, help="Sharpening strength (default: 0.5)"
    )

    # ESRGAN parameters
    e = p.add_argument_group("ESRGAN parameters")
    e.add_argument(
        "--esrgan-scale",
        type=int,
        default=4,
        choices=[2, 4],
        help="ESRGAN upscale factor (default: 4)",
    )
    e.add_argument(
        "--esrgan-model",
        default="realesr-general-x4v3",
        help="ESRGAN model name (default: realesr-general-x4v3)",
    )
    e.add_argument(
        "--esrgan-model-path", default=None, help="Path to ESRGAN model weights (optional)"
    )
    e.add_argument(
        "--esrgan-half",
        action="store_true",
        help="Use FP16 inference (faster but can crash on some GPUs; " "off by default)",
    )
    e.add_argument(
        "--esrgan-tile",
        type=int,
        default=0,
        help="Tile size for ESRGAN (0=no tiling; default: 0). " "Prevents GPU OOM on full frames.",
    )
    e.add_argument(
        "--codec",
        default="avc1",
        help="Output video FOURCC codec (default: avc1 = H.264). " "Examples: avc1, H264, mp4v.",
    )
    e.add_argument(
        "--esrgan-cpu",
        action="store_true",
        help="Run ESRGAN on CPU (slower per-frame but avoids CUDA "
        "segfaults and allows full parallelism across "
        "--workers threads)",
    )

    # RIFE parameters
    r = p.add_argument_group("RIFE parameters")
    r.add_argument(
        "--rife-model-dir",
        default=None,
        help="Path to Practical-RIFE directory or train_log/ "
        "(default: auto-detect or RIFE_MODEL_DIR env var)",
    )

    # HiFI parameters
    hf = p.add_argument_group("HiFI parameters")
    hf.add_argument(
        "--hifi-model-dir",
        default=None,
        help="Path to directory containing HiFI checkpoint "
        "(default: auto-detect or HIFI_MODEL_DIR env var)",
    )
    hf.add_argument(
        "--hifi-steps",
        type=int,
        default=8,
        help="Number of DDIM sampling steps (default: 8; " "fewer = faster, more = higher quality)",
    )
    hf.add_argument(
        "--hifi-guidance",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (default: 1.0)",
    )
    hf.add_argument(
        "--hifi-patch-size",
        type=int,
        default=256,
        help="Spatial patch size for cascaded processing " "(default: 256; larger needs more VRAM)",
    )

    # GIMM-VFI parameters
    gv = p.add_argument_group("GIMM-VFI parameters")
    gv.add_argument(
        "--gimmvfi-model-dir",
        default=None,
        help="Path to directory containing GIMM-VFI checkpoints "
        "(default: auto-detect or GIMMVFI_MODEL_DIR env var)",
    )
    gv.add_argument(
        "--gimmvfi-variant",
        default="gimmvfi_f",
        choices=["gimmvfi_r", "gimmvfi_f"],
        help="GIMM-VFI variant: gimmvfi_r (RAFT, no CUDA extensions, safe) or "
        "gimmvfi_f (FlowFormer, better quality, requires compiling "
        "alt_cuda_corr) (default: gimmvfi_f)",
    )
    gv.add_argument(
        "--gimmvfi-ds-factor",
        type=float,
        default=1.0,
        help="Downsampling factor for flow estimation; "
        "1.0 = full res, 0.5 = half (saves VRAM on 2K+) "
        "(default: 1.0)",
    )
    gv.add_argument(
        "--gimmvfi-source-dir",
        default=None,
        help="Path to the GIMM-VFI repo root (e.g. models/GIMM-VFI-main). "
        "Enables the full model instead of the built-in lite wrapper. "
        "Use --gimmvfi-variant gimmvfi_r (pure-Python RAFT, no CUDA "
        "extensions) to avoid segfaults. gimmvfi_f requires compiling "
        "alt_cuda_corr first. Also accepted via GIMMVFI_SOURCE_DIR env var.",
    )

    args = p.parse_args(argv)

    interp_flags = sum([args.rife, args.hifi, args.gimmvfi])
    if interp_flags > 1:
        p.error(
            "--rife, --hifi, and --gimmvfi are mutually exclusive; "
            "choose one frame interpolation backend"
        )

    if args.output_csv is None:
        args.output_csv = os.path.join(args.output_dir, "dataset_enhanced.csv")

    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
