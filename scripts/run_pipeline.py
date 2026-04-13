"""
scripts/run_pipeline.py — Run the hand-landmark inference pipeline.

Example
-------
    # Using the default config lookup (PS_CONFIG_PATH env var, then
    # configs/config.yaml, then configs/config.example.yaml):
    python scripts/run_pipeline.py

    # Or with an explicit config file:
    python scripts/run_pipeline.py --config configs/my_config.yaml

Pipeline configuration lives in YAML files under configs/. See
configs/config.example.yaml for the full schema.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from ps_kinematics import HandLandmarkProcessor
from ps_kinematics.config import load_pipeline_config

multiprocessing.freeze_support()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _validate_config(cfg: dict, yolo_only: bool = False) -> None:
    """Raise an error if required CONFIG paths are not set."""
    required = ["vid_score_path", "log_csv_path", "save_dir"]
    if not yolo_only:
        # hand_path is only needed for the MediaPipe primary tracker.
        required.append("hand_path")
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise RuntimeError(
            f"Required pipeline config paths are not set: {missing}. "
            "Fill them in under 'paths:' in your configs/config.yaml "
            "(see configs/config.example.yaml for a template)."
        )


def _normalize_video_path_for_matching(path_value) -> str:
    """Build a stable key for matching videos across CSV sources."""
    if pd.isna(path_value):
        return ""

    out = str(path_value).strip().replace("\\", "/").lower()
    out = re.sub(r"_cl_sh_sr4x(?:_gimmvfi)?(?=\.[^./\\]+$|$)", "", out)
    name = os.path.basename(out)

    if "segmented_video_" in name:
        tail = name.split("segmented_video_", 1)[1]
        tail = re.sub(r"_cropped(?=\.[^./\\]+$|$)", "", tail)
        return f"seg::{tail}"

    visit_match = re.search(r"visit\s*(\d+)", out)
    pom_match = re.search(r"(pom\d+vd\d+)", out)
    hand_tail_match = re.search(r"((?:on|off)_4[lr].*)", name)
    if pom_match and hand_tail_match:
        visit_part = visit_match.group(1) if visit_match else "na"
        hand_tail = re.sub(r"\s+", " ", hand_tail_match.group(1)).strip()
        hand_tail = re.sub(r"_cropped(?=\.[^./\\]+$|$)", "", hand_tail)
        return f"clip::v{visit_part}_{pom_match.group(1)}_{hand_tail}"

    name = re.sub(r"^enhanced_videos\d*_", "", name)
    name = re.sub(r"enhanced_videos\d*", "cropped_videos_output", name)
    name = re.sub(r"^clahe_sharpen_esrgan_videos\d*_", "", name)
    name = re.sub(r"clahe_sharpen_esrgan_videos\d*", "cropped_videos_output", name)
    if name.startswith("visit "):
        name = f"cropped_videos_output_{name}"
    return re.sub(r"\s+", " ", name).strip()


def _prepare_video_quality_filtered_csv(
    video_csv_path: str,
    video_quality_labels_csv_path: str | None,
    video_quality_threshold: int,
    output_dir: str,
) -> str:
    """Filter the input video CSV by manual quality labels and write a temp CSV."""
    if video_quality_threshold not in (1, 2, 3):
        raise ValueError("video_quality_threshold must be one of: 1, 2, 3")

    if video_quality_labels_csv_path is None:
        if video_quality_threshold < 3:
            raise ValueError("video_quality_threshold < 3 requires video_quality_labels_csv_path")
        return video_csv_path

    if not os.path.exists(video_quality_labels_csv_path):
        raise FileNotFoundError(
            f"Video quality labels CSV not found: {video_quality_labels_csv_path}"
        )
    if os.path.getsize(video_quality_labels_csv_path) == 0:
        raise ValueError(
            "Video quality labels CSV is empty (0 bytes): " f"{video_quality_labels_csv_path}"
        )

    video_df = pd.read_csv(video_csv_path)
    if "video_path" not in video_df.columns:
        raise RuntimeError(
            "Input video CSV must contain a 'video_path' column for quality filtering"
        )

    try:
        labels_df = pd.read_csv(video_quality_labels_csv_path)
    except EmptyDataError as exc:
        raise ValueError(
            "Video quality labels CSV has no parseable columns: "
            f"{video_quality_labels_csv_path}. Expected columns: video_path, quality_label"
        ) from exc

    required_cols = {"video_path", "quality_label"}
    missing = required_cols.difference(labels_df.columns)
    if missing:
        raise RuntimeError(
            "Video quality labels CSV is missing required column(s): " + ", ".join(sorted(missing))
        )

    labels_df = labels_df.copy()
    labels_df["quality_label"] = pd.to_numeric(labels_df["quality_label"], errors="coerce")
    labels_df = labels_df.dropna(subset=["video_path", "quality_label"]).copy()
    labels_df["quality_label"] = labels_df["quality_label"].astype(int)
    labels_df = labels_df[labels_df["quality_label"].between(1, 3)].copy()
    if labels_df.empty:
        raise RuntimeError(
            "No valid rows in video quality labels CSV after cleaning "
            "(expected quality_label values 1..3)"
        )

    labels_df["_quality_key"] = labels_df["video_path"].map(_normalize_video_path_for_matching)
    labels_df = labels_df.dropna(subset=["_quality_key"]).copy()
    labels_df = (
        labels_df.sort_values(["_quality_key", "quality_label"])
        .drop_duplicates(subset=["_quality_key"], keep="first")
        .copy()
    )

    video_df = video_df.copy()
    video_df["_quality_key"] = video_df["video_path"].map(_normalize_video_path_for_matching)
    quality_map = labels_df.set_index("_quality_key")["quality_label"]
    matched_labels = video_df["_quality_key"].map(quality_map)
    keep_mask = matched_labels.notna() & (matched_labels <= int(video_quality_threshold))

    n_before = len(video_df)
    n_matched = int(matched_labels.notna().sum())
    filtered_df = video_df[keep_mask].copy()
    filtered_df = filtered_df.drop(columns=["_quality_key"], errors="ignore")

    print(
        "[Pipeline] Manual video quality filter "
        f"(threshold <= {video_quality_threshold}): matched {n_matched}/{n_before}, "
        f"kept {len(filtered_df)}/{n_before} ({n_before - len(filtered_df)} dropped)"
    )

    if filtered_df.empty:
        raise RuntimeError("No videos remain after applying manual video quality filter")

    os.makedirs(output_dir, exist_ok=True)
    filtered_csv_path = os.path.join(
        output_dir,
        f"video_quality_filtered_input_q{video_quality_threshold}.csv",
    )
    filtered_df.to_csv(filtered_csv_path, index=False)
    return filtered_csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PD-PS landmark pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the pipeline YAML config. "
            "Defaults to $PS_CONFIG_PATH, then configs/config.yaml, "
            "then configs/config.example.yaml."
        ),
    )
    parser.add_argument(
        "--n", type=int, default=None, help="number of videos to process (None = all)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling videos")
    parser.add_argument(
        "--workers", type=int, default=None, help="number of parallel workers to use"
    )
    parser.add_argument(
        "--video-quality-labels-csv",
        type=str,
        default=None,
        help=(
            "Path to manual video quality labels CSV with columns "
            "video_path, quality_label (1=best, 3=worst)."
        ),
    )
    parser.add_argument(
        "--video-quality-threshold",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help=(
            "Keep only videos with manual quality_label <= threshold. "
            "Defaults to the config's processing.video_quality_threshold (or 3)."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        nargs="?",
        const=True,
        default=False,
        metavar="LOG_PATH",
        help=(
            "Re-run only TIMEOUT/ERROR/CRASH videos from an existing tracking log "
            "with no per-video time limit. "
            "Optionally provide a log path; defaults to log_csv_path from the config. "
            "Successful (VIDEO) records from the log are preserved in the output."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        metavar="N",
        help=(
            "After the initial run (or --retry-failed pass), keep retrying "
            "remaining TIMEOUT/ERROR/CRASH videos up to N additional times. "
            "Use with --retry-failed to loop until all videos succeed. "
            "Set to a large number (e.g. 99) to retry indefinitely until done. "
            "Default: 0 (no automatic looping)."
        ),
    )
    # ── Multi-node support ────────────────────────────────────────────────
    # Videos are striped across nodes (node k processes indices k, k+N, k+2N, ...).
    # Each node writes a separate tracking_logs_rank<k>.csv.
    parser.add_argument(
        "--node-rank",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="0-based rank of this node in the job array (default: $SLURM_ARRAY_TASK_ID or 0)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)),
        help="total nodes in the job array (default: $SLURM_ARRAY_TASK_COUNT or 1)",
    )
    parser.add_argument(
        "--workers-list",
        type=str,
        default=os.environ.get("WORKERS_LIST", ""),
        help="Comma-separated worker counts for all nodes (e.g. '120,94,30,33'). "
        "When provided, each node receives a contiguous video slice sized "
        "proportionally to its worker count so all nodes finish at the same time.",
    )
    parser.add_argument(
        "--yolo-pd-finetune",
        action="store_true",
        default=False,
        help=(
            "Run the YOLO PD fine-tuning pipeline: (1) standard MediaPipe run → "
            "(2) pseudo-label extraction from top-quality videos → "
            "(3) YOLO-Pose fine-tuning on pseudo-labels → "
            "(4) re-run pipeline with YOLO refinement enabled. "
            "Requires the standard pipeline to have been run first "
            "(tracking_logs.csv must exist)."
        ),
    )
    parser.add_argument(
        "--enable-cudnn",
        action="store_true",
        default=False,
        help=(
            "Enable cuDNN in the YOLO training subprocess. "
            "Disabled by default to work around a SIGSEGV in certain cuDNN "
            "versions on specific GPU compute capabilities. "
            "Safe to enable on GPUs where cuDNN is known to be stable, "
            "giving ~30%% faster convolutions."
        ),
    )
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        default=False,
        help=(
            "Run a fully YOLO-based inference pipeline, bypassing MediaPipe "
            "entirely. YOLO-Pose is used as the primary detector on every "
            "frame; frame-to-frame tracking and main-track selection logic "
            "(rotation-score, choose_main_track) are identical to the "
            "MediaPipe pipeline. Requires a trained YOLO-Pose checkpoint "
            "(set YOLO_HAND_MODEL_PATH in the tuning profile or pass via "
            "--yolo-hand-model-path). Run --yolo-pd-finetune first to "
            "produce models/yolo_pd_hand_pose.pt if you don't have a "
            "checkpoint yet."
        ),
    )
    parser.add_argument(
        "--yolo-hand-model-path",
        type=str,
        default=None,
        help=(
            "Path to the YOLO-Pose hand model checkpoint (.pt) used by "
            "--yolo-only (and --yolo-pd-finetune refinement pass). "
            "Overrides YOLO_HAND_MODEL_PATH in the tuning profile."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # ── Load pipeline config from YAML ─────────────────────────────────────
    config = load_pipeline_config(args.config)
    tuning_overrides = dict(config.get("tuning_overrides") or {})

    # Validate config now that we know which mode is active.
    _validate_config(config, yolo_only=args.yolo_only)

    # ── Video quality filter ───────────────────────────────────────────────
    quality_labels_csv = (
        args.video_quality_labels_csv
        if args.video_quality_labels_csv is not None
        else config.get("video_quality_labels_csv_path")
    )
    quality_threshold = (
        int(args.video_quality_threshold)
        if args.video_quality_threshold is not None
        else int(config.get("video_quality_threshold", 3))
    )

    config["vid_score_path"] = _prepare_video_quality_filtered_csv(
        video_csv_path=config["vid_score_path"],
        video_quality_labels_csv_path=quality_labels_csv,
        video_quality_threshold=quality_threshold,
        output_dir=config["save_dir"],
    )

    # Propagate --enable-cudnn to inference worker subprocesses via env var.
    if args.enable_cudnn:
        os.environ["ENABLE_CUDNN"] = "1"

    # Resolve the repo root for SUPERRES/YOLO model path fallbacks. When the
    # config is loaded from the shipped configs/ directory this is the repo
    # root; users pointing --config at an arbitrary location can still use
    # absolute model paths in their tuning YAML.
    _config_path = Path(config.get("_config_path", ""))
    repo_root = _config_path.parent.parent if _config_path.exists() else Path.cwd()

    # ── YOLO-only inference pipeline ───────────────────────────────────────
    if args.yolo_only:
        from ps_kinematics.utils import YOLO_PD_MODEL_PATH

        _yolo_model_path = args.yolo_hand_model_path
        if _yolo_model_path is None:
            _yolo_model_path = tuning_overrides.get(
                "YOLO_HAND_MODEL_PATH",
                str(repo_root / YOLO_PD_MODEL_PATH),
            )
        _yolo_model_path = str(Path(_yolo_model_path).resolve())

        if not os.path.exists(_yolo_model_path):
            raise FileNotFoundError(
                f"[--yolo-only] YOLO model not found: {_yolo_model_path}\n"
                "Run --yolo-pd-finetune first to produce a fine-tuned model, or\n"
                "point --yolo-hand-model-path at an existing YOLO-Pose .pt checkpoint."
            )

        print(f"[YOLO-Only] Primary detector: {_yolo_model_path}")

        tuning_overrides["USE_YOLO_ONLY"] = True
        tuning_overrides["USE_YOLO_HAND"] = True
        tuning_overrides["YOLO_HAND_MODEL_PATH"] = _yolo_model_path

        # Use a separate output log so the YOLO-only results don't overwrite
        # an existing MediaPipe baseline log.
        _stem, _ext = os.path.splitext(config["log_csv_path"])
        config["log_csv_path"] = f"{_stem}_yolo_only{_ext}"
        print(f"[YOLO-Only] Output log: {config['log_csv_path']}")

    # ── YOLO PD fine-tuning orchestration ──────────────────────────────────
    if args.yolo_pd_finetune:
        from ps_kinematics.refinement.yolo import train_yolo_pd_hand_model
        from ps_kinematics.utils import (
            YOLO_PD_MODEL_PATH,
            YOLO_PD_TRAIN_BATCH,
            YOLO_PD_TRAIN_EPOCHS,
            YOLO_PD_TRAIN_IMGSZ,
        )
        from scripts.prepare_yolo_pseudolabels import extract_dataset

        log_csv = config["log_csv_path"]
        if not os.path.exists(log_csv):
            raise FileNotFoundError(
                f"tracking_logs.csv not found at {log_csv}. "
                "Run the standard pipeline first before --yolo-pd-finetune."
            )

        dataset_dir = os.path.join(os.path.dirname(log_csv), "yolo_pd_dataset")
        dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")
        if os.path.exists(dataset_yaml):
            print(
                f"[YOLO-PD] Step 1/3: Existing dataset found at {dataset_dir}, skipping extraction."
            )
        else:
            print(f"[YOLO-PD] Step 1/3: Extracting pseudo-labels to {dataset_dir} ...")
            dataset_yaml = extract_dataset(
                tracking_logs_csv=log_csv,
                output_dir=dataset_dir,
                hand_model_path=config.get("hand_path"),
            )

        # Base model for fine-tuning: set YOLO_HAND_MODEL_PATH in the tuning
        # profile to point to an existing YOLO-Pose checkpoint, or leave unset
        # to auto-download the default Ultralytics yolov8x-pose.pt base model.
        base_model = tuning_overrides.get("YOLO_HAND_MODEL_PATH", "yolov8x-pose.pt")
        output_model = str(repo_root / YOLO_PD_MODEL_PATH)
        # Default to GPU 0 — multi-GPU DDP requires container IPC namespace
        # sharing (--container-ipc=host in SLURM) which is not available on
        # all clusters.  Single-GPU training with AMP is reliable everywhere.
        device = tuning_overrides.get("YOLO_HAND_TRAIN_DEVICE", 0)

        print(f"[YOLO-PD] Step 2/3: Fine-tuning YOLO from {base_model} ...")
        train_yolo_pd_hand_model(
            dataset_yaml=dataset_yaml,
            base_model_path=base_model,
            output_path=output_model,
            epochs=YOLO_PD_TRAIN_EPOCHS,
            imgsz=YOLO_PD_TRAIN_IMGSZ,
            batch=YOLO_PD_TRAIN_BATCH,
            device=device,
            # AMP disabled: Ultralytics runs a pre-training validation forward
            # pass that can SIGSEGV with certain cuDNN versions, regardless
            # of GPU architecture.
            amp=False,
            cudnn_enabled=args.enable_cudnn,
        )

        print("[YOLO-PD] Step 3/3: Re-running pipeline with fine-tuned model ...")
        tuning_overrides["USE_YOLO_HAND"] = True
        tuning_overrides["YOLO_HAND_MODEL_PATH"] = output_model

        # Update log path to avoid overwriting the original
        _stem, _ext = os.path.splitext(config["log_csv_path"])
        config["log_csv_path"] = f"{_stem}_yolo_refined{_ext}"
        print(f"[YOLO-PD] Output log: {config['log_csv_path']}")

    # ── Persist tuning overrides back into config for worker propagation ───
    config["tuning_overrides"] = tuning_overrides

    # When distributing across nodes, append rank to log path so each node
    # writes its own CSV (avoids concurrent writes to the same file).
    if args.num_nodes > 1:
        _log = config["log_csv_path"]
        _stem, _ext = os.path.splitext(_log)
        config["log_csv_path"] = f"{_stem}_rank{args.node_rank:03d}{_ext}"

    # Compute proportional video slice boundaries for this node.
    # Each node gets a contiguous chunk whose size is proportional to its
    # worker count, so all nodes finish at roughly the same wall-clock time.
    start_frac = end_frac = None
    if args.workers_list and args.num_nodes > 1:
        import itertools

        try:
            wlist = [int(x) for x in args.workers_list.split(",") if x.strip()]
            if len(wlist) == args.num_nodes and sum(wlist) > 0:
                total_w = sum(wlist)
                cumsum = [0] + list(itertools.accumulate(wlist))
                start_frac = cumsum[args.node_rank] / total_w
                end_frac = cumsum[args.node_rank + 1] / total_w
        except (ValueError, IndexError):
            pass  # fall back to even striping if list is malformed

    # Determine the number of videos to process (CLI > config > all).
    n_videos = args.n if args.n is not None else config.get("n_videos")

    # ── Instantiate and run ────────────────────────────────────────────────
    processor = HandLandmarkProcessor(config)
    processor.process_hand_videos_random(
        n=n_videos,
        seed=args.seed,
        num_workers=args.workers,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
        start_frac=start_frac,
        end_frac=end_frac,
        retry_failed=args.retry_failed,
    )

    # ── Auto-retry loop (--max-retries) ────────────────────────────────────
    # After the initial run, keep retrying remaining TIMEOUT/ERROR/CRASH
    # records until all succeed or the retry budget is exhausted.
    # Each retry pass doubles the worker timeout so that videos which
    # narrowly missed the previous deadline get more headroom.
    if args.max_retries > 0:
        _log_path = processor.log_csv_path
        _FAILED = {"ERROR", "TIMEOUT", "CRASH"}
        _base_retry_timeout = 3600  # first retry: 1 h
        for _retry_n in range(1, args.max_retries + 1):
            try:
                _df = pd.read_csv(_log_path)
            except Exception as _e:
                print(f"[RetryLoop] Cannot read log '{_log_path}': {_e}; stopping.")
                break
            _n_failed = (
                int(_df["record_type"].isin(_FAILED).sum()) if "record_type" in _df.columns else 0
            )
            if _n_failed == 0:
                print(
                    f"[RetryLoop] All videos succeeded — stopping after {_retry_n - 1} extra retry pass(es)."
                )
                break
            _retry_timeout = _base_retry_timeout * (2 ** (_retry_n - 1))
            print(
                f"[RetryLoop] Retry pass {_retry_n}/{args.max_retries}: "
                f"{_n_failed} failed video(s) remaining (timeout={_retry_timeout}s)."
            )
            processor.process_hand_videos_random(
                n=None,
                seed=args.seed,
                num_workers=args.workers,
                node_rank=args.node_rank,
                num_nodes=args.num_nodes,
                start_frac=None,
                end_frac=None,
                retry_failed=_log_path,
                retry_timeout=_retry_timeout,
            )
        else:
            # Loop exhausted — report any remaining failures.
            try:
                _df = pd.read_csv(_log_path)
                _n_failed = (
                    int(_df["record_type"].isin(_FAILED).sum())
                    if "record_type" in _df.columns
                    else 0
                )
                if _n_failed:
                    print(
                        f"[RetryLoop] {_n_failed} video(s) still failed after {args.max_retries} retry pass(es). "
                        "Increase --max-retries or investigate errors in the log."
                    )
            except Exception:
                pass


if __name__ == "__main__":
    main()
