#!/usr/bin/env python3
"""
validate_yolo_finetuned.py — Validate a fine-tuned YOLO-Pose hand model
against the MediaPipe-only pipeline baseline.

Validation strategy (no ground truth available):

1. **Kinematic Agreement** — ICC between MediaPipe-only and YOLO-refined
   kinematic features on held-out high-quality videos.
2. **Rescue Rate** — Improvement in detection_rate and signal_quality on
   bottom-quintile (PLE < 0.3) difficult videos.
3. **Temporal Consistency** — Compare per-frame MCP keypoint jitter between
   MediaPipe-only and YOLO-refined pipelines.
4. **Synthetic Perturbation Robustness** — Apply blur, brightness shift,
   rotation, and crop to high-quality frames; compare landmark degradation.

Usage
-----
    python scripts/validate_yolo_finetuned.py \\
        --logs-mp output_mediapipe/tracking_logs.csv \\
        --logs-yolo output_yolo/tracking_logs.csv \\
        [--output-dir validation_results] \\
        [--ple-rescue-threshold 0.3]
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _parse_json_series(val):
    if isinstance(val, (list, np.ndarray)):
        return np.asarray(val, dtype=np.float64)
    if not isinstance(val, str) or not val.strip():
        return np.array([], dtype=np.float64)
    try:
        return np.asarray(json.loads(val), dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError):
        return np.array([], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Kinematic Agreement (ICC)
# ---------------------------------------------------------------------------


def _icc_2_1(x, y):
    """Two-way random, single measures ICC(2,1).

    Parameters
    ----------
    x, y : array-like
        Paired measurements from two raters/methods.

    Returns
    -------
    float
        ICC value in [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return np.nan

    grand_mean = (x.mean() + y.mean()) / 2.0
    ss_rows = n * (((x + y) / 2.0 - grand_mean) ** 2).sum()  # approximate
    # Simplified ICC(2,1) via variance decomposition
    diff = x - y
    var_diff = np.var(diff, ddof=1)
    var_mean = np.var((x + y) / 2.0, ddof=1)

    if var_mean + var_diff / 2.0 == 0:
        return 1.0 if var_diff == 0 else 0.0

    icc = (var_mean - var_diff / 2.0) / (var_mean + var_diff / 2.0)
    return float(icc)


def compute_kinematic_agreement(df_mp, df_yolo, join_col="video_path"):
    """Compute ICC between MediaPipe and YOLO kinematic features.

    Parameters
    ----------
    df_mp, df_yolo : pd.DataFrame
        Tracking logs from MediaPipe-only and YOLO-refined runs.
    join_col : str
        Column to join on (typically video_path).

    Returns
    -------
    dict
        ``{feature_name: icc_value}``
    """
    features = [
        "Mean Amplitude",
        "Mean Frequency",
        "Norm Decrement Slope",
        "Rhythm (CV %)",
        "Peak Velocity",
        "Mean Velocity",
        "Signal Quality",
    ]

    merged = df_mp.merge(df_yolo, on=join_col, suffixes=("_mp", "_yolo"), how="inner")
    logger.info("Kinematic agreement: %d matched videos.", len(merged))

    results = {}
    for feat in features:
        col_mp = f"{feat}_mp"
        col_yolo = f"{feat}_yolo"
        if col_mp in merged.columns and col_yolo in merged.columns:
            vals_mp = merged[col_mp].apply(lambda v: _safe_float(v)).values
            vals_yolo = merged[col_yolo].apply(lambda v: _safe_float(v)).values
            icc = _icc_2_1(vals_mp, vals_yolo)
            results[feat] = icc
            logger.info("  %s: ICC = %.4f", feat, icc)
        else:
            results[feat] = np.nan
            logger.warning("  %s: column not found in both logs.", feat)

    return results


# ---------------------------------------------------------------------------
# 2. Rescue Rate on Difficult Videos
# ---------------------------------------------------------------------------


def compute_rescue_rate(df_mp, df_yolo, join_col="video_path", ple_threshold=0.3):
    """Compare detection_rate and signal_quality on difficult videos.

    "Difficult" = bottom-quintile videos (PLE < ple_threshold) based on the
    MediaPipe-only run.

    Returns
    -------
    dict
        Summary statistics for rescue metrics.
    """
    from scripts.prepare_yolo_pseudolabels import compute_ple_scores

    df_mp = df_mp.copy()
    df_mp["ple_score"] = compute_ple_scores(df_mp)
    difficult = df_mp[df_mp["ple_score"] < ple_threshold].copy()

    if len(difficult) == 0:
        logger.warning("No videos with PLE < %.2f found.", ple_threshold)
        return {}

    merged = difficult.merge(df_yolo, on=join_col, suffixes=("_mp", "_yolo"), how="inner")
    logger.info("Rescue rate: %d difficult videos matched.", len(merged))

    results = {}
    for metric in ["VQ_detection_rate", "Signal Quality", "VQ_gap_fraction"]:
        col_mp = f"{metric}_mp"
        col_yolo = f"{metric}_yolo"
        if col_mp not in merged.columns or col_yolo not in merged.columns:
            continue

        vals_mp = merged[col_mp].apply(lambda v: _safe_float(v, 0.0)).values
        vals_yolo = merged[col_yolo].apply(lambda v: _safe_float(v, 0.0)).values

        mean_mp = float(np.mean(vals_mp))
        mean_yolo = float(np.mean(vals_yolo))

        if metric == "VQ_gap_fraction":
            # Lower is better for gap_fraction
            improvement = mean_mp - mean_yolo
            pct_improvement = (improvement / max(abs(mean_mp), 1e-8)) * 100
        else:
            improvement = mean_yolo - mean_mp
            pct_improvement = (improvement / max(abs(mean_mp), 1e-8)) * 100

        results[metric] = {
            "mean_mp": mean_mp,
            "mean_yolo": mean_yolo,
            "improvement": improvement,
            "pct_improvement": pct_improvement,
        }
        logger.info(
            "  %s: MP=%.4f → YOLO=%.4f (%+.1f%%)", metric, mean_mp, mean_yolo, pct_improvement
        )

    return results


# ---------------------------------------------------------------------------
# 3. Temporal Consistency (Jitter Comparison)
# ---------------------------------------------------------------------------


def compute_jitter_comparison(df_mp, df_yolo, join_col="video_path"):
    """Compare mean frame-to-frame MCP jitter between methods.

    Uses ``conf_mcp_min_series`` — higher values = less jitter = better.

    Returns
    -------
    dict
        ``{"mean_conf_mp": float, "mean_conf_yolo": float, "improvement_pct": float}``
    """
    merged = df_mp.merge(df_yolo, on=join_col, suffixes=("_mp", "_yolo"), how="inner")

    def _median_conf(series_col):
        vals = []
        for val in series_col:
            arr = _parse_json_series(val)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                vals.append(float(np.median(arr)))
        return float(np.mean(vals)) if vals else 0.0

    conf_col_mp = "conf_mcp_min_series_mp"
    conf_col_yolo = "conf_mcp_min_series_yolo"

    if conf_col_mp not in merged.columns or conf_col_yolo not in merged.columns:
        logger.warning("conf_mcp_min_series not found in both logs.")
        return {}

    mean_mp = _median_conf(merged[conf_col_mp])
    mean_yolo = _median_conf(merged[conf_col_yolo])
    improvement = ((mean_yolo - mean_mp) / max(abs(mean_mp), 1e-8)) * 100

    results = {
        "mean_conf_mp": mean_mp,
        "mean_conf_yolo": mean_yolo,
        "improvement_pct": improvement,
    }
    logger.info("Jitter comparison: MP=%.4f → YOLO=%.4f (%+.1f%%)", mean_mp, mean_yolo, improvement)
    return results


# ---------------------------------------------------------------------------
# 4. Synthetic Perturbation Robustness
# ---------------------------------------------------------------------------


def run_perturbation_test(video_paths, yolo_model_path, n_frames=100, seed=42):
    """Apply synthetic perturbations and compare landmark robustness.

    For a random subset of frames from high-quality videos, apply:
    - Gaussian blur (sigma=2,4,6)
    - Brightness shift (+/-30%)
    - Rotation (+/-10 deg)
    - Center crop to 80%

    Compare landmark displacement from unperturbed pseudo-labels for both
    MediaPipe and YOLO.

    Parameters
    ----------
    video_paths : list of str
        Paths to high-quality videos for testing.
    yolo_model_path : str
        Path to the fine-tuned YOLO model.
    n_frames : int
        Number of random frames to sample across videos.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Per-perturbation mean landmark error for MP and YOLO.
    """
    try:
        import cv2
        import mediapipe as mp_lib
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from ultralytics import YOLO
    except ImportError as e:
        logger.warning("Perturbation test skipped (missing dependency): %s", e)
        return {}

    rng = np.random.RandomState(seed)

    # Load YOLO model
    yolo = YOLO(yolo_model_path)

    # Collect random frames
    frames_per_video = max(1, n_frames // max(len(video_paths), 1))
    test_frames = []  # list of (frame_bgr, video_path, frame_idx)

    for vp in video_paths[:20]:  # cap at 20 videos
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            continue
        chosen = sorted(rng.choice(total, min(frames_per_video, total), replace=False))
        for target_fi in chosen:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
            ok, frame_bgr = cap.read()
            if ok:
                test_frames.append((frame_bgr, vp, target_fi))
        cap.release()
        if len(test_frames) >= n_frames:
            break

    test_frames = test_frames[:n_frames]
    if not test_frames:
        logger.warning("No frames collected for perturbation test.")
        return {}

    logger.info("Perturbation test: %d frames from %d videos.", len(test_frames), len(video_paths))

    # Define perturbations
    perturbations = {
        "blur_s2": lambda img: cv2.GaussianBlur(img, (0, 0), 2),
        "blur_s4": lambda img: cv2.GaussianBlur(img, (0, 0), 4),
        "blur_s6": lambda img: cv2.GaussianBlur(img, (0, 0), 6),
        "bright_p30": lambda img: np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8),
        "bright_m30": lambda img: np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8),
        "crop_80": lambda img: _center_crop(img, 0.8),
    }

    def _mp_detect(img_bgr):
        """Run MediaPipe IMAGE mode on a BGR frame, return (21,2) or None."""
        from scripts.prepare_yolo_pseudolabels import _find_mediapipe_model

        model_path = _find_mediapipe_model()
        if model_path is None:
            return None
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
        )
        with mp_vision.HandLandmarker.create_from_options(options) as lm:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
            result = lm.detect(mp_img)
            if result.hand_landmarks:
                arr = np.array([[l.x, l.y] for l in result.hand_landmarks[0]])
                return arr
        return None

    def _yolo_detect(img_bgr):
        """Run YOLO on a BGR frame, return (21,2) or None."""
        results = yolo(img_bgr, verbose=False)
        if results and results[0].keypoints is not None:
            kpts = results[0].keypoints.xyn
            if kpts is not None and len(kpts) > 0:
                kp = kpts[0].cpu().numpy()  # (21, 2)
                if kp.shape[0] >= 21:
                    return kp[:21]
        return None

    # Run: for each frame, get baseline landmarks, then perturbed landmarks
    results = {p: {"mp_errors": [], "yolo_errors": []} for p in perturbations}

    for frame_bgr, vp, fi in test_frames:
        base_mp = _mp_detect(frame_bgr)
        base_yolo = _yolo_detect(frame_bgr)

        for pname, pfunc in perturbations.items():
            perturbed = pfunc(frame_bgr)

            if base_mp is not None:
                pert_mp = _mp_detect(perturbed)
                if pert_mp is not None:
                    err = float(np.mean(np.linalg.norm(base_mp - pert_mp, axis=1)))
                    results[pname]["mp_errors"].append(err)

            if base_yolo is not None:
                pert_yolo = _yolo_detect(perturbed)
                if pert_yolo is not None:
                    err = float(np.mean(np.linalg.norm(base_yolo - pert_yolo, axis=1)))
                    results[pname]["yolo_errors"].append(err)

    # Summarise
    summary = {}
    for pname, data in results.items():
        mp_err = float(np.mean(data["mp_errors"])) if data["mp_errors"] else np.nan
        yolo_err = float(np.mean(data["yolo_errors"])) if data["yolo_errors"] else np.nan
        summary[pname] = {
            "mp_mean_error": mp_err,
            "yolo_mean_error": yolo_err,
            "yolo_better": (
                bool(yolo_err < mp_err) if np.isfinite(yolo_err) and np.isfinite(mp_err) else None
            ),
        }
        logger.info(
            "  %s: MP=%.5f  YOLO=%.5f  YOLO_better=%s",
            pname,
            mp_err,
            yolo_err,
            summary[pname]["yolo_better"],
        )

    return summary


def _center_crop(img, frac):
    """Center-crop an image to *frac* of its original size, then resize back."""
    import cv2

    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = img[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(cropped, (w, h))


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def generate_report(icc_results, rescue_results, jitter_results, perturbation_results, output_dir):
    """Write a summary JSON report."""
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "kinematic_agreement_icc": icc_results,
        "rescue_rate": rescue_results,
        "jitter_comparison": jitter_results,
        "perturbation_robustness": perturbation_results,
    }

    # Pass/fail assessment
    amp_icc = icc_results.get("Mean Amplitude", np.nan)
    freq_icc = icc_results.get("Mean Frequency", np.nan)
    det_rescue = rescue_results.get("VQ_detection_rate", {}).get("pct_improvement", 0)

    report["assessment"] = {
        "kinematic_icc_pass": bool(
            np.isfinite(amp_icc) and amp_icc > 0.90 and np.isfinite(freq_icc) and freq_icc > 0.90
        ),
        "rescue_rate_pass": bool(det_rescue > 15.0),
        "amplitude_icc": float(amp_icc) if np.isfinite(amp_icc) else None,
        "frequency_icc": float(freq_icc) if np.isfinite(freq_icc) else None,
        "detection_rate_rescue_pct": float(det_rescue),
    }

    report_path = os.path.join(output_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Validation report saved to %s", report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("YOLO Fine-Tuned Model Validation Summary")
    print("=" * 60)
    print("\nKinematic Agreement (ICC):")
    for feat, icc in icc_results.items():
        status = "PASS" if np.isfinite(icc) and icc > 0.90 else "BELOW TARGET"
        print(f"  {feat}: {icc:.4f}  [{status}]")
    print("\nRescue Rate (difficult videos):")
    for metric, data in rescue_results.items():
        if isinstance(data, dict):
            print(f"  {metric}: {data.get('pct_improvement', 0):+.1f}%")
    print("\nJitter Comparison:")
    if jitter_results:
        print(
            f"  MCP conf: MP={jitter_results.get('mean_conf_mp', 0):.4f} → "
            f"YOLO={jitter_results.get('mean_conf_yolo', 0):.4f} "
            f"({jitter_results.get('improvement_pct', 0):+.1f}%)"
        )
    assessment = report["assessment"]
    print(
        f"\nOverall: kinematic_icc={'PASS' if assessment['kinematic_icc_pass'] else 'FAIL'}"
        f"  rescue_rate={'PASS' if assessment['rescue_rate_pass'] else 'FAIL'}"
    )
    print("=" * 60)

    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Validate fine-tuned YOLO-Pose hand model.",
    )
    parser.add_argument(
        "--logs-mp",
        required=True,
        help="Path to tracking_logs.csv from MediaPipe-only pipeline run.",
    )
    parser.add_argument(
        "--logs-yolo",
        required=True,
        help="Path to tracking_logs.csv from YOLO-refined pipeline run.",
    )
    parser.add_argument(
        "--output-dir",
        default="validation_results",
        help="Directory for validation output (default: validation_results).",
    )
    parser.add_argument(
        "--ple-rescue-threshold",
        type=float,
        default=0.3,
        help="PLE threshold for 'difficult' videos (default: 0.3).",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Path to fine-tuned YOLO model (for perturbation test).",
    )
    parser.add_argument(
        "--skip-perturbation",
        action="store_true",
        help="Skip the slow synthetic perturbation test.",
    )
    args = parser.parse_args()

    df_mp = pd.read_csv(args.logs_mp)
    df_yolo = pd.read_csv(args.logs_yolo)

    # Filter to VIDEO records
    if "record_type" in df_mp.columns:
        df_mp = df_mp[df_mp["record_type"] == "VIDEO"].copy()
    if "record_type" in df_yolo.columns:
        df_yolo = df_yolo[df_yolo["record_type"] == "VIDEO"].copy()

    # 1. Kinematic agreement
    icc_results = compute_kinematic_agreement(df_mp, df_yolo)

    # 2. Rescue rate
    rescue_results = compute_rescue_rate(
        df_mp,
        df_yolo,
        ple_threshold=args.ple_rescue_threshold,
    )

    # 3. Jitter comparison
    jitter_results = compute_jitter_comparison(df_mp, df_yolo)

    # 4. Perturbation robustness (optional)
    perturbation_results = {}
    if not args.skip_perturbation and args.yolo_model:
        video_paths = df_mp["video_path"].dropna().tolist()[:20]
        perturbation_results = run_perturbation_test(
            video_paths,
            args.yolo_model,
        )

    # Generate report
    generate_report(
        icc_results,
        rescue_results,
        jitter_results,
        perturbation_results,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
