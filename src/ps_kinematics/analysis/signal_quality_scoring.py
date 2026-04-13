"""Signal quality scoring: label-free objective for tracking quality."""

import json as _json

import numpy as np
import pandas as pd

from ._constants import PS_FREQ_RANGE


def _parse_json_series(json_str) -> np.ndarray | None:
    """Parse a JSON-serialised 1D array from a CSV cell."""
    if json_str is None or (isinstance(json_str, float) and np.isnan(json_str)):
        return None
    if isinstance(json_str, str):
        try:
            vals = _json.loads(json_str)
            if isinstance(vals, list):
                return np.array(vals, dtype=float)
        except Exception:
            return None
    return None


def _trim_to_keypoint_valid_window(raw_sig, filt_sig):
    """Trim signals to [first valid keypoint, last valid keypoint].

    Validity is determined from ``raw_sig`` (which preserves NaNs from missing
    keypoints prior to interpolation). If ``raw_sig`` is unavailable, falls back
    to valid (non-NaN) indices in ``filt_sig``.

    Returns
    -------
    tuple
        (raw_trimmed, filt_trimmed, start_idx, end_idx)
    """
    raw_arr = np.asarray(raw_sig, dtype=float) if raw_sig is not None else None
    filt_arr = np.asarray(filt_sig, dtype=float) if filt_sig is not None else None

    if raw_arr is not None:
        valid_idx = np.where(~np.isnan(raw_arr))[0]
    elif filt_arr is not None:
        valid_idx = np.where(~np.isnan(filt_arr))[0]
    else:
        return raw_sig, filt_sig, 0, -1

    if valid_idx.size == 0:
        return None, None, 0, -1

    start = int(valid_idx[0])
    end = int(valid_idx[-1])

    raw_out = raw_arr[start : end + 1] if raw_arr is not None else None
    if filt_arr is None:
        filt_out = None
    else:
        end_eff = min(end, len(filt_arr) - 1)
        if end_eff < start:
            return raw_out, None, start, end
        filt_out = filt_arr[start : end_eff + 1]

    return raw_out, filt_out, start, end


def _spectral_concentration(signal: np.ndarray, fps: float) -> float | None:
    """Fraction of FFT power in a +/-0.3 Hz band around the dominant frequency.

    Only considers the physiological PS range (0.3-4 Hz).
    Returns a value in [0, 1]; higher = more periodic / less noisy.
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig[~np.isnan(sig)]
    if len(sig) < 30:
        return None
    sig = sig - np.mean(sig)
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(sig)) ** 2

    # Restrict to physiological range
    mask = (freqs >= PS_FREQ_RANGE[0]) & (freqs <= PS_FREQ_RANGE[1])
    if mask.sum() < 2:
        return None
    physio_power = power[mask]
    physio_freqs = freqs[mask]
    total_physio = float(np.sum(physio_power))
    if total_physio < 1e-12:
        return None

    # Find dominant frequency
    dom_idx = int(np.argmax(physio_power))
    dom_freq = physio_freqs[dom_idx]

    # Power in +/-0.3 Hz band around dominant
    band_mask = (physio_freqs >= dom_freq - 0.3) & (physio_freqs <= dom_freq + 0.3)
    band_power = float(np.sum(physio_power[band_mask]))
    return band_power / total_physio


def _autocorr_peak_strength(signal: np.ndarray, fps: float) -> float | None:
    """Peak normalised autocorrelation at the dominant period.

    Searches for the first positive autocorrelation peak in the lag range
    corresponding to 0.3-4 Hz.  Returns a value in [0, 1].
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig[~np.isnan(sig)]
    if len(sig) < 30:
        return None
    sig = sig - np.mean(sig)
    var = float(np.dot(sig, sig))
    if var < 1e-12:
        return None

    # Lag range: 1/4 Hz to 1/0.3 Hz in frames
    min_lag = max(1, int(fps / PS_FREQ_RANGE[1]))
    max_lag = min(len(sig) // 2, int(fps / PS_FREQ_RANGE[0]))
    if min_lag >= max_lag:
        return None

    # Compute autocorrelation for the relevant lags
    acf = np.array(
        [
            float(np.dot(sig[: len(sig) - lag], sig[lag:])) / var
            for lag in range(min_lag, max_lag + 1)
        ]
    )
    if acf.size == 0:
        return None
    peak_val = float(np.max(acf))
    return float(np.clip(peak_val, 0.0, 1.0))


def _cycle_shape_consistency(
    filt_sig: np.ndarray, peak_times_json, fps: float, start_frame: int = 0
) -> float | None:
    """Mean Pearson r between consecutive time-warped half-cycles.

    Uses detected peak times to segment the filtered signal into
    inter-peak intervals, resamples each to a fixed length, then
    computes pairwise correlation between consecutive segments.
    """
    if peak_times_json is None or (
        isinstance(peak_times_json, float) and np.isnan(peak_times_json)
    ):
        return None
    try:
        peak_times = _json.loads(peak_times_json) if isinstance(peak_times_json, str) else None
    except Exception:
        return None
    if peak_times is None or len(peak_times) < 3:
        return None

    sig = np.asarray(filt_sig, dtype=float)
    # Convert peak times (seconds) to frame indices
    peak_frames = [int(round(t * fps)) - int(start_frame) for t in peak_times]
    peak_frames = [p for p in peak_frames if 0 <= p < len(sig)]
    if len(peak_frames) < 3:
        return None

    # Extract inter-peak segments and resample to fixed length
    resample_len = 50
    segments = []
    for i in range(len(peak_frames) - 1):
        lo, hi = peak_frames[i], peak_frames[i + 1]
        if hi - lo < 3:
            continue
        seg = sig[lo:hi]
        if np.any(np.isnan(seg)):
            continue
        # Resample to fixed length via linear interpolation
        x_old = np.linspace(0, 1, len(seg))
        x_new = np.linspace(0, 1, resample_len)
        resampled = np.interp(x_new, x_old, seg)
        segments.append(resampled)

    if len(segments) < 2:
        return None

    # Pairwise correlation between consecutive segments
    correlations = []
    for i in range(len(segments) - 1):
        s1, s2 = segments[i], segments[i + 1]
        std1, std2 = float(np.std(s1)), float(np.std(s2))
        if std1 < 1e-9 or std2 < 1e-9:
            continue
        r = float(np.corrcoef(s1, s2)[0, 1])
        if np.isfinite(r):
            correlations.append(r)

    if not correlations:
        return None
    mean_r = float(np.mean(correlations))
    return float(np.clip((mean_r + 1.0) / 2.0, 0.0, 1.0))


def _empty_quality_result(reason: str) -> dict:
    return {
        "signal_quality_score": 0.0,
        "sub_scores": {},
        "weights": {},
        "n_total": 0,
        "n_with_features": 0,
        "n_signal_evaluated": 0,
        "error": reason,
    }


def compute_signal_quality_score(
    kinematics_csv_path: str,
    restrict_to_valid_window: bool = True,
    verbose: bool = False,
) -> dict:
    """Evaluate intrinsic tracking / signal quality -- no clinical labels needed.

    Reads the tracking log CSV produced by the pipeline and computes
    per-video quality metrics from the raw + filtered angle signals and
    tracking metadata.  Returns an aggregate quality score that rewards
    configurations producing clean, periodic, high-coverage signals.

    Sub-scores (all in [0, 1], higher = better):
      1. **yield_rate** -- fraction of videos that produced kinematic features.
      2. **coverage** -- mean landmark detection rate within each video's
         active window (adjusted_appearance_pct / 100).
      3. **spectral_concentration** -- fraction of bandpass signal power in
         a narrow band around the dominant frequency (auto-detected).
      4. **autocorr_strength** -- peak normalised autocorrelation at the
         dominant period.
      5. **cycle_shape_consistency** -- mean Pearson correlation between
         consecutive time-warped half-cycles.
      6. **freq_physio_rate** -- fraction of videos whose detected
         frequency falls within the physiological PS range.
      7. **low_nan_frac** -- mean (1 - nan_fraction) across videos.

    Aggregate:
        signal_quality_score = weighted mean of sub-scores
    """
    kin = pd.read_csv(kinematics_csv_path)
    n_total = len(kin)
    if n_total == 0:
        return _empty_quality_result("Empty CSV")

    # Guard: 'Mean Amplitude' will be absent when ALL worker processes crashed
    # (the CSV only contains ERROR/TIMEOUT records which have no kinematic columns).
    # Surface the actual worker error messages so callers can diagnose the crash.
    if "Mean Amplitude" not in kin.columns:
        error_sample: list[str] = []
        if "error" in kin.columns:
            error_sample = (
                kin["error"].dropna().astype(str).unique().tolist()[:3]
            )
        msg = (
            f"No kinematic features in CSV ({n_total} rows, all appear to be "
            "worker error/timeout records)."
        )
        if error_sample:
            msg += f" Worker errors (first {len(error_sample)}): {error_sample}"
        return _empty_quality_result(msg)

    # --- 1. Feature yield ---
    has_features = kin["Mean Amplitude"].notna()
    n_with = int(has_features.sum())
    yield_rate = n_with / n_total if n_total > 0 else 0.0

    # --- 2. Coverage ---
    if "adjusted_appearance_pct" in kin.columns:
        cov_vals = pd.to_numeric(kin["adjusted_appearance_pct"], errors="coerce").dropna()
        coverage = float(cov_vals.mean()) / 100.0 if len(cov_vals) > 0 else 0.0
    else:
        coverage = yield_rate  # fallback

    # Per-video signal metrics
    spectral_concentrations = []
    autocorr_strengths = []
    cycle_shape_scores = []
    freq_physio_hits = 0
    nan_fracs = []
    n_signal = 0

    for _, row in kin.iterrows():
        if pd.isna(row.get("Mean Amplitude")):
            continue
        raw_json = row.get("raw_rotation_series")
        filt_json = row.get("filtered_rotation_series")
        fps = float(row.get("fps", 25.0)) if pd.notna(row.get("fps")) else 25.0

        raw_sig = _parse_json_series(raw_json)
        filt_sig = _parse_json_series(filt_json)
        if restrict_to_valid_window:
            raw_sig, filt_sig, win_start, _win_end = _trim_to_keypoint_valid_window(
                raw_sig, filt_sig
            )
        else:
            win_start = 0
        if filt_sig is None or len(filt_sig) < 20:
            continue
        n_signal += 1

        # --- NaN fraction from raw signal ---
        if raw_sig is not None:
            nf = float(np.sum(np.isnan(raw_sig))) / max(len(raw_sig), 1)
        else:
            nf = 0.0
        nan_fracs.append(1.0 - nf)

        # --- 3. Spectral concentration ---
        sc = _spectral_concentration(filt_sig, fps)
        if sc is not None:
            spectral_concentrations.append(sc)

        # --- 4. Autocorrelation strength ---
        ac = _autocorr_peak_strength(filt_sig, fps)
        if ac is not None:
            autocorr_strengths.append(ac)

        # --- 5. Cycle shape consistency ---
        peak_times_json = row.get("cycle_peak_times_s")
        cs = _cycle_shape_consistency(filt_sig, peak_times_json, fps, start_frame=win_start)
        if cs is not None:
            cycle_shape_scores.append(cs)

        # --- 6. Frequency in physiological range ---
        freq = row.get("Mean Frequency")
        if pd.notna(freq):
            f = float(freq)
            if PS_FREQ_RANGE[0] <= f <= PS_FREQ_RANGE[1]:
                freq_physio_hits += 1

    # Aggregate sub-scores
    spectral = float(np.mean(spectral_concentrations)) if spectral_concentrations else 0.0
    autocorr = float(np.mean(autocorr_strengths)) if autocorr_strengths else 0.0
    cycle_shape = float(np.mean(cycle_shape_scores)) if cycle_shape_scores else 0.0
    freq_physio_rate = freq_physio_hits / n_signal if n_signal > 0 else 0.0
    low_nan = float(np.mean(nan_fracs)) if nan_fracs else 0.0

    # Weighted aggregate
    weights = {
        "yield_rate": 1.5,
        "coverage": 1.0,
        "spectral_concentration": 2.0,
        "autocorr_strength": 2.0,
        "cycle_shape_consistency": 1.5,
        "freq_physio_rate": 1.0,
        "low_nan_frac": 1.0,
    }
    sub_scores = {
        "yield_rate": float(np.clip(yield_rate, 0, 1)),
        "coverage": float(np.clip(coverage, 0, 1)),
        "spectral_concentration": float(np.clip(spectral, 0, 1)),
        "autocorr_strength": float(np.clip(autocorr, 0, 1)),
        "cycle_shape_consistency": float(np.clip(cycle_shape, 0, 1)),
        "freq_physio_rate": float(np.clip(freq_physio_rate, 0, 1)),
        "low_nan_frac": float(np.clip(low_nan, 0, 1)),
    }
    w_total = sum(weights.values())
    quality_score = sum(weights[k] * sub_scores[k] for k in weights) / w_total

    if verbose:
        print(f"  Signal quality \u2013 n_total={n_total} n_with={n_with} n_signal={n_signal}")
        for k, v in sub_scores.items():
            print(f"    {k:30s} = {v:.4f}  (w={weights[k]:.1f})")
        print(f"    {'AGGREGATE':30s} = {quality_score:.4f}")

    return {
        "signal_quality_score": float(quality_score),
        "sub_scores": sub_scores,
        "weights": weights,
        "n_total": n_total,
        "n_with_features": n_with,
        "n_signal_evaluated": n_signal,
    }
