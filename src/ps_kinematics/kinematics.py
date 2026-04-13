"""
ps_kinematics.kinematics — Kinematic feature computation.

Contains KinematicAnalyzer and angle computation functions for extracting
clinically oriented cycle metrics from hand-landmark tracking data.
"""

import numpy as np

from .utils import (
    ADAPTIVE_VIS_FLOOR,
    ADAPTIVE_VIS_PERCENTILE,
    ADAPTIVE_VISIBILITY,
    AMP_DECREMENT_SPLIT_FRAC,
    ARREST_MIN_DURATION_S,
    BASE_FPS,
    CYCLE_NAN_THRESHOLD,
    DETREND_POLY_ORDER,
    HAMPEL_SIGMA,
    HAMPEL_WINDOW,
    PEAK_VELOCITY_PERCENTILE,
    SCIPY_OK,
    STARTUP_TRANSIENT_FACTOR,
    USE_WRIST_Z_CONFIRMATION,
    VISIBILITY_THRESHOLD,
    WRIST_Z_CONFIRM_WINDOW,
    WRIST_Z_MIN_SNR,
    WRIST_Z_MIN_VALID_FRAC,
    WRIST_Z_PROM_BOOST,
    ZCR_DC_WINDOW_CYCLES,
    _unwrap_segments,
    hampel_filter,
)

if SCIPY_OK:
    from scipy.signal import butter, find_peaks, sosfiltfilt
    from scipy.stats import linregress


# Lazy import — only needed for landmark index constants
try:
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
except ImportError:
    # Provide stub constants when mediapipe is not installed (e.g. test env)
    class HandLandmark:
        WRIST = 0
        INDEX_FINGER_MCP = 5
        MIDDLE_FINGER_MCP = 9
        RING_FINGER_MCP = 13
        PINKY_MCP = 17


# ============================================================
# KinematicAnalyzer
# ============================================================


class KinematicAnalyzer:
    """Extract clinically oriented cycle metrics from a 1D angle signal."""

    _HESITATION_RATIO = 1.25
    _HESITATION_ABS_FLOOR_S = 0.40

    def __init__(
        self,
        timestamps,
        angles_deg,
        fps=30,
        cutoff_hz=2.5,
        filter_order=4,
        highpass_hz=0.10,
        wrist_z=None,
        use_parabolic_interp=False,
    ):
        self.t = np.asarray(timestamps, dtype=float)
        self.fps = float(fps) if fps else 30.0
        self.use_parabolic_interp = bool(use_parabolic_interp)
        # Scale the wrist-z confirmation window (calibrated at BASE_FPS) to
        # the actual video fps so the ±seconds tolerance stays constant.
        _fps_scale = self.fps / BASE_FPS
        self._wrist_z_confirm_window = max(1, round(WRIST_Z_CONFIRM_WINDOW * _fps_scale))

        ang = np.asarray(angles_deg, dtype=float)
        self.nan_mask = np.isnan(ang)
        if np.isnan(ang).any():
            valid = ~np.isnan(ang)
            if valid.sum() >= 2:
                ang[np.isnan(ang)] = np.interp(self.t[np.isnan(ang)], self.t[valid], ang[valid])
            else:
                ang[:] = 0.0

        self.raw_signal = ang.copy()

        # ── Pre-filter detrending ──────────────────────────────────────
        # Accumulated phase-unwrap errors and slow tracking drift can
        # produce angle trends spanning hundreds of degrees in 10–15 s
        # recordings.  The bandpass highpass component (often 0.05–0.30 Hz)
        # has insufficient roll-off so close to DC to suppress them.
        # A low-order polynomial fit captures the drift without affecting
        # the faster PS oscillation (0.25–3.5 Hz).
        ang = self._detrend(ang, self.t)
        ang = hampel_filter(ang, HAMPEL_WINDOW, HAMPEL_SIGMA)

        # Lowpass-only signal: preserves long pauses for arrest detection
        self._lowpass_signal = self._bandpass(
            ang, cutoff=cutoff_hz, highpass_hz=0.0, fs=self.fps, order=filter_order
        )
        self.clean_signal = self._bandpass(
            ang, cutoff=cutoff_hz, highpass_hz=highpass_hz, fs=self.fps, order=filter_order
        )

        # --- Wrist-Z confirmation channel ---
        # Filtered wrist z signal used to validate detected peaks/valleys.
        # During pronation-supination the wrist z oscillates at the PS
        # frequency; genuine angle extrema should coincide with wrist z
        # extrema (peak or valley — phase depends on camera angle).
        self._wrist_z_valid = False
        self._wrist_z_clean = None
        self._wrist_z_extrema = None
        if wrist_z is not None and USE_WRIST_Z_CONFIRMATION:
            wz = np.asarray(wrist_z, dtype=float)
            wz_nan = np.isnan(wz)
            valid_frac = float((~wz_nan).sum()) / max(len(wz), 1)
            if valid_frac >= WRIST_Z_MIN_VALID_FRAC:
                # Interpolate NaN gaps
                if wz_nan.any():
                    wz_valid = ~wz_nan
                    if wz_valid.sum() >= 2:
                        wz[wz_nan] = np.interp(self.t[wz_nan], self.t[wz_valid], wz[wz_valid])
                    else:
                        wz[:] = 0.0
                wz_filt = self._bandpass(
                    wz, cutoff=cutoff_hz, highpass_hz=highpass_hz, fs=self.fps, order=filter_order
                )
                # SNR check: signal range vs noise (std of residual after filtering)
                wz_range = float(np.ptp(wz_filt))
                wz_noise = float(np.std(wz - wz_filt)) if wz.size > 10 else 1e9
                if wz_noise > 1e-9 and (wz_range / wz_noise) >= WRIST_Z_MIN_SNR:
                    self._wrist_z_clean = wz_filt
                    self._wrist_z_valid = True
                    # Pre-compute wrist z extrema (both peaks and valleys)
                    self._wrist_z_extrema = self._detect_wrist_z_extrema(wz_filt)

    def _bandpass(self, data, cutoff, highpass_hz, fs, order=4):
        """Butterworth bandpass (or lowpass when highpass_hz is 0 or None)."""
        data = np.asarray(data, dtype=float)
        min_len = 3 * (2 * order + 3)
        if data.size < min_len:
            return data.copy()
        if SCIPY_OK:
            nyq = 0.5 * fs
            high = min(max(cutoff / nyq, 1e-4), 0.999)
            low = float(highpass_hz) / nyq if highpass_hz and float(highpass_hz) > 0 else 0.0
            if low < 1e-4:
                # Pure lowpass — no highpass component
                sos = butter(order, high, btype="low", analog=False, output="sos")
            elif low >= high:
                sos = butter(order, high, btype="low", analog=False, output="sos")
            else:
                sos = butter(order, [low, high], btype="band", analog=False, output="sos")
            return sosfiltfilt(sos, data)
        win_lp = max(3, int(fs / max(cutoff, 1e-3)))
        win_hp = max(3, int(fs / max(float(highpass_hz or 0.3), 1e-3) * 0.5))
        if win_lp % 2 == 0:
            win_lp += 1
        if win_hp % 2 == 0:
            win_hp += 1
        lp = np.convolve(data, np.ones(win_lp) / win_lp, mode="same")
        bas = np.convolve(data, np.ones(win_hp) / win_hp, mode="same")
        return lp - bas

    @staticmethod
    def _detrend(ang, t):
        """Remove low-frequency polynomial drift from angle signal.

        Accumulated phase-unwrap errors and slow tracking drift can
        produce trends spanning hundreds of degrees in 10–15 s
        recordings.  A Butterworth highpass at 0.05–0.30 Hz lacks
        sufficient roll-off at these very-low frequencies.

        A degree-2 polynomial captures the dominant drift shape (linear
        ramp from accumulated ±360° unwrap errors, or V/U-shaped from
        transient tracking switches) without affecting the faster PS
        oscillation (0.25–3.5 Hz).

        For a sinusoidal PS signal with n full cycles over the recording
        duration, the polynomial coefficients for the oscillatory part
        integrate to ≈ 0 (orthogonality of polynomials and
        trigonometric functions), so the fit captures only the drift.

        Parameters
        ----------
        ang : np.ndarray
            Angle signal in degrees (NaN-interpolated).
        t : np.ndarray
            Timestamp array (seconds).

        Returns
        -------
        np.ndarray
            Detrended angle signal.
        """
        if len(ang) < 10 or DETREND_POLY_ORDER < 1:
            return ang
        coeffs = np.polyfit(t, ang, DETREND_POLY_ORDER)
        residual = ang - np.polyval(coeffs, t)

        # Second pass: running-median baseline subtraction to remove large
        # DC offsets that survive polynomial detrending (e.g. multi-revolution
        # signals in the range 0°–2500°).  Window is ~2 cycle lengths so the
        # median tracks the slow baseline without attenuating the oscillation.
        try:
            from scipy.ndimage import median_filter as _mf

            # Estimate dominant oscillation period from FFT of residual
            n = len(residual)
            freqs = np.fft.rfftfreq(n, d=float(t[1] - t[0])) if n > 1 else np.array([1.0])
            power = np.abs(np.fft.rfft(residual)) ** 2
            # Restrict to physiologically plausible PS range (0.25–3.5 Hz)
            fps_est = 1.0 / float(t[1] - t[0]) if n > 1 else 25.0
            valid = (freqs >= 0.25) & (freqs <= 3.5)
            if valid.any():
                dom_freq = float(freqs[valid][np.argmax(power[valid])])
            else:
                dom_freq = 1.5  # fallback
            period_frames = max(5, int(fps_est / dom_freq * 2))
            if period_frames % 2 == 0:
                period_frames += 1
            baseline = _mf(residual, size=period_frames)
            residual = residual - baseline
        except Exception:
            pass  # fall back to polynomial-only detrend

        return residual

    def _detect_wrist_z_extrema(self, wz_filt):
        """Detect all extrema (peaks + valleys) in filtered wrist z signal.

        Returns a sorted array of frame indices where wrist z has a local
        extremum.  Used for confirming angle-based peak/valley detections.
        """
        extrema = []
        if SCIPY_OK:
            # Use a relaxed prominence — we want *all* genuine extrema
            wz_range = float(np.ptp(wz_filt))
            wz_prom = max(0.05 * wz_range, 1e-6)
            pk, _ = find_peaks(wz_filt, prominence=wz_prom)
            vl, _ = find_peaks(-wz_filt, prominence=wz_prom)
            extrema = np.sort(np.concatenate([pk, vl]))
        else:
            for i in range(1, len(wz_filt) - 1):
                if (wz_filt[i] > wz_filt[i - 1] and wz_filt[i] > wz_filt[i + 1]) or (
                    wz_filt[i] < wz_filt[i - 1] and wz_filt[i] < wz_filt[i + 1]
                ):
                    extrema.append(i)
            extrema = np.array(extrema, dtype=int)
        return extrema

    # ------------------------------------------------------------------ #
    #  Dominant-period estimation                                         #
    # ------------------------------------------------------------------ #
    def _estimate_dominant_period(self, sig):
        """Estimate the dominant oscillation period in frames via autocorrelation.

        The autocorrelation function (ACF) is computed efficiently via FFT.
        The first significant peak in the ACF (lag > physiological minimum)
        corresponds to the dominant PS period.

        Returns
        -------
        int or None
            Estimated dominant period in frames, or ``None`` if estimation
            fails (signal too short, flat, or aperiodic).
        """
        n = len(sig)
        if n < 20:
            return None

        sig_c = sig - np.mean(sig)
        energy = float(np.sum(sig_c**2))
        if energy < 1e-12:
            return None

        # Autocorrelation via FFT
        n_fft = 1
        while n_fft < 2 * n:
            n_fft *= 2
        fft_sig = np.fft.rfft(sig_c, n=n_fft)
        acf = np.fft.irfft(np.abs(fft_sig) ** 2, n=n_fft)[:n]
        acf = acf / acf[0]

        # Physiological search range
        # Fastest PS ≈ 3.5 Hz → period ≈ fps / 3.5
        # Slowest PS ≈ 0.25 Hz → period ≈ fps / 0.25
        min_lag = max(3, int(self.fps / 3.5))
        max_lag = min(n // 2, int(self.fps / 0.25))

        if max_lag <= min_lag + 2:
            return None

        acf_seg = acf[min_lag : max_lag + 1]
        if len(acf_seg) < 3:
            return None

        if SCIPY_OK:
            pks, props = find_peaks(acf_seg, prominence=0.03)
            if len(pks) == 0:
                return None
            # Take the most prominent peak (dominant period)
            best_pk = pks[np.argmax(props["prominences"])]
            return int(best_pk + min_lag)
        else:
            for i in range(1, len(acf_seg) - 1):
                if (
                    acf_seg[i] > acf_seg[i - 1]
                    and acf_seg[i] > acf_seg[i + 1]
                    and acf_seg[i] > 0.05
                ):
                    return int(i + min_lag)
            return None

    # ------------------------------------------------------------------ #
    #  Zero-crossing based half-cycle detection (primary method)          #
    # ------------------------------------------------------------------ #
    def _detect_half_cycles_zcr(self, sig, est_period, min_half_cycle_s, max_half_cycle_s=4.0):
        """Detect half-cycles via zero-crossing detection.

        The bandpass filter's high-pass component guarantees the signal is
        approximately zero-mean, so zero crossings naturally delineate
        half-cycle boundaries.  Within each half-cycle the true extremum
        (peak or valley) is located on the original filtered signal.

        Advantages over peak-based detection
        -------------------------------------
        * No prominence threshold required — immune to amplitude decrement.
        * No width constraint — works at all physiological PS frequencies.
        * Natural alternation — positive half-cycles always alternate with
          negative half-cycles (no post-hoc alternation enforcement needed).

        Parameters
        ----------
        sig : np.ndarray
            Bandpass-filtered angle signal (``self.clean_signal``).
        est_period : int or None
            Estimated dominant period in frames.
        min_half_cycle_s : float
            Minimum physiological half-cycle duration (seconds).
        max_half_cycle_s : float
            Maximum physiological half-cycle duration (seconds).

        Returns
        -------
        list of tuple
            ``[(extremum_index, extremum_type), ...]`` where type is
            ``+1`` (peak) or ``-1`` (valley), in temporal order with
            alternation guaranteed.
        """
        n = len(sig)
        if n < 6:
            return []

        # Remove any residual DC (should be near-zero after bandpass).
        # A local running mean handles cases where the bandpass high-pass
        # corner is very low relative to the signal duration.
        # Uses reflect-padding to avoid edge effects: the default
        # zero-padding caused the moving mean to be biased at the signal
        # boundaries, preventing small-amplitude oscillations after a
        # large startup transient from crossing zero.
        if est_period is not None and est_period > 4:
            dc_win = max(5, int(ZCR_DC_WINDOW_CYCLES * est_period))
        else:
            dc_win = max(5, int(2.0 * self.fps))
        if dc_win % 2 == 0:
            dc_win += 1
        dc_win = min(dc_win, n if n % 2 == 1 else n - 1)
        if dc_win < 3:
            dc_win = 3
        if dc_win % 2 == 0:
            dc_win += 1

        # Reflect-pad before convolution so the running mean at boundaries
        # uses mirrored signal values instead of implicit zeros.
        pad = dc_win // 2
        sig_padded = np.pad(sig, pad, mode="reflect")
        kernel = np.ones(dc_win) / dc_win
        dc_padded = np.convolve(sig_padded, kernel, mode="same")
        dc = dc_padded[pad : pad + n]
        sig_c = sig - dc

        # --- Detect zero crossings ---
        min_half_frames = max(2, int(self.fps * min_half_cycle_s))
        max_half_frames = int(self.fps * max_half_cycle_s)

        sign = np.sign(sig_c)
        sign[sign == 0] = 1  # treat exact zero as positive
        sign_diff = np.diff(sign)
        crossing_indices = np.where(sign_diff != 0)[0]  # crossings between [i] and [i+1]

        if len(crossing_indices) < 3:
            return []

        # --- Build half-cycles from consecutive zero crossings ---
        result = []
        for i in range(len(crossing_indices) - 1):
            zc_start = crossing_indices[i]
            zc_end = crossing_indices[i + 1]

            span = zc_end - zc_start
            if span < min_half_frames or span > max_half_frames:
                continue

            # Sign of the mid-segment determines peak vs valley
            mid = zc_start + 1 + (span - 1) // 2
            segment = sig[zc_start : zc_end + 1]

            if sig_c[mid] > 0:
                ext_local = int(np.argmax(segment))
                ext_type = 1  # peak
            else:
                ext_local = int(np.argmin(segment))
                ext_type = -1  # valley

            ext_idx = zc_start + ext_local
            result.append((ext_idx, ext_type))

        # --- Enforce strict alternation (handles edge-case gaps) ---
        alt = []
        for idx, etype in result:
            if alt and alt[-1][1] == etype:
                prev_idx = alt[-1][0]
                if (etype == 1 and sig[idx] > sig[prev_idx]) or (
                    etype == -1 and sig[idx] < sig[prev_idx]
                ):
                    alt[-1] = (idx, etype)
            else:
                alt.append((idx, etype))

        return alt

    # ------------------------------------------------------------------ #
    #  Enhanced peak-based half-cycle detection (fallback method)         #
    # ------------------------------------------------------------------ #
    def _detect_half_cycles_peaks(
        self,
        sig,
        est_period,
        prominence_deg,
        adaptive_prom_frac,
        min_half_cycle_s,
        max_movement_hz=2.0,
    ):
        """Fallback cycle detection via peak/valley finding.

        Key differences from the original ``extract_features`` peak detection:

        1. **No width constraint** — the old ``width = fps × 0.30`` parameter
           rejected every peak at PS frequencies above ~1.4 Hz.  Removed.
        2. **Autocorrelation-guided distance** — ``min_distance`` is set from
           the estimated period rather than a fixed ``fps / max_hz``.
        3. **Wrist-Z gating** preserved (when enabled).
        """
        if not SCIPY_OK:
            return []

        q75, q25 = float(np.percentile(sig, 75)), float(np.percentile(sig, 25))
        iqr = q75 - q25
        med = float(np.median(sig))
        mad = float(np.median(np.abs(sig - med)))
        robust_scale = 1.4826 * mad if mad > 1e-12 else iqr / 1.349 if iqr > 1e-12 else 0.0
        capped_iqr = min(iqr, 2.5 * robust_scale) if robust_scale > 0 else iqr
        adaptive_prom = adaptive_prom_frac * max(capped_iqr, 0.0)
        effective_prom = max(0.5, float(prominence_deg), adaptive_prom)
        boosted_prom = effective_prom * WRIST_Z_PROM_BOOST

        # Minimum distance between peaks: ~35 % of estimated period
        phys_dist_cap = max(2, int(self.fps / max(float(max_movement_hz), 1e-6) * 0.85))
        if est_period is not None and est_period > 3:
            min_dist = max(2, min(int(est_period * 0.35), phys_dist_cap))
        else:
            min_dist = max(2, int(phys_dist_cap * 0.6))

        # ── NO WIDTH CONSTRAINT ──
        peaks, _ = find_peaks(sig, distance=min_dist, prominence=effective_prom)
        valleys, _ = find_peaks(-sig, distance=min_dist, prominence=effective_prom)

        # Wrist-Z confirmation gating (optional, active when enabled)
        if self._wrist_z_valid and len(peaks) > 0:
            from scipy.signal import peak_prominences

            try:
                _proms, _, _ = peak_prominences(sig, peaks)
            except Exception:
                _proms = np.full(len(peaks), effective_prom)
            peaks = np.array(
                [
                    pk
                    for pk, pr in zip(peaks, _proms)
                    if self._wrist_z_confirms(pk) or pr >= boosted_prom
                ],
                dtype=int,
            )

        if self._wrist_z_valid and len(valleys) > 0:
            from scipy.signal import peak_prominences

            try:
                _vproms, _, _ = peak_prominences(-sig, valleys)
            except Exception:
                _vproms = np.full(len(valleys), effective_prom)
            valleys = np.array(
                [
                    vl
                    for vl, pr in zip(valleys, _vproms)
                    if self._wrist_z_confirms(vl) or pr >= boosted_prom
                ],
                dtype=int,
            )

        if len(peaks) < 2 or len(valleys) < 1:
            return []

        peaks = np.sort(peaks)
        valleys = np.sort(valleys)

        # Build alternating P-V sequence
        _all = sorted(
            [(int(idx), 1) for idx in peaks] + [(int(idx), -1) for idx in valleys],
            key=lambda x: x[0],
        )
        _alt = []
        for idx, etype in _all:
            if _alt and _alt[-1][1] == etype:
                prev_idx = _alt[-1][0]
                if (etype == 1 and sig[idx] > sig[prev_idx]) or (
                    etype == -1 and sig[idx] < sig[prev_idx]
                ):
                    _alt[-1] = (idx, etype)
            else:
                _alt.append((idx, etype))

        if len(_alt) < 3:
            return _alt

        # Remove too-short half-cycles
        min_half_frames = max(1, int(self.fps * min_half_cycle_s))
        _changed = True
        while _changed:
            _changed = False
            for i in range(len(_alt) - 1):
                if abs(_alt[i + 1][0] - _alt[i][0]) < min_half_frames:
                    idx_a, type_a = _alt[i]
                    idx_b, type_b = _alt[i + 1]
                    if type_a == type_b:
                        if (type_a == 1 and sig[idx_b] >= sig[idx_a]) or (
                            type_a == -1 and sig[idx_b] <= sig[idx_a]
                        ):
                            _alt.pop(i)
                        else:
                            _alt.pop(i + 1)
                    else:
                        # Use median-relative amplitude (more robust than mean)
                        med = float(np.median(sig))
                        amp_a = abs(sig[idx_a] - med)
                        amp_b = abs(sig[idx_b] - med)
                        if amp_a <= amp_b:
                            _alt.pop(i)
                        else:
                            _alt.pop(i + 1)
                    _changed = True
                    break

        return _alt

    def _wrist_z_confirms(self, idx):
        """Check whether frame *idx* is near a wrist z extremum.

        Returns True if there is a wrist z extremum within
        ±WRIST_Z_CONFIRM_WINDOW frames of *idx*, or if wrist z
        validation is inactive.
        """
        if not self._wrist_z_valid or self._wrist_z_extrema is None:
            return True  # no wrist z available → don't penalise
        if len(self._wrist_z_extrema) == 0:
            return False
        # Binary search for closest extremum
        pos = np.searchsorted(self._wrist_z_extrema, idx)
        for candidate in (pos - 1, pos):
            if 0 <= candidate < len(self._wrist_z_extrema):
                if (
                    abs(int(self._wrist_z_extrema[candidate]) - int(idx))
                    <= self._wrist_z_confirm_window
                ):
                    return True
        return False

    # ------------------------------------------------------------------ #
    #  Parabolic peak interpolation                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parabolic_peak_value(sig: np.ndarray, idx: int) -> float:
        """Refine an extremum value via 3-point parabolic interpolation.

        Given a discrete extremum at *idx*, fit a parabola through
        ``sig[idx-1], sig[idx], sig[idx+1]`` and return the vertex
        value.  This recovers sub-sample peak precision and partially
        compensates for the lowpass filter rounding off true peaks.

        Falls back to ``sig[idx]`` when *idx* is at a signal boundary
        or the parabola opens in the wrong direction (concavity check).

        Parameters
        ----------
        sig : np.ndarray
            1-D signal array.
        idx : int
            Index of the discrete extremum.

        Returns
        -------
        float
            Interpolated extremum value (always >= discrete peak for
            maxima, <= discrete valley for minima due to concavity).
        """
        if idx <= 0 or idx >= len(sig) - 1:
            return float(sig[idx])

        y_prev = float(sig[idx - 1])
        y_curr = float(sig[idx])
        y_next = float(sig[idx + 1])

        denom = y_prev - 2.0 * y_curr + y_next
        if abs(denom) < 1e-12:
            return y_curr

        # Vertex offset from idx (in fractional samples)
        offset = 0.5 * (y_prev - y_next) / denom
        # Interpolated value at the vertex
        interp_val = y_curr - 0.25 * (y_prev - y_next) * offset

        # Concavity guard: for a maximum the parabola must open downward
        # (denom < 0), for a minimum it must open upward (denom > 0).
        # If the interpolated value is *worse* than the discrete value,
        # the parabola fit is unreliable — fall back.
        is_peak = (y_curr >= y_prev) and (y_curr >= y_next)
        if is_peak and interp_val < y_curr:
            return y_curr
        is_valley = (y_curr <= y_prev) and (y_curr <= y_next)
        if is_valley and interp_val > y_curr:
            return y_curr

        return float(interp_val)

    # ------------------------------------------------------------------ #
    #  Cycle building from alternating extrema                            #
    # ------------------------------------------------------------------ #
    def _build_cycles_from_extrema(self, sig, _alt):
        """Build full cycles from detected alternating extrema.

        Anchors cycles on same-polarity extrema (peak-to-peak OR
        valley-to-valley), estimates each cycle's amplitude using the
        opposite turning point inside that interval, and applies startup
        transient gating.

        Returns
        -------
        dict or None
            Keys: amplitudes, peak_times, cycle_bounds, intervals,
            peaks_idx, detected_peak_times, detected_trough_times, n_full.
        """
        peak_idxs = np.array([idx for idx, et in _alt if et == 1], dtype=int)
        valley_idxs = np.array([idx for idx, et in _alt if et == -1], dtype=int)
        detected_peak_times = self.t[peak_idxs] if peak_idxs.size else np.array([], dtype=float)
        detected_trough_times = (
            self.t[valley_idxs] if valley_idxs.size else np.array([], dtype=float)
        )

        def _build_anchor_cycles(anchor_type):
            anchor_idxs = peak_idxs if anchor_type == 1 else valley_idxs
            if len(anchor_idxs) < 3:
                return None

            amps, times, bounds = [], [], []
            for i in range(len(anchor_idxs) - 1):
                a = int(anchor_idxs[i])
                b = int(anchor_idxs[i + 1])
                lo, hi = (a, b) if a < b else (b, a)
                if hi - lo < 2:
                    continue

                opp = [idx for idx, et in _alt if et == -anchor_type and lo < idx < hi]
                if opp:
                    if self.use_parabolic_interp:
                        opp_vals = np.array(
                            [self._parabolic_peak_value(sig, idx) for idx in opp],
                            dtype=float,
                        )
                    else:
                        opp_vals = np.array([sig[idx] for idx in opp], dtype=float)
                    opp_val = (
                        float(np.min(opp_vals)) if anchor_type == 1 else float(np.max(opp_vals))
                    )
                else:
                    seg = sig[lo : hi + 1]
                    if seg.size < 3:
                        continue
                    opp_val = float(np.min(seg)) if anchor_type == 1 else float(np.max(seg))

                # Read anchor values — optionally with parabolic refinement
                val_a = (
                    self._parabolic_peak_value(sig, a) if self.use_parabolic_interp else float(sig[a])
                )
                val_b = (
                    self._parabolic_peak_value(sig, b) if self.use_parabolic_interp else float(sig[b])
                )

                if anchor_type == 1:
                    amp = ((val_a - opp_val) + (val_b - opp_val)) / 2.0
                else:
                    amp = ((opp_val - val_a) + (opp_val - val_b)) / 2.0

                if not np.isfinite(amp) or amp <= 0:
                    continue

                amps.append(float(amp))
                times.append(float(self.t[a]))
                bounds.append((lo, hi))

            if len(amps) < 2:
                return None

            anchor_t = self.t[anchor_idxs]
            intervals = np.diff(anchor_t)
            intervals = intervals[intervals > 1e-6]
            if intervals.size < 1:
                return None

            return {
                "anchor_type": anchor_type,
                "anchor_idxs": np.array(anchor_idxs, dtype=int),
                "amplitudes": np.array(amps, dtype=float),
                "peak_times": np.array(times, dtype=float),
                "cycle_bounds": bounds,
                "intervals": np.array(intervals, dtype=float),
            }

        peak_cycles = _build_anchor_cycles(anchor_type=1)
        valley_cycles = _build_anchor_cycles(anchor_type=-1)

        candidates = [c for c in (peak_cycles, valley_cycles) if c is not None]
        if not candidates:
            return None

        def _candidate_score(cand):
            n = len(cand["amplitudes"])
            ints = cand["intervals"]
            if ints.size >= 2 and float(np.mean(ints)) > 1e-9:
                cv = float(np.std(ints) / np.mean(ints))
            else:
                cv = 1e9
            return (n, -cv)

        chosen = max(candidates, key=_candidate_score)

        amplitudes = chosen["amplitudes"]
        peak_times = chosen["peak_times"]
        cycle_bounds = chosen["cycle_bounds"]
        cycle_intervals = chosen["intervals"]
        peaks_out = chosen["anchor_idxs"]

        # Startup transient gating: the first PS movement is often a large,
        # non-representative initial sweep.
        if STARTUP_TRANSIENT_FACTOR > 0 and len(amplitudes) >= 4:
            rest_median = float(np.median(amplitudes[1:]))
            if rest_median > 1e-6 and amplitudes[0] > STARTUP_TRANSIENT_FACTOR * rest_median:
                amplitudes = amplitudes[1:]
                peak_times = peak_times[1:]
                cycle_bounds = cycle_bounds[1:]
                if len(cycle_intervals) > 1:
                    cycle_intervals = cycle_intervals[1:]
                if len(peaks_out) > 1:
                    peaks_out = peaks_out[1:]

        if len(amplitudes) < 2:
            return None

        return {
            "amplitudes": amplitudes,
            "peak_times": peak_times,
            "cycle_bounds": cycle_bounds,
            "intervals": cycle_intervals,
            "peaks_idx": peaks_out,
            "n_full": len(amplitudes),
            "detected_peak_times": detected_peak_times,
            "detected_trough_times": detected_trough_times,
        }

    # ------------------------------------------------------------------ #
    #  NaN quality gating                                                 #
    # ------------------------------------------------------------------ #
    def _apply_nan_quality_gating(self, sig, cycle_bounds, amplitudes, cycle_intervals, n_full):
        """Quality-gate cycles based on NaN fraction within each cycle.

        Returns
        -------
        tuple
            (full_quality, quality_amplitudes, quality_intervals, n_quality_cycles)
        """
        nan_mask_arr = getattr(self, "nan_mask", np.zeros(len(sig), dtype=bool))
        full_quality = np.ones(n_full, dtype=bool)
        for _ci, (lo_i, hi_i) in enumerate(cycle_bounds):
            lo_i = max(0, int(lo_i))
            hi_i = min(len(nan_mask_arr), int(hi_i) + 1)
            if hi_i > lo_i:
                nan_frac = float(np.mean(nan_mask_arr[lo_i:hi_i]))
                if nan_frac > CYCLE_NAN_THRESHOLD:
                    full_quality[_ci] = False

        quality_amplitudes = amplitudes[full_quality]
        if quality_amplitudes.size < 2:
            quality_amplitudes = amplitudes
        n_quality_cycles = int(full_quality.sum())

        # Intervals align with cycles: interval[i] spans cycle i to i+1,
        # so an interval is quality-gated if BOTH adjacent cycles pass.
        if len(cycle_intervals) == len(full_quality) - 1 and len(full_quality) > 1:
            _qi_mask = full_quality[:-1] & full_quality[1:]
            quality_intervals = cycle_intervals[_qi_mask]
            if quality_intervals.size < 2:
                quality_intervals = cycle_intervals
        else:
            quality_intervals = cycle_intervals

        return full_quality, quality_amplitudes, quality_intervals, n_quality_cycles

    # ------------------------------------------------------------------ #
    #  Amplitude feature computation                                      #
    # ------------------------------------------------------------------ #
    def _compute_amplitude_features(
        self, amplitudes, quality_amplitudes, peak_times, cycle_intervals, quality_intervals
    ):
        """Compute amplitude and timing features from cycle data.

        Returns
        -------
        dict
            Keys: avg_amp, amp_cv, freq, cv, norm_decrement_slope,
            amp_decrement_onset, amp_decrement_pct, norm_ti_slope,
            trend_line, slope, intercept.
        """
        _ham = float(np.mean(amplitudes))
        _qham = float(np.mean(quality_amplitudes))

        if SCIPY_OK and len(amplitudes) >= 3:
            slope, intercept, _r, _p, _se = linregress(peak_times, amplitudes)
        else:
            slope, intercept = np.polyfit(peak_times, amplitudes, 1)
        slope, intercept = float(slope), float(intercept)
        # Raw (unnormalised) amplitude slope — paper-aligned (Zarrat Ehsan et al. 2024)
        raw_amp_slope = slope if len(amplitudes) >= 3 else float(np.nan)

        if cycle_intervals.size < 1:
            return None

        avg_freq = 1.0 / float(np.mean(cycle_intervals))
        avg_cycle_duration = float(np.mean(cycle_intervals))

        _hm = float(np.mean(quality_intervals))
        rhythm_cv = (float(np.std(quality_intervals)) / _hm) * 100.0 if _hm > 1e-9 else 0.0

        amp_cv = (float(np.std(quality_amplitudes)) / _qham) * 100.0 if _qham > 1e-9 else 0.0

        trend = intercept + slope * peak_times

        if _ham > 1.0:
            norm_decrement_slope = float(np.clip(slope / _ham * 100.0, -50.0, 50.0))
        else:
            norm_decrement_slope = float(np.nan)

        # Normalised timing decrement slope (%/cycle)
        if cycle_intervals.size >= 3 and _hm > 1e-9:
            _hdt_idx = np.arange(len(cycle_intervals), dtype=float)
            if SCIPY_OK:
                _ti_slope, _ti_icept, *_ = linregress(_hdt_idx, cycle_intervals)
            else:
                _ti_slope, _ti_icept = np.polyfit(_hdt_idx, cycle_intervals, 1)
            norm_ti_slope = float(np.clip(float(_ti_slope) / _hm * 100.0, -50.0, 50.0))
            # Raw (unnormalised) cycle-duration slope — paper-aligned
            raw_cycle_duration_slope = float(_ti_slope)
        else:
            norm_ti_slope = float(np.nan)
            raw_cycle_duration_slope = float(np.nan)

        # Full-cycle CV (paper-aligned: CV of Cycle Duration, Zarrat Ehsan et al. 2024)
        if cycle_intervals.size >= 2 and avg_cycle_duration > 1e-9:
            cycle_duration_cv = float(np.std(cycle_intervals, ddof=1) / avg_cycle_duration * 100.0)
        else:
            cycle_duration_cv = float(np.nan)

        amp_decrement_onset = self._compute_amp_decrement_onset(amplitudes)
        amp_decrement_pct = self._compute_sequence_effect_pct(quality_amplitudes)

        return {
            "avg_amp": float(np.mean(quality_amplitudes)),
            "amp_cv": float(amp_cv),
            "freq": float(avg_freq),
            "avg_cycle_duration": avg_cycle_duration,
            "cv": float(rhythm_cv),
            "cycle_duration_cv": cycle_duration_cv,
            "norm_decrement_slope": norm_decrement_slope,
            "raw_amp_slope": raw_amp_slope,
            "amp_decrement_onset": amp_decrement_onset,
            "amp_decrement_pct": amp_decrement_pct,
            "norm_ti_slope": norm_ti_slope,
            "raw_cycle_duration_slope": raw_cycle_duration_slope,
            "trend_line": trend,
        }

    # ------------------------------------------------------------------ #
    #  Velocity feature computation                                       #
    # ------------------------------------------------------------------ #
    def _compute_velocity_features(self, sig, cycle_bounds, full_quality):
        """Compute per-cycle and global velocity features.

        Returns
        -------
        dict
            Keys: peak_velocity, mean_velocity, peak_velocity_cv,
            mean_velocity_cv, norm_velocity_decrement_slope,
            velocity_decrement_onset, velocity_decrement_pct,
            global_velocity, cycle_peak_velocities.
        """
        dt = 1.0 / self.fps
        dtheta_dt = np.gradient(sig, dt)
        global_velocity = float(np.mean(np.abs(dtheta_dt)))

        cycle_peak_velocities = []
        cycle_mean_velocities = []
        for lo_i, hi_i in cycle_bounds:
            lo_i = max(0, int(lo_i))
            hi_i = min(len(dtheta_dt), int(hi_i) + 1)
            if hi_i - lo_i < 2:
                continue
            seg_vel = np.abs(dtheta_dt[lo_i:hi_i])
            # Use p95 (paper-aligned: CMS = Cycle Maximum Speed at p95) for
            # robustness to momentary keypoint noise (Zarrat Ehsan et al. 2024).
            cycle_peak_velocities.append(float(np.percentile(seg_vel, PEAK_VELOCITY_PERCENTILE)))
            cycle_mean_velocities.append(float(np.mean(seg_vel)))

        _nan = float(np.nan)
        if len(cycle_peak_velocities) >= 2:
            _cpv = np.array(cycle_peak_velocities, dtype=float)
            _cmv = np.array(cycle_mean_velocities, dtype=float)
            if len(_cpv) == len(full_quality):
                quality_cpv = _cpv[full_quality]
                quality_cmv = _cmv[full_quality]
                if quality_cpv.size < 2:
                    quality_cpv, quality_cmv = _cpv, _cmv
            else:
                quality_cpv, quality_cmv = _cpv, _cmv

            peak_velocity = float(np.mean(quality_cpv))
            mean_velocity = float(np.mean(quality_cmv))
            peak_velocity_cv = (
                float(np.std(quality_cpv, ddof=1) / np.mean(quality_cpv) * 100.0)
                if np.mean(quality_cpv) > 1e-9 and len(quality_cpv) >= 2
                else _nan
            )
            mean_velocity_cv = (
                float(np.std(quality_cmv, ddof=1) / np.mean(quality_cmv) * 100.0)
                if np.mean(quality_cmv) > 1e-9 and len(quality_cmv) >= 2
                else _nan
            )
            if len(quality_cpv) >= 3 and peak_velocity > 1e-9:
                _vel_idx = np.arange(len(quality_cpv), dtype=float)
                if SCIPY_OK:
                    _vd_slope, *_ = linregress(_vel_idx, quality_cpv)
                else:
                    _vd_slope = float(np.polyfit(_vel_idx, quality_cpv, 1)[0])
                norm_velocity_decrement_slope = float(
                    np.clip(float(_vd_slope) / peak_velocity * 100.0, -50.0, 50.0)
                )
                # Raw (unnormalised) peak-velocity slope — paper-aligned (M6a)
                raw_velocity_slope = float(_vd_slope)
            else:
                norm_velocity_decrement_slope = _nan
                raw_velocity_slope = _nan

            # Raw mean-velocity (CAS) slope — paper-aligned speed slope (M6b, Zarrat Ehsan et al. 2024)
            if len(quality_cmv) >= 3:
                _spd_idx = np.arange(len(quality_cmv), dtype=float)
                if SCIPY_OK:
                    _speed_slope, *_ = linregress(_spd_idx, quality_cmv)
                else:
                    _speed_slope = float(np.polyfit(_spd_idx, quality_cmv, 1)[0])
                raw_speed_slope = float(_speed_slope)
            else:
                raw_speed_slope = _nan

            velocity_decrement_onset = self._compute_decrement_onset(quality_cpv)
            velocity_decrement_pct = self._compute_sequence_effect_pct(quality_cpv)
        elif len(cycle_peak_velocities) == 1:
            peak_velocity = float(cycle_peak_velocities[0])
            mean_velocity = float(cycle_mean_velocities[0])
            peak_velocity_cv = mean_velocity_cv = _nan
            norm_velocity_decrement_slope = velocity_decrement_onset = velocity_decrement_pct = _nan
            raw_velocity_slope = raw_speed_slope = _nan
        else:
            peak_velocity = mean_velocity = _nan
            peak_velocity_cv = mean_velocity_cv = _nan
            norm_velocity_decrement_slope = velocity_decrement_onset = velocity_decrement_pct = _nan
            raw_velocity_slope = raw_speed_slope = _nan

        return {
            "peak_velocity": peak_velocity,
            "mean_velocity": mean_velocity,
            "peak_velocity_cv": peak_velocity_cv,
            "mean_velocity_cv": mean_velocity_cv,
            "norm_velocity_decrement_slope": norm_velocity_decrement_slope,
            "raw_velocity_slope": raw_velocity_slope,
            "raw_speed_slope": raw_speed_slope,
            "velocity_decrement_onset": velocity_decrement_onset,
            "velocity_decrement_pct": velocity_decrement_pct,
            "global_velocity": global_velocity,
            "cycle_peak_velocities": cycle_peak_velocities,
        }

    # ------------------------------------------------------------------ #
    #  Coupling features                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_amp_vel_coupling(amplitudes, cycle_peak_velocities):
        """Pearson r between per-cycle amplitude and peak velocity.

        In healthy subjects larger cycles are also faster (coupled);
        PD disrupts this coupling.
        """
        if len(cycle_peak_velocities) < 4 or len(amplitudes) != len(cycle_peak_velocities):
            return float(np.nan)
        _a = np.array(amplitudes, dtype=float)
        _v = np.array(cycle_peak_velocities, dtype=float)
        _finite = np.isfinite(_a) & np.isfinite(_v)
        if _finite.sum() < 4:
            return float(np.nan)
        _a_f, _v_f = _a[_finite], _v[_finite]
        if float(np.std(_a_f)) < 1e-9 or float(np.std(_v_f)) < 1e-9:
            return float(np.nan)
        return float(np.corrcoef(_a_f, _v_f)[0, 1])

    # ------------------------------------------------------------------ #
    #  Hilbert envelope amplitude                                         #
    # ------------------------------------------------------------------ #
    def _compute_hilbert_amplitude(
        self,
        sig: np.ndarray,
        cycle_bounds: list,
        full_quality: np.ndarray,
    ) -> float:
        """Compute mean amplitude via the Hilbert analytic-signal envelope.

        The Hilbert transform yields an instantaneous amplitude envelope
        ``A(t) = |z(t)|`` where ``z(t) = sig(t) + j * H[sig(t)]``.
        Unlike peak-based amplitude, this uses the *entire* waveform
        shape per cycle and is therefore less sensitive to peak clipping
        caused by the lowpass filter or keypoint tracking noise at high
        movement velocities.

        Returns the quality-gated mean of per-cycle median envelope
        values, or ``NaN`` if the Hilbert transform is unavailable or
        too few cycles are present.

        Parameters
        ----------
        sig : np.ndarray
            Bandpass-filtered angle signal.
        cycle_bounds : list of tuple
            ``(lo, hi)`` frame-index pairs for each full cycle.
        full_quality : np.ndarray
            Boolean mask indicating quality-gated cycles.

        Returns
        -------
        float
            Hilbert envelope amplitude (degrees), or ``NaN``.
        """
        if not SCIPY_OK or len(cycle_bounds) < 2:
            return float(np.nan)

        try:
            from scipy.signal import hilbert as _hilbert
        except ImportError:
            return float(np.nan)

        analytic = _hilbert(sig)
        envelope = np.abs(analytic)

        cycle_amps: list[float] = []
        for lo_i, hi_i in cycle_bounds:
            lo_i = max(0, int(lo_i))
            hi_i = min(len(envelope), int(hi_i) + 1)
            if hi_i - lo_i < 3:
                continue
            # Median of envelope within the cycle: robust to edge
            # ringing that the Hilbert transform can produce.
            # Scale by 2× to match peak-to-trough convention used
            # by the standard Mean Amplitude feature.
            cycle_amps.append(float(np.median(envelope[lo_i:hi_i])) * 2.0)

        if len(cycle_amps) < 2:
            return float(np.nan)

        amps_arr = np.array(cycle_amps, dtype=float)
        if len(amps_arr) == len(full_quality):
            quality_amps = amps_arr[full_quality]
            if quality_amps.size < 2:
                quality_amps = amps_arr
        else:
            quality_amps = amps_arr

        return float(np.mean(quality_amps))

    # ------------------------------------------------------------------ #
    #  Half-cycle integral amplitude                                      #
    # ------------------------------------------------------------------ #
    def _compute_integral_amplitude(
        self,
        sig: np.ndarray,
        _alt: list,
        cycle_bounds: list,
        full_quality: np.ndarray,
    ) -> float:
        """Compute mean amplitude by integrating |velocity| over half-cycles.

        For each half-cycle delimited by consecutive alternating extrema,
        the total angular excursion is:

            amp_half = integral_{t_start}^{t_end} |d theta / dt| dt

        A full-cycle amplitude is the sum of two consecutive half-cycle
        excursions.  This measure uses the entire velocity profile rather
        than only the discrete peak/trough values, making it less
        vulnerable to peak clipping from the lowpass filter or from
        noisy keypoint estimates during fast movement.

        Parameters
        ----------
        sig : np.ndarray
            Bandpass-filtered angle signal.
        _alt : list of tuple
            ``[(idx, type), ...]`` alternating extrema from cycle detection.
        cycle_bounds : list of tuple
            ``(lo, hi)`` frame-index pairs for each full cycle (used for
            quality-gating alignment).
        full_quality : np.ndarray
            Boolean mask indicating quality-gated cycles.

        Returns
        -------
        float
            Integral amplitude (degrees), or ``NaN``.
        """
        if len(_alt) < 3:
            return float(np.nan)

        dt = 1.0 / self.fps
        velocity = np.gradient(sig, dt)

        # Compute half-cycle excursions
        half_excursions: list[float] = []
        for i in range(len(_alt) - 1):
            idx_start = int(_alt[i][0])
            idx_end = int(_alt[i + 1][0])
            if idx_end <= idx_start:
                continue
            seg_vel = np.abs(velocity[idx_start : idx_end + 1])
            # np.trapz was removed in NumPy 2.0; use np.trapezoid if available.
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            excursion = float(_trapz(seg_vel, dx=dt))
            half_excursions.append(excursion)

        if len(half_excursions) < 2:
            return float(np.nan)

        # Each half-cycle excursion is the peak-to-trough angular
        # displacement (≈ Mean Amplitude for a sinusoidal signal).
        # Average all half-cycle excursions directly.
        amps_arr = np.array(half_excursions, dtype=float)

        # Align with quality gating: half-excursions are indexed by
        # consecutive extremum pairs, so quality alignment with
        # full-cycle bounds is approximate.  Use all when dimensions
        # don't match.
        if len(amps_arr) == len(full_quality):
            quality_amps = amps_arr[full_quality]
            if quality_amps.size < 2:
                quality_amps = amps_arr
        else:
            quality_amps = amps_arr

        return float(np.mean(quality_amps))

    # ------------------------------------------------------------------ #
    #  Main feature extraction (orchestrates sub-methods)                 #
    # ------------------------------------------------------------------ #
    def extract_features(
        self,
        prominence_deg=15.0,
        max_movement_hz=2.0,
        adaptive_prom_frac=0.20,
        min_half_cycle_s=0.20,
    ):
        """Extract cycle metrics using zero-crossing-guided detection.

        Two-tier approach: primary zero-crossing detection, with peak/valley
        fallback when ZCR yields too few cycles.

        Parameters
        ----------
        prominence_deg : float
            Minimum prominence (degrees) for peak-based fallback.
        max_movement_hz : float
            Maximum expected PS frequency (used only in the peak fallback).
        adaptive_prom_frac : float
            Fraction of signal IQR used as alternative prominence floor.
        min_half_cycle_s : float
            Minimum physiological half-cycle duration (seconds).

        Returns
        -------
        dict or None
            Clinical feature dict, or None if signal is too short / acyclic.
        """
        sig = np.asarray(self.clean_signal, dtype=float)
        if sig.size < 10:
            return None

        # --- Detect extrema ---
        est_period = self._estimate_dominant_period(sig)
        _alt_zcr = self._detect_half_cycles_zcr(sig, est_period, min_half_cycle_s)
        _alt_pk = self._detect_half_cycles_peaks(
            sig,
            est_period,
            prominence_deg,
            adaptive_prom_frac,
            min_half_cycle_s,
            max_movement_hz=max_movement_hz,
        )
        if len(_alt_pk) >= len(_alt_zcr) + 2:
            _alt = _alt_pk
        else:
            _alt = _alt_zcr if len(_alt_zcr) >= 3 else _alt_pk
        if len(_alt) < 3:
            return None

        # --- Build cycles ---
        cycles = self._build_cycles_from_extrema(sig, _alt)
        if cycles is None:
            return None

        amplitudes = cycles["amplitudes"]
        peak_times = cycles["peak_times"]
        cycle_bounds = cycles["cycle_bounds"]
        cycle_intervals = cycles["intervals"]
        n_full = cycles["n_full"]

        # --- NaN quality gating ---
        full_quality, quality_amplitudes, quality_intervals, n_quality_cycles = (
            self._apply_nan_quality_gating(sig, cycle_bounds, amplitudes, cycle_intervals, n_full)
        )

        # --- Amplitude & timing features ---
        amp_feats = self._compute_amplitude_features(
            amplitudes, quality_amplitudes, peak_times, cycle_intervals, quality_intervals
        )
        if amp_feats is None:
            return None

        # --- Timing interruptions ---
        interruption_metrics = self._summarize_timing_interruptions_from_lowpass(
            prominence_deg, max_movement_hz, adaptive_prom_frac, min_half_cycle_s
        )

        # --- Paper-aligned interruption count: full-cycle durations > 2x median ---
        # Zarrat Ehsan et al. (2024) define interruptions using 2× median of
        # full-cycle (peak-to-peak) durations, whereas our two-tier system uses
        # half-cycle intervals with different thresholds.
        if cycle_intervals.size >= 2:
            _median_cd = float(np.median(cycle_intervals))
            num_interruptions = int(np.sum(cycle_intervals > 2.0 * _median_cd))
        else:
            num_interruptions = 0

        # --- Velocity features ---
        vel_feats = self._compute_velocity_features(sig, cycle_bounds, full_quality)

        # --- Signal complexity & coupling ---
        sample_entropy = self._compute_sample_entropy(sig, m=2, r_mult=0.2)
        amp_vel_coupling = self._compute_amp_vel_coupling(
            amplitudes, vel_feats["cycle_peak_velocities"]
        )

        # --- Alternative amplitude estimators ---
        hilbert_amplitude = self._compute_hilbert_amplitude(sig, cycle_bounds, full_quality)
        integral_amplitude = self._compute_integral_amplitude(
            sig, _alt, cycle_bounds, full_quality
        )

        return {
            # --- Core MDS-UPDRS 3.6 clinical features ---
            "avg_amp": amp_feats["avg_amp"],
            "amp_cv": amp_feats["amp_cv"],
            "freq": amp_feats["freq"],
            # Bradykinesia: explicit cycle duration (inverse of frequency)
            "avg_cycle_duration": amp_feats["avg_cycle_duration"],
            "cv": amp_feats["cv"],
            # Full-cycle CV (paper-aligned: Zarrat Ehsan et al. 2024)
            "cycle_duration_cv": amp_feats["cycle_duration_cv"],
            "norm_decrement_slope": amp_feats["norm_decrement_slope"],
            # Raw (unnormalised) amplitude slope — paper-aligned
            "raw_amp_slope": amp_feats["raw_amp_slope"],
            "amp_decrement_onset": amp_feats["amp_decrement_onset"],
            "amp_decrement_pct": amp_feats["amp_decrement_pct"],
            "norm_ti_slope": amp_feats["norm_ti_slope"],
            # Raw (unnormalised) cycle-duration slope — paper-aligned
            "raw_cycle_duration_slope": amp_feats["raw_cycle_duration_slope"],
            "num_hesitations": int(interruption_metrics["num_hesitations"]),
            "num_arrests": int(interruption_metrics["num_arrests"]),
            # Paper-aligned interruption count (2× median full-cycle threshold)
            "num_interruptions": num_interruptions,
            "max_pause_duration_s": float(interruption_metrics["max_pause_duration_s"]),
            "pause_time_ratio": float(interruption_metrics["pause_time_ratio"]),
            # --- Speed-based features ---
            "peak_velocity": vel_feats["peak_velocity"],
            "mean_velocity": vel_feats["mean_velocity"],
            "peak_velocity_cv": vel_feats["peak_velocity_cv"],
            "mean_velocity_cv": vel_feats["mean_velocity_cv"],
            "norm_velocity_decrement_slope": vel_feats["norm_velocity_decrement_slope"],
            # Raw (unnormalised) velocity slopes — paper-aligned
            "raw_velocity_slope": vel_feats["raw_velocity_slope"],
            "raw_speed_slope": vel_feats["raw_speed_slope"],
            "velocity_decrement_onset": vel_feats["velocity_decrement_onset"],
            "velocity_decrement_pct": vel_feats["velocity_decrement_pct"],
            "global_velocity": vel_feats["global_velocity"],
            # --- Signal complexity & coupling features ---
            "sample_entropy": sample_entropy,
            "amp_vel_coupling": amp_vel_coupling,
            # --- Alternative amplitude estimators ---
            "hilbert_amplitude": hilbert_amplitude,
            "integral_amplitude": integral_amplitude,
            # --- Cycle counts (for QC / inspection) ---
            "total_cycles": int(n_full),
            "quality_cycles": n_quality_cycles,
            # --- Diagnostics ---
            "wrist_z_used": self._wrist_z_valid,
            # --- Internal arrays (used for plotting/visualization only) ---
            "peak_times": peak_times,
            "trough_times": cycles["detected_trough_times"],
            "detected_peak_times": cycles["detected_peak_times"],
            "amplitudes": amplitudes,
            "trend_line": amp_feats["trend_line"],
            "peaks_idx": cycles["peaks_idx"],
        }

    # ------------------------------------------------------------------ #
    #  SSL signal export                                                  #
    # ------------------------------------------------------------------ #

    def get_ssl_signal(self, max_len: int = 1024) -> np.ndarray:
        """Return the bandpass-filtered angle signal for SSL pretraining.

        The signal is amplitude-normalised (zero-mean, unit std) so that
        the MAE encoder learns shape rather than absolute scale.  It is
        then cropped or zero-padded to ``max_len`` samples so all videos
        produce a fixed-size input tensor.

        Parameters
        ----------
        max_len : int
            Fixed output length in samples (default 1024 ≈ 10 s at 100 fps
            or 34 s at 30 fps — covers full recordings).

        Returns
        -------
        np.ndarray, shape (max_len,)
            Normalised, padded/cropped signal.  Padding region is zero.
        """
        sig = np.asarray(self.clean_signal, dtype=np.float32)
        # Replace any remaining NaNs with zero (padding value)
        sig = np.where(np.isfinite(sig), sig, 0.0)
        # Amplitude-normalise: zero-mean, unit std (avoid divide-by-zero)
        std = float(np.std(sig))
        if std > 1e-9:
            sig = (sig - float(np.mean(sig))) / std
        # Crop or zero-pad to max_len
        out = np.zeros(max_len, dtype=np.float32)
        n = min(len(sig), max_len)
        out[:n] = sig[:n]
        return out

    # ------------------------------------------------------------------ #
    #  Sample entropy                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_sample_entropy(sig, m=2, r_mult=0.2):
        """Compute sample entropy (SampEn) of a 1D signal.

        Sample entropy measures the complexity / predictability of a time
        series.  A perfectly periodic signal has SampEn near 0; a random
        signal has high SampEn.  Unlike CV, SampEn captures the *temporal
        structure* of irregularity and is not affected by the amplitude
        floor effect.

        Parameters
        ----------
        sig : np.ndarray
            1D signal (bandpass-filtered angle).
        m : int
            Embedding dimension (template length).  Default 2.
        r_mult : float
            Tolerance as fraction of signal std.  Default 0.2 (20%).

        Returns
        -------
        float
            Sample entropy value, or NaN if signal is too short or constant.
        """
        x = np.asarray(sig, dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n < 2 * (m + 1) + 1:
            return float(np.nan)

        sd = float(np.std(x, ddof=1))
        if sd < 1e-12:
            return float(np.nan)
        r = r_mult * sd

        # Downsample long signals for performance (SampEn is O(N^2))
        max_n = 500
        if n > max_n:
            step = n // max_n
            x = x[::step]
            n = len(x)
            if n < 2 * (m + 1) + 1:
                return float(np.nan)

        def _count_matches(length):
            templates = np.array([x[i : i + length] for i in range(n - length)])
            count = 0
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            return count

        a = _count_matches(m + 1)
        b = _count_matches(m)

        if b == 0:
            return float(np.nan)
        if a == 0:
            # No matches at length m+1: SampEn is very high
            # Return ln(b) as a conservative upper bound
            return float(np.log(b))

        return float(-np.log(a / b))

    # ------------------------------------------------------------------ #
    #  Amplitude decrement onset detection                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_decrement_onset(values):
        """Estimate where sustained decrement begins in a cycle sequence.

        Captures MDS-UPDRS sequence-effect timing as a continuous value in
        [0, 1], where 0 means decrement begins immediately and 1 means no
        decrement or decrement only at the very end.

        Algorithm
        ---------
        1. Smooth the series with a 3-cycle moving average.
        2. Check overall slope: if the linear regression slope across the
           full series is not meaningfully negative (normalised slope
           > −2 %/cycle), return 1.0 (no decrement).
        3. Otherwise, slide a window across the smoothed series and find
           the first position where the local normalised slope is
           sufficiently negative (< −3 %/cycle), indicating the start of
           sustained decrement.
        4. Return onset_index / (n_cycles − 1).

        Parameters
        ----------
        values : np.ndarray
            Per-cycle values, in temporal order.

        Returns
        -------
        float
            Fractional onset position in [0, 1], or NaN if fewer than
            3 cycles.
        """
        amps = np.asarray(values, dtype=float)
        n = len(amps)
        if n < 3:
            return float(np.nan)

        # Smooth with a 3-cycle moving average (reflect-padded)
        win = min(3, n)
        if win < 2:
            smoothed = amps.copy()
        else:
            pad = win // 2
            padded = np.pad(amps, pad, mode="reflect")
            kernel = np.ones(win) / win
            smoothed = np.convolve(padded, kernel, mode="same")[pad : pad + n]

        # Check overall slope first: if not meaningfully negative, no decrement
        overall_mean = float(np.mean(smoothed))
        if overall_mean < 1e-6:
            return float(np.nan)

        if SCIPY_OK and n >= 3:
            _idx = np.arange(n, dtype=float)
            _sl, _, _, _, _ = linregress(_idx, smoothed)
            overall_norm_slope = float(_sl) / overall_mean * 100.0
        else:
            _sl, _ = np.polyfit(np.arange(n, dtype=float), smoothed, 1)
            overall_norm_slope = float(_sl) / overall_mean * 100.0

        # If overall slope is not meaningfully negative, no decrement
        if overall_norm_slope > -1.0:
            return 1.0

        # Sliding window: find onset of sustained negative slope
        slope_win = max(3, n // 3)  # use ~1/3 of cycles as window
        slope_win = min(slope_win, n - 1)
        neg_slope_thresh = -2.0  # %/cycle normalised slope threshold

        for start in range(n - slope_win + 1):
            seg = smoothed[start : start + slope_win]
            seg_mean = float(np.mean(seg))
            if seg_mean < 1e-6:
                continue
            _idx = np.arange(len(seg), dtype=float)
            if SCIPY_OK:
                _local_sl, _, _, _, _ = linregress(_idx, seg)
            else:
                _local_sl, _ = np.polyfit(_idx, seg, 1)
            local_norm_slope = float(_local_sl) / seg_mean * 100.0
            if local_norm_slope < neg_slope_thresh:
                return float(start) / float(max(n - 1, 1))

        # If no sustained local decrement found despite negative overall slope,
        # report onset near the end (decrement is gradual/late)
        return 0.85

    @staticmethod
    def _compute_amp_decrement_onset(amplitudes):
        """Estimate where amplitude decrement begins across the task."""
        return KinematicAnalyzer._compute_decrement_onset(amplitudes)

    @staticmethod
    def _compute_sequence_effect_pct(values):
        """Early-to-late percent decrement for amplitude or velocity.

        Returns positive values when late-task performance is lower than
        early-task performance.  The early/late window size is controlled
        by ``AMP_DECREMENT_SPLIT_FRAC`` (default 1/3).
        """
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = len(arr)
        if n < 4:
            return float(np.nan)

        win = max(2, min(4, int(n * AMP_DECREMENT_SPLIT_FRAC)))
        early = float(np.median(arr[:win]))
        late = float(np.median(arr[-win:]))
        if early <= 1e-6:
            return float(np.nan)
        return float((early - late) / early * 100.0)

    def _extract_half_cycle_intervals(
        self, sig, prominence_deg, max_movement_hz, adaptive_prom_frac, min_half_cycle_s
    ):
        """Return consecutive-extremum half-cycle durations in seconds."""
        sig = np.asarray(sig, dtype=float)
        if sig.size < 10:
            return np.array([], dtype=float)

        est_period = self._estimate_dominant_period(sig)
        alt = self._detect_half_cycles_zcr(sig, est_period, min_half_cycle_s)
        if len(alt) < 3:
            alt = self._detect_half_cycles_peaks(
                sig,
                est_period,
                prominence_deg,
                adaptive_prom_frac,
                min_half_cycle_s,
                max_movement_hz=max_movement_hz,
            )
        if len(alt) < 3:
            return np.array([], dtype=float)

        ext_times = self.t[np.array([idx for idx, _ in alt], dtype=int)]
        half_dts = np.diff(ext_times)
        return half_dts[half_dts > 1e-6]

    def _summarize_timing_interruptions_from_lowpass(
        self, prominence_deg, max_movement_hz, adaptive_prom_frac, min_half_cycle_s
    ):
        """Summarise hesitations, arrests, and overall pause burden.

        MDS-UPDRS 3.6 explicitly mentions interruptions/hesitations as well
        as arrests/halts.  The existing arrest count only captured the most
        severe timing failures.  This summary adds:

        * `num_hesitations`: moderately prolonged half-cycles.
        * `num_arrests`: more severe prolonged half-cycles.
        * `max_pause_duration_s`: longest hesitation/arrest duration.
        * `pause_time_ratio`: fraction of total half-cycle time spent in
          hesitation/arrest intervals.
        """
        lp = np.asarray(self._lowpass_signal, dtype=float)
        half_dts = self._extract_half_cycle_intervals(
            lp,
            prominence_deg,
            max_movement_hz,
            adaptive_prom_frac,
            min_half_cycle_s,
        )
        if half_dts.size < 2:
            return {
                "num_hesitations": 0,
                "num_arrests": 0,
                "max_pause_duration_s": 0.0,
                "pause_time_ratio": 0.0,
            }

        median_half_dt = float(np.median(half_dts))
        hesitation_thresh = max(
            self._HESITATION_RATIO * median_half_dt,
            self._HESITATION_ABS_FLOOR_S,
        )
        arrest_thresh = max(1.5 * median_half_dt, ARREST_MIN_DURATION_S)

        arrest_mask = half_dts > arrest_thresh
        hesitation_mask = (half_dts > hesitation_thresh) & ~arrest_mask
        pause_mask = hesitation_mask | arrest_mask

        total_pause_time = float(np.sum(half_dts[pause_mask])) if np.any(pause_mask) else 0.0
        total_half_cycle_time = float(np.sum(half_dts)) if np.sum(half_dts) > 1e-9 else 0.0

        return {
            "num_hesitations": int(np.sum(hesitation_mask)),
            "num_arrests": int(np.sum(arrest_mask)),
            "max_pause_duration_s": (
                float(np.max(half_dts[pause_mask])) if np.any(pause_mask) else 0.0
            ),
            "pause_time_ratio": (
                float(total_pause_time / total_half_cycle_time)
                if total_half_cycle_time > 1e-9
                else 0.0
            ),
        }

    # ------------------------------------------------------------------ #
    #  Per-video signal quality score                                     #
    # ------------------------------------------------------------------ #
    _PS_FREQ_LO = 0.3  # Hz — lower bound of physiological PS range
    _PS_FREQ_HI = 4.0  # Hz — upper bound of physiological PS range

    @staticmethod
    def _valid_window_from_nan_mask(nan_mask):
        """Return (start, end) indices of the valid keypoint window.

        The window is inclusive and spans from the first non-NaN frame to
        the last non-NaN frame in the original (pre-interpolation) signal.
        Returns ``None`` when no valid frame exists.
        """
        valid_idx = np.where(~np.asarray(nan_mask, dtype=bool))[0]
        if valid_idx.size == 0:
            return None
        return int(valid_idx[0]), int(valid_idx[-1])

    def compute_signal_quality(self, metrics=None, ps_start_frame=None, ps_end_frame=None):
        """Compute a per-video signal quality score in [0, 1].

        Measures how closely the tracked angle signal resembles an ideal
        pronation-supination oscillation, **independent of clinical severity**.
        A perfect quasi-sinusoidal PS signal scores ~1; random noise or
        accumulated-drift artefacts score near 0.

        Sub-scores (all mapped to [0, 1], higher = better):

        1. **spectral_concentration** — fraction of FFT power in a ±0.3 Hz
           band around the dominant frequency, restricted to the
           physiological PS range (0.3–4 Hz).  Rewards periodic signals.
        2. **autocorr_strength** — peak normalised autocorrelation at the
           dominant period.  Rewards clean repeating waveforms.
        3. **cycle_regularity** — mean Pearson *r* between consecutive
           time-warped inter-peak segments.  Rewards consistent cycle
           shape.
        4. **signal_coverage** — 1 − NaN fraction of the raw angle signal
           within the evaluation window.  Rewards complete landmark detection.
        5. **freq_plausibility** — continuous measure of whether the
           detected dominant frequency is within the physiological PS
           range, using a smooth sigmoid falloff at the boundaries.
        6. **cycle_yield** — sigmoid ramp based on the number of detected
           full cycles (3 → 0.5, 6+ → ~1.0).  Rewards sufficient data
           for reliable feature extraction.

        The aggregate score is the weighted mean of the sub-scores.

        Parameters
        ----------
        metrics : dict or None
            Feature dict returned by ``extract_features()``.  If None,
            only signal-level sub-scores are computed and cycle-dependent
            sub-scores default to 0.
        ps_start_frame : int or None
            First frame of the PS-active segment (i.e., the trimmed window
            start returned by ``trim_track_to_ps_segment``).  When provided
            together with ``ps_end_frame``, the quality evaluation window is
            restricted to ``[ps_start_frame, ps_end_frame]`` instead of the
            full first-to-last keypoint window.  This prevents the zeroed
            non-task frames outside the PS segment from contaminating
            spectral concentration and autocorrelation sub-scores.
        ps_end_frame : int or None
            Last frame (inclusive) of the PS-active segment.

        Returns
        -------
        dict
            ``"signal_quality"`` : float in [0, 1]
            ``"sq_sub_scores"`` : dict of individual sub-scores
        """
        sig_full = np.asarray(self.clean_signal, dtype=float)
        nan_mask_full = getattr(self, "nan_mask", np.zeros(len(sig_full), dtype=bool))

        _null_sub = {
            "spectral_concentration": 0.0,
            "autocorr_strength": 0.0,
            "cycle_regularity": 0.0,
            "signal_coverage": 0.0,
            "freq_plausibility": 0.0,
            "cycle_yield": 0.0,
        }

        # Determine evaluation window.
        # When PS-activity trimming was applied, restrict quality evaluation to
        # [ps_start_frame, ps_end_frame] so that zeroed pre/post-task frames do
        # not pollute spectral concentration or autocorrelation sub-scores.
        # Without trimming, fall back to the keypoint-valid window
        # (first → last non-NaN raw frame).
        if ps_start_frame is not None and ps_end_frame is not None:
            n = len(sig_full)
            win_start = int(max(0, min(ps_start_frame, n - 1)))
            win_end = int(max(0, min(ps_end_frame, n - 1)))
            if win_start > win_end:
                return {"signal_quality": 0.0, "sq_sub_scores": _null_sub}
        else:
            win = self._valid_window_from_nan_mask(nan_mask_full)
            if win is None:
                return {"signal_quality": 0.0, "sq_sub_scores": _null_sub}
            win_start, win_end = win

        sig = sig_full[win_start : win_end + 1]
        nan_mask = nan_mask_full[win_start : win_end + 1]

        metrics_in_win = metrics
        if metrics is not None and "peaks_idx" in metrics:
            peaks_idx = np.asarray(metrics.get("peaks_idx", []), dtype=int)
            peaks_idx = peaks_idx[(peaks_idx >= win_start) & (peaks_idx <= win_end)]
            metrics_in_win = dict(metrics)
            metrics_in_win["peaks_idx"] = peaks_idx - win_start

        # --- Sub-score 1: Spectral concentration ---
        spectral = self._sq_spectral_concentration(sig)

        # --- Sub-score 2: Autocorrelation strength ---
        autocorr = self._sq_autocorr_strength(sig)

        # --- Sub-score 3: Cycle regularity ---
        cycle_reg = self._sq_cycle_regularity(sig, metrics_in_win)

        # --- Sub-score 4: Signal coverage ---
        coverage = 1.0 - (float(np.sum(nan_mask)) / max(len(nan_mask), 1))

        # --- Sub-score 5: Frequency plausibility ---
        freq_plaus = self._sq_freq_plausibility(sig)

        # --- Sub-score 6: Cycle yield ---
        if metrics is not None and "peak_times" in metrics and len(metrics.get("peak_times", [])):
            peak_times = np.asarray(metrics.get("peak_times", []), dtype=float)
            t0 = float(self.t[win_start])
            t1 = float(self.t[win_end])
            n_cycles = int(np.sum((peak_times >= t0) & (peak_times <= t1)))
        else:
            n_cycles = int(metrics.get("total_cycles", 0)) if metrics else 0
        # Sigmoid: 0 cycles → 0, 3 → 0.5, 6 → ~0.88, 10+ → ~0.98
        cycle_yield = 1.0 / (1.0 + np.exp(-0.7 * (n_cycles - 3.0)))

        weights = {
            "spectral_concentration": 2.0,
            "autocorr_strength": 2.0,
            "cycle_regularity": 1.5,
            "signal_coverage": 1.0,
            "freq_plausibility": 1.0,
            "cycle_yield": 1.5,
        }
        sub_scores = {
            "spectral_concentration": float(np.clip(spectral, 0, 1)),
            "autocorr_strength": float(np.clip(autocorr, 0, 1)),
            "cycle_regularity": float(np.clip(cycle_reg, 0, 1)),
            "signal_coverage": float(np.clip(coverage, 0, 1)),
            "freq_plausibility": float(np.clip(freq_plaus, 0, 1)),
            "cycle_yield": float(np.clip(cycle_yield, 0, 1)),
        }
        w_total = sum(weights.values())
        quality = sum(weights[k] * sub_scores[k] for k in weights) / w_total

        return {
            "signal_quality": float(np.clip(quality, 0.0, 1.0)),
            "sq_sub_scores": sub_scores,
        }

    def _sq_spectral_concentration(self, sig):
        """Fraction of power in ±0.3 Hz around dominant freq (physiological range)."""
        sig = sig[~np.isnan(sig)]
        if len(sig) < 30:
            return 0.0
        sig = sig - np.mean(sig)
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        power = np.abs(np.fft.rfft(sig)) ** 2

        mask = (freqs >= self._PS_FREQ_LO) & (freqs <= self._PS_FREQ_HI)
        if mask.sum() < 2:
            return 0.0
        physio_power = power[mask]
        physio_freqs = freqs[mask]
        total = float(np.sum(physio_power))
        if total < 1e-12:
            return 0.0

        dom_idx = int(np.argmax(physio_power))
        dom_freq = physio_freqs[dom_idx]

        band = (physio_freqs >= dom_freq - 0.3) & (physio_freqs <= dom_freq + 0.3)
        return float(np.sum(physio_power[band])) / total

    def _sq_autocorr_strength(self, sig):
        """Peak normalised ACF at dominant period in physiological range."""
        sig = sig[~np.isnan(sig)]
        if len(sig) < 30:
            return 0.0
        sig = sig - np.mean(sig)
        var = float(np.dot(sig, sig))
        if var < 1e-12:
            return 0.0

        min_lag = max(1, int(self.fps / self._PS_FREQ_HI))
        max_lag = min(len(sig) // 2, int(self.fps / self._PS_FREQ_LO))
        if min_lag >= max_lag:
            return 0.0

        acf = np.array(
            [
                float(np.dot(sig[: len(sig) - lag], sig[lag:])) / var
                for lag in range(min_lag, max_lag + 1)
            ]
        )
        if acf.size == 0:
            return 0.0
        return float(np.clip(np.max(acf), 0.0, 1.0))

    def _sq_cycle_regularity(self, sig, metrics):
        """Mean Pearson r between consecutive time-warped inter-peak segments."""
        if metrics is None:
            return 0.0
        peak_indices = metrics.get("peaks_idx", None)
        if peak_indices is None or len(peak_indices) < 3:
            return 0.0

        # Scale resample_len proportionally to fps so we always oversample
        # cycle segments ~2× regardless of frame rate.  At 25 fps this gives
        # 50 (original behaviour); at 50 fps (GIMMVFI) it gives 100, matching
        # the same smoothing effect and keeping the metric comparable.
        resample_len = max(50, round(self.fps * 2.0))
        segments = []
        for i in range(len(peak_indices) - 1):
            lo, hi = int(peak_indices[i]), int(peak_indices[i + 1])
            if hi - lo < 3 or hi > len(sig):
                continue
            seg = sig[lo:hi]
            if np.any(np.isnan(seg)):
                continue
            x_old = np.linspace(0, 1, len(seg))
            x_new = np.linspace(0, 1, resample_len)
            segments.append(np.interp(x_new, x_old, seg))

        if len(segments) < 2:
            return 0.0

        correlations = []
        for i in range(len(segments) - 1):
            s1, s2 = segments[i], segments[i + 1]
            if float(np.std(s1)) < 1e-9 or float(np.std(s2)) < 1e-9:
                continue
            r = float(np.corrcoef(s1, s2)[0, 1])
            if np.isfinite(r):
                correlations.append(r)

        if not correlations:
            return 0.0
        # Map r from [-1, 1] to [0, 1]
        return float(np.clip((np.mean(correlations) + 1.0) / 2.0, 0.0, 1.0))

    def _sq_freq_plausibility(self, sig):
        """Smooth plausibility score for whether dominant freq is in PS range."""
        sig = sig[~np.isnan(sig)]
        if len(sig) < 30:
            return 0.0
        sig = sig - np.mean(sig)
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        power = np.abs(np.fft.rfft(sig)) ** 2

        # Only consider the wider search range, then check if dominant is
        # within the physiological window
        mask = (freqs >= 0.1) & (freqs <= 8.0)
        if mask.sum() < 2:
            return 0.0
        search_power = power[mask]
        search_freqs = freqs[mask]
        if float(np.sum(search_power)) < 1e-12:
            return 0.0

        dom_freq = float(search_freqs[int(np.argmax(search_power))])

        # Sigmoid falloff: 1.0 inside [0.3, 4.0], smooth decay outside
        # Decay constant k controls steepness; k=10 → half-score at ±0.1 Hz
        k = 10.0
        if dom_freq < self._PS_FREQ_LO:
            return float(1.0 / (1.0 + np.exp(-k * (dom_freq - self._PS_FREQ_LO))))
        elif dom_freq > self._PS_FREQ_HI:
            return float(1.0 / (1.0 + np.exp(k * (dom_freq - self._PS_FREQ_HI))))
        return 1.0


# ============================================================
# Angle computation helpers
# ============================================================


def _knuckle_line_angle_rad_standalone(lm_arr):
    """Robust knuckle-line angle: circular mean of three overlapping MCP pairs.

    Returns radians; apply np.unwrap across the frame sequence.
    """
    pairs = [
        (HandLandmark.INDEX_FINGER_MCP, HandLandmark.PINKY_MCP),
        (HandLandmark.INDEX_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
        (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.PINKY_MCP),
    ]
    cos_sum, sin_sum, n = 0.0, 0.0, 0
    for a, b in pairs:
        dx = float(lm_arr[b, 0] - lm_arr[a, 0])
        dy = float(lm_arr[b, 1] - lm_arr[a, 1])
        mag = (dx * dx + dy * dy) ** 0.5
        if mag > 1e-9:
            cos_sum += dx / mag
            sin_sum += dy / mag
            n += 1
    if n == 0:
        dx = float(lm_arr[HandLandmark.PINKY_MCP, 0] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 0])
        dy = float(lm_arr[HandLandmark.PINKY_MCP, 1] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 1])
    else:
        dx, dy = cos_sum / n, sin_sum / n
    return float(np.arctan2(dy, dx))


def _hand_roll_angle_standalone(lm_arr):
    """3D palm roll proxy around the wrist->middle-MCP axis.

    Returns the roll angle in radians, or NaN when degenerate.
    """

    def _su(v, eps=1e-9):
        n = float(np.linalg.norm(v))
        return v / n if n >= eps else None

    w = lm_arr[HandLandmark.WRIST, :3]
    m = lm_arr[HandLandmark.MIDDLE_FINGER_MCP, :3]
    i = lm_arr[HandLandmark.INDEX_FINGER_MCP, :3]
    p = lm_arr[HandLandmark.PINKY_MCP, :3]

    u = _su(m - w)
    if u is None:
        return np.nan
    n_vec = _su(np.cross(i - w, p - w))
    if n_vec is None:
        return np.nan
    idx_dir = _su(i - w)
    if idx_dir is None:
        return np.nan
    b1 = idx_dir - np.dot(idx_dir, u) * u
    b1 = _su(b1)
    if b1 is None:
        return np.nan
    b2 = _su(np.cross(u, b1))
    if b2 is None:
        return np.nan
    return float(np.arctan2(float(np.dot(n_vec, b2)), float(np.dot(n_vec, b1))))


def _compute_adaptive_vis_threshold(frames_dict, vis_lm_indices):
    """Compute a per-video adaptive visibility threshold.

    Returns the effective threshold: the maximum of the hard floor and
    the ``ADAPTIVE_VIS_PERCENTILE``-th percentile of per-frame minimum
    MCP visibility, capped at the global ``VISIBILITY_THRESHOLD`` so
    that easy videos are not degraded.

    When ``ADAPTIVE_VISIBILITY`` is False, returns the fixed global
    ``VISIBILITY_THRESHOLD``.
    """
    if not ADAPTIVE_VISIBILITY:
        return VISIBILITY_THRESHOLD

    min_vis_per_frame = []
    for f, lm_arr in frames_dict.items():
        if lm_arr.shape[1] >= 4:
            vis = min(float(lm_arr[li, 3]) for li in vis_lm_indices)
            min_vis_per_frame.append(vis)

    if not min_vis_per_frame:
        return VISIBILITY_THRESHOLD

    vis_arr = np.array(min_vis_per_frame)
    adaptive = max(ADAPTIVE_VIS_FLOOR, float(np.percentile(vis_arr, ADAPTIVE_VIS_PERCENTILE)))
    # Don't exceed the global threshold — easy videos keep the strict gate
    return min(adaptive, VISIBILITY_THRESHOLD)


def _reject_angle_spikes(raw_rad, max_jump_rad=np.pi / 2, fps=None):
    """Mark single-frame angle spikes as NaN before phase unwrapping.

    A *spike* is a frame whose angle differs from **both** neighbours by
    more than ``max_jump_rad`` (wrap-aware).  Such jumps are non-physical
    for pronation-supination motion at any clinically observed frequency
    and typically arise from:

    * Landmark detection errors (wrong hand detected, severely
      mislocated MCPs).
    * Transient hand-track identity swaps in the tracker.

    Removing them **before** ``np.unwrap`` prevents accumulation of
    ±2π correction errors that would otherwise produce the massive
    monotonic drift seen in problematic videos (e.g., 800–1300° range
    in a 10 s recording).

    Parameters
    ----------
    raw_rad : np.ndarray, shape (N,)
        Wrapped angles in radians (may contain NaN).
    max_jump_rad : float
        Maximum plausible angular change between adjacent valid frames
        **at BASE_FPS (25 fps)**.  Default π/2 (90°).  At 25 fps and the
        fastest observed PS (3.5 Hz, ≈60° amplitude), the actual
        per-frame change is ≈ 60° × 2π × 3.5 / 25 ≈ 53°, well below 90°.
    fps : float or None
        Actual video frame rate.  When provided, ``max_jump_rad`` is
        scaled by ``BASE_FPS / fps`` so the angular-velocity ceiling
        (rad/s) stays constant regardless of frame rate.  At 80 fps
        the effective per-frame threshold becomes ≈ 28° instead of 90°.

    Returns
    -------
    np.ndarray
        Copy of ``raw_rad`` with spike frames set to NaN.
    """
    if fps is not None and fps > 0:
        max_jump_rad = max_jump_rad * BASE_FPS / fps
    result = raw_rad.copy()
    valid = np.where(~np.isnan(result))[0]
    if len(valid) < 3:
        return result

    angles = result[valid]
    to_nan = []
    for i in range(1, len(angles) - 1):
        d_prev = abs(angles[i] - angles[i - 1])
        d_next = abs(angles[i] - angles[i + 1])
        # Wrap-aware angular distance (shortest arc on the unit circle)
        d_prev = min(d_prev, 2.0 * np.pi - d_prev)
        d_next = min(d_next, 2.0 * np.pi - d_next)
        if d_prev > max_jump_rad and d_next > max_jump_rad:
            to_nan.append(valid[i])

    for idx in to_nan:
        result[idx] = np.nan

    return result


def _unwrap_axial_angle_segments(raw_rad):
    """Unwrap a π-periodic orientation signal stored as wrapped radians.

    Knuckle-line orientation is axial: ``theta`` and ``theta + π`` represent
    the same physical pose.  Applying standard 2π unwrapping directly can
    accumulate offsets when frame-level orientation occasionally flips by π,
    producing monotonic drift in the downstream filtered signal.

    Robust approach:
      1) map to directional phase with angle doubling,
      2) perform standard gap-aware unwrapping,
      3) map back by halving.

    Parameters
    ----------
    raw_rad : np.ndarray
        Wrapped orientation angles in radians (NaN allowed).

    Returns
    -------
    np.ndarray
        Unwrapped orientation angles in radians with NaNs preserved.
    """
    raw_rad = np.asarray(raw_rad, dtype=np.float64)
    valid = ~np.isnan(raw_rad)
    if valid.sum() < 2:
        return raw_rad.copy()
    doubled = raw_rad.copy()
    doubled[valid] = 2.0 * doubled[valid]
    return 0.5 * _unwrap_segments(doubled)


def _build_unwrapped_angle_deg(frames_dict, total_frames, roll_fn, fps=None):
    """Collect per-frame roll angles, unwrap the phase, and convert to degrees."""
    _VIS_LMS = [
        int(HandLandmark.INDEX_FINGER_MCP),
        int(HandLandmark.MIDDLE_FINGER_MCP),
        int(HandLandmark.RING_FINGER_MCP),
        int(HandLandmark.PINKY_MCP),
    ]
    vis_thresh = _compute_adaptive_vis_threshold(frames_dict, _VIS_LMS)
    raw_rad = np.full(total_frames, np.nan)
    for f, lm_arr in frames_dict.items():
        if f < total_frames:
            if lm_arr.shape[1] >= 4:
                if any(float(lm_arr[li, 3]) < vis_thresh for li in _VIS_LMS):
                    continue
            ang = roll_fn(lm_arr)
            if ang is not None:
                raw_rad[f] = float(ang)
    raw_rad = _reject_angle_spikes(raw_rad, fps=fps)
    valid = ~np.isnan(raw_rad)
    if valid.sum() < 2:
        return np.degrees(raw_rad)
    return np.degrees(_unwrap_axial_angle_segments(raw_rad))


def _build_pca_angle_deg(frames_dict, total_frames, fps=None):
    """View-invariant PS angle via per-video PCA on 2D (x,y) knuckle-line vectors.

    For each frame, computes the mean unit 2D direction vector across three
    overlapping MCP-to-MCP pairs (circular mean in x,y only).  The z coordinate
    is deliberately excluded because:
      - x,y are directly observed image coordinates — accurate and stable.
      - z is estimated monocularly — noisy, with error magnitude comparable to
        the PS signal for frontal recordings.
      - Including z in the 3D SVD causes σ1 (PS arc) ≈ σ2 (z-noise) for many
        patients, making the PCA eigenvectors ill-conditioned and producing
        high cycle-to-cycle amplitude variability.

    The 2D PCA still provides the key benefits of the PCA approach:
      - Per-video centering by median direction removes mean-pose bias.
      - SVD finds the principal axis of motion in image space, which robustly
        handles hands that are not perfectly horizontally oriented.
      - arctan2 projection gives a smooth 1D signal that is phase-unwrappable.

    Because x,y are directly measured, amplitude scale is correct and stable.
    σ1 >> σ2 is reliably true in 2D (the transverse motion is much smaller
    than the PS arc), so the eigenvectors are always well-conditioned.

    Falls back to _build_unwrapped_angle_deg + _knuckle_line_angle_rad_standalone
    (also 2D) when fewer than 10 valid frames are available.
    """
    _PAIRS = [
        (HandLandmark.INDEX_FINGER_MCP, HandLandmark.PINKY_MCP),
        (HandLandmark.INDEX_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
        (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.PINKY_MCP),
    ]
    _VIS = [
        int(HandLandmark.INDEX_FINGER_MCP),
        int(HandLandmark.MIDDLE_FINGER_MCP),
        int(HandLandmark.RING_FINGER_MCP),
        int(HandLandmark.PINKY_MCP),
    ]

    vis_thresh = _compute_adaptive_vis_threshold(frames_dict, _VIS)

    raw_vecs = {}
    for f, lm_arr in frames_dict.items():
        if f >= total_frames:
            continue
        if lm_arr.shape[1] >= 4:
            if any(float(lm_arr[li, 3]) < vis_thresh for li in _VIS):
                continue
        cos_sum = np.zeros(2, dtype=np.float64)  # 2D: x,y only
        n_v = 0
        for a, b in _PAIRS:
            v = lm_arr[b, :2].astype(np.float64) - lm_arr[a, :2].astype(np.float64)  # x,y only
            mag = float(np.linalg.norm(v))
            if mag > 1e-9:
                cos_sum += v / mag
                n_v += 1
        if n_v == 0:
            continue
        raw_vecs[f] = cos_sum / float(n_v)

    if len(raw_vecs) < 10:
        return _build_unwrapped_angle_deg(
            frames_dict, total_frames, _knuckle_line_angle_rad_standalone, fps=fps
        )

    frame_order = sorted(raw_vecs.keys())
    V = np.array([raw_vecs[f] for f in frame_order], dtype=np.float64)

    V_centred = V - np.median(V, axis=0)
    try:
        _, _, Vt = np.linalg.svd(V_centred, full_matrices=False)
        e1, e2 = Vt[0], Vt[1]
    except np.linalg.LinAlgError:
        return _build_unwrapped_angle_deg(
            frames_dict, total_frames, _knuckle_line_angle_rad_standalone, fps=fps
        )

    raw_rad = np.full(total_frames, np.nan)
    for f in frame_order:
        v = raw_vecs[f]
        mag = float(np.linalg.norm(v))
        if mag < 1e-9:
            continue
        v_n = v / mag
        raw_rad[f] = float(np.arctan2(float(np.dot(v_n, e2)), float(np.dot(v_n, e1))))

    raw_rad = _reject_angle_spikes(raw_rad, fps=fps)
    valid = ~np.isnan(raw_rad)
    if valid.sum() < 2:
        return np.degrees(raw_rad)
    return np.degrees(_unwrap_axial_angle_segments(raw_rad))


def _compute_inter_mcp_span(frames_dict, total_frames):
    """Compute the mean inter-MCP span (Index-to-Pinky distance) across valid frames.

    The inter-MCP span (Euclidean distance between Index MCP and Pinky MCP
    in normalised image coordinates) serves as a camera-distance proxy
    used for computing the arm swing index.

    Returns
    -------
    float or None
        Mean inter-MCP span in normalised image units, or None if fewer
        than 5 valid frames.
    """
    spans = []
    for f, lm_arr in frames_dict.items():
        if f >= total_frames:
            continue
        try:
            dx = float(lm_arr[HandLandmark.PINKY_MCP, 0] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 0])
            dy = float(lm_arr[HandLandmark.PINKY_MCP, 1] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 1])
            span = (dx * dx + dy * dy) ** 0.5
            if span > 1e-6:
                spans.append(span)
        except (IndexError, TypeError):
            continue
    if len(spans) < 5:
        return None
    return float(np.median(spans))


def _compute_inter_mcp_span_px(frames_dict, total_frames, frame_width, frame_height):
    """Compute the median index-to-pinky MCP distance in pixels.

    Converts the normalized landmark coordinates to pixel space using the
    actual frame dimensions, producing a concrete, recording-protocol-checkable
    measure of how large the hand appears in the video.  Unlike the normalized
    span, this value scales with camera distance and video resolution, making
    it suitable for pre-processing quality checks (e.g., "was the hand close
    enough to the camera to be reliably tracked?").

    Parameters
    ----------
    frames_dict : dict
        Frame index → landmark array (normalized MediaPipe coords).
    total_frames : int
        Total number of frames in the video.
    frame_width : int or float or None
        Video width in pixels.
    frame_height : int or float or None
        Video height in pixels.

    Returns
    -------
    float or None
        Median inter-MCP span in pixels, or None if fewer than 5 valid
        frames or if frame dimensions are unavailable.
    """
    if not (frame_width and frame_height and frame_width > 0 and frame_height > 0):
        return None
    w = float(frame_width)
    h = float(frame_height)
    spans_px = []
    for f, lm_arr in frames_dict.items():
        if f >= total_frames:
            continue
        try:
            dx_norm = float(
                lm_arr[HandLandmark.PINKY_MCP, 0] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 0]
            )
            dy_norm = float(
                lm_arr[HandLandmark.PINKY_MCP, 1] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 1]
            )
            span_px = ((dx_norm * w) ** 2 + (dy_norm * h) ** 2) ** 0.5
            if span_px > 1e-3:
                spans_px.append(span_px)
        except (IndexError, TypeError):
            continue
    if len(spans_px) < 5:
        return None
    return float(np.median(spans_px))


def _compute_arm_swing_index(
    frames_dict, total_frames, ps_start_frame=None, ps_end_frame=None, inter_mcp_span=None
):
    """Compute the arm-swing index from wrist XY positional spread.

    During a correct PS task the forearm rotates while the wrist stays nearly
    stationary in image space.  Patients who flail or drift their arm produce
    a large trajectory for the wrist landmark.  This function quantifies that
    drift as the 2-D RMS spread of the wrist position, normalised by the
    inter-MCP span so the result is dimensionless and camera-distance
    invariant.

    Algorithm
    ---------
    1. Collect per-frame wrist (x, y) image coordinates restricted to the
       PS-active window (``ps_start_frame``..``ps_end_frame``).  If those
       bounds are not known, all detected frames are used.
    2. Compute ``arm_swing_rms = sqrt(var(wx) + var(wy))``, the RMS of 2-D
       deviations from the mean wrist position (equivalent to the mean
       squared displacement from the centroid).
    3. If ``inter_mcp_span`` is available and > 0, return the normalised
       value ``arm_swing_rms / inter_mcp_span``; otherwise return the raw
       pixel-normalised value (coordinates are already in [0,1] range as
       MediaPipe landmark coordinates are normalised to frame dimensions).

    Returns
    -------
    float or None
        Arm swing index (normalised RMS wrist displacement, dimensionless),
        or None if fewer than 5 valid frames are available.
    """
    lo = int(ps_start_frame) if ps_start_frame is not None else 0
    hi = int(ps_end_frame) if ps_end_frame is not None else total_frames - 1

    wx_list: list[float] = []
    wy_list: list[float] = []
    for f, lm_arr in frames_dict.items():
        fi = int(f)
        if fi < lo or fi > hi or fi >= total_frames:
            continue
        try:
            wx_list.append(float(lm_arr[HandLandmark.WRIST, 0]))
            wy_list.append(float(lm_arr[HandLandmark.WRIST, 1]))
        except (IndexError, TypeError):
            continue

    if len(wx_list) < 5:
        return None

    wx = np.array(wx_list, dtype=float)
    wy = np.array(wy_list, dtype=float)
    rms_spread = float(np.sqrt(np.var(wx) + np.var(wy)))

    if inter_mcp_span is not None and inter_mcp_span > 1e-6:
        return float(rms_spread / inter_mcp_span)
    return float(rms_spread)


def _proxy_angle_deg(lm_arr):
    """2D proxy: angle between Index MCP and Pinky MCP (degrees)."""
    dx = float(lm_arr[HandLandmark.PINKY_MCP, 0] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 0])
    dy = float(lm_arr[HandLandmark.PINKY_MCP, 1] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 1])
    return float(np.degrees(np.arctan2(dy, dx)))


def _build_wrist_z_signal(frames_dict, total_frames):
    """Extract per-frame wrist z coordinate as a 1D signal.

    During pronation-supination the wrist z (monocular depth estimate)
    oscillates at the PS frequency because the forearm rotation changes
    the depth of the wrist landmark.  While z is too noisy for amplitude
    computation (which uses only x,y), the oscillation provides a valuable
    confirmation channel for cycle boundaries.

    Returns
    -------
    np.ndarray of shape (total_frames,)
        Wrist z per frame; NaN where no landmark data is available.
    """
    wrist_z = np.full(total_frames, np.nan, dtype=float)
    for f, lm_arr in frames_dict.items():
        if f < total_frames and lm_arr.shape[1] >= 3:
            wrist_z[f] = float(lm_arr[HandLandmark.WRIST, 2])
    return wrist_z


# ============================================================
# High-level API
# ============================================================


def compute_kinematic_features(input_data, *, config=None):
    """Compute kinematic features from tracking data.

    Parameters
    ----------
    input_data : dict
        Must contain ``"frames"`` (dict mapping frame_idx -> (21, 3+) array),
        ``"total_frames"`` (int), and ``"fps"`` (float).
    config : dict or None
        Optional overrides for kinematic analyzer params. Supported keys:
        ``cutoff_hz``, ``highpass_hz``, ``prominence_deg``, ``filter_order``,
        ``max_movement_hz``, ``adaptive_prom_frac``, ``use_pca_angle``,
        ``use_parabolic_interp``.

    Returns
    -------
    dict or None
        Kinematic feature dict as returned by KinematicAnalyzer.extract_features,
        or None if extraction fails.
    """
    if config is None:
        config = {}

    frames_dict = input_data.get("frames", {})
    total_frames = input_data.get("total_frames", 0)
    fps = input_data.get("fps", 30.0)

    if not frames_dict or total_frames < 10:
        return None

    use_pca = config.get("use_pca_angle", True)
    if use_pca:
        raw_deg = _build_pca_angle_deg(frames_dict, total_frames, fps=fps)
    else:
        raw_deg = _build_unwrapped_angle_deg(
            frames_dict, total_frames, _knuckle_line_angle_rad_standalone, fps=fps
        )

    # Extract wrist z confirmation signal
    wrist_z = _build_wrist_z_signal(frames_dict, total_frames)

    # Compute inter-MCP span for distance normalisation
    inter_mcp_span = _compute_inter_mcp_span(frames_dict, total_frames)

    time_s = np.arange(total_frames, dtype=float) / float(fps)

    analyzer = KinematicAnalyzer(
        time_s,
        raw_deg,
        fps=fps,
        cutoff_hz=config.get("cutoff_hz", 2.5),
        filter_order=int(config.get("filter_order", 4)),
        highpass_hz=config.get("highpass_hz", 0.10),
        wrist_z=wrist_z,
        use_parabolic_interp=config.get("use_parabolic_interp", False),
    )

    metrics = analyzer.extract_features(
        prominence_deg=config.get("prominence_deg", 15.0),
        max_movement_hz=config.get("max_movement_hz", 2.0),
        adaptive_prom_frac=config.get("adaptive_prom_frac", 0.20),
        min_half_cycle_s=config.get("min_half_cycle_s", 0.20),
    )

    # Attach inter-MCP span
    if metrics is not None:
        metrics["inter_mcp_span"] = inter_mcp_span

    return metrics
