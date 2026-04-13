"""Shared constants for the analysis sub-package.

Data-only module: no imports from the analysis package, only stdlib + numpy.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Age-group binning
# ---------------------------------------------------------------------------
AGE_BINS = [0, 60, 70, 200]
AGE_LABELS = ["<60", "60-69", "70+"]

# ---------------------------------------------------------------------------
# Physiological frequency range for pronation-supination
# ---------------------------------------------------------------------------
PS_FREQ_RANGE = (0.3, 4.0)  # Hz

# ---------------------------------------------------------------------------
# Composite score component definitions
# ---------------------------------------------------------------------------
COMPOSITE_COMPONENTS = [
    ("Mean Amplitude", -1.0),
    ("Peak Velocity", -1.0),
    ("Mean Velocity", -1.0),
]

COMPOSITE_V2_COMPONENTS = [
    ("Peak Velocity", -1.0, 2.0),   # weight 2: strongest clinical signal
    ("Mean Amplitude", -1.0, 1.5),  # weight 1.5: core criterion
    ("Global Velocity", -1.0, 1.5), # weight 1.5: cycle-agnostic speed
    ("Rhythm (CV %)", +1.0, 1.0),   # weight 1: irregularity
    ("Amp Decrement %", +1.0, 1.0), # weight 1: sequence effect
    ("Pause Time Ratio", +1.0, 1.0),# weight 1: interruptions
    ("Arm Swing Index", +1.0, 0.5), # weight 0.5: posture
]

# ---------------------------------------------------------------------------
# Video quality factor column definitions
# ---------------------------------------------------------------------------
VQ_FACTOR_COLS = [
    # (csv_col,              display_label,                           description)
    (
        "VQ_inter_mcp_span_px",
        "Inter-MCP Span (px)",
        "Median index-to-pinky MCP pixel distance — anatomically grounded hand size; protocol-checkable recording quality indicator",
    ),
    (
        "VQ_hand_bbox_area_median_px",
        "Median Hand BBox Area (px\u00b2)",
        "Hand pixel size — larger = closer / higher resolution",
    ),
    (
        "VQ_sharpness_median",
        "Median Sharpness (Laplacian \u03c3\u00b2)",
        "Laplacian variance — higher = sharper, lower = motion blur",
    ),
    (
        "VQ_luminance_median",
        "Median Luminance (0-255)",
        "Brightness in hand ROI — very low or high = poor lighting",
    ),
    (
        "VQ_luminance_cv",
        "Luminance CV (frame-to-frame)",
        "Lighting stability — higher = more flickering / variation",
    ),
    (
        "VQ_saturation_frac_median",
        "Saturation Fraction",
        "Fraction of over/underexposed pixels in hand ROI",
    ),
    (
        "VQ_detection_rate",
        "Detection Rate (0-1)",
        "Fraction of frames with successful hand detection",
    ),
    ("VQ_gap_fraction", "Gap Fraction (0-1)", "Fraction of analysis window lost to detection gaps"),
    ("VQ_longest_gap_s", "Longest Detection Gap (s)", "Duration of worst occlusion event"),
    ("VQ_n_gaps", "Number of Detection Gaps", "Count of separate occlusion episodes"),
    (
        "VQ_sharpness_q10",
        "P10 Sharpness (worst frames)",
        "10th percentile sharpness — captures blurriest frames",
    ),
]

# Key kinematic features to correlate with quality factors
KIN_COLS_FOR_VQ = [
    "Signal Quality",
    "Mean Amplitude",
    "Peak Velocity",
    "Mean Velocity",
    "Rhythm (CV %)",
    "Amplitude CV",
    "Total Cycles",
    "Quality Cycles",
]

# ---------------------------------------------------------------------------
# Clinical domain organisation (Zarrat Ehsan et al. 2024 / Bologna 2023)
# ---------------------------------------------------------------------------
FEATURE_DOMAINS = {
    "Hypokinesia": [
        "Mean Amplitude",
        "Hilbert Amplitude",
        "Integral Amplitude",
    ],
    "Bradykinesia": [
        "Mean Frequency",
        "Avg Cycle Duration",
    ],
    "Hypo- & Bradykinesia (Speed)": [
        "Peak Velocity",
        "Mean Velocity",
        "Global Velocity",
    ],
    "Sequence Effect": [
        "Norm Decrement Slope",
        "Raw Amp Slope",
        "Norm TI Slope",
        "Raw Cycle Duration Slope",
        "Norm Velocity Decrement Slope",
        "Raw Velocity Slope",
        "Raw Speed Slope",
        "Amp Decrement Onset",
        "Velocity Decrement Onset",
        "Amp Decrement %",
        "Velocity Decrement %",
    ],
    "Hesitation-Halts": [
        "Amplitude CV",
        "Rhythm (CV %)",
        "Cycle Duration CV",
        "Peak Velocity CV",
        "Mean Velocity CV",
        "Num Hesitations",
        "Num Arrests",
        "Num Interruptions (2x)",
        "Max Pause Duration (s)",
        "Pause Time Ratio",
        "Sample Entropy",
        "Amp-Vel Coupling",
    ],
    "Task-Specific (PS)": [
        "Arm Swing Index",
    ],
    "Composite": [
        "Composite Score",
        "Clinical Composite v2",
        "Movement Vigor",
        "Amp-Vel Product",
        "Fatigue Index",
        "Irregularity Burden",
    ],
}

# ---------------------------------------------------------------------------
# Kinematic feature definitions: (column, label, description, can_normalize)
# ---------------------------------------------------------------------------
KINEMATIC_FEATURES = [
    # --- Hypokinesia ---
    (
        "Mean Amplitude",
        "Mean Amplitude (degrees)",
        "Mean amplitude of PS movement [Hypokinesia]",
        False,
    ),
    (
        "Hilbert Amplitude",
        "Hilbert Envelope Amplitude (degrees)",
        "Per-cycle median of the Hilbert analytic-signal envelope; "
        "less sensitive to peak clipping from tracking noise [Hypokinesia]",
        False,
    ),
    (
        "Integral Amplitude",
        "Integral Amplitude (degrees)",
        "Half-cycle velocity-integral amplitude; uses entire velocity "
        "profile instead of discrete peak/trough values [Hypokinesia]",
        False,
    ),
    # --- Bradykinesia ---
    ("Mean Frequency", "Mean Frequency (Hz)", "Average frequency [Bradykinesia]", False),
    (
        "Avg Cycle Duration",
        "Average Cycle Duration (s)",
        "Mean full-cycle (peak-to-peak) duration; inverse of frequency \u2014 paper-aligned [Bradykinesia]",
        False,
    ),
    # --- Hypo- & Bradykinesia (Speed) ---
    (
        "Peak Velocity",
        "Peak Angular Velocity (deg/s)",
        "Mean p95 peak angular velocity per cycle (CMS; paper-aligned) [Hypo- & Bradykinesia]",
        False,
    ),
    (
        "Mean Velocity",
        "Mean Angular Velocity (deg/s)",
        "Mean angular velocity per cycle (CAS; paper-aligned) [Hypo- & Bradykinesia]",
        False,
    ),
    (
        "Global Velocity",
        "Global Angular Velocity (deg/s)",
        "Mean |d\u03b8/dt| over the entire keypoint-detected period \u2014 cycle-agnostic [Hypo- & Bradykinesia]",
        False,
    ),
    # --- Sequence Effect ---
    (
        "Norm Decrement Slope",
        "Normalised Amplitude Decrement (%/s)",
        "Dimensionless amplitude fatigue (normalised+clipped) [Sequence Effect]",
        False,
    ),
    (
        "Raw Amp Slope",
        "Raw Amplitude Slope (deg/s)",
        "Unnormalised linear regression slope of amplitude over time \u2014 paper-aligned [Sequence Effect]",
        False,
    ),
    (
        "Amp Decrement Onset",
        "Amplitude Decrement Onset (frac)",
        "Fractional cycle position where amplitude decrement begins (0=immediate, 1=end/none) [Sequence Effect]",
        False,
    ),
    (
        "Amp Decrement %",
        "Amplitude Decrement (%)",
        "Percent loss in amplitude from early to late task performance [Sequence Effect]",
        False,
    ),
    (
        "Norm TI Slope",
        "Normalised Timing Decrement (%/cyc)",
        "Dimensionless rhythmic fatigue (timing, normalised+clipped) [Sequence Effect]",
        False,
    ),
    (
        "Raw Cycle Duration Slope",
        "Raw Cycle Duration Slope (s/cyc)",
        "Unnormalised linear regression slope of cycle duration over cycle index \u2014 paper-aligned [Sequence Effect]",
        False,
    ),
    (
        "Norm Velocity Decrement Slope",
        "Normalised Velocity Decrement (%/cyc)",
        "Normalised slope of per-cycle peak velocity over cycle index [Sequence Effect]",
        False,
    ),
    (
        "Raw Velocity Slope",
        "Raw Peak-Velocity Slope (deg/s/cyc)",
        "Unnormalised slope of per-cycle peak velocity over cycle index [Sequence Effect]",
        False,
    ),
    (
        "Raw Speed Slope",
        "Raw Mean-Velocity Slope (deg/s/cyc)",
        "Unnormalised slope of per-cycle mean velocity (CAS) \u2014 paper-aligned speed slope [Sequence Effect]",
        False,
    ),
    (
        "Velocity Decrement Onset",
        "Velocity Decrement Onset (frac)",
        "Fractional cycle position where sustained peak-velocity decrement begins [Sequence Effect]",
        False,
    ),
    (
        "Velocity Decrement %",
        "Velocity Decrement (%)",
        "Percent loss in peak velocity from early to late task performance [Sequence Effect]",
        False,
    ),
    # --- Hesitation-Halts ---
    (
        "Amplitude CV",
        "Amplitude Variability (CV %)",
        "CV of cycle amplitude [Hesitation-Halts]",
        False,
    ),
    (
        "Rhythm (CV %)",
        "Rhythm Variability (CV %)",
        "CV of half-cycle timing intervals [Hesitation-Halts]",
        False,
    ),
    (
        "Cycle Duration CV",
        "Cycle Duration CV (%)",
        "CV of full-cycle (peak-to-peak) durations \u2014 paper-aligned [Hesitation-Halts]",
        False,
    ),
    (
        "Peak Velocity CV",
        "Peak Velocity Variability (CV %)",
        "CV of per-cycle peak angular velocity [Hesitation-Halts]",
        False,
    ),
    (
        "Mean Velocity CV",
        "Mean Velocity Variability (CV %)",
        "CV of per-cycle mean angular velocity [Hesitation-Halts]",
        False,
    ),
    (
        "Num Hesitations",
        "Hesitation Count",
        "Moderately prolonged half-cycles (PS-specific two-tier) [Hesitation-Halts]",
        False,
    ),
    (
        "Num Arrests",
        "Arrest Count",
        "Half-cycles exceeding 1.5x median interval (PS-specific) [Hesitation-Halts]",
        False,
    ),
    (
        "Num Interruptions (2x)",
        "Interruption Count (2\u00d7 median)",
        "Full-cycle durations > 2\u00d7 median \u2014 paper-aligned interruption count [Hesitation-Halts]",
        False,
    ),
    (
        "Max Pause Duration (s)",
        "Longest Pause Duration (s)",
        "Longest hesitation or arrest duration across the task [Hesitation-Halts]",
        False,
    ),
    (
        "Pause Time Ratio",
        "Pause Time Ratio",
        "Fraction of half-cycle time spent in hesitation/arrest intervals [Hesitation-Halts]",
        False,
    ),
    (
        "Sample Entropy",
        "Sample Entropy (SampEn)",
        "Signal complexity/predictability; low = regular (healthy), high = irregular (PD) [Hesitation-Halts]",
        False,
    ),
    (
        "Amp-Vel Coupling",
        "Amplitude-Velocity Coupling (r)",
        "Pearson r between per-cycle amplitude and peak velocity; PD disrupts coupling [Hesitation-Halts]",
        False,
    ),
    # --- Task-Specific (PS) ---
    (
        "Arm Swing Index",
        "Arm Swing Index (RMS wrist drift / span)",
        "2-D RMS spread of wrist image position normalised by inter-MCP span [Task-Specific PS]",
        False,
    ),
    # --- Composite scores ---
    (
        "Composite Score",
        "Composite Severity Score (Z)",
        "Equal-weight Z-score composite of \u2212Z(MeanAmp) \u2212 Z(PeakVel) \u2212 Z(MeanVel) [Composite]",
        False,
    ),
    # --- Interaction features ---
    (
        "Movement Vigor",
        "Movement Vigor (amp \u00d7 peakvel / 100)",
        "Amplitude \u00d7 Peak Velocity; overall movement energy [Composite]",
        False,
    ),
    (
        "Amp-Vel Product",
        "Amp-Vel Product (amp \u00d7 meanvel / 100)",
        "Amplitude \u00d7 Mean Velocity [Composite]",
        False,
    ),
    (
        "Fatigue Index",
        "Fatigue Index (%)",
        "Combined amplitude + velocity decrement [Composite]",
        False,
    ),
    (
        "Irregularity Burden",
        "Irregularity Burden",
        "CV weighted by pause burden [Composite]",
        False,
    ),
    (
        "Clinical Composite v2",
        "Clinical Composite v2 (weighted Z)",
        "Weighted Z-score combining velocity, amplitude, regularity, decrement, and pauses [Composite]",
        False,
    ),
]

# ---------------------------------------------------------------------------
# Expected direction of change with clinical severity
# ---------------------------------------------------------------------------
EXPECTED_DIRECTIONS = {
    # --- Hypokinesia ---
    "Mean Amplitude": -1,
    "Hilbert Amplitude": -1,
    "Integral Amplitude": -1,
    # --- Bradykinesia ---
    "Mean Frequency": -1,
    "Avg Cycle Duration": +1,
    # --- Hypo- & Bradykinesia (Speed) ---
    "Peak Velocity": -1,
    "Mean Velocity": -1,
    "Global Velocity": -1,
    # --- Sequence Effect ---
    "Norm Decrement Slope": -1,
    "Raw Amp Slope": -1,
    "Amp Decrement Onset": -1,
    "Amp Decrement %": +1,
    "Norm TI Slope": +1,
    "Raw Cycle Duration Slope": +1,
    "Norm Velocity Decrement Slope": -1,
    "Raw Velocity Slope": -1,
    "Raw Speed Slope": -1,
    "Velocity Decrement Onset": -1,
    "Velocity Decrement %": +1,
    # --- Hesitation-Halts ---
    "Amplitude CV": +1,
    "Rhythm (CV %)": +1,
    "Cycle Duration CV": +1,
    "Peak Velocity CV": +1,
    "Mean Velocity CV": +1,
    "Num Hesitations": +1,
    "Num Arrests": +1,
    "Num Interruptions (2x)": +1,
    "Max Pause Duration (s)": +1,
    "Pause Time Ratio": +1,
    "Sample Entropy": +1,
    "Amp-Vel Coupling": -1,
    # --- Task-Specific (PS) ---
    "Arm Swing Index": +1,
    # --- Composite scores ---
    "Composite Score": +1,
    # --- Interaction features ---
    "Movement Vigor": -1,
    "Amp-Vel Product": -1,
    "Fatigue Index": +1,
    "Irregularity Burden": +1,
    "Clinical Composite v2": +1,
}
