"""CLI entry point for kinematic analysis.

Creates box plots of kinematic features grouped by MDS-UPDRS score,
matching video IDs to patient IDs via id2vid.csv and merging with
clinical scores from final_scores_summary.csv.

All analysis logic lives in ``ps_kinematics.analysis``; this script
is a thin CLI wrapper around the public API functions.
"""

from ps_kinematics.analysis import (
    compute_clinical_alignment_score,
    compute_cycle_detection_accuracy,
    compute_signal_quality_score,
    create_kinematic_boxplots_by_score,
)
from ps_kinematics.analysis._constants import EXPECTED_DIRECTIONS
from ps_kinematics.analysis._filtering import (
    load_recording_angle_labels as _load_recording_angle_labels,
    normalize_recording_angle_filter as _normalize_recording_angle_filter,
)
from ps_kinematics.io import normalize_video_path_for_matching as _normalize_video_path_for_matching

if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="Kinematic analysis entry point.")
    _ap.add_argument("--kinematics-csv", required=True, help="Path to tracking_logs.csv")
    _ap.add_argument("--score-csv", required=True, help="Path to final_scores_summary.csv")
    _ap.add_argument("--id2vid-csv", required=True, help="Path to id2vid.csv")
    _ap.add_argument("--output-dir", required=True, help="Directory for analysis output")
    _ap.add_argument("--recording-angle-csv", default="", help="Path to angle_annotations.csv")
    _ap.add_argument("--video-quality-csv", default="", help="Path to video_quality_labels.csv")
    _ap.add_argument("--age-gender-csv", default="", help="Path to age_gender.csv")
    _ap.add_argument("--validation-csv", default="", help="Path to validation_peaks.xlsx")
    _ap.add_argument("--score-column", default="ProS", help="Score column name")
    _args = _ap.parse_args()

    # Toggle expensive steps during exploratory runs (defaults preserve behavior).
    RUN_MAIN_ANALYSIS = True
    RUN_CYCLE_VALIDATION = True

    RUN_PER_VIDEO_DIAGNOSTIC_PLOTS = True
    RUN_VIDEO_QUALITY_ANALYSIS = True
    RUN_RECORDING_ANGLE_ANALYSIS = True
    RUN_LONGITUDINAL_REPORT = True
    RUN_MEDICATION_EFFECT_REPORT = True
    RUN_AGE_GENDER_ADJUSTMENT = False

    SEGMENTED_ONLY = False
    NON_SEGMENTED = False

    RECORDING_ANGLE_CSV_PATH = _args.recording_angle_csv
    SELECTED_RECORDING_ANGLES = ["Front"]
    VIDEO_QUALITY_LABELS_CSV_PATH = _args.video_quality_csv
    VIDEO_QUALITY_THRESHOLD = 1
    AGE_GENDER_CSV_PATH = _args.age_gender_csv

    if RUN_MAIN_ANALYSIS:
        result = create_kinematic_boxplots_by_score(
            kinematics_csv_path=_args.kinematics_csv,
            score_csv_path=_args.score_csv,
            id2vid_csv_path=_args.id2vid_csv,
            score_column=_args.score_column,
            output_dir=_args.output_dir,
            save_plots=True,
            show_plots=False,
            normalize_by_cycles=True,
            extreme_iqr_multiplier=3.0,
            age_gender_csv_path=AGE_GENDER_CSV_PATH if RUN_AGE_GENDER_ADJUSTMENT else None,
            signal_quality_threshold=0.0,
            signal_quality_sub_thresholds={
                "SQ_spectral_concentration": 0,
                "SQ_autocorr_strength": 0,
                "SQ_cycle_regularity": 0.0,
                "SQ_signal_coverage": 0,
                "SQ_freq_plausibility": 0,
                "SQ_cycle_yield": 0,
            },
            min_cycles=0,
            min_quality_cycles=0,
            min_inter_mcp_span_px=0,
            min_detection_rate=0.0,
            recording_angle_csv_path=RECORDING_ANGLE_CSV_PATH,
            selected_recording_angles=SELECTED_RECORDING_ANGLES,
            video_quality_labels_csv_path=VIDEO_QUALITY_LABELS_CSV_PATH,
            video_quality_threshold=VIDEO_QUALITY_THRESHOLD,
            run_per_video_diagnostic_plots=RUN_PER_VIDEO_DIAGNOSTIC_PLOTS,
            run_video_quality_analysis=RUN_VIDEO_QUALITY_ANALYSIS,
            run_recording_angle_analysis=RUN_RECORDING_ANGLE_ANALYSIS,
            run_longitudinal_report=RUN_LONGITUDINAL_REPORT,
            run_medicine_effect_report=RUN_MEDICATION_EFFECT_REPORT,
            run_age_gender_adjustment=RUN_AGE_GENDER_ADJUSTMENT,
            segmented_only=SEGMENTED_ONLY,
            non_segmented=NON_SEGMENTED,
        )

    if RUN_CYCLE_VALIDATION:
        accuracy = compute_cycle_detection_accuracy(
            tracking_csv_path=_args.kinematics_csv,
            validation_csv_path=_args.validation_csv,
            recording_angle_csv_path=RECORDING_ANGLE_CSV_PATH,
            selected_recording_angles=SELECTED_RECORDING_ANGLES,
            video_quality_labels_csv_path=VIDEO_QUALITY_LABELS_CSV_PATH,
            video_quality_threshold=VIDEO_QUALITY_THRESHOLD,
            segmented_only=SEGMENTED_ONLY,
            non_segmented=NON_SEGMENTED,
        )
