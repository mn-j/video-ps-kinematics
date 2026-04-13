"""ps_kinematics.analysis — statistical analysis and visualization pipeline."""

from .clinical_alignment import compute_clinical_alignment_score
from .cycle_detection_accuracy import compute_cycle_detection_accuracy
from .orchestrator import create_kinematic_boxplots_by_score
from .pca_varimax import run_pca_varimax_analysis
from .signal_quality_scoring import compute_signal_quality_score

__all__ = [
    "create_kinematic_boxplots_by_score",
    "run_pca_varimax_analysis",
    "compute_clinical_alignment_score",
    "compute_signal_quality_score",
    "compute_cycle_detection_accuracy",
]
