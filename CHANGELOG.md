# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-04-10

### Changed
- Migrated package to `src/` layout (`src/ps_kinematics/`).
- Renamed package from `mediapipe_analysis` to `ps_kinematics`.
- Externalized pipeline configuration from Python dicts to YAML files under `configs/`.
- Moved `run_pipeline.py` into `scripts/` directory.
- Grouped refinement backends (YOLO, RTMPose, OpenPose, Real-ESRGAN) into
  `ps_kinematics.refinement` subpackage.
- Converted all intra-package imports to relative imports.
- Added `pyproject.toml` with full metadata, optional dependency extras, and
  tool configuration (ruff, black, pytest).

### Removed
- Removed `sys.path` manipulation hacks from all scripts.
- Removed `requirements.in` hard-pin duplication (now managed via `pyproject.toml`).

### Added
- `configs/config.example.yaml` with annotated pipeline configuration template.
- `configs/tuning_25fps.yaml` and `configs/tuning_enhanced_50fps.yaml` tuning profiles.
- Editable install support (`pip install -e .`).
- LICENSE, CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md.
