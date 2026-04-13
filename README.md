# video-ps-kinematics

Quantitative kinematic feature extraction from consumer-camera video recordings of the **hand pronation-supination (PS) motor task** (MDS-UPDRS item 3.6) for Parkinson's disease and PSP research.

---

## Repository layout

```
video-ps-kinematics/
├── src/ps_kinematics/           # Main package (src-layout)
│   ├── processor.py             #   Pipeline orchestrator, multiprocessing pool
│   ├── workers.py               #   Per-video multiprocessing workers
│   ├── core.py                  #   Per-frame landmark utilities, ROI re-detection
│   ├── tracker.py               #   MultiHandOfflineTracker — spatial track matching
│   ├── kinematics.py            #   KinematicAnalyzer + feature computation
│   ├── utils.py                 #   Tuning constants + PipelineConfig
│   ├── io.py                    #   Path parsing, CSV loaders
│   ├── plotting.py              #   OpenCV time-series overlay renderer
│   ├── video_quality.py         #   Per-video quality factor computation
│   ├── gpu_manager.py           #   GPU concurrency semaphore
│   ├── yolo_tracker.py          #   YOLO-Pose hand tracking
│   ├── config/                  #   YAML config loader
│   ├── refinement/              #   Optional keypoint-refinement backends
│   │   ├── rtmpose.py           #     RTMPose-Hand (MMPose, GPU)
│   │   ├── openpose.py          #     OpenPose-Hand (OpenCV DNN)
│   │   ├── superres.py          #     Real-ESRGAN hand-ROI super-resolution
│   │   └── yolo.py              #     YOLO-Pose hand landmark refinement
│   └── analysis/                #   Statistical analysis & visualization
│       ├── orchestrator.py      #     Kinematic boxplots by clinical score
│       ├── clinical_alignment.py
│       ├── cycle_detection_accuracy.py
│       ├── pca_varimax.py
│       └── ...                  #     medication, longitudinal, recording_angle, etc.
├── configs/
│   ├── config.example.yaml      #   Annotated template — copy to config.yaml
│   ├── config_tulip.example.yaml#   TULIP dataset template
│   ├── tuning_25fps.yaml        #   Pre-optimised tuning profile (25 fps)
│   └── tuning_enhanced_50fps.yaml
├── scripts/
│   ├── run_pipeline.py          #   Main pipeline entry point
│   ├── analyze.py               #   Box-plot creation + clinical alignment
│   ├── classify.py              #   ML classification (LightGBM, SVM, LogReg)
│   ├── classify_nn.py           #   Neural network classification (CORAL, MLP)
│   ├── preprocess_videos.py     #   Offline video enhancement
│   ├── prepare_tulip.py         #   TULIP dataset preparation
│   ├── prepare_yolo_pseudolabels.py  # YOLO pseudo-label generation
│   └── validate_yolo_finetuned.py    # YOLO fine-tuning validation
├── pyproject.toml
├── LICENSE
└── CITATION.cff
```

---

## Pipeline overview

```
Video CSV
    │
    ▼
[Optional] scripts/preprocess_videos.py
    ├── CLAHE contrast enhancement
    ├── Denoising (fastNlMeans)
    ├── Sharpening (unsharp mask)
    ├── ESRGAN super-resolution (GPU)
    └── Frame interpolation: RIFE / HiFI / GIMM-VFI
    │
    ▼
HandLandmarkProcessor (processor.py)
    ├── Parallel video workers (ProcessPoolExecutor)
    ├── MediaPipe Hand Landmarker → 21 3D landmarks per frame
    ├── Optional keypoint refinement:
    │   ├── YOLO-Pose  (USE_YOLO_HAND, GPU)
    │   ├── RTMPose    (USE_RTMPOSE, GPU)
    │   └── OpenPose   (USE_OPENPOSE, CPU/GPU)
    ├── Optional hand-ROI super-resolution (USE_SUPERRES, GPU)
    ├── MultiHandOfflineTracker → spatial track matching, gap filling
    └── KinematicAnalyzer
            ├── Polynomial drift removal
            ├── Butterworth bandpass filter
            ├── Zero-crossing half-cycle detection
            ├── Peak/valley detection + wrist-Z confirmation
            └── Cycle-level feature extraction
    │
    ▼
Outputs
    ├── Per-video kinematic CSVs
    ├── Optional plot-overlay MP4 videos
    └── Aggregate tracking_logs.csv
    │
    ▼
scripts/analyze.py → box-plot figures by MDS-UPDRS ProS score
```

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Configure
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml with your paths

# 3. Run pipeline
python scripts/run_pipeline.py

# 4. Analyse results
python scripts/analyze.py \
    --kinematics-csv path/to/tracking_logs.csv \
    --score-csv path/to/scores.csv \
    --id2vid-csv path/to/id2vid.csv \
    --output-dir results/
```

---

## Installation

```bash
# Core pipeline (MediaPipe, OpenCV, NumPy, pandas, scipy, matplotlib):
pip install -e .

# With GPU refinement backends (YOLO, RTMPose, Real-ESRGAN):
pip install -e ".[refinement]"

# With ML / classification extras (torch, LightGBM, Optuna):
pip install -e ".[ml,plots]"

# Development (ruff, black):
pip install -e ".[dev]"
```

---

## Configuration

Copy the example config and fill in your paths:

```bash
cp configs/config.example.yaml configs/config.yaml
```

```yaml
paths:
  vid_score_path: ~/data/PS_video_path_diag.csv
  score_csv_path: ~/data/final_scores_summary.csv
  hand_path:      ~/models/hand_landmarker.task
  log_csv_path:   ~/results/tracking_logs.csv
  save_dir:       ~/results/

tuning_profile: tuning_25fps.yaml
```

Override at runtime:

```bash
python scripts/run_pipeline.py --config path/to/my_config.yaml
```

---

## Input file formats

### `vid_score_path` — video list CSV

| Column | Required | Description |
|---|---|---|
| `video_path` | yes | Absolute or relative path to each video file |
| `visit` | no | Visit number (integer) |

### `score_csv_path` — clinical scores CSV

| Column | Required | Description |
|---|---|---|
| `ids` | yes | Patient identifier (string) |
| `visit` | no | Visit number (integer) |
| `medication_state` | no | `"On"` or `"Off"` |
| `hand` | no | `"Left"` or `"Right"` |
| `<score_column>` | no | Clinical score (integer); column name set by `score_column` in config (default `ProS`) |

### `id2vid_csv_path` — patient-to-video ID mapping

Headerless CSV with two columns:

| Column | Description |
|---|---|
| 1 | Patient ID (string) |
| 2 | Python-style tuple or list of video IDs, e.g. `('VID001', 'VID002')` |

### `video_quality_labels_csv_path` — manual quality labels (optional)

| Column | Required | Description |
|---|---|---|
| `video_path` | yes | Path to the video file |
| `quality_label` | yes | Integer: 1 (best), 2, or 3 (worst) |

---

## Running the pipeline

```bash
python scripts/run_pipeline.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--n N` | all | Cap number of videos |
| `--seed N` | 42 | Random seed for subset selection |
| `--workers N` | auto | Parallel workers |
| `--retry-failed [LOG]` | — | Re-run only TIMEOUT/ERROR/CRASH videos |
| `--yolo-pd-finetune` | off | Run YOLO fine-tuning pipeline |
| `--enable-cudnn` | off | Enable cuDNN |
| `--node-rank N` | 0 | Node index for multi-node processing |
| `--num-nodes N` | 1 | Total nodes |

### Multi-node processing

Videos are striped across nodes for balanced load:

```bash
python scripts/run_pipeline.py \
    --node-rank $SLURM_ARRAY_TASK_ID \
    --num-nodes $SLURM_ARRAY_TASK_COUNT \
    --workers-list "120,94,30,33"
```

Each node writes a separate `tracking_logs_rank<k>.csv`. Merge per-node CSVs afterwards with `pandas.concat` or similar.

---

## Keypoint refinement backends

| Backend | Flag | Hardware | Notes |
|---|---|---|---|
| **YOLO-Pose** | `USE_YOLO_HAND=True` | GPU | Ultralytics; auto-trains if model absent |
| **RTMPose-Hand** | `USE_RTMPOSE=True` | GPU | MMPose-based |
| **OpenPose-Hand** | `USE_OPENPOSE=True` | CPU or GPU | OpenCV DNN |
| **Real-ESRGAN** | `USE_SUPERRES=True` | GPU | Upscales hand ROI before inference |

---

## YOLO fine-tuning

```bash
python scripts/run_pipeline.py --yolo-pd-finetune
```

1. Standard MediaPipe run on all videos
2. Pseudo-label extraction from top-quality frames
3. YOLO-Pose fine-tuning on pseudo-labels
4. Re-run with `USE_YOLO_HAND=True`

Validate with `scripts/validate_yolo_finetuned.py`.

---

## Video preprocessing

```bash
python scripts/preprocess_videos.py --csv input.csv --output-dir ./enhanced \
    --clahe --denoise --sharpen --esrgan --rife
```

Enhancement stages: CLAHE, denoising, sharpening, ESRGAN super-resolution, frame interpolation (RIFE / HiFI / GIMM-VFI).

---

## TULIP dataset

Convert the TULIP dataset to pipeline-compatible CSVs:

```bash
python scripts/prepare_tulip.py \
    --tulip-root /path/to/tulip-dataset \
    --output-dir /path/to/tulip-dataset/pipeline_csvs

python scripts/run_pipeline.py --config configs/config_tulip.yaml
```

See `configs/config_tulip.example.yaml` for the configuration template.

---

## Scripts

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Main pipeline — landmark inference, tracking, kinematic features |
| `analyze.py` | Box plots by MDS-UPDRS score; medication and longitudinal analysis |
| `classify.py` | ML classification (LightGBM, SVM, LogReg) with LOSO-CV |
| `classify_nn.py` | Neural network classification (CORAL ordinal regression, ResidualMLP) |
| `preprocess_videos.py` | Offline video enhancement (CLAHE, denoise, ESRGAN, RIFE/GIMM-VFI) |
| `prepare_tulip.py` | TULIP dataset preparation |
| `prepare_yolo_pseudolabels.py` | YOLO pseudo-label extraction from MediaPipe detections |
| `validate_yolo_finetuned.py` | Fine-tuned YOLO validation against MediaPipe baseline |

---

## Extracted kinematic features

### Amplitude

| Feature | Description |
|---|---|
| `avg_amp` | Mean full-cycle amplitude (degrees) |
| `amp_cv` | Amplitude coefficient of variation (%) |
| `norm_decrement_slope` | Normalised amplitude decrement over time (%/s) |
| `amp_decrement_onset` | Fractional position where amplitude decrement begins |
| `amp_decrement_pct` | Percent amplitude loss from early to late task |

### Frequency & Timing

| Feature | Description |
|---|---|
| `freq` | Mean PS frequency (Hz) |
| `cv` | Inter-half-cycle timing CV (%) |
| `norm_ti_slope` | Normalised slope of cycle interval (%/cycle) |
| `num_hesitations` | Count of prolonged half-cycles |
| `num_arrests` | Count of intervals exceeding 1.5x median |
| `max_pause_duration_s` | Longest hesitation/arrest (seconds) |
| `pause_time_ratio` | Fraction of time in hesitation/arrest |

### Velocity

| Feature | Description |
|---|---|
| `peak_velocity` | Mean per-cycle peak angular velocity (deg/s) |
| `mean_velocity` | Mean per-cycle mean angular velocity (deg/s) |
| `peak_velocity_cv` | Peak velocity variability (%) |
| `mean_velocity_cv` | Mean velocity variability (%) |
| `global_velocity` | Mean \|dtheta/dt\| over entire PS period |
| `norm_velocity_decrement_slope` | Velocity decrement over cycle index (%/cycle) |

### Video quality metrics

| Metric | Description |
|---|---|
| `hand_bbox_area_median_px` | Median hand bounding-box area (pixels) |
| `sharpness_median` | Median Laplacian variance in hand ROI |
| `luminance_median` | Median hand-ROI luminance |
| `detection_rate` | Fraction of frames with hand detected |
| `longest_gap_s` | Longest detection gap (seconds) |

---

## Key tuning constants

All constants can be overridden via the tuning profile YAML.

| Group | Constants |
|---|---|
| **Tracking** | `TRACK_MATCH_THRESH`, `MAX_GAP`, `MAX_JUMP_PER_FRAME`, `FILL_MAX_DIST` |
| **Landmark quality** | `VISIBILITY_THRESHOLD`, `ADAPTIVE_VISIBILITY`, `LM_OUTLIER_WINDOW` |
| **Signal processing** | `cutoff_hz`, `highpass_hz`, `filter_order`, `DETREND_POLY_ORDER` |
| **Cycle detection** | `ZCR_DC_WINDOW_CYCLES`, `ARREST_MIN_DURATION_S`, `CYCLE_NAN_THRESHOLD` |
| **GPU / refinement** | `GPU_CONCURRENCY`, `USE_SUPERRES`, `USE_RTMPOSE`, `USE_YOLO_HAND` |

---

## Output files

| File | Description |
|---|---|
| `<save_dir>/<video_id>_kinematics.csv` | Per-video kinematic features |
| `<save_dir>/<video_id>_plot.mp4` | Two-panel overlay video |
| `<log_csv_path>` | Aggregate CSV: one row per video |
| `<output_dir>/boxplot_<feature>.png` | Per-feature box plots by score |

---

## Design decisions

- **PCA angle**: PCA on 2D knuckle-line direction vectors; excludes z to avoid monocular depth noise.
- **Zero-crossing cycle detection**: Half-cycles bounded by zero-crossings; immune to amplitude decrement.
- **Wrist-Z confirmation**: Angle-based peaks require a corresponding wrist z extremum.
- **Polynomial detrending**: Pre-filter quadratic detrend removes slow drift.
- **Adaptive visibility**: Per-video threshold from MCP-visibility distribution.
- **One-Euro filter**: Adaptive low-pass for landmark smoothing.
- **PS-activity trimming**: Retains only frames with active task performance.
- **CLAHE fill-pass**: Missing frames contrast-enhanced before re-detection.
- **Cycle NaN gating**: Excludes cycles with >40% interpolated frames.
- **GPU concurrency**: Process-level semaphore prevents GPU OOM.

---

## Citation

```bibtex
@software{aljalab_video_ps_kinematics,
  author  = {Aljalab, Mohamad Naseb},
  title   = {video-ps-kinematics},
  url     = {https://github.com/mn-j/video-ps-kinematics},
  license = {MIT}
}
```

See `CITATION.cff` for the full citation metadata.

---

## License

MIT — see [LICENSE](LICENSE).
