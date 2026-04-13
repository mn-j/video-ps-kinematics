"""
ps_kinematics.processor — HandLandmarkProcessor: orchestrates the full
video-processing pipeline (landmark detection, tracking, refinement,
kinematic analysis, and output generation).
"""

import json
import logging
import os
import re
import time

import numpy as np
import pandas as pd

try:
    import resource as _resource

    _RESOURCE_OK = True
except Exception:
    _resource = None
    _RESOURCE_OK = False
try:
    import psutil as _psutil

    _PSUTIL_OK = True
except ImportError:
    _psutil = None
    _PSUTIL_OK = False

import multiprocessing as mp_proc

try:
    import cv2

    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarksConnections

    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

    class HandLandmark:
        WRIST = 0
        INDEX_FINGER_MCP = 5
        MIDDLE_FINGER_MCP = 9
        RING_FINGER_MCP = 13
        PINKY_MCP = 17


from .core import (
    _compute_mcp_confidence_proxy,
    _extract_conf,
    _format_clinical_score_overlay_line,
    _lm_vis,
    _parse_visible_gpu_ids,
    _roi_zoom_detect,
    _runtime_gpu_config,
    ensure_track_visibility_channel,
    reject_landmark_outliers,
    smooth_track_landmarks,
    trim_track_to_ps_segment,
)
from .io import (
    canonicalize_video_id,
    load_clinical_scores_table,
    load_videoid_to_patientid_map,
    parse_hand_from_path,
    parse_ids_and_visit,
    parse_medication_state_from_path,
    resolve_video_clinical_score,
)
from .kinematics import (
    KinematicAnalyzer,
    _build_pca_angle_deg,
    _build_unwrapped_angle_deg,
    _build_wrist_z_signal,
    _compute_arm_swing_index,
    _compute_inter_mcp_span,
    _compute_inter_mcp_span_px,
)
from .plotting import render_two_plot_panel
from .tracker import MultiHandOfflineTracker
from .utils import (
    EXPORT_PLOT_VIDEO,
    FILL_ITERATIONS,
    MAX_PS_DURATION_S,
    MIN_PS_DURATION_S,
    NUM_WORKERS,
    PLOT_WIDTH_RATIO,
    ROI_REDETECT_PADDING,
    SUPERRES_HALF,
    SUPERRES_MODEL_NAME,
    SUPERRES_MODEL_PATH,
    SUPERRES_SCALE,
    TRIM_TO_PS_ACTIVITY,
    USE_CLAHE_ON_FILL,
    USE_PCA_ANGLE,
    _clahe_enhance,
    apply_tuning_overrides,
    series_to_json,
)
from .video_quality import compute_video_quality_metrics
from .workers import _robust_worker_entry

logger = logging.getLogger(__name__)


class HandLandmarkProcessor:
    MARGIN = 10
    FONT_SIZE = 0.7
    FONT_THICKNESS = 2
    KINEMATIC_LANDMARK_COLOR = (0, 165, 255)
    DEFAULT_LANDMARK_COLOR = (0, 255, 0)
    CONNECTION_COLOR = (0, 255, 0)

    def __init__(self, config, num_videos_to_process=20):
        self.config = config
        tuning = self.config.get("tuning_overrides", {})
        if tuning:
            apply_tuning_overrides(tuning)
        self._cutoff_hz = tuning.get("cutoff_hz", 2.5)
        self._highpass_hz = tuning.get("highpass_hz", 0.1)
        self._prominence_deg = tuning.get("prominence_deg", 10.0)
        self._filter_order = int(tuning.get("filter_order", 4))
        self._max_movement_hz = tuning.get("max_movement_hz", 3.0)
        self._adaptive_prom_frac = tuning.get("adaptive_prom_frac", 0.20)
        self._trim_to_ps = tuning.get("trim_to_ps_activity", TRIM_TO_PS_ACTIVITY)
        self._max_ps_duration_s = tuning.get("max_ps_duration_s", MAX_PS_DURATION_S)
        self._min_ps_duration_s = tuning.get("min_ps_duration_s", MIN_PS_DURATION_S)
        self.id2vid_csv_path = self.config.get("id2vid_csv_path", self.config.get("id2vid"))
        if not self.id2vid_csv_path:
            raise RuntimeError("Config must include 'id2vid_csv_path' (or 'id2vid').")
        self.video_to_patient = load_videoid_to_patientid_map(self.id2vid_csv_path)
        self.score_csv_path = self.config.get("score_csv_path")
        self.score_column = self.config.get("score_column", "ProS")
        self.scores_df = load_clinical_scores_table(self.score_csv_path, self.score_column)
        self.vid_score = pd.read_csv(self.config["vid_score_path"])
        self.hand_path = self.config["hand_path"]
        self.save_dir = self.config["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_csv_path = self.config.get(
            "log_csv_path", os.path.join(self.save_dir, "tracking_logs.csv")
        )
        self.num_videos_to_process = num_videos_to_process
        self.hand_to_track = None
        self.options_video, self.options_image = self._load_hand_landmarker_options_pair()

    def _load_hand_landmarker_options_pair(self):
        if not MEDIAPIPE_OK:
            raise RuntimeError("mediapipe is required for HandLandmarkProcessor")
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        base = None
        tried = []
        candidates = []
        try:
            candidates.append({"delegate": mp.tasks.BaseOptions.Delegate.GPU})
        except Exception:
            pass
        try:
            candidates.append({"delegate": mp.tasks.BaseOptions.Delegate.TFLITE_GPU})
        except Exception:
            pass
        candidates.append({"delegate": "GPU"})

        for cand in candidates:
            try:
                base = BaseOptions(model_asset_path=self.hand_path, **cand)
                tried.append(cand)
                break
            except Exception:
                tried.append(cand)

        if base is None:
            base = BaseOptions(model_asset_path=self.hand_path)

        tuning = self.config.get("tuning_overrides", {})
        common_kwargs = dict(
            base_options=base,
            num_hands=2,
            min_hand_detection_confidence=tuning.get("min_hand_detection_confidence", 0.35),
            min_hand_presence_confidence=tuning.get("min_hand_presence_confidence", 0.35),
            min_tracking_confidence=tuning.get("min_tracking_confidence", 0.35),
        )

        options_video = HandLandmarkerOptions(running_mode=VisionRunningMode.VIDEO, **common_kwargs)
        options_image = HandLandmarkerOptions(running_mode=VisionRunningMode.IMAGE, **common_kwargs)
        return options_video, options_image

    @staticmethod
    def _frame_timestamp_ms(frame_idx, fps):
        return int(frame_idx * 1000.0 / fps)

    @staticmethod
    def _lm_to_pixel_xy(lm_x, lm_y, width, height):
        return int(lm_x * width), int(lm_y * height)

    def _draw_top_right_text(self, image_bgr, lines):
        if isinstance(lines, str):
            lines = [lines]
        h, w = image_bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        sizes = [cv2.getTextSize(t, font, self.FONT_SIZE, self.FONT_THICKNESS)[0] for t in lines]
        max_tw = max((tw for tw, th in sizes), default=0)
        x = max(self.MARGIN, w - max_tw - self.MARGIN)
        y = self.MARGIN
        for i, text in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(text, font, self.FONT_SIZE, self.FONT_THICKNESS)
            yy = y + (i + 1) * (th + 6)
            cv2.putText(
                image_bgr,
                text,
                (x, yy),
                font,
                self.FONT_SIZE,
                (0, 0, 0),
                self.FONT_THICKNESS + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr,
                text,
                (x, yy),
                font,
                self.FONT_SIZE,
                (255, 255, 255),
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

    def _draw_hand_from_array(self, image_bgr, lm_arr):
        h, w = image_bgr.shape[:2]
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            x1, y1 = lm_arr[conn.start, 0], lm_arr[conn.start, 1]
            x2, y2 = lm_arr[conn.end, 0], lm_arr[conn.end, 1]
            p1 = self._lm_to_pixel_xy(x1, y1, w, h)
            p2 = self._lm_to_pixel_xy(x2, y2, w, h)
            cv2.line(image_bgr, p1, p2, self.CONNECTION_COLOR, 1)

        kinematic_lms = {
            HandLandmark.WRIST,
            HandLandmark.INDEX_FINGER_MCP,
            HandLandmark.MIDDLE_FINGER_MCP,
            HandLandmark.RING_FINGER_MCP,
            HandLandmark.PINKY_MCP,
        }
        for i in range(lm_arr.shape[0]):
            x, y = lm_arr[i, 0], lm_arr[i, 1]
            p = self._lm_to_pixel_xy(x, y, w, h)
            if i in kinematic_lms:
                color, radius = self.KINEMATIC_LANDMARK_COLOR, 3
            else:
                color, radius = self.DEFAULT_LANDMARK_COLOR, 2
            cv2.circle(image_bgr, p, radius, color, -1)

    @staticmethod
    def _wrist_xy(lm_arr):
        return float(lm_arr[HandLandmark.WRIST, 0]), float(lm_arr[HandLandmark.WRIST, 1])

    @staticmethod
    def _dist(a, b):
        return float((((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5))

    def _nearest_reference_wrist(self, track, frame_idx):
        if track is None or not track.get("frames"):
            return None
        frames = sorted(track["frames"].keys())
        best_f = min(frames, key=lambda f: abs(f - frame_idx))
        return self._wrist_xy(track["frames"][best_f])

    def _choose_detection_for_track_spatial(self, detections, ref_wrist, fill_max_dist=None):
        if not detections:
            return None
        if ref_wrist is None:
            if self.hand_to_track is not None:
                for lm_arr, handedness in detections:
                    if handedness and handedness[0].category_name == self.hand_to_track:
                        return (lm_arr, handedness)
            return None

        from . import utils as _utils

        _fill_max_dist = fill_max_dist if fill_max_dist is not None else _utils.FILL_MAX_DIST

        best = None
        best_score = float("inf")
        for lm_arr, handedness in detections:
            w = self._wrist_xy(lm_arr)
            d = self._dist(w, ref_wrist)
            if self.hand_to_track is not None and handedness:
                if handedness[0].category_name != self.hand_to_track:
                    d *= _utils.HANDEDNESS_PENALTY_MULT
            if d < best_score:
                best_score = d
                best = (lm_arr, handedness, d)

        if best is None:
            return None
        lm_arr, handedness, d = best
        if d > _fill_max_dist:
            return None
        return (lm_arr, handedness)

    @staticmethod
    def _compute_appearance_metrics(track, total_frames):
        if track is None or not track.get("frames"):
            return None, None, None, None, 0, 0, 0
        frames = sorted(track["frames"].keys())
        first, last = frames[0], frames[-1]
        window_len = last - first + 1
        detected_in_window = sum(1 for f in range(first, last + 1) if f in track["frames"])
        detected_total = len(track["frames"])
        adjusted = 100.0 * detected_in_window / window_len if window_len > 0 else None
        non_adjusted = 100.0 * detected_total / total_frames if total_frames > 0 else None
        return adjusted, non_adjusted, first, last, detected_in_window, window_len, detected_total

    @staticmethod
    def _avg_confidence_from_track(track, first, last):
        if track is None or first is None or last is None:
            return None, 0.0, 0
        vals = []
        for f in range(first, last + 1):
            if f in track.get("conf", {}):
                vals.append(track["conf"][f])
        if not vals:
            return None, 0.0, 0
        s = float(np.sum(vals))
        c = len(vals)
        return (s / c), s, c

    @staticmethod
    def _compute_proxy_angle_deg(lm_arr):
        dx = float(lm_arr[HandLandmark.PINKY_MCP, 0] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 0])
        dy = float(lm_arr[HandLandmark.PINKY_MCP, 1] - lm_arr[HandLandmark.INDEX_FINGER_MCP, 1])
        return float(np.degrees(np.arctan2(dy, dx)))

    @staticmethod
    def _knuckle_line_angle_rad(lm_arr):
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

    def _infer_tracks_offline(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        from . import utils as _utils

        tracker = MultiHandOfflineTracker(
            expected_label=self.hand_to_track,
            match_thresh=_utils.TRACK_MATCH_THRESH,
            max_gap=_utils.MAX_GAP,
            max_jump_per_frame=_utils.MAX_JUMP_PER_FRAME,
            fps=fps,
        )

        frame_idx = 0
        total_frames = 0
        with mp.tasks.vision.HandLandmarker.create_from_options(self.options_video) as landmarker:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                total_frames += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ts_ms = self._frame_timestamp_ms(frame_idx, fps)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                detections = []
                for hand_i, lm_list in enumerate(result.hand_landmarks):
                    lm_arr = np.array(
                        [[lm.x, lm.y, lm.z, _lm_vis(lm)] for lm in lm_list], dtype=np.float32
                    )
                    handedness = (
                        result.handedness[hand_i] if hand_i < len(result.handedness) else []
                    )
                    detections.append((lm_arr, handedness))

                tracker.associate_frame(frame_idx, detections)
                frame_idx += 1

        cap.release()
        main_track = tracker.choose_main_track()

        if main_track is None:
            return None, total_frames, fps, 0, 0, (frame_h, frame_w)

        detected_before_fill = len(main_track.get("frames", {}))

        missing = [f for f in range(total_frames) if f not in main_track["frames"]]
        if not missing:
            return main_track, total_frames, fps, 0, detected_before_fill, (frame_h, frame_w)

        fill_added = 0
        with mp.tasks.vision.HandLandmarker.create_from_options(
            self.options_image
        ) as landmarker_img:
            for _fill_iter in range(max(1, int(FILL_ITERATIONS))):
                missing = [f for f in range(total_frames) if f not in main_track["frames"]]
                if not missing:
                    break
                missing_set = set(missing)
                cap2 = cv2.VideoCapture(video_path)
                frame_idx = 0
                while True:
                    ok, frame_bgr = cap2.read()
                    if not ok:
                        break
                    if frame_idx in missing_set:
                        if USE_CLAHE_ON_FILL:
                            frame_bgr = _clahe_enhance(frame_bgr)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                        result = landmarker_img.detect(mp_image)

                        detections = []
                        for hand_i, lm_list in enumerate(result.hand_landmarks):
                            lm_arr = np.array(
                                [[lm.x, lm.y, lm.z, _lm_vis(lm)] for lm in lm_list],
                                dtype=np.float32,
                            )
                            handedness = (
                                result.handedness[hand_i] if hand_i < len(result.handedness) else []
                            )
                            detections.append((lm_arr, handedness))

                        if detections:
                            ref_wrist = self._nearest_reference_wrist(main_track, frame_idx)
                            from . import utils as _utils

                            _fps_scale = max(1.0, fps) / _utils.BASE_FPS

                            _scaled_fill = _utils.FILL_MAX_DIST / _fps_scale
                            chosen = self._choose_detection_for_track_spatial(
                                detections, ref_wrist, fill_max_dist=_scaled_fill
                            )
                            if chosen is not None:
                                lm_arr, handedness = chosen
                                if frame_idx not in main_track["frames"]:
                                    fill_added += 1
                                main_track["frames"][frame_idx] = lm_arr
                                main_track.setdefault("conf", {})[frame_idx] = _extract_conf(
                                    handedness, self.hand_to_track
                                )

                        if frame_idx in missing_set and frame_idx not in main_track["frames"]:
                            zoomed = _roi_zoom_detect(
                                frame_bgr, frame_idx, main_track, landmarker_img, self.hand_to_track
                            )
                            if zoomed is not None:
                                main_track["frames"][frame_idx] = zoomed
                                main_track.setdefault("conf", {})[frame_idx] = 0.3
                                fill_added += 1

                    frame_idx += 1
                cap2.release()

        return main_track, total_frames, fps, fill_added, detected_before_fill, (frame_h, frame_w)

    def _render_video_with_plots(
        self,
        video_path,
        save_vid_path,
        track,
        overlay_lines,
        time_s,
        filtered_deg,
        metrics,
        plot_width_ratio=PLOT_WIDTH_RATIO,
    ):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Unable to read video: {video_path}")

        h, w = first_frame.shape[:2]
        plot_w = int(w * float(plot_width_ratio))
        out_w = w + plot_w
        out_h = h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_vid_path, fourcc, fps, (out_w, out_h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            annotated = frame_bgr.copy()

            if track is not None and frame_idx in track.get("frames", {}):
                self._draw_hand_from_array(annotated, track["frames"][frame_idx])

            if overlay_lines:
                self._draw_top_right_text(annotated, overlay_lines)

            plot_panel = render_two_plot_panel(
                time_s=time_s,
                filtered_deg=filtered_deg,
                metrics=metrics,
                current_frame_idx=frame_idx,
                fps=fps,
                panel_h=h,
                panel_w=plot_w,
            )

            combined = cv2.hconcat([annotated, plot_panel])
            out.write(combined)

            frame_idx += 1

        cap.release()
        out.release()
        if CV2_OK:
            cv2.destroyAllWindows()

    def _process_video(self, video_path):
        normalized_path = os.path.normpath(video_path)
        base_name = os.path.basename(normalized_path)
        subfolder_name = os.path.basename(os.path.dirname(normalized_path))
        # Include subject folder (e.g. "Subject_1") in the output name when
        # present, so videos from different subjects don't overwrite each other.
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(normalized_path)))
        if re.match(r"(?i)subject_\d+", grandparent):
            new_filename = f"{grandparent}_{subfolder_name}_{base_name}"
        else:
            new_filename = f"{subfolder_name}_{base_name}"
        save_vid_path = os.path.join(self.save_dir, new_filename)

        main_track, total_frames, fps, fill_added, detected_before_fill, frame_hw = (
            self._infer_tracks_offline(video_path)
        )

        # ── Landmark refinement passes (SuperRes / RTMPose / OpenPose / YOLO) ──
        superres_refined = 0
        rtmpose_refined = 0
        openpose_refined = 0
        yolo_refined = 0
        gpu_cfg = _runtime_gpu_config()
        need_refinement = main_track is not None and (
            gpu_cfg["use_superres"]
            or gpu_cfg["use_rtmpose"]
            or gpu_cfg["use_openpose"]
            or gpu_cfg["use_yolo_hand"]
        )
        if need_refinement:
            need_gpu_lock = bool(
                gpu_cfg["use_superres"] or gpu_cfg["use_rtmpose"] or gpu_cfg["use_yolo_hand"]
            )
            gpu_acquired = False
            if need_gpu_lock:
                from .gpu_manager import acquire_gpu, cleanup_gpu, release_gpu

                gpu_acquired = acquire_gpu()
                if not gpu_acquired:
                    need_refinement = False
                    logger.warning(
                        "[GPU] acquire_gpu() timed out; skipping SuperRes/RTMPose refinement for this video "
                        "to avoid deadlock."
                    )
            else:
                gpu_acquired = True

        if need_refinement and gpu_acquired:
            try:
                frame_set = set(main_track.get("frames", {}).keys())
                sr_landmarker = None
                if gpu_cfg["use_superres"]:
                    try:
                        from .refinement.superres import superres_refine_landmarks

                        sr_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                            self.options_image
                        )
                    except Exception as e:
                        logger.error("[SuperRes] Could not create landmarker: %s", e)
                rtm_inferencer_ok = False
                if gpu_cfg["use_rtmpose"]:
                    try:
                        from .refinement.rtmpose import _get_inferencer, refine_landmarks_rtmpose

                        infer_result = _get_inferencer(
                            gpu_cfg["rtmpose_model_cfg"],
                            checkpoint_path=gpu_cfg["rtmpose_checkpoint_path"],
                        )
                        if infer_result is not None:
                            rtm_inferencer_ok = True
                        else:
                            logger.warning(
                                "[RTMPose] inferencer returned None unexpectedly. "
                                "RTMPose refinement will be skipped for this video."
                            )
                    except Exception as e:
                        logger.error(
                            "[RTMPose] Could not initialise inferencer — RTMPose refinement skipped: %s",
                            e,
                        )

                openpose_ready = False
                if gpu_cfg["use_openpose"]:
                    try:
                        from .refinement.openpose import (
                            _get_openpose_net,
                            refine_landmarks_openpose,
                        )

                        openpose_ready = (
                            _get_openpose_net(
                                gpu_cfg["openpose_proto_path"],
                                gpu_cfg["openpose_weights_path"],
                                use_cuda=gpu_cfg["openpose_use_cuda"],
                            )
                            is not None
                        )
                    except Exception as e:
                        logger.error(
                            "[OpenPose] Could not initialise network — OpenPose refinement skipped: %s",
                            e,
                        )

                yolo_ready = False
                if gpu_cfg["use_yolo_hand"]:
                    try:
                        from .refinement.yolo import _get_yolo_model, refine_landmarks_yolo

                        if _get_yolo_model(gpu_cfg["yolo_hand_model_path"]) is not None:
                            yolo_ready = True
                    except Exception as e:
                        logger.error(
                            "[YOLO-Hand] Could not initialise model — YOLO refinement skipped: %s",
                            e,
                        )

                cap_gpu = cv2.VideoCapture(video_path)
                fidx = 0
                while True:
                    ok, frame_bgr = cap_gpu.read()
                    if not ok:
                        break
                    if fidx in frame_set:
                        if sr_landmarker is not None:
                            try:
                                refined_sr = superres_refine_landmarks(
                                    frame_bgr,
                                    main_track,
                                    fidx,
                                    sr_landmarker,
                                    hand_to_track=self.hand_to_track,
                                    scale=SUPERRES_SCALE,
                                    model_name=SUPERRES_MODEL_NAME,
                                    model_path=SUPERRES_MODEL_PATH,
                                    half=SUPERRES_HALF,
                                    padding=ROI_REDETECT_PADDING,
                                )
                                if refined_sr is not None:
                                    main_track["frames"][fidx] = refined_sr
                                    superres_refined += 1
                            except Exception:
                                pass
                        if rtm_inferencer_ok:
                            try:
                                lm_current = main_track["frames"][fidx]
                                refined_rtm = refine_landmarks_rtmpose(
                                    frame_bgr,
                                    lm_current,
                                    model_cfg=gpu_cfg["rtmpose_model_cfg"],
                                    checkpoint_path=gpu_cfg["rtmpose_checkpoint_path"],
                                    padding=gpu_cfg["rtmpose_bbox_padding"],
                                )
                                if refined_rtm is not None:
                                    main_track["frames"][fidx] = refined_rtm
                                    rtmpose_refined += 1
                            except Exception:
                                pass
                        if openpose_ready:
                            try:
                                lm_current = main_track["frames"][fidx]
                                refined_openpose = refine_landmarks_openpose(
                                    frame_bgr,
                                    lm_current,
                                    proto_path=gpu_cfg["openpose_proto_path"],
                                    weights_path=gpu_cfg["openpose_weights_path"],
                                    padding=gpu_cfg["openpose_bbox_padding"],
                                    conf_thresh=gpu_cfg["openpose_conf_threshold"],
                                    input_size=gpu_cfg["openpose_input_size"],
                                    use_cuda=gpu_cfg["openpose_use_cuda"],
                                )
                                if refined_openpose is not None:
                                    main_track["frames"][fidx] = refined_openpose
                                    openpose_refined += 1
                            except Exception:
                                pass
                        if yolo_ready:
                            try:
                                lm_current = main_track["frames"][fidx]
                                refined_yolo = refine_landmarks_yolo(
                                    frame_bgr,
                                    lm_current,
                                    model_path=gpu_cfg["yolo_hand_model_path"],
                                    padding=gpu_cfg["yolo_hand_bbox_padding"],
                                    conf_thresh=gpu_cfg["yolo_hand_conf_threshold"],
                                )
                                if refined_yolo is not None:
                                    main_track["frames"][fidx] = refined_yolo
                                    yolo_refined += 1
                            except Exception:
                                pass
                    fidx += 1
                cap_gpu.release()

                if sr_landmarker is not None:
                    try:
                        sr_landmarker.close()
                    except Exception:
                        pass
                if gpu_cfg["use_openpose"]:
                    try:
                        from .refinement.openpose import cleanup_openpose

                        cleanup_openpose()
                    except Exception:
                        pass
                if gpu_cfg["use_yolo_hand"]:
                    try:
                        from .refinement.yolo import cleanup_yolo

                        cleanup_yolo()
                    except Exception:
                        pass
            finally:
                if need_gpu_lock:
                    cleanup_gpu()
                    release_gpu()

        main_track = ensure_track_visibility_channel(main_track)
        main_track = smooth_track_landmarks(main_track, total_frames, fps=fps)
        main_track = reject_landmark_outliers(main_track, fps=fps)

        ps_trimmed = False
        ps_start_frame = None
        ps_end_frame = None
        ps_duration_s = None
        if self._trim_to_ps and main_track is not None:
            main_track, ps_start_frame, ps_end_frame = trim_track_to_ps_segment(
                main_track,
                total_frames,
                fps,
                max_duration_s=self._max_ps_duration_s,
                min_duration_s=self._min_ps_duration_s,
            )
            if ps_start_frame is not None:
                ps_trimmed = True
                ps_duration_s = (ps_end_frame - ps_start_frame + 1) / fps

        medication_state = parse_medication_state_from_path(video_path)
        hand = parse_hand_from_path(video_path)
        if hand is None and main_track is not None:
            hand = main_track.get("hand_label")

        visit = None
        try:
            if (
                hasattr(self, "vid_score")
                and "video_path" in self.vid_score.columns
                and "visit" in self.vid_score.columns
            ):
                sel = self.vid_score[self.vid_score["video_path"] == video_path]
                if sel.empty:
                    norm = os.path.normpath(video_path)
                    try:
                        sel = self.vid_score[
                            self.vid_score["video_path"].apply(
                                lambda x: os.path.normpath(str(x)) == norm
                            )
                        ]
                    except Exception:
                        sel = pd.DataFrame()
                if not sel.empty:
                    v = sel.iloc[0]["visit"]
                    if pd.notna(v):
                        visit = v
        except Exception:
            visit = None

        ids_parsed, visit_parsed = parse_ids_and_visit(video_path)
        ids = canonicalize_video_id(ids_parsed)
        if visit is None:
            visit = visit_parsed

        clinical_score_info = resolve_video_clinical_score(
            video_path=video_path,
            video_to_patient=self.video_to_patient,
            scores_df=self.scores_df,
            score_column=self.score_column,
            visit=visit,
            medication_state=medication_state,
            hand=hand,
        )
        patient_id = clinical_score_info.get("patient_id")
        clinical_score = clinical_score_info.get("score_clean")
        clinical_score_raw = clinical_score_info.get("score_raw")
        score_line = None
        if self.score_csv_path:
            score_line = _format_clinical_score_overlay_line(self.score_column, clinical_score)

        adj_app, non_adj_app, first, last, det_win, win_len, det_total = (
            self._compute_appearance_metrics(main_track, total_frames)
        )
        avg_conf, conf_sum, conf_count = self._avg_confidence_from_track(main_track, first, last)

        time_s = np.arange(total_frames, dtype=float) / float(fps)

        metrics = None
        filtered_deg_full = np.full((total_frames,), np.nan, dtype=float)
        raw_deg_full = np.full((total_frames,), np.nan, dtype=float)
        signal_quality = 0.0
        sq_sub_scores = {}
        _mcp_indices = [
            int(HandLandmark.INDEX_FINGER_MCP),
            int(HandLandmark.MIDDLE_FINGER_MCP),
            int(HandLandmark.RING_FINGER_MCP),
            int(HandLandmark.PINKY_MCP),
        ]
        conf_idx_series = np.full((total_frames,), np.nan, dtype=float)
        conf_mid_series = np.full((total_frames,), np.nan, dtype=float)
        conf_ring_series = np.full((total_frames,), np.nan, dtype=float)
        conf_pinky_series = np.full((total_frames,), np.nan, dtype=float)
        conf_min_series = np.full((total_frames,), np.nan, dtype=float)
        conf_used_mask_series = np.zeros((total_frames,), dtype=float)
        conf_low_thresh = 0.5

        if main_track is not None:
            _frames_dict = main_track.get("frames", {})
            _conf_map, _conf_min, conf_low_thresh = _compute_mcp_confidence_proxy(
                _frames_dict,
                total_frames,
                _mcp_indices,
            )
            conf_idx_series = _conf_map[_mcp_indices[0]]
            conf_mid_series = _conf_map[_mcp_indices[1]]
            conf_ring_series = _conf_map[_mcp_indices[2]]
            conf_pinky_series = _conf_map[_mcp_indices[3]]
            conf_min_series = _conf_min
            conf_used_mask_series = np.where(
                np.isfinite(_conf_min) & (_conf_min >= conf_low_thresh),
                1.0,
                0.0,
            )

            if USE_PCA_ANGLE:
                raw_deg_full = _build_pca_angle_deg(_frames_dict, total_frames, fps=fps)
            else:
                raw_deg_full = _build_unwrapped_angle_deg(
                    _frames_dict, total_frames, self._knuckle_line_angle_rad, fps=fps
                )

            wrist_z = _build_wrist_z_signal(_frames_dict, total_frames)

            analyzer = KinematicAnalyzer(
                time_s,
                raw_deg_full,
                fps=fps,
                cutoff_hz=self._cutoff_hz,
                filter_order=self._filter_order,
                highpass_hz=self._highpass_hz,
                wrist_z=wrist_z,
            )
            if ps_trimmed and ps_start_frame is not None:
                if ps_start_frame > 0:
                    analyzer.clean_signal[:ps_start_frame] = 0.0
                if ps_end_frame + 1 < total_frames:
                    analyzer.clean_signal[ps_end_frame + 1 :] = 0.0
            filtered_deg_full = analyzer.clean_signal.copy()
            if ps_trimmed and ps_start_frame is not None:
                filtered_deg_full[:ps_start_frame] = np.nan
                filtered_deg_full[ps_end_frame + 1 :] = np.nan
            metrics = analyzer.extract_features(
                prominence_deg=self._prominence_deg,
                max_movement_hz=self._max_movement_hz,
                adaptive_prom_frac=self._adaptive_prom_frac,
            )
            sq_result = analyzer.compute_signal_quality(
                metrics,
                ps_start_frame=ps_start_frame if ps_trimmed else None,
                ps_end_frame=ps_end_frame if ps_trimmed else None,
            )
            signal_quality = sq_result["signal_quality"]
            sq_sub_scores = sq_result["sq_sub_scores"]

            inter_mcp_span = _compute_inter_mcp_span(_frames_dict, total_frames)
            arm_swing_index = _compute_arm_swing_index(
                _frames_dict,
                total_frames,
                ps_start_frame=ps_start_frame if ps_trimmed else None,
                ps_end_frame=ps_end_frame if ps_trimmed else None,
                inter_mcp_span=inter_mcp_span,
            )
            if metrics is not None:
                metrics["inter_mcp_span"] = inter_mcp_span
                metrics["arm_swing_index"] = arm_swing_index

        # ── Video quality factor metrics ──────────────────────────────────
        vq_metrics = {}
        if main_track is not None:
            _vq_frames = main_track.get("frames", {})
            vq_metrics = compute_video_quality_metrics(
                video_path,
                _vq_frames,
                total_frames,
                fps,
                ps_start_frame=ps_start_frame if ps_trimmed else None,
                ps_end_frame=ps_end_frame if ps_trimmed else None,
                frame_hw=frame_hw,
            )

        if main_track is None:
            overlay_lines = [
                "No main track",
                f"FPS: {fps:.1f} Frames: {total_frames}",
            ]
        else:
            overlay_lines = [
                f"AdjApp: {adj_app:5.1f}%" if adj_app is not None else "AdjApp: N/A",
                f"NonAdjApp: {non_adj_app:5.1f}%" if non_adj_app is not None else "NonAdjApp: N/A",
                (
                    f"FillAdded: {fill_added} Conf: {avg_conf:.2f}"
                    if avg_conf is not None
                    else f"FillAdded: {fill_added}"
                ),
            ]
            if score_line:
                overlay_lines.insert(0, score_line)
            if metrics is not None:
                overlay_lines.extend(
                    [
                        f"MeanAmp: {metrics['avg_amp']:.1f} deg  AmpCV: {metrics['amp_cv']:.1f}%",
                        f"MeanFreq: {metrics['freq']:.2f} Hz",
                        f"NormDecSlope: {metrics['norm_decrement_slope']:.2f}%/s  RhythmCV: {metrics['cv']:.1f}%",
                        f"NormTISlope: {metrics['norm_ti_slope']:.2f}%/cyc  Arrests: {metrics['num_arrests']}",
                        f"SigQuality: {signal_quality:.3f}",
                    ]
                )
            else:
                overlay_lines.append("Metrics: insufficient cycles")
            if ps_trimmed:
                overlay_lines.append(f"PS: {ps_duration_s:.1f}s [{ps_start_frame}-{ps_end_frame}]")
        if main_track is None and score_line:
            overlay_lines.insert(0, score_line)

        render_ok = False
        if EXPORT_PLOT_VIDEO:
            try:
                self._render_video_with_plots(
                    video_path=video_path,
                    save_vid_path=save_vid_path,
                    track=main_track,
                    overlay_lines=overlay_lines,
                    time_s=time_s,
                    filtered_deg=filtered_deg_full,
                    metrics=metrics,
                )
                render_ok = True
            except Exception:
                render_ok = False

        if main_track is None:
            rot_total = None
            rot_amp = None
            det_frames = 0
            avg_rot = None
            chosen_track_id = None
        else:
            rot_total = float(main_track.get("rot_total", 0.0))
            rot_amp = float(main_track.get("angle_max", 0.0) - main_track.get("angle_min", 0.0))
            det_frames = len(main_track.get("frames", {}))
            avg_rot = rot_total / max(1, det_frames)
            chosen_track_id = int(main_track.get("id", -1))

        from .utils import ROT_AVG_THRESH
        from .utils import SCIPY_OK as _scipy_ok

        log_row = {
            "record_type": "VIDEO",
            "video_path": video_path,
            "output_video": save_vid_path,
            "patient_id": patient_id,
            "clinical_score_column": self.score_column if self.score_csv_path else None,
            "clinical_score": clinical_score,
            "clinical_score_raw": clinical_score_raw,
            "hand": hand,
            "fps": float(fps) if fps else None,
            "total_frames": int(total_frames),
            "main_track_found": bool(main_track is not None),
            "chosen_track_id": chosen_track_id,
            "rot_total_rad": rot_total,
            "rot_amp_rad": rot_amp,
            "detected_frames": int(det_frames),
            "avg_rot_rad_per_frame": avg_rot,
            "rot_avg_thresh": float(ROT_AVG_THRESH),
            "adjusted_appearance_pct": adj_app,
            "non_adjusted_appearance_pct": non_adj_app,
            "avg_conf": avg_conf,
            "conf_sum": conf_sum,
            "conf_count": int(conf_count),
            "detected_in_window": int(det_win),
            "window_len": int(win_len),
            "detected_total": int(det_total),
            "detected_before_fill": int(detected_before_fill),
            "fill_added_frames": int(fill_added),
            "keypoint_backend": gpu_cfg["keypoint_backend"],
            "superres_enabled": bool(gpu_cfg["use_superres"]),
            "rtmpose_enabled": bool(gpu_cfg["use_rtmpose"]),
            "openpose_enabled": bool(gpu_cfg["use_openpose"]),
            "superres_refined_frames": int(superres_refined),
            "rtmpose_refined_frames": int(rtmpose_refined),
            "openpose_refined_frames": int(openpose_refined),
            "render_ok": bool(render_ok),
            "scipy_ok": bool(_scipy_ok),
            "log_medication_state": medication_state,
            "log_hand": hand,
            "ids": ids,
            "visit": visit,
            "ps_trimmed": ps_trimmed,
            "ps_start_frame": ps_start_frame,
            "ps_end_frame": ps_end_frame,
            "ps_duration_s": ps_duration_s,
        }

        if metrics is not None:
            log_row.update(
                {
                    "Mean Amplitude": float(metrics["avg_amp"]),
                    "Amplitude CV": float(metrics["amp_cv"]),
                    "Mean Frequency": float(metrics["freq"]),
                    "Avg Cycle Duration": float(metrics.get("avg_cycle_duration", float("nan"))),
                    "Rhythm (CV %)": float(metrics["cv"]),
                    "Cycle Duration CV": float(metrics.get("cycle_duration_cv", float("nan"))),
                    "Norm Decrement Slope": float(metrics["norm_decrement_slope"]),
                    "Raw Amp Slope": float(metrics.get("raw_amp_slope", float("nan"))),
                    "Amp Decrement Onset": float(metrics["amp_decrement_onset"]),
                    "Amp Decrement %": float(metrics.get("amp_decrement_pct", float("nan"))),
                    "Norm TI Slope": float(metrics["norm_ti_slope"]),
                    "Raw Cycle Duration Slope": float(
                        metrics.get("raw_cycle_duration_slope", float("nan"))
                    ),
                    "Num Hesitations": int(metrics.get("num_hesitations", 0)),
                    "Num Arrests": int(metrics["num_arrests"]),
                    "Num Interruptions (2x)": int(metrics.get("num_interruptions", 0)),
                    "Max Pause Duration (s)": float(
                        metrics.get("max_pause_duration_s", float("nan"))
                    ),
                    "Pause Time Ratio": float(metrics.get("pause_time_ratio", float("nan"))),
                    "Peak Velocity": float(metrics.get("peak_velocity", float("nan"))),
                    "Mean Velocity": float(metrics.get("mean_velocity", float("nan"))),
                    "Peak Velocity CV": float(metrics.get("peak_velocity_cv", float("nan"))),
                    "Mean Velocity CV": float(metrics.get("mean_velocity_cv", float("nan"))),
                    "Norm Velocity Decrement Slope": float(
                        metrics.get("norm_velocity_decrement_slope", float("nan"))
                    ),
                    "Raw Velocity Slope": float(metrics.get("raw_velocity_slope", float("nan"))),
                    "Raw Speed Slope": float(metrics.get("raw_speed_slope", float("nan"))),
                    "Velocity Decrement Onset": float(
                        metrics.get("velocity_decrement_onset", float("nan"))
                    ),
                    "Velocity Decrement %": float(
                        metrics.get("velocity_decrement_pct", float("nan"))
                    ),
                    "Global Velocity": float(metrics.get("global_velocity", float("nan"))),
                    "Arm Swing Index": (
                        float(metrics["arm_swing_index"])
                        if metrics.get("arm_swing_index") is not None
                        else float("nan")
                    ),
                    "Sample Entropy": float(metrics.get("sample_entropy", float("nan"))),
                    "Amp-Vel Coupling": float(metrics.get("amp_vel_coupling", float("nan"))),
                    "Inter-MCP Span": (
                        float(metrics["inter_mcp_span"])
                        if metrics.get("inter_mcp_span") is not None
                        else None
                    ),
                    "Total Cycles": int(metrics["total_cycles"]),
                    "Quality Cycles": int(metrics["quality_cycles"]),
                    "Signal Quality": float(signal_quality),
                }
            )
            for sq_key, sq_val in sq_sub_scores.items():
                log_row[f"SQ_{sq_key}"] = float(sq_val)
        else:
            log_row.update(
                {
                    "Mean Amplitude": None,
                    "Amplitude CV": None,
                    "Mean Frequency": None,
                    "Avg Cycle Duration": None,
                    "Rhythm (CV %)": None,
                    "Cycle Duration CV": None,
                    "Norm Decrement Slope": None,
                    "Raw Amp Slope": None,
                    "Amp Decrement Onset": None,
                    "Amp Decrement %": None,
                    "Norm TI Slope": None,
                    "Raw Cycle Duration Slope": None,
                    "Num Hesitations": None,
                    "Num Arrests": None,
                    "Num Interruptions (2x)": None,
                    "Max Pause Duration (s)": None,
                    "Pause Time Ratio": None,
                    "Peak Velocity": None,
                    "Mean Velocity": None,
                    "Peak Velocity CV": None,
                    "Mean Velocity CV": None,
                    "Norm Velocity Decrement Slope": None,
                    "Raw Velocity Slope": None,
                    "Raw Speed Slope": None,
                    "Velocity Decrement Onset": None,
                    "Velocity Decrement %": None,
                    "Global Velocity": None,
                    "Arm Swing Index": None,
                    "Sample Entropy": None,
                    "Amp-Vel Coupling": None,
                    "Inter-MCP Span": None,
                    "Total Cycles": 0,
                    "Quality Cycles": 0,
                    "Signal Quality": 0.0,
                }
            )

        log_row.update(
            {
                "raw_rotation_series": series_to_json(raw_deg_full),
                "filtered_rotation_series": series_to_json(filtered_deg_full),
                "conf_index_mcp_series": series_to_json(conf_idx_series, decimals=4),
                "conf_middle_mcp_series": series_to_json(conf_mid_series, decimals=4),
                "conf_ring_mcp_series": series_to_json(conf_ring_series, decimals=4),
                "conf_pinky_mcp_series": series_to_json(conf_pinky_series, decimals=4),
                "conf_mcp_min_series": series_to_json(conf_min_series, decimals=4),
                "conf_mcp_used_mask_series": series_to_json(conf_used_mask_series, decimals=0),
                "conf_mcp_low_threshold": float(conf_low_thresh),
            }
        )
        if metrics is not None:
            log_row.update(
                {
                    "cycle_peak_times_s": series_to_json(
                        metrics.get("detected_peak_times", metrics.get("peak_times", [])),
                        decimals=4,
                    ),
                    "cycle_trough_times_s": series_to_json(
                        metrics.get("trough_times", []), decimals=4
                    ),
                    "cycle_amplitudes_deg": series_to_json(metrics["amplitudes"], decimals=4),
                    "cycle_trendline_deg": series_to_json(metrics["trend_line"], decimals=4),
                }
            )

        # ── Video quality factor columns ──────────────────────────────────
        _frames_dict_vq = main_track.get("frames", {}) if main_track is not None else {}
        log_row.update(
            {
                "VQ_video_width_px": vq_metrics.get("video_width_px"),
                "VQ_video_height_px": vq_metrics.get("video_height_px"),
                "VQ_inter_mcp_span_px": _compute_inter_mcp_span_px(
                    _frames_dict_vq,
                    total_frames,
                    vq_metrics.get("video_width_px"),
                    vq_metrics.get("video_height_px"),
                ),
                "VQ_hand_bbox_area_median_px": vq_metrics.get(
                    "hand_bbox_area_median_px", float("nan")
                ),
                "VQ_hand_bbox_area_q25_px": vq_metrics.get("hand_bbox_area_q25_px", float("nan")),
                "VQ_sharpness_median": vq_metrics.get("sharpness_median", float("nan")),
                "VQ_sharpness_q10": vq_metrics.get("sharpness_q10", float("nan")),
                "VQ_sharpness_q25": vq_metrics.get("sharpness_q25", float("nan")),
                "VQ_luminance_median": vq_metrics.get("luminance_median", float("nan")),
                "VQ_luminance_mean": vq_metrics.get("luminance_mean", float("nan")),
                "VQ_luminance_cv": vq_metrics.get("luminance_cv", float("nan")),
                "VQ_luminance_uniformity_median": vq_metrics.get(
                    "luminance_uniformity_median", float("nan")
                ),
                "VQ_saturation_frac_median": vq_metrics.get("saturation_frac_median", float("nan")),
                "VQ_detection_rate": vq_metrics.get("detection_rate", float("nan")),
                "VQ_n_gaps": vq_metrics.get("n_gaps", 0),
                "VQ_longest_gap_frames": vq_metrics.get("longest_gap_frames", 0),
                "VQ_longest_gap_s": vq_metrics.get("longest_gap_s", float("nan")),
                "VQ_mean_gap_frames": vq_metrics.get("mean_gap_frames", float("nan")),
                "VQ_total_gap_frames": vq_metrics.get("total_gap_frames", 0),
                "VQ_gap_fraction": vq_metrics.get("gap_fraction", float("nan")),
                # Global (full-frame) metrics — no landmark dependency
                "VQ_global_sharpness_median": vq_metrics.get(
                    "global_sharpness_median", float("nan")
                ),
                "VQ_global_sharpness_q10": vq_metrics.get("global_sharpness_q10", float("nan")),
                "VQ_global_contrast_median": vq_metrics.get("global_contrast_median", float("nan")),
                "VQ_global_luminance_median": vq_metrics.get(
                    "global_luminance_median", float("nan")
                ),
                "VQ_global_luminance_cv": vq_metrics.get("global_luminance_cv", float("nan")),
                "VQ_global_saturation_frac_median": vq_metrics.get(
                    "global_saturation_frac_median", float("nan")
                ),
                "VQ_temporal_diff_median": vq_metrics.get("temporal_diff_median", float("nan")),
                "VQ_temporal_diff_cv": vq_metrics.get("temporal_diff_cv", float("nan")),
                "VQ_bitrate_kbps": vq_metrics.get("bitrate_kbps", float("nan")),
                "VQ_resolution_area_mpx": vq_metrics.get("resolution_area_mpx", float("nan")),
            }
        )

        return log_row

    def process_hand_videos_random(
        self,
        n=20,
        seed=None,
        num_workers=None,
        node_rank=0,
        num_nodes=1,
        start_frac=None,
        end_frac=None,
        retry_failed=False,
        retry_timeout=3600,
    ):
        # ── Retry-failed mode ─────────────────────────────────────────────
        # When retry_failed is set, build video_tasks exclusively from
        # TIMEOUT/ERROR/CRASH entries in an existing tracking log.
        # retry_timeout sets the parent-side kill cap in seconds (default 1 h);
        # pass 0 to disable the cap entirely (original behaviour, not recommended).
        _worker_timeout: float | None = 1800  # default: 30-min parent-side cap
        if retry_failed:
            _retry_log_path = retry_failed if isinstance(retry_failed, str) else self.log_csv_path
            try:
                _log_df = pd.read_csv(_retry_log_path)
            except Exception as _e:
                logger.error("[RetryFailed] Cannot read log '%s': %s", _retry_log_path, _e)
                return pd.DataFrame()
            _FAILED = {"ERROR", "TIMEOUT", "CRASH"}
            if "record_type" not in _log_df.columns:
                logger.warning(
                    "[RetryFailed] Log '%s' has no 'record_type' column — nothing to retry.",
                    _retry_log_path,
                )
                return _log_df

            # ── Collect tasks from two sources ────────────────────────────
            # 1) Explicit failure records (ERROR / TIMEOUT / CRASH) in the log.
            _failed_rows = _log_df[_log_df["record_type"].isin(_FAILED)]
            video_tasks = []
            for _, _r in _failed_rows.iterrows():
                _vp = str(_r.get("video_path", ""))
                _h = _r.get("hand", None)
                if isinstance(_h, str) and _h.strip() not in ("", "nan", "None"):
                    _h = _h.strip()
                else:
                    _h = None
                if _vp:
                    video_tasks.append((_vp, _h))

            # 2) Videos present in the source input CSV but entirely absent
            #    from the log (no record of any kind).  This catches videos
            #    whose TIMEOUT rows were lost due to a mid-run job cancel.
            if self.vid_score is not None and not self.vid_score.empty:
                _logged_paths = set(_log_df["video_path"].astype(str))
                for _, _row in self.vid_score.iterrows():
                    _vp = str(_row.get("video_path", ""))
                    if _vp and _vp not in _logged_paths:
                        _h = parse_hand_from_path(_vp)
                        video_tasks.append((_vp, _h))
                        logger.debug("[RetryFailed] Source CSV has unlogged video: %s", _vp)

            # Deduplicate while preserving order.
            _seen: set = set()
            _deduped = []
            for _vp, _h in video_tasks:
                _key = (str(_vp), str(_h))
                if _key not in _seen:
                    _seen.add(_key)
                    _deduped.append((_vp, _h))
            video_tasks = _deduped

            if not video_tasks:
                logger.info(
                    "[RetryFailed] No failed or missing videos found. Nothing to do.",
                )
                return _log_df

            # Apply a generous-but-finite parent-side kill cap so genuinely
            # broken videos (stale NFS handle, corrupt codec) don't hang forever.
            # The SIGALRM inside the worker is suppressed; the parent SIGKILL is
            # more reliable (can't be caught) and sufficient.
            _worker_timeout = retry_timeout if retry_timeout > 0 else None
            _cap_str = f"{_worker_timeout:.0f}s" if _worker_timeout else "unlimited"
            logger.info(
                "[RetryFailed] %d video(s) to re-run (%d explicit failures, "
                "%d missing from log) with timeout=%s.",
                len(video_tasks),
                len(_failed_rows),
                len(video_tasks) - len(_failed_rows),
                _cap_str,
            )
            # Point self.log_csv_path at the retry log so the resume logic
            # loads VIDEO records from the right file and writes output there.
            self.log_csv_path = _retry_log_path
            # Suppress SIGALRM inside workers — parent SIGKILL handles the cap.
            self.config["_no_worker_timeout"] = True
            # Keep the full retry task list so we can restore TIMEOUT records
            # for any tasks that were killed before producing a result (e.g.
            # SLURM job cancel or OOM) — without this, those videos vanish from
            # the CSV and a subsequent --retry-failed finds nothing to do.
            _retry_video_tasks = list(video_tasks)
        else:
            _retry_video_tasks = []

            if n is None:
                n = len(self.vid_score)
            else:
                try:
                    n = int(n)
                except Exception:
                    n = len(self.vid_score)
                n = max(0, min(n, len(self.vid_score)))
            random_rows = self.vid_score.sample(n=n, random_state=seed)

            video_tasks = []
            for _, row in random_rows.iterrows():
                video_path = row["video_path"]
                h_parsed = parse_hand_from_path(video_path)
                hand_to_track = h_parsed
                video_tasks.append((video_path, hand_to_track))

        # ── Multi-node partition ───────────────────────────────────────────
        num_nodes = max(1, int(num_nodes))
        node_rank = max(0, min(int(node_rank), num_nodes - 1))
        if num_nodes > 1:
            total = len(video_tasks)
            if start_frac is not None and end_frac is not None:
                # Proportional contiguous slice: each node receives a chunk
                # sized to match its worker count so all nodes finish together.
                i_start = round(total * start_frac)
                i_end = round(total * end_frac)
                video_tasks = video_tasks[i_start:i_end]
                logger.info(
                    "[MultiNode] node %d/%d — processing %d/%d videos "
                    "(proportional slice %.1f%%–%.1f%%)",
                    node_rank,
                    num_nodes,
                    len(video_tasks),
                    total,
                    start_frac * 100,
                    end_frac * 100,
                )
            else:
                # Fallback: even striping when no worker counts were provided.
                video_tasks = video_tasks[node_rank::num_nodes]
                logger.info(
                    "[MultiNode] node %d/%d — processing %d/%d videos (striped)",
                    node_rank,
                    num_nodes,
                    len(video_tasks),
                    total,
                )

        # ── Resume: skip videos already successfully processed ───────────
        _resume_logs: list = []
        if os.path.isfile(self.log_csv_path):
            try:
                _existing = pd.read_csv(self.log_csv_path)

                # Self-heal any duplicates that crept in from previous
                # interrupted runs.  Keep the last entry for each
                # (video_path, hand) pair — the most recent completion is
                # the most likely to be valid.
                if {"video_path", "hand"}.issubset(_existing.columns):
                    _before = len(_existing)
                    _existing = _existing.drop_duplicates(
                        subset=["video_path", "hand"], keep="last"
                    ).reset_index(drop=True)
                    _n_dropped = _before - len(_existing)
                    if _n_dropped:
                        logger.warning(
                            "[Resume] Removed %d duplicate row(s) from existing log "
                            "(kept most recent entry per video/hand pair).",
                            _n_dropped,
                        )

                def _norm_hand_key(h) -> str:
                    """Canonical hand key: maps None / NaN / '' / 'nan' / 'None' → ''."""
                    if h is None:
                        return ""
                    s = str(h).strip()
                    return "" if s in ("", "nan", "None") else s

                def _has_valid_confidence_payload(_row) -> bool:
                    try:
                        if not bool(_row.get("main_track_found", False)):
                            return True
                        conf_min = _row.get("conf_mcp_min_series", None)
                        conf_used = _row.get("conf_mcp_used_mask_series", None)
                        if conf_min is None or conf_used is None:
                            return False
                        if (isinstance(conf_min, float) and np.isnan(conf_min)) or (
                            isinstance(conf_used, float) and np.isnan(conf_used)
                        ):
                            return False
                        conf_min_arr = (
                            json.loads(conf_min) if isinstance(conf_min, str) else conf_min
                        )
                        conf_used_arr = (
                            json.loads(conf_used) if isinstance(conf_used, str) else conf_used
                        )
                        if not isinstance(conf_min_arr, list) or not isinstance(
                            conf_used_arr, list
                        ):
                            return False
                        if len(conf_min_arr) == 0 or len(conf_used_arr) == 0:
                            return False
                        return any(v is not None for v in conf_min_arr)
                    except Exception:
                        return False

                if "record_type" in _existing.columns:
                    _done_mask = _existing["record_type"] == "VIDEO"
                else:
                    _done_mask = pd.Series([True] * len(_existing), index=_existing.index)
                # Normalise paths so that symlink vs. resolved, double-slash,
                # or mount-point differences don't prevent resume matching.
                _done_pairs: set = set()
                for _, _erow in _existing[_done_mask].iterrows():
                    if not _has_valid_confidence_payload(_erow):
                        continue
                    _vp = os.path.normpath(str(_erow.get("video_path", "")))
                    _h = _norm_hand_key(_erow.get("hand"))
                    if _vp:
                        _done_pairs.add((_vp, _h))
                _resume_logs = [
                    r
                    for r in _existing.to_dict("records")
                    if r.get("record_type") == "VIDEO" and _has_valid_confidence_payload(r)
                ]
                _original_task_count = len(video_tasks)
                video_tasks = [
                    (vp, h)
                    for vp, h in video_tasks
                    if (os.path.normpath(str(vp)), _norm_hand_key(h)) not in _done_pairs
                ]
                _n_skipped = _original_task_count - len(video_tasks)
                if _n_skipped > 0:
                    logger.info(
                        "[Resume] %d/%d videos already completed — skipping them. %d remaining.",
                        _n_skipped,
                        _original_task_count,
                        len(video_tasks),
                    )
                elif _original_task_count > 0:
                    logger.info(
                        "[Resume] Existing log found but no completed videos overlap "
                        "with the current task list — processing all videos."
                    )
            except Exception as _resume_exc:
                logger.warning(
                    "[Resume] Could not read existing log '%s' (%s); starting fresh.",
                    self.log_csv_path,
                    _resume_exc,
                )
                _resume_logs = []

        if not video_tasks:
            logger.info("[Resume] All videos already processed. Nothing to do.")
            _df_done = pd.DataFrame(_resume_logs)
            if not _df_done.empty:
                _log_dir = os.path.dirname(self.log_csv_path)
                if _log_dir:
                    os.makedirs(_log_dir, exist_ok=True)
                _df_done.to_csv(self.log_csv_path, index=False)
                logger.info("Results saved to %s", self.log_csv_path)
            return _df_done

        if num_workers is None:
            num_workers = NUM_WORKERS if NUM_WORKERS is not None else os.cpu_count()
        num_workers = max(1, min(num_workers, len(video_tasks)))

        # ── FD-aware worker cap (Linux/Unix) ────────────────────────────
        if _RESOURCE_OK:
            try:
                soft_nofile, _ = _resource.getrlimit(_resource.RLIMIT_NOFILE)
                reserve_fds = 256
                fds_per_worker = 8
                max_workers_by_fd = max(1, int((soft_nofile - reserve_fds) // fds_per_worker))
                if num_workers > max_workers_by_fd:
                    logger.info(
                        "[FD] Capping workers from %d to %d based on RLIMIT_NOFILE=%d",
                        num_workers,
                        max_workers_by_fd,
                        soft_nofile,
                    )
                    num_workers = max_workers_by_fd
            except Exception:
                pass

        logger.info(
            "Processing %d videos using %d parallel workers...", len(video_tasks), num_workers
        )

        gpu_concurrency_per_device = _runtime_gpu_config()["gpu_concurrency"]
        visible_gpu_ids = _parse_visible_gpu_ids()
        n_visible_gpus = max(1, len(visible_gpu_ids))
        gpu_concurrency_total = int(gpu_concurrency_per_device) * n_visible_gpus
        logger.info(
            "[GPU] visible=%s GPU_CONCURRENCY(per-device)=%s total_gpu_slots=%d",
            visible_gpu_ids,
            gpu_concurrency_per_device,
            gpu_concurrency_total,
        )

        _mp_ctx = mp_proc.get_context("spawn")
        gpu_sem = _mp_ctx.Semaphore(gpu_concurrency_total)

        logs = list(_resume_logs)
        completed = 0

        # ── Atomic CSV helper ────────────────────────────────────────────────
        def _atomic_csv_flush():
            """Write logs to a .tmp file then rename over the final path (atomic)."""
            try:
                _tmp = self.log_csv_path + ".tmp"
                _flush_df = pd.DataFrame(logs)
                if {"video_path", "hand"}.issubset(_flush_df.columns):
                    _before = len(_flush_df)
                    _flush_df = _flush_df.drop_duplicates(
                        subset=["video_path", "hand"], keep="last"
                    ).reset_index(drop=True)
                    if len(_flush_df) < _before:
                        logger.warning(
                            "[Flush] Removed %d duplicate row(s) before writing CSV.",
                            _before - len(_flush_df),
                        )
                _flush_df.to_csv(_tmp, index=False)
                os.replace(_tmp, self.log_csv_path)
            except Exception as _e:
                logger.warning("Could not save intermediate CSV: %s", _e)

        # ── SIGTERM handler (SLURM sends SIGTERM before SIGKILL) ─────────────
        # Flush whatever records we have so a job-time-limit cancel doesn't
        # lose completions that happened after the last intermediate flush.
        import signal as _signal

        _sigterm_flushed = False

        def _on_sigterm(signum, frame):
            nonlocal _sigterm_flushed
            if not _sigterm_flushed:
                _sigterm_flushed = True
                logger.warning("[SIGTERM] Caught — flushing %d records to CSV.", len(logs))
                _atomic_csv_flush()
            raise SystemExit(0)

        try:
            _signal.signal(_signal.SIGTERM, _on_sigterm)
        except Exception:
            pass  # Windows / environments that don't support SIGTERM

        MIN_FREE_MEM_GIB = 4.0
        _mem_warn_printed = False
        gpu_holders: set = set()
        # pid → monotonic time when the parent received the _gpu_acquired sentinel.
        # Used to pro-actively kill workers that hold the GPU semaphore for too long
        # so that other waiting workers are not stranded past the acquire timeout.
        gpu_acquire_times: dict = {}
        from .gpu_manager import _GPU_ACQUIRE_TIMEOUT_S

        # Kill a GPU-holding worker this many seconds before the acquire timeout
        # fires for waiting workers, giving the semaphore time to be recovered.
        _GPU_HOLD_KILL_S = _GPU_ACQUIRE_TIMEOUT_S - 30

        task_iter = iter(enumerate(video_tasks))
        tasks_exhausted = False
        active = {}

        def _free_mem_gib():
            if not _PSUTIL_OK:
                return float("inf")
            return _psutil.virtual_memory().available / (1024**3)

        def _launch_next():
            nonlocal tasks_exhausted, _mem_warn_printed
            free_gib = _free_mem_gib()
            if free_gib < MIN_FREE_MEM_GIB:
                if not _mem_warn_printed:
                    logger.warning(
                        "[MEM] Only %.1f GiB free (threshold %s GiB) "
                        "— holding back new workers until memory is available. "
                        "Active workers: %d",
                        free_gib,
                        MIN_FREE_MEM_GIB,
                        len(active),
                    )
                    _mem_warn_printed = True
                return False
            _mem_warn_printed = False
            try:
                task_idx, (video_path, hand_to_track) = next(task_iter)
            except StopIteration:
                tasks_exhausted = True
                return False
            result_q = _mp_ctx.Queue()
            report_q = _mp_ctx.Queue()
            p = _mp_ctx.Process(
                target=_robust_worker_entry,
                args=(
                    video_path,
                    hand_to_track,
                    self.config,
                    self.save_dir,
                    gpu_sem,
                    visible_gpu_ids[task_idx % n_visible_gpus],
                    result_q,
                    report_q,
                ),
                daemon=False,
            )
            try:
                p.start()
            except OSError as e:
                emfile = getattr(e, "errno", None) == 24
                if emfile:
                    logger.warning(
                        "[FD] Process start failed with EMFILE (too many open files). "
                        "Will retry launching after current workers finish."
                    )
                else:
                    logger.error("[WorkerStart] Failed to start worker for %s: %s", video_path, e)
                try:
                    result_q.close()
                except Exception:
                    pass
                try:
                    report_q.close()
                except Exception:
                    pass
                return False
            try:
                result_q.cancel_join_thread()
            except Exception:
                pass
            try:
                report_q.cancel_join_thread()
            except Exception:
                pass
            active[p.pid] = {
                "proc": p,
                "result_q": result_q,
                "report_q": report_q,
                "video_path": video_path,
                "hand": hand_to_track,
                "start": time.monotonic(),
                "task_idx": task_idx,
            }
            return True

        def _collect_result(info, record_type="OK", error_msg=None):
            nonlocal completed
            if record_type == "OK":
                try:
                    result = info["result_q"].get(timeout=30)
                    logs.append(result)
                except Exception:
                    logs.append(
                        {
                            "record_type": "ERROR",
                            "video_path": info["video_path"],
                            "hand": info["hand"],
                            "error": "worker finished but returned no result",
                        }
                    )
            else:
                logs.append(
                    {
                        "record_type": record_type,
                        "video_path": info["video_path"],
                        "hand": info["hand"],
                        "error": error_msg or "unknown",
                    }
                )
            completed += 1
            # Flush after every completion so records are never lost on job cancel.
            _atomic_csv_flush()
            if completed % 5 == 0 or completed == len(video_tasks):
                logger.info("Progress: %d/%d videos processed.", completed, len(video_tasks))

        # Fill initial worker slots
        for _ in range(num_workers):
            if not _launch_next():
                break

        # Main monitoring loop
        while active:
            time.sleep(2)
            now = time.monotonic()
            done_pids = []

            for pid, info in list(active.items()):
                proc = info["proc"]
                elapsed = now - info["start"]

                try:
                    q = info["report_q"]
                    while True:
                        msg = q.get_nowait()
                        if isinstance(msg, dict) and msg.get("_gpu_acquired"):
                            gpu_holders.add(pid)
                            gpu_acquire_times.setdefault(pid, now)
                        elif isinstance(msg, dict) and msg.get("_gpu_released"):
                            gpu_holders.discard(pid)
                            gpu_acquire_times.pop(pid, None)
                except Exception:
                    pass

                if not proc.is_alive():
                    proc.join(timeout=5)
                    # Drain any remaining report-queue messages BEFORE checking
                    # gpu_holders.  Without this drain, a worker that sends
                    # _gpu_acquired and then immediately SIGSEGVs can die between
                    # the sentinel being enqueued and the parent's 2-second poll
                    # processing it.  The parent would then see pid not in
                    # gpu_holders and skip the semaphore release, causing a
                    # permanent 300-second stall for every subsequent worker.
                    try:
                        q = info["report_q"]
                        while True:
                            msg = q.get_nowait()
                            if isinstance(msg, dict) and msg.get("_gpu_acquired"):
                                gpu_holders.add(pid)
                                gpu_acquire_times.setdefault(pid, now)
                            elif isinstance(msg, dict) and msg.get("_gpu_released"):
                                gpu_holders.discard(pid)
                                gpu_acquire_times.pop(pid, None)
                    except Exception:
                        pass
                    gpu_acquire_times.pop(pid, None)  # clean up on worker exit
                    if proc.exitcode == 0:
                        _collect_result(info)
                    else:
                        exit_reason = (
                            "OOM-killed by kernel (SIGKILL)"
                            if proc.exitcode == -9
                            else f"exit code {proc.exitcode}"
                        )
                        _collect_result(
                            info,
                            "CRASH",
                            f"worker {exit_reason} after {elapsed:.0f}s",
                        )
                        logger.error(
                            "[CRASH] Video %s — worker pid=%s %s after %.0fs",
                            info["video_path"],
                            pid,
                            exit_reason,
                            elapsed,
                        )
                    if pid in gpu_holders:
                        gpu_holders.discard(pid)
                        try:
                            gpu_sem.release()
                        except Exception:
                            pass
                        logger.warning("[GPU] Recovered semaphore from dead worker pid=%s", pid)
                    elif proc.exitcode != 0:
                        # Worker crashed without the _gpu_acquired sentinel being
                        # delivered.  Unconditionally release one slot — in YOLO-only
                        # mode the worker acquires the GPU as its very first action, so
                        # any crash with exitcode != 0 almost certainly consumed a slot.
                        # The old "probe-then-put-back" approach was broken: with N
                        # workers dying simultaneously the probe succeeds (count > 0),
                        # restores the slot it just took, and nets zero recovery per
                        # worker — leaving all N slots permanently lost.
                        # Worst-case cost of an unconditional release: the worker crashed
                        # before acquire_gpu() (very rare, < 1s elapsed), giving the
                        # semaphore a count one above GPU_CONCURRENCY momentarily.  That
                        # is harmless compared to a permanent multi-slot stall.
                        try:
                            gpu_sem.release()
                        except Exception:
                            pass
                        logger.warning(
                            "[GPU] Released semaphore for crashed worker pid=%s"
                            " (no sentinel; exitcode=%s, elapsed=%.0fs).",
                            pid,
                            proc.exitcode,
                            elapsed,
                        )
                    done_pids.append(pid)

                elif (
                    pid in gpu_holders
                    and (now - gpu_acquire_times.get(pid, now)) > _GPU_HOLD_KILL_S
                ):
                    # Worker is alive but has held the GPU semaphore for nearly as
                    # long as the acquire timeout.  Waiting workers will give up in
                    # ~30s, causing a cascade of skipped videos.  Kill this worker
                    # now so the semaphore can be recovered in time.
                    # Root cause: CUDA kernel hangs cannot be interrupted by
                    # SIGALRM, so the worker-internal alarm (840s) never fires.
                    _gpu_held_s = now - gpu_acquire_times.get(pid, now)
                    logger.warning(
                        "[GPU] Worker pid=%s has held GPU semaphore for %.0fs "
                        "(limit=%ss, acquire timeout=%ss) — killing now to unblock "
                        "waiting workers.",
                        pid,
                        _gpu_held_s,
                        _GPU_HOLD_KILL_S,
                        _GPU_ACQUIRE_TIMEOUT_S,
                    )
                    proc.kill()
                    proc.join(timeout=10)
                    # Drain any sentinels buffered before the kill.
                    try:
                        q = info["report_q"]
                        while True:
                            msg = q.get_nowait()
                            if isinstance(msg, dict) and msg.get("_gpu_acquired"):
                                gpu_holders.add(pid)
                                gpu_acquire_times.setdefault(pid, now)
                            elif isinstance(msg, dict) and msg.get("_gpu_released"):
                                gpu_holders.discard(pid)
                                gpu_acquire_times.pop(pid, None)
                    except Exception:
                        pass
                    gpu_acquire_times.pop(pid, None)
                    if pid in gpu_holders:
                        gpu_holders.discard(pid)
                        try:
                            gpu_sem.release()
                        except Exception:
                            pass
                        logger.warning(
                            "[GPU] Recovered semaphore from GPU-hold-timeout worker pid=%s", pid
                        )
                    _collect_result(
                        info,
                        "TIMEOUT",
                        f"killed after holding GPU semaphore for {_gpu_held_s:.0f}s "
                        f"(limit {_GPU_HOLD_KILL_S}s — CUDA kernel likely hung)",
                    )
                    done_pids.append(pid)

                elif _worker_timeout is not None and elapsed > _worker_timeout:
                    logger.warning(
                        "[TIMEOUT] Video %s — killing pid=%s after %.0fs (stuck in cap.read or inference)",
                        info["video_path"],
                        pid,
                        elapsed,
                    )
                    proc.kill()
                    proc.join(timeout=10)
                    # Drain any report-queue messages buffered before the kill.
                    try:
                        q = info["report_q"]
                        while True:
                            msg = q.get_nowait()
                            if isinstance(msg, dict) and msg.get("_gpu_acquired"):
                                gpu_holders.add(pid)
                                gpu_acquire_times.setdefault(pid, now)
                            elif isinstance(msg, dict) and msg.get("_gpu_released"):
                                gpu_holders.discard(pid)
                                gpu_acquire_times.pop(pid, None)
                    except Exception:
                        pass
                    gpu_acquire_times.pop(pid, None)  # clean up on worker exit
                    if pid in gpu_holders:
                        gpu_holders.discard(pid)
                        try:
                            gpu_sem.release()
                        except Exception:
                            pass
                        logger.warning("[GPU] Recovered semaphore from killed worker pid=%s", pid)
                    else:
                        # Killed worker had no sentinel — unconditionally release one
                        # slot for the same reason as the crashed-worker path above.
                        try:
                            gpu_sem.release()
                        except Exception:
                            pass
                        logger.warning(
                            "[GPU] Released semaphore for killed worker pid=%s"
                            " (no sentinel; elapsed=%.0fs).",
                            pid,
                            elapsed,
                        )
                    _collect_result(
                        info,
                        "TIMEOUT",
                        f"killed after {elapsed:.0f}s (exceeded {_worker_timeout:.0f}s limit)",
                    )
                    done_pids.append(pid)

            for pid in done_pids:
                info = active.pop(pid)
                try:
                    info["result_q"].close()
                except Exception:
                    pass
                try:
                    info["result_q"].join_thread()
                except Exception:
                    pass
                try:
                    info["report_q"].close()
                except Exception:
                    pass
                try:
                    info["report_q"].join_thread()
                except Exception:
                    pass
                if not tasks_exhausted:
                    _launch_next()

        # ── Restore TIMEOUT placeholders for cancelled retry tasks ────────
        # If this was a --retry-failed run and the job was killed before all
        # workers finished (SLURM cancel, OOM, Ctrl-C), those tasks never
        # produced a result row.  Without this step they silently vanish from
        # the CSV, making a subsequent --retry-failed find nothing to do.
        if _retry_video_tasks:
            # Any record produced during *this* run (not the pre-loaded resume
            # rows) means the task completed in some form (VIDEO/ERROR/CRASH/TIMEOUT).
            # Normalise hand to "" for None/nan so both sides of the comparison
            # use the same representation and None-handed videos don't appear
            # spuriously in _missing after a successful run.
            def _norm_hand(h):
                if h is None:
                    return ""
                s = str(h).strip()
                return "" if s in ("", "nan", "None") else s

            _new_result_pairs = {
                (str(r.get("video_path", "")), _norm_hand(r.get("hand")))
                for r in logs[len(_resume_logs) :]
            }
            _missing = [
                (vp, h)
                for vp, h in _retry_video_tasks
                if (str(vp), _norm_hand(h)) not in _new_result_pairs
            ]
            if _missing:
                logger.warning(
                    "[RetryFailed] %d task(s) cancelled before completion — "
                    "writing back as TIMEOUT so they can be retried.",
                    len(_missing),
                )
                for vp, h in _missing:
                    logs.append(
                        {
                            "record_type": "TIMEOUT",
                            "video_path": vp,
                            "hand": str(h) if h is not None else "",
                            "error": "job cancelled before worker completed",
                        }
                    )

        df = pd.DataFrame(logs)
        _log_dir = os.path.dirname(self.log_csv_path)
        if _log_dir:
            os.makedirs(_log_dir, exist_ok=True)
        _atomic_csv_flush()
        logger.info("Results saved to %s", self.log_csv_path)
        return df

    def process_hand_videos_sequential(self, n=20, seed=None):
        n = min(n, len(self.vid_score))
        random_rows = self.vid_score.sample(n=n, random_state=seed)
        logs = []
        for idx, (_, row) in enumerate(random_rows.iterrows()):
            video_path = row["video_path"]
            h_parsed = parse_hand_from_path(video_path)
            self.hand_to_track = h_parsed
            logs.append(self._process_video(video_path))
            if (idx + 1) % 10 == 0:
                logger.info("Progress: %d/%d videos processed", idx + 1, n)

        df = pd.DataFrame(logs)
        os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)
        df.to_csv(self.log_csv_path, index=False)
        return df
