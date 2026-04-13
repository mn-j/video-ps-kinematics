"""
ps_kinematics.tracker — Multi-hand offline tracking with spatial binding.

Contains MultiHandOfflineTracker: frame-to-frame spatial matching of hand
tracks using wrist distance, handedness consistency, and rotation activity
score.
"""

import logging
from collections import Counter

import numpy as np

from .utils import (
    BASE_FPS,
    HANDEDNESS_PENALTY_MULT,
    MAX_GAP,
    MAX_JUMP_PER_FRAME,
    TRACK_MATCH_THRESH,
)

try:
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
except ImportError:

    class HandLandmark:
        WRIST = 0
        INDEX_FINGER_MCP = 5
        MIDDLE_FINGER_MCP = 9
        RING_FINGER_MCP = 13
        PINKY_MCP = 17


logger = logging.getLogger(__name__)


class MultiHandOfflineTracker:
    """Offline multi-hand tracking with spatial binding and rotation-based main-track selection."""

    def __init__(
        self,
        expected_label=None,
        match_thresh=TRACK_MATCH_THRESH,
        max_gap=MAX_GAP,
        max_jump_per_frame=MAX_JUMP_PER_FRAME,
        fps=BASE_FPS,
    ):
        self.expected_label = expected_label
        self.match_thresh = match_thresh
        # Scale frame-count / per-frame parameters to the actual video fps.
        # The defaults in utils.py are calibrated for BASE_FPS (25 fps); at a
        # different frame rate (e.g. 50 fps after GIMMVFI interpolation) the
        # same wall-clock tolerance must map to more frames, and the per-frame
        # displacement budget must shrink proportionally.
        _fps = max(1.0, float(fps))
        _scale = _fps / BASE_FPS
        self.max_gap = max(1, round(max_gap * _scale))
        self.max_jump_per_frame = max_jump_per_frame / _scale
        # rot_avg_thresh is stored as a per-frame value calibrated at BASE_FPS;
        # divide by the same scale so the equivalent rad/s threshold is preserved.
        # Read ROT_AVG_THRESH via a live module reference so that any value set
        # by apply_tuning_overrides() is respected.  A bare `from ... import`
        # binding is frozen at import time and cannot be updated by overrides.
        from . import utils as _utils

        self.rot_avg_thresh = _utils.ROT_AVG_THRESH / _scale
        self.tracks = []
        self.next_id = 0

    @staticmethod
    def _top_category(categories):
        return categories[0] if categories else None

    def _confidence_for_expected(self, categories):
        if not categories:
            return 0.0
        if self.expected_label is not None:
            for c in categories:
                if c.category_name == self.expected_label:
                    return float(c.score)
        top = self._top_category(categories)
        return float(top.score) if top is not None else 0.0

    @staticmethod
    def _wrist_xy(lm_arr):
        return float(lm_arr[HandLandmark.WRIST, 0]), float(lm_arr[HandLandmark.WRIST, 1])

    @staticmethod
    def _safe_unit(v, eps=1e-9):
        n = float(np.linalg.norm(v))
        if n < eps:
            return None
        return v / n

    @staticmethod
    def _wrap_to_pi(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _hand_roll_angle(self, lm_arr):
        """Palm roll proxy around wrist->middle_mcp axis."""
        w = lm_arr[HandLandmark.WRIST, :3]
        m = lm_arr[HandLandmark.MIDDLE_FINGER_MCP, :3]
        i = lm_arr[HandLandmark.INDEX_FINGER_MCP, :3]
        p = lm_arr[HandLandmark.PINKY_MCP, :3]

        u = self._safe_unit(m - w)
        if u is None:
            return None
        n = self._safe_unit(np.cross(i - w, p - w))
        if n is None:
            return None
        idx_dir = self._safe_unit(i - w)
        if idx_dir is None:
            return None

        b1 = idx_dir - np.dot(idx_dir, u) * u
        b1 = self._safe_unit(b1)
        if b1 is None:
            return None
        b2 = self._safe_unit(np.cross(u, b1))
        if b2 is None:
            return None

        x = float(np.dot(n, b1))
        y = float(np.dot(n, b2))
        return float(np.arctan2(y, x))

    def _init_rotation_state(self, tr, lm_arr):
        ang = self._hand_roll_angle(lm_arr)
        tr["last_angle_raw"] = ang
        tr["angle_unwrapped"] = 0.0
        tr["angle_min"] = 0.0
        tr["angle_max"] = 0.0
        tr["rot_total"] = 0.0

    def _update_rotation_state(self, tr, lm_arr):
        ang = self._hand_roll_angle(lm_arr)
        if ang is None or tr.get("last_angle_raw") is None:
            tr["last_angle_raw"] = ang
            return
        delta = self._wrap_to_pi(ang - tr["last_angle_raw"])
        tr["last_angle_raw"] = ang
        tr["angle_unwrapped"] += delta
        tr["rot_total"] += abs(delta)
        tr["angle_min"] = min(tr["angle_min"], tr["angle_unwrapped"])
        tr["angle_max"] = max(tr["angle_max"], tr["angle_unwrapped"])

    @staticmethod
    def _rotation_amplitude(tr):
        return float(tr.get("angle_max", 0.0) - tr.get("angle_min", 0.0))

    def _create_track(self, frame_idx, lm_arr, handedness_categories):
        wrist = self._wrist_xy(lm_arr)
        tr = {
            "id": self.next_id,
            "last_frame": frame_idx,
            "last_lm": lm_arr,
            "last_wrist": wrist,
            "vel_wrist": (0.0, 0.0),
            "frames": {frame_idx: lm_arr},
            "conf": {frame_idx: self._confidence_for_expected(handedness_categories)},
            "det_count": 1,
            "handedness_history": [
                handedness_categories[0].category_name if handedness_categories else None
            ],
        }
        self._init_rotation_state(tr, lm_arr)
        self.next_id += 1
        self.tracks.append(tr)

    def _update_track(self, tr, frame_idx, lm_arr, handedness_categories):
        new_wrist = self._wrist_xy(lm_arr)
        old_wrist = tr["last_wrist"]
        dt = max(1, frame_idx - tr["last_frame"])
        tr["vel_wrist"] = ((new_wrist[0] - old_wrist[0]) / dt, (new_wrist[1] - old_wrist[1]) / dt)
        tr["last_lm"] = lm_arr
        tr["last_wrist"] = new_wrist
        tr["last_frame"] = frame_idx
        tr["frames"][frame_idx] = lm_arr
        tr["conf"][frame_idx] = self._confidence_for_expected(handedness_categories)
        tr["det_count"] += 1
        self._update_rotation_state(tr, lm_arr)
        if handedness_categories:
            try:
                tr.setdefault("handedness_history", []).append(
                    handedness_categories[0].category_name
                )
            except Exception:
                tr.setdefault("handedness_history", []).append(None)

    def _active_tracks(self, frame_idx):
        return [tr for tr in self.tracks if (frame_idx - tr["last_frame"]) <= self.max_gap]

    def _predicted_wrist(self, tr, frame_idx):
        dt = frame_idx - tr["last_frame"]
        vx, vy = tr.get("vel_wrist", (0.0, 0.0))
        wx, wy = tr["last_wrist"]
        return (wx + vx * dt, wy + vy * dt)

    @staticmethod
    def _euclid(a, b):
        return float((((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5))

    def associate_frame(self, frame_idx, detections):
        """detections: list of tuples (lm_arr (21,3), handedness_categories)."""
        if not detections:
            return
        active = self._active_tracks(frame_idx)
        if not active:
            for lm_arr, handedness in detections:
                self._create_track(frame_idx, lm_arr, handedness)
            return

        pairs = []
        for ti, tr in enumerate(active):
            pred = self._predicted_wrist(tr, frame_idx)
            dt = max(1, frame_idx - tr["last_frame"])
            gate = max(self.match_thresh, self.max_jump_per_frame * dt)
            for di, (lm_arr, handedness) in enumerate(detections):
                w = self._wrist_xy(lm_arr)
                dist = self._euclid(w, pred)
                if dist > gate:
                    continue
                if self.expected_label is not None and handedness:
                    top = handedness[0].category_name
                    if top != self.expected_label:
                        dist *= HANDEDNESS_PENALTY_MULT
                pairs.append((dist, ti, di))

        if not pairs:
            for lm_arr, handedness in detections:
                self._create_track(frame_idx, lm_arr, handedness)
            return

        pairs.sort(key=lambda x: x[0])
        used_tracks, used_dets = set(), set()
        for dist, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            tr = active[ti]
            lm_arr, handedness = detections[di]
            self._update_track(tr, frame_idx, lm_arr, handedness)

        for di, (lm_arr, handedness) in enumerate(detections):
            if di not in used_dets:
                self._create_track(frame_idx, lm_arr, handedness)

    def choose_main_track(self):
        """Main hand = the one with highest total rotation."""
        if not self.tracks:
            return None

        def avg_conf(tr):
            vals = list(tr["conf"].values())
            return float(np.mean(vals)) if vals else 0.0

        def detected_frames(tr):
            fr = tr.get("frames", {})
            return len(fr) if fr else int(tr.get("det_count", 0))

        def avg_rotation(tr):
            det = max(1, detected_frames(tr))
            return float(tr.get("rot_total", 0.0)) / det

        ranked = sorted(
            self.tracks,
            key=lambda tr: (
                float(tr.get("rot_total", 0.0)),
                self._rotation_amplitude(tr),
                int(tr.get("det_count", 0)),
                avg_conf(tr) if self.expected_label is not None else 0.0,
            ),
            reverse=True,
        )

        for tr in ranked:
            if avg_rotation(tr) >= self.rot_avg_thresh:
                tr["detected_frames"] = detected_frames(tr)
                tr["avg_rot_per_frame"] = avg_rotation(tr)
                hh = tr.get("handedness_history", [])
                if hh:
                    try:
                        most = Counter([h for h in hh if h is not None]).most_common(1)
                        tr["hand_label"] = most[0][0] if most else None
                    except Exception:
                        tr["hand_label"] = None
                else:
                    tr["hand_label"] = None
                return tr
        return None
