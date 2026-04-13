"""
ps_kinematics.plotting — Plotting helpers (wrap OpenCV/matplotlib calls).

Optional dependency guards are preserved: if matplotlib is not available,
the module still imports but plotting functions become no-ops or return None.
"""

import numpy as np

try:
    import cv2

    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import matplotlib.pyplot as plt  # noqa: F401
    import matplotlib.ticker as ticker  # noqa: F401

    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    import seaborn as sns  # noqa: F401

    SEABORN_OK = True
except ImportError:
    SEABORN_OK = False


# ============================================================
# OpenCV-based plot rendering (used in video overlays)
# ============================================================


def _nice_limits(y, pad_ratio=0.08):
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if y.size < 5:
        return -1.0, 1.0
    lo = float(np.nanpercentile(y, 2))
    hi = float(np.nanpercentile(y, 98))
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad


def _draw_axis_frame(img, rect, title, xlabel, ylabel):
    if not CV2_OK:
        return
    x0, y0, x1, y1 = rect
    cv2.putText(
        img, title, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
    )
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
    cv2.putText(
        img, ylabel, (x0 + 6, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
    )
    cv2.putText(
        img, xlabel, (x1 - 140, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
    )


def _draw_timeseries(
    img,
    x,
    y,
    cursor_x,
    rect,
    title,
    xlabel="Time (s)",
    ylabel="Angle (deg)",
    line_color=(40, 120, 220),
    cursor_color=(0, 0, 255),
):
    if not CV2_OK:
        return
    x0, y0, x1, y1 = rect
    _draw_axis_frame(img, rect, title, xlabel, ylabel)

    left = x0 + 60
    right = x1 - 10
    top = y0 + 26
    bottom = y1 - 25
    plot_w = max(10, right - left)
    plot_h = max(10, bottom - top)
    cv2.rectangle(img, (left, top), (left + plot_w, top + plot_h), (0, 0, 0), 1)

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0
    y_min, y_max = _nice_limits(y)

    def x_to_px(xv):
        return int(left + (float(xv) - x_min) / (x_max - x_min) * plot_w)

    def y_to_py(yv):
        return int(top + (1.0 - (float(yv) - y_min) / max(1e-9, (y_max - y_min))) * plot_h)

    for frac, lab in [(0.0, y_min), (0.5, (y_min + y_max) / 2.0), (1.0, y_max)]:
        yy = int(top + (1.0 - frac) * plot_h)
        cv2.line(img, (left - 5, yy), (left, yy), (0, 0, 0), 1)
        cv2.putText(
            img,
            f"{lab:.1f}",
            (x0 + 5, yy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    pts = []
    for xv, yv in zip(x, y):
        if np.isnan(xv) or np.isnan(yv):
            continue
        pts.append((x_to_px(xv), y_to_py(yv)))
    if len(pts) >= 2:
        cv2.polylines(
            img, [np.array(pts, dtype=np.int32)], isClosed=False, color=line_color, thickness=2
        )

    cx = x_to_px(np.clip(cursor_x, x_min, x_max))
    cv2.line(img, (cx, top), (cx, top + plot_h), cursor_color, 2)


def _draw_cycle_bars(
    img,
    cycle_nums,
    amps,
    trend,
    cursor_cycle,
    rect,
    title,
    xlabel="Cycle Number",
    ylabel="Amplitude (deg)",
    bar_color=(40, 180, 40),
    trend_color=(0, 0, 255),
    cursor_color=(0, 0, 255),
):
    if not CV2_OK:
        return
    x0, y0, x1, y1 = rect
    _draw_axis_frame(img, rect, title, xlabel, ylabel)

    left = x0 + 60
    right = x1 - 10
    top = y0 + 26
    bottom = y1 - 25
    plot_w = max(10, right - left)
    plot_h = max(10, bottom - top)
    cv2.rectangle(img, (left, top), (left + plot_w, top + plot_h), (0, 0, 0), 1)

    if cycle_nums is None or len(cycle_nums) == 0:
        cx = int(left)
        cv2.line(img, (cx, top), (cx, top + plot_h), cursor_color, 2)
        return

    cycle_nums = np.asarray(cycle_nums, float)
    amps = np.asarray(amps, float)
    trend = np.asarray(trend, float) if trend is not None else None

    x_min = 0.0
    x_max = float(max(1.0, np.nanmax(cycle_nums)))
    _, y_max = _nice_limits(amps)
    y_min = 0.0

    def x_to_px(xv):
        return int(left + (float(xv) - x_min) / max(1e-9, (x_max - x_min)) * plot_w)

    def y_to_py(yv):
        return int(top + (1.0 - (float(yv) - y_min) / max(1e-9, (y_max - y_min))) * plot_h)

    for frac, lab in [(0.0, y_min), (0.5, (y_min + y_max) / 2.0), (1.0, y_max)]:
        yy = int(top + (1.0 - frac) * plot_h)
        cv2.line(img, (left - 5, yy), (left, yy), (0, 0, 0), 1)
        cv2.putText(
            img,
            f"{lab:.1f}",
            (x0 + 5, yy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    n = len(cycle_nums)
    if n > 0:
        bar_w = max(2, int(plot_w / max(10, n * 1.2)))
        for cn, a in zip(cycle_nums, amps):
            if np.isnan(a):
                continue
            cx = x_to_px(cn)
            x_left = int(cx - bar_w // 2)
            x_right = int(cx + bar_w // 2)
            y_base = y_to_py(y_min)
            y_top = max(top, y_to_py(a))
            cv2.rectangle(img, (x_left, y_top), (x_right, y_base), bar_color, -1)

    if trend is not None and len(trend) == len(cycle_nums):
        pts = []
        for cn, tv in zip(cycle_nums, trend):
            if np.isnan(tv):
                continue
            pts.append((x_to_px(cn), y_to_py(tv)))
        if len(pts) >= 2:
            cv2.polylines(
                img, [np.array(pts, dtype=np.int32)], isClosed=False, color=trend_color, thickness=2
            )

    cursor_cycle = float(np.clip(cursor_cycle, x_min, x_max))
    cx = x_to_px(cursor_cycle)
    cv2.line(img, (cx, top), (cx, top + plot_h), cursor_color, 2)


def render_two_plot_panel(time_s, filtered_deg, metrics, current_frame_idx, fps, panel_h, panel_w):
    """Create right-side panel with 2 live plots and synced scrubbers."""
    if not CV2_OK:
        return np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

    bg = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

    header_h = 32
    cv2.putText(
        bg,
        "Kinematics (synced)",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.line(bg, (0, header_h), (panel_w, header_h), (0, 0, 0), 1)

    avail_h = panel_h - header_h
    slot_h = avail_h // 2

    t_cursor = current_frame_idx / float(fps)

    rect1 = (0, header_h, panel_w, header_h + slot_h)
    _draw_timeseries(
        bg,
        x=time_s,
        y=filtered_deg,
        cursor_x=t_cursor,
        rect=rect1,
        title="Filtered Hand Rotation",
        xlabel="Time (s)",
        ylabel="Angle (deg)",
    )

    rect2 = (0, header_h + slot_h, panel_w, panel_h)

    if metrics is None:
        _draw_cycle_bars(
            bg,
            cycle_nums=np.array([], dtype=float),
            amps=np.array([], dtype=float),
            trend=None,
            cursor_cycle=0.0,
            rect=rect2,
            title="Amplitude Decrement (Fatigue)",
            xlabel="Cycle Number",
            ylabel="Amplitude (deg)",
        )
        return bg

    amps = metrics["amplitudes"]
    trend = metrics["trend_line"]
    cycle_nums = np.arange(1, len(amps) + 1, dtype=float)

    peak_times = metrics["peak_times"]
    cycles_done = int(np.searchsorted(peak_times, t_cursor, side="right"))
    cursor_cycle = float(cycles_done)

    _draw_cycle_bars(
        bg,
        cycle_nums=cycle_nums,
        amps=amps,
        trend=trend,
        cursor_cycle=cursor_cycle,
        rect=rect2,
        title="Amplitude Decrement (Fatigue)",
        xlabel="Cycle Number",
        ylabel="Amplitude (deg)",
    )
    return bg
