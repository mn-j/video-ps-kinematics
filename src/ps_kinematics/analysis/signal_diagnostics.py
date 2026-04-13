"""Signal-quality and confidence diagnostic plots.

Utilities for visualising per-video signal quality scores, MCP landmark
confidence, handedness confidence, and amplitude-vs-cycle-time
relationships.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_json_float_array(value) -> np.ndarray:
    """Parse a JSON-encoded numeric list into a float numpy array."""
    import json

    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return np.array([], dtype=float)
    if isinstance(value, list):
        try:
            return np.array([float(v) if v is not None else np.nan for v in value], dtype=float)
        except Exception:
            return np.array([], dtype=float)
    try:
        parsed = json.loads(str(value))
        return np.array([float(v) if v is not None else np.nan for v in parsed], dtype=float)
    except Exception:
        return np.array([], dtype=float)


def plot_signal_quality_distribution(
    kin_df: "pd.DataFrame",
    output_dir: str,
    threshold: float = 0.0,
    save_plots: bool = True,
    show_plots: bool = True,
):
    """Plot the distribution of per-video Signal Quality scores.

    Produces a histogram + KDE of the ``Signal Quality`` column from the
    tracking log CSV, with a vertical line at the active threshold.
    Also prints summary statistics (mean, median, percentiles) so the
    user can choose an informed threshold.
    """
    if "Signal Quality" not in kin_df.columns:
        print("  (Signal Quality column not present in CSV — skipping distribution plot)")
        return

    sq = pd.to_numeric(kin_df["Signal Quality"], errors="coerce").dropna()
    if sq.empty:
        print("  (No valid Signal Quality values — skipping distribution plot)")
        return

    print(f"\n{'=' * 60}")
    print("SIGNAL QUALITY DISTRIBUTION")
    print(f"{'=' * 60}")
    print(f"  N videos with score : {len(sq)}")
    print(f"  Mean                : {sq.mean():.3f}")
    print(f"  Median              : {sq.median():.3f}")
    print(f"  Std                 : {sq.std():.3f}")
    print(f"  Min / Max           : {sq.min():.3f} / {sq.max():.3f}")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct:<2d}                : {sq.quantile(pct / 100):.3f}")
    if threshold > 0.0:
        n_pass = int((sq >= threshold).sum())
        print(
            f"  >= threshold {threshold:.2f}  : {n_pass}/{len(sq)} "
            f"({100 * n_pass / len(sq):.1f}%)"
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(sq, bins=30, kde=True, color="steelblue", edgecolor="white", alpha=0.7, ax=ax)
    ax.set_xlabel("Signal Quality Score", fontsize=12)
    ax.set_ylabel("Number of Videos", fontsize=12)
    ax.set_title("Per-Video Signal Quality Distribution", fontsize=14)
    ax.set_xlim(0, 1)

    if threshold > 0.0:
        ax.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold = {threshold:.2f}",
        )
        n_pass = int((sq >= threshold).sum())
        ax.legend(title=f"{n_pass}/{len(sq)} pass", fontsize=10)

    # Add summary annotation
    stats_text = (
        f"Mean={sq.mean():.3f}  Median={sq.median():.3f}\n" f"Std={sq.std():.3f}  N={len(sq)}"
    )
    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    if save_plots:
        plot_path = os.path.join(output_dir, "signal_quality_distribution.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        # print(f"  Saved: {plot_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_confidence_video_averages(
    merged_video_df: "pd.DataFrame",
    output_dir: str,
    save_plots: bool = True,
    show_plots: bool = True,
):
    """Plot per-video mean MCP landmark confidence (stability proxy) by score."""
    required = {
        "score_clean",
        "conf_index_mcp_series",
        "conf_middle_mcp_series",
        "conf_ring_mcp_series",
        "conf_pinky_mcp_series",
        "conf_mcp_used_mask_series",
    }
    if not required.issubset(set(merged_video_df.columns)):
        print("  (MCP confidence series columns not present — skipping confidence plots)")
        return

    rows = []
    for _, row in merged_video_df.iterrows():
        idx = parse_json_float_array(row.get("conf_index_mcp_series"))
        mid = parse_json_float_array(row.get("conf_middle_mcp_series"))
        ring = parse_json_float_array(row.get("conf_ring_mcp_series"))
        pinky = parse_json_float_array(row.get("conf_pinky_mcp_series"))
        used_mask = parse_json_float_array(row.get("conf_mcp_used_mask_series"))
        if min(len(idx), len(mid), len(ring), len(pinky)) == 0:
            continue

        min_len = min(len(idx), len(mid), len(ring), len(pinky))
        idx = idx[:min_len]
        mid = mid[:min_len]
        ring = ring[:min_len]
        pinky = pinky[:min_len]
        if len(used_mask) < min_len:
            mask = np.ones(min_len, dtype=bool)
        else:
            mask = used_mask[:min_len] > 0.5

        def _safe_mean(arr: np.ndarray, sel: np.ndarray) -> float:
            vals = arr[sel] if sel.any() else arr
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return float("nan")
            return float(np.mean(vals))

        row_dict = {
            "score_clean": row.get("score_clean"),
            "video_ids": row.get("video_ids", row.get("ids")),
            "conf_mean_index_mcp": _safe_mean(idx, mask),
            "conf_mean_middle_mcp": _safe_mean(mid, mask),
            "conf_mean_ring_mcp": _safe_mean(ring, mask),
            "conf_mean_pinky_mcp": _safe_mean(pinky, mask),
        }
        row_dict["conf_mean_mcp_overall"] = float(
            np.nanmean(
                [
                    row_dict["conf_mean_index_mcp"],
                    row_dict["conf_mean_middle_mcp"],
                    row_dict["conf_mean_ring_mcp"],
                    row_dict["conf_mean_pinky_mcp"],
                ]
            )
        )
        rows.append(row_dict)

    conf_df = pd.DataFrame(rows)
    if conf_df.empty:
        print("  (No valid MCP confidence data parsed — skipping confidence plots)")
        return

    conf_csv = os.path.join(output_dir, "video_level_mcp_confidence_means.csv")
    conf_df.to_csv(conf_csv, index=False)
    print(f"  Saved video-level MCP confidence summary: {conf_csv}")

    # Plot 1: overall mean MCP confidence by clinical score
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=conf_df, x="score_clean", y="conf_mean_mcp_overall", color="lightsteelblue", ax=ax
    )
    sns.stripplot(
        data=conf_df,
        x="score_clean",
        y="conf_mean_mcp_overall",
        color="black",
        alpha=0.35,
        size=3,
        ax=ax,
    )
    ax.set_xlabel("MDS-UPDRS score")
    ax.set_ylabel("Mean MCP confidence (stability proxy)")
    ax.set_title("Per-video mean MCP landmark confidence by score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_plots:
        p1 = os.path.join(output_dir, "confidence_mean_mcp_by_score.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Plot 2: per-keypoint mean confidence by score
    key_cols = [
        "conf_mean_index_mcp",
        "conf_mean_middle_mcp",
        "conf_mean_ring_mcp",
        "conf_mean_pinky_mcp",
    ]
    key_label_map = {
        "conf_mean_index_mcp": "Index MCP",
        "conf_mean_middle_mcp": "Middle MCP",
        "conf_mean_ring_mcp": "Ring MCP",
        "conf_mean_pinky_mcp": "Pinky MCP",
    }
    long_conf = conf_df.melt(
        id_vars=["score_clean", "video_ids"],
        value_vars=key_cols,
        var_name="keypoint",
        value_name="mean_confidence",
    )
    long_conf["keypoint"] = long_conf["keypoint"].map(key_label_map)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=long_conf,
        x="score_clean",
        y="mean_confidence",
        hue="keypoint",
        ax=ax,
    )
    ax.set_xlabel("MDS-UPDRS score")
    ax.set_ylabel("Mean keypoint confidence (stability proxy)")
    ax.set_title("Per-video mean MCP landmark confidence by keypoint and score")
    ax.set_ylim(0, 1)
    ax.legend(title="Keypoint", fontsize=9)
    plt.tight_layout()
    if save_plots:
        p2 = os.path.join(output_dir, "confidence_keypoint_means_by_score.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_handedness_confidence(
    merged_video_df: "pd.DataFrame",
    output_dir: str,
    save_plots: bool = True,
    show_plots: bool = True,
):
    """Plot per-video MediaPipe handedness confidence (avg_conf) by score.

    ``avg_conf`` is the mean per-frame handedness classification confidence
    returned by the MediaPipe Hand Landmarker — the only true model-level
    confidence score available per detection.
    """
    if "avg_conf" not in merged_video_df.columns or "score_clean" not in merged_video_df.columns:
        print(
            "  (avg_conf or score_clean column not present — skipping handedness confidence plots)"
        )
        return

    df = merged_video_df[["score_clean", "avg_conf"]].copy()
    df["avg_conf"] = pd.to_numeric(df["avg_conf"], errors="coerce")
    df = df.dropna(subset=["avg_conf"])
    if df.empty:
        print("  (No valid avg_conf values — skipping handedness confidence plots)")
        return

    # --- Plot 1: box plot by score ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="score_clean", y="avg_conf", color="lightsteelblue", ax=ax)
    sns.stripplot(data=df, x="score_clean", y="avg_conf", color="black", alpha=0.35, size=3, ax=ax)
    ax.set_xlabel("MDS-UPDRS score")
    ax.set_ylabel("Mean handedness confidence")
    ax.set_title("Per-video MediaPipe handedness confidence by score")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_plots:
        p = os.path.join(output_dir, "handedness_confidence_by_score.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # --- Plot 2: histogram of avg_conf across all videos ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["avg_conf"].values, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Mean handedness confidence")
    ax.set_ylabel("Number of videos")
    ax.set_title("Distribution of per-video handedness confidence")
    ax.axvline(
        df["avg_conf"].median(),
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Median = {df['avg_conf'].median():.3f}",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_plots:
        p = os.path.join(output_dir, "handedness_confidence_distribution.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()


# ============================================================
# Amplitude vs cycle-time scatter
# ============================================================


def plot_amplitude_vs_cycle_time_scatter(
    merged_video_level: "pd.DataFrame",
    output_dir: str,
    score_column: str = "ProS",
    save_plots: bool = True,
    show_plots: bool = False,
) -> None:
    """Scatter plot of per-cycle amplitude vs cycle duration.

    Each point represents one detected cycle from one video recording.
    Cycle duration is derived as the inter-peak interval (``np.diff`` of
    ``cycle_peak_times_s``).  Colors encode the MDS-UPDRS score.

    No pipeline changes are required: ``cycle_amplitudes_deg`` and
    ``cycle_peak_times_s`` are already stored as JSON arrays in the
    tracking-log CSV and are present in ``merged_video_level``.

    Parameters
    ----------
    merged_video_level : pd.DataFrame
        Pre-aggregation DataFrame (one row per video) containing
        ``cycle_amplitudes_deg``, ``cycle_peak_times_s``, and
        ``score_clean``.
    output_dir : str
        Directory to save the plot.
    score_column : str
        Label for the score column used in the plot title.
    save_plots : bool
        Whether to save the figure to disk.
    show_plots : bool
        Whether to display the figure interactively.
    """
    needed = {"cycle_amplitudes_deg", "cycle_peak_times_s", "score_clean"}
    missing = needed - set(merged_video_level.columns)
    if missing:
        print(f"  ⚠ Skipping amplitude-vs-cycle-time scatter: missing columns {missing}")
        return

    rows = []
    for _, row in merged_video_level.iterrows():
        amps = parse_json_float_array(row["cycle_amplitudes_deg"])
        peaks = parse_json_float_array(row["cycle_peak_times_s"])
        score = row["score_clean"]
        if len(amps) < 2 or len(peaks) < 2:
            continue
        intervals = np.diff(peaks)  # N-1 inter-peak intervals = cycle durations
        n = min(len(amps) - 1, len(intervals))
        for i in range(n):
            if np.isfinite(amps[i]) and np.isfinite(intervals[i]) and intervals[i] > 0:
                rows.append(
                    {
                        "amplitude_deg": float(amps[i]),
                        "cycle_duration_s": float(intervals[i]),
                        "score_clean": int(score),
                    }
                )

    if not rows:
        print("  ⚠ No valid per-cycle data for amplitude-vs-cycle-time scatter.")
        return

    cyc_df = pd.DataFrame(rows)
    score_vals = sorted(cyc_df["score_clean"].unique())
    palette = sns.color_palette("Blues_d", n_colors=len(score_vals))
    color_map = {s: palette[i] for i, s in enumerate(score_vals)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for score in score_vals:
        sub = cyc_df[cyc_df["score_clean"] == score]
        ax.scatter(
            sub["cycle_duration_s"],
            sub["amplitude_deg"],
            c=[color_map[score]],
            label=f"Score {score} (n={len(sub)})",
            alpha=0.4,
            s=20,
            edgecolors="none",
        )

    valid = cyc_df.dropna(subset=["cycle_duration_s", "amplitude_deg"])
    if len(valid) >= 3:
        try:
            from scipy.stats import pearsonr

            r, p = pearsonr(valid["cycle_duration_s"], valid["amplitude_deg"])
            ax.annotate(
                f"Pearson r = {r:.3f}  (p = {p:.3g}, n = {len(valid)} cycles)",
                xy=(0.02, 0.97),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )
        except ImportError:
            pass

    ax.set_xlabel("Cycle Duration (s)", fontsize=12)
    ax.set_ylabel("Amplitude (degrees)", fontsize=12)
    ax.set_title(
        f"Per-Cycle Amplitude vs Cycle Duration\n(colored by MDS-UPDRS {score_column} score)",
        fontsize=13,
    )
    ax.legend(title="Score", fontsize=9, title_fontsize=9, markerscale=1.5)
    plt.tight_layout()

    if save_plots:
        plot_path = os.path.join(output_dir, "scatter_amplitude_vs_cycle_duration.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
