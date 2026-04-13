"""Video quality factor analysis and visualisation."""

import os
import warnings as _w

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._constants import KIN_COLS_FOR_VQ, VQ_FACTOR_COLS


def plot_video_quality_factors(
    df: "pd.DataFrame",
    output_dir: str,
    recording_angle_col: str = "recording_angle",
    save_plots: bool = True,
    show_plots: bool = True,
):
    """Analyse and visualise video quality factors and their interaction with
    kinematic features.

    Produces:
    1. Distribution histograms for each VQ factor.
    2. VQ factors by recording angle (if available).
    3. VQ factors by MDS-UPDRS score (to check for clinical confounding).
    4. Correlation heatmap between VQ factors and kinematic features.
    5. Scatter plots of key VQ factors vs Signal Quality.
    """
    vq_dir = os.path.join(output_dir, "video_quality")
    os.makedirs(vq_dir, exist_ok=True)

    # Determine which VQ columns are present
    available_vq = [(c, l, d) for c, l, d in VQ_FACTOR_COLS if c in df.columns]
    if not available_vq:
        print("  (No VQ_ columns present in data \u2014 skipping video quality analysis)")
        return

    # Coerce to numeric
    for col, _, _ in available_vq:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    has_angle = recording_angle_col in df.columns
    has_score = "score_clean" in df.columns

    # -- 1. Summary statistics table --
    print(f"\n{'=' * 60}")
    print("VIDEO QUALITY FACTOR SUMMARY")
    print(f"{'=' * 60}")
    print(f"  N videos: {len(df)}")
    for col, label, _ in available_vq:
        vals = df[col].dropna()
        if vals.empty:
            continue
        print(f"  {label}:")
        print(
            f"    mean={vals.mean():.2f}  median={vals.median():.2f}  "
            f"std={vals.std():.2f}  min={vals.min():.2f}  max={vals.max():.2f}"
        )

    # -- 2. Distribution histograms --
    n_vq = len(available_vq)
    n_cols_h = min(3, n_vq)
    n_rows_h = (n_vq + n_cols_h - 1) // n_cols_h
    fig, axes = plt.subplots(n_rows_h, n_cols_h, figsize=(6 * n_cols_h, 4 * n_rows_h))
    axes_flat = np.array(axes).flatten() if n_vq > 1 else [axes]

    for idx, (col, label, desc) in enumerate(available_vq):
        ax = axes_flat[idx]
        vals = df[col].dropna()
        if vals.empty:
            ax.set_visible(False)
            continue
        with _w.catch_warnings():
            _w.simplefilter("ignore", FutureWarning)
            sns.histplot(vals, bins=30, kde=True, color="teal", edgecolor="white", alpha=0.7, ax=ax)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        stats_txt = f"N={len(vals)} med={vals.median():.1f}"
        ax.text(
            0.98,
            0.95,
            stats_txt,
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.5),
        )

    for idx in range(n_vq, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle("Video Quality Factor Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(vq_dir, "vq_distributions.png"), dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -- 3. VQ factors by recording angle --
    if has_angle:
        angle_vals = df[recording_angle_col].dropna()
        n_labelled = len(angle_vals)
        angle_categories = sorted(angle_vals.unique())
        print(f"\n  Recording angle labels available: {n_labelled}/{len(df)} videos")
        for cat in angle_categories:
            print(f"    {cat}: {int((df[recording_angle_col] == cat).sum())}")

        if n_labelled >= 5 and len(angle_categories) >= 2:
            # Boxplots of each VQ factor split by angle
            n_cols_a = min(3, n_vq)
            n_rows_a = (n_vq + n_cols_a - 1) // n_cols_a
            fig, axes = plt.subplots(n_rows_a, n_cols_a, figsize=(6 * n_cols_a, 4.5 * n_rows_a))
            axes_flat = np.array(axes).flatten() if n_vq > 1 else [axes]
            angle_palette = sns.color_palette("Set2", n_colors=len(angle_categories))

            for idx, (col, label, desc) in enumerate(available_vq):
                ax = axes_flat[idx]
                plot_df = df.dropna(subset=[col, recording_angle_col])
                if plot_df.empty:
                    ax.set_visible(False)
                    continue
                with _w.catch_warnings():
                    _w.simplefilter("ignore", FutureWarning)
                    sns.boxplot(
                        data=plot_df,
                        x=recording_angle_col,
                        y=col,
                        hue=recording_angle_col,
                        palette=angle_palette,
                        legend=False,
                        ax=ax,
                        showfliers=True,
                    )
                sns.stripplot(
                    data=plot_df,
                    x=recording_angle_col,
                    y=col,
                    color="black",
                    alpha=0.3,
                    size=3,
                    ax=ax,
                    jitter=True,
                )
                ax.set_xlabel("Recording Angle", fontsize=9)
                ax.set_ylabel(label, fontsize=9)
                ax.set_title(label, fontsize=10, fontweight="bold")
                for gi, cat in enumerate(sorted(plot_df[recording_angle_col].unique())):
                    n = int((plot_df[recording_angle_col] == cat).sum())
                    ax.annotate(
                        f"n={n}",
                        xy=(gi, ax.get_ylim()[1]),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="gray",
                    )

            for idx in range(n_vq, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            fig.suptitle("Video Quality Factors by Recording Angle", fontsize=13, fontweight="bold")
            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    os.path.join(vq_dir, "vq_by_recording_angle.png"), dpi=150, bbox_inches="tight"
                )
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Signal Quality by recording angle
            if "Signal Quality" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                plot_df = df.dropna(subset=["Signal Quality", recording_angle_col])
                with _w.catch_warnings():
                    _w.simplefilter("ignore", FutureWarning)
                    sns.boxplot(
                        data=plot_df,
                        x=recording_angle_col,
                        y="Signal Quality",
                        hue=recording_angle_col,
                        palette=angle_palette,
                        legend=False,
                        ax=ax,
                        showfliers=True,
                    )
                sns.stripplot(
                    data=plot_df,
                    x=recording_angle_col,
                    y="Signal Quality",
                    color="black",
                    alpha=0.3,
                    size=4,
                    ax=ax,
                    jitter=True,
                )
                ax.set_xlabel("Recording Angle", fontsize=12)
                ax.set_ylabel("Signal Quality (0-1)", fontsize=12)
                ax.set_title("Signal Quality by Recording Angle", fontsize=14)
                for gi, cat in enumerate(sorted(plot_df[recording_angle_col].unique())):
                    n = int((plot_df[recording_angle_col] == cat).sum())
                    med = float(
                        plot_df.loc[plot_df[recording_angle_col] == cat, "Signal Quality"].median()
                    )
                    ax.annotate(
                        f"n={n}\nmed={med:.2f}",
                        xy=(gi, ax.get_ylim()[1]),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="gray",
                    )
                plt.tight_layout()
                if save_plots:
                    plt.savefig(
                        os.path.join(vq_dir, "signal_quality_by_recording_angle.png"),
                        dpi=150,
                        bbox_inches="tight",
                    )
                if show_plots:
                    plt.show()
                else:
                    plt.close()

            # Key kinematic features by recording angle
            kin_by_angle = [
                c for c in KIN_COLS_FOR_VQ if c in df.columns and c != "Signal Quality"
            ]
            if kin_by_angle:
                n_ka = len(kin_by_angle)
                n_cols_ka = min(3, n_ka)
                n_rows_ka = (n_ka + n_cols_ka - 1) // n_cols_ka
                fig, axes = plt.subplots(
                    n_rows_ka, n_cols_ka, figsize=(6 * n_cols_ka, 4.5 * n_rows_ka)
                )
                axes_flat = np.array(axes).flatten() if n_ka > 1 else [axes]
                for idx, kcol in enumerate(kin_by_angle):
                    ax = axes_flat[idx]
                    plot_df = df.dropna(subset=[kcol, recording_angle_col])
                    if plot_df.empty:
                        ax.set_visible(False)
                        continue
                    with _w.catch_warnings():
                        _w.simplefilter("ignore", FutureWarning)
                        sns.boxplot(
                            data=plot_df,
                            x=recording_angle_col,
                            y=kcol,
                            hue=recording_angle_col,
                            palette=angle_palette,
                            legend=False,
                            ax=ax,
                            showfliers=True,
                        )
                    sns.stripplot(
                        data=plot_df,
                        x=recording_angle_col,
                        y=kcol,
                        color="black",
                        alpha=0.3,
                        size=3,
                        ax=ax,
                        jitter=True,
                    )
                    ax.set_xlabel("Recording Angle", fontsize=9)
                    ax.set_ylabel(kcol, fontsize=9)
                    ax.set_title(f"{kcol} by Recording Angle", fontsize=10, fontweight="bold")
                for idx in range(n_ka, len(axes_flat)):
                    axes_flat[idx].set_visible(False)
                fig.suptitle(
                    "Kinematic Features by Recording Angle", fontsize=13, fontweight="bold"
                )
                plt.tight_layout()
                if save_plots:
                    plt.savefig(
                        os.path.join(vq_dir, "kinematics_by_recording_angle.png"),
                        dpi=150,
                        bbox_inches="tight",
                    )
                if show_plots:
                    plt.show()
                else:
                    plt.close()

    # -- 4. VQ factors by MDS-UPDRS score (confounding check) --
    if has_score:
        score_values = sorted(df["score_clean"].dropna().unique())
        if len(score_values) >= 2:
            n_cols_s = min(3, n_vq)
            n_rows_s = (n_vq + n_cols_s - 1) // n_cols_s
            fig, axes = plt.subplots(n_rows_s, n_cols_s, figsize=(6 * n_cols_s, 4.5 * n_rows_s))
            axes_flat = np.array(axes).flatten() if n_vq > 1 else [axes]
            score_palette = sns.color_palette("Blues_d", n_colors=len(score_values))

            for idx, (col, label, desc) in enumerate(available_vq):
                ax = axes_flat[idx]
                plot_df = df.dropna(subset=[col, "score_clean"])
                if plot_df.empty:
                    ax.set_visible(False)
                    continue
                with _w.catch_warnings():
                    _w.simplefilter("ignore", FutureWarning)
                    sns.boxplot(
                        data=plot_df,
                        x="score_clean",
                        y=col,
                        hue="score_clean",
                        palette=score_palette,
                        legend=False,
                        ax=ax,
                        showfliers=True,
                    )
                ax.set_xlabel("MDS-UPDRS Score", fontsize=9)
                ax.set_ylabel(label, fontsize=9)
                ax.set_title(label, fontsize=10, fontweight="bold")

            for idx in range(n_vq, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            fig.suptitle(
                "Video Quality Factors by MDS-UPDRS Score\n"
                "(check for clinical confounding \u2014 should be flat if technical-only)",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    os.path.join(vq_dir, "vq_by_clinical_score.png"), dpi=150, bbox_inches="tight"
                )
            if show_plots:
                plt.show()
            else:
                plt.close()

    # -- 5. Correlation heatmap: VQ factors x kinematic features --
    vq_cols_present = [c for c, _, _ in available_vq if df[c].notna().sum() >= 5]
    kin_cols_present = [c for c in KIN_COLS_FOR_VQ if c in df.columns and df[c].notna().sum() >= 5]

    if vq_cols_present and kin_cols_present:
        corr_df = df[vq_cols_present + kin_cols_present].dropna()
        if len(corr_df) >= 10:
            vq_short = {c: l.split("(")[0].strip() for c, l, _ in available_vq}

            corr_matrix = corr_df[vq_cols_present + kin_cols_present].corr()
            cross_corr = corr_matrix.loc[vq_cols_present, kin_cols_present]
            cross_corr = cross_corr.rename(index=vq_short)

            fig, ax = plt.subplots(
                figsize=(max(8, len(kin_cols_present) * 1.2), max(6, len(vq_cols_present) * 0.6))
            )
            sns.heatmap(
                cross_corr,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                ax=ax,
                linewidths=0.5,
                linecolor="white",
                annot_kws={"fontsize": 9},
            )
            ax.set_title(
                "Correlation: Video Quality Factors vs Kinematic Features",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_ylabel("Video Quality Factor", fontsize=10)
            ax.set_xlabel("Kinematic Feature", fontsize=10)
            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    os.path.join(vq_dir, "vq_kinematic_correlation_heatmap.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
            if show_plots:
                plt.show()
            else:
                plt.close()

    # -- 6. Scatter: key VQ factors vs Signal Quality --
    _key_vq_scatter = [
        "VQ_hand_bbox_area_median_px",
        "VQ_sharpness_median",
        "VQ_luminance_median",
        "VQ_detection_rate",
        "VQ_gap_fraction",
    ]
    scatter_cols = [c for c in _key_vq_scatter if c in df.columns and df[c].notna().sum() >= 5]
    if scatter_cols and "Signal Quality" in df.columns:
        n_sc = len(scatter_cols)
        n_cols_sc = min(3, n_sc)
        n_rows_sc = (n_sc + n_cols_sc - 1) // n_cols_sc
        fig, axes = plt.subplots(n_rows_sc, n_cols_sc, figsize=(5.5 * n_cols_sc, 4.5 * n_rows_sc))
        axes_flat = np.array(axes).flatten() if n_sc > 1 else [axes]

        for idx, col in enumerate(scatter_cols):
            ax = axes_flat[idx]
            plot_df = df.dropna(subset=[col, "Signal Quality"])
            if plot_df.empty:
                ax.set_visible(False)
                continue
            hue_col = (
                recording_angle_col
                if has_angle and recording_angle_col in plot_df.columns
                else ("score_clean" if has_score else None)
            )
            with _w.catch_warnings():
                _w.simplefilter("ignore", FutureWarning)
                sns.scatterplot(
                    data=plot_df,
                    x=col,
                    y="Signal Quality",
                    hue=hue_col,
                    alpha=0.6,
                    s=30,
                    ax=ax,
                    palette="Set2" if hue_col == recording_angle_col else "Blues_d",
                )
            # Trend line
            x_vals = plot_df[col].values
            y_vals = plot_df["Signal Quality"].values
            finite = np.isfinite(x_vals) & np.isfinite(y_vals)
            if finite.sum() >= 5:
                from scipy.stats import spearmanr

                rho, p = spearmanr(x_vals[finite], y_vals[finite])
                z = np.polyfit(x_vals[finite], y_vals[finite], 1)
                x_line = np.linspace(np.min(x_vals[finite]), np.max(x_vals[finite]), 50)
                ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5, alpha=0.7)
                ax.text(
                    0.02,
                    0.02,
                    f"rho={rho:.2f} p={p:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.7),
                )
            label = col
            for c, l, _ in available_vq:
                if c == col:
                    label = l
                    break
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("Signal Quality", fontsize=9)
            ax.set_title(
                f"Signal Quality vs {label.split('(')[0].strip()}", fontsize=10, fontweight="bold"
            )

        for idx in range(n_sc, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.suptitle("Signal Quality vs Video Quality Factors", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_plots:
            plt.savefig(
                os.path.join(vq_dir, "vq_vs_signal_quality_scatter.png"),
                dpi=150,
                bbox_inches="tight",
            )
        if show_plots:
            plt.show()
        else:
            plt.close()

    # -- 7. Print factor-level exclusion analysis --
    if "Signal Quality" in df.columns:
        sq = pd.to_numeric(df["Signal Quality"], errors="coerce")
        sq_median = float(sq.median())
        high_sq = sq >= sq_median
        low_sq = sq < sq_median

        print(f"\n  VQ factor comparison: high-SQ (>={sq_median:.2f}) vs low-SQ videos:")
        print(f"  {'Factor':<35s}  {'High-SQ median':>14s}  {'Low-SQ median':>14s}  {'Ratio':>8s}")
        print(f"  {'-'*35}  {'-'*14}  {'-'*14}  {'-'*8}")
        for col, label, _ in available_vq:
            high_vals = df.loc[high_sq, col].dropna()
            low_vals = df.loc[low_sq, col].dropna()
            if high_vals.empty or low_vals.empty:
                continue
            h_med = float(high_vals.median())
            l_med = float(low_vals.median())
            ratio = h_med / l_med if abs(l_med) > 1e-9 else float("nan")
            short_label = label.split("(")[0].strip()[:35]
            print(f"  {short_label:<35s}  {h_med:>14.2f}  {l_med:>14.2f}  {ratio:>8.2f}")

    print(f"\n  Video quality plots saved to: {vq_dir}")
