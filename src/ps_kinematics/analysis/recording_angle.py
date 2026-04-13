"""Recording angle effect analysis on kinematic features."""

import os
import warnings as _w

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, spearmanr


def run_recording_angle_analysis(
    merged: "pd.DataFrame",
    available_features: list,
    output_dir: str,
    recording_angle_col: str = "recording_angle",
    save_plots: bool = True,
    show_plots: bool = True,
):
    """Analyse and visualise the effect of recording angle on kinematic features.

    Creates a dedicated ``recording_angle/`` sub-folder containing:
    1. Per-feature boxplots split by recording angle.
    2. Combined multi-panel summary of all features by angle.
    3. Angle distribution bar chart.
    4. Score distribution per angle (confounding check).
    5. Kruskal-Wallis + Spearman statistical tests per feature.
    6. Correlation heatmap: recording angle (ordinal) vs kinematic features.
    7. Summary CSV with per-angle descriptive statistics and test results.
    """
    if recording_angle_col not in merged.columns:
        print("  (No recording_angle column \u2014 skipping angle analysis)")
        return

    angle_data = merged.dropna(subset=[recording_angle_col]).copy()
    if len(angle_data) < 5:
        print(f"  (Only {len(angle_data)} videos with angle labels \u2014 skipping angle analysis)")
        return

    angle_dir = os.path.join(output_dir, "recording_angle")
    os.makedirs(angle_dir, exist_ok=True)

    angle_categories = sorted(angle_data[recording_angle_col].unique())
    # Ordinal encoding for correlation analysis (front=0, angled=1, lateral=2, other=3)
    _ANGLE_ORDER = {"front": 0, "angled": 1, "lateral": 2, "other": 3}
    angle_data["_angle_ordinal"] = angle_data[recording_angle_col].map(_ANGLE_ORDER)

    angle_palette = sns.color_palette("Set2", n_colors=len(angle_categories))

    print(f"\n{'=' * 60}")
    print("RECORDING ANGLE EFFECT ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  N videos with angle labels: {len(angle_data)}")
    for cat in angle_categories:
        n = int((angle_data[recording_angle_col] == cat).sum())
        print(f"    {cat}: {n}")

    # -- 1. Angle distribution bar chart --
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = (
        angle_data[recording_angle_col]
        .value_counts()
        .reindex([c for c in ["front", "angled", "lateral", "other"] if c in angle_categories])
        .dropna()
    )
    bars = ax.bar(
        counts.index,
        counts.values,
        color=angle_palette[: len(counts)],
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(val)),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlabel("Recording Angle", fontsize=12)
    ax.set_ylabel("Number of Videos", fontsize=12)
    ax.set_title("Recording Angle Distribution", fontsize=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(angle_dir, "angle_distribution.png"), dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -- 2. Score distribution per angle (confounding check) --
    if "score_clean" in angle_data.columns:
        score_values = sorted(angle_data["score_clean"].dropna().unique())
        if len(score_values) >= 2:
            ct = pd.crosstab(
                angle_data[recording_angle_col], angle_data["score_clean"], margins=True
            )
            print("\n  Score x Angle contingency table:")
            print(ct.to_string())

            fig, ax = plt.subplots(figsize=(10, 6))
            ct_no_margins = ct.drop("All", axis=0).drop("All", axis=1)
            ct_pct = ct_no_margins.div(ct_no_margins.sum(axis=1), axis=0) * 100
            ct_pct.plot(
                kind="bar", stacked=True, ax=ax, colormap="Blues", edgecolor="white", linewidth=0.5
            )
            ax.set_xlabel("Recording Angle", fontsize=12)
            ax.set_ylabel("Percentage of Videos (%)", fontsize=12)
            ax.set_title(
                "MDS-UPDRS Score Distribution by Recording Angle\n"
                "(check for confounding \u2014 should be similar across angles)",
                fontsize=12,
            )
            ax.legend(title="Score", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    os.path.join(angle_dir, "score_distribution_by_angle.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
            if show_plots:
                plt.show()
            else:
                plt.close()

    # -- 3. Per-feature boxplots by recording angle --
    features_for_angle = [
        (col, label, desc, cn)
        for col, label, desc, cn in available_features
        if col in angle_data.columns and angle_data[col].notna().sum() >= 3
    ]
    if not features_for_angle:
        print("  No kinematic features available for angle analysis.")
        return

    stat_rows = []

    for col, ylabel, description, _ in features_for_angle:
        plot_df = angle_data.dropna(subset=[col])
        if len(plot_df) < 3:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        with _w.catch_warnings():
            _w.simplefilter("ignore", FutureWarning)
            sns.boxplot(
                data=plot_df,
                x=recording_angle_col,
                y=col,
                order=[
                    c
                    for c in ["front", "angled", "lateral", "other"]
                    if c in plot_df[recording_angle_col].unique()
                ],
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
            order=[
                c
                for c in ["front", "angled", "lateral", "other"]
                if c in plot_df[recording_angle_col].unique()
            ],
            color="black",
            alpha=0.35,
            size=4,
            ax=ax,
            jitter=True,
        )

        # Annotate n and median per group
        ordered_cats = [
            c
            for c in ["front", "angled", "lateral", "other"]
            if c in plot_df[recording_angle_col].unique()
        ]
        for gi, cat in enumerate(ordered_cats):
            grp = plot_df.loc[plot_df[recording_angle_col] == cat, col].dropna()
            n = len(grp)
            med = float(grp.median()) if n > 0 else float("nan")
            ax.annotate(
                f"n={n}\nmed={med:.2f}",
                xy=(gi, ax.get_ylim()[1]),
                ha="center",
                va="bottom",
                fontsize=9,
                color="gray",
            )

        ax.set_xlabel("Recording Angle", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # Statistical test annotation
        groups = [
            plot_df.loc[plot_df[recording_angle_col] == cat, col].dropna().values
            for cat in ordered_cats
            if len(plot_df.loc[plot_df[recording_angle_col] == cat, col].dropna()) >= 2
        ]
        kw_text = ""
        kw_h, kw_p = float("nan"), float("nan")
        rho, rho_p = float("nan"), float("nan")
        if len(groups) >= 2:
            kw_h, kw_p = kruskal(*groups)
            sig = "***" if kw_p < 0.001 else "**" if kw_p < 0.01 else "*" if kw_p < 0.05 else "ns"
            kw_text = f"Kruskal-Wallis H={kw_h:.2f}, p={kw_p:.3g} ({sig})"
        ordinal_valid = plot_df.dropna(subset=["_angle_ordinal", col])
        if len(ordinal_valid) >= 5:
            rho, rho_p = spearmanr(ordinal_valid["_angle_ordinal"], ordinal_valid[col])
            rho_sig = (
                "***" if rho_p < 0.001 else "**" if rho_p < 0.01 else "*" if rho_p < 0.05 else "ns"
            )
            kw_text += f"\nSpearman rho={rho:+.3f}, p={rho_p:.3g} ({rho_sig})"

        ax.set_title(f"{ylabel} by Recording Angle\n{kw_text}", fontsize=12)
        plt.tight_layout()
        if save_plots:
            safe_col = col.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
            plt.savefig(
                os.path.join(angle_dir, f"boxplot_{safe_col}_by_angle.png"),
                dpi=150,
                bbox_inches="tight",
            )
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Collect per-angle statistics
        row = {"feature": col}
        for cat in ordered_cats:
            grp = plot_df.loc[plot_df[recording_angle_col] == cat, col].dropna()
            row[f"{cat}_n"] = len(grp)
            row[f"{cat}_mean"] = float(grp.mean()) if len(grp) > 0 else float("nan")
            row[f"{cat}_median"] = float(grp.median()) if len(grp) > 0 else float("nan")
            row[f"{cat}_std"] = float(grp.std()) if len(grp) > 1 else float("nan")
        row["kruskal_H"] = kw_h
        row["kruskal_p"] = kw_p
        row["spearman_rho"] = rho
        row["spearman_p"] = rho_p
        stat_rows.append(row)

    # -- 4. Combined multi-panel summary --
    n_feat = len(features_for_angle)
    if n_feat > 1:
        n_cols_p = 3
        n_rows_p = (n_feat + n_cols_p - 1) // n_cols_p
        fig, axes = plt.subplots(n_rows_p, n_cols_p, figsize=(6 * n_cols_p, 4.5 * n_rows_p))
        axes_flat = np.array(axes).flatten() if n_feat > 1 else [axes]

        for idx, (col, ylabel, desc, _) in enumerate(features_for_angle):
            ax = axes_flat[idx]
            plot_df = angle_data.dropna(subset=[col])
            if plot_df.empty:
                ax.set_visible(False)
                continue
            ordered = [
                c
                for c in ["front", "angled", "lateral", "other"]
                if c in plot_df[recording_angle_col].unique()
            ]
            with _w.catch_warnings():
                _w.simplefilter("ignore", FutureWarning)
                sns.boxplot(
                    data=plot_df,
                    x=recording_angle_col,
                    y=col,
                    order=ordered,
                    hue=recording_angle_col,
                    palette=angle_palette,
                    legend=False,
                    ax=ax,
                    showfliers=True,
                )
            ax.set_xlabel("", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(col, fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=8)

        for idx in range(n_feat, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.suptitle("All Kinematic Features by Recording Angle", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_plots:
            plt.savefig(
                os.path.join(angle_dir, "all_features_by_angle_combined.png"),
                dpi=150,
                bbox_inches="tight",
            )
        if show_plots:
            plt.show()
        else:
            plt.close()

    # -- 5. Correlation heatmap: angle (ordinal) vs kinematic features --
    feat_cols = [col for col, _, _, _ in features_for_angle if col in angle_data.columns]
    corr_data = angle_data[["_angle_ordinal"] + feat_cols].dropna()
    if len(corr_data) >= 10 and len(feat_cols) >= 2:
        corr_matrix = corr_data.corr(method="spearman")
        angle_corr = corr_matrix.loc[["_angle_ordinal"], feat_cols]
        angle_corr.index = ["Recording Angle\n(front\u2192lateral)"]

        fig, ax = plt.subplots(figsize=(max(10, len(feat_cols) * 0.8), 3))
        sns.heatmap(
            angle_corr,
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
            "Spearman Correlation: Recording Angle (ordinal) vs Kinematic Features",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        if save_plots:
            plt.savefig(
                os.path.join(angle_dir, "angle_kinematic_correlation_heatmap.png"),
                dpi=150,
                bbox_inches="tight",
            )
        if show_plots:
            plt.show()
        else:
            plt.close()

    # -- 6. Score-stratified angle effect (within-score angle comparison) --
    if "score_clean" in angle_data.columns:
        key_features = [
            col
            for col, _, _, _ in features_for_angle
            if col
            in (
                "Mean Amplitude",
                "Peak Velocity",
                "Mean Velocity",
                "Rhythm (CV %)",
                "Global Velocity",
                "Arm Swing Index",
            )
            and col in angle_data.columns
        ]
        if key_features and len(angle_categories) >= 2:
            for feat in key_features:
                plot_df = angle_data.dropna(subset=[feat, "score_clean", recording_angle_col])
                if len(plot_df) < 10:
                    continue
                scores_present = sorted(plot_df["score_clean"].unique())
                if len(scores_present) < 2:
                    continue
                fig, ax = plt.subplots(figsize=(12, 6))
                with _w.catch_warnings():
                    _w.simplefilter("ignore", FutureWarning)
                    sns.boxplot(
                        data=plot_df,
                        x="score_clean",
                        y=feat,
                        hue=recording_angle_col,
                        palette=angle_palette,
                        ax=ax,
                        showfliers=True,
                    )
                ax.set_xlabel("MDS-UPDRS Score", fontsize=12)
                ax.set_ylabel(feat, fontsize=12)
                ax.set_title(f"{feat} by Score, stratified by Recording Angle", fontsize=13)
                ax.legend(title="Angle", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                if save_plots:
                    safe_col = (
                        feat.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
                    )
                    plt.savefig(
                        os.path.join(angle_dir, f"score_stratified_{safe_col}_by_angle.png"),
                        dpi=150,
                        bbox_inches="tight",
                    )
                if show_plots:
                    plt.show()
                else:
                    plt.close()

    # -- 7. Save summary CSV --
    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        csv_path = os.path.join(angle_dir, "angle_effect_statistics.csv")
        stat_df.to_csv(csv_path, index=False)
        print(f"\n  Angle effect statistics saved to: {csv_path}")

        sig_feats = stat_df[stat_df["kruskal_p"] < 0.05].sort_values("kruskal_p")
        if not sig_feats.empty:
            print("\n  Features significantly affected by recording angle (p < 0.05):")
            for _, r in sig_feats.iterrows():
                print(
                    f"    {r['feature']}: H={r['kruskal_H']:.2f}, "
                    f"p={r['kruskal_p']:.3g}, rho={r['spearman_rho']:+.3f}"
                )
        else:
            print("\n  No features significantly affected by recording angle (p < 0.05).")

    angle_data.drop(columns=["_angle_ordinal"], inplace=True, errors="ignore")
    print(f"\n  Recording angle analysis plots saved to: {angle_dir}")
