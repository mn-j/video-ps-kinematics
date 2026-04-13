"""Medication effect analysis: strict Off->On filename pairing."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ps_kinematics.io import normalize_med_state
from ps_kinematics.analysis._demographics import (
    age_group_labels,
    normalize_gender_series,
    subgroup_delta_stats,
    ttest_rel_p,
    wilcoxon_p,
)


def medication_neutral_filename_key(video_path: str):
    """Build medication-neutral key by replacing On/Off token in filename."""
    if video_path is None or (isinstance(video_path, float) and np.isnan(video_path)):
        return None
    name = os.path.basename(str(video_path))
    if not name:
        return None
    neutral = re.sub(
        r"(?<![A-Za-z])(On|Off)(?![A-Za-z])",
        "__MED__",
        name,
        count=1,
        flags=re.IGNORECASE,
    )
    if neutral == name:
        return None
    return neutral.lower()


def run_medicine_effect_report(
    merged: "pd.DataFrame",
    available_features,
    output_dir: str,
    save_plots: bool,
    show_plots: bool,
) -> dict:
    """Measure medication effect by strict Off->On filename pairing.

    Mapping rule:
      - video filenames must match exactly after replacing one On/Off token.
      - each key must map to exactly one Off video and one On video.

    Effect size per feature:
      effect_size = avg(x) / std(x), where x = (On - Off) across paired videos.
    """
    required = ["video_ids", "medication_state"]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        print(f"  (Medicine effect report skipped: missing columns {missing})")
        return {"error": f"missing columns: {missing}"}

    if "video_path" not in merged.columns:
        print("  (Medicine effect report skipped: video_path not found in merged data)")
        return {"error": "missing video_path column"}

    med_df = merged.copy()
    med_df["medication_state"] = med_df["medication_state"].apply(normalize_med_state)
    med_df = med_df[med_df["medication_state"].isin(["Off", "On"])].copy()
    med_df["med_pair_key"] = med_df["video_path"].apply(medication_neutral_filename_key)
    med_df = med_df.dropna(subset=["med_pair_key"]).copy()

    # --- Demographics: Gender / age_group for subgroup analysis ---
    if "Gender" in med_df.columns:
        med_df["gender_norm"] = normalize_gender_series(med_df["Gender"])
    if "age" in med_df.columns:
        med_df["age_group"] = age_group_labels(med_df["age"])
    _med_demog_cols = [c for c in ["gender_norm", "age_group"] if c in med_df.columns]
    _has_med_demog = len(_med_demog_cols) > 0

    if med_df.empty:
        print("  (Medicine effect report: no Off/On filename pairs found)")
        return {
            "n_rows_input": int(len(merged)),
            "n_candidate_rows": 0,
            "n_pair_keys_total": 0,
            "n_pair_keys_mapped": 0,
            "n_pair_keys_unmatched_off_only": 0,
            "n_pair_keys_unmatched_on_only": 0,
            "n_pair_keys_excluded_multi_off_or_on": 0,
            "mapping_csv": None,
            "paired_csv": None,
            "summary_csv": None,
        }

    key_cols = ["med_pair_key"]
    if "ids" in med_df.columns:
        key_cols.append("ids")
    if "visit" in med_df.columns:
        key_cols.append("visit")
    if "hand" in med_df.columns:
        key_cols.append("hand")

    # Collapse repeated rows for same video/med state.
    feature_cols = [
        col for col, _label, _desc, _can_norm in available_features if col in med_df.columns
    ]
    has_score = "score_clean" in med_df.columns
    agg_dict = {"video_ids": "first", "video_path": "first"}
    if has_score:
        agg_dict["score_clean"] = "first"
    for c in feature_cols:
        agg_dict[c] = "mean"
    for _dc in _med_demog_cols:
        agg_dict[_dc] = "first"
    collapsed = med_df.groupby(
        key_cols + ["medication_state", "video_ids"], dropna=False, as_index=False
    ).agg(agg_dict)

    pair_counts = (
        collapsed.groupby(key_cols + ["medication_state"], dropna=False)["video_ids"]
        .nunique(dropna=True)
        .reset_index(name="n_videos")
    )

    # Reshape counts for mapping diagnostics.
    counts_wide = pair_counts.pivot_table(
        index=key_cols,
        columns="medication_state",
        values="n_videos",
        aggfunc="sum",
    ).reset_index()
    counts_wide.columns.name = None
    if "Off" not in counts_wide.columns:
        counts_wide["Off"] = 0
    if "On" not in counts_wide.columns:
        counts_wide["On"] = 0

    off_only = counts_wide[(counts_wide["Off"] >= 1) & (counts_wide["On"] == 0)]
    on_only = counts_wide[(counts_wide["On"] >= 1) & (counts_wide["Off"] == 0)]
    multi_side = counts_wide[(counts_wide["Off"] > 1) | (counts_wide["On"] > 1)]
    valid_keys = counts_wide[(counts_wide["Off"] == 1) & (counts_wide["On"] == 1)].copy()

    if valid_keys.empty:
        print("  (Medicine effect report: no valid 1:1 Off->On mappings)")
        return {
            "n_rows_input": int(len(merged)),
            "n_candidate_rows": int(len(med_df)),
            "n_pair_keys_total": int(len(counts_wide)),
            "n_pair_keys_mapped": 0,
            "n_pair_keys_unmatched_off_only": int(len(off_only)),
            "n_pair_keys_unmatched_on_only": int(len(on_only)),
            "n_pair_keys_excluded_multi_off_or_on": int(len(multi_side)),
            "mapping_csv": None,
            "paired_csv": None,
            "summary_csv": None,
        }

    valid_pairs = collapsed.merge(valid_keys[key_cols], on=key_cols, how="inner")

    # Wide paired table with Off and On columns.
    off_df = valid_pairs[valid_pairs["medication_state"] == "Off"].copy()
    on_df = valid_pairs[valid_pairs["medication_state"] == "On"].copy()

    # Demographics lookup (one row per pairing key — demographics constant per patient).
    _med_demog_lookup = (
        off_df[key_cols + _med_demog_cols].drop_duplicates(subset=key_cols)
        if _has_med_demog
        else None
    )

    mapping_cols = key_cols + ["video_ids", "video_path"]
    off_map = off_df[mapping_cols].rename(
        columns={
            "video_ids": "off_video_id",
            "video_path": "off_video_path",
        }
    )
    on_map = on_df[mapping_cols].rename(
        columns={
            "video_ids": "on_video_id",
            "video_path": "on_video_path",
        }
    )
    mapping_df = off_map.merge(on_map, on=key_cols, how="inner")

    paired_df = mapping_df.copy()

    # Add MDS-UPDRS scores per medication state to the paired table.
    if has_score and "score_clean" in off_df.columns:
        off_score = (
            off_df[key_cols + ["score_clean"]]
            .drop_duplicates(subset=key_cols)
            .rename(columns={"score_clean": "score_clean__off"})
        )
        on_score = (
            on_df[key_cols + ["score_clean"]]
            .drop_duplicates(subset=key_cols)
            .rename(columns={"score_clean": "score_clean__on"})
        )
        paired_df = paired_df.merge(off_score, on=key_cols, how="left")
        paired_df = paired_df.merge(on_score, on=key_cols, how="left")
        paired_df["score_clean__delta_on_minus_off"] = pd.to_numeric(
            paired_df["score_clean__on"], errors="coerce"
        ) - pd.to_numeric(paired_df["score_clean__off"], errors="coerce")

    summary_rows = []

    for feat in feature_cols:
        off_feat = off_df[key_cols + [feat]].rename(columns={feat: f"{feat}__off"})
        on_feat = on_df[key_cols + [feat]].rename(columns={feat: f"{feat}__on"})
        feat_pair = off_feat.merge(on_feat, on=key_cols, how="inner")
        feat_pair[f"{feat}__delta_on_minus_off"] = (
            feat_pair[f"{feat}__on"] - feat_pair[f"{feat}__off"]
        )
        paired_df = paired_df.merge(feat_pair, on=key_cols, how="left")

        _off_num = pd.to_numeric(feat_pair[f"{feat}__off"], errors="coerce")
        _on_num = pd.to_numeric(feat_pair[f"{feat}__on"], errors="coerce")
        _valid_mask = _off_num.notna() & _on_num.notna()
        deltas = (_on_num - _off_num)[_valid_mask]
        effect_size = np.nan
        if len(deltas) >= 2:
            std = float(deltas.std(ddof=1))
            if std > 0:
                effect_size = float(deltas.mean() / std)

        _med_row = {
            "feature": feat,
            "n_pairs": int(len(deltas)),
            "off_mean": float(_off_num[_valid_mask].mean()) if len(deltas) else np.nan,
            "on_mean": float(_on_num[_valid_mask].mean()) if len(deltas) else np.nan,
            "delta_mean_on_minus_off": float(deltas.mean()) if len(deltas) else np.nan,
            "delta_median_on_minus_off": float(deltas.median()) if len(deltas) else np.nan,
            "delta_std_on_minus_off": float(deltas.std(ddof=1)) if len(deltas) >= 2 else np.nan,
            "medicine_effect_size_mean_over_std": effect_size,
            "wilcoxon_p": wilcoxon_p(deltas),
            "ttest_rel_p": ttest_rel_p(
                _off_num[_valid_mask].reset_index(drop=True),
                _on_num[_valid_mask].reset_index(drop=True),
            ),
        }

        # --- Per-gender and per-age-group subgroup analysis ---
        if _has_med_demog and _med_demog_lookup is not None and len(feat_pair) > 0:
            _fp_valid = feat_pair[_valid_mask].merge(_med_demog_lookup, on=key_cols, how="left")
            _d_indexed = pd.to_numeric(
                _fp_valid[f"{feat}__on"] - _fp_valid[f"{feat}__off"], errors="coerce"
            )
            for _dc in _med_demog_cols:
                if _dc in _fp_valid.columns:
                    _med_row.update(subgroup_delta_stats(_d_indexed, _fp_valid[_dc], _dc))

        summary_rows.append(_med_row)

        if len(feat_pair) > 0 and save_plots:
            # Paired Off->On trajectory plot.
            fig, ax = plt.subplots(figsize=(7, 5))
            for _, row in feat_pair.iterrows():
                off_val = row[f"{feat}__off"]
                on_val = row[f"{feat}__on"]
                if pd.notna(off_val) and pd.notna(on_val):
                    ax.plot([0, 1], [off_val, on_val], color="gray", alpha=0.35, linewidth=1)
            off_mean = pd.to_numeric(feat_pair[f"{feat}__off"], errors="coerce").mean()
            on_mean = pd.to_numeric(feat_pair[f"{feat}__on"], errors="coerce").mean()
            ax.plot(
                [0, 1],
                [off_mean, on_mean],
                color="tab:blue",
                marker="o",
                linewidth=2.5,
                label="Cohort mean",
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Off", "On"])
            ax.set_ylabel(feat)
            ax.set_title(f"Medication pairing: {feat} (Off vs On)")
            ax.legend(loc="best")
            plt.tight_layout()

            safe_col = feat.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
            pair_plot_path = os.path.join(output_dir, f"medicine_pairing_{safe_col}.png")
            plt.savefig(pair_plot_path, dpi=150, bbox_inches="tight")
            # print(f"  Saved: {pair_plot_path}")
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Delta distribution plot.
            delta_vals = deltas
            if len(delta_vals) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(delta_vals, bins=20, kde=True, color="mediumpurple", ax=ax)
                ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
                ax.set_xlabel(f"{feat}: On - Off")
                ax.set_ylabel("Count")
                ax.set_title(f"Medication effect distribution: {feat}")
                plt.tight_layout()
                delta_plot_path = os.path.join(output_dir, f"medicine_effect_delta_{safe_col}.png")
                plt.savefig(delta_plot_path, dpi=150, bbox_inches="tight")
                # print(f"  Saved: {delta_plot_path}")
                if show_plots:
                    plt.show()
                else:
                    plt.close()

    # Append MDS-UPDRS score summary row.
    if has_score and "score_clean__delta_on_minus_off" in paired_df.columns:
        sc_off = pd.to_numeric(paired_df["score_clean__off"], errors="coerce").dropna()
        sc_on = pd.to_numeric(paired_df["score_clean__on"], errors="coerce").dropna()
        sc_deltas = pd.to_numeric(
            paired_df["score_clean__delta_on_minus_off"], errors="coerce"
        ).dropna()
        sc_effect = np.nan
        if len(sc_deltas) >= 2:
            _sc_std = float(sc_deltas.std(ddof=1))
            if _sc_std > 0:
                sc_effect = float(sc_deltas.mean() / _sc_std)
        sc_off_paired = pd.to_numeric(paired_df["score_clean__off"], errors="coerce")
        sc_on_paired = pd.to_numeric(paired_df["score_clean__on"], errors="coerce")
        summary_rows.append(
            {
                "feature": "score_clean (MDS-UPDRS)",
                "n_pairs": int(len(sc_deltas)),
                "off_mean": float(sc_off.mean()) if len(sc_off) else np.nan,
                "on_mean": float(sc_on.mean()) if len(sc_on) else np.nan,
                "delta_mean_on_minus_off": float(sc_deltas.mean()) if len(sc_deltas) else np.nan,
                "delta_median_on_minus_off": (
                    float(sc_deltas.median()) if len(sc_deltas) else np.nan
                ),
                "delta_std_on_minus_off": (
                    float(sc_deltas.std(ddof=1)) if len(sc_deltas) >= 2 else np.nan
                ),
                "medicine_effect_size_mean_over_std": sc_effect,
                "wilcoxon_p": wilcoxon_p(sc_deltas),
                "ttest_rel_p": ttest_rel_p(sc_off_paired, sc_on_paired),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    med_dir = os.path.join(output_dir, "medicine_effect")
    os.makedirs(med_dir, exist_ok=True)
    mapping_csv = os.path.join(med_dir, "medicine_effect_mapping_off_on.csv")
    paired_csv = os.path.join(med_dir, "medicine_effect_pairs_per_video.csv")
    summary_csv = os.path.join(med_dir, "medicine_effect_summary.csv")
    mapping_df.to_csv(mapping_csv, index=False)
    paired_df.to_csv(paired_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    # ── Spearman correlation: MDS-UPDRS delta vs kinematic feature deltas ──
    # This reveals whether patients who improve clinically also improve
    # kinematically, even when group-level means cancel out.
    updrs_corr_csv = os.path.join(med_dir, "medicine_effect_updrs_correlations.csv")
    updrs_corr_fig = os.path.join(med_dir, "medicine_effect_updrs_correlations.png")
    if has_score and "score_clean__delta_on_minus_off" in paired_df.columns:
        from scipy.stats import spearmanr

        updrs_delta = pd.to_numeric(paired_df["score_clean__delta_on_minus_off"], errors="coerce")
        delta_cols = [
            c
            for c in paired_df.columns
            if c.endswith("__delta_on_minus_off") and c != "score_clean__delta_on_minus_off"
        ]
        corr_rows = []
        for col in delta_cols:
            feat_name = col.replace("__delta_on_minus_off", "")
            feat_delta = pd.to_numeric(paired_df[col], errors="coerce")
            valid = updrs_delta.notna() & feat_delta.notna()
            n_valid = int(valid.sum())
            if n_valid < 5:
                continue
            rho, pval = spearmanr(updrs_delta[valid], feat_delta[valid])
            corr_rows.append(
                {
                    "feature": feat_name,
                    "n": n_valid,
                    "spearman_rho": float(rho),
                    "p_value": float(pval),
                }
            )

        if corr_rows:
            corr_df = (
                pd.DataFrame(corr_rows)
                .sort_values("spearman_rho", key=abs, ascending=False)
                .reset_index(drop=True)
            )
            corr_df.to_csv(updrs_corr_csv, index=False)

            # Bar chart of Spearman rho values, coloured by sign
            n_feats = len(corr_df)
            fig_h = max(4, n_feats * 0.35)
            fig, ax = plt.subplots(figsize=(8, fig_h))
            colors = ["#d62728" if r > 0 else "#1f77b4" for r in corr_df["spearman_rho"]]
            bars = ax.barh(
                corr_df["feature"][::-1],
                corr_df["spearman_rho"][::-1],
                color=colors[::-1],
                edgecolor="white",
                linewidth=0.4,
            )
            # Mark significant bars (p < 0.05) with a star
            for i, (bar, pval) in enumerate(zip(bars, corr_df["p_value"][::-1])):
                if pval < 0.05:
                    x = bar.get_width()
                    offset = 0.01 if x >= 0 else -0.01
                    ha = "left" if x >= 0 else "right"
                    ax.text(
                        x + offset,
                        bar.get_y() + bar.get_height() / 2,
                        "*",
                        va="center",
                        ha=ha,
                        fontsize=10,
                        color="black",
                    )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spearman ρ  (MDS-UPDRS Δ vs feature Δ)")
            ax.set_title(
                "Individual levodopa response tracking\n"
                "(* p < 0.05;  red = same direction as UPDRS worsening)"
            )
            ax.set_xlim(-1, 1)
            plt.tight_layout()
            plt.savefig(updrs_corr_fig, dpi=150, bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Scatter plots for top-5 features by |rho|
            top5 = corr_df.head(5)
            for _, row in top5.iterrows():
                col = f"{row['feature']}__delta_on_minus_off"
                if col not in paired_df.columns:
                    continue
                feat_delta = pd.to_numeric(paired_df[col], errors="coerce")
                valid = updrs_delta.notna() & feat_delta.notna()
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(
                    updrs_delta[valid],
                    feat_delta[valid],
                    alpha=0.7,
                    edgecolors="k",
                    linewidths=0.4,
                )
                # Regression line
                _x = updrs_delta[valid].values
                _y = feat_delta[valid].values
                _m, _b = np.polyfit(_x, _y, 1)
                _xr = np.array([_x.min(), _x.max()])
                ax.plot(_xr, _m * _xr + _b, color="crimson", linewidth=1.2)
                ax.set_xlabel("MDS-UPDRS Δ (On − Off)")
                ax.set_ylabel(f"{row['feature']} Δ (On − Off)")
                ax.set_title(
                    f"ρ = {row['spearman_rho']:.3f},  p = {row['p_value']:.3f}" f"  (n={row['n']})"
                )
                ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
                ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
                plt.tight_layout()
                safe_feat = row["feature"].replace(" ", "_").replace("/", "-")
                scatter_path = os.path.join(med_dir, f"updrs_corr_scatter_{safe_feat}.png")
                plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
                if show_plots:
                    plt.show()
                else:
                    plt.close()

    print(f"\n{'=' * 60}")
    print("MEDICATION EFFECT (OFF -> ON PAIRING)")
    print(f"{'=' * 60}")
    print(f"  Candidate rows with Off/On + filename token: {len(med_df)}")
    print(f"  Total filename keys seen: {len(counts_wide)}")
    print(f"  Mapped 1:1 Off->On keys: {len(valid_keys)}")
    print(f"  Unmatched Off-only keys: {len(off_only)}")
    print(f"  Unmatched On-only keys: {len(on_only)}")
    print(f"  Excluded keys (>1 Off or >1 On): {len(multi_side)}")
    print("  Effect size formula: mean(On-Off) / std(On-Off)")
    # print(f"  Saved mapping CSV: {mapping_csv}")
    # print(f"  Saved paired CSV: {paired_csv}")
    # print(f"  Saved summary CSV: {summary_csv}")

    return {
        "n_rows_input": int(len(merged)),
        "n_candidate_rows": int(len(med_df)),
        "n_pair_keys_total": int(len(counts_wide)),
        "n_pair_keys_mapped": int(len(valid_keys)),
        "n_pair_keys_unmatched_off_only": int(len(off_only)),
        "n_pair_keys_unmatched_on_only": int(len(on_only)),
        "n_pair_keys_excluded_multi_off_or_on": int(len(multi_side)),
        "mapping_csv": mapping_csv,
        "paired_csv": paired_csv,
        "summary_csv": summary_csv,
    }
