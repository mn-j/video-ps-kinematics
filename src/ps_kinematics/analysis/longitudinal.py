"""Longitudinal progression analysis across visits 1-3."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ps_kinematics.io import normalize_visit_to_int
from ps_kinematics.analysis._demographics import (
    age_group_labels,
    effect_size_from_deltas,
    normalize_gender_series,
    subgroup_delta_stats,
    wilcoxon_p,
)


def extract_visit_from_pom_token(x):
    """Extract visit number from a token like ``POM2...`` (supports 1..3)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NA
    s = str(x)
    m = re.search(r"\bPOM([1-3])(?=[A-Za-z0-9])", s, flags=re.IGNORECASE)
    if not m:
        return pd.NA
    try:
        return int(m.group(1))
    except Exception:
        return pd.NA


def ensure_visit_from_pom_token(df: "pd.DataFrame") -> "pd.DataFrame":
    """Fill missing visit values using ``POMx`` token in ids/video path.

    Priority:
      1) existing ``visit`` column (normalized)
      2) parse from ``ids`` token (e.g., ``POM2VD...``)
      3) parse from ``video_path`` if available
    """
    out = df.copy()
    if "visit" not in out.columns:
        out["visit"] = pd.NA

    out["visit"] = out["visit"].apply(normalize_visit_to_int)

    from_ids = (
        out["ids"].apply(extract_visit_from_pom_token)
        if "ids" in out.columns
        else pd.Series([pd.NA] * len(out), index=out.index)
    )
    from_path = (
        out["video_path"].apply(extract_visit_from_pom_token)
        if "video_path" in out.columns
        else pd.Series([pd.NA] * len(out), index=out.index)
    )

    missing_mask = out["visit"].isna()
    out.loc[missing_mask, "visit"] = from_ids[missing_mask]
    missing_mask = out["visit"].isna()
    out.loc[missing_mask, "visit"] = from_path[missing_mask]

    # Report conflicts for transparency but keep explicit visit column as source of truth.
    conflict_ids = out["visit"].notna() & from_ids.notna() & (out["visit"] != from_ids)
    n_conflict_ids = int(conflict_ids.sum())
    if n_conflict_ids > 0:
        print(f"Visit conflicts with ids POM token: {n_conflict_ids} rows (keeping visit column)")

    conflict_path = out["visit"].notna() & from_path.notna() & (out["visit"] != from_path)
    n_conflict_path = int(conflict_path.sum())
    if n_conflict_path > 0:
        print(
            f"Visit conflicts with video_path POM token: {n_conflict_path} rows (keeping visit column)"
        )

    try:
        out["visit"] = out["visit"].astype("Int64")
    except Exception:
        pass
    return out


def run_longitudinal_progression_report(
    merged: "pd.DataFrame",
    available_features,
    output_dir: str,
    save_plots: bool,
    show_plots: bool,
) -> dict:
    """Track longitudinal progression across visits 1-3.

    Enforces one-video-per-visit per patient/medication/hand tuple.
    Tuples violating this rule are excluded from longitudinal analysis.
    """
    if merged.empty:
        return {
            "n_input_rows": 0,
            "n_rows_visits_1_3": 0,
            "n_keys_total": 0,
            "n_keys_excluded_multi_video_per_visit": 0,
            "n_keys_mapped_ge2_visits": 0,
            "n_keys_mapped_all_3_visits": 0,
            "mapping_csv": None,
            "per_subject_csv": None,
            "summary_csv": None,
        }

    key_cols = ["ids", "medication_state", "hand"]
    req_cols = key_cols + ["visit", "video_ids"]
    missing = [c for c in req_cols if c not in merged.columns]
    if missing:
        print(f"  (Longitudinal report skipped: missing columns {missing})")
        return {"error": f"missing columns: {missing}"}

    long_df = merged.copy()
    long_df["visit"] = long_df["visit"].apply(normalize_visit_to_int)
    try:
        long_df["visit"] = long_df["visit"].astype("Int64")
    except Exception:
        pass
    long_df = long_df[long_df["visit"].isin([1, 2, 3])].copy()
    if long_df.empty:
        print("  (Longitudinal report: no rows with visit in {1,2,3})")
        return {
            "n_input_rows": int(len(merged)),
            "n_rows_visits_1_3": 0,
            "n_keys_total": 0,
            "n_keys_excluded_multi_video_per_visit": 0,
            "n_keys_mapped_ge2_visits": 0,
            "n_keys_mapped_all_3_visits": 0,
            "mapping_csv": None,
            "per_subject_csv": None,
            "summary_csv": None,
        }

    # Enforce: at most one unique video per visit for each patient+medication+hand tuple.
    uniq_vid_counts = (
        long_df.groupby(key_cols + ["visit"], dropna=False)["video_ids"]
        .nunique(dropna=True)
        .reset_index(name="n_unique_videos")
    )
    violating_pairs = uniq_vid_counts[uniq_vid_counts["n_unique_videos"] > 1].copy()
    violating_keys = violating_pairs[key_cols].drop_duplicates().copy()
    n_violating_keys = int(len(violating_keys))

    if n_violating_keys > 0:
        long_df = long_df.merge(
            violating_keys.assign(_exclude_longitudinal=1),
            on=key_cols,
            how="left",
        )
        long_df = long_df[long_df["_exclude_longitudinal"].isna()].drop(
            columns=["_exclude_longitudinal"]
        )

    if long_df.empty:
        print("  (Longitudinal report: all keys excluded due to >1 video per visit)")
        return {
            "n_input_rows": int(len(merged)),
            "n_rows_visits_1_3": 0,
            "n_keys_total": int(merged[key_cols].drop_duplicates().shape[0]),
            "n_keys_excluded_multi_video_per_visit": n_violating_keys,
            "n_keys_mapped_ge2_visits": 0,
            "n_keys_mapped_all_3_visits": 0,
            "mapping_csv": None,
            "per_subject_csv": None,
            "summary_csv": None,
        }

    feature_cols = [
        col for col, _label, _desc, _can_norm in available_features if col in long_df.columns
    ]
    has_score = "score_clean" in long_df.columns

    # --- Demographics: Gender / age_group subgroup columns ---
    if "Gender" in long_df.columns:
        long_df = long_df.copy()
        long_df["gender_norm"] = normalize_gender_series(long_df["Gender"])
    if "age" in long_df.columns:
        long_df = long_df.copy() if "Gender" not in long_df.columns else long_df
        long_df["age_group"] = age_group_labels(long_df["age"])
    _long_demog_cols = [c for c in ["gender_norm", "age_group"] if c in long_df.columns]
    _has_long_demog = len(_long_demog_cols) > 0

    # Collapse any duplicated rows for the same key+visit (same video) by numeric mean.
    agg_dict = {"video_ids": "first"}
    if has_score:
        agg_dict["score_clean"] = "first"
    for c in feature_cols:
        agg_dict[c] = "mean"
    for _dc in _long_demog_cols:
        agg_dict[_dc] = "first"
    collapsed = long_df.groupby(key_cols + ["visit"], dropna=False, as_index=False).agg(agg_dict)

    mapping = collapsed.pivot_table(
        index=key_cols,
        columns="visit",
        values="video_ids",
        aggfunc="first",
    )
    mapping = mapping.rename(columns={1: "video_visit_1", 2: "video_visit_2", 3: "video_visit_3"})
    mapping = mapping.reset_index()
    visit_cols = ["video_visit_1", "video_visit_2", "video_visit_3"]
    mapping["n_visits_mapped"] = mapping[visit_cols].notna().sum(axis=1)

    n_keys_total = int(mapping.shape[0])
    n_ge2 = int((mapping["n_visits_mapped"] >= 2).sum())
    n_all3 = int((mapping["n_visits_mapped"] == 3).sum())

    long_dir = os.path.join(output_dir, "longitudinal")
    os.makedirs(long_dir, exist_ok=True)
    mapping_csv = os.path.join(long_dir, "longitudinal_visit_mapping.csv")
    mapping.to_csv(mapping_csv, index=False)

    per_subject_df = mapping.copy()

    # Add demographics to per_subject_df for subgroup analysis.
    if _has_long_demog:
        _long_demog_lookup = collapsed[key_cols + _long_demog_cols].drop_duplicates(subset=key_cols)
        per_subject_df = per_subject_df.merge(_long_demog_lookup, on=key_cols, how="left")

    # Add per-visit MDS-UPDRS scores to the per-subject table.
    if has_score and "score_clean" in collapsed.columns:
        score_wide = collapsed.pivot_table(
            index=key_cols, columns="visit", values="score_clean", aggfunc="first"
        )
        score_wide = score_wide.rename(
            columns={1: "score_visit_1", 2: "score_visit_2", 3: "score_visit_3"}
        ).reset_index()
        for _sc in ("score_visit_1", "score_visit_2", "score_visit_3"):
            if _sc not in score_wide.columns:
                score_wide[_sc] = np.nan
        per_subject_df = per_subject_df.merge(score_wide, on=key_cols, how="left")

    summary_rows = []

    for feat in feature_cols:
        wide = collapsed.pivot_table(index=key_cols, columns="visit", values=feat, aggfunc="mean")
        wide = wide.rename(columns={1: "v1", 2: "v2", 3: "v3"}).reset_index()
        for _vcol in ("v1", "v2", "v3"):
            if _vcol not in wide.columns:
                wide[_vcol] = np.nan
        wide[f"{feat}__d12"] = wide["v2"] - wide["v1"]
        wide[f"{feat}__d23"] = wide["v3"] - wide["v2"]
        wide[f"{feat}__d13"] = wide["v3"] - wide["v1"]

        def _slope_row(r):
            x = []
            y = []
            for v_idx, c_name in ((1, "v1"), (2, "v2"), (3, "v3")):
                val = r[c_name]
                if pd.notna(val):
                    x.append(float(v_idx))
                    y.append(float(val))
            if len(x) < 2:
                return np.nan
            return float(np.polyfit(x, y, 1)[0])

        wide[f"{feat}__slope_per_visit"] = wide.apply(_slope_row, axis=1)
        wide = wide.drop(columns=["v1", "v2", "v3"])

        per_subject_df = per_subject_df.merge(wide, on=key_cols, how="left")

        feat_wide = collapsed.pivot_table(
            index=key_cols, columns="visit", values=feat, aggfunc="mean"
        )
        n_v1 = (
            int(feat_wide.get(1, pd.Series(dtype=float)).notna().sum())
            if 1 in feat_wide.columns
            else 0
        )
        n_v2 = (
            int(feat_wide.get(2, pd.Series(dtype=float)).notna().sum())
            if 2 in feat_wide.columns
            else 0
        )
        n_v3 = (
            int(feat_wide.get(3, pd.Series(dtype=float)).notna().sum())
            if 3 in feat_wide.columns
            else 0
        )

        d12 = (
            (feat_wide[2] - feat_wide[1]).dropna()
            if {1, 2}.issubset(set(feat_wide.columns))
            else pd.Series(dtype=float)
        )
        d23 = (
            (feat_wide[3] - feat_wide[2]).dropna()
            if {2, 3}.issubset(set(feat_wide.columns))
            else pd.Series(dtype=float)
        )
        d13 = (
            (feat_wide[3] - feat_wide[1]).dropna()
            if {1, 3}.issubset(set(feat_wide.columns))
            else pd.Series(dtype=float)
        )
        n_triplets = (
            int(feat_wide[[1, 2, 3]].notna().all(axis=1).sum())
            if {1, 2, 3}.issubset(set(feat_wide.columns))
            else 0
        )

        slopes = per_subject_df[f"{feat}__slope_per_visit"].dropna()

        # Friedman test across all 3 visits (subjects with complete triplets only).
        _friedman_p = np.nan
        if n_triplets >= 4 and {1, 2, 3}.issubset(set(feat_wide.columns)):
            try:
                from scipy.stats import friedmanchisquare

                _triplet_mask = feat_wide[[1, 2, 3]].notna().all(axis=1)
                _v1 = feat_wide.loc[_triplet_mask, 1].values
                _v2 = feat_wide.loc[_triplet_mask, 2].values
                _v3 = feat_wide.loc[_triplet_mask, 3].values
                _, _friedman_p = friedmanchisquare(_v1, _v2, _v3)
                _friedman_p = float(_friedman_p)
            except Exception:
                _friedman_p = np.nan

        _long_row = {
            "feature": feat,
            "n_visit1": n_v1,
            "n_visit2": n_v2,
            "n_visit3": n_v3,
            "n_pairs_1_2": int(len(d12)),
            "n_pairs_2_3": int(len(d23)),
            "n_pairs_1_3": int(len(d13)),
            "n_triplets_1_2_3": n_triplets,
            "delta_1_2_mean": float(d12.mean()) if len(d12) else np.nan,
            "delta_1_2_median": float(d12.median()) if len(d12) else np.nan,
            "delta_1_2_std": float(d12.std(ddof=1)) if len(d12) >= 2 else np.nan,
            "effect_size_d12": effect_size_from_deltas(d12),
            "wilcoxon_p_d12": wilcoxon_p(d12),
            "delta_2_3_mean": float(d23.mean()) if len(d23) else np.nan,
            "delta_2_3_median": float(d23.median()) if len(d23) else np.nan,
            "delta_2_3_std": float(d23.std(ddof=1)) if len(d23) >= 2 else np.nan,
            "effect_size_d23": effect_size_from_deltas(d23),
            "wilcoxon_p_d23": wilcoxon_p(d23),
            "delta_1_3_mean": float(d13.mean()) if len(d13) else np.nan,
            "delta_1_3_median": float(d13.median()) if len(d13) else np.nan,
            "delta_1_3_std": float(d13.std(ddof=1)) if len(d13) >= 2 else np.nan,
            "effect_size_d13": effect_size_from_deltas(d13),
            "wilcoxon_p_d13": wilcoxon_p(d13),
            "friedman_p": _friedman_p,
            "mean_subject_slope_per_visit": float(slopes.mean()) if len(slopes) else np.nan,
            "median_subject_slope_per_visit": float(slopes.median()) if len(slopes) else np.nan,
        }

        # --- Per-gender and per-age-group subgroup analysis ---
        if _has_long_demog and f"{feat}__d12" in per_subject_df.columns:
            _d12_all = per_subject_df[f"{feat}__d12"]
            _d13_all = per_subject_df[f"{feat}__d13"]
            _slp_all = per_subject_df[f"{feat}__slope_per_visit"]
            for _dc in _long_demog_cols:
                if _dc not in per_subject_df.columns:
                    continue
                _grp = per_subject_df[_dc]
                # d1->2 deltas per group
                _long_row.update(subgroup_delta_stats(_d12_all, _grp, f"delta_1_2_{_dc}"))
                # d1->3 deltas per group
                _long_row.update(subgroup_delta_stats(_d13_all, _grp, f"delta_1_3_{_dc}"))
                # per-subject slope per group
                _long_row.update(subgroup_delta_stats(_slp_all, _grp, f"slope_{_dc}"))

        summary_rows.append(_long_row)

        # Visual report: individual trajectories + cohort mean by visit.
        feat_for_plot = feat_wide.copy()
        if len(feat_for_plot) > 0 and save_plots:
            fig, ax = plt.subplots(figsize=(8, 5))
            for _, row in feat_for_plot.iterrows():
                xs = []
                ys = []
                for v in (1, 2, 3):
                    if v in feat_for_plot.columns and pd.notna(row.get(v, np.nan)):
                        xs.append(v)
                        ys.append(float(row[v]))
                if len(xs) >= 2:
                    ax.plot(xs, ys, color="gray", alpha=0.25, linewidth=1)

            mean_vals = []
            n_vals = []
            for v in (1, 2, 3):
                if v in feat_for_plot.columns:
                    vals = pd.to_numeric(feat_for_plot[v], errors="coerce").dropna()
                    mean_vals.append(float(vals.mean()) if len(vals) else np.nan)
                    n_vals.append(int(len(vals)))
                else:
                    mean_vals.append(np.nan)
                    n_vals.append(0)

            ax.plot(
                [1, 2, 3],
                mean_vals,
                color="tab:blue",
                marker="o",
                linewidth=2.5,
                label="Cohort mean",
            )
            for x, n in zip([1, 2, 3], n_vals):
                ax.annotate(
                    f"n={n}",
                    xy=(x, ax.get_ylim()[1]),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="dimgray",
                )

            ax.set_xticks([1, 2, 3])
            ax.set_xlabel("Visit")
            ax.set_ylabel(feat)
            ax.set_title(f"Longitudinal progression: {feat}")
            ax.legend(loc="best")
            plt.tight_layout()

            safe_col = feat.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
            plot_path = os.path.join(output_dir, f"longitudinal_{safe_col}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            # print(f"  Saved: {plot_path}")
            if show_plots:
                plt.show()
            else:
                plt.close()

    # Append MDS-UPDRS score summary row to the longitudinal summary.
    if has_score and "score_clean" in collapsed.columns:
        sc_wide = collapsed.pivot_table(
            index=key_cols, columns="visit", values="score_clean", aggfunc="first"
        )
        sc_n_v1 = (
            int(sc_wide.get(1, pd.Series(dtype=float)).notna().sum()) if 1 in sc_wide.columns else 0
        )
        sc_n_v2 = (
            int(sc_wide.get(2, pd.Series(dtype=float)).notna().sum()) if 2 in sc_wide.columns else 0
        )
        sc_n_v3 = (
            int(sc_wide.get(3, pd.Series(dtype=float)).notna().sum()) if 3 in sc_wide.columns else 0
        )
        sc_d12 = (
            (sc_wide[2] - sc_wide[1]).dropna()
            if {1, 2}.issubset(set(sc_wide.columns))
            else pd.Series(dtype=float)
        )
        sc_d23 = (
            (sc_wide[3] - sc_wide[2]).dropna()
            if {2, 3}.issubset(set(sc_wide.columns))
            else pd.Series(dtype=float)
        )
        sc_d13 = (
            (sc_wide[3] - sc_wide[1]).dropna()
            if {1, 3}.issubset(set(sc_wide.columns))
            else pd.Series(dtype=float)
        )
        sc_n_triplets = (
            int(sc_wide[[1, 2, 3]].notna().all(axis=1).sum())
            if {1, 2, 3}.issubset(set(sc_wide.columns))
            else 0
        )

        def _sc_slope(r):
            xs, ys = [], []
            for v_idx, col_name in ((1, 1), (2, 2), (3, 3)):
                val = r.get(col_name, np.nan)
                if pd.notna(val):
                    xs.append(float(v_idx))
                    ys.append(float(val))
            return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else np.nan

        sc_slopes = sc_wide.apply(_sc_slope, axis=1).dropna()
        _sc_friedman_p = np.nan
        if sc_n_triplets >= 4 and {1, 2, 3}.issubset(set(sc_wide.columns)):
            try:
                from scipy.stats import friedmanchisquare

                _sc_tmask = sc_wide[[1, 2, 3]].notna().all(axis=1)
                _, _sc_friedman_p = friedmanchisquare(
                    sc_wide.loc[_sc_tmask, 1].values,
                    sc_wide.loc[_sc_tmask, 2].values,
                    sc_wide.loc[_sc_tmask, 3].values,
                )
                _sc_friedman_p = float(_sc_friedman_p)
            except Exception:
                _sc_friedman_p = np.nan
        summary_rows.append(
            {
                "feature": "score_clean (MDS-UPDRS)",
                "n_visit1": sc_n_v1,
                "n_visit2": sc_n_v2,
                "n_visit3": sc_n_v3,
                "n_pairs_1_2": int(len(sc_d12)),
                "n_pairs_2_3": int(len(sc_d23)),
                "n_pairs_1_3": int(len(sc_d13)),
                "n_triplets_1_2_3": sc_n_triplets,
                "delta_1_2_mean": float(sc_d12.mean()) if len(sc_d12) else np.nan,
                "delta_1_2_median": float(sc_d12.median()) if len(sc_d12) else np.nan,
                "delta_1_2_std": float(sc_d12.std(ddof=1)) if len(sc_d12) >= 2 else np.nan,
                "effect_size_d12": effect_size_from_deltas(sc_d12),
                "wilcoxon_p_d12": wilcoxon_p(sc_d12),
                "delta_2_3_mean": float(sc_d23.mean()) if len(sc_d23) else np.nan,
                "delta_2_3_median": float(sc_d23.median()) if len(sc_d23) else np.nan,
                "delta_2_3_std": float(sc_d23.std(ddof=1)) if len(sc_d23) >= 2 else np.nan,
                "effect_size_d23": effect_size_from_deltas(sc_d23),
                "wilcoxon_p_d23": wilcoxon_p(sc_d23),
                "delta_1_3_mean": float(sc_d13.mean()) if len(sc_d13) else np.nan,
                "delta_1_3_median": float(sc_d13.median()) if len(sc_d13) else np.nan,
                "delta_1_3_std": float(sc_d13.std(ddof=1)) if len(sc_d13) >= 2 else np.nan,
                "effect_size_d13": effect_size_from_deltas(sc_d13),
                "wilcoxon_p_d13": wilcoxon_p(sc_d13),
                "friedman_p": _sc_friedman_p,
                "mean_subject_slope_per_visit": (
                    float(sc_slopes.mean()) if len(sc_slopes) else np.nan
                ),
                "median_subject_slope_per_visit": (
                    float(sc_slopes.median()) if len(sc_slopes) else np.nan
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    per_subject_csv = os.path.join(long_dir, "longitudinal_progression_per_subject.csv")
    summary_csv = os.path.join(long_dir, "longitudinal_progression_summary.csv")
    per_subject_df.to_csv(per_subject_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n{'=' * 60}")
    print("LONGITUDINAL PROGRESSION (VISITS 1-3)")
    print(f"{'=' * 60}")
    print(f"  Rows considered (visit 1..3): {len(merged)} -> {len(collapsed)}")
    print(f"  Unique patient+med+hand keys: {n_keys_total}")
    print(f"  Excluded keys (>1 video within a visit): {n_violating_keys}")
    print(f"  Mapped keys with >=2 visits: {n_ge2}")
    print(f"  Mapped keys with all 3 visits: {n_all3}")
    # print(f"  Saved mapping CSV: {mapping_csv}")
    # print(f"  Saved per-subject CSV: {per_subject_csv}")
    # print(f"  Saved summary CSV: {summary_csv}")

    return {
        "n_input_rows": int(len(merged)),
        "n_rows_visits_1_3": int(len(collapsed)),
        "n_keys_total": n_keys_total,
        "n_keys_excluded_multi_video_per_visit": n_violating_keys,
        "n_keys_mapped_ge2_visits": n_ge2,
        "n_keys_mapped_all_3_visits": n_all3,
        "mapping_csv": mapping_csv,
        "per_subject_csv": per_subject_csv,
        "summary_csv": summary_csv,
    }
