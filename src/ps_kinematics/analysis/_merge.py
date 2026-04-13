"""Data loading and UPDRS total score utilities."""

import os

import numpy as np
import pandas as pd

from ps_kinematics.io import normalize_med_state


def load_updrs_total_wide(
    csv_path: str,
    score_col_name: str = "UPDRS_Total",
) -> "pd.DataFrame":
    """Load a wide-format MDS-UPDRS Part-III total-score CSV into long form.

    Expected input columns
    -----------------------
    ``ids``             - patient IDs (not video IDs)
    ``updrs_v1_off``    - Part-III total, visit 1, medication Off
    ``updrs_v2_off``    - Part-III total, visit 2, medication Off
    ``updrs_v3_off``    - Part-III total, visit 3, medication Off
    ``updrs_v1_on``     - Part-III total, visit 1, medication On
    ``updrs_v2_on``     - Part-III total, visit 2, medication On
    ``updrs_v3_on``     - Part-III total, visit 3, medication On

    Returns a long-form DataFrame with columns:
    ``ids``, ``visit`` (Int64), ``medication_state`` (normalised), ``score_col_name``.
    Rows with missing score values are dropped.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    # Map raw wide column names -> (visit, medication_state)
    _COL_MAP = {
        "updrs_v1_off": (1, "Off"),
        "updrs_v2_off": (2, "Off"),
        "updrs_v3_off": (3, "Off"),
        "updrs_v1_on": (1, "On"),
        "updrs_v2_on": (2, "On"),
        "updrs_v3_on": (3, "On"),
    }

    found_cols = [c for c in _COL_MAP if c in df.columns]
    if not found_cols:
        raise RuntimeError(
            f"UPDRS-total CSV '{csv_path}' contains none of the expected wide columns "
            f"({list(_COL_MAP.keys())}).  Columns present: {list(df.columns)}"
        )

    if "ids" not in df.columns:
        raise RuntimeError(
            f"UPDRS-total CSV '{csv_path}' is missing the 'ids' column (patient IDs)."
        )

    rows: list[dict] = []
    for _, row in df.iterrows():
        patient_id = row["ids"]
        for col_name, (visit_num, med_state) in _COL_MAP.items():
            if col_name not in df.columns:
                continue
            val = row[col_name]
            if pd.isna(val):
                continue
            rows.append(
                {
                    "ids": patient_id,
                    "visit": visit_num,
                    "medication_state": normalize_med_state(med_state),
                    score_col_name: val,
                }
            )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df

    long_df["visit"] = long_df["visit"].astype("Int64")
    long_df[score_col_name] = pd.to_numeric(long_df[score_col_name], errors="coerce")
    long_df = long_df.dropna(subset=[score_col_name]).copy()
    print(f"  UPDRS-total wide CSV: {len(df)} patient rows \u2192 {len(long_df)} long-form score rows")
    return long_df
