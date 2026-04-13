"""
ps_kinematics.io — I/O utilities: CSV/JSON read-write, path parsing utilities.
"""

import ast
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from pathlib import PurePath
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================
# Path / value parsing helpers
# ============================================================


def parse_medication_state_from_path(video_path: Optional[str]) -> Optional[str]:
    """Extract medication state token 'On' or 'Off' from path."""
    if not video_path:
        return None
    s = str(video_path).replace("\\\\", "/")
    m = re.search(r"(?<![A-Za-z])(On|Off)(?![A-Za-z])", s, flags=re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def parse_hand_from_path(video_path: Optional[str]) -> Optional[str]:
    """Extract hand from path.

    Recognises two conventions:
    - Primary dataset: ``_4R`` / ``_4L`` token in filename.
    - TULIP dataset: ``_left`` / ``_right`` in a parent folder name
      (e.g. ``13. Pronation_and_supination_left``).
    """
    if not video_path:
        return None
    s = str(video_path)
    # Primary dataset tokens (checked first — more specific)
    if "_4R" in s:
        return "Right"
    if "_4L" in s:
        return "Left"
    # TULIP-style: folder name ends with _left / _right
    s_lower = s.replace("\\", "/").lower()
    if "_right/" in s_lower or s_lower.endswith("_right"):
        return "Right"
    if "_left/" in s_lower or s_lower.endswith("_left"):
        return "Left"
    return None


def parse_ids_and_visit(video_path: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Parse ids and visit from a video path.

    Supports two dataset conventions:

    - **Primary dataset**: ``Video/visit X/<ids>/...`` directory structure,
      or ``POM``-token filenames.
    - **TULIP dataset**: ``Subject_N/...`` directory structure → returns
      ``TULIP_NNN`` as the id with visit = 1 (single-session dataset).
    """
    if not video_path:
        return None, None
    p = PurePath(video_path)
    parts = list(p.parts)

    # ── Primary dataset: Video/visit X/<ids>/ ────────────────────────
    idx_video = None
    for i, seg in enumerate(parts):
        if str(seg).lower() == "video":
            idx_video = i
            break
    if idx_video is not None and len(parts) >= idx_video + 3:
        visit_seg = str(parts[idx_video + 1])
        ids_seg = str(parts[idx_video + 2])
        m = re.search(r"visit\s*(\d+)", visit_seg, flags=re.IGNORECASE)
        if m:
            return ids_seg, int(m.group(1))
        # If "visit" not matched, check for TULIP Subject_N in remaining parts
        # before returning a potentially wrong ids_seg.

    # ── TULIP dataset: .../Subject_N/... ─────────────────────────────
    for seg in parts:
        m_subj = re.fullmatch(r"Subject_(\d+)", str(seg), flags=re.IGNORECASE)
        if m_subj:
            return f"TULIP_{int(m_subj.group(1)):03d}", 1

    # ── Primary dataset fallback: regex on full path ─────────────────
    s = str(video_path).replace("\\\\", "/")
    m = re.search(r"/Video/visit\s*(\d+)/([^/]+)/", s, flags=re.IGNORECASE)
    if m:
        return m.group(2), int(m.group(1))

    # Fallback for paths like: .../segmented_video_..._<PATIENT_ID>_PS_<VISIT>_<TASK>.mp4
    m_id = re.search(r"(POM\w+)", s, flags=re.IGNORECASE)
    if m_id:
        id_val = m_id.group(1)
        visit_val = None
        m_vis = re.search(r"PS_(\d+)", s, flags=re.IGNORECASE)
        if m_vis:
            try:
                visit_val = int(m_vis.group(1))
            except ValueError:
                pass
        return id_val, visit_val

    return None, None


def canonicalize_video_id(video_id: Optional[str]) -> Optional[str]:
    """Return canonical video id before any underscore or file extension.

    Examples:
      'SUBJ001_PS_002_4' -> 'SUBJ001'
      'SUBJ002' -> 'SUBJ002'
      'TULIP_001' -> 'TULIP_001'  (TULIP ids kept intact)
    """
    if video_id is None:
        return None
    s = str(video_id).strip()
    if not s:
        return None
    # remove file extension if present
    s = s.split(".")[0]
    # TULIP ids (TULIP_NNN) are already canonical — keep intact
    if re.fullmatch(r"TULIP_\d+", s, flags=re.IGNORECASE):
        return s
    # take part before first underscore
    s = s.split("_")[0]
    return s


# ============================================================
# Normalization helpers
# ============================================================


def normalize_med_state(x) -> Optional[str]:
    """Normalize medication state to 'On' or 'Off'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s.startswith("on"):
        return "On"
    if s.startswith("off"):
        return "Off"
    return None


def normalize_hand(x) -> Optional[str]:
    """Normalize hand to 'Left' or 'Right'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"l", "left", "lhs"}:
        return "Left"
    if s in {"r", "right", "rhs"}:
        return "Right"
    if "right" in s or "4r" in s:
        return "Right"
    if "left" in s or "4l" in s:
        return "Left"
    return None


def normalize_visit_to_int(v):
    """Normalize visit values to integer (nullable). Returns ``pd.NA`` for missing."""
    if pd.isna(v):
        return pd.NA
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip()
    if s == "":
        return pd.NA
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
    except Exception:
        pass
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return pd.NA


# ============================================================
# Mapping & scoring helpers
# ============================================================


def load_videoid_to_patientid_map(id2vid_csv_path: str) -> dict[str, str]:
    """Load id2vid.csv and create a mapping from video_id -> patient_id."""
    df = pd.read_csv(id2vid_csv_path, header=None)
    if df.shape[1] < 2:
        raise RuntimeError(
            "id2vid.csv must have at least 2 columns: patient_id and tuple(video_ids)"
        )

    video_to_patient = {}
    for _, row in df.iterrows():
        patient_id = str(row.iloc[0]).strip()
        vids_raw = row.iloc[1]

        vids = []
        if isinstance(vids_raw, (tuple, list)):
            vids = list(vids_raw)
        else:
            s = str(vids_raw).strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (tuple, list)):
                    vids = list(parsed)
                else:
                    vids = [parsed]
            except Exception:
                vids = re.findall(r"POM\d+VD\d+", s)

        for v in vids:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            video_id = str(v).strip().strip("'").strip('"')
            if video_id:
                video_to_patient[video_id] = patient_id

    return video_to_patient


def coerce_int_score(x) -> Optional[int]:
    """Coerce a score value to integer, return None if invalid."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if s == "":
        return None
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        return None
    except Exception:
        return None


def load_clinical_scores_table(score_csv_path: str, score_column: str = "ProS") -> pd.DataFrame:
    """Load and normalize the clinical scores table used by overlay rendering."""
    if not score_csv_path:
        return pd.DataFrame()

    scores = pd.read_csv(score_csv_path, sep=None, engine="python")
    if scores.empty:
        return scores

    if "ids" in scores.columns:
        scores["ids"] = scores["ids"].astype(str).str.strip()

    if "visit" in scores.columns:
        scores["visit"] = scores["visit"].apply(normalize_visit_to_int)
        try:
            scores["visit"] = scores["visit"].astype("Int64")
        except Exception:
            pass

    if "medication_state" in scores.columns:
        scores["medication_state"] = scores["medication_state"].apply(normalize_med_state)

    if "hand" in scores.columns:
        scores["hand"] = scores["hand"].apply(normalize_hand)

    if score_column in scores.columns:
        scores["score_clean"] = scores[score_column].apply(coerce_int_score)
    else:
        scores["score_clean"] = pd.NA

    return scores


def resolve_video_clinical_score(
    video_path: str,
    video_to_patient: dict,
    scores_df: pd.DataFrame,
    score_column: str = "ProS",
    visit=None,
    medication_state=None,
    hand=None,
):
    """Resolve a unique clinical score for a video using patient/task metadata."""
    result = {
        "video_id": None,
        "patient_id": None,
        "score_raw": None,
        "score_clean": None,
    }
    if not video_path or not video_to_patient or scores_df is None or scores_df.empty:
        return result

    ids_parsed, visit_parsed = parse_ids_and_visit(video_path)
    video_id = canonicalize_video_id(ids_parsed)
    result["video_id"] = video_id
    if not video_id:
        return result

    patient_id = video_to_patient.get(str(video_id))
    result["patient_id"] = patient_id
    if patient_id is None or "ids" not in scores_df.columns:
        return result

    visit_norm = normalize_visit_to_int(visit if visit is not None else visit_parsed)
    med_norm = normalize_med_state(
        medication_state
        if medication_state is not None
        else parse_medication_state_from_path(video_path)
    )
    hand_norm = normalize_hand(hand if hand is not None else parse_hand_from_path(video_path))

    candidates = scores_df[
        scores_df["ids"].astype(str).str.strip() == str(patient_id).strip()
    ].copy()
    if candidates.empty:
        return result

    if "visit" in candidates.columns and pd.notna(visit_norm):
        candidates = candidates[candidates["visit"] == visit_norm]
    if "medication_state" in candidates.columns and med_norm is not None:
        candidates = candidates[candidates["medication_state"] == med_norm]
    if "hand" in candidates.columns and hand_norm is not None:
        candidates = candidates[candidates["hand"] == hand_norm]
    if candidates.empty:
        return result

    raw_values = (
        candidates[score_column].dropna().tolist() if score_column in candidates.columns else []
    )
    clean_values = (
        sorted({int(x) for x in candidates["score_clean"].dropna().tolist()})
        if "score_clean" in candidates.columns
        else []
    )

    if len(clean_values) == 1:
        result["score_clean"] = clean_values[0]
    elif len(raw_values) == 1:
        result["score_clean"] = coerce_int_score(raw_values[0])

    if len(raw_values) == 1:
        result["score_raw"] = raw_values[0]
    elif result["score_clean"] is not None:
        result["score_raw"] = result["score_clean"]

    return result


def detect_extreme_outliers(data, iqr_multiplier=3.0):
    """Detect extreme outliers using IQR method.

    Returns
    -------
    tuple : (has_extremes, mask_without_extremes, lower_bound, upper_bound)
    """
    valid_data = data.dropna()
    if len(valid_data) < 4:
        return False, pd.Series([True] * len(data), index=data.index), None, None

    q1 = valid_data.quantile(0.25)
    q3 = valid_data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    is_extreme = (data < lower_bound) | (data > upper_bound)
    n_extreme = is_extreme.sum()

    if n_extreme > 0:
        data_range = valid_data.max() - valid_data.min()
        non_extreme_data = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]
        if len(non_extreme_data) > 0:
            non_extreme_range = non_extreme_data.max() - non_extreme_data.min()
            has_extremes = data_range > 2 * non_extreme_range if non_extreme_range > 0 else False
        else:
            has_extremes = False
    else:
        has_extremes = False

    mask_without_extremes = ~is_extreme
    return has_extremes, mask_without_extremes, lower_bound, upper_bound


# ============================================================
# Age / gender helpers
# ============================================================


def load_age_gender(age_gender_csv_path: str) -> pd.DataFrame:
    """Load age_gender.csv and return a tidy DataFrame."""
    ag = pd.read_csv(age_gender_csv_path)
    first_col = ag.columns[0]
    if first_col != "ids":
        ag = ag.rename(columns={first_col: "ids"})
    ag["ids"] = ag["ids"].astype(str).str.strip()
    for col in ("age", "Gender"):
        if col in ag.columns:
            ag[col] = pd.to_numeric(ag[col], errors="coerce")
        else:
            ag[col] = np.nan
    return ag[["ids", "age", "Gender"]].drop_duplicates(subset="ids").copy()


def residualize_for_age_gender(df: pd.DataFrame, feature_col: str) -> pd.Series:
    """Return age- and gender-adjusted values for *feature_col* via OLS residuals."""
    result = pd.Series(np.nan, index=df.index, dtype=float)
    mask = df[feature_col].notna() & df["age"].notna() & df["Gender"].notna()
    if mask.sum() < 5:
        result.loc[df[feature_col].notna()] = df.loc[df[feature_col].notna(), feature_col].to_numpy(
            dtype=float
        )
        return result

    sub = df.loc[mask]
    y = sub[feature_col].to_numpy(dtype=float)
    gender_dummy = (sub["Gender"] == 2).astype(float).to_numpy()
    age_vals = sub["age"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(sub)), age_vals, gender_dummy])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        grand_mean = float(y.mean())
        result.loc[sub.index] = residuals + grand_mean
    except Exception:
        result.loc[sub.index] = y
    return result


# ============================================================
# Stdlib-only xlsx reader (avoids openpyxl dependency)
# ============================================================


def read_xlsx_stdlib(path: str) -> "pd.DataFrame":
    """Read the first sheet of an xlsx file into a DataFrame.

    Uses only Python's standard library (``zipfile`` + ``xml.etree.ElementTree``),
    so no ``openpyxl`` installation is required.  Handles shared-string lookups,
    boolean cells, and blank cells.  Sufficient for flat tabular validation files.
    """
    NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

        # -- Shared strings --
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            ss_root = ET.parse(zf.open("xl/sharedStrings.xml")).getroot()
            for si in ss_root:
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS}}}t")))

        # -- First worksheet --
        sheet_file = "xl/worksheets/sheet1.xml"
        try:
            wb_rels = ET.parse(zf.open("xl/_rels/workbook.xml.rels")).getroot()
            for rel in wb_rels:
                target = rel.get("Target", "")
                if target.startswith("worksheets/"):
                    sheet_file = f"xl/{target}"
                    break
        except KeyError:
            pass

        ws_root = ET.parse(zf.open(sheet_file)).getroot()

        rows_data: list[list] = []
        for row_el in ws_root.iter(f"{{{NS}}}row"):
            row: list = []
            for c in row_el:
                cell_type = c.get("t", "")
                v_el = c.find(f"{{{NS}}}v")
                raw = v_el.text if v_el is not None else None
                if raw is None:
                    row.append(None)
                elif cell_type == "s":  # shared string index
                    row.append(shared[int(raw)])
                elif cell_type == "b":  # boolean
                    row.append(bool(int(raw)))
                elif cell_type == "str":  # inline formula result
                    row.append(raw)
                else:  # numeric / date serial
                    try:
                        row.append(float(raw))
                    except ValueError:
                        row.append(raw)
            rows_data.append(row)

    if not rows_data:
        return pd.DataFrame()

    header = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows_data[0])]
    return pd.DataFrame(rows_data[1:], columns=header)


# ============================================================
# Video path normalization for cross-file matching
# ============================================================


def normalize_video_path_for_matching(path_value) -> str:
    """Build a canonical filename-level key for cross-file video matching.

    Ensures file matching does not break when tracked outputs and
    label files use different prefixes/suffixes, without collapsing distinct
    videos into a coarse patient-level key.
    """
    if pd.isna(path_value):
        return ""

    out = str(path_value).strip().replace("\\", "/").lower()
    # Handle both processing suffix variants seen in tracked outputs.
    out = re.sub(r"_cl_sh_sr4x(?:_gimmvfi)?(?=\.[^./\\]+$|$)", "", out)
    name = os.path.basename(out)

    # Segmented clips: match on stable tail after "segmented_video_" so
    # enhanced/window/cropped prefixes become equivalent while preserving
    # clip-level uniqueness.
    if "segmented_video_" in name:
        tail = name.split("segmented_video_", 1)[1]
        tail = re.sub(r"_cropped(?=\.[^./\\]+$|$)", "", tail)
        return f"seg::{tail}"

    # Non-segmented clips: build a stable key from visit + patient token +
    # hand-side filename tail to avoid many-to-one collisions.
    visit_match = re.search(r"visit\s*(\d+)", out)
    pom_match = re.search(r"(pom\d+vd\d+)", out)
    hand_tail_match = re.search(r"((?:on|off)_4[lr].*)", name)
    if pom_match and hand_tail_match:
        visit_part = visit_match.group(1) if visit_match else "na"
        hand_tail = re.sub(r"\s+", " ", hand_tail_match.group(1)).strip()
        hand_tail = re.sub(r"_cropped(?=\.[^./\\]+$|$)", "", hand_tail)
        return f"clip::v{visit_part}_{pom_match.group(1)}_{hand_tail}"

    # Fallback for uncommon naming patterns.
    name = re.sub(r"^enhanced_videos\d*_", "", name)
    name = re.sub(r"enhanced_videos\d*", "cropped_videos_output", name)
    name = re.sub(r"^clahe_sharpen_esrgan_videos\d*_", "", name)
    name = re.sub(r"clahe_sharpen_esrgan_videos\d*", "cropped_videos_output", name)
    if name.startswith("visit "):
        name = f"cropped_videos_output_{name}"
    return re.sub(r"\s+", " ", name).strip()


@lru_cache(maxsize=16384)
def normalize_video_path_for_matching_cached(path_value_str: str) -> str:
    """Cached variant for repeated path-key normalization calls."""
    return normalize_video_path_for_matching(path_value_str)


def normalize_video_path_series_for_matching(series: "pd.Series") -> "pd.Series":
    """Vectorized/cached normalization for video-path matching keys."""
    if series.empty:
        return series.astype(str)

    out = pd.Series(index=series.index, dtype="object")
    _notna = series.notna()
    out.loc[~_notna] = ""

    if _notna.any():
        unique_vals = pd.unique(series.loc[_notna].astype(str))
        mapped = {v: normalize_video_path_for_matching_cached(v) for v in unique_vals}
        out.loc[_notna] = series.loc[_notna].astype(str).map(mapped)

    return out
