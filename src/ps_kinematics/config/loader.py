"""YAML pipeline-configuration loader.

Loads pipeline configuration from YAML files under ``configs/``, resolves
paths, expands ``~``, and merges the environment.

Layout
------
  configs/
    config.example.yaml          — pipeline paths + processing options
    tuning_25fps.yaml            — tuning-overrides profile #1
    tuning_enhanced_50fps.yaml   — tuning-overrides profile #2

Typical use
-----------
    from ps_kinematics.config import load_pipeline_config

    cfg = load_pipeline_config("configs/config.yaml")
    # cfg is a dict with the standard CONFIG keys plus two extras:
    #   'tuning_overrides'        — dict (the primary profile)
    #   'tuning_overrides_50fps'  — dict or None (optional secondary profile)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml  # PyYAML
except ImportError as exc:  # pragma: no cover — surfaced at runtime
    raise ImportError(
        "PyYAML is required for config loading. Install with: pip install pyyaml"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────


def _expand(p: str | None) -> str | None:
    """Expand ~ and environment variables in a path string."""
    if p is None:
        return None
    return os.path.expanduser(os.path.expandvars(str(p)))


def _resolve_relative(path: str, base_dir: Path) -> Path:
    """Resolve *path* against *base_dir* if it is not already absolute."""
    p = Path(_expand(path))
    return p if p.is_absolute() else (base_dir / p)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────


def load_tuning_profile(path: str | Path) -> dict[str, Any]:
    """Load a standalone tuning-overrides YAML file.

    The file is a flat mapping of `CONSTANT_NAME: value` pairs; it is passed
    directly to ``apply_tuning_overrides`` at worker startup.
    """
    p = Path(_expand(str(path)))
    if not p.exists():
        raise FileNotFoundError(f"Tuning profile not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Tuning profile must be a mapping, got {type(data).__name__}: {p}")
    return data


def load_pipeline_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the pipeline YAML config and return a flat CONFIG dict.

    Resolution order for *path*:
        1. Explicit ``path`` argument.
        2. ``PS_CONFIG_PATH`` environment variable.
        3. ``configs/config.yaml`` next to the current working directory.
        4. ``configs/config.example.yaml`` as a last-resort fallback.

    The returned dict uses the standard pipeline configuration keys that ``run_pipeline.py``
    CONFIG expected (``vid_score_path``, ``log_csv_path`` …) and also embeds
    the two optional tuning profiles under ``tuning_overrides`` and
    ``tuning_overrides_50fps``.
    """
    # 1. Resolve the config file location.
    if path is None:
        path = os.environ.get("PS_CONFIG_PATH")
    if path is None:
        cwd_cfg = Path("configs") / "config.yaml"
        example_cfg = Path("configs") / "config.example.yaml"
        if cwd_cfg.exists():
            path = cwd_cfg
        elif example_cfg.exists():
            path = example_cfg
        else:
            raise FileNotFoundError(
                "No pipeline config found. Pass --config, set PS_CONFIG_PATH, "
                "or create configs/config.yaml (see configs/config.example.yaml)."
            )

    cfg_path = Path(_expand(str(path))).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    paths_section: dict[str, Any] = raw.get("paths", {}) or {}
    processing: dict[str, Any] = raw.get("processing", {}) or {}

    # 2. Flatten paths into the standard CONFIG shape and expand ~ / env vars.
    config: dict[str, Any] = {
        "vid_score_path": _expand(paths_section.get("vid_score_path")),
        "score_csv_path": _expand(paths_section.get("score_csv_path")),
        "score_column": paths_section.get("score_column", "ProS"),
        "hand_path": _expand(paths_section.get("hand_path")),
        "log_csv_path": _expand(paths_section.get("log_csv_path")),
        "save_dir": _expand(paths_section.get("save_dir")),
        "id2vid_csv_path": _expand(paths_section.get("id2vid_csv_path")),
        "video_quality_labels_csv_path": _expand(
            paths_section.get("video_quality_labels_csv_path")
        ),
        "video_quality_threshold": int(processing.get("video_quality_threshold", 3)),
        "n_videos": processing.get("n_videos"),
    }

    # 3. Load tuning profiles. Relative paths are resolved relative to the
    #    directory containing the YAML config file.
    profile_dir = cfg_path.parent
    tuning_name = raw.get("tuning_profile")
    if tuning_name:
        config["tuning_overrides"] = load_tuning_profile(
            _resolve_relative(tuning_name, profile_dir)
        )
    else:
        config["tuning_overrides"] = {}

    tuning_name_50 = raw.get("tuning_profile_50fps")
    if tuning_name_50:
        config["tuning_overrides_50fps"] = load_tuning_profile(
            _resolve_relative(tuning_name_50, profile_dir)
        )
    else:
        config["tuning_overrides_50fps"] = None

    # 4. Record where the config was loaded from so downstream code (e.g. the
    #    SUPERRES_MODEL_PATH resolver) can reference the repo root.
    config["_config_path"] = str(cfg_path)

    return config
