from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in {path}")
    return data


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    cfg = _read_yaml(path)

    inherits = cfg.pop("inherits", None)
    if not inherits:
        return cfg

    parent_path = (path.parent / inherits).resolve()
    parent_cfg = load_config(parent_path)
    return _deep_merge(parent_cfg, cfg)
