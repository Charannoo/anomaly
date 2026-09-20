"""Minimal YAML config loader with `_base_` inheritance.

Supports a `_base_` (or `base`) key naming a sibling file in the same
directory, e.g. ``_base_: default.yaml``. Values in the child override the
base via a recursive dict merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving one level of ``_base_`` inheritance."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_key = cfg.pop("_base_", cfg.pop("base", None))
    if base_key is not None:
        base_path = (path.parent / str(base_key)).resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg
