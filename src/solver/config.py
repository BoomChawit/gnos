from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def get_path(config_path: str | Path, maybe_relative: str | Path) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (Path(config_path).resolve().parent / p).resolve()

