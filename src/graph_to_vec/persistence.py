"""Run artifact persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_yaml(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_joblib(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    return path


def load_joblib(path: str | Path) -> Any:
    return joblib.load(Path(path))


def save_torch(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(obj, path)
    return path


def load_torch(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def save_embeddings(
    embeddings: np.ndarray | dict[str, np.ndarray],
    path: str | Path,
    ids: list[Any] | dict[str, list[Any]] | None = None,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)

    if isinstance(embeddings, dict):
        frames = []
        for node_type, matrix in embeddings.items():
            frame = pd.DataFrame(matrix)
            frame.insert(0, "node_type", node_type)
            if isinstance(ids, dict) and node_type in ids:
                frame.insert(1, "node_id", ids[node_type])
            frames.append(frame)
        output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        output = pd.DataFrame(embeddings)
        if isinstance(ids, list):
            output.insert(0, "id", ids)

    if path.suffix.lower() == ".parquet":
        output.to_parquet(path, index=False)
    else:
        output.to_csv(path, index=False)
    return path
