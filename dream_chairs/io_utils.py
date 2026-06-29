"""Loading dreams/symbols and saving point clouds + results."""
from __future__ import annotations

import json
from typing import Any

import numpy as np


def load_dreams(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dreams = data["dreams"] if isinstance(data, dict) else data
    out = []
    for i, d in enumerate(dreams):
        if isinstance(d, str):
            d = {"id": f"d{i:03d}", "text": d}
        out.append({"id": d.get("id", f"d{i:03d}"), "text": d["text"]})
    return out


def load_symbols(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    symbols = data["symbols"] if isinstance(data, dict) else data
    for s in symbols:
        s.setdefault("descriptors", [s.get("label", s["id"])])
    return symbols


def save_point_cloud(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    if colors is None:
        np.savez_compressed(path, points=points.astype(np.float32))
    else:
        np.savez_compressed(path, points=points.astype(np.float32), colors=colors.astype(np.float32))


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
