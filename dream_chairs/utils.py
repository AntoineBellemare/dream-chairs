"""Small shared helpers."""
from __future__ import annotations

import os

import numpy as np


def resolve_device(preference: str = "cuda") -> str:
    """Return 'cuda' if requested and available, else 'cpu'."""
    import torch

    if preference == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)
