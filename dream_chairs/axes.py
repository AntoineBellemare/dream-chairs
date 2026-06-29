"""Bipolar posture-axis projection — the dream<->body<->chair bridge.

Projects a shape embedding (OpenShape) or a text embedding (CLIP) onto each bipolar
axis as a signed score = cos(emb, high_pole) - cos(emb, low_pole). Scores are z-scored
across a corpus so they're comparable regardless of the backend's cosine scale, and a
position -> chair-parameter map turns an axis coordinate into readable chair geometry.

Shapes and texts live in the SAME CLIP-ViT-L/14 space (OpenShape aligns 3D to it), so a
dream report and a generated form can be placed on the same posture axes.
"""
from __future__ import annotations

import json

import numpy as np

from .utils import l2_normalize


def load_axes(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["axes"] if isinstance(data, dict) else data


def embed_axes(axes: list[dict], embedder):
    """-> list of {name, hi, lo, axis} with hi/lo = normalized mean pole embeddings."""
    out = []
    for ax in axes:
        hi = l2_normalize(embedder.embed_texts(ax["high"]).mean(axis=0))
        lo = l2_normalize(embedder.embed_texts(ax["low"]).mean(axis=0))
        out.append({"name": ax["name"], "hi": hi, "lo": lo, "axis": ax})
    return out


def project_raw(embs: np.ndarray, axis_embs) -> np.ndarray:
    """embs: (n, d) normalized -> raw scores (n, a) = cos(hi) - cos(lo) per axis."""
    E = l2_normalize(embs, axis=1)
    cols = []
    for ae in axis_embs:
        cols.append(E @ ae["hi"] - E @ ae["lo"])
    return np.stack(cols, axis=1)


def standardize(raw: np.ndarray):
    """z-score each axis across the corpus. Returns (z, mean, std) so the same
    reference can later place a single new dream/shape on the corpus scale."""
    mean = raw.mean(axis=0)
    std = raw.std(axis=0) + 1e-9
    return (raw - mean) / std, mean, std


def position_unit(z: np.ndarray) -> np.ndarray:
    """Map z-score to [0,1] along the axis (clipped at +-2 sd) for chair interpolation."""
    return np.clip((z + 2.0) / 4.0, 0.0, 1.0)


# chair-wording per axis: (low-pole phrase, high-pole phrase) — used to turn a target
# posture into a chair generation prompt.
CHAIR_PHRASES = {
    "recumbent_upright": ("deeply reclined and low", "upright and tall"),
    "exposed_cocooned": ("open and exposed", "with a high enclosing wrap-around back"),
    "floating_grounded": ("on slender light legs", "on a heavy grounded base"),
    "collapsed_taut": ("soft, slumped and rounded", "rigid with sharp taut edges"),
    "contracted_dilated": ("compact and contracted", "wide, open and expansive"),
    "stable_unstable": ("balanced and symmetric", "tilted and off-balance"),
}


def chair_prompt(axis_names: list[str], scores, thresh: float = 0.35) -> str:
    """Build a Shap-E/Point-E chair prompt from a target posture (scores in [-1,1])."""
    parts = []
    for i, name in enumerate(axis_names):
        lo_hi = CHAIR_PHRASES.get(name)
        if lo_hi is None:
            continue
        v = float(scores[i])
        if v >= thresh:
            parts.append(lo_hi[1])
        elif v <= -thresh:
            parts.append(lo_hi[0])
    return "a chair, " + (", ".join(parts) if parts else "simple balanced form")


def posture_to_chair(axis_names: list[str], z_scores: np.ndarray, axes: list[dict]) -> dict:
    """Turn one form's axis z-scores into chair parameters via each axis's [lo,hi] map."""
    by_name = {a["name"]: a for a in axes}
    t = position_unit(z_scores)
    params = {}
    for i, name in enumerate(axis_names):
        for pname, (lo, hi) in by_name[name].get("chair", {}).items():
            params[pname] = round(float(lo + t[i] * (hi - lo)), 3)
    return params
