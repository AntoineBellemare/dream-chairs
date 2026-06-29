"""Position shapes within families of design descriptors.

A "family" is a set of related design form-language terms (e.g. Volume =
{monolith, shell, lattice, membrane, layered}). For a shape embedding we compute a
soft position *within* each family — a distribution over its members — plus a
salience (how strongly the family applies at all). Scale-robust across embedding
backends because positions use within-family z-scored softmax, not raw cosine.
"""
from __future__ import annotations

import json

import numpy as np

from .utils import l2_normalize


def load_families(path: str) -> list[tuple[str, list[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fams = data["families"] if isinstance(data, dict) else data
    return [(f["name"], list(f["members"])) for f in fams]


def embed_families(families, embedder):
    """-> list of (name, members, member_embeddings[normalized])."""
    out = []
    for name, members in families:
        embs = l2_normalize(embedder.embed_texts(members), axis=1)
        out.append((name, members, embs))
    return out


def position(sims: np.ndarray, sharpness: float = 2.0) -> np.ndarray:
    """Within-family soft position: z-score the member cosines, then softmax.
    Independent of the absolute cosine scale, so it reads the same whether the
    backend produces ~0.1 (OpenShape) or ~0.25 (CLIP multi-view) similarities."""
    s = np.asarray(sims, dtype=float)
    z = (s - s.mean()) / (s.std() + 1e-6)
    p = np.exp((z - z.max()) * sharpness)
    return p / p.sum()


def shape_profile(shape_emb: np.ndarray, fam_embs, sharpness: float = 2.0) -> list[dict]:
    e = l2_normalize(shape_emb)
    profile = []
    for name, members, embs in fam_embs:
        sims = embs @ e
        p = position(sims, sharpness)
        j = int(np.argmax(p))
        profile.append({
            "family": name,
            "dominant": members[j],
            "dominant_p": float(p[j]),
            "salience": float(sims.max()),
            "distribution": {members[i]: float(p[i]) for i in range(len(members))},
            "cosines": {members[i]: float(sims[i]) for i in range(len(members))},
        })
    return profile
