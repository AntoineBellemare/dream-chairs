"""Match shape embeddings to symbol descriptors via cosine similarity."""
from __future__ import annotations

import numpy as np

from .utils import l2_normalize


def build_symbol_embeddings(symbols: list[dict], embedder, aggregation: str = "mean"):
    """Return (S, D) matrix: one embedding per symbol, aggregated over its descriptors."""
    vectors = []
    for s in symbols:
        descriptors = s.get("descriptors") or [s.get("label", s["id"])]
        feats = embedder.embed_texts(descriptors)  # (n, D) normalized
        if aggregation == "max":
            vec = feats.max(axis=0)
        else:
            vec = feats.mean(axis=0)
        vectors.append(vec)
    return l2_normalize(np.stack(vectors, axis=0), axis=1)


def classify_shape(shape_emb: np.ndarray, symbol_embs: np.ndarray,
                   symbols: list[dict], top_k: int = 5) -> list[dict]:
    sims = symbol_embs @ l2_normalize(shape_emb)  # (S,) cosine
    order = np.argsort(-sims)[:top_k]
    return [
        {"id": symbols[i]["id"], "label": symbols[i].get("label", symbols[i]["id"]),
         "score": float(sims[i])}
        for i in order
    ]


def rank_terms(shape_emb: np.ndarray, vocab_embs: np.ndarray,
               vocab_terms: list[str], k: int = 8) -> list[dict]:
    """Top-k descriptor terms from a (large) dictionary for one shape embedding.

    This is the per-shape side of Approach 1: cosine-match a generated form against
    a dictionary of visual descriptors to read off its characteristic qualities.
    """
    sims = vocab_embs @ l2_normalize(shape_emb)
    order = np.argsort(-sims)[:k]
    return [{"term": vocab_terms[i], "score": float(sims[i])} for i in order]
