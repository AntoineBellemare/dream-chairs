"""Cluster shape embeddings and surface each cluster's emergent visual terms.

This is the data-driven 'emergent visual qualities per dream cluster' analysis
Antoine proposed to Mark: group generated shapes, then describe each group by the
descriptor terms its centroid is closest to.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np

from .utils import l2_normalize


def cluster_shapes(shape_embs: np.ndarray, algorithm: str = "kmeans",
                   num_clusters: int = 3) -> np.ndarray:
    n = shape_embs.shape[0]
    k = max(1, min(num_clusters, n))
    # loky probes for physical cores via a subprocess absent on modern Windows and
    # dumps a harmless traceback to stderr; the result is unused here, so mute stderr
    # for the fit (real failures still raise and surface after the block).
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        if algorithm == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            model = AgglomerativeClustering(n_clusters=k)
        else:
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=k, n_init=10, random_state=0)
        return model.fit_predict(shape_embs)


def describe_clusters(shape_embs: np.ndarray, labels: np.ndarray,
                      vocab_terms: list[str], vocab_embs: np.ndarray,
                      top_terms: int = 8) -> list[dict]:
    """For each cluster, rank the descriptor vocabulary by similarity to the centroid."""
    out = []
    for c in sorted(set(int(x) for x in labels)):
        mask = labels == c
        centroid = l2_normalize(shape_embs[mask].mean(axis=0))
        sims = vocab_embs @ centroid
        order = np.argsort(-sims)[:top_terms]
        out.append({
            "cluster": int(c),
            "size": int(mask.sum()),
            "top_terms": [{"term": vocab_terms[i], "score": float(sims[i])} for i in order],
        })
    return out


def build_vocab(symbols: list[dict], embedder):
    """Unique descriptor phrases across all symbols + their embeddings."""
    terms: list[str] = []
    seen = set()
    for s in symbols:
        for d in s.get("descriptors", []):
            if d not in seen:
                seen.add(d)
                terms.append(d)
    embs = embedder.embed_texts(terms) if terms else np.zeros((0, 1), dtype=np.float32)
    return terms, l2_normalize(embs, axis=1)
