"""End-to-end orchestration: dreams -> shapes -> embeddings -> symbols (+ clusters)."""
from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

from . import classify as classify_mod
from . import cluster as cluster_mod
from . import generate as generate_mod
from . import io_utils
from .config import DotDict
from .embed import build_embedder
from .utils import ensure_dir, resolve_device


def run(cfg: DotDict, limit: int | None = None, dry_run: bool = False,
        do_cluster: bool | None = None) -> dict:
    # avoid joblib/loky probing for physical cores (no wmic on modern Windows)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
    device = resolve_device(cfg.get("device", "cuda"))
    print(f"[dream-chairs] device={device}  dry_run={dry_run}")

    dreams = io_utils.load_dreams(cfg.paths["dreams"])
    symbols = io_utils.load_symbols(cfg.paths["symbols"])
    if limit:
        dreams = dreams[:limit]
    print(f"[dream-chairs] {len(dreams)} dreams, {len(symbols)} symbols")

    out_dir = ensure_dir(cfg.paths.get("output_dir", "outputs"))
    pc_dir = ensure_dir(os.path.join(out_dir, "pointclouds"))

    embedder = build_embedder(cfg, device)
    symbol_embs = classify_mod.build_symbol_embeddings(
        symbols, embedder, cfg.classification.get("descriptor_aggregation", "mean")
    )

    generator = None
    if not dry_run:
        generator = generate_mod.build_generator(cfg, device)

    num_points = int(cfg.generation.get("num_points", 4096))
    top_k = int(cfg.classification.get("top_k", 5))

    results = []
    shape_embs = []
    for i, dream in enumerate(tqdm(dreams, desc="dreams")):
        if dry_run:
            shape = generate_mod.random_shape(dream["id"], dream["text"], num_points, seed=i)
        else:
            shape = generator.generate(dream["id"], dream["text"])

        pc_path = os.path.join(pc_dir, f"{dream['id']}.npz")
        io_utils.save_point_cloud(pc_path, shape.points, shape.colors)

        emb = embedder.embed_shape(shape)
        shape_embs.append(emb)
        matches = classify_mod.classify_shape(emb, symbol_embs, symbols, top_k)
        results.append({
            "id": dream["id"],
            "text": dream["text"],
            "point_cloud": os.path.relpath(pc_path, out_dir),
            "matches": matches,
        })

    shape_embs = np.stack(shape_embs, axis=0)
    np.save(os.path.join(out_dir, "shape_embeddings.npy"), shape_embs)
    io_utils.save_json(os.path.join(out_dir, "results.json"), {"results": results})

    if do_cluster is None:
        do_cluster = bool(cfg.clustering.get("enabled", True))
    clusters = None
    if do_cluster and len(dreams) >= 2:
        labels = cluster_mod.cluster_shapes(
            shape_embs,
            cfg.clustering.get("algorithm", "kmeans"),
            int(cfg.clustering.get("num_clusters", 3)),
        )
        vocab_terms, vocab_embs = cluster_mod.build_vocab(symbols, embedder)
        descriptions = cluster_mod.describe_clusters(
            shape_embs, labels, vocab_terms, vocab_embs,
            int(cfg.clustering.get("top_terms", 8)),
        )
        clusters = {
            "assignments": {results[i]["id"]: int(labels[i]) for i in range(len(results))},
            "clusters": descriptions,
        }
        io_utils.save_json(os.path.join(out_dir, "clusters.json"), clusters)

    print(f"[dream-chairs] wrote results to {out_dir}/")
    return {"results": results, "clusters": clusters}
