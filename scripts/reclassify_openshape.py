"""Re-classify already-generated point clouds with the OpenShape backend.

Loads the forms in outputs/series/pointclouds/*.npz (from dream_series_demo.py) and
classifies them in OpenShape's native point-cloud<->text space — no rendering, no
regeneration. Writes report_openshape.json/.md alongside the originals so you can
compare against the CLIP multi-view results.

Usage:
  python scripts/reclassify_openshape.py
  python scripts/reclassify_openshape.py --swap-yz       # force y/z swap to test axis
  python scripts/reclassify_openshape.py --checkpoint openshape-pointbert-vitg14-rgb \
        --clip-backend openclip --clip-model ViT-bigG-14
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream_chairs import classify as C
from dream_chairs import cluster as CL
from dream_chairs import io_utils
from dream_chairs.config import load_config
from dream_chairs.embed import OpenShapeEmbedder
from dream_chairs.generate import Shape
from dream_chairs.utils import l2_normalize, resolve_device


def load_cloud(path: str) -> Shape:
    d = np.load(path)
    cid = os.path.splitext(os.path.basename(path))[0]
    colors = d["colors"] if "colors" in d.files else None
    return Shape(id=cid, prompt="", points=d["points"], colors=colors)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--pc-dir", default=str(ROOT / "outputs" / "series" / "pointclouds"))
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--descriptors", default=str(ROOT / "data" / "descriptors" / "visual_descriptors.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "series"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--clip-backend", default=None)
    ap.add_argument("--clip-model", default=None)
    ap.add_argument("--swap-yz", action="store_true")
    ap.add_argument("--num-clusters", type=int, default=4)
    args = ap.parse_args(argv)

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "cuda"))
    os_cfg = cfg.embedding.get("openshape", {})

    paths = sorted(glob.glob(os.path.join(args.pc_dir, "*.npz")))
    if not paths:
        print(f"no .npz point clouds in {args.pc_dir}; run dream_series_demo.py first")
        return 1
    shapes = [load_cloud(p) for p in paths]
    symbols = io_utils.load_symbols(args.symbols or cfg.paths["symbols"])
    import json
    with open(args.descriptors, encoding="utf-8") as f:
        dd = json.load(f)
    descriptors = list(dict.fromkeys(dd["descriptors"] if isinstance(dd, dict) else dd))

    checkpoint = args.checkpoint or os_cfg.get("checkpoint", "openshape-pointbert-vitl14-rgb")
    print(f"[openshape] device={device} checkpoint={checkpoint} clouds={len(shapes)} "
          f"symbols={len(symbols)} descriptors={len(descriptors)}")
    print("[openshape] loading encoder + CLIP text (first run downloads the encoder)...")

    embedder = OpenShapeEmbedder(
        device=device, checkpoint=checkpoint,
        clip_backend=args.clip_backend or os_cfg.get("clip_backend", "openai"),
        clip_model_name=args.clip_model or os_cfg.get("clip_model_name", "ViT-L/14"),
        clip_pretrained=os_cfg.get("clip_pretrained", "laion2b_s39b_b160k"),
        num_points=int(os_cfg.get("num_points", 10000)),
        swap_yz=True if args.swap_yz else os_cfg.get("swap_yz", None),
    )

    symbol_embs = C.build_symbol_embeddings(symbols, embedder,
                                            cfg.classification.get("descriptor_aggregation", "mean"))
    desc_embs = l2_normalize(embedder.embed_texts(descriptors), axis=1)

    report, shape_embs = [], []
    for shape in shapes:
        emb = embedder.embed_shape(shape)
        shape_embs.append(emb)
        sym = C.classify_shape(emb, symbol_embs, symbols, top_k=3)
        terms = C.rank_terms(emb, desc_embs, descriptors, k=8)
        print(f"  {shape.id}: " + ", ".join(f"{m['label']}({m['score']:.3f})" for m in sym)
              + "  |  " + ", ".join(t["term"] for t in terms[:5]))
        report.append({"id": shape.id, "symbol_matches": sym, "visual_qualities": terms})

    shape_embs = np.stack(shape_embs, axis=0)
    clusters = None
    if len(shapes) >= 2:
        labels = CL.cluster_shapes(shape_embs, "kmeans", min(args.num_clusters, len(shapes)))
        desc = CL.describe_clusters(shape_embs, labels, descriptors, desc_embs, top_terms=10)
        clusters = {"assignments": {report[i]["id"]: int(labels[i]) for i in range(len(report))},
                    "clusters": desc}

    io_utils.save_json(os.path.join(args.out, "report_openshape.json"),
                       {"backend": checkpoint, "results": report, "clusters": clusters})
    _write_md(os.path.join(args.out, "report_openshape.md"), checkpoint, report, clusters)
    print(f"\n[openshape] wrote {args.out}/report_openshape.(json|md)")
    if clusters:
        print("[openshape] clusters:", clusters["assignments"])
    return 0


def _write_md(path, checkpoint, report, clusters):
    lines = [f"# OpenShape classification ({checkpoint})\n",
             "Native point-cloud <-> text cosine similarity (no rendering).\n"]
    for r in report:
        sym = ", ".join(f"{m['label']} ({m['score']:.3f})" for m in r["symbol_matches"])
        q = ", ".join(f"{t['term']} ({t['score']:.3f})" for t in r["visual_qualities"][:8])
        lines += [f"## {r['id']}", f"- **Form -> symbols:** {sym}",
                  f"- **Form -> emergent qualities:** {q}\n"]
    if clusters:
        lines.append("## Emergent clusters\n")
        for c in clusters["clusters"]:
            members = [k for k, v in clusters["assignments"].items() if v == c["cluster"]]
            terms = ", ".join(t["term"] for t in c["top_terms"])
            lines += [f"- **Cluster {c['cluster']}** ({c['size']}): {', '.join(members)}",
                      f"  - characteristic qualities: {terms}\n"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
