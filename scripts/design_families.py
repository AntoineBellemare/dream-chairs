"""Position each generated dream shape within families of design descriptors.

Loads the saved point clouds, embeds them (OpenShape by default), and for each shape
computes a soft position within every design family. Writes:
  outputs/series/design_profiles.json   full profiles (distribution + salience)
  outputs/series/design_profiles.md     readable per-dream + per-family spread
  outputs/series/design_matrix.json     dreams x members matrix (for the heatmap)
  outputs/series/shape_embeddings_openshape.npz   cached embeddings (reused next time)

Usage:
  python scripts/design_families.py
  python scripts/design_families.py --backend clip_multiview
  python scripts/design_families.py --sharpness 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream_chairs import families as FAM
from dream_chairs import io_utils
from dream_chairs.config import load_config
from dream_chairs.embed import build_embedder
from dream_chairs.generate import Shape
from dream_chairs.utils import resolve_device


def load_cloud(path: str) -> Shape:
    d = np.load(path)
    cid = os.path.splitext(os.path.basename(path))[0]
    colors = d["colors"] if "colors" in d.files else None
    return Shape(id=cid, prompt="", points=d["points"], colors=colors)


def dream_texts(path: str) -> dict:
    try:
        return {d["id"]: d["text"] for d in io_utils.load_dreams(path)}
    except Exception:
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--pc-dir", default=str(ROOT / "outputs" / "series" / "pointclouds"))
    ap.add_argument("--families", default=str(ROOT / "data" / "descriptors" / "design_families.json"))
    ap.add_argument("--dreams", default=str(ROOT / "data" / "dreams" / "dream_series.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "series"))
    ap.add_argument("--backend", default="openshape", help="openshape | clip_multiview")
    ap.add_argument("--sharpness", type=float, default=2.0)
    ap.add_argument("--no-cache", action="store_true", help="ignore cached embeddings")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.embedding["backend"] = args.backend
    device = resolve_device(cfg.get("device", "cuda"))

    paths = sorted(glob.glob(os.path.join(args.pc_dir, "*.npz")))
    if not paths:
        print(f"no point clouds in {args.pc_dir}; run dream_series_demo.py first")
        return 1
    shapes = [load_cloud(p) for p in paths]
    ids = [s.id for s in shapes]
    families = FAM.load_families(args.families)
    texts = dream_texts(args.dreams)

    cache = os.path.join(args.out, f"shape_embeddings_{args.backend}.npz")
    embedder = None
    fam_embs = None
    shape_embs = None

    if os.path.exists(cache) and not args.no_cache:
        c = np.load(cache, allow_pickle=True)
        if list(c["ids"]) == ids:
            shape_embs = c["embs"]
            print(f"[families] loaded cached shape embeddings: {cache}")

    print(f"[families] backend={args.backend} shapes={len(shapes)} "
          f"families={len(families)} sharpness={args.sharpness}")
    if shape_embs is None:
        print("[families] building embedder + embedding shapes...")
        embedder = build_embedder(cfg, device)
        shape_embs = np.stack([embedder.embed_shape(s) for s in shapes], axis=0)
        np.savez_compressed(cache, ids=np.array(ids), embs=shape_embs)
        print(f"[families] cached embeddings -> {cache}")
    if embedder is None:
        embedder = build_embedder(cfg, device)  # needed for family text embeddings
    fam_embs = FAM.embed_families(families, embedder)

    results = []
    for i, s in enumerate(shapes):
        prof = FAM.shape_profile(shape_embs[i], fam_embs, args.sharpness)
        results.append({"id": s.id, "text": texts.get(s.id, ""), "profile": prof})

    # full profiles + matrix for viz
    io_utils.save_json(os.path.join(args.out, "design_profiles.json"),
                       {"backend": args.backend, "sharpness": args.sharpness, "results": results})
    matrix = {
        "dreams": [{"id": r["id"], "text": r["text"]} for r in results],
        "families": [{"name": n, "members": m} for n, m in families],
        "values": {r["id"]: {mem: r["profile"][fi]["distribution"][mem]
                             for fi, (n, m) in enumerate(families) for mem in m}
                   for r in results},
        "salience": {r["id"]: {r["profile"][fi]["family"]: r["profile"][fi]["salience"]
                               for fi in range(len(families))} for r in results},
    }
    io_utils.save_json(os.path.join(args.out, "design_matrix.json"), matrix)
    _write_md(os.path.join(args.out, "design_profiles.md"), args.backend, results, families)

    # console summary
    print("\n=== design family positioning (dominant member per family) ===")
    for r in results:
        print(f"\n{r['id']}  {r['text']}")
        for p in r["profile"]:
            print(f"   {p['family']:<12}: {p['dominant']:<26} "
                  f"(p={p['dominant_p']:.2f}, salience={p['salience']:.3f})")
    print(f"\n[families] wrote design_profiles.(json|md) + design_matrix.json -> {args.out}")
    return 0


def _write_md(path, backend, results, families):
    lines = [f"# Dream shapes positioned in design families ({backend})\n",
             "For each shape, the dominant member within each family (with its "
             "within-family probability and the family's salience).\n"]
    for r in results:
        lines.append(f"## {r['id']} — {r['text']}")
        for p in r["profile"]:
            dist = ", ".join(f"{k} {v:.2f}" for k, v in
                             sorted(p["distribution"].items(), key=lambda kv: -kv[1]))
            lines.append(f"- **{p['family']}** → _{p['dominant']}_ "
                         f"(p={p['dominant_p']:.2f}, salience={p['salience']:.3f}) — {dist}")
        lines.append("")
    # per-family spread across dreams
    lines.append("## Spread across dreams (who dominates each family member)\n")
    for fi, (name, members) in enumerate(families):
        lines.append(f"### {name}")
        for mem in members:
            who = [r["id"] for r in results if r["profile"][fi]["dominant"] == mem]
            lines.append(f"- {mem}: {', '.join(who) if who else '—'}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
