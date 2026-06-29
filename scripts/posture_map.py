"""Project generated dream forms onto the body-posture axes (the dream<->chair bridge).

Reuses the cached OpenShape embeddings of a dataset (no re-embedding), projects each
form onto every bipolar posture axis, and writes per-form posture coordinates + chair
parameters. Prints the extreme exemplars per axis so you can sanity-check that the axes
read meaningfully on the geometry.

  python scripts/posture_map.py --tag pointe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream_chairs import axes as AX
from dream_chairs import io_utils
from dream_chairs.config import load_config
from dream_chairs.embed import build_embedder
from dream_chairs.utils import resolve_device


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--tag", default="pointe")
    ap.add_argument("--axes", default=str(ROOT / "data" / "descriptors" / "body_posture_axes.json"))
    ap.add_argument("--dreams", default=str(ROOT / "data" / "dreams" / "dream_dataset.json"))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    ap.add_argument("--clean", action="store_true", help="restrict to clean_ids.json; writes *_clean")
    args = ap.parse_args(argv)
    suffix = "_clean" if args.clean else ""

    cfg = load_config(args.config)
    cfg.embedding["backend"] = "openshape"
    device = resolve_device(cfg.get("device", "cuda"))
    base = os.path.join(args.out_root, f"dataset_{args.tag}")

    cache = os.path.join(base, "shape_embeddings_openshape.npz")
    if not os.path.exists(cache):
        print(f"no cached OpenShape embeddings at {cache}; run cluster_dataset.py --tag {args.tag} first")
        return 1
    c = np.load(cache, allow_pickle=True)
    ids = list(c["ids"])
    E = c["embs"]
    if args.clean:
        cids = set(json.load(open(os.path.join(base, "clean_ids.json"), encoding="utf-8"))["ids"])
        keep = [i for i, d in enumerate(ids) if d in cids]
        ids = [ids[i] for i in keep]; E = E[keep]
    print(f"[posture] {len(ids)} forms, embeddings {E.shape}")

    meta = {d["id"]: d for d in json.load(open(args.dreams, encoding="utf-8"))["dreams"]}
    clusters = {}
    rep_path = os.path.join(base, f"cluster_report{suffix}.json")
    if os.path.exists(rep_path):
        rep = json.load(open(rep_path, encoding="utf-8"))
        for cl in rep["clusters"]:
            for m in cl["members"]:
                clusters[m] = cl["cluster"]

    axes = AX.load_axes(args.axes)
    axis_names = [a["name"] for a in axes]
    embedder = build_embedder(cfg, device)  # for pole text embeddings (same CLIP space)
    axis_embs = AX.embed_axes(axes, embedder)

    raw = AX.project_raw(E, axis_embs)
    Z, mean, std = AX.standardize(raw)

    profiles = []
    for i, did in enumerate(ids):
        zrow = Z[i]
        profiles.append({
            "id": did, "text": meta.get(did, {}).get("text", ""),
            "tag": meta.get(did, {}).get("tag", ""),
            "cluster": clusters.get(did, -1),
            "scores": {axis_names[a]: float(zrow[a]) for a in range(len(axis_names))},
            "chair": AX.posture_to_chair(axis_names, zrow, axes),
        })

    io_utils.save_json(os.path.join(base, f"posture_profiles{suffix}.json"),
                       {"tag": args.tag, "axes": axis_names, "profiles": profiles})
    io_utils.save_json(os.path.join(base, f"posture_matrix{suffix}.json"), {
        "axes": [{"name": a["name"], "low": a["low"][0], "high": a["high"][0],
                  "phenomenology": a["phenomenology"]} for a in axes],
        "dreams": [{"id": p["id"], "text": p["text"], "tag": p["tag"], "cluster": p["cluster"],
                    "z": [round(p["scores"][n], 3) for n in axis_names]} for p in profiles],
    })

    # sanity check: extremes per axis
    print("\n=== axis extremes (does geometry read as posture?) ===")
    for a, name in enumerate(axis_names):
        order = np.argsort(Z[:, a])
        lo = ids[order[0]]
        hi = ids[order[-1]]
        ax = axes[a]
        print(f"\n{name}  [{ax['low'][0]}  <->  {ax['high'][0]}]")
        print(f"   low : {lo}  {meta.get(lo, {}).get('text', '')}")
        print(f"   high: {hi}  {meta.get(hi, {}).get('text', '')}")

    # example chair params for two extreme dreams on axis 0
    print("\n=== example dream -> chair params ===")
    o0 = np.argsort(Z[:, 0])
    for did_i in (o0[0], o0[-1]):
        did = ids[did_i]
        print(f"\n{did}: {meta.get(did, {}).get('text', '')}")
        print("  ", AX.posture_to_chair(axis_names, Z[did_i], axes))
    print(f"\n[posture] wrote posture_profiles{suffix}.json + posture_matrix{suffix}.json -> {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
