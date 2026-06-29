"""Dream posture -> chair: generate-and-select loop (the body side of the bridge).

For each dream's target posture: build a chair prompt, generate K candidate forms,
embed each with OpenShape, project onto the posture axes, and keep the candidate whose
body best matches the dream's target. Candidates are standardized against a reference
corpus of forms so their posture scores are comparable to the target.

  python scripts/posture_chair.py --ids D003,D016 --variants 4 --backend point_e
Default backend is point_e (light; coexists with a running Shap-E job). Use
--backend shap_e --steps 64 for fabricable meshes when the GPU is free.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream_chairs import axes as AX
from dream_chairs import io_utils
from dream_chairs.config import load_config
from dream_chairs.embed import build_embedder, render_point_views
from dream_chairs.generate import build_generator
from dream_chairs.utils import ensure_dir, resolve_device


def _font(sz):
    for p in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--targets", default=str(ROOT / "outputs" / "dream_posture_targets.json"))
    ap.add_argument("--axes", default=str(ROOT / "data" / "descriptors" / "body_posture_axes.json"))
    ap.add_argument("--ids", default=None, help="comma-separated dream ids (default: first 2)")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--backend", default="point_e", help="point_e | shap_e")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--ref-tag", default="pointe", help="dataset whose forms set the posture reference")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "chairs"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.embedding["backend"] = "openshape"
    cfg.generation["backend"] = args.backend
    if args.steps is not None:
        cfg.generation.setdefault("shap_e", {})["karras_steps"] = args.steps
    device = resolve_device(cfg.get("device", "cuda"))

    axes = AX.load_axes(args.axes)
    names = [a["name"] for a in axes]
    targets = {d["id"]: d for d in json.load(open(args.targets, encoding="utf-8"))["dreams"]}
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else list(targets.keys())[:2]

    out = ensure_dir(args.out)
    embedder = build_embedder(cfg, device)
    axis_embs = AX.embed_axes(axes, embedder)

    # posture reference from an existing corpus of forms -> comparable candidate scores
    ref = np.load(os.path.join(ROOT, "outputs", f"dataset_{args.ref_tag}",
                               "shape_embeddings_openshape.npz"), allow_pickle=True)
    ref_raw = AX.project_raw(ref["embs"], axis_embs)
    rmean, rstd = ref_raw.mean(0), ref_raw.std(0) + 1e-9

    gen = build_generator(cfg, device)
    report = []
    for did in ids:
        tgt = np.array([targets[did]["scores"][n] for n in names])
        prompt = AX.chair_prompt(names, tgt)
        print(f"\n{did}: {targets[did]['text']}\n  target posture -> {prompt!r}")
        cands = []
        for k in range(args.variants):
            shape = gen.generate(f"{did}_v{k}", prompt)
            emb = embedder.embed_shape(shape)
            raw = AX.project_raw(emb[None], axis_embs)[0]
            score = np.tanh(((raw - rmean) / rstd) / 2.0)
            dist = float(np.linalg.norm(score - tgt))
            view = render_point_views(shape.points, shape.colors, 1, 220, 2)[0]
            cands.append({"k": k, "dist": dist, "score": score, "view": view, "shape": shape})
            print(f"    v{k}: posture-distance {dist:.2f}")
        cands.sort(key=lambda c: c["dist"])
        best = cands[0]
        # save best geometry
        io_utils.save_point_cloud(os.path.join(out, f"{did}_best.npz"),
                                  best["shape"].points, best["shape"].colors)
        if best["shape"].mesh is not None:
            best["shape"].mesh.export(os.path.join(out, f"{did}_best.obj"))
        _render_selection(os.path.join(out, f"{did}.png"), did, targets[did]["text"],
                          prompt, cands, names, tgt)
        report.append({"id": did, "text": targets[did]["text"], "prompt": prompt,
                       "target": {names[a]: round(float(tgt[a]), 2) for a in range(len(names))},
                       "chosen_variant": best["k"], "chosen_distance": round(best["dist"], 3),
                       "chosen_posture": {names[a]: round(float(best["score"][a]), 2) for a in range(len(names))},
                       "candidates": [{"k": c["k"], "distance": round(c["dist"], 3)} for c in cands]})

    io_utils.save_json(os.path.join(out, "chairs_report.json"), {"backend": args.backend, "results": report})
    print(f"\n[posture-chair] wrote chairs -> {out} (per-dream PNG, *_best.npz/.obj, chairs_report.json)")
    return 0


def _render_selection(path, did, text, prompt, cands, names, tgt):
    cell, pad = 220, 8
    K = len(cands)
    fid, ftx, fsm = _font(16), _font(13), _font(12)
    W = max(760, pad + K * (cell + pad))
    H = 70 + cell + 26
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((pad, 6), f"{did}  —  {text}", fill=(20, 20, 20), font=fid)
    d.text((pad, 28), f"chair prompt: {prompt}", fill=(90, 90, 90), font=ftx)
    d.text((pad, 48), "generated variants ranked by posture-distance to the dream (chosen = green):",
           fill=(120, 120, 120), font=fsm)
    for i, c in enumerate(cands):
        x = pad + i * (cell + pad)
        y = 70
        img.paste(c["view"], (x, y))
        chosen = (i == 0)
        col = (29, 158, 117) if chosen else (200, 200, 200)
        d.rectangle([x, y, x + cell - 1, y + cell - 1], outline=col, width=3 if chosen else 1)
        tag = f"dist {c['dist']:.2f}" + ("  ✓ chosen" if chosen else "")
        d.text((x + 4, y + cell + 4), tag, fill=col if chosen else (120, 120, 120), font=fsm)
    img.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
