"""Dream series -> Shap-E 3D forms -> classification + emergent-quality analysis.

Implements both approaches from the project emails:

  Approach 2 (symbol-guided generation): pick the top symbol for each dream from its
  TEXT, then fold that symbol's spatial/architectural qualities into the Shap-E prompt.

  Approach 1 (inductive / emergent qualities): embed each generated point cloud and
  cosine-match it against a LARGE dictionary of visual descriptors, then cluster the
  forms and describe each cluster by its characteristic emergent descriptors.

Outputs (outputs/series/):
  meshes/<id>.obj          generated mesh (open in Windows 3D Viewer / Blender)
  pointclouds/<id>.npz     sampled point cloud (points [+ colors])
  previews/<id>.png        multi-view montage of the form
  overview.png             all forms side by side
  report.json / report.md  full results + cluster analysis

Usage:
  python scripts/dream_series_demo.py                 # all dreams
  python scripts/dream_series_demo.py --limit 2       # first 2 (quick test)
  python scripts/dream_series_demo.py --steps 48      # faster, lower quality
  python scripts/dream_series_demo.py --no-guidance   # plain dream prompts
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream_chairs import classify as C
from dream_chairs import cluster as CL
from dream_chairs import io_utils
from dream_chairs.config import load_config
from dream_chairs.embed import build_embedder, render_point_views
from dream_chairs.generate import ShapEGenerator
from dream_chairs.utils import ensure_dir, l2_normalize, resolve_device


def load_descriptors(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    terms = data["descriptors"] if isinstance(data, dict) else data
    return list(dict.fromkeys(terms))  # dedupe, preserve order


def augment_prompt(text: str, symbol: dict) -> str:
    """Approach 2: fold the guiding symbol's spatial qualities into the prompt."""
    quals = (symbol.get("spatial_qualities") or [])[:4]
    prompt = f"a sculptural form of {text}"
    if quals:
        prompt += ", " + ", ".join(quals)
    return prompt


def montage(images: list[Image.Image], cols: int = 4, pad: int = 4) -> Image.Image:
    n = len(images)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    w, h = images[0].size
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad),
                       (255, 255, 255))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return canvas


def labeled(image: Image.Image, text: str) -> Image.Image:
    im = image.copy()
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, im.width, 14], fill=(20, 20, 20))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return im


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dream-series-demo")
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--dreams", default=str(ROOT / "data" / "dreams" / "dream_series.json"))
    ap.add_argument("--symbols", default=None, help="defaults to paths.symbols in config")
    ap.add_argument("--descriptors", default=str(ROOT / "data" / "descriptors" / "visual_descriptors.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "series"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=64, help="Shap-E karras steps")
    ap.add_argument("--guidance-scale", type=float, default=15.0)
    ap.add_argument("--no-guidance", action="store_true", help="disable symbol-guided prompts")
    ap.add_argument("--num-clusters", type=int, default=4)
    args = ap.parse_args(argv)

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "cuda"))

    dreams = io_utils.load_dreams(args.dreams)
    if args.limit:
        dreams = dreams[: args.limit]
    symbols = io_utils.load_symbols(args.symbols or cfg.paths["symbols"])
    descriptors = load_descriptors(args.descriptors)

    print(f"[demo] device={device}  dreams={len(dreams)}  symbols={len(symbols)}  "
          f"descriptors={len(descriptors)}  guidance={not args.no_guidance}  steps={args.steps}")

    out = ensure_dir(args.out)
    mesh_dir = ensure_dir(os.path.join(out, "meshes"))
    pc_dir = ensure_dir(os.path.join(out, "pointclouds"))
    prev_dir = ensure_dir(os.path.join(out, "previews"))

    # --- embedding space: text (symbols + descriptor dictionary) ---
    embedder = build_embedder(cfg, device)
    symbol_embs = C.build_symbol_embeddings(
        symbols, embedder, cfg.classification.get("descriptor_aggregation", "mean"))
    desc_embs = l2_normalize(embedder.embed_texts(descriptors), axis=1)

    print("[demo] loading Shap-E (first run downloads weights)...")
    gen = ShapEGenerator(device=device, guidance_scale=args.guidance_scale,
                         karras_steps=args.steps,
                         num_points=int(cfg.generation.get("num_points", 4096)))

    report = []
    shape_embs = []
    overview_tiles = []
    # preview render params come from config (independent of the embedding backend, so
    # previews work with both clip_multiview and openshape)
    mv = cfg.embedding.get("multiview", {})
    nv = int(mv.get("num_views", 8))
    prev_size = int(mv.get("image_size", 224))
    prev_psize = int(mv.get("point_size", 2))

    for i, d in enumerate(dreams):
        # Approach 2: choose guiding symbol from the dream TEXT
        dtext_emb = l2_normalize(embedder.embed_texts([d["text"]])[0])
        sym_text_sims = symbol_embs @ dtext_emb
        guide = symbols[int(np.argmax(sym_text_sims))]
        prompt = d["text"] if args.no_guidance else augment_prompt(d["text"], guide)

        print(f"\n[{i+1}/{len(dreams)}] {d['id']}: {d['text']}")
        print(f"        guide-symbol={guide['label']}  prompt={prompt!r}")
        shape = gen.generate(d["id"], prompt)

        shape.mesh.export(os.path.join(mesh_dir, f"{d['id']}.obj"))
        io_utils.save_point_cloud(os.path.join(pc_dir, f"{d['id']}.npz"),
                                  shape.points, shape.colors)

        emb = embedder.embed_shape(shape)
        shape_embs.append(emb)

        # multi-view preview
        views = render_point_views(shape.points, shape.colors, nv, prev_size, prev_psize)
        montage(views).save(os.path.join(prev_dir, f"{d['id']}.png"))
        overview_tiles.append(labeled(views[0], d["id"]))

        # classification of the FORM against symbols
        sym_matches = C.classify_shape(emb, symbol_embs, symbols, top_k=3)
        # Approach 1: characteristic visual descriptors of the FORM
        form_terms = C.rank_terms(emb, desc_embs, descriptors, k=8)

        print("        form->symbols: " +
              ", ".join(f"{m['label']}({m['score']:.2f})" for m in sym_matches))
        print("        form->qualities: " +
              ", ".join(t["term"] for t in form_terms[:6]))

        report.append({
            "id": d["id"], "text": d["text"],
            "guidance_symbol": guide["id"], "prompt": prompt,
            "symbol_matches": sym_matches,
            "visual_qualities": form_terms,
        })

    shape_embs = np.stack(shape_embs, axis=0)
    np.save(os.path.join(out, "shape_embeddings.npy"), shape_embs)

    if overview_tiles:
        montage(overview_tiles, cols=4).save(os.path.join(out, "overview.png"))

    # --- Approach 1: cluster forms + describe with the large dictionary ---
    clusters = None
    if len(dreams) >= 2:
        labels = CL.cluster_shapes(shape_embs, "kmeans",
                                   min(args.num_clusters, len(dreams)))
        cluster_desc = CL.describe_clusters(shape_embs, labels, descriptors, desc_embs,
                                            top_terms=10)
        clusters = {
            "assignments": {report[i]["id"]: int(labels[i]) for i in range(len(report))},
            "clusters": cluster_desc,
        }

    io_utils.save_json(os.path.join(out, "report.json"),
                       {"results": report, "clusters": clusters})
    write_markdown(os.path.join(out, "report.md"), report, clusters)

    print(f"\n[demo] done -> {out}")
    print(f"[demo] view forms: outputs/series/overview.png  +  meshes/*.obj")
    if clusters:
        print("[demo] cluster assignments:", clusters["assignments"])
    return 0


def write_markdown(path: str, report: list[dict], clusters: dict | None) -> None:
    lines = ["# Dream series — Shap-E forms & symbol classification\n"]
    for r in report:
        lines.append(f"## {r['id']} — {r['text']}")
        lines.append(f"- **Guiding symbol (from text):** {r['guidance_symbol']}")
        lines.append(f"- **Prompt:** `{r['prompt']}`")
        sym = ", ".join(f"{m['label']} ({m['score']:.3f})" for m in r["symbol_matches"])
        lines.append(f"- **Form → symbols:** {sym}")
        q = ", ".join(f"{t['term']} ({t['score']:.3f})" for t in r["visual_qualities"][:8])
        lines.append(f"- **Form → emergent qualities:** {q}")
        lines.append(f"- ![preview](previews/{r['id']}.png)\n")
    if clusters:
        lines.append("## Emergent clusters\n")
        for c in clusters["clusters"]:
            members = [k for k, v in clusters["assignments"].items() if v == c["cluster"]]
            terms = ", ".join(t["term"] for t in c["top_terms"])
            lines.append(f"- **Cluster {c['cluster']}** ({c['size']}): {', '.join(members)}")
            lines.append(f"  - characteristic qualities: {terms}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
