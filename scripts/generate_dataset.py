"""Generate a 3D form for every dream in a corpus (resumable).

Generator-agnostic (Point-E or Shap-E); outputs are namespaced by --tag so multiple
generators can be kept side by side for comparison. Saves a point cloud per dream, a
thumbnail, and a contact sheet of the whole set.

  python scripts/generate_dataset.py --backend point_e --tag pointe
  python scripts/generate_dataset.py --backend shap_e --steps 64 --tag shape64
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream_chairs import io_utils
from dream_chairs.config import load_config
from dream_chairs.embed import render_point_views
from dream_chairs.generate import Shape, build_generator
from dream_chairs.utils import ensure_dir, resolve_device


def contact_sheet(thumbs, cols=10, pad=3) -> Image.Image:
    if not thumbs:
        return Image.new("RGB", (10, 10), (255, 255, 255))
    cols = min(cols, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    w, h = thumbs[0].size
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), (255, 255, 255))
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        canvas.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return canvas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--dreams", default=str(ROOT / "data" / "dreams" / "dream_dataset.json"))
    ap.add_argument("--backend", default="point_e", help="point_e | shap_e")
    ap.add_argument("--tag", default=None, help="output namespace (default = backend)")
    ap.add_argument("--steps", type=int, default=None, help="shap_e karras steps override")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thumb", type=int, default=150, help="thumbnail size px")
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    args = ap.parse_args(argv)

    tag = args.tag or args.backend
    cfg = load_config(args.config)
    cfg.generation["backend"] = args.backend
    if args.steps is not None:
        cfg.generation.setdefault("shap_e", {})["karras_steps"] = args.steps
    device = resolve_device(cfg.get("device", "cuda"))

    dreams = io_utils.load_dreams(args.dreams)
    if args.limit:
        dreams = dreams[: args.limit]
    out = ensure_dir(os.path.join(args.out_root, f"dataset_{tag}"))
    pc_dir = ensure_dir(os.path.join(out, "pointclouds"))
    th_dir = ensure_dir(os.path.join(out, "thumbs"))
    mesh_dir = ensure_dir(os.path.join(out, "meshes"))  # populated when the generator yields meshes (Shap-E)

    todo = [d for d in dreams if not os.path.exists(os.path.join(pc_dir, f"{d['id']}.npz"))]
    print(f"[gen] backend={args.backend} tag={tag} device={device} "
          f"dreams={len(dreams)} to_generate={len(todo)} (resuming {len(dreams)-len(todo)})")

    gen = build_generator(cfg, device) if todo else None

    thumbs = []
    for d in tqdm(dreams, desc=f"gen[{tag}]"):
        pc_path = os.path.join(pc_dir, f"{d['id']}.npz")
        if os.path.exists(pc_path):
            z = np.load(pc_path)
            points = z["points"]
            colors = z["colors"] if "colors" in z.files else None
        else:
            shape = gen.generate(d["id"], d["text"])
            points, colors = shape.points, shape.colors
            io_utils.save_point_cloud(pc_path, points, colors)
            if shape.mesh is not None:
                try:
                    shape.mesh.export(os.path.join(mesh_dir, f"{d['id']}.obj"))
                except Exception as e:
                    print(f"  [warn] mesh export failed for {d['id']}: {e}")
        th = render_point_views(points, colors, 1, args.thumb, 2)[0]
        th.save(os.path.join(th_dir, f"{d['id']}.png"))
        thumbs.append(th)

    contact_sheet(thumbs).save(os.path.join(out, "contact_sheet.png"))
    print(f"[gen] done -> {out}  (contact_sheet.png, pointclouds/, thumbs/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
