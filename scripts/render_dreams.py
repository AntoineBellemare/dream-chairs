"""Render saved dream point clouds as labeled multi-view montages, to eyeball forms.

  python scripts/render_dreams.py --tag pointe --exemplars      # cluster exemplars
  python scripts/render_dreams.py --tag pointe --ids D003,D041  # specific dreams
  python scripts/render_dreams.py --tag pointe --first 16       # first N
  python scripts/render_dreams.py --tag pointe --sample 12      # random N
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream_chairs.embed import render_point_views


def _font(size):
    for p in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap(draw, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pointe")
    ap.add_argument("--dreams", default=str(ROOT / "data" / "dreams" / "dream_dataset.json"))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    ap.add_argument("--ids", default=None, help="comma-separated dream ids")
    ap.add_argument("--exemplars", action="store_true", help="use cluster_report exemplars")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--size", type=int, default=170)
    ap.add_argument("--matplotlib", action="store_true",
                    help="render in the 3D matplotlib style (coolwarm cube) instead of the fast splat view")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    base = os.path.join(args.out_root, f"dataset_{args.tag}")
    pc_dir = os.path.join(base, "pointclouds")
    texts = {d["id"]: d["text"] for d in json.load(open(args.dreams, encoding="utf-8"))["dreams"]}
    all_ids = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(pc_dir, "*.npz")))]

    if args.ids:
        ids = [i.strip() for i in args.ids.split(",")]
    elif args.exemplars:
        rep = json.load(open(os.path.join(base, "cluster_report.json"), encoding="utf-8"))
        ids = [e["id"] for cl in rep["clusters"] for e in cl["exemplars"]]
    elif args.first:
        ids = all_ids[: args.first]
    elif args.sample:
        ids = random.Random(0).sample(all_ids, min(args.sample, len(all_ids)))
    else:
        ids = all_ids[:12]
    ids = [i for i in ids if os.path.exists(os.path.join(pc_dir, f"{i}.npz"))]

    cell, V, labw, pad = args.size, args.views, 250, 6
    font_id, font_tx = _font(15), _font(13)
    rowh = cell
    W = labw + V * cell + (V + 1) * pad
    H = len(ids) * (rowh + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for r, did in enumerate(ids):
        z = np.load(os.path.join(pc_dir, f"{did}.npz"))
        pts = z["points"]
        cols = z["colors"] if "colors" in z.files else None
        if args.matplotlib:
            from dream_chairs.render3d import render_points_3d
            azims = np.linspace(-62, -62 + 300, V)
            views = [render_points_3d(pts, px=cell, point_size=6, azim=a).resize((cell, cell))
                     for a in azims]
        else:
            views = render_point_views(pts, cols, V, cell, 2)
        y = pad + r * (rowh + pad)
        draw.text((pad, y + 4), did, fill=(20, 20, 20), font=font_id)
        for li, line in enumerate(_wrap(draw, texts.get(did, ""), font_tx, labw - 16)):
            draw.text((pad, y + 26 + li * 17), line, fill=(90, 90, 90), font=font_tx)
        for v, im in enumerate(views):
            canvas.paste(im, (labw + pad + v * (cell + pad), y))

    out = args.out or os.path.join(base, "render_montage.png")
    canvas.save(out)
    print(f"rendered {len(ids)} dreams x {V} views -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
