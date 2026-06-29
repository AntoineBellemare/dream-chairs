"""Detect degenerate 'grey blob' forms — generations that collapsed into a generic,
redundant, featureless mass instead of a distinctive shape.

Signals (per form, from the OpenShape embeddings + point geometry):
  centrality  — cosine to the corpus mean embedding   (high = generic)
  redundancy  — mean cosine to its 10 nearest neighbors (high = many near-duplicates)
  anisotropy  — 1 - λ3/λ1 from a PCA of the points     (low  = round / featureless)
blobbiness = z(centrality) + z(redundancy) - z(anisotropy); the top fraction is flagged.

Distinguishing a *degenerate* blob from a *legitimately round* dream (a boulder, a moon)
is inherently fuzzy — this targets generic + redundant + round forms; review the output.

  python scripts/flag_blobs.py --tag pointe --frac 0.22
Writes blob_report.json, clean_ids.json, blob_compare.png to outputs/dataset_<tag>/.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dream_chairs import io_utils  # noqa: E402
from dream_chairs.render3d import render_points_3d  # noqa: E402
from dream_chairs.utils import l2_normalize  # noqa: E402


def z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-9)


def anisotropy(points):
    p = points - points.mean(0)
    w = np.sort(np.linalg.eigvalsh(np.cov(p.T)))[::-1]  # l1>=l2>=l3
    return float(1 - w[2] / (w[0] + 1e-9))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pointe")
    ap.add_argument("--frac", type=float, default=0.22, help="fraction flagged as blobs")
    ap.add_argument("--show", type=int, default=12, help="worst/best to render")
    args = ap.parse_args(argv)

    base = ROOT / "outputs" / f"dataset_{args.tag}"
    cache = base / "shape_embeddings_openshape.npz"
    if not cache.exists():
        print(f"need {cache}; run cluster_dataset.py --tag {args.tag} first")
        return 1
    c = np.load(cache, allow_pickle=True)
    ids = list(c["ids"]); E = l2_normalize(c["embs"], axis=1)
    pc = {os.path.splitext(os.path.basename(p))[0]: p
          for p in glob.glob(str(base / "pointclouds" / "*.npz"))}

    gc = l2_normalize(E.mean(0))
    centrality = E @ gc
    S = E @ E.T
    np.fill_diagonal(S, -1.0)
    redundancy = np.sort(S, axis=1)[:, -10:].mean(axis=1)
    aniso = np.array([anisotropy(np.load(pc[i])["points"]) for i in ids])

    blobbiness = z(centrality) + z(redundancy) - z(aniso)
    order = np.argsort(-blobbiness)
    n_flag = int(round(args.frac * len(ids)))
    flagged = set(order[:n_flag].tolist())

    report = [{"id": ids[i], "blobbiness": round(float(blobbiness[i]), 3),
               "centrality": round(float(centrality[i]), 3),
               "redundancy": round(float(redundancy[i]), 3),
               "anisotropy": round(float(aniso[i]), 3),
               "flag": i in flagged}
              for i in range(len(ids))]
    report.sort(key=lambda r: -r["blobbiness"])
    io_utils.save_json(base / "blob_report.json",
                       {"tag": args.tag, "frac": args.frac, "n_flagged": n_flag, "forms": report})
    io_utils.save_json(base / "clean_ids.json",
                       {"ids": [ids[i] for i in range(len(ids)) if i not in flagged]})

    print(f"[blobs] flagged {n_flag}/{len(ids)} at frac={args.frac}")
    print("  most blobby:", ", ".join(ids[i] for i in order[:8]))
    print("  most distinctive:", ", ".join(ids[i] for i in order[-8:]))

    # worst-vs-best comparison rendered in the new style
    k = args.show
    cols = 6
    rows = 2 * ((k + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 1.9))
    sel = list(order[:k]) + list(order[::-1][:k])
    border = ["#d62728"] * k + ["#2ca02c"] * k
    for ax, idx, col in zip(axes.ravel(), sel, border):
        img = render_points_3d(np.load(pc[ids[idx]])["points"], px=240, point_size=6)
        ax.imshow(np.asarray(img))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(col); s.set_linewidth(2.2)
        ax.set_title(f"{ids[idx]}  b={blobbiness[idx]:+.1f}", fontsize=7.5, color=col)
    for ax in axes.ravel()[len(sel):]:
        ax.axis("off")
    fig.suptitle(f"Degenerate blobs (red) vs most distinctive forms (green) — {args.tag}",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(base / "blob_compare.png", dpi=200, bbox_inches="tight")
    fig.savefig(base / "blob_compare.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[blobs] wrote blob_report.json, clean_ids.json, blob_compare.png -> {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
