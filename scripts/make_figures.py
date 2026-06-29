"""Build paper-ready figures (vector PDF + 300 dpi PNG) from existing results.
Uses the matplotlib-3D renderer (render3d) for presentation-quality forms.

  python scripts/make_figures.py
Outputs to docs/figures/. Each figure is isolated; failures are skipped.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dream_chairs.render3d import render_mesh_3d, render_points_3d

plt.rcParams.update({
    "savefig.dpi": 300, "figure.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "font.family": "DejaVu Sans",
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

TAG = "pointe"
DS = ROOT / "outputs" / f"dataset_{TAG}"
FIGDIR = ROOT / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
CMAP = plt.get_cmap("tab10")
_CACHE = {}
CLEAN = "--clean" in sys.argv
SUF = "_clean" if CLEAN else ""


def loadj(p):
    return json.load(open(p, encoding="utf-8"))


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote docs/figures/{name}.pdf + .png")


def render_id(did, tag=TAG, mesh=False, px=300):
    key = (tag, did, mesh)
    if key in _CACHE:
        return _CACHE[key]
    base = ROOT / "outputs" / f"dataset_{tag}"
    if mesh:
        import trimesh
        m = trimesh.load(base / "meshes" / f"{did}.obj", process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        img = np.asarray(render_mesh_3d(m, px=px))
    else:
        z = np.load(base / "pointclouds" / f"{did}.npz")
        img = np.asarray(render_points_3d(z["points"], px=px, point_size=6))
    _CACHE[key] = img
    return img


def blob_flags(tag=TAG):
    p = ROOT / "outputs" / f"dataset_{tag}" / "blob_report.json"
    return {f["id"] for f in loadj(p)["forms"] if f["flag"]} if p.exists() else set()


def cluster_labels(report):
    labels, used = {}, set()
    for c in report["clusters"]:
        terms = [t["term"] for t in c["distinctive_descriptors"]]
        lab = next((t for t in terms if t not in used), terms[0])
        used.add(lab)
        labels[c["cluster"]] = lab
    return labels


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.5",
                                    linewidth=1.1, edgecolor="#3a3a3a", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111")

    def arrow(p1, p2, ls="-", color="#555", lw=1.3):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                     lw=lw, color=color, linestyle=ls, shrinkA=1, shrinkB=1))

    blue, teal, sand, rose = "#cfe0f3", "#cdeae0", "#f3e6cf", "#f4d9e2"
    box(2, 76, 19, 14, "Dream\nreport", blue)
    box(27, 76, 19, 14, "Text → 3D\nPoint-E / Shap-E", blue)
    box(52, 76, 19, 14, "3D form\npoint cloud / mesh", blue)
    box(77, 76, 21, 14, "OpenShape\nshared 3D ↔ text", teal)
    for x1, x2 in [(21, 27), (46, 52), (71, 77)]:
        arrow((x1, 83), (x2, 83))
    box(3, 40, 27, 15, "Cluster forms →\nfamilies of shapes", sand)
    box(36.5, 40, 27, 15, "Design-family\nsignatures", sand)
    box(70, 40, 27, 15, "Posture axes →\nchair parameters", rose)
    ax.plot([87.5, 87.5], [76, 68], color="#555", lw=1.3)
    ax.plot([16.5, 87.5], [68, 68], color="#555", lw=1.3)
    for x in (16.5, 50, 83.5):
        arrow((x, 68), (x, 55))
    box(34, 6, 32, 14, "Chair\ngenerate & select", rose)
    for x in (16.5, 50, 83.5):
        arrow((x, 40), (50, 20))
    arrow((11.5, 76), (75, 55), ls=(0, (4, 3)), color="#1d9e75", lw=1.2)
    ax.text(34, 67, "text → posture", color="#0F6E56", fontsize=8, rotation=-19, style="italic")
    ax.text(50, 97, "Dream → Chairs pipeline", ha="center", fontsize=12, color="#111")
    save(fig, "fig1_pipeline")


def fig_gallery(rows=5, cols=8):
    ids = [f"D{i:03d}" for i in range(rows * cols)]
    flags = blob_flags()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.35))
    for ax, did in zip(axes.ravel(), ids):
        ax.imshow(render_id(did)); ax.set_xticks([]); ax.set_yticks([])
        fl = did in flags
        for s in ax.spines.values():
            s.set_visible(fl); s.set_color("#d62728"); s.set_linewidth(2)
    fig.suptitle("Generated dream-form corpus (Point-E) — red outline = flagged degenerate blob",
                 y=1.004, fontsize=11)
    fig.subplots_adjust(wspace=0.04, hspace=0.1)
    save(fig, "fig2_corpus_gallery")


def fig_tsne():
    sc = loadj(DS / f"cluster_scatter{SUF}.json")["points"]
    rep = loadj(DS / f"cluster_report{SUF}.json")
    labels = cluster_labels(rep)
    sizes = {c["cluster"]: c["size"] for c in rep["clusters"]}
    exemplar = {c["cluster"]: c["exemplars"][0]["id"] for c in rep["clusters"]}
    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    for k in sorted(labels):
        pts = [(p["x"], p["y"]) for p in sc if p["cluster"] == k]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=30, color=CMAP(k % 10), alpha=0.85, linewidths=0,
                   label=f"C{k} · {labels[k]} (n={sizes[k]})")
        im = OffsetImage(render_id(exemplar[k], px=220), zoom=0.13)
        ab = AnnotationBbox(im, (np.mean(xs), np.mean(ys)), frameon=True, pad=0.05,
                            bboxprops=dict(edgecolor=CMAP(k % 10), lw=1.6))
        ax.add_artist(ab)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("Families of dream-forms — OpenShape embeddings (t-SNE), %d clusters" % len(labels))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    save(fig, f"fig3_cluster_tsne{SUF}")


def fig_signatures():
    cls = loadj(DS / f"cluster_report{SUF}.json")["clusters"]
    n = len(cls); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 2.05))
    for i, c in enumerate(cls):
        ax = axes.ravel()[i]
        sig = sorted(c["family_signature"], key=lambda s: -s["lift"])[:5][::-1]
        names = [s["distinctive"] for s in sig]
        lifts = [s["lift"] for s in sig]
        ax.barh(range(len(names)), lifts, color=CMAP(c["cluster"] % 10), alpha=0.9)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"C{c['cluster']}  (n={c['size']})", fontsize=9.5)
        ax.tick_params(axis="x", labelsize=7); ax.set_xlabel("design-family lift", fontsize=7.5)
        ax.set_xlim(0, max(lifts) * 1.15 if max(lifts) > 0 else 1)
    for j in range(n, rows * cols):
        axes.ravel()[j].axis("off")
    fig.suptitle("Cluster characterization — distinctive design-family qualities (contrastive)",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    save(fig, f"fig4_cluster_signatures{SUF}")


def fig_posture():
    m = loadj(DS / f"posture_matrix{SUF}.json")
    an = [a["name"] for a in m["axes"]]
    xi, yi = an.index("recumbent_upright"), an.index("exposed_cocooned")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    xs = [d["z"][xi] for d in m["dreams"]]; ys = [d["z"][yi] for d in m["dreams"]]
    cs = [CMAP(d["cluster"] % 10) for d in m["dreams"]]
    ax.axhline(0, color="#ccc", lw=0.8, ls="--"); ax.axvline(0, color="#ccc", lw=0.8, ls="--")
    ax.scatter(xs, ys, s=34, color=cs, alpha=0.85, linewidths=0)
    arr = np.array([xs, ys]).T
    for idx in {int(np.argmax(arr[:, 0])), int(np.argmin(arr[:, 0])), int(np.argmax(arr[:, 1])),
                int(np.argmin(arr[:, 1])), int(np.argmax(arr.sum(1))), int(np.argmin(arr.sum(1)))}:
        ax.annotate(m["dreams"][idx]["tag"], (xs[idx], ys[idx]), fontsize=7.5, color="#333",
                    xytext=(4, 4), textcoords="offset points")
    lim = max(3, max(abs(v) for v in xs + ys) * 1.05)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("← horizontal / reclined          upright / vertical →")
    ax.set_ylabel("← exposed / open          cocooned / enclosed →")
    ax.set_title("Posture map — dream-forms on two body axes (color = shape cluster)")
    save(fig, f"fig5_posture_map{SUF}")


def fig_chair():
    res = loadj(ROOT / "outputs" / "chairs" / "chairs_report.json")["results"]
    axn = list(res[0]["target"].keys())
    fig, axes = plt.subplots(len(res), 2, figsize=(8.4, 3.1 * len(res)),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    if len(res) == 1:
        axes = axes[None, :]
    for r, x in enumerate(res):
        axb = axes[r, 0]
        vals = [x["target"][a] for a in axn]
        axb.barh(range(len(axn)), vals, alpha=0.9,
                 color=["#d85a30" if v < 0 else "#1d9e75" for v in vals])
        axb.set_yticks(range(len(axn))); axb.set_yticklabels([a.replace("_", " ↔ ") for a in axn], fontsize=8)
        axb.axvline(0, color="#999", lw=0.8); axb.set_xlim(-1, 1)
        axb.set_xlabel("target posture (−1 … +1)", fontsize=8)
        axb.set_title(f"{x['id']}  —  {x['text']}", fontsize=9.5, loc="left"); axb.invert_yaxis()
        axi = axes[r, 1]
        z = np.load(ROOT / "outputs" / "chairs" / f"{x['id']}_best.npz")
        axi.imshow(np.asarray(render_points_3d(z["points"], px=320, point_size=6)))
        axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_visible(False)
        axi.set_xlabel("selected chair\n" + "\n".join(textwrap.wrap(x["prompt"].replace("a chair, ", ""), 38)),
                       fontsize=7.5)
    fig.suptitle("Dream → chair: posture-matched generate-and-select", y=1.005, fontsize=11)
    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    save(fig, "fig6_dream_to_chair")


def fig_generators():
    ids = ["D003", "D010", "D025", "D050", "D081"]
    rows = [("Point-E (points)", False, "pointe"), ("Shap-E 64 (mesh)", True, "shape64")]
    fig, axes = plt.subplots(len(rows), len(ids), figsize=(len(ids) * 1.7, len(rows) * 1.85))
    for ri, (name, mesh, tag) in enumerate(rows):
        for ci, did in enumerate(ids):
            ax = axes[ri, ci]
            try:
                ax.imshow(render_id(did, tag=tag, mesh=mesh))
            except Exception:
                pass
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if ci == 0:
                ax.set_ylabel(name, fontsize=10)
            if ri == 0:
                ax.set_title(did, fontsize=8)
    fig.suptitle("Same dreams, two generators", y=1.01, fontsize=11)
    fig.tight_layout()
    save(fig, "fig7_pointe_vs_shape")


def fig_mesh_gallery(rows=3, cols=8):
    ids = [f"D{i:03d}" for i in range(rows * cols)]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.35))
    for ax, did in zip(axes.ravel(), ids):
        try:
            ax.imshow(render_id(did, tag="shape64", mesh=True))
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Shap-E mesh corpus (surface-sampled render)", y=1.004, fontsize=11)
    fig.subplots_adjust(wspace=0.04, hspace=0.1)
    save(fig, "fig8_mesh_gallery")


def main():
    funcs = ((fig_tsne, fig_signatures, fig_posture) if CLEAN else
             (fig_pipeline, fig_gallery, fig_tsne, fig_signatures, fig_posture,
              fig_chair, fig_generators, fig_mesh_gallery))
    for fn in funcs:
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {e}")
    print(f"figures in {FIGDIR}")


if __name__ == "__main__":
    main()
