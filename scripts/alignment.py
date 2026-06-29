"""Measure how faithfully text -> 3D preserves a dream's structure.

Embeds each dream's TEXT and its generated FORM in the same OpenShape/CLIP space,
projects BOTH onto the same dimension sets (posture axes, design families, symbols),
and compares the two profiles. Reports:
  per-dimension : across the corpus, which axes track between text and form
                  (which properties survive translation)
  per-dream     : how well each form kept its own dream's signature
                  (use to pick the strongest dreams for the one-dream -> one-chair route)

Cosines are standardized within modality (text<->text and form<->text live on different
scales), and a shuffled baseline (dream paired with a random other form) shows whether
alignment is above chance.

  python scripts/alignment.py --tag pointe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dream_chairs import axes as AX  # noqa: E402
from dream_chairs import classify as C  # noqa: E402
from dream_chairs import families as FAM  # noqa: E402
from dream_chairs import io_utils  # noqa: E402
from dream_chairs.config import load_config  # noqa: E402
from dream_chairs.embed import build_embedder  # noqa: E402
from dream_chairs.utils import l2_normalize, resolve_device  # noqa: E402


def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def align_set(T, F, rng, n_perm=50):
    """T, F: (n, k) text/form profiles -> (per-dream r, mean true, mean shuffled, per-dim r)."""
    zT, zF = zscore(T), zscore(F)
    n = len(T)
    per_dream = np.array([corr(zT[d], zF[d]) for d in range(n)])
    mean_true = float(np.nanmean(per_dream))
    shuf = []
    for _ in range(n_perm):
        idx = rng.permutation(n)
        shuf.append(np.nanmean([corr(zT[d], zF[idx[d]]) for d in range(n)]))
    mean_shuf = float(np.nanmean(shuf))
    per_dim = [corr(zT[:, i], zF[:, i]) for i in range(T.shape[1])]
    return per_dream, mean_true, mean_shuf, per_dim


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--tag", default="pointe")
    ap.add_argument("--dreams", default=str(ROOT / "data" / "dreams" / "dream_dataset.json"))
    ap.add_argument("--families", default=str(ROOT / "data" / "descriptors" / "design_families.json"))
    ap.add_argument("--axes", default=str(ROOT / "data" / "descriptors" / "body_posture_axes.json"))
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = load_config(args.config)
    cfg.embedding["backend"] = "openshape"
    device = resolve_device(cfg.get("device", "cuda"))
    base = ROOT / "outputs" / f"dataset_{args.tag}"

    cache = base / "shape_embeddings_openshape.npz"
    if not cache.exists():
        print(f"need {cache}; run cluster_dataset.py --tag {args.tag} first")
        return 1
    c = np.load(cache, allow_pickle=True)
    ids = list(c["ids"]); shape_E = l2_normalize(c["embs"], axis=1)
    meta = {d["id"]: d for d in json.load(open(args.dreams, encoding="utf-8"))["dreams"]}
    texts = [meta[i]["text"] for i in ids]

    print(f"[align] {len(ids)} dreams — embedding text + dimension sets (OpenShape/CLIP)...")
    embedder = build_embedder(cfg, device)
    text_E = l2_normalize(embedder.embed_texts(texts), axis=1)

    # dimension sets, projected for BOTH text and form
    axis_embs = AX.embed_axes(AX.load_axes(args.axes), embedder)
    posture_dims = [a["name"] for a in AX.load_axes(args.axes)]
    sets = {
        "posture": (AX.project_raw(text_E, axis_embs), AX.project_raw(shape_E, axis_embs), posture_dims),
    }
    fams = FAM.load_families(args.families)
    members = list(dict.fromkeys([m for _, ms in fams for m in ms]))
    M = l2_normalize(embedder.embed_texts(members), axis=1)
    sets["design_families"] = (text_E @ M.T, shape_E @ M.T, members)

    symbols = io_utils.load_symbols(cfg.paths["symbols"])
    SY = C.build_symbol_embeddings(symbols, embedder)
    sym_names = [s.get("label", s["id"]) for s in symbols]
    sets["symbols"] = (text_E @ SY.T, shape_E @ SY.T, sym_names)

    rng = np.random.default_rng(0)
    report = {"tag": args.tag, "n": len(ids), "axis_sets": {}}
    per_dream_by_set = {}
    print("\n=== faithfulness: text → form (true vs shuffled) ===")
    for name, (T, F, dims) in sets.items():
        pd, mt, ms, pdim = align_set(T, F, rng)
        per_dream_by_set[name] = pd
        report["axis_sets"][name] = {
            "k": len(dims), "mean_alignment": round(mt, 3), "shuffled": round(ms, 3),
            "per_dimension": sorted(
                [{"dim": dims[i], "r": round(float(pdim[i]), 3)} for i in range(len(dims))],
                key=lambda d: -(d["r"] if d["r"] == d["r"] else -9)),
        }
        print(f"  {name:16s} true r={mt:+.3f}   shuffled r={ms:+.3f}   (k={len(dims)})")

    # per-dream + top-aligned dreams (mean across sets)
    stacked = np.vstack([per_dream_by_set[s] for s in sets]).T  # (n, n_sets)
    mean_align = np.nanmean(stacked, axis=1)
    per_dream = [{"id": ids[i], "text": meta[ids[i]]["text"],
                  **{s: round(float(per_dream_by_set[s][i]), 3) for s in sets},
                  "mean": round(float(mean_align[i]), 3)} for i in range(len(ids))]
    report["per_dream"] = per_dream
    order = np.argsort(-np.nan_to_num(mean_align, nan=-9))
    report["top_aligned"] = [ids[i] for i in order[:15]]

    print("\n  which properties survive (posture, per dimension):")
    for d in report["axis_sets"]["posture"]["per_dimension"]:
        print(f"     {d['dim']:20s} r={d['r']:+.2f}")
    print("\n  top dreams to keep (best-preserved signature):")
    for i in order[:8]:
        print(f"     {ids[i]}  align={mean_align[i]:+.2f}  {meta[ids[i]]['text']}")

    io_utils.save_json(base / "alignment_report.json", report)
    _figure(report, sets, per_dream_by_set, args.tag)
    print(f"\n[align] wrote alignment_report.json -> {base}  +  docs/figures/fig9_alignment.*")
    return 0


def _figure(report, sets, per_dream_by_set, tag):
    figdir = ROOT / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    names = list(sets.keys())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(names))
    true = [report["axis_sets"][n]["mean_alignment"] for n in names]
    shuf = [report["axis_sets"][n]["shuffled"] for n in names]
    a1.bar(x - 0.2, true, 0.4, label="true pairs", color="#1d9e75")
    a1.bar(x + 0.2, shuf, 0.4, label="shuffled", color="#bbbbbb")
    a1.set_xticks(x); a1.set_xticklabels(names, fontsize=9)
    a1.axhline(0, color="#999", lw=0.8); a1.set_ylabel("mean dream↔form alignment (r)")
    a1.set_title("Faithfulness of text → 3D, by axis set"); a1.legend(frameon=False, fontsize=8)
    # per-dimension posture preservation
    pd = report["axis_sets"]["posture"]["per_dimension"][::-1]
    a2.barh(range(len(pd)), [d["r"] for d in pd], color="#7f77dd")
    a2.set_yticks(range(len(pd))); a2.set_yticklabels([d["dim"] for d in pd], fontsize=8)
    a2.axvline(0, color="#999", lw=0.8); a2.set_xlabel("text↔form correlation (r)")
    a2.set_title("Which posture axes survive translation")
    fig.suptitle(f"Dream ↔ form alignment ({tag})", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "fig9_alignment.pdf"); fig.savefig(figdir / "fig9_alignment.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
