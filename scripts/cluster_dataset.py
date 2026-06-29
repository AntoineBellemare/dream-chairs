"""Cluster a corpus of 3D forms by their OpenShape embeddings, then characterize each
cluster from the design descriptor families.

Pipeline: load point clouds (tag) -> OpenShape embed (cached) -> pick k by silhouette
-> KMeans -> per-cluster design-family signature + emergent descriptors + exemplars,
plus a 2D map (t-SNE) of all forms colored by cluster.

  python scripts/cluster_dataset.py --tag pointe
  python scripts/cluster_dataset.py --tag shape64 --k 8
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
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
from dream_chairs.utils import l2_normalize, resolve_device


def load_cloud(path: str) -> Shape:
    z = np.load(path)
    cid = os.path.splitext(os.path.basename(path))[0]
    return Shape(id=cid, prompt="", points=z["points"],
                 colors=z["colors"] if "colors" in z.files else None)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stderr(io.StringIO()):
        yield


def pick_k(E, kmin, kmax):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best = (-2.0, None, None)
    scores = {}
    with quiet():
        for k in range(kmin, kmax + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(E)
            s = silhouette_score(E, labels, metric="cosine")
            scores[k] = float(s)
            if s > best[0]:
                best = (s, k, labels)
    return best[1], best[2], scores


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--tag", default="pointe")
    ap.add_argument("--dreams", default=str(ROOT / "data" / "dreams" / "dream_dataset.json"))
    ap.add_argument("--families", default=str(ROOT / "data" / "descriptors" / "design_families.json"))
    ap.add_argument("--descriptors", default=str(ROOT / "data" / "descriptors" / "visual_descriptors.json"))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    ap.add_argument("--k", type=int, default=None, help="fixed cluster count (else silhouette)")
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--sharpness", type=float, default=2.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="restrict to clean_ids.json (drop flagged blobs); writes *_clean outputs")
    args = ap.parse_args(argv)
    suffix = "_clean" if args.clean else ""

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
    cfg = load_config(args.config)
    cfg.embedding["backend"] = "openshape"
    device = resolve_device(cfg.get("device", "cuda"))

    out = os.path.join(args.out_root, f"dataset_{args.tag}")
    pc_dir = os.path.join(out, "pointclouds")
    paths = sorted(glob.glob(os.path.join(pc_dir, "*.npz")))
    if not paths:
        print(f"no point clouds in {pc_dir}; run generate_dataset.py --tag {args.tag} first")
        return 1
    shapes = [load_cloud(p) for p in paths]
    ids = [s.id for s in shapes]
    raw = json.load(open(args.dreams, encoding="utf-8"))
    meta = {d["id"]: d for d in (raw["dreams"] if isinstance(raw, dict) else raw)}

    families = FAM.load_families(args.families)
    with open(args.descriptors, encoding="utf-8") as f:
        dd = json.load(f)
    descriptors = list(dict.fromkeys(dd["descriptors"] if isinstance(dd, dict) else dd))

    # --- embed (cached) ---
    cache = os.path.join(out, "shape_embeddings_openshape.npz")
    E = None
    if os.path.exists(cache) and not args.no_cache:
        c = np.load(cache, allow_pickle=True)
        if list(c["ids"]) == ids:
            E = c["embs"]
            print(f"[cluster] loaded cached embeddings ({E.shape}) from {cache}")
    print(f"[cluster] tag={args.tag} shapes={len(shapes)} families={len(families)} "
          f"descriptors={len(descriptors)}")
    embedder = build_embedder(cfg, device)
    if E is None:
        print("[cluster] embedding shapes with OpenShape...")
        E = np.stack([embedder.embed_shape(s) for s in shapes], axis=0)
        np.savez_compressed(cache, ids=np.array(ids), embs=E)
        print(f"[cluster] cached embeddings -> {cache}")
    E = l2_normalize(E, axis=1)

    if args.clean:
        cids = set(json.load(open(os.path.join(out, "clean_ids.json"), encoding="utf-8"))["ids"])
        keep = [i for i, d in enumerate(ids) if d in cids]
        ids = [ids[i] for i in keep]; shapes = [shapes[i] for i in keep]; E = E[keep]
        print(f"[cluster] clean mode: {len(ids)} forms (flagged blobs removed)")

    fam_embs = FAM.embed_families(families, embedder)
    desc_embs = l2_normalize(embedder.embed_texts(descriptors), axis=1)

    # --- choose k + cluster ---
    from sklearn.cluster import KMeans
    if args.k:
        with quiet():
            labels = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(E)
        k, scores = args.k, {}
    else:
        k, labels, scores = pick_k(E, args.kmin, min(args.kmax, len(shapes) - 1))
        print(f"[cluster] silhouette by k: " +
              ", ".join(f"{kk}:{vv:.3f}" for kk, vv in scores.items()))
    print(f"[cluster] k={k}")

    # --- per-shape family profiles (for aggregation) ---
    profiles = [FAM.shape_profile(E[i], fam_embs, args.sharpness) for i in range(len(shapes))]

    # --- 2D map ---
    from sklearn.manifold import TSNE
    with quiet():
        perp = max(5, min(30, len(shapes) // 3))
        xy = TSNE(n_components=2, init="pca", perplexity=perp, random_state=0,
                  metric="cosine").fit_transform(E)
    xy = (xy - xy.min(0)) / (xy.ptp(0) + 1e-9)  # normalize to [0,1]

    # --- global baselines for contrastive characterization ---
    # (when an attractor dominates every cluster, what separates clusters is how much
    #  each one deviates from the corpus average, not its raw dominant member)
    global_centroid = l2_normalize(E.mean(0))
    global_term = desc_embs @ global_centroid
    global_fdist = []
    for fi, (fname, members) in enumerate(families):
        arr = np.zeros(len(members))
        for i in range(len(shapes)):
            d = profiles[i][fi]["distribution"]
            arr += np.array([d[m] for m in members])
        global_fdist.append(arr / len(shapes))

    # --- characterize each cluster ---
    fam_names = [n for n, _ in families]
    clusters = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        centroid = l2_normalize(E[idx].mean(0))
        signature = []
        for fi, (fname, members) in enumerate(families):
            dist = np.zeros(len(members))
            for i in idx:
                d = profiles[i][fi]["distribution"]
                dist += np.array([d[m] for m in members])
            dist /= len(idx)
            sal = float(np.mean([profiles[i][fi]["salience"] for i in idx]))
            j = int(np.argmax(dist))
            dev = dist - global_fdist[fi]      # lift vs corpus average
            jd = int(np.argmax(dev))
            signature.append({"family": fname,
                              "dominant": members[j], "mean_p": float(dist[j]),
                              "distinctive": members[jd], "lift": float(dev[jd]),
                              "mean_salience": sal,
                              "distribution": {members[m]: float(dist[m]) for m in range(len(members))}})
        terms_sims = desc_embs @ centroid
        top_terms = [{"term": descriptors[t], "score": float(terms_sims[t])}
                     for t in np.argsort(-terms_sims)[:10]]
        term_dev = terms_sims - global_term
        distinctive_terms = [{"term": descriptors[t], "lift": float(term_dev[t])}
                             for t in np.argsort(-term_dev)[:8]]
        sims = E[idx] @ centroid
        reps = [ids[idx[t]] for t in np.argsort(-sims)[:3]]
        clusters.append({
            "cluster": c, "size": int(len(idx)),
            "members": [ids[i] for i in idx],
            "family_signature": signature,
            "top_descriptors": top_terms,
            "distinctive_descriptors": distinctive_terms,
            "exemplars": [{"id": r, "text": meta.get(r, {}).get("text", "")} for r in reps],
        })

    report = {"tag": args.tag, "k": k, "n": len(shapes), "silhouette": scores,
              "clusters": clusters}
    io_utils.save_json(os.path.join(out, f"cluster_report{suffix}.json"), report)
    scatter = {"points": [{"id": ids[i], "x": float(xy[i, 0]), "y": float(xy[i, 1]),
                           "cluster": int(labels[i]),
                           "tag": meta.get(ids[i], {}).get("tag", ""),
                           "text": meta.get(ids[i], {}).get("text", "")} for i in range(len(shapes))]}
    io_utils.save_json(os.path.join(out, f"cluster_scatter{suffix}.json"), scatter)
    _write_md(os.path.join(out, f"cluster_report{suffix}.md"), report, fam_names)

    print(f"\n=== {k} clusters ({args.tag}) ===")
    for cl in clusters:
        dist = " | ".join(f"{s['family']}:{s['distinctive']}(+{s['lift']:.2f})"
                          for s in sorted(cl["family_signature"], key=lambda s: -s["lift"])[:4])
        print(f"\ncluster {cl['cluster']} (n={cl['size']})")
        print(f"  distinctive: {dist}")
        print(f"  vs-corpus terms: {', '.join(t['term'] for t in cl['distinctive_descriptors'][:6])}")
        print(f"  exemplars: {'; '.join(e['text'] for e in cl['exemplars'])}")
    print(f"\n[cluster] wrote cluster_report.(json|md) + cluster_scatter.json -> {out}")
    return 0


def _write_md(path, report, fam_names):
    L = [f"# Volume clusters & design-family signatures ({report['tag']})\n",
         f"{report['n']} forms -> {report['k']} clusters (OpenShape embeddings, KMeans).\n"]
    for cl in report["clusters"]:
        L.append(f"## Cluster {cl['cluster']} — {cl['size']} forms")
        L.append("**Design-family signature** (dominant member; distinctive = lift vs corpus avg):")
        for s in cl["family_signature"]:
            L.append(f"- {s['family']}: _{s['dominant']}_ (p={s['mean_p']:.2f}) · "
                     f"distinctive _{s['distinctive']}_ (+{s['lift']:.2f}, salience={s['mean_salience']:.3f})")
        L.append(f"\n**Emergent qualities:** {', '.join(t['term'] for t in cl['top_descriptors'])}")
        L.append(f"\n**Most distinctive vs corpus:** {', '.join(t['term'] for t in cl['distinctive_descriptors'])}")
        L.append("\n**Exemplar dreams:**")
        for e in cl["exemplars"]:
            L.append(f"- {e['id']}: {e['text']}")
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
