"""Dream report -> body-posture axis scores (the dream side of the bridge).

Two methods, same output schema (scores in [-1, 1] per axis, low pole -1 / high pole +1):
  clip : zero-shot — embed the dream text and project onto the axis poles in the shared
         OpenShape/CLIP space, z-scored across the corpus. No API key. Best for short,
         concrete reports; weaker on long/negated phenomenology.
  llm  : Claude reads the full report against the rubric and returns calibrated scores.
         Robust for real dream reports. Needs ANTHROPIC_API_KEY.

  python scripts/extract_posture.py --method clip
  python scripts/extract_posture.py --method llm --model claude-opus-4-8
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


def clip_scores(dreams, axes, cfg, device):
    cfg.embedding["backend"] = "openshape"
    embedder = build_embedder(cfg, device)
    axis_embs = AX.embed_axes(axes, embedder)
    texts = [d["text"] for d in dreams]
    raw = AX.project_raw(embedder.embed_texts(texts), axis_embs)
    z, _, _ = AX.standardize(raw)
    return np.tanh(z / 2.0)  # -> [-1, 1]


def llm_scores(dreams, axes, model):
    import anthropic
    client = anthropic.Anthropic()
    names = [a["name"] for a in axes]
    rubric = "\n".join(
        f"- {a['name']}: -1 = {a['low'][0]}; +1 = {a['high'][0]}.  ({a['phenomenology']})"
        for a in axes)
    out = np.zeros((len(dreams), len(axes)))
    for i, d in enumerate(dreams):
        prompt = (
            "You are scoring the BODILY PHENOMENOLOGY of a dream on bipolar axes, to "
            "inform the design of a chair that proposes the same posture.\n\n"
            f"Axes (score each from -1 to +1):\n{rubric}\n\n"
            f"Dream report:\n\"\"\"{d['text']}\"\"\"\n\n"
            "Return ONLY a JSON object mapping each axis name to a float in [-1, 1]. "
            f"Axis names: {names}.")
        msg = client.messages.create(model=model, max_tokens=400,
                                     messages=[{"role": "user", "content": prompt}])
        txt = msg.content[0].text
        s = txt[txt.find("{"): txt.rfind("}") + 1]
        obj = json.loads(s)
        out[i] = [float(np.clip(obj.get(n, 0.0), -1, 1)) for n in names]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--dreams", default=str(ROOT / "data" / "dreams" / "dream_dataset.json"))
    ap.add_argument("--axes", default=str(ROOT / "data" / "descriptors" / "body_posture_axes.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dream_posture_targets.json"))
    ap.add_argument("--method", default="clip", choices=["clip", "llm"])
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "cuda"))
    axes = AX.load_axes(args.axes)
    names = [a["name"] for a in axes]
    dreams = json.load(open(args.dreams, encoding="utf-8"))["dreams"]
    if args.limit:
        dreams = dreams[: args.limit]

    print(f"[posture-extract] method={args.method} dreams={len(dreams)} axes={len(axes)}")
    scores = (llm_scores(dreams, axes, args.model) if args.method == "llm"
              else clip_scores(dreams, axes, cfg, device))

    targets = {"method": args.method, "axes": names,
               "dreams": [{"id": d["id"], "text": d["text"],
                           "scores": {names[a]: round(float(scores[i, a]), 3) for a in range(len(names))}}
                          for i, d in enumerate(dreams)]}
    io_utils.save_json(args.out, targets)
    print(f"[posture-extract] wrote {args.out}")
    for t in targets["dreams"][:5]:
        top = sorted(t["scores"].items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  {t['id']}: " + ", ".join(f"{k}={v:+.2f}" for k, v in top) + f"  | {t['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
