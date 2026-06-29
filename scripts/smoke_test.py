"""Quick wiring check: runs the embed + classify + cluster path on random point
clouds (no Point-E/Shap-E downloads). Exercises CLIP though, so the first run
downloads the OpenCLIP weights (~600 MB for ViT-B-32).

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dream_chairs.config import load_config
from dream_chairs.pipeline import run

if __name__ == "__main__":
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "config" / "config.yaml"))
    out = run(cfg, dry_run=True)
    print("\n=== smoke test results ===")
    for r in out["results"]:
        top = ", ".join(f"{m['label']}({m['score']:.2f})" for m in r["matches"][:3])
        print(f"  {r['id']}: {top}")
    if out["clusters"]:
        print("clusters:", out["clusters"]["assignments"])
