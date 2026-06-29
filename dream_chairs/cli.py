"""Command-line entry point.

  python -m dream_chairs.cli run                 # full pipeline
  python -m dream_chairs.cli run --dry-run       # skip generation (random clouds)
  python -m dream_chairs.cli run --limit 1       # first dream only
"""
from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run as run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dream-chairs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the dream -> shape -> symbol pipeline")
    p_run.add_argument("--config", default="config/config.yaml")
    p_run.add_argument("--limit", type=int, default=None, help="process only the first N dreams")
    p_run.add_argument("--dry-run", action="store_true",
                       help="use random point clouds instead of Point-E/Shap-E (no model downloads)")
    p_run.add_argument("--no-cluster", action="store_true", help="skip clustering stage")

    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = load_config(args.config)
        do_cluster = False if args.no_cluster else None
        run_pipeline(cfg, limit=args.limit, dry_run=args.dry_run, do_cluster=do_cluster)
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
