#!/usr/bin/env python3
"""Runner to generate all reports and export figures and LaTeX tables."""

from pathlib import Path
import sys
import argparse

# ensure repo root is on sys.path so imports like `src.reports` work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reports import (
    generate_performance_fob,
    generate_performance_weights,
    generate_gap_tables,
    generate_sensitivity_tables,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="out")
    p.add_argument(
        "--only", nargs="*", help="Which reports to run: fob, weights, gap, sensitivity"
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_run = set(args.only) if args.only else {"fob", "weights", "gap", "sensitivity"}

    if "fob" in to_run:
        print("Running performance fob...")
        if not args.dry_run:
            generate_performance_fob(out_dir)

    if "weights" in to_run:
        print("Running performance weights...")
        if not args.dry_run:
            generate_performance_weights(out_dir)

    if "gap" in to_run:
        print("Running gap analysis...")
        if not args.dry_run:
            generate_gap_tables(out_dir)

    if "sensitivity" in to_run:
        print("Running sensitivity analysis...")
        if not args.dry_run:
            generate_sensitivity_tables(out_dir)


if __name__ == "__main__":
    main()
