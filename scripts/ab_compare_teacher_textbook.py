#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from radjax_tome.parity.runner import run_ab_parity


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy TeacherTextbook A/B parity harness against the "
            "archived qrwkv-xla builder."
        )
    )
    parser.add_argument(
        "--old-repo",
        type=Path,
        default=os.environ.get("QRWKV_XLA_OLD_REPO"),
        help="Path to a local read-only qrwkv-xla clone.",
    )
    parser.add_argument(
        "--new-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the RADJAX-Tome checkout. Defaults to this repository.",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case-set", choices=("fake-default",), default="fake-default")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.old_repo is None:
        parser.error("--old-repo or QRWKV_XLA_OLD_REPO is required")
    summary = run_ab_parity(
        old_repo=args.old_repo,
        new_repo=args.new_repo,
        work_dir=args.work_dir,
        case_set=args.case_set,
        overwrite=args.overwrite,
    )
    print(
        f"status={summary.status} cases={len(summary.cases)} "
        f"work_dir={summary.work_dir}"
    )
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
