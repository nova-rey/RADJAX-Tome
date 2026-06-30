#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from radjax_tome.audit import run_extraction_audit, write_audit_reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit qrwkv-xla Tome Generator extraction into RADJAX-Tome."
    )
    parser.add_argument("--old-repo", type=Path, required=True)
    parser.add_argument(
        "--new-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-blockers", action="store_true")
    args = parser.parse_args()

    if not args.old_repo.exists():
        parser.error(f"--old-repo does not exist: {args.old_repo}")
    if not args.new_repo.exists():
        parser.error(f"--new-repo does not exist: {args.new_repo}")

    report = run_extraction_audit(args.old_repo, args.new_repo)
    if not report.old_repo_ok:
        parser.error(f"--old-repo does not look like qrwkv-xla: {args.old_repo}")
    if not report.new_repo_ok:
        parser.error(f"--new-repo does not look like RADJAX-Tome: {args.new_repo}")

    write_audit_reports(report, args.output_dir, overwrite=args.overwrite)
    print(
        "status=complete "
        f"producer_relevant={report.summary['producer_relevant_old_files']} "
        f"missing={report.summary['missing']} "
        f"partial={report.summary['partial']} "
        f"blockers={len(report.blockers_before_spec3)} "
        f"output={args.output_dir}"
    )
    if args.fail_on_blockers and report.blockers_before_spec3:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
