#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from radjax_tome.builder import validate_teacher_textbook
from radjax_tome.targets import inspect_target_store


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Tome-side producer artifacts only."
    )
    parser.add_argument("--teacher-textbook", type=Path)
    parser.add_argument("--target-store", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    if args.teacher_textbook is None and args.target_store is None:
        parser.error("pass --teacher-textbook, --target-store, or both")
    report: dict[str, object] = {"status": "pass", "checks": {}, "blockers": []}
    blockers: list[str] = []
    if args.teacher_textbook is not None:
        textbook_report = validate_teacher_textbook(args.teacher_textbook)
        report["checks"]["teacher_textbook"] = textbook_report.to_dict()  # type: ignore[index]
        if textbook_report.status != "pass":
            blockers.extend(textbook_report.blockers)
    if args.target_store is not None:
        try:
            report["checks"]["target_store"] = inspect_target_store(args.target_store)  # type: ignore[index]
        except ValueError as exc:
            blockers.append(str(exc))
    if blockers:
        report["status"] = "fail"
        report["blockers"] = blockers
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"status={report['status']} blockers={len(blockers)}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Producer validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
