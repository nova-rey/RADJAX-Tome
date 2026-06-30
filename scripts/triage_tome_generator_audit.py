#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from radjax_tome.audit import (
    build_migration_map,
    emit_doc_summary,
    has_untriaged_high_risk,
    write_migration_map,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Triage Tome Generator extraction audit findings."
    )
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-untriaged-high-risk", action="store_true")
    parser.add_argument("--emit-doc-summary", type=Path)
    args = parser.parse_args()

    if not args.audit_json.is_file():
        parser.error(f"--audit-json does not exist: {args.audit_json}")

    migration_map = build_migration_map(args.audit_json)
    write_migration_map(migration_map, args.output_dir, overwrite=args.overwrite)
    if args.emit_doc_summary is not None:
        emit_doc_summary(migration_map, args.emit_doc_summary)

    print(
        "status=complete "
        f"items={len(migration_map.triage_items)} "
        f"high_risk={sum(migration_map.high_risk_by_bucket.values())} "
        f"spec3_gate={migration_map.spec3_gate.passed} "
        f"output={args.output_dir}"
    )
    if args.fail_on_untriaged_high_risk and has_untriaged_high_risk(migration_map):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
