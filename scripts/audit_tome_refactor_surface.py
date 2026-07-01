#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from radjax_tome.audit.refactor_surface import (
    run_refactor_audit,
    write_refactor_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the RADJAX-Tome adversarial refactor surface audit."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    audit = run_refactor_audit(args.repo_root)
    write_refactor_audit(audit, json_out=args.json_out, md_out=args.md_out)
    print(
        "status=complete "
        f"files={len(audit.file_metrics)} "
        f"checklist={len(audit.checklist)} "
        f"spec3_blocked={audit.summary['spec3_blocked']} "
        f"json_out={args.json_out} md_out={args.md_out}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Refactor audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
