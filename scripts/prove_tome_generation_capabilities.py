#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from radjax_tome.capabilities import (
    prove_tome_generation_capabilities,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prove active RADJAX-Tome teacher-side generation capabilities."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-optional-hf-local", action="store_true")
    args = parser.parse_args(argv)

    result = prove_tome_generation_capabilities(
        work_dir=args.work_dir,
        matrix_json=args.matrix_json,
        report_md=args.report_md,
        overwrite=args.overwrite,
        run_optional_hf_local=args.run_optional_hf_local,
    )
    print(result.status_line())
    return result.exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Capability proof failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
