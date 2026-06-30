#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from radjax_tome.targets.export import export_synthetic_teacher_targets
from radjax_tome.targets.inspection import inspect_target_store


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export producer-side teacher targets without live teacher deps."
    )
    parser.add_argument("--out", "--output", dest="output", type=Path, required=True)
    parser.add_argument("--backend", choices=("synthetic",), default="synthetic")
    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    export_synthetic_teacher_targets(
        args.output,
        num_examples=args.num_examples,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
        overwrite=args.overwrite,
    )
    report = inspect_target_store(args.output)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"status=pass output={args.output} target_type={report['target_type']} "
        f"examples={report['num_examples']} shards={report['shard_count']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Export teacher targets failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
