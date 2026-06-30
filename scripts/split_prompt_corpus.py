#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from radjax_tome.corpora import (
    assign_prompt_splits,
    read_prompt_corpus,
    write_prompt_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic prompt splits")
    parser.add_argument("corpus", help="Input prompt corpus JSONL")
    parser.add_argument("--out", required=True, help="Output prompt corpus JSONL")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction assigned to validation",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Fraction assigned to test",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        corpus = read_prompt_corpus(args.corpus)
        split_corpus = assign_prompt_splits(
            corpus,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        output_path = write_prompt_corpus(
            split_corpus,
            Path(args.out),
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"Prompt corpus split failed: {exc}", file=sys.stderr)
        return 1

    print(f"corpus: {output_path}")
    print(f"prompt_count: {len(split_corpus.records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
