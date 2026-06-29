from __future__ import annotations

import argparse
from pathlib import Path

from radjax_tome.backends import FakeTeacherBackend
from radjax_tome.corpora import load_jsonl_corpus
from radjax_tome.emit import emit_toy_teacher_tome


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a toy RADJAX teacher tome.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=8)
    args = parser.parse_args()

    records = load_jsonl_corpus(args.corpus)
    emit_toy_teacher_tome(
        output_dir=args.output_dir,
        backend=FakeTeacherBackend(vocab_size=args.vocab_size),
        records=records,
        sequence_length=args.sequence_length,
    )
    print(f"artifact_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
