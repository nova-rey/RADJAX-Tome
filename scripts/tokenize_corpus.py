#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from radjax_tome.corpora import (
    TokenizerConfig,
    create_tokenizer,
    tokenize_jsonl_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tokenize a producer JSONL corpus with stable source identity."
    )
    parser.add_argument("corpus", nargs="?")
    parser.add_argument("--input", dest="corpus_input")
    parser.add_argument(
        "--out", "--output", dest="output_dir", type=Path, required=True
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--tokenizer-backend", "--tokenizer", default="smoke")
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--tokenizer-vocab-size", type=int)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--pad-token-id", type=int)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--slow-tokenizer", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    corpus = args.corpus_input or args.corpus
    if corpus is None:
        parser.error("corpus path is required")
    tokenizer = create_tokenizer(
        TokenizerConfig(
            backend=args.tokenizer_backend,
            tokenizer_id=args.tokenizer_id,
            vocab_size=args.tokenizer_vocab_size,
            eos_token_id=args.eos_token_id,
            pad_token_id=args.pad_token_id,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            use_fast=not args.slow_tokenizer,
        )
    )
    manifest = tokenize_jsonl_corpus(
        corpus,
        args.output_dir,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        overwrite=args.overwrite,
    )
    print("Wrote tokenized corpus:")
    print(f" output: {args.output_dir}")
    print(f" tokenizer: {manifest.tokenizer['backend']}")
    print(f" sequence_length: {manifest.sequence_length}")
    print(f" sequences: {manifest.num_sequences}")
    print(f" tokens: {manifest.num_tokens}")
    print(f" shards: {manifest.num_shards}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Tokenize corpus failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
