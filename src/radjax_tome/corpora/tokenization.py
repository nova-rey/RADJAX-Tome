from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from radjax_tome.corpora.loaders import load_jsonl_corpus
from radjax_tome.corpora.tokenizer import Tokenizer
from radjax_tome.provenance.hashes import stable_hash


@dataclass(frozen=True)
class TokenizedCorpusManifest:
    tokenizer: dict[str, object]
    sequence_length: int
    num_examples: int
    num_sequences: int
    num_tokens: int
    num_shards: int
    source_text_hash: str
    token_ids_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def tokenize_jsonl_corpus(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: Tokenizer,
    sequence_length: int,
    overwrite: bool = False,
) -> TokenizedCorpusManifest:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    root = Path(output_dir)
    if root.exists():
        if not overwrite:
            raise ValueError(f"tokenized corpus already exists: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rows = load_jsonl_corpus(input_path)
    token_rows: list[dict[str, object]] = []
    flat_tokens: list[int] = []
    for row in rows:
        token_ids = tokenizer.encode(row["text"], max_length=sequence_length)
        flat_tokens.extend(token_ids)
        token_rows.append(
            {
                "example_id": row["example_id"],
                "source_text_hash": stable_hash(row["text"]),
                "token_ids": token_ids,
                "token_count": len(token_ids),
            }
        )
    shard_path = root / "tokens-00000.jsonl"
    shard_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in token_rows),
        encoding="utf-8",
    )
    manifest = TokenizedCorpusManifest(
        tokenizer=asdict(tokenizer.metadata),
        sequence_length=sequence_length,
        num_examples=len(rows),
        num_sequences=len(token_rows),
        num_tokens=len(flat_tokens),
        num_shards=1,
        source_text_hash=stable_hash([row["text"] for row in rows]),
        token_ids_hash=stable_hash(np.asarray(flat_tokens, dtype=np.int64).tolist()),
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
