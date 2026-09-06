"""Read-only corpus feasibility checks used by CLI and TUI preflight."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from radjax_tome.corpora.config import CorpusBuildIntent
from radjax_tome.corpora.lifecycle import preflight_corpus_build
from radjax_tome.corpora.tokenizer import create_tokenizer


def assess_corpus_feasibility(intent: CorpusBuildIntent) -> dict[str, Any]:
    """Check declared inputs and tokenizer construction without transforming data."""
    preflight = preflight_corpus_build(intent)
    for source in intent.sources:
        path = source.path
        if path.is_symlink() or not path.exists():
            raise ValueError(f"source is not a readable owned path: {source.source_id}")
        if path.is_file():
            with path.open("rb") as handle:
                handle.read(1)
        elif path.is_dir():
            for directory, dirnames, filenames in os.walk(path):
                dirnames.sort()
                for filename in sorted(filenames):
                    candidate = Path(directory) / filename
                    if candidate.is_symlink():
                        raise ValueError(f"source contains a symlink: {candidate}")
                    if candidate.suffix.lower() in {
                        ".txt",
                        ".md",
                        ".markdown",
                        ".py",
                        ".jsonl",
                    }:
                        with candidate.open("rb") as handle:
                            handle.read(1)
        if source.adapter == "local_jsonl_text_v1":
            if not path.is_file():
                raise ValueError(
                    f"JSONL source must be a readable file: {source.source_id}"
                )
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise ValueError(
                            f"JSONL source contains a blank line at {line_number}"
                        )
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"JSONL source is invalid at line {line_number}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"JSONL source record {line_number} must be an object"
                        )
                    text = record.get(source.text_field)
                    if not isinstance(text, str):
                        raise ValueError(
                            f"JSONL source record {line_number} text field must be "
                            "a string"
                        )
    tokenizer = create_tokenizer(str(intent.policy["tokenizer"]))
    return {
        "status": "pass",
        "preflight": preflight,
        "tokenizer": type(tokenizer).__name__,
        "sources": len(intent.sources),
        "transformed": False,
    }


__all__ = ["assess_corpus_feasibility"]
