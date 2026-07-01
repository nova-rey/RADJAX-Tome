from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from radjax_tome.builder import (
    TeacherTextbookBuildConfig,
    build_fake_teacher_textbook,
)
from radjax_tome.corpora import (
    PromptCorpus,
    PromptRecord,
    build_prompt_corpus_manifest,
    write_prompt_corpus,
    write_prompt_corpus_manifest,
)
from radjax_tome.fingerprint import build_minimal_fingerprint_artifact_from_target_store
from radjax_tome.targets.export import export_synthetic_teacher_targets
from radjax_tome.targets.store import TeacherTargetStore


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(record), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_prompt_corpus_fixture(tmp_path: Path) -> Path:
    prompt_corpus = PromptCorpus(
        corpus_id="test-prompt-corpus",
        records=(
            PromptRecord(id="p0", text="alpha", split="train", tags=("fixture",)),
            PromptRecord(id="p1", text="beta", split="validation", tags=("fixture",)),
        ),
    )
    path = write_prompt_corpus(
        prompt_corpus,
        tmp_path / "prompts.jsonl",
        overwrite=True,
    )
    manifest = build_prompt_corpus_manifest(
        prompt_corpus,
        description="Shared test prompt corpus.",
    )
    write_prompt_corpus_manifest(
        manifest,
        tmp_path / "prompt_manifest.json",
        overwrite=True,
    )
    return path


def build_fake_teacher_textbook_artifact(
    tmp_path: Path,
    *,
    max_examples: int = 2,
    sequence_length: int = 8,
) -> Path:
    output = tmp_path / "fake_tome"
    build_fake_teacher_textbook(
        TeacherTextbookBuildConfig(
            output_dir=output,
            teacher_mode="fake",
            max_examples=max_examples,
            sequence_length=sequence_length,
            overwrite=True,
        )
    )
    return output


def build_minimal_target_store(
    tmp_path: Path,
    *,
    num_examples: int = 2,
    sequence_length: int = 3,
    vocab_size: int = 8,
) -> TeacherTargetStore:
    return export_synthetic_teacher_targets(
        tmp_path / "dense",
        num_examples=num_examples,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        overwrite=True,
    )


def build_minimal_fingerprint_artifact(tmp_path: Path) -> Path:
    store = build_minimal_target_store(tmp_path)
    return build_minimal_fingerprint_artifact_from_target_store(
        store,
        tmp_path / "fingerprint",
        overwrite=True,
    )
