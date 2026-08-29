"""One production corpus reader for v1 compatibility and v2 artifacts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.corpora.config import CorpusArtifactReference
from radjax_tome.corpora.loaders import load_jsonl_corpus
from radjax_tome.corpora.validation import open_verified_corpus


@dataclass(frozen=True)
class ResolvedCorpusInput:
    kind: str
    artifact_path: Path
    semantic_identity: str | None
    manifest_path: Path | None = None


def resolve_corpus_input(
    value: CorpusArtifactReference | Mapping[str, Any] | Path | str,
    *,
    expected_semantic_identity: str | None = None,
    corpus_manifest_path: Path | None = None,
) -> ResolvedCorpusInput:
    """Resolve and validate a corpus before callers load a teacher model."""

    if isinstance(value, CorpusArtifactReference):
        artifact = value.artifact_path.resolve()
        expected = value.expected_semantic_identity
    elif isinstance(value, Mapping):
        artifact_value = value.get("artifact_path")
        expected = value.get("expected_semantic_identity", expected_semantic_identity)
        if not isinstance(artifact_value, (str, Path)):
            raise ValueError("corpus artifact_path is required")
        artifact = Path(artifact_value).resolve()
    else:
        artifact = Path(value).resolve()
        expected = expected_semantic_identity
    if (artifact / "corpus_cover.json").is_file():
        result = __import__(
            "radjax_tome.corpora.validation", fromlist=["validate_corpus_artifact_v2"]
        ).validate_corpus_artifact_v2(artifact)
        if not result.ok:
            raise ValueError(
                "invalid corpus v2 artifact: " + "; ".join(result.blockers)
            )
        if expected is not None and result.semantic_identity != expected:
            raise ValueError("corpus semantic identity mismatch")
        return ResolvedCorpusInput("v2", artifact, result.semantic_identity)
    manifest = corpus_manifest_path.resolve() if corpus_manifest_path else artifact
    from radjax_tome.corpora.builder import read_corpus_manifest

    parsed = read_corpus_manifest(manifest)
    return ResolvedCorpusInput("v1", artifact, str(parsed.get("corpus_hash")), manifest)


def open_corpus_input(resolved: ResolvedCorpusInput) -> Any:
    if resolved.kind == "v2":
        return open_verified_corpus(resolved.artifact_path)
    return load_jsonl_corpus(resolved.artifact_path)


def iter_corpus_examples(resolved: ResolvedCorpusInput) -> Iterator[dict[str, Any]]:
    source = open_corpus_input(resolved)
    if resolved.kind == "v2":
        yield from source
    else:
        yield from source


__all__ = [
    "CorpusArtifactReference",
    "ResolvedCorpusInput",
    "iter_corpus_examples",
    "open_corpus_input",
    "resolve_corpus_input",
]
