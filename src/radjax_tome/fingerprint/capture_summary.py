from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json


@dataclass(frozen=True)
class TeacherIdentity:
    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    backend: str = "hf_causal_lm"
    revision: str | None = None
    dtype: str | None = None
    vocab_size: int | None = None
    local_files_only: bool = True
    allow_downloads: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TeacherIdentity:
        return cls(
            model_name_or_path=str(payload.get("model_name_or_path", "")),
            tokenizer_name_or_path=(
                str(payload["tokenizer_name_or_path"])
                if payload.get("tokenizer_name_or_path") is not None
                else None
            ),
            backend=str(payload.get("backend", "hf_causal_lm")),
            revision=str(payload["revision"]) if payload.get("revision") else None,
            dtype=str(payload["dtype"]) if payload.get("dtype") else None,
            vocab_size=(
                int(payload["vocab_size"])
                if payload.get("vocab_size") is not None
                else None
            ),
            local_files_only=bool(payload.get("local_files_only", True)),
            allow_downloads=bool(payload.get("allow_downloads", False)),
            metadata=_metadata_excluding(
                payload,
                {
                    "model_name_or_path",
                    "tokenizer_name_or_path",
                    "backend",
                    "revision",
                    "dtype",
                    "vocab_size",
                    "local_files_only",
                    "allow_downloads",
                },
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: value for key, value in asdict(self).items() if value is not None
        }
        metadata = payload.pop("metadata", {})
        return {**payload, **metadata}


@dataclass(frozen=True)
class SourceCorpusReference:
    path: str
    corpus_id: str | None = None
    prompt_count: int | None = None
    sha256: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SourceCorpusReference:
        return cls(
            path=str(payload.get("path", "")),
            corpus_id=str(payload["corpus_id"]) if payload.get("corpus_id") else None,
            prompt_count=(
                int(payload["prompt_count"])
                if payload.get("prompt_count") is not None
                else None
            ),
            sha256=str(payload["sha256"]) if payload.get("sha256") else None,
            split=str(payload["split"]) if payload.get("split") else None,
            metadata=_metadata_excluding(
                payload,
                {"path", "corpus_id", "prompt_count", "sha256", "split"},
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: value for key, value in asdict(self).items() if value is not None
        }
        metadata = payload.pop("metadata", {})
        return {**payload, **metadata}


@dataclass(frozen=True)
class RealTeacherCaptureSummary:
    status: str
    artifact_dir: str
    teacher: TeacherIdentity
    source_corpus: SourceCorpusReference
    manifest_path: str
    modes_path: str | None
    target_shards: tuple[str, ...]
    exemplar_shards: tuple[str, ...] = ()
    local_files_only: bool = True
    allow_downloads: bool = False
    examples_processed: int = 0
    tokens_processed: int = 0
    target_positions_processed: int = 0
    modes_discovered: int = 0
    exemplars_retained: int = 0
    corridor_bounds_method: str | None = None
    exemplar_selection_policy: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RealTeacherCaptureSummary:
        return cls(
            status=str(payload.get("status", "")),
            artifact_dir=str(payload.get("artifact_dir", "")),
            teacher=TeacherIdentity.from_payload(_mapping(payload.get("teacher"))),
            source_corpus=SourceCorpusReference.from_payload(
                _mapping(payload.get("source_corpus"))
            ),
            manifest_path=str(payload.get("manifest_path", "")),
            modes_path=(
                str(payload["modes_path"]) if payload.get("modes_path") else None
            ),
            target_shards=tuple(str(item) for item in payload.get("target_shards", ())),
            exemplar_shards=tuple(
                str(item) for item in payload.get("exemplar_shards", ())
            ),
            local_files_only=bool(payload.get("local_files_only", True)),
            allow_downloads=bool(payload.get("allow_downloads", False)),
            examples_processed=_non_negative_int(
                payload.get("examples_processed", 0),
                "examples_processed",
            ),
            tokens_processed=_non_negative_int(
                payload.get("tokens_processed", 0),
                "tokens_processed",
            ),
            target_positions_processed=_non_negative_int(
                payload.get("target_positions_processed", 0),
                "target_positions_processed",
            ),
            modes_discovered=_non_negative_int(
                payload.get("modes_discovered", 0),
                "modes_discovered",
            ),
            exemplars_retained=_non_negative_int(
                payload.get("exemplars_retained", 0),
                "exemplars_retained",
            ),
            corridor_bounds_method=(
                str(payload["corridor_bounds_method"])
                if payload.get("corridor_bounds_method")
                else None
            ),
            exemplar_selection_policy=(
                str(payload["exemplar_selection_policy"])
                if payload.get("exemplar_selection_policy")
                else None
            ),
            warnings=tuple(str(item) for item in payload.get("warnings", ())),
            metadata=_mapping(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifact_dir": self.artifact_dir,
            "teacher": self.teacher.to_dict(),
            "source_corpus": self.source_corpus.to_dict(),
            "manifest_path": self.manifest_path,
            "modes_path": self.modes_path,
            "target_shards": list(self.target_shards),
            "exemplar_shards": list(self.exemplar_shards),
            "local_files_only": self.local_files_only,
            "allow_downloads": self.allow_downloads,
            "examples_processed": self.examples_processed,
            "tokens_processed": self.tokens_processed,
            "target_positions_processed": self.target_positions_processed,
            "modes_discovered": self.modes_discovered,
            "exemplars_retained": self.exemplars_retained,
            "corridor_bounds_method": self.corridor_bounds_method,
            "exemplar_selection_policy": self.exemplar_selection_policy,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def read_real_teacher_capture_summary(path: str | Path) -> RealTeacherCaptureSummary:
    return RealTeacherCaptureSummary.from_payload(read_json_object(Path(path)))


def write_real_teacher_capture_summary(
    path: str | Path,
    summary: RealTeacherCaptureSummary | dict[str, Any],
) -> Path:
    output = Path(path)
    payload = (
        summary.to_dict()
        if isinstance(summary, RealTeacherCaptureSummary)
        else dict(summary)
    )
    write_json(output, payload)
    return output


def validate_real_teacher_capture_summary(
    summary: RealTeacherCaptureSummary | dict[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    try:
        parsed = (
            summary
            if isinstance(summary, RealTeacherCaptureSummary)
            else RealTeacherCaptureSummary.from_payload(summary)
        )
    except ValueError as exc:
        return False, (str(exc),)
    blockers: list[str] = []
    if parsed.status not in {"pass", "fail", "dry_run", "unavailable"}:
        blockers.append("status must be pass, fail, dry_run, or unavailable")
    if not parsed.artifact_dir:
        blockers.append("artifact_dir is required")
    if not parsed.teacher.model_name_or_path:
        blockers.append("teacher.model_name_or_path is required")
    if not parsed.source_corpus.path:
        blockers.append("source_corpus.path is required")
    if parsed.allow_downloads and parsed.local_files_only:
        blockers.append("allow_downloads and local_files_only cannot both be true")
    if parsed.status == "pass" and not parsed.target_shards:
        blockers.append("pass summaries must list target_shards")
    return not blockers, tuple(blockers)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _metadata_excluding(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items() if key not in keys}


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    raise ValueError(f"{name} must be a non-negative integer")
