from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json


@dataclass(frozen=True)
class HFTeacherExportConfig:
    resolved_model_id: str
    tokenizer_id: str | None = None
    policy_label: str | None = None
    fallback_label: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = True
    allow_downloads: bool = False
    dtype: str = "fp32"
    device: str = "cpu"
    sequence_length: int = 8
    vocab_size: int | None = None
    include_logits: bool = True
    include_attention_targets: bool = False
    prompt_count: int = 0
    batch_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HFTeacherExportConfig:
        return cls(
            resolved_model_id=str(payload.get("resolved_model_id", "")),
            tokenizer_id=(
                str(payload["tokenizer_id"]) if payload.get("tokenizer_id") else None
            ),
            policy_label=(
                str(payload["policy_label"]) if payload.get("policy_label") else None
            ),
            fallback_label=(
                str(payload["fallback_label"])
                if payload.get("fallback_label")
                else None
            ),
            revision=str(payload["revision"]) if payload.get("revision") else None,
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
            local_files_only=bool(payload.get("local_files_only", True)),
            allow_downloads=bool(payload.get("allow_downloads", False)),
            dtype=str(payload.get("dtype", "fp32")),
            device=str(payload.get("device", "cpu")),
            sequence_length=int(payload.get("sequence_length", 8)),
            vocab_size=(
                int(payload["vocab_size"])
                if payload.get("vocab_size") is not None
                else None
            ),
            include_logits=bool(payload.get("include_logits", True)),
            include_attention_targets=bool(
                payload.get("include_attention_targets", False)
            ),
            prompt_count=int(payload.get("prompt_count", 0)),
            batch_size=int(payload.get("batch_size", 1)),
            metadata={
                str(key): value
                for key, value in payload.items()
                if key
                not in {
                    "resolved_model_id",
                    "tokenizer_id",
                    "policy_label",
                    "fallback_label",
                    "revision",
                    "trust_remote_code",
                    "local_files_only",
                    "allow_downloads",
                    "dtype",
                    "device",
                    "sequence_length",
                    "vocab_size",
                    "include_logits",
                    "include_attention_targets",
                    "prompt_count",
                    "batch_size",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: value for key, value in asdict(self).items() if value is not None
        }
        metadata = payload.pop("metadata", {})
        return {**payload, **metadata}


@dataclass(frozen=True)
class HFTeacherExportMetadata:
    schema_version: str
    teacher_family: str
    teacher_model_id: str
    tokenizer_id: str
    sequence_length: int
    dtype: str
    local_files_only: bool
    allow_downloads: bool
    targets: dict[str, bool]
    prompt_count: int
    created_by: str = "RADJAX-Tome HFTeacherExportMetadata"
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HFTeacherExportMetadata:
        return cls(
            schema_version=str(payload.get("schema_version", "0.1")),
            teacher_family=str(payload.get("teacher_family", "hf_causal_lm")),
            teacher_model_id=str(payload.get("teacher_model_id", "")),
            tokenizer_id=str(payload.get("tokenizer_id", "")),
            sequence_length=int(payload.get("sequence_length", 0)),
            dtype=str(payload.get("dtype", "")),
            local_files_only=bool(payload.get("local_files_only", True)),
            allow_downloads=bool(payload.get("allow_downloads", False)),
            targets={
                str(key): bool(value)
                for key, value in _mapping(payload.get("targets")).items()
            },
            prompt_count=int(payload.get("prompt_count", 0)),
            created_by=str(
                payload.get("created_by", "RADJAX-Tome HFTeacherExportMetadata")
            ),
            notes=tuple(str(item) for item in payload.get("notes", ())),
            extra=_mapping(payload.get("extra")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "teacher_family": self.teacher_family,
            "teacher_model_id": self.teacher_model_id,
            "tokenizer_id": self.tokenizer_id,
            "sequence_length": self.sequence_length,
            "dtype": self.dtype,
            "local_files_only": self.local_files_only,
            "allow_downloads": self.allow_downloads,
            "targets": dict(self.targets),
            "prompt_count": self.prompt_count,
            "created_by": self.created_by,
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }


def validate_hf_export_config(
    config: HFTeacherExportConfig,
) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    if not config.resolved_model_id:
        blockers.append("resolved_model_id is required")
    if config.sequence_length <= 0:
        blockers.append("sequence_length must be > 0")
    if config.prompt_count < 0:
        blockers.append("prompt_count must be non-negative")
    if config.batch_size <= 0:
        blockers.append("batch_size must be > 0")
    if config.vocab_size is not None and config.vocab_size <= 0:
        blockers.append("vocab_size must be > 0 when provided")
    if config.allow_downloads and config.local_files_only:
        blockers.append("allow_downloads and local_files_only cannot both be true")
    return not blockers, tuple(blockers)


def build_hf_export_metadata(config: HFTeacherExportConfig) -> HFTeacherExportMetadata:
    ok, blockers = validate_hf_export_config(config)
    if not ok:
        raise ValueError("invalid HF export config: " + "; ".join(blockers))
    return HFTeacherExportMetadata(
        schema_version="0.1",
        teacher_family="hf_causal_lm",
        teacher_model_id=config.resolved_model_id,
        tokenizer_id=config.tokenizer_id or config.resolved_model_id,
        sequence_length=config.sequence_length,
        dtype=config.dtype,
        local_files_only=config.local_files_only,
        allow_downloads=config.allow_downloads,
        targets={
            "input_ids": True,
            "attention_mask": True,
            "loss_mask": True,
            "hidden_states": True,
            "logits": config.include_logits,
            "attention_targets": config.include_attention_targets,
        },
        prompt_count=config.prompt_count,
        notes=("huggingface teacher exporter metadata",),
        extra={
            "policy_label": config.policy_label,
            "fallback_label": config.fallback_label,
            "revision": config.revision,
            "trust_remote_code": config.trust_remote_code,
            "device": config.device,
            "vocab_size": config.vocab_size,
            "batch_size": config.batch_size,
            **dict(config.metadata),
        },
    )


def read_hf_export_metadata(path: str | Path) -> HFTeacherExportMetadata:
    return HFTeacherExportMetadata.from_payload(read_json_object(Path(path)))


def write_hf_export_metadata(
    path: str | Path,
    metadata: HFTeacherExportMetadata | dict[str, Any],
) -> Path:
    output = Path(path)
    payload = (
        metadata.to_dict()
        if isinstance(metadata, HFTeacherExportMetadata)
        else dict(metadata)
    )
    write_json(output, payload)
    return output


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
