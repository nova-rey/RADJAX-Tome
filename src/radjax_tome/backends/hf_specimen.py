from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json

DEFAULT_HF_SPECIMEN_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
HF_SPECIMEN_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "qwen_specific_support",
    "gpt2_specific_architecture",
    "student_consumption_proven",
    "training_ready",
    "tokenizer_remapping_supported",
    "production_distillation_ready",
    "full_model_quality_proven",
)


@dataclass(frozen=True)
class HFTeacherSpecimenConfig:
    model_id: str = DEFAULT_HF_SPECIMEN_MODEL_ID
    prompts: tuple[str, ...] = ("hello",)
    sequence_length: int = 8
    local_files_only: bool = True
    allow_downloads: bool = False
    tokenizer_id: str | None = None
    revision: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HFTeacherSpecimenConfig:
        return cls(
            model_id=str(payload.get("model_id", DEFAULT_HF_SPECIMEN_MODEL_ID)),
            prompts=tuple(str(item) for item in payload.get("prompts", ("hello",))),
            sequence_length=int(payload.get("sequence_length", 8)),
            local_files_only=bool(payload.get("local_files_only", True)),
            allow_downloads=bool(payload.get("allow_downloads", False)),
            tokenizer_id=(
                str(payload["tokenizer_id"]) if payload.get("tokenizer_id") else None
            ),
            revision=str(payload["revision"]) if payload.get("revision") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class HFTeacherSpecimenSmokeResult:
    status: str
    scope: str
    model_id: str
    local_files_only: bool
    allow_downloads: bool
    target_store_path: str
    target_store_validated: bool
    vocab_contract_extracted: bool
    tokenizer_id: str | None
    tokenizer_hash: str | None
    vocab_size: int | None
    sequence_length: int
    num_examples: int
    target_type: str | None
    logits_shape: tuple[int, ...] | None
    claims_not_made: tuple[str, ...]
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    phase: str = "P104"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HFTeacherSpecimenSmokeResult:
        return cls(
            status=str(payload.get("status", "")),
            scope=str(payload.get("scope", "")),
            model_id=str(payload.get("model_id", "")),
            local_files_only=bool(payload.get("local_files_only", True)),
            allow_downloads=bool(payload.get("allow_downloads", False)),
            target_store_path=str(payload.get("target_store_path", "")),
            target_store_validated=bool(payload.get("target_store_validated", False)),
            vocab_contract_extracted=bool(
                payload.get("vocab_contract_extracted", False)
            ),
            tokenizer_id=(
                str(payload["tokenizer_id"]) if payload.get("tokenizer_id") else None
            ),
            tokenizer_hash=(
                str(payload["tokenizer_hash"])
                if payload.get("tokenizer_hash")
                else None
            ),
            vocab_size=(
                int(payload["vocab_size"])
                if payload.get("vocab_size") is not None
                else None
            ),
            sequence_length=int(payload.get("sequence_length", 0)),
            num_examples=int(payload.get("num_examples", 0)),
            target_type=(
                str(payload["target_type"]) if payload.get("target_type") else None
            ),
            logits_shape=(
                tuple(int(item) for item in payload["logits_shape"])
                if payload.get("logits_shape") is not None
                else None
            ),
            claims_not_made=tuple(
                str(item)
                for item in payload.get("claims_not_made", HF_SPECIMEN_CLAIMS_NOT_MADE)
            ),
            reason=str(payload["reason"]) if payload.get("reason") else None,
            error_type=(
                str(payload["error_type"]) if payload.get("error_type") else None
            ),
            error_message=(
                str(payload["error_message"]) if payload.get("error_message") else None
            ),
            phase=str(payload.get("phase", "P104")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class HFTeacherSpecimenSwapReport:
    status: str
    scope: str
    specimen_count: int
    passed: int
    unavailable: int
    failed: int
    model_ids: tuple[str, ...]
    specimens: tuple[HFTeacherSpecimenSmokeResult, ...]
    claims_not_made: tuple[str, ...] = HF_SPECIMEN_CLAIMS_NOT_MADE
    phase: str = "P105"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HFTeacherSpecimenSwapReport:
        return cls(
            status=str(payload.get("status", "")),
            scope=str(payload.get("scope", "")),
            specimen_count=int(payload.get("specimen_count", 0)),
            passed=int(payload.get("passed", 0)),
            unavailable=int(payload.get("unavailable", 0)),
            failed=int(payload.get("failed", 0)),
            model_ids=tuple(str(item) for item in payload.get("model_ids", ())),
            specimens=tuple(
                HFTeacherSpecimenSmokeResult.from_payload(dict(item))
                for item in payload.get("specimens", ())
            ),
            claims_not_made=tuple(
                str(item)
                for item in payload.get("claims_not_made", HF_SPECIMEN_CLAIMS_NOT_MADE)
            ),
            phase=str(payload.get("phase", "P105")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scope": self.scope,
            "specimen_count": self.specimen_count,
            "passed": self.passed,
            "unavailable": self.unavailable,
            "failed": self.failed,
            "model_ids": list(self.model_ids),
            "specimens": [specimen.to_dict() for specimen in self.specimens],
            "claims_not_made": list(self.claims_not_made),
            "phase": self.phase,
        }


SpecimenRunner = Callable[[HFTeacherSpecimenConfig, Path], HFTeacherSpecimenSmokeResult]


def validate_hf_teacher_specimen_config(config: HFTeacherSpecimenConfig) -> None:
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if not config.prompts:
        raise ValueError("prompts must contain at least one prompt")
    if config.allow_downloads and config.local_files_only:
        raise ValueError("allow_downloads and local_files_only cannot both be true")


def build_hf_teacher_specimen_dry_run(
    config: HFTeacherSpecimenConfig,
    *,
    target_store: str | Path,
) -> HFTeacherSpecimenSmokeResult:
    validate_hf_teacher_specimen_config(config)
    return HFTeacherSpecimenSmokeResult(
        status="dry_run",
        scope="tiny_hf_causal_lm_teacher_specimen_smoke",
        model_id=config.model_id,
        local_files_only=config.local_files_only,
        allow_downloads=config.allow_downloads,
        target_store_path=str(target_store),
        target_store_validated=False,
        vocab_contract_extracted=False,
        tokenizer_id=config.tokenizer_id,
        tokenizer_hash=None,
        vocab_size=None,
        sequence_length=config.sequence_length,
        num_examples=len(config.prompts),
        target_type=None,
        logits_shape=None,
        claims_not_made=HF_SPECIMEN_CLAIMS_NOT_MADE,
        reason="dry_run_no_hf_backend_invoked",
    )


def run_hf_teacher_specimen_smoke(
    config: HFTeacherSpecimenConfig,
    *,
    target_store: str | Path,
    runner: SpecimenRunner | None = None,
) -> HFTeacherSpecimenSmokeResult:
    validate_hf_teacher_specimen_config(config)
    target_store_path = Path(target_store)
    if runner is None:
        return HFTeacherSpecimenSmokeResult(
            status="unavailable",
            scope="tiny_hf_causal_lm_teacher_specimen_smoke",
            model_id=config.model_id,
            local_files_only=config.local_files_only,
            allow_downloads=config.allow_downloads,
            target_store_path=str(target_store_path),
            target_store_validated=False,
            vocab_contract_extracted=False,
            tokenizer_id=config.tokenizer_id,
            tokenizer_hash=None,
            vocab_size=None,
            sequence_length=config.sequence_length,
            num_examples=len(config.prompts),
            target_type=None,
            logits_shape=None,
            claims_not_made=HF_SPECIMEN_CLAIMS_NOT_MADE,
            reason="optional_hf_runner_not_provided",
        )
    try:
        return runner(config, target_store_path)
    except Exception as exc:
        return HFTeacherSpecimenSmokeResult(
            status="fail",
            scope="tiny_hf_causal_lm_teacher_specimen_smoke",
            model_id=config.model_id,
            local_files_only=config.local_files_only,
            allow_downloads=config.allow_downloads,
            target_store_path=str(target_store_path),
            target_store_validated=False,
            vocab_contract_extracted=False,
            tokenizer_id=config.tokenizer_id,
            tokenizer_hash=None,
            vocab_size=None,
            sequence_length=config.sequence_length,
            num_examples=len(config.prompts),
            target_type=None,
            logits_shape=None,
            claims_not_made=HF_SPECIMEN_CLAIMS_NOT_MADE,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def build_hf_teacher_specimen_swap_report(
    specimens: Sequence[HFTeacherSpecimenSmokeResult],
) -> HFTeacherSpecimenSwapReport:
    if not specimens:
        raise ValueError("specimens must contain at least one specimen")
    passed = sum(1 for item in specimens if item.status == "pass")
    unavailable = sum(1 for item in specimens if item.status == "unavailable")
    failed = sum(1 for item in specimens if item.status == "fail")
    dry_run = sum(1 for item in specimens if item.status == "dry_run")
    status = "pass" if failed == 0 else "fail"
    if passed == 0 and dry_run:
        status = "dry_run"
    return HFTeacherSpecimenSwapReport(
        status=status,
        scope="tiny_hf_causal_lm_teacher_specimen_swap_smoke",
        specimen_count=len(specimens),
        passed=passed,
        unavailable=unavailable + dry_run,
        failed=failed,
        model_ids=tuple(item.model_id for item in specimens),
        specimens=tuple(specimens),
    )


def read_hf_teacher_specimen_report(
    path: str | Path,
) -> HFTeacherSpecimenSmokeResult | HFTeacherSpecimenSwapReport:
    payload = read_json_object(Path(path))
    if "specimens" in payload:
        return HFTeacherSpecimenSwapReport.from_payload(payload)
    return HFTeacherSpecimenSmokeResult.from_payload(payload)


def write_hf_teacher_specimen_report(
    path: str | Path,
    report: HFTeacherSpecimenSmokeResult | HFTeacherSpecimenSwapReport,
) -> Path:
    output = Path(path)
    write_json(output, report.to_dict())
    return output


def _unused_mapping_type_hint(_: Mapping[str, Any]) -> None:
    return None
