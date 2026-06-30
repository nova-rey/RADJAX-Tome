from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from radjax_tome.fingerprint.artifacts import (
    BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE,
    BEHAVIORAL_FINGERPRINT_VERSION,
    BUDGET_SUBSET_SCHEMA_VERSION,
    FingerprintBudgetSubsetReceipt,
    FingerprintByteAccounting,
    FingerprintManifest,
    validate_fingerprint_artifact,
    validate_fingerprint_byte_accounting,
    write_fingerprint_byte_accounting,
    write_fingerprint_manifest,
)
from radjax_tome.fingerprint.corridor import (
    CorridorMeasurementRecord,
    CorridorMeasurementReport,
    validate_corridor_measurement_report,
    write_corridor_measurement_report,
)
from radjax_tome.fingerprint.exemplars import (
    FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE,
    FingerprintExemplarManifest,
    FingerprintExemplarRecord,
    FingerprintExemplarShard,
    summarize_exemplar_reservoir,
    validate_fingerprint_exemplar_records,
    write_fingerprint_exemplar_manifest,
    write_fingerprint_exemplar_records,
)
from radjax_tome.targets.store import TeacherTargetStore


def build_minimal_fingerprint_artifact_from_target_store(
    target_store: TeacherTargetStore | str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    store = (
        target_store
        if isinstance(target_store, TeacherTargetStore)
        else TeacherTargetStore.open(target_store)
    )
    store.validate()
    root = Path(output_dir)
    if root.exists():
        if not overwrite:
            raise ValueError(f"fingerprint artifact already exists: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    arrays = store.read_shard(0)
    input_ids = arrays["input_ids"]
    modes_path = root / "modes.json"
    modes_path.write_text(
        json.dumps(
            {
                "modes": [
                    {
                        "mode_id": 0,
                        "entropy": 0.0,
                        "top1_margin": 0.0,
                        "top8_mass": 1.0,
                        "top32_mass": 1.0,
                        "tail_mass": 0.0,
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    target_path = root / "targets-00000.jsonl"
    rows = []
    for index, token_ids in enumerate(input_ids):
        rows.append(
            {
                "example_id": f"ex-{index}",
                "input_ids": [int(value) for value in token_ids],
                "mode_id": 0,
                "position": 0,
                "bounds": {},
            }
        )
    target_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = FingerprintManifest(
        artifact_type=BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE,
        artifact_version=BEHAVIORAL_FINGERPRINT_VERSION,
        created_by="radjax_tome.fingerprint.generation",
        teacher={
            "model_name": store.metadata.model_id,
            "tokenizer_name": store.metadata.tokenizer_id,
            "vocab_size": store.metadata.vocab_size,
        },
        sequence={
            "max_seq_len": store.metadata.sequence_length,
            "target_positions": len(rows),
        },
        stats={
            "tracked": [
                "entropy",
                "top1_margin",
                "top8_mass",
                "top32_mass",
                "tail_mass",
            ]
        },
        modes_file="modes.json",
        target_shards=({"path": "targets-00000.jsonl", "num_records": len(rows)},),
    )
    write_fingerprint_manifest(root, manifest)
    accounting = FingerprintByteAccounting(
        physical_subset_bytes=target_path.stat().st_size,
        logical_payload_bytes_selected=target_path.stat().st_size,
        logical_payload_bytes_consumed=target_path.stat().st_size,
        arm_charged_bytes=target_path.stat().st_size,
        corridor_charged_bytes=target_path.stat().st_size,
        exemplar_charged_bytes=0,
    )
    write_fingerprint_byte_accounting(root / "byte_accounting.json", accounting)
    validation = validate_fingerprint_artifact(root)
    if not validation.ok:
        raise ValueError("; ".join(validation.blockers))
    accounting_validation = validate_fingerprint_byte_accounting(accounting)
    if not accounting_validation.ok:
        raise ValueError("; ".join(accounting_validation.blockers))
    return root


def generate_corridor_subset_receipt(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    declared_byte_budget: int = 4096,
) -> FingerprintBudgetSubsetReceipt:
    root = Path(artifact_dir)
    manifest_path = root / "manifest.json"
    shard_path = root / "targets-00000.jsonl"
    receipt = FingerprintBudgetSubsetReceipt(
        schema_version=BUDGET_SUBSET_SCHEMA_VERSION,
        subset_role="corridor_subset",
        declared_byte_budget=declared_byte_budget,
        selection_policy="deterministic_first_shard_v1",
        selected_record_count=_count_jsonl_records(shard_path),
        artifact_manifest_sha256=_file_sha256(manifest_path),
        shard_hashes={"targets-00000.jsonl": _file_sha256(shard_path)},
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def generate_corridor_measurement_report(
    artifact_dir: str | Path,
    output_path: str | Path,
) -> CorridorMeasurementReport:
    records = (
        CorridorMeasurementRecord(
            example_id="ex-0",
            position=0,
            mode_id=0,
            stats={
                "entropy": 0.0,
                "top1_margin": 0.0,
                "top8_mass": 1.0,
                "top32_mass": 1.0,
                "tail_mass": 0.0,
            },
            bounds={
                "entropy": (0.0, 0.0),
                "top1_margin": (0.0, 0.0),
                "top8_mass": (1.0, 1.0),
                "top32_mass": (1.0, 1.0),
                "tail_mass": (0.0, 0.0),
            },
        ),
    )
    report = CorridorMeasurementReport(
        status="pass",
        artifact_dir=str(artifact_dir),
        records_measured=len(records),
        modes_measured=1,
        measurements=records,
        metadata={"generation": "deterministic_producer_subset"},
    )
    ok, blockers = validate_corridor_measurement_report(report)
    if not ok:
        raise ValueError("; ".join(blockers))
    write_corridor_measurement_report(output_path, report)
    return report


def generate_exemplar_reservoir(
    artifact_dir: str | Path,
    *,
    max_seq_len: int,
    vocab_size: int,
) -> FingerprintExemplarManifest:
    root = Path(artifact_dir)
    exemplar_dir = root / "exemplars"
    exemplar_dir.mkdir(parents=True, exist_ok=True)
    records = (
        FingerprintExemplarRecord(
            example_id="ex-0",
            input_ids=tuple(range(max_seq_len)),
            position=0,
            weight=1.0,
            teacher_probs=tuple([1.0 / vocab_size] * vocab_size),
            reason_codes=("deterministic_smoke",),
        ),
    )
    validation = validate_fingerprint_exemplar_records(
        records,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        payload_type=FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE,
    )
    if not validation.ok:
        raise ValueError("; ".join(validation.blockers))
    shard_path = exemplar_dir / "exemplars-00000.jsonl"
    write_fingerprint_exemplar_records(shard_path, records)
    manifest = FingerprintExemplarManifest(
        payload_type=FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE,
        shards=(
            FingerprintExemplarShard(
                path="exemplars/exemplars-00000.jsonl",
                num_records=len(records),
                payload_type=FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE,
                sha256=_file_sha256(shard_path),
            ),
        ),
        num_records=len(records),
        selection_policy="deterministic_first_record_v1",
    )
    write_fingerprint_exemplar_manifest(exemplar_dir / "manifest.json", manifest)
    original = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    original["exemplar_reservoir"] = manifest.to_dict()
    (root / "manifest.json").write_text(
        json.dumps(original, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = summarize_exemplar_reservoir(root)
    if summary.num_records != len(records):
        raise ValueError("exemplar reservoir summary count mismatch")
    return manifest


def _count_jsonl_records(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
