"""Ordinary Path-B production keeps the governed v4 M7 bytes reproducible."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from radjax_contract.tome import open_streaming_tome, validate_streaming_tome

from radjax_tome.builder import ProductionBuildConfig, build_production_gpu_tome
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance
from radjax_tome.tome.packaging import package_tome_artifact


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True)
    sources = []
    for index in range(4):
        path = root / f"source-{index:03d}.txt"
        path.write_text(
            f"ordinary Path-B deterministic record {index}; stable input text\n",
            encoding="utf-8",
        )
        sources.append(path)
    corpus = root / "corpus"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(sources), output_dir=corpus, overwrite=True)
    )
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "tiny"}), encoding="utf-8"
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    provenance = root / "teacher_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/m7-determinism"), provenance
    )
    return corpus, model, provenance


def _config(
    *, corpus: Path, model: Path, provenance: Path, output: Path
) -> ProductionBuildConfig:
    return ProductionBuildConfig(
        teacher_model=str(model),
        tokenizer_id="radjax/m7-determinism",
        dataset_path=corpus / "corpus.jsonl",
        corpus_manifest_path=corpus / "corpus_manifest.json",
        teacher_model_provenance_path=provenance,
        output_dir=output,
        teacher_backend="smoke_tokenizer",
        runtime_mode="cpu",
        target_policy="corridor_exemplar_v1",
        sequence_length=8,
        vocab_size=512,
        top_k=4,
        num_buckets=3,
        gpu_batch_size_mode="preset",
        gpu_batch_size_preset=2,
        shard_size_examples=2,
        max_examples=4,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        selected_exemplar_budget=4,
        total_selected_exemplar_budget=4,
        selection_integration_policy="corridor_first_global_backfill_v1",
        retain_unselected_exemplar_payloads=False,
    )


def test_ordinary_path_b_v4_m7_bytes_are_stable_across_fresh_builds(
    tmp_path: Path,
) -> None:
    corpus, model, provenance = _inputs(tmp_path / "inputs")
    outputs = []
    student_archives = []
    for name in ("build-a", "build-b"):
        config = _config(
            corpus=corpus,
            model=model,
            provenance=provenance,
            output=tmp_path / name / "producer_artifact",
        )
        report = build_production_gpu_tome(config)
        assert report["status"] == "pass", report["blockers"]
        archive = config.output_dir.with_name(f"{config.output_dir.name}.v4.tgz")
        assert validate_streaming_tome(archive, strict=True).ok
        outputs.append(archive)
        student = tmp_path / name / "student.v6.tgz"
        package_tome_artifact(
            config.output_dir,
            student,
            profile="student",
            archive="tgz",
            overwrite=True,
            student_contract_profile="v6",
        )
        student_archives.append(student)

    first, second = (path.read_bytes() for path in outputs)
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    with open_streaming_tome(outputs[0]) as reader:
        rows = list(reader)
        assert reader.verification_state == "fully_verified"
    with open_streaming_tome(outputs[1]) as reader:
        assert rows == list(reader)
        assert reader.verification_state == "fully_verified"
    with tarfile.open(outputs[0], "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
    assert not any(
        member.startswith(("c6/", "reports/"))
        or member
        in {
            "delivery_report.json",
            "progress_log.jsonl",
            "run_manifest.json",
            "run_plan.json",
        }
        for member in members
    )
    assert student_archives[0].read_bytes() == student_archives[1].read_bytes()
    first_student_digest = hashlib.sha256(student_archives[0].read_bytes()).hexdigest()
    second_student_digest = hashlib.sha256(student_archives[1].read_bytes()).hexdigest()
    assert first_student_digest == second_student_digest
