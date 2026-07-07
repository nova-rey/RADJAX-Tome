from __future__ import annotations

import json
from pathlib import Path

from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.builder import (
    BackendTeacherTextbookBuildConfig,
    TeacherTextbookBuildConfig,
    build_backend_teacher_textbook,
    build_teacher_textbook,
)
from radjax_tome.reports import (
    build_artifact_metadata_sanity_report,
    build_runtime_doctor_report,
    render_artifact_metadata_sanity_summary,
    render_runtime_doctor_summary,
    write_artifact_metadata_sanity_report,
    write_runtime_doctor_report,
)
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _backend_config(
    tmp_path: Path,
    *,
    output_name: str = "backend_tome",
    **overrides: object,
) -> BackendTeacherTextbookBuildConfig:
    payload = {
        "output_dir": tmp_path / output_name,
        "teacher_backend": "cpu_reference",
        "runtime_mode": "cpu",
        "target_policy": "dynamic_cascaded_soft_labels_v1",
        "sequence_length": 5,
        "batch_size": 2,
        "max_examples": 2,
        "vocab_size": 11,
        "top_k": 4,
        "num_buckets": 3,
        "overwrite": True,
    }
    payload.update(overrides)
    return BackendTeacherTextbookBuildConfig(**payload)


def _read_metadata(path: Path) -> dict[str, object]:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _set_target_param(path: Path, key: str, value: object) -> None:
    metadata = _read_metadata(path)
    params = dict(metadata["target_params"])  # type: ignore[arg-type]
    params[key] = value
    metadata["target_params"] = params
    _write_metadata(path, metadata)


def test_runtime_doctor_report_for_cpu_reference_is_json_serializable(
    tmp_path: Path,
) -> None:
    report = build_runtime_doctor_report(
        TeacherBackendConfig(
            backend_id="cpu_reference",
            runtime_mode="cpu",
            target_policy="dense_logits",
        )
    )
    output = tmp_path / "runtime_doctor.json"
    write_runtime_doctor_report(report, output)

    assert report["report_schema"] == "runtime_doctor_report_v1"
    assert report["backend_id"] == "cpu_reference"
    assert report["can_emit"] is True
    assert report["failure_stage"] == "none"
    assert report["dependency_status"] == "not_required"
    assert json.loads(output.read_text(encoding="utf-8"))["can_emit"] is True


def test_runtime_doctor_gpu_torch_unavailable_reports_remediation() -> None:
    report = build_runtime_doctor_report(
        TeacherBackendConfig(
            backend_id="gpu_torch",
            runtime_mode="cpu_gpu",
            target_policy="corridor_exemplar_v1",
            model_id="missing-local-hf-model",
            tokenizer_id="missing-local-hf-model",
            local_files_only=True,
            allow_downloads=False,
        )
    )
    rendered = "\n".join(render_runtime_doctor_summary(report))

    assert report["report_schema"] == "runtime_doctor_report_v1"
    assert "torch_available" in report
    assert "transformers_available" in report
    assert "accelerator_available" in report
    assert report["can_emit"] in {True, False}
    if not report["can_emit"]:
        assert report["remediation_hint"]
        assert "can_emit=false" in rendered


def test_artifact_metadata_sanity_passes_backend_routed_dynamic(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dynamic"
    build_backend_teacher_textbook(_backend_config(tmp_path, output_dir=output))

    report = build_artifact_metadata_sanity_report(output)

    assert report["report_schema"] == "artifact_metadata_sanity_report_v1"
    assert report["status"] == "pass"
    assert report["target_policy"] == "dynamic_cascaded_soft_labels_v1"
    assert report["backend_id"] == "cpu_reference"
    assert report["effective_backend_id"] == "cpu_reference"
    assert report["multidevice_enabled"] is False
    assert report["batch_partition_strategy"] == "single_device"


def test_artifact_metadata_sanity_passes_one_pass_selector(
    tmp_path: Path,
) -> None:
    output = tmp_path / "one_pass_selector"
    build_backend_teacher_textbook(
        _backend_config(
            tmp_path,
            output_dir=output,
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="one_pass_candidate",
            exemplar_selection_enabled=True,
            exemplar_selection_board_capacity=2,
        )
    )

    report = build_artifact_metadata_sanity_report(output)

    assert report["status"] == "pass"
    assert report["exemplar_selection_enabled"] is True
    assert report["exemplar_fulfillment_policy"] == "select_from_existing_capture"
    assert report["deduplication_policy"] == (
        "rank_aware_board_assignment_with_backfill_v1"
    )
    assert report["production_global_selector"] is False


def test_artifact_metadata_sanity_passes_score_pass_rerun_selector(
    tmp_path: Path,
) -> None:
    output = tmp_path / "score_pass"
    build_backend_teacher_textbook(
        _backend_config(
            tmp_path,
            output_dir=output,
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="two_pass_sparse_exemplar",
            exemplar_selection_enabled=True,
            exemplar_selection_board_capacity=2,
        )
    )

    report = build_artifact_metadata_sanity_report(output)

    assert report["status"] == "pass"
    assert report["target_type"] == "corridor_exemplar_score_pass_v1"
    assert report["exemplar_capture_stage"] == "score_pass"
    assert report["requires_second_pass_for_final_exemplars"] is True
    assert report["exemplar_fulfillment_policy"] == "rerun_selected_capture"


def test_artifact_metadata_sanity_fails_inconsistent_score_pass(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bad_score_pass"
    build_backend_teacher_textbook(
        _backend_config(
            tmp_path,
            output_dir=output,
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="two_pass_sparse_exemplar",
        )
    )
    _set_target_param(output, "exemplar_capture_stage", "selected_exemplar_pass")

    report = build_artifact_metadata_sanity_report(output)

    assert report["status"] == "fail"
    assert any("score_pass" in blocker for blocker in report["blockers"])


def test_artifact_metadata_sanity_fails_gpu_backend_mismatch_without_fallback(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bad_backend"
    build_backend_teacher_textbook(_backend_config(tmp_path, output_dir=output))
    _set_target_param(output, "requested_backend_id", "gpu_torch")
    _set_target_param(output, "backend_id", "gpu_torch")
    _set_target_param(output, "effective_backend_id", "cpu_reference")
    _set_target_param(output, "fallback_used", "false")

    report = build_artifact_metadata_sanity_report(output)

    assert report["status"] == "fail"
    assert any("requested gpu_torch" in blocker for blocker in report["blockers"])


def test_artifact_metadata_sanity_checks_false_claims_and_multidevice(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bad_claims"
    build_backend_teacher_textbook(_backend_config(tmp_path, output_dir=output))
    _set_target_param(output, "production_global_selector", "true")
    _set_target_param(output, "multidevice_enabled", "false")
    _set_target_param(output, "batch_partition_strategy", "round_robin")

    report = build_artifact_metadata_sanity_report(output)

    assert report["status"] == "fail"
    assert any("production_global_selector" in item for item in report["blockers"])
    assert any("single_device" in item for item in report["blockers"])


def test_artifact_metadata_sanity_does_not_fail_old_dense_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "old_dense"
    build_teacher_textbook(
        TeacherTextbookBuildConfig(
            output_dir=output,
            max_examples=2,
            sequence_length=5,
            overwrite=True,
        )
    )

    report = build_artifact_metadata_sanity_report(output)

    assert report["status"] in {"pass", "warn"}
    assert not report["blockers"]


def test_artifact_metadata_sanity_render_and_write_json(tmp_path: Path) -> None:
    output = tmp_path / "dynamic"
    build_backend_teacher_textbook(_backend_config(tmp_path, output_dir=output))
    report = build_artifact_metadata_sanity_report(output)
    report_path = tmp_path / "metadata_sanity.json"
    write_artifact_metadata_sanity_report(report, report_path)
    rendered = "\n".join(render_artifact_metadata_sanity_summary(report))

    assert "metadata_sanity_status=pass" in rendered
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "pass"


def test_cli_inspect_and_validate_metadata_sanity(tmp_path: Path) -> None:
    output = tmp_path / "cli_sanity"
    build_backend_teacher_textbook(_backend_config(tmp_path, output_dir=output))

    inspect = run_cli(ROOT, "inspect", "--path", str(output), "--metadata-sanity")
    validate = run_cli(
        ROOT,
        "validate",
        "--path",
        str(output),
        "--metadata-sanity",
        "--write-report",
    )

    assert inspect.returncode == 0, inspect.stderr
    assert "metadata_sanity_status=pass" in inspect.stdout
    assert validate.returncode == 0, validate.stderr
    assert "metadata_sanity_status=pass" in validate.stdout
    assert (output / "metadata_sanity_report.json").is_file()


def test_cli_doctor_accepts_backend_runtime_target_flags(tmp_path: Path) -> None:
    report_path = tmp_path / "doctor.json"
    result = run_cli(
        ROOT,
        "doctor",
        "--teacher-backend",
        "gpu_torch",
        "--runtime-mode",
        "cpu_gpu",
        "--target-policy",
        "corridor_exemplar_v1",
        "--write-report",
        str(report_path),
    )

    assert result.returncode == 0, result.stderr
    assert "backend=gpu_torch" in result.stdout
    assert "runtime_mode=cpu_gpu" in result.stdout
    assert "target_policy=corridor_exemplar_v1" in result.stdout
    assert report_path.is_file()
