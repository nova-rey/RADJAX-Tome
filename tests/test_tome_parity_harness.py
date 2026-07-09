from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from radjax_tome.builder import (
    BackendTeacherTextbookBuildConfig,
    TeacherTextbookBuildConfig,
    build_backend_teacher_textbook,
    build_teacher_textbook,
)
from radjax_tome.reports import (
    TomeParityConfig,
    compare_tome_artifacts,
)
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _fake_artifact(path: Path, **overrides: object) -> Path:
    config = {
        "output_dir": path,
        "max_examples": 2,
        "sequence_length": 6,
        "batch_size": 2,
        "overwrite": True,
    }
    config.update(overrides)
    build_teacher_textbook(TeacherTextbookBuildConfig(**config))
    return path


def _backend_selector_artifact(path: Path) -> Path:
    build_backend_teacher_textbook(
        BackendTeacherTextbookBuildConfig(
            output_dir=path,
            teacher_backend="cpu_reference",
            runtime_mode="cpu",
            target_policy="corridor_exemplar_v1",
            exemplar_selection_enabled=True,
            exemplar_capture_mode="one_pass_candidate",
            exemplar_fulfillment_policy="select_from_existing_capture",
            max_examples=3,
            sequence_length=5,
            batch_size=2,
            vocab_size=64,
            top_k=4,
            num_buckets=3,
            overwrite=True,
        )
    )
    return path


def _copy(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mutate_json(path: Path, updates: dict[str, object]) -> None:
    payload = _json(path)
    payload.update(updates)
    _write_json(path, payload)


def _mutate_target_params(path: Path, updates: dict[str, object]) -> None:
    payload = _json(path / "metadata.json")
    params = dict(payload.get("target_params", {}))
    params.update(updates)
    payload["target_params"] = params
    _write_json(path / "metadata.json", payload)


def _write_first_shard(path: Path, **arrays: np.ndarray) -> None:
    shard = path / "shards" / "shard-00000.npz"
    with np.load(shard, allow_pickle=False) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    payload.update(arrays)
    np.savez(shard, **payload)


def _add_corpus_provenance(path: Path, corpus_hash: str) -> None:
    provenance = {
        "source_corpus_hash": corpus_hash,
        "source_corpus_manifest_hash": corpus_hash.replace("corpus", "manifest"),
        "source_corpus_num_examples": 2,
    }
    manifest = _json(path / "teacher_manifest.json")
    manifest["corpus_provenance"] = provenance
    _write_json(path / "teacher_manifest.json", manifest)
    cover = _json(path / "cover_page.json")
    cover["corpus"] = provenance
    _write_json(path / "cover_page.json", cover)


def _add_teacher_model_provenance(path: Path, weights_hash: str) -> None:
    provenance = {
        "schema_version": "teacher_model_provenance_v1",
        "model_source_kind": "local_directory",
        "model_identity_confidence": "declared",
        "model_provenance_mode": "user_declared",
        "model_name": "teacher",
        "model_name_source": "user_declared",
        "model_revision": "rev",
        "model_revision_source": "user_declared",
        "config_hash": "sha256:config",
        "tokenizer_hash": "sha256:tokenizer",
        "weights_hash": weights_hash,
        "model_directory_hash": "sha256:directory",
        "network_used": False,
        "local_files_only": True,
        "allow_downloads": False,
    }
    manifest = _json(path / "teacher_manifest.json")
    manifest["teacher_model_provenance"] = provenance
    _write_json(path / "teacher_manifest.json", manifest)
    cover = _json(path / "cover_page.json")
    cover["teacher_model_provenance"] = provenance
    _write_json(path / "cover_page.json", cover)


def test_parity_report_passes_for_identical_fake_artifacts(tmp_path: Path) -> None:
    left = _fake_artifact(tmp_path / "left")
    right = _copy(left, tmp_path / "right")

    report = compare_tome_artifacts(left, right)

    assert report.status == "pass"
    assert report.summary["schema_parity"] == "pass"
    assert report.summary["array_parity"] == "pass"
    assert report.summary["metadata_truth"] == "pass"


def test_created_at_differences_do_not_break_parity(tmp_path: Path) -> None:
    left = _fake_artifact(tmp_path / "left")
    right = _copy(left, tmp_path / "right")
    _mutate_json(right / "metadata.json", {"created_at": "2026-07-07T00:00:01+00:00"})
    _mutate_json(
        right / "teacher_manifest.json",
        {"created_at": "2026-07-07T00:00:01+00:00"},
    )

    report = compare_tome_artifacts(left, right)

    assert report.status == "pass"


def test_target_type_missing_sidecar_shape_and_nonfinite_failures(
    tmp_path: Path,
) -> None:
    left = _fake_artifact(tmp_path / "left")

    target_mismatch = _copy(left, tmp_path / "target_mismatch")
    _mutate_json(
        target_mismatch / "metadata.json",
        {"target_type": "topk_with_tail_v0"},
    )
    assert compare_tome_artifacts(left, target_mismatch).status == "fail"

    missing_sidecar = _copy(left, tmp_path / "missing_sidecar")
    (missing_sidecar / "cover_page.json").unlink()
    missing_report = compare_tome_artifacts(left, missing_sidecar)
    assert missing_report.status == "fail"
    assert any("missing required sidecar" in item for item in missing_report.blockers)

    shape_mismatch = _copy(left, tmp_path / "shape_mismatch")
    _write_first_shard(
        shape_mismatch,
        logits=np.zeros((1, 6, 32), dtype=np.float32),
    )
    assert compare_tome_artifacts(left, shape_mismatch).status == "fail"

    nonfinite = _copy(left, tmp_path / "nonfinite")
    logits = np.zeros((2, 6, 32), dtype=np.float32)
    logits[0, 0, 0] = np.nan
    _write_first_shard(nonfinite, logits=logits)
    nonfinite_report = compare_tome_artifacts(left, nonfinite)
    assert nonfinite_report.status == "fail"
    assert any("non-finite" in item for item in nonfinite_report.blockers)


def test_numeric_diffs_and_no_compare_values_mode(tmp_path: Path) -> None:
    left = _fake_artifact(tmp_path / "left")
    right = _copy(left, tmp_path / "right")
    with np.load(right / "shards" / "shard-00000.npz", allow_pickle=False) as loaded:
        logits = loaded["logits"].copy()
    logits[0, 0, 0] += 0.25
    _write_first_shard(right, logits=logits)

    strict_report = compare_tome_artifacts(left, right)
    assert strict_report.status == "fail"
    assert strict_report.array_comparisons
    assert any(
        item.get("max_abs_diff") == 0.25 for item in strict_report.array_comparisons
    )

    shape_only = compare_tome_artifacts(
        left,
        right,
        TomeParityConfig(compare_values=False),
    )
    assert shape_only.status == "pass"
    assert shape_only.summary["numeric_parity"] == "pass"


def test_corpus_and_teacher_model_provenance_comparisons(tmp_path: Path) -> None:
    left = _fake_artifact(tmp_path / "left")
    right = _copy(left, tmp_path / "right")
    _add_corpus_provenance(left, "sha256:corpus-a")
    _add_corpus_provenance(right, "sha256:corpus-b")
    assert compare_tome_artifacts(left, right).status == "fail"

    one_sided_corpus = _copy(left, tmp_path / "one_sided_corpus")
    no_corpus = _copy(left, tmp_path / "no_corpus")
    manifest = _json(no_corpus / "teacher_manifest.json")
    manifest.pop("corpus_provenance")
    _write_json(no_corpus / "teacher_manifest.json", manifest)
    cover = _json(no_corpus / "cover_page.json")
    cover.pop("corpus")
    _write_json(no_corpus / "cover_page.json", cover)
    assert compare_tome_artifacts(one_sided_corpus, no_corpus).status == "warn"

    teacher_left = _fake_artifact(tmp_path / "teacher_left")
    teacher_right = _copy(teacher_left, tmp_path / "teacher_right")
    _add_teacher_model_provenance(teacher_left, "sha256:weights-a")
    _add_teacher_model_provenance(teacher_right, "sha256:weights-b")
    assert compare_tome_artifacts(teacher_left, teacher_right).status == "fail"

    one_sided_teacher = _copy(teacher_left, tmp_path / "one_sided_teacher")
    no_teacher = _copy(teacher_left, tmp_path / "no_teacher")
    manifest = _json(no_teacher / "teacher_manifest.json")
    manifest.pop("teacher_model_provenance")
    _write_json(no_teacher / "teacher_manifest.json", manifest)
    cover = _json(no_teacher / "cover_page.json")
    cover.pop("teacher_model_provenance")
    _write_json(no_teacher / "cover_page.json", cover)
    assert compare_tome_artifacts(one_sided_teacher, no_teacher).status == "warn"


def test_selector_manifest_overlap_and_forbidden_claims(tmp_path: Path) -> None:
    left = _backend_selector_artifact(tmp_path / "left")
    right = _copy(left, tmp_path / "right")

    report = compare_tome_artifacts(left, right)

    assert report.status == "pass"
    selector = report.selector_manifest_comparison
    assert selector["fields"]["selection_policy"]["status"] == "pass"
    assert selector["selected_example_jaccard"] == 1.0
    assert selector["selected_position_jaccard"] == 1.0

    manifest = _json(right / "exemplar_selection_manifest.json")
    manifest["production_global_selector"] = True
    _write_json(right / "exemplar_selection_manifest.json", manifest)
    assert compare_tome_artifacts(left, right).status == "fail"


def test_metadata_sanity_and_forbidden_claims_fail(tmp_path: Path) -> None:
    left = _fake_artifact(tmp_path / "left")
    right = _copy(left, tmp_path / "right")
    _write_json(
        right / "metadata_sanity_report.json",
        {
            "report_schema": "artifact_metadata_sanity_report_v1",
            "status": "fail",
            "blockers": ["bad"],
            "warnings": [],
        },
    )
    assert compare_tome_artifacts(left, right).status == "fail"

    forbidden = _copy(left, tmp_path / "forbidden")
    _mutate_target_params(forbidden, {"production_global_selector": "true"})
    forbidden_report = compare_tome_artifacts(left, forbidden)
    assert forbidden_report.status == "fail"
    assert any("forbidden" in item for item in forbidden_report.blockers)


def test_parity_cli_writes_report_and_fails_on_mismatch(tmp_path: Path) -> None:
    left = _fake_artifact(tmp_path / "left")
    right = _copy(left, tmp_path / "right")
    output = tmp_path / "parity_report.json"

    result = run_cli(
        ROOT,
        "parity",
        "--left",
        str(left),
        "--right",
        str(right),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert "status=pass" in result.stdout
    assert "schema_parity=pass" in result.stdout
    assert output.is_file()
    payload = _json(output)
    assert payload["report_schema"] == "tome_parity_report_v1"

    bad = _copy(left, tmp_path / "bad")
    _mutate_json(bad / "metadata.json", {"target_type": "topk_with_tail_v0"})
    fail_output = tmp_path / "parity_fail.json"
    failed = run_cli(
        ROOT,
        "parity",
        "--left",
        str(left),
        "--right",
        str(bad),
        "--output",
        str(fail_output),
    )
    assert failed.returncode == 1
    assert "status=fail" in failed.stdout


def test_docs_and_bible_mention_spec_4_3() -> None:
    bible = (ROOT / "bible.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "PARITY_HARNESS.md").read_text(encoding="utf-8")
    cli = (ROOT / "docs" / "CLI_GUIDE.md").read_text(encoding="utf-8")

    assert "Spec 4.3" in bible
    assert "parity" in bible
    assert "radjax-tome parity" in docs
    assert "exact floating equality is not required" in docs
    assert "corpus/model provenance affects parity" in docs
    assert "radjax-tome parity" in cli
