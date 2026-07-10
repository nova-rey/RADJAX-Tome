from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from radjax_tome.builder import validate_teacher_textbook
from radjax_tome.io.json import write_json
from radjax_tome.tome import validate_tome_cover_page, write_cover_page
from tests.helpers.fixtures import build_fake_teacher_textbook_artifact
from tests.helpers.subprocess import run_cli, run_repo_python

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TOP_LEVEL_FIELDS = {
    "artifact_kind",
    "cover_page_version",
    "tome_version",
    "layout",
    "created_by",
    "created_at",
    "source_artifact_type",
    "teacher",
    "tokenizer",
    "targets",
    "contents",
    "behavioral_surfaces",
    "recommended_training_plan",
    "validation",
    "claims_not_made",
}
REQUIRED_CONTENTS = {
    "metadata.json",
    "vocab_contract.json",
    "teacher_manifest.json",
    "emission_config.json",
    "validation_report.json",
    "shards/shard-00000.npz",
}


def test_fake_build_writes_deterministic_cover_page(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    cover_page_path = artifact / "cover_page.json"
    raw = cover_page_path.read_text(encoding="utf-8")
    cover_page = json.loads(raw)

    assert cover_page_path.is_file()
    assert raw.endswith("\n")
    assert raw == json.dumps(cover_page, indent=2, sort_keys=True) + "\n"
    assert REQUIRED_TOP_LEVEL_FIELDS <= set(cover_page)
    assert cover_page["artifact_kind"] == "radjax_tome"
    assert cover_page["cover_page_version"] == 2
    assert cover_page["tome_version"] == 1
    assert cover_page["layout"] == "unpacked_directory"
    assert cover_page["source_artifact_type"] == "teacher_textbook"
    assert cover_page["teacher"]["model_id"] == "fake-deterministic-teacher"
    assert cover_page["teacher"]["backend_type"] == "fake"
    assert cover_page["tokenizer"]["vocab_contract_path"] == "vocab_contract.json"
    assert cover_page["targets"]["target_type"] == "dense_logits"
    assert cover_page["targets"]["num_examples"] == 2
    assert cover_page["behavioral_surfaces"] == []
    assert cover_page["recommended_training_plan"]["passes"] == []


def test_cover_page_contents_include_sidecars_and_matching_hashes(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    cover_page = json.loads((artifact / "cover_page.json").read_text(encoding="utf-8"))
    contents = {entry["path"]: entry for entry in cover_page["contents"]}

    assert REQUIRED_CONTENTS <= set(contents)
    assert "cover_page.json" not in contents
    for relative_path, entry in contents.items():
        path = artifact / relative_path
        assert entry["size_bytes"] == path.stat().st_size
        assert entry["sha256"] == _sha256(path)
        assert isinstance(entry["required"], bool)
        assert entry["classification"] in {
            "training_critical",
            "integrity_or_provenance",
            "diagnostic",
            "human_readable",
            "operational",
        }


def test_cover_page_validation_passes_and_fails_on_tampering(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)

    report = validate_tome_cover_page(artifact)

    assert report.status == "pass"
    assert report.artifact_kind_ok
    assert report.version_ok
    assert report.layout_ok
    assert report.contents_ok
    assert report.hashes_ok
    assert report.required_fields_ok

    validation_report = artifact / "validation_report.json"
    validation_report.write_text(
        validation_report.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    tampered = validate_tome_cover_page(artifact)
    assert tampered.status == "fail"
    assert any("sha256 mismatch" in blocker for blocker in tampered.blockers)


def test_legacy_teacher_textbook_without_cover_page_still_validates(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    (artifact / "cover_page.json").unlink()

    report = validate_teacher_textbook(artifact)
    cli = run_cli(ROOT, "validate", "--path", str(artifact))

    assert report.status == "pass"
    assert cli.returncode == 0, cli.stderr
    assert "status=pass" in cli.stdout
    assert "cover_page_status" not in cli.stdout


def test_public_cli_validate_and_inspect_are_cover_page_aware(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "cli_tome"
    build = run_cli(
        ROOT,
        "build",
        "--output",
        str(artifact),
        "--teacher-mode",
        "fake",
        "--max-examples",
        "2",
        "--sequence-length",
        "8",
        "--overwrite",
    )
    assert build.returncode == 0, build.stderr

    validate = run_cli(ROOT, "validate", "--path", str(artifact))
    inspect = run_cli(ROOT, "inspect", "--path", str(artifact))

    assert validate.returncode == 0, validate.stderr
    assert "status=pass" in validate.stdout
    assert "cover_page_status=pass" in validate.stdout
    assert inspect.returncode == 0, inspect.stderr
    assert "artifact_type=teacher_textbook" in inspect.stdout
    assert "tome_artifact_kind=radjax_tome" in inspect.stdout
    assert "cover_page_version=2" in inspect.stdout
    assert "tome_version=1" in inspect.stdout
    assert "layout=unpacked_directory" in inspect.stdout
    assert "target_type=dense_logits" in inspect.stdout
    assert "num_examples=2" in inspect.stdout
    assert "shard_count=1" in inspect.stdout


def test_cover_page_module_does_not_import_heavy_optional_dependencies() -> None:
    script = (
        "import sys; "
        "import radjax_tome.tome.cover_page; "
        "bad=[name for name in ('torch','transformers','jax') if name in sys.modules]; "
        "raise SystemExit(1 if bad else 0)"
    )

    result = run_repo_python(ROOT, "-c", script)

    assert result.returncode == 0, result.stderr


def test_production_cover_page_indexes_surfaces_and_packed_files(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    _add_behavioral_surface_files(artifact)

    write_cover_page(artifact)
    cover = json.loads((artifact / "cover_page.json").read_text(encoding="utf-8"))
    roles = {entry["role"] for entry in cover["contents"]}

    assert {
        "corridor_mode_table",
        "corridor_assignment_manifest",
        "corridor_assignment_position_example_index",
        "corridor_assignment_position",
        "corridor_assignment_mode_id",
        "corridor_assignment_weight",
        "corridor_assignment_examples_metadata",
        "selected_exemplar_index",
        "selected_exemplar_payload_shard",
    } <= roles
    assert [item["surface_id"] for item in cover["behavioral_surfaces"]] == [
        "corridor",
        "exemplar",
    ]
    assert [
        item["surface_id"] for item in cover["recommended_training_plan"]["passes"]
    ] == ["corridor", "exemplar"]
    assert all(
        item["checkpoint_after"]
        for item in cover["recommended_training_plan"]["passes"]
    )
    assert validate_tome_cover_page(artifact).status == "pass"


def test_cover_page_rejects_missing_surface_role_and_unknown_pass_surface(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    _add_behavioral_surface_files(artifact)
    write_cover_page(artifact)
    cover_path = artifact / "cover_page.json"
    cover = json.loads(cover_path.read_text(encoding="utf-8"))
    cover["contents"] = [
        item
        for item in cover["contents"]
        if item["role"] != "corridor_assignment_mode_id"
    ]
    cover["recommended_training_plan"]["passes"][0]["surface_id"] = "missing"
    write_json(cover_path, cover)

    report = validate_tome_cover_page(artifact)

    assert report.status == "fail"
    missing_role = (
        "surface corridor missing required content role: corridor_assignment_mode_id"
    )
    assert missing_role in report.blockers
    assert "recommended pass references unknown surface: missing" in report.blockers


def test_cover_page_rejects_unsafe_and_duplicate_paths(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    cover_path = artifact / "cover_page.json"
    cover = json.loads(cover_path.read_text(encoding="utf-8"))
    cover["contents"][0]["path"] = "../metadata.json"
    cover["contents"].append(dict(cover["contents"][1]))
    write_json(cover_path, cover)

    report = validate_tome_cover_page(artifact)

    assert report.status == "fail"
    assert any("path escapes artifact root" in item for item in report.blockers)
    assert any("duplicate path" in item for item in report.blockers)


def _add_behavioral_surface_files(artifact: Path) -> None:
    corridors = artifact / "corridors"
    assignments = corridors / "mode_assignments"
    selected = artifact / "selected_exemplars"
    leaderboards = artifact / "leaderboards"
    assignments.mkdir(parents=True)
    selected.mkdir()
    leaderboards.mkdir()
    write_json(
        corridors / "corridor_summary.json",
        {
            "corridor_mode_policy": "stat_bands_v0",
            "corridor_assignment_storage_kind": "packed_numpy_v1",
        },
    )
    write_json(corridors / "corridor_modes.json", {"modes": []})
    write_json(corridors / "corridor_fingerprints.json", {"fingerprints": []})
    (corridors / "corridor_summary.txt").write_text("summary\n", encoding="utf-8")
    arrays = {
        "position_example_index": np.asarray([0], dtype=np.int32),
        "position": np.asarray([0], dtype=np.int32),
        "mode_id": np.asarray([0], dtype=np.int32),
        "weight": np.asarray([1.0], dtype=np.float32),
        "fingerprint_index": np.asarray([0], dtype=np.int32),
    }
    for name, values in arrays.items():
        np.save(assignments / f"{name}.npy", values)
    (assignments / "examples_metadata.jsonl").write_text(
        '{"example_id": "example-0", "example_index": 0}\n',
        encoding="utf-8",
    )
    write_json(
        corridors / "mode_assignments.json",
        {
            "arrays": {
                name: {
                    "path": f"corridors/mode_assignments/{name}.npy",
                    "dtype": str(values.dtype),
                    "shape": list(values.shape),
                }
                for name, values in arrays.items()
            },
            "examples_metadata": {
                "path": "corridors/mode_assignments/examples_metadata.jsonl"
            },
        },
    )
    write_json(
        leaderboards / "selected_exemplars.json",
        {"selected_exemplars": []},
    )
    write_json(leaderboards / "leaderboard_report.json", {"status": "pass"})
    write_json(
        selected / "selected-exemplars-00000.json",
        {"selected_exemplars": []},
    )
    write_json(artifact / "delivery_report.json", {"status": "pass"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
