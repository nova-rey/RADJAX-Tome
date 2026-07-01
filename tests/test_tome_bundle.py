from __future__ import annotations

import io
import tarfile
from pathlib import Path

from radjax_tome.tome import (
    inspect_tome_bundle,
    pack_tome_bundle,
    unpack_tome_bundle,
    validate_tome_bundle,
)
from tests.helpers.fixtures import build_fake_teacher_textbook_artifact
from tests.helpers.subprocess import run_cli, run_repo_python

ROOT = Path(__file__).resolve().parents[1]


def test_pack_tome_bundle_creates_rtome_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    bundle = tmp_path / "fake_tome.rtome"

    result = pack_tome_bundle(artifact, bundle)

    assert result == bundle
    assert bundle.is_file()
    try:
        pack_tome_bundle(artifact, bundle)
    except ValueError as exc:
        assert "bundle already exists" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("packing should refuse overwrite by default")


def test_pack_tome_bundle_is_byte_deterministic(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    first = pack_tome_bundle(artifact, tmp_path / "first.rtome")
    second = pack_tome_bundle(artifact, tmp_path / "second.rtome")

    assert first.read_bytes() == second.read_bytes()


def test_bundle_contains_cover_page_and_only_listed_contents(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    (artifact / "junk.txt").write_text("not part of the Tome\n", encoding="utf-8")
    bundle = pack_tome_bundle(artifact, tmp_path / "fake_tome.rtome")

    with tarfile.open(bundle, mode="r:*") as archive:
        names = archive.getnames()

    assert "cover_page.json" in names
    assert "junk.txt" not in names
    assert {
        "metadata.json",
        "vocab_contract.json",
        "teacher_manifest.json",
        "emission_config.json",
        "validation_report.json",
        "shards/shard-00000.npz",
    } <= set(names)


def test_bundle_validation_passes_and_inspects_without_extraction(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    bundle = pack_tome_bundle(artifact, tmp_path / "fake_tome.rtome")

    report = validate_tome_bundle(bundle)
    summary = inspect_tome_bundle(bundle)

    assert report.status == "pass"
    assert report.format_ok
    assert report.cover_page_ok
    assert report.contents_ok
    assert report.deterministic_layout_ok
    assert summary["artifact_kind"] == "radjax_tome"
    assert summary["cover_page_version"] == 1
    assert summary["tome_version"] == 1
    assert summary["layout"] == "unpacked_directory"
    assert summary["target_type"] == "dense_logits"
    assert summary["num_examples"] == 2
    assert summary["shard_count"] == 1
    assert summary["content_count"] == 6
    assert not (tmp_path / "inspection_extract").exists()


def test_bundle_validation_fails_on_tampered_member(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    bundle = tmp_path / "tampered.rtome"

    _write_archive(
        bundle,
        {
            "cover_page.json": (artifact / "cover_page.json").read_bytes(),
            "metadata.json": b"tampered\n",
            "vocab_contract.json": (artifact / "vocab_contract.json").read_bytes(),
            "teacher_manifest.json": (artifact / "teacher_manifest.json").read_bytes(),
            "emission_config.json": (artifact / "emission_config.json").read_bytes(),
            "validation_report.json": (
                artifact / "validation_report.json"
            ).read_bytes(),
            "shards/shard-00000.npz": (
                artifact / "shards" / "shard-00000.npz"
            ).read_bytes(),
        },
    )

    report = validate_tome_bundle(bundle)

    assert report.status == "fail"
    assert any("sha256 mismatch" in blocker for blocker in report.blockers)


def test_bundle_validation_rejects_unsafe_and_duplicate_members(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    unsafe = tmp_path / "unsafe.rtome"
    duplicate = tmp_path / "duplicate.rtome"

    _write_archive(
        unsafe,
        {
            "cover_page.json": (artifact / "cover_page.json").read_bytes(),
            "../evil.txt": b"nope",
        },
    )
    _write_archive(
        duplicate,
        {
            "cover_page.json": (artifact / "cover_page.json").read_bytes(),
            "metadata.json": (artifact / "metadata.json").read_bytes(),
        },
        duplicate_name="metadata.json",
        duplicate_data=b"duplicate",
    )

    unsafe_report = validate_tome_bundle(unsafe)
    duplicate_report = validate_tome_bundle(duplicate)

    assert unsafe_report.status == "fail"
    assert any(
        "unsafe bundle member path" in blocker for blocker in unsafe_report.blockers
    )
    assert duplicate_report.status == "fail"
    assert any(
        "duplicate bundle member" in blocker for blocker in duplicate_report.blockers
    )


def test_unpack_tome_bundle_extracts_safely_and_validates(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    bundle = pack_tome_bundle(artifact, tmp_path / "fake_tome.rtome")
    output = tmp_path / "unpacked"

    result = unpack_tome_bundle(bundle, output)

    assert result == output
    assert (output / "cover_page.json").is_file()
    assert validate_tome_bundle(
        pack_tome_bundle(output, tmp_path / "repacked.rtome")
    ).ok
    try:
        unpack_tome_bundle(bundle, output)
    except ValueError as exc:
        assert "output directory is not empty" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("unpack should refuse non-empty output by default")


def test_public_cli_pack_validate_inspect_and_unpack_bundle(tmp_path: Path) -> None:
    artifact = tmp_path / "cli_tome"
    bundle = tmp_path / "cli_tome.rtome"
    unpacked = tmp_path / "cli_tome_unpacked"

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
    pack = run_cli(
        ROOT,
        "pack",
        "--input",
        str(artifact),
        "--output",
        str(bundle),
        "--overwrite",
    )
    validate = run_cli(ROOT, "validate", "--path", str(bundle))
    inspect = run_cli(ROOT, "inspect", "--path", str(bundle))
    unpack = run_cli(
        ROOT,
        "unpack",
        "--input",
        str(bundle),
        "--output",
        str(unpacked),
        "--overwrite",
    )
    directory_inspect = run_cli(ROOT, "inspect", "--path", str(artifact))

    assert build.returncode == 0, build.stderr
    assert pack.returncode == 0, pack.stderr
    assert "status=pass" in pack.stdout
    assert validate.returncode == 0, validate.stderr
    assert "status=pass" in validate.stdout
    assert "bundle_format_ok=True" in validate.stdout
    assert inspect.returncode == 0, inspect.stderr
    assert "RADJAX-Tome bundle summary" in inspect.stdout
    assert "tome_artifact_kind=radjax_tome" in inspect.stdout
    assert "compression=none" in inspect.stdout
    assert unpack.returncode == 0, unpack.stderr
    assert (unpacked / "cover_page.json").is_file()
    assert directory_inspect.returncode == 0, directory_inspect.stderr
    assert "RADJAX-Tome artifact summary" in directory_inspect.stdout


def test_bundle_module_does_not_import_heavy_optional_dependencies() -> None:
    script = (
        "import sys; "
        "import radjax_tome.tome.bundle; "
        "bad=[name for name in ('torch','transformers','jax') if name in sys.modules]; "
        "raise SystemExit(1 if bad else 0)"
    )

    result = run_repo_python(ROOT, "-c", script)

    assert result.returncode == 0, result.stderr


def _write_archive(
    path: Path,
    members: dict[str, bytes],
    *,
    duplicate_name: str | None = None,
    duplicate_data: bytes = b"",
) -> None:
    with tarfile.open(path, mode="w") as archive:
        for name, data in members.items():
            _add_bytes(archive, name, data)
        if duplicate_name is not None:
            _add_bytes(archive, duplicate_name, duplicate_data)


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))
