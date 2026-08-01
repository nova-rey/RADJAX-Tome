from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from radjax_contract.tome import validate_and_resolve_student_consumption

from radjax_tome.tome import (
    pack_tome_bundle,
    validate_tome_bundle,
    validate_tome_cover_page,
    validate_tome_package,
)
from radjax_tome.tome.packaging import package_tome_artifact
from tests.helpers.fixtures import build_fake_teacher_textbook_artifact
from tests.test_tome_packaging_profiles import _artifact as build_package_artifact

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_radjax_tome_contract.py"


def _portable(path: Path, *flags: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), *flags, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return {"returncode": result.returncode, **json.loads(result.stdout)}


def test_m6d_native_and_portable_accept_canonical_directory_and_transports(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    rtome = pack_tome_bundle(artifact, tmp_path / "artifact.rtome")
    tgz = pack_tome_bundle(artifact, tmp_path / "artifact.tgz", compression="gz")

    assert validate_tome_cover_page(artifact).ok
    assert validate_tome_bundle(rtome).ok
    assert validate_tome_bundle(tgz).ok
    for candidate in (artifact, rtome, tgz):
        report = _portable(candidate)
        assert report == {"returncode": 0, "errors": [], "ok": True, "warnings": []}


def test_m6d_profiles_preserve_identity_and_validate_in_both_paths(
    tmp_path: Path,
) -> None:
    artifact = build_package_artifact(tmp_path / "source")
    student = package_tome_artifact(
        artifact, tmp_path / "student", profile="student"
    ).output_path
    debug = package_tome_artifact(
        artifact, tmp_path / "debug", profile="full_debug_provenance"
    ).output_path
    student_cover = json.loads((student / "cover_page.json").read_text())
    debug_cover = json.loads((debug / "cover_page.json").read_text())

    assert validate_tome_package(student, profile="student").ok
    assert validate_tome_package(debug, profile="full_debug_provenance").ok
    assert (
        student_cover["identity"]["semantic_digest"]
        == debug_cover["identity"]["semantic_digest"]
    )
    assert (
        student_cover["manifests"]["content"]["manifest_digest"]
        != debug_cover["manifests"]["content"]["manifest_digest"]
    )
    # The student package now carries the closed v3 consumption extension;
    # its portable authority is the published Contract v3 resolver, not the
    # deliberately base-v3-only M6 standalone checker.
    assert validate_and_resolve_student_consumption(
        student,
        profile_id="native_v3_student_v3",
        strict=True,
    ).ok
    assert _portable(debug)["ok"] is True


def test_m6d_safe_noncanonical_transport_is_documented_intentional_difference(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    archive = tmp_path / "noncanonical.rtome"
    with tarfile.open(archive, "w") as output:
        for source in sorted(path for path in artifact.rglob("*") if path.is_file()):
            data = source.read_bytes()
            info = tarfile.TarInfo(source.relative_to(artifact).as_posix())
            info.size = len(data)
            info.mtime = 1
            output.addfile(info, io.BytesIO(data))

    native = validate_tome_bundle(archive)
    portable = _portable(archive)
    assert not native.ok
    assert portable == {
        "returncode": 0,
        "errors": [],
        "ok": True,
        "warnings": ["transport_noncanonical"],
    }
