from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from radjax_tome.tome import pack_tome_bundle
from tests.helpers.fixtures import build_fake_teacher_textbook_artifact

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "radjax_tome" / "v1"
VALIDATOR = ROOT / "tools" / "validate_radjax_tome_contract.py"


def _run(path: Path, *flags: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), *flags, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return {"returncode": result.returncode, **json.loads(result.stdout)}


def test_m6c_digest_vector_is_independently_reproducible() -> None:
    vector = json.loads(
        (CONTRACT / "vectors" / "canonical_json_digest_v1.json").read_text()
    )
    encoded = json.dumps(vector["payload"], sort_keys=True, separators=(",", ":"))
    assert encoded == vector["canonical_json_utf8"]
    assert "sha256:" + hashlib.sha256(encoded.encode()).hexdigest() == vector["sha256"]


def test_m6c_valid_directory_and_transports_pass_portable_validator(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    rtome = pack_tome_bundle(artifact, tmp_path / "artifact.rtome")
    tgz = pack_tome_bundle(artifact, tmp_path / "artifact.tgz", compression="gz")
    for path in (artifact, rtome, tgz):
        report = _run(path)
        assert report["returncode"] == 0, report
        assert report["ok"] is True
        assert report["warnings"] == []


def test_m6c_safe_noncanonical_transport_warns_or_strictly_rejects(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    archive = tmp_path / "noncanonical.rtome"
    with tarfile.open(archive, "w") as output:
        for source in sorted(path for path in artifact.rglob("*") if path.is_file()):
            relative = source.relative_to(artifact).as_posix()
            info = tarfile.TarInfo(relative)
            data = source.read_bytes()
            info.size = len(data)
            info.mtime = 99
            output.addfile(info, io.BytesIO(data))
    default = _run(archive)
    strict = _run(archive, "--strict-canonicality")
    assert default["returncode"] == 0
    assert default["warnings"] == ["transport_noncanonical"]
    assert strict["returncode"] == 1
    assert strict["errors"] == ["transport_noncanonical"]


def test_m6c_stale_digest_and_unsafe_archive_fail_closed(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    cover = json.loads((artifact / "cover_page.json").read_text())
    cover["identity"]["training_contract"]["vocab_size"] = -1
    (artifact / "cover_page.json").write_text(json.dumps(cover), encoding="utf-8")
    assert _run(artifact)["errors"] == ["digest_mismatch"]

    unsafe = tmp_path / "unsafe.rtome"
    with tarfile.open(unsafe, "w") as output:
        payload = json.dumps(cover).encode()
        info = tarfile.TarInfo("../cover_page.json")
        info.size = len(payload)
        output.addfile(info, io.BytesIO(payload))
    assert _run(unsafe)["errors"] == ["path_unsafe"]


def test_m6c_catalog_states_historical_nonpromotion() -> None:
    catalog = json.loads((CONTRACT / "fixtures" / "catalog.json").read_text())
    historical = next(
        case for case in catalog["cases"] if case["id"] == "historical-v2"
    )
    assert historical["expected"] == "incomplete_descriptor"
