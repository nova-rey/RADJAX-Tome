from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "contracts" / "radjax_tome" / "student_consumption" / "v3"
PIN_DOCUMENT = ROOT / "docs" / "C3_STUDENT_CONSUMPTION_CONTRACT_PIN.md"
PINNED_COMMIT = "f9c9278b6a467a6ba7a3972e1644bfc3d13abd6b"


def _entries(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_c5_v3_mirror_is_checksum_pinned_to_contract_v050() -> None:
    assert "v0.5.1" in PIN_DOCUMENT.read_text(encoding="utf-8")
    assert PINNED_COMMIT in PIN_DOCUMENT.read_text(encoding="utf-8")
    sums = {
        relative: digest
        for digest, relative in (
            line.split("  ", maxsplit=1)
            for line in (MIRROR / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        )
    }
    observed = _entries(MIRROR)
    observed.pop("SHA256SUMS")
    assert observed == sums


def test_c5_pyproject_dependency_is_pinned_to_v050_contract_commit() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(PINNED_COMMIT in item for item in project["project"]["dependencies"])


def test_c5_v3_contract_source_assets_match_offline_mirror_when_available() -> None:
    configured = os.environ.get("RADJAX_CONTRACT_STUDENT_CONSUMPTION_V3_ROOT")
    if configured is not None:
        assert _entries(Path(configured)) == _entries(MIRROR)
