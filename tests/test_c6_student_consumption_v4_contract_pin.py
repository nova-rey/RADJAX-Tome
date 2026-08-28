from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path

from radjax_contract.tome import validate_and_resolve_student_consumption

from radjax_tome.tome import package_tome_artifact
from tests.test_tome_packaging_profiles import _artifact

ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "contracts" / "radjax_tome" / "student_consumption" / "v4"
PIN_DOCUMENT = ROOT / "docs" / "C3_STUDENT_CONSUMPTION_CONTRACT_PIN.md"
PINNED_COMMIT = "b1209f21fef9405776a757f1a5749d3152bbc3c6"
CURRENT_CONTRACT_COMMIT = "373e3d17060d4ce1c4a0db6065c9289da714bde7"


def _entries(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_c6_v4_mirror_is_checksum_pinned_to_contract_v060() -> None:
    assert "v0.6.0" in PIN_DOCUMENT.read_text(encoding="utf-8")
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


def test_c6_v4_mirror_remains_historical_while_runtime_pin_advances() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "v0.6.0" in PIN_DOCUMENT.read_text(encoding="utf-8")
    assert PINNED_COMMIT in PIN_DOCUMENT.read_text(encoding="utf-8")
    assert any(
        CURRENT_CONTRACT_COMMIT in item for item in project["project"]["dependencies"]
    )


def test_c6_v4_remains_an_explicit_historical_contract_validation_path(
    tmp_path: Path,
) -> None:
    source = _artifact(tmp_path / "source")
    package = package_tome_artifact(
        source,
        tmp_path / "student-v4",
        profile="student",
        student_contract_profile="v4",
    ).output_path

    result = validate_and_resolve_student_consumption(
        package,
        profile_id="native_v3_student_v4",
        strict=True,
    )

    assert result.ok, result.issues


def test_c6_v4_contract_source_assets_match_offline_mirror_when_available() -> None:
    configured = os.environ.get("RADJAX_CONTRACT_STUDENT_CONSUMPTION_V4_ROOT")
    if configured is not None:
        assert _entries(Path(configured)) == _entries(MIRROR)
