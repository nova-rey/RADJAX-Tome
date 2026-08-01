from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "contracts" / "radjax_tome" / "student_consumption" / "v2"


def _entries(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _checksum_inventory(root: Path) -> dict[str, str]:
    return {
        relative: digest
        for digest, relative in (
            line.split("  ", maxsplit=1)
            for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        )
    }


def test_c3_v2_mirror_remains_checksum_closed_for_historical_validation() -> None:
    observed = _entries(MIRROR)
    observed.pop("SHA256SUMS")
    assert observed == _checksum_inventory(MIRROR)


def test_c3_v2_contract_source_assets_match_offline_mirror_when_available() -> None:
    configured = os.environ.get("RADJAX_CONTRACT_STUDENT_CONSUMPTION_V2_ROOT")
    if configured is None:
        return
    source_root = Path(configured)
    assert source_root.is_dir()
    assert _entries(source_root) == _entries(MIRROR)
