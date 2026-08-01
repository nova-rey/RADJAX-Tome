from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "contracts" / "radjax_tome" / "student_consumption" / "v3"


def _entries(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_c5_v3_mirror_is_checksum_pinned_to_contract_v050() -> None:
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


def test_c5_v3_contract_source_assets_match_offline_mirror_when_available() -> None:
    configured = os.environ.get("RADJAX_CONTRACT_STUDENT_CONSUMPTION_V3_ROOT")
    if configured is not None:
        assert _entries(Path(configured)) == _entries(MIRROR)
