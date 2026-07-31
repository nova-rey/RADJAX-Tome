from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "contracts" / "radjax_tome" / "v2"
PIN_DOCUMENT = ROOT / "docs" / "M7_CONTRACT_PUBLICATION_PIN.md"


def _entries(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_m7e_v2_mirror_is_checksum_pinned_to_contract_release() -> None:
    text = PIN_DOCUMENT.read_text(encoding="utf-8")
    assert "v0.3.1" in text
    assert "f8ca8c0" in text
    expected = {}
    for line in (MIRROR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        expected[relative] = digest
    observed = _entries(MIRROR)
    observed.pop("SHA256SUMS")
    assert observed == expected


def test_m7e_contract_source_assets_match_offline_mirror_when_available() -> None:
    configured = os.environ.get("RADJAX_CONTRACT_STREAMING_ROOT")
    if configured is None:
        return
    source_root = Path(configured)
    assert source_root.is_dir()
    assert _entries(source_root) == _entries(MIRROR)
