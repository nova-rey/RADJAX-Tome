from __future__ import annotations

import os
from pathlib import Path

import pytest

from radjax_tome.parity.runner import run_ab_parity


def test_ab_parity_integration_with_archived_repo_when_configured(
    tmp_path: Path,
) -> None:
    old_repo = os.environ.get("QRWKV_XLA_OLD_REPO")
    if not old_repo:
        pytest.skip("QRWKV_XLA_OLD_REPO not set")

    summary = run_ab_parity(
        old_repo=Path(old_repo),
        new_repo=Path(__file__).resolve().parents[1],
        work_dir=tmp_path / "ab",
        overwrite=True,
    )

    assert summary.status == "pass"
    assert (tmp_path / "ab" / "ab_summary.json").is_file()
