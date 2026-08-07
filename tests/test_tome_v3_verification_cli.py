from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from radjax_contract.tome.v3.codec import digest
from radjax_contract.tome.v3.journal import journal_restart_disposition_v3
from radjax_contract.tome.v3.models import JournalStateV3

from radjax_tome.tome.artifact_v3 import (
    FinalizedV3Handoff,
    publish_v3_from_handoff,
)


def _artifact(tmp_path: Path) -> Path:
    vectors = json.loads(
        Path(
            "contracts/radjax_tome/v3/vectors/tome_provenance_v3_vectors.json"
        ).read_text()
    )
    vector = vectors["normative_root_vectors"][0]
    records = (
        {
            key: value
            for key, value in vector["ordered_records"][0].items()
            if key != "selection_index"
        },
    )
    return publish_v3_from_handoff(
        FinalizedV3Handoff(
            records,
            vector["semantic_context"]["authority"],
            vector["semantic_context"]["behavioral_policy"],
            (0,),
            1,
        ),
        tmp_path / "cli",
    ).directory


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{Path('src').resolve()}:/Users/Cooper/code/RADJAX-Contract/src"
    )
    return subprocess.run(
        [sys.executable, "-m", "radjax_tome.cli.main", *args],
        text=True,
        capture_output=True,
        env=env,
    )


def test_verify_artifact_cli_standard(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    result = _run("verify-artifact", "--artifact", str(artifact), "--mode", "standard")
    assert result.returncode == 0, result.stderr
    assert '"mode": "standard"' in result.stdout


def test_verify_artifact_cli_requires_external_inputs(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    governed = _run(
        "verify-artifact", "--artifact", str(artifact), "--mode", "governed"
    )
    assert governed.returncode == 2
    external = _run(
        "verify-artifact", "--artifact", str(artifact), "--mode", "external-attestation"
    )
    assert external.returncode == 2


def test_private_journal_restart_requires_promoted_marker() -> None:
    state = JournalStateV3(
        "tx",
        "cfg",
        digest(b"authority", "authority"),
        "COMPLETE_INTENT",
        (),
        0,
        True,
        False,
    )
    disposition = journal_restart_disposition_v3(state, public_location_present=False)
    assert disposition.action == "derive_public_evidence"
    assert disposition.public_visible is False
