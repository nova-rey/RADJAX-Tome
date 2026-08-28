"""Public result, stream, and exit-code contract checks."""

from __future__ import annotations

import json
from pathlib import Path

from radjax_tome.cli.mainline import parser, run


def test_machine_result_has_one_stable_top_level_document() -> None:
    result = run(parser().parse_args(["doctor"]))
    document = result.to_dict()
    assert document["schema_version"] == "radjax_tome_cli_result_v1"
    assert document["versions"]["radjax_contract_commit"].startswith("373e3d1")
    assert json.loads(json.dumps(document, sort_keys=True)) == document


def test_external_attestation_requires_attestation_and_evaluation_time(
    tmp_path: Path,
) -> None:
    result = run(
        parser().parse_args(
            [
                "validate",
                str(tmp_path / "missing.tgz"),
                "--mode",
                "external-attestation",
            ]
        )
    )
    assert result.exit_code == 4
    assert result.error is not None
    assert (
        "evaluation-time" in result.error.message
        or "attestation" in result.error.message
    )
