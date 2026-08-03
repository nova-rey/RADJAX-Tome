"""Private M8B.1 raw-evidence driver checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from radjax_tome.builder.delivery.measurement import validate_m8b_statistics_receipt

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "m8b_driver", scripts / "run_m8b_selected_staging_baseline.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _run(staging_seconds: float, wall_seconds: float) -> dict[str, object]:
    phases = {
        name: {"status": "measured_host_wall", "seconds": staging_seconds / 5}
        for name in (
            "canonical_body_encoding_hash",
            "staging_json_encoding",
            "temporary_file_write",
            "temporary_file_close",
            "atomic_replacement",
        )
    }
    return {
        "metrics": {
            "selected_pass_execution_v1": {
                "selected_pass_wall_seconds": wall_seconds,
                "phases": phases,
            }
        }
    }


def test_m8b_driver_freezes_statistics_and_evaluates_initial_staging_gate() -> None:
    driver = _driver()
    assert driver._initial_staging_seconds(_run(8.0, 10.0)) == 8.0
    checkpoint = type(
        "Checkpoint",
        (),
        {
            "digest": "sha256:checkpoint",
            "file_digests": {"one": "digest"},
            "manifest_path": Path(__file__),
        },
    )()
    receipt = driver._receipt(
        checkpoint=checkpoint,
        runs=[_run(8.0, 10.0), _run(7.0, 10.0), _run(9.0, 10.0)],
        expected_tome_commit="f" * 40,
    )
    validate_m8b_statistics_receipt(receipt)
    assert receipt["initial_staging_gate_passes"] is True
