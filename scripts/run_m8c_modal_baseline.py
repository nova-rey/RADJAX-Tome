#!/usr/bin/env python3
"""Maintained Modal wrapper for the private M8C baseline driver.

The source, Contract, checkpoint, and model roots are supplied explicitly by
environment variables so no provider-local path or credential is part of the
repository.  This wrapper only runs the existing measurement driver; it does
not change Tome production behavior.
"""

from __future__ import annotations

import json
import gzip
import base64
import os
import shutil
import subprocess
from pathlib import Path

import modal


def _local_input(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        # The module is imported again inside the remote container. Local
        # mount roots are intentionally unavailable there; Modal has already
        # serialized the image mounts from the local entrypoint.
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


TOME_ROOT = _local_input("M8C_TOME_ROOT")
CONTRACT_ROOT = _local_input("M8C_CONTRACT_ROOT")
CHECKPOINT_ROOT = _local_input("M8C_CHECKPOINT_ROOT")
MODEL_ROOT = _local_input("M8C_MODEL_ROOT")
EXPECTED_TOME_COMMIT = os.environ.get("M8C_EXPECTED_TOME_COMMIT", "")

app = modal.App("radjax-m8c-selected-staging-baseline")
evidence_volume = modal.Volume.from_name("radjax-m8c-evidence", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy>=1.26",
    "PyYAML>=6.0",
    "torch",
    "transformers",
    "safetensors",
    "jsonschema",
)
if all(
    root is not None for root in (TOME_ROOT, CONTRACT_ROOT, CHECKPOINT_ROOT, MODEL_ROOT)
):
    image = (
        image.add_local_dir(TOME_ROOT / "src", "/workspace/tome/src")
        .add_local_dir(TOME_ROOT / "scripts", "/workspace/tome/scripts")
        .add_local_dir(CONTRACT_ROOT / "src", "/workspace/contract/src")
        .add_local_dir(CHECKPOINT_ROOT, "/inputs/checkpoint")
        .add_local_dir(MODEL_ROOT, "/inputs/model")
    )


@app.function(
    image=image, gpu="T4", timeout=3600, volumes={"/mnt/evidence": evidence_volume}
)
def run_baseline(expected_commit: str) -> str:
    marker = Path("/mnt/evidence/m8c_runner_started.json")
    marker.write_text(
        json.dumps({"expected_commit": expected_commit, "phase": "started"}),
        encoding="utf-8",
    )
    evidence_volume.commit()
    evidence = Path("/tmp/m8c-evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    checkpoint = Path("/inputs/checkpoint")
    command = [
        "python",
        "/workspace/tome/scripts/run_m8b_selected_staging_baseline.py",
        "--anchor",
        str(checkpoint),
        "--checkpoint-root",
        "/tmp/m8c-checkpoint",
        "--evidence-dir",
        str(evidence),
        "--teacher-model",
        "/inputs/model",
        "--tokenizer-id",
        "/inputs/model",
        "--dataset",
        str(checkpoint / "input/corpus.jsonl"),
        "--corpus-manifest",
        str(checkpoint / "input/corpus_manifest.json"),
        "--teacher-model-provenance",
        str(checkpoint / "input/teacher_model_provenance.json"),
        "--expected-tome-commit",
        expected_commit,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        "/workspace/tome/src:/workspace/tome/scripts:/workspace/contract/src"
    )
    # The source mount intentionally omits .git. The driver only asks for
    # HEAD, so bind that one expected identity rather than mounting history.
    git_shim = Path("/tmp/git")
    git_shim.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = rev-parse ] && [ "$2" = HEAD ]; then '
        f"echo {expected_commit}; exit 0; fi\n"
        "exit 127\n"
    )
    git_shim.chmod(0o755)
    env["PATH"] = "/tmp:" + env.get("PATH", "")
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=3500
    )
    if result.returncode:
        marker.write_text(
            json.dumps(
                {
                    "expected_commit": expected_commit,
                    "phase": "driver_failed",
                    "returncode": result.returncode,
                    "stderr_tail": result.stderr[-2000:],
                }
            ),
            encoding="utf-8",
        )
        evidence_volume.commit()
        raise RuntimeError(result.stderr or result.stdout)
    report = evidence / "m8b_selected_staging_baseline.json"
    # Use the established M8B evidence filename so the volume's existing
    # retrieval path remains authoritative and no second mutable report name
    # can be mistaken for a separate baseline.
    destination = "/mnt/evidence/m8b_selected_staging_baseline_current.json"
    shutil.copy2(report, destination)
    shutil.copy2(report, "/mnt/evidence/m8c_selected_staging_baseline_current.json")
    marker.write_text(
        json.dumps(
            {
                "expected_commit": expected_commit,
                "phase": "completed",
                "report_bytes": report.stat().st_size,
                "operation_counts": "operation_counts" in json.loads(
                    report.read_text(encoding="utf-8")
                ),
            }
        ),
        encoding="utf-8",
    )
    evidence_volume.commit()
    payload = json.loads(report.read_text(encoding="utf-8"))
    summary = json.dumps(
        {
            "schema_version": payload.get("schema_version"),
            "checkpoint": payload.get("checkpoint"),
            "initial_staging_seconds": payload.get("initial_staging_seconds"),
            "selected_pass_wall_seconds": payload.get("selected_pass_wall_seconds"),
        },
        sort_keys=True,
    )
    # Return a compressed copy through the authenticated Modal result as a
    # second transfer path.  The volume remains the durable remote evidence
    # path; this avoids losing a completed report if a concurrent volume
    # snapshot races the final commit.
    encoded_report = base64.b64encode(gzip.compress(report.read_bytes())).decode(
        "ascii"
    )
    return json.dumps({"summary": json.loads(summary), "report_gzip_b64": encoded_report})


@app.local_entrypoint()
def main() -> None:
    if not EXPECTED_TOME_COMMIT:
        raise RuntimeError("M8C_EXPECTED_TOME_COMMIT is required")
    result = json.loads(run_baseline.remote(EXPECTED_TOME_COMMIT))
    out = Path(os.environ.get("M8C_LOCAL_REPORT", "/tmp/m8c_modal_report.json"))
    out.write_bytes(gzip.decompress(base64.b64decode(result["report_gzip_b64"])))
    print(json.dumps(result["summary"], sort_keys=True))
    print(f"wrote_report={out}")
