#!/usr/bin/env python3
"""Maintained Modal wrapper for the private M8C baseline driver.

The source, Contract, checkpoint, and model roots are supplied explicitly by
environment variables so no provider-local path or credential is part of the
repository.  This wrapper only runs the existing measurement driver; it does
not change Tome production behavior.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


def _required(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must identify an explicit local input root")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


TOME_ROOT = _required("M8C_TOME_ROOT")
CONTRACT_ROOT = _required("M8C_CONTRACT_ROOT")
CHECKPOINT_ROOT = _required("M8C_CHECKPOINT_ROOT")
MODEL_ROOT = _required("M8C_MODEL_ROOT")
EXPECTED_TOME_COMMIT = os.environ.get("M8C_EXPECTED_TOME_COMMIT", "")
if not EXPECTED_TOME_COMMIT:
    raise RuntimeError("M8C_EXPECTED_TOME_COMMIT is required")

app = modal.App("radjax-m8c-selected-staging-baseline")
evidence_volume = modal.Volume.from_name("radjax-m8c-evidence", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26",
        "PyYAML>=6.0",
        "torch",
        "transformers",
        "safetensors",
        "jsonschema",
    )
    .add_local_dir(TOME_ROOT / "src", "/workspace/tome/src")
    .add_local_dir(TOME_ROOT / "scripts", "/workspace/tome/scripts")
    .add_local_dir(CONTRACT_ROOT / "src", "/workspace/contract/src")
    .add_local_dir(CHECKPOINT_ROOT, "/inputs/checkpoint")
    .add_local_dir(MODEL_ROOT, "/inputs/model")
)


@app.function(
    image=image, gpu="T4", timeout=3600, volumes={"/mnt/evidence": evidence_volume}
)
def run_baseline() -> str:
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
        EXPECTED_TOME_COMMIT,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        "/workspace/tome/src:/workspace/tome/scripts:/workspace/contract/src"
    )
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=3500
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    report = evidence / "m8b_selected_staging_baseline.json"
    destination = "/mnt/evidence/m8c_selected_staging_baseline.json"
    shutil.copy2(report, destination)
    evidence_volume.commit()
    payload = json.loads(report.read_text(encoding="utf-8"))
    return json.dumps(
        {
            "schema_version": payload.get("schema_version"),
            "checkpoint": payload.get("checkpoint"),
            "initial_staging_seconds": payload.get("initial_staging_seconds"),
            "selected_pass_wall_seconds": payload.get("selected_pass_wall_seconds"),
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def main() -> None:
    print(run_baseline.remote())
