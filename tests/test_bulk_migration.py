from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from radjax_tome.backends import resolve_qwen_policy
from radjax_tome.corpora import (
    assign_prompt_splits,
    build_prompt_corpus_manifest,
    read_prompt_corpus,
    write_prompt_corpus,
)
from radjax_tome.fingerprint import validate_fingerprint_artifact

ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def test_prompt_corpus_split_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "prompts.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "alpha", "tags": ["smoke"]}),
                json.dumps({"id": "b", "text": "beta", "tags": ["smoke"]}),
                json.dumps({"id": "c", "text": "gamma", "tags": ["holdout"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    corpus = read_prompt_corpus(source, corpus_id="unit")
    split = assign_prompt_splits(corpus, validation_fraction=0.34, seed=7)
    out = tmp_path / "split.jsonl"
    write_prompt_corpus(split, out)
    reloaded = read_prompt_corpus(out, corpus_id="unit")
    manifest = build_prompt_corpus_manifest(reloaded)

    assert manifest.prompt_count == 3
    assert manifest.splits["validation"] == 1
    assert manifest.splits["train"] == 2
    assert manifest.sha256 == build_prompt_corpus_manifest(reloaded).sha256

    inspect_result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_prompt_corpus.py",
            str(out),
            "--json",
        ],
        check=True,
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
    )
    assert json.loads(inspect_result.stdout)["prompt_count"] == 3


def test_qwen_policy_resolution_cli(tmp_path: Path) -> None:
    policy = tmp_path / "qwen_policy.yaml"
    policy.write_text(
        """
schema_version: "0.1"
policies:
  local.smoke:
    description: Local smoke model
    resolved_model_id: local/qwen
    tokenizer_id: local/tokenizer
    trust_remote_code: false
    dtype: fp32
    device: cpu
    requires_manual_resolution: false
""".lstrip(),
        encoding="utf-8",
    )

    resolution = resolve_qwen_policy("local.smoke", policy_path=policy)
    assert resolution.resolved_model_id == "local/qwen"
    assert resolution.tokenizer_id == "local/tokenizer"
    assert resolution.is_resolved

    cli = subprocess.run(
        [
            sys.executable,
            "scripts/resolve_qwen_policy.py",
            "local.smoke",
            "--policy",
            str(policy),
            "--json",
        ],
        check=True,
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
    )
    assert json.loads(cli.stdout)["resolved_model_id"] == "local/qwen"


def test_fingerprint_artifact_validation_and_inspection(tmp_path: Path) -> None:
    artifact = tmp_path / "fingerprint"
    artifact.mkdir()
    (artifact / "modes.json").write_text(
        json.dumps({"modes": [{"mode_id": 0, "top1_margin": 0.2}]}) + "\n",
        encoding="utf-8",
    )
    (artifact / "targets-00000.jsonl").write_text(
        json.dumps({"example_id": "a", "mode_id": 0}) + "\n",
        encoding="utf-8",
    )
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "behavioral_fingerprint",
                "artifact_version": "0.1",
                "created_by": "test",
                "teacher": {
                    "model_name": "teacher",
                    "tokenizer_name": "tok",
                    "vocab_size": 8,
                },
                "sequence": {"max_seq_len": 4, "target_positions": 1},
                "stats": {"tracked": ["top1_margin"]},
                "modes_file": "modes.json",
                "target_shards": [{"path": "targets-00000.jsonl", "num_records": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_fingerprint_artifact(artifact)
    assert result.ok
    assert result.metadata["records"] == 1

    cli = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_fingerprint_artifact.py",
            str(artifact),
            "--json",
        ],
        check=True,
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
    )
    assert json.loads(cli.stdout)["num_corridor_records"] == 1
