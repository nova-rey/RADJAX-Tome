#!/usr/bin/env python3
"""Validate a durable M8G workload manifest without resolving private paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


SCHEMA = "m8g_benchmark_workload_bundle_v1"
REQUIRED = {
    "schema_version",
    "provenance",
    "normalized_inputs",
    "selected_sources",
    "selected_coordinates",
    "authorities",
    "expected_counts",
    "replay_policy",
}


def validate(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if set(document) != REQUIRED:
        raise ValueError("bundle fields are not closed")
    if document["schema_version"] != SCHEMA:
        raise ValueError("unsupported workload bundle schema")
    provenance = document["provenance"]
    if not isinstance(provenance, dict) or provenance.get("status") not in {
        "HISTORICAL_M8_WORKLOAD_RECOVERED",
        "NEW_PAIRED_M8G_WORKLOAD_ESTABLISHED",
        "BLOCKED_NO_REPRODUCIBLE_WORKLOAD",
    }:
        raise ValueError("invalid workload provenance status")
    sources = document["selected_sources"]
    coordinates = document["selected_coordinates"]
    if not isinstance(sources, dict) or not isinstance(coordinates, dict):
        raise ValueError("selected identity manifests must be objects")
    source_count = int(sources["count"])
    coordinate_count = int(coordinates["count"])
    expected = document["expected_counts"]
    if not isinstance(expected, dict):
        raise ValueError("expected_counts must be an object")
    if expected.get("selected_coordinates") != coordinate_count:
        raise ValueError("selected coordinate count mismatch")
    if expected.get("selected_sources") != source_count:
        raise ValueError("selected source count mismatch")
    policy = document["replay_policy"]
    if not isinstance(policy, dict) or policy != {
        "requested_batch_size": 8,
        "batching": "fixed_source_count",
        "oom_fallback": "halving_only",
        "length_bucketing": False,
    }:
        raise ValueError("replay policy is not the frozen M8 policy")
    # The manifest is portable; byte verification is enabled by passing the
    # separately governed artifact root to the CLI.
    artifact_root = getattr(validate, "artifact_root", None)
    if artifact_root is not None:
        layout = provenance.get("artifact_layout")
        if not isinstance(layout, dict):
            raise ValueError("artifact layout is missing")

        def contained(relative: object) -> Path:
            candidate = (artifact_root / str(relative)).resolve()
            try:
                candidate.relative_to(artifact_root.resolve())
            except ValueError as exc:
                raise ValueError("artifact layout escapes bundle root") from exc
            return candidate

        replay_root = contained(layout["replay_root"])
        checkpoint_manifest = artifact_root / "checkpoint-manifest.json"
        if (
            _sha256(checkpoint_manifest)
            != document["normalized_inputs"]["checkpoint_manifest_sha256"]
        ):
            raise ValueError("checkpoint manifest digest mismatch")
        checkpoint_doc = json.loads(checkpoint_manifest.read_text())
        from radjax_tome.builder.delivery.replay import _checkpoint_digest

        checkpoint_identity = dict(checkpoint_doc)
        declared_checkpoint_digest = checkpoint_identity.pop("checkpoint_digest", None)
        if _checkpoint_digest(checkpoint_identity) != declared_checkpoint_digest:
            raise ValueError("checkpoint digest does not recompute")
        if (
            checkpoint_doc.get("checkpoint_digest")
            != document["normalized_inputs"]["checkpoint_digest"]
        ):
            raise ValueError("checkpoint digest mismatch")
        if len(checkpoint_doc.get("file_digests", {})) != int(
            document["normalized_inputs"]["checkpoint_file_count"]
        ):
            raise ValueError("checkpoint file count mismatch")
        actual_checkpoint_files = {
            path.relative_to(replay_root).as_posix()
            for path in replay_root.rglob("*")
            if path.is_file()
        }
        if any(path.is_symlink() for path in replay_root.rglob("*")):
            raise ValueError("checkpoint tree contains symlinked resources")
        if actual_checkpoint_files != set(checkpoint_doc["file_digests"]):
            raise ValueError("checkpoint tree has missing or extra files")
        for relative, expected_digest in checkpoint_doc["file_digests"].items():
            candidate = replay_root / relative
            if (
                not candidate.is_file()
                or _sha256(candidate) != "sha256:" + expected_digest
            ):
                raise ValueError(f"checkpoint member mismatch: {relative}")
        required = {
            "c6/claims/selected_coordinates.jsonl": "sha256:d29e9f5642a808e5a439ec21beadbfc3793c2c2c6f14467fe3c1a8447784c5cb",  # noqa: E501
            "c6/multi-role-selection/selected_exemplars.jsonl": "sha256:255d5507668f8c6eed6f220563cc8bd3641fa134d721f3d9d9047161d899b35c",  # noqa: E501
            "c6/source-passports.jsonl": "sha256:b23bc3d9dbfb9845a77efab3819e99c7d410b8243153d1ba81120adf3987226b",  # noqa: E501
            "c6/authority_manifest.json": "sha256:765f3f997c1c56ff4333f4ed8dd61ec9696dcd6bef27c149c6569858c828027f",  # noqa: E501
        }
        for relative, expected_digest in required.items():
            candidate = replay_root / relative
            if not candidate.is_file() or _sha256(candidate) != expected_digest:
                raise ValueError(f"artifact member digest mismatch: {relative}")
        coordinates = replay_root / "c6/claims/selected_coordinates.jsonl"
        selected = replay_root / "c6/multi-role-selection/selected_exemplars.jsonl"
        passports = replay_root / "c6/source-passports.jsonl"
        coordinate_rows = [
            json.loads(line) for line in coordinates.read_text().splitlines()
        ]
        selected_rows = [json.loads(line) for line in selected.read_text().splitlines()]
        passport_keys = {
            (str(row["example_id"]), int(row["position"]))
            for row in (json.loads(line) for line in passports.read_text().splitlines())
        }
        coordinate_keys = {
            (
                str(row.get("selected_example_id", row.get("example_id"))),
                int(row.get("selected_position", row.get("position"))),
            )
            for row in coordinate_rows
        }
        selected_keys = {
            (
                str(row.get("selected_example_id", row.get("example_id"))),
                int(row.get("selected_position", row.get("position"))),
            )
            for row in selected_rows
        }
        if len(coordinate_keys) != coordinate_count or selected_keys != coordinate_keys:
            raise ValueError("selected coordinate identity mismatch")
        if not coordinate_keys <= passport_keys:
            raise ValueError("selected coordinate lacks source passport")
        if len({key[0] for key in coordinate_keys}) != source_count:
            raise ValueError("selected source count mismatch")
        for row in coordinate_rows:
            if int(row.get("position", row.get("selected_position", -1))) < 0:
                raise ValueError("selected coordinate position is invalid")
        authority = json.loads((replay_root / "c6/authority_manifest.json").read_text())
        if (
            authority.get("score_pass_authority_hash")
            != document["normalized_inputs"]["authority_hash"]
        ):
            raise ValueError("selection authority mismatch")
        corpus_manifest = json.loads(
            (replay_root / "input/corpus_manifest.json").read_text()
        )
        if (
            corpus_manifest.get("corpus_hash")
            != document["normalized_inputs"]["corpus_hash"]
        ):
            raise ValueError("corpus authority mismatch")
        normalized = document["normalized_inputs"]
        if not isinstance(normalized.get("model_file_manifest"), list):
            raise ValueError("complete model-file manifest is missing")
        if (
            int(normalized["sequence_length"]) != 128
            or int(normalized["vocab_size"]) != 262144
        ):
            raise ValueError("normalized sequence/vocabulary authority mismatch")
        if int(corpus_manifest["num_examples"]) != 99989:
            raise ValueError("corpus example-count authority mismatch")
        # Reject duplicate raw rows rather than relying on set cardinality.
        coordinate_list = [
            (str(row.get("example_id")), int(row.get("position")))
            for row in coordinate_rows
        ]
        selected_list = [
            (str(row.get("example_id")), int(row.get("position")))
            for row in selected_rows
        ]
        if len(coordinate_list) != len(set(coordinate_list)):
            raise ValueError("duplicate selected coordinate rows")
        if len(selected_list) != len(set(selected_list)):
            raise ValueError("duplicate selected exemplar rows")
        if any(
            position < 0 or position >= int(normalized["sequence_length"])
            for _, position in coordinate_list
        ):
            raise ValueError("selected coordinate position exceeds sequence length")
        if any(
            position < 0 or position >= int(normalized["sequence_length"])
            for _, position in selected_list
        ):
            raise ValueError("selected exemplar position exceeds sequence length")
        # Bind the canonical delivery-record identity, not merely the raw JSONL
        # file digest.  This is the digest used by the frozen M8 authority.
        try:
            from radjax_tome.builder.c6_integration import c5_records_for_delivery
            from radjax_tome.fingerprint.multi_role_selection import (
                load_multi_role_selection_artifact,
            )

            artifact = load_multi_role_selection_artifact(
                replay_root / "c6/multi-role-selection"
            )
            records = c5_records_for_delivery(
                artifact, delivery_path="two_pass_rerun_selected"
            )
            canonical_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            )
        except Exception as exc:  # pragma: no cover - authority failure is reported
            raise ValueError(
                f"cannot compute canonical selected-record digest: {exc}"
            ) from exc
        if canonical_digest != normalized["selected_record_digest"]:
            raise ValueError("canonical selected-record digest mismatch")
        if (
            _sha256(replay_root / "input/corpus_manifest.json")
            != document["authorities"]["corpus_manifest_sha256"]
        ):
            raise ValueError("corpus manifest identity mismatch")
        passport_by_key = {
            (str(row["example_id"]), int(row["position"])): row
            for row in (json.loads(line) for line in passports.read_text().splitlines())
        }
        selected_keys_raw = set()
        for row in coordinate_rows:
            raw_key = (str(row["example_id"]), int(row["position"]))
            selected_key = (
                str(row.get("selected_example_id", row["example_id"])),
                int(row.get("selected_position", row["position"])),
            )
            if raw_key != selected_key:
                raise ValueError("selected coordinate aliases its source identity")
            selected_keys_raw.add(raw_key)
            passport = passport_by_key.get(raw_key)
            if passport is None or int(passport["source_row"]) < 0:
                raise ValueError("selected coordinate source mapping is invalid")
        corpus_path = replay_root / "input/corpus.jsonl"
        needed = {example_id for example_id, _ in selected_keys_raw}
        found: dict[str, int] = {}
        with corpus_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not needed:
                    break
                row = json.loads(line)
                if row.get("example_id") in needed:
                    found[str(row["example_id"])] = index
                    needed.remove(str(row["example_id"]))
        if needed:
            raise ValueError("selected source is absent from corpus")
        for example_id, _ in selected_keys_raw:
            passport = next(
                value for key, value in passport_by_key.items() if key[0] == example_id
            )
            if int(passport["source_row"]) != found[example_id]:
                raise ValueError("selected source row does not bind to corpus")
        # Verify the model/tokenizer identity files against the retained teacher
        # provenance, and bind the declared architecture dimensions.
        model_root = contained(layout["model_root"])
        provenance_doc = _load_json(replay_root / "input/teacher_model_provenance.json")
        if (
            provenance_doc.get("model_directory_hash")
            != normalized["model_directory_hash"]
        ):
            raise ValueError("teacher model directory identity mismatch")
        if provenance_doc.get("tokenizer_hash") != normalized["tokenizer_hash"]:
            raise ValueError("tokenizer identity mismatch")
        if provenance_doc.get("weights_hash") != normalized["weights_hash"]:
            raise ValueError("weights identity mismatch")
        if provenance_doc.get("config_hash") != normalized["config_hash"]:
            raise ValueError("config identity mismatch")
        expected_files = {
            "model/config.json": "config_sha256",
            "model/model.safetensors": "weights_file_sha256",
            "model/tokenizer.json": "tokenizer_json_sha256",
            "model/tokenizer.model": "tokenizer_model_sha256",
            "verify-checkpoint/input/teacher_model_provenance.json": (
                "teacher_provenance_sha256"
            ),
        }
        for relative, field in expected_files.items():
            digest = _sha256(artifact_root / relative)
            if digest != normalized[field]:
                raise ValueError(f"teacher authority digest mismatch: {relative}")
        model_records = normalized["model_file_manifest"]
        if len({record["relative_path"] for record in model_records}) != len(
            model_records
        ):
            raise ValueError("duplicate model-file manifest entry")
        actual_model_files = {
            path.name for path in model_root.iterdir() if path.is_file()
        }
        if any(path.is_symlink() for path in model_root.iterdir()):
            raise ValueError("model tree contains symlinked resources")
        if actual_model_files != {record["relative_path"] for record in model_records}:
            raise ValueError("model tree has missing or extra files")
        for record in model_records:
            if _sha256(model_root / record["relative_path"]) != record["sha256"]:
                raise ValueError(
                    f"complete model-file manifest mismatch: {record['relative_path']}"
                )
        model_manifest = {
            "config_files": [
                {
                    "relative_path": x["relative_path"],
                    "sha256": x["sha256"],
                    "size_bytes": x["size_bytes"],
                }
                for x in normalized["model_file_manifest"]
                if x["relative_path"] in {"config.json", "generation_config.json"}
            ],
            "tokenizer_files": [
                {
                    "relative_path": x["relative_path"],
                    "sha256": x["sha256"],
                    "size_bytes": x["size_bytes"],
                }
                for x in normalized["model_file_manifest"]
                if x["relative_path"]
                in {
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "tokenizer.model",
                }
            ],
            "weight_files": [
                {
                    "relative_path": x["relative_path"],
                    "sha256": x["sha256"],
                    "size_bytes": x["size_bytes"],
                }
                for x in normalized["model_file_manifest"]
                if x["relative_path"].endswith(".safetensors")
            ],
        }
        from radjax_tome.provenance.teacher_model import _category_hash, _directory_hash

        if _category_hash(model_manifest["config_files"]) != provenance_doc.get(
            "config_hash"
        ):
            raise ValueError("model config identity does not recompute")
        if _category_hash(model_manifest["tokenizer_files"]) != provenance_doc.get(
            "tokenizer_hash"
        ):
            raise ValueError("tokenizer identity does not recompute")
        if _category_hash(model_manifest["weight_files"]) != provenance_doc.get(
            "weights_hash"
        ):
            raise ValueError("weights identity does not recompute")
        if _directory_hash(**model_manifest) != provenance_doc.get(
            "model_directory_hash"
        ):
            raise ValueError("model directory identity does not recompute")
        config_doc = _load_json(model_root / "config.json")
        if int(config_doc.get("vocab_size", -1)) != int(normalized["vocab_size"]):
            raise ValueError("model vocabulary authority mismatch")
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": SCHEMA,
        "status": provenance["status"],
        "selected_sources": source_count,
        "selected_coordinates": coordinate_count,
        "manifest_sha256": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "artifact_verified": artifact_root is not None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args()
    validate.artifact_root = args.bundle_root.resolve() if args.bundle_root else None
    print(json.dumps(validate(args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
