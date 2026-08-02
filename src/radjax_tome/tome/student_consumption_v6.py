"""Materialize the explicit v6 behavioral-authority composition.

This adapter deliberately composes the immutable v5 tokenizer/language
binding with existing native-v3 evidence.  It does not change producer
selection, re-run a teacher, or reinterpret corridor diagnostics as policy.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.tome.student_consumption_v2 import (
    NativeV3StudentConsumptionV2Materialization,
    materialize_native_v3_student_consumption_v2,
)

MANIFEST_PATH = "manifests/behavioral_resource_binding_v1.json"
_ROOT = "student_consumption/v6"


@dataclass(frozen=True)
class NativeV3StudentConsumptionV6Materialization:
    """The v6 manifest path and Contract-computed composition identities."""

    manifest_path: str
    behavioral_authority_digest: str
    composition_digest: str


def materialize_native_v3_student_consumption_v6(
    artifact_root: Path,
    *,
    destination_root: Path,
    package_semantic_identity: str,
) -> NativeV3StudentConsumptionV6Materialization:
    """Emit the closed v6 registry using existing producer facts only."""

    root = Path(destination_root)
    v2 = materialize_native_v3_student_consumption_v2(
        artifact_root, destination_root=root
    )
    paths = _write_v6_projections(root, v2, source_root=Path(artifact_root))
    resources = _resources(root, paths)
    from radjax_contract.tome import validate_and_resolve_language_tokenizer_binding
    from radjax_contract.tome.student_consumption_v6 import (
        canonical_behavioral_authority_digest,
        canonical_behavioral_source_identity,
        canonical_composition_digest,
    )

    language = validate_and_resolve_language_tokenizer_binding(root, strict=True)
    if not language.ok or language.descriptor is None:
        raise ValueError("v6 requires an admitted v5 language binding")
    by_role = {item["role"]: item for item in resources}
    authority = _registry(resources, authority=True)
    non_authority = _registry(resources, authority=False)
    target = np.load(root / paths["input_ids"], allow_pickle=False)
    source = canonical_behavioral_source_identity(
        language_binding_digest=language.descriptor.canonical_binding_digest,
        target_semantic_identity=by_role["target_shard"]["semantic_identity"],
        example_registry_semantic_identity=by_role["example_registry"][
            "semantic_identity"
        ],
        target_shape=tuple(target.shape),
        target_axes=("example", "sequence_position"),
    )
    joins = (
        "assignment_to_target_grid",
        "selected_passport_to_target",
        "selected_exemplar_to_passport",
    )
    behavioral = canonical_behavioral_authority_digest(
        language_binding_digest=language.descriptor.canonical_binding_digest,
        behavioral_source_identity=source,
        authority_registry=authority,
        required_joins=joins,
        selection_authority_digest=by_role["authority_reference"]["semantic_identity"],
    )
    composition = canonical_composition_digest(
        behavioral_authority_digest=behavioral,
        authority_registry=authority,
        non_authority_registry=non_authority,
        package_semantic_identity=package_semantic_identity,
    )
    write_json(
        root / MANIFEST_PATH,
        {
            "schema_version": "radjax_behavioral_resource_binding_v1",
            "profile_id": "native_v3_student_v6",
            "resources": resources,
            "package_semantic_identity": package_semantic_identity,
            "behavioral_authority_digest": behavioral,
            "composition_digest": composition,
        },
    )
    return NativeV3StudentConsumptionV6Materialization(
        MANIFEST_PATH, behavioral, composition
    )


def _write_v6_projections(
    root: Path,
    v2: NativeV3StudentConsumptionV2Materialization,
    *,
    source_root: Path,
) -> dict[str, str]:
    """Write only closed logical projections; source arrays stay native NPY."""

    directory = root / _ROOT
    directory.mkdir(parents=True, exist_ok=True)
    with np.load(root / v2.role_paths["target_shard"], allow_pickle=False) as target:
        input_ids = np.asarray(target["input_ids"], dtype=np.int32)
        mask = np.asarray(target["attention_mask"], dtype=np.int8)
    np.save(directory / "input_ids.npy", input_ids)
    np.save(directory / "attention_mask.npy", mask)
    with np.load(
        root / v2.role_paths["corridor_assignment"], allow_pickle=False
    ) as source:
        mode = np.asarray(source["mode_id"], dtype=np.int32)
        weight = np.asarray(source["weight"], dtype=np.float32)
    n, length = input_ids.shape
    existing = {
        (int(example), int(position)): (int(mode_id), float(value))
        for example, position, mode_id, value in zip(
            np.load(root / v2.role_paths["corridor_assignment"], allow_pickle=False)[
                "position_example_index"
            ],
            np.load(root / v2.role_paths["corridor_assignment"], allow_pickle=False)[
                "position"
            ],
            mode,
            weight,
            strict=True,
        )
    }
    example = np.repeat(np.arange(n, dtype=np.int32), length)
    position = np.tile(np.arange(length, dtype=np.int32), n)
    if set(existing) != set(zip(example.tolist(), position.tolist(), strict=True)):
        raise ValueError("native v3 source lacks the required v6 full assignment grid")
    v6_mode = np.asarray(
        [existing[(int(e), int(p))][0] for e, p in zip(example, position, strict=True)],
        dtype=np.int32,
    )
    v6_weight = np.asarray(
        [existing[(int(e), int(p))][1] for e, p in zip(example, position, strict=True)],
        dtype=np.float32,
    )
    for name, values in (
        ("example_index", example),
        ("position", position),
        ("mode_id", v6_mode),
        ("weight", v6_weight),
    ):
        np.save(directory / f"{name}.npy", values)
    registry = read_json_object(root / v2.role_paths["example_registry"])["examples"]
    _write_jsonl(
        directory / "example_registry.jsonl",
        [{"example_id": row["selected_example_id"]} for row in registry],
    )
    _write_jsonl(directory / "selected_passport_index.jsonl", _passports(root, v2))
    m7_archive = source_root.with_name(f"{source_root.name}.v4.tgz")
    if m7_archive.is_file():
        shutil.copyfile(m7_archive, directory / "selected_exemplar_payload.m7.tgz")
        exemplar_name = "selected_exemplar_payload.m7.tgz"
    else:
        _write_jsonl(
            directory / "selected_exemplar_payload.jsonl", _exemplars(root, v2)
        )
        exemplar_name = "selected_exemplar_payload.jsonl"
    write_json(directory / "corridor_mode_table.json", _mode_table(root, v6_mode))
    authority = _authority_reference(root, v2)
    write_json(directory / "authority_reference.json", authority)
    delivery = read_json_object(root / v2.role_paths["delivery_receipt"])
    source_delivery = (
        read_json_object(root / "delivery_report.json")
        if (root / "delivery_report.json").is_file()
        else {}
    )
    write_json(
        directory / "delivery_receipt.json",
        {
            "delivery_path": source_delivery.get(
                "delivery_path",
                delivery.get("delivery_path", "two_pass_rerun_selected"),
            )
        },
    )
    return {
        name: f"{_ROOT}/{name}"
        for name in (
            "input_ids.npy",
            "attention_mask.npy",
            "example_index.npy",
            "position.npy",
            "mode_id.npy",
            "weight.npy",
            "example_registry.jsonl",
            "selected_passport_index.jsonl",
            exemplar_name,
            "corridor_mode_table.json",
            "authority_reference.json",
            "delivery_receipt.json",
        )
    } | {
        "input_ids": f"{_ROOT}/input_ids.npy",
        "attention_mask": f"{_ROOT}/attention_mask.npy",
    }


def _passports(
    root: Path, v2: NativeV3StudentConsumptionV2Materialization
) -> list[dict[str, Any]]:
    rows = read_json_object(root / v2.role_paths["selected_passport_index"])[
        "selected_exemplars"
    ]
    authority = read_json_object(root / v2.role_paths["authority_reference"])
    selection_hash = authority.get("selection_integration_config_hash")
    if not isinstance(selection_hash, str):
        raise ValueError("native v3 selected passport has no selection authority hash")
    return [
        {
            "schema_version": "radjax_selected_passport_v6",
            "selected_example_id": row["selected_example_id"],
            "selected_position": row["selected_position"],
            "rank": rank,
            "selected_score": row["selected_score"],
            "selected_policy": str(row.get("selected_policy") or "native_v3"),
            "corridor_mode_id": int(row["corridor_mode_id"]),
            "corridor_fingerprint_id": str(
                row.get("corridor_fingerprint_id") or "native_v3"
            ),
            "corridor_assignment_status": "selected",
            "selection_integration_config_hash": selection_hash,
        }
        for rank, row in enumerate(rows, start=1)
    ]


def _exemplars(
    root: Path, v2: NativeV3StudentConsumptionV2Materialization
) -> list[dict[str, Any]]:
    return read_json_object(root / v2.role_paths["selected_exemplar_payload"])[
        "selected_exemplars"
    ]


def _mode_table(root: Path, mode_ids: np.ndarray) -> dict[str, Any]:
    source = read_json_object(root / "corridors/corridor_modes.json")["modes"]
    stats = {"entropy", "top1_margin", "top8_mass", "top32_mass", "tail_mass"}
    source_by_id = {int(row["mode_id"]): row for row in source}
    modes = []
    for mode_id in sorted(set(int(value) for value in mode_ids)):
        source_stats = source_by_id[mode_id].get("statistics", {})
        normalized = {}
        for name in stats:
            value = source_stats.get(name, {})
            minimum, mean, maximum = (
                float(value.get(key, 0.0)) for key in ("min", "mean", "max")
            )
            normalized[name] = {"min": minimum, "mean": mean, "max": maximum}
        modes.append({"mode_id": mode_id, "statistics": normalized})
    return {"modes": modes}


def _authority_reference(
    root: Path, v2: NativeV3StudentConsumptionV2Materialization
) -> dict[str, Any]:
    candidates = [
        read_json_object(root / v2.role_paths["authority_reference"]),
        *(
            read_json_object(root / name)
            for name in (
                "production_build_report.json",
                "delivery_report.json",
                "metadata.json",
            )
            if (root / name).is_file()
        ),
    ]
    values: dict[str, Any] = {}
    for candidate in candidates:
        values.update(
            {key: value for key, value in candidate.items() if key not in values}
        )
    if "score_pass_authority_hash" not in values:
        values["score_pass_authority_hash"] = values.get("score_pass_authority_hash_v1")
    required = (
        "selection_integration_config_hash",
        "score_pass_authority_hash",
        "delivery_authority_hash",
    )
    if any(
        not isinstance(values.get(key), str) or not values[key].startswith("sha256:")
        for key in required
    ):
        raise ValueError("native v3 artifact lacks closed v6 authority evidence")
    return {
        "schema_version": "radjax_behavioral_authority_reference_v6",
        **{key: values[key] for key in required},
    }


def _resources(root: Path, paths: dict[str, str]) -> list[dict[str, Any]]:
    from radjax_contract.tome.language_tokenizer_binding_v1 import canonical_json_bytes
    from radjax_contract.tome.streaming_validation import open_streaming_tome
    from radjax_contract.tome.student_consumption_v6 import (
        canonical_authority_reference_identity,
        canonical_multipart_npy_identity,
        canonical_npy_component_identity,
        canonical_record_sequence_identity,
        canonical_selected_passport_identity,
        sha256_identity,
    )

    multipart = {
        "target_shard": (
            (
                "attention_mask",
                paths["attention_mask"],
                ["example", "sequence_position"],
            ),
            ("input_ids", paths["input_ids"], ["example", "sequence_position"]),
        ),
        "corridor_assignment": (
            ("example_index", paths["example_index.npy"], ["assignment"]),
            ("mode_id", paths["mode_id.npy"], ["assignment"]),
            ("position", paths["position.npy"], ["assignment"]),
            ("weight", paths["weight.npy"], ["assignment"]),
        ),
    }
    rows: list[dict[str, Any]] = []
    for role, components in multipart.items():
        declared = []
        for name, locator, axes in components:
            path = root / locator
            declared.append(
                {
                    "component": name,
                    "locator": locator,
                    "axes": axes,
                    "raw_sha256": _sha(path),
                    "raw_size_bytes": path.stat().st_size,
                    "semantic_identity": canonical_npy_component_identity(
                        role=role,
                        component=name,
                        array=np.load(path, allow_pickle=False),
                        axes=tuple(axes),
                    ),
                }
            )
        rows.append(
            _resource(
                root,
                role,
                "multipart_npy",
                components[0][1],
                canonical_multipart_npy_identity(role=role, components=declared),
                components=[
                    {
                        key: value
                        for key, value in item.items()
                        if key != "semantic_identity"
                    }
                    for item in declared
                ],
            )
        )
    jsonl_roles = ("example_registry", "selected_passport_index")
    for role in jsonl_roles:
        locator = paths[f"{role}.jsonl"]
        records = [
            json.loads(line)
            for line in (root / locator).read_text(encoding="utf-8").splitlines()
        ]
        identity = (
            canonical_selected_passport_identity(records)
            if role == "selected_passport_index"
            else canonical_record_sequence_identity(role=role, records=records)
        )
        rows.append(_resource(root, role, "jsonl", locator, identity))
    m7_locator = paths.get("selected_exemplar_payload.m7.tgz")
    if m7_locator is not None:
        with open_streaming_tome(root / m7_locator) as reader:
            for _ in reader:
                pass
            if reader.verification_state != "fully_verified":
                raise ValueError("native M7 payload did not fully verify")
            identity = sha256_identity(
                canonical_json_bytes(reader.descriptor.semantic_identity)
            )
        rows.append(
            _resource(
                root,
                "selected_exemplar_payload",
                "m7_tome_archive",
                m7_locator,
                identity,
            )
        )
    else:
        locator = paths["selected_exemplar_payload.jsonl"]
        records = [
            json.loads(line)
            for line in (root / locator).read_text(encoding="utf-8").splitlines()
        ]
        rows.append(
            _resource(
                root,
                "selected_exemplar_payload",
                "jsonl",
                locator,
                canonical_record_sequence_identity(
                    role="selected_exemplar_payload", records=records
                ),
            )
        )
    mode = read_json_object(root / paths["corridor_mode_table.json"])
    projection = [
        {"mode_id": item["mode_id"], "statistic_names": sorted(item["statistics"])}
        for item in mode["modes"]
    ]
    rows.append(
        _resource(
            root,
            "corridor_mode_table",
            "json",
            paths["corridor_mode_table.json"],
            sha256_identity(
                json.dumps(
                    {"modes": projection}, sort_keys=True, separators=(",", ":")
                ).encode()
            ),
        )
    )
    reference = read_json_object(root / paths["authority_reference.json"])
    rows.append(
        _resource(
            root,
            "authority_reference",
            "json",
            paths["authority_reference.json"],
            canonical_authority_reference_identity(reference),
        )
    )
    rows.append(
        _resource(
            root,
            "delivery_receipt",
            "json",
            paths["delivery_receipt.json"],
            sha256_identity(
                json.dumps(
                    read_json_object(root / paths["delivery_receipt.json"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ),
        )
    )
    return sorted(rows, key=lambda row: (row["role"], row["resource_id"]))


def _resource(
    root: Path,
    role: str,
    encoding: str,
    locator: str,
    semantic_identity: str,
    *,
    components: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path = root / locator
    result: dict[str, Any] = {
        "resource_id": f"{role}/default",
        "role": role,
        "schema": f"radjax_{role}_v6",
        "encoding": encoding,
        "semantic_identity": semantic_identity,
        "raw_sha256": _sha(path),
        "raw_size_bytes": path.stat().st_size,
        "locator": locator,
    }
    if components is not None:
        result["components"] = components
    return result


def _registry(
    resources: list[dict[str, Any]], *, authority: bool
) -> list[dict[str, str]]:
    authority_roles = {
        "target_shard",
        "example_registry",
        "corridor_mode_table",
        "corridor_assignment",
        "selected_passport_index",
        "selected_exemplar_payload",
        "authority_reference",
    }
    return [
        {
            key: row[key]
            for key in ("resource_id", "role", "schema", "semantic_identity")
        }
        for row in resources
        if (row["role"] in authority_roles) == authority
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
