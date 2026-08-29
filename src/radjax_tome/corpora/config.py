"""Strict configuration and identity contracts for the M10 corpus artifact.

The v1 loader in :mod:`radjax_tome.builder.config_io` is intentionally left
alone.  This module owns the new corpus contract so adding corpus v2 cannot
silently change the historical M5 selection projection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORPUS_BUILD_INTENT_SCHEMA_V1 = "radjax_tome_corpus_build_intent_v1"
CORPUS_ARTIFACT_SCHEMA_V2 = "radjax_tome_corpus_artifact_v2"
CORPUS_AUTHORITY_SCHEMA_V2 = "selection_authority_payload_v2"


@dataclass(frozen=True)
class CorpusSourceSpec:
    """One declared local source, in declaration order."""

    source_id: str
    adapter: str
    path: Path
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    text_field: str = "text"
    record_id_field: str | None = None


@dataclass(frozen=True)
class CorpusBuildIntent:
    """Validated corpus v1 build intent.

    The section mappings are retained as immutable plain values because they
    are also the canonical, versioned input to semantic identity.  The
    ``source_path`` is configuration provenance only and never enters that
    identity.
    """

    artifact: Mapping[str, Any]
    sources: tuple[CorpusSourceSpec, ...]
    policy: Mapping[str, Any]
    layout: Mapping[str, Any]
    resources: Mapping[str, Any]
    output: Mapping[str, Any]
    execution: Mapping[str, Any]
    reporting: Mapping[str, Any]
    source_path: Path | None = None
    schema_version: str = CORPUS_BUILD_INTENT_SCHEMA_V1

    @property
    def output_path(self) -> Path:
        value = self.output.get("artifact_path", self.output.get("output_dir"))
        if not isinstance(value, Path):
            raise ValueError("output.artifact_path must be a path")
        return value

    @property
    def resume(self) -> bool:
        return bool(self.execution.get("resume", False))

    @property
    def overwrite(self) -> bool:
        return bool(self.execution.get("overwrite", False))


@dataclass(frozen=True)
class CorpusArtifactReference:
    """A production-facing reference to an already validated corpus v2."""

    artifact_path: Path
    expected_semantic_identity: str
    max_examples: int | None = None


@dataclass(frozen=True)
class CorpusBuildIntentV2:
    """The explicit M5 production projection for a corpus v2 input."""

    corpus: CorpusArtifactReference
    source_path: Path | None = None
    schema_version: str = "radjax_tome_build_intent_v2"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _tuple_strings(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def _source(value: Any, *, base: Path, index: int) -> CorpusSourceSpec:
    if not isinstance(value, dict):
        raise ValueError(f"sources[{index}] must be an object")
    allowed = {
        "source_id",
        "adapter",
        "path",
        "include",
        "exclude",
        "text_field",
        "record_id_field",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown sources[{index}] fields: {', '.join(unknown)}")
    for field in ("source_id", "adapter", "path"):
        if field not in value:
            raise ValueError(f"sources[{index}] missing required field: {field}")
    source_id = value["source_id"]
    adapter = value["adapter"]
    path = value["path"]
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"sources[{index}].source_id must be a non-empty string")
    if not isinstance(adapter, str) or adapter not in {
        "local_text_tree_v1",
        "local_jsonl_text_v1",
    }:
        raise ValueError(f"sources[{index}].adapter is unsupported")
    if not isinstance(path, str) or not path:
        raise ValueError(f"sources[{index}].path must be a string")
    resolved = (base / path).resolve() if not Path(path).is_absolute() else Path(path)
    text_field = value.get("text_field", "text")
    record_id_field = value.get("record_id_field")
    if not isinstance(text_field, str) or not text_field:
        raise ValueError(f"sources[{index}].text_field must be a non-empty string")
    if record_id_field is not None and not isinstance(record_id_field, str):
        raise ValueError(f"sources[{index}].record_id_field must be a string or null")
    return CorpusSourceSpec(
        source_id=source_id,
        adapter=adapter,
        path=resolved,
        include=_tuple_strings(value.get("include"), f"sources[{index}].include"),
        exclude=_tuple_strings(value.get("exclude"), f"sources[{index}].exclude"),
        text_field=text_field,
        record_id_field=record_id_field,
    )


def load_corpus_build_intent(path: str | Path) -> CorpusBuildIntent:
    """Load strict JSON/YAML corpus intent v1.

    This boundary rejects duplicate keys, unknown top-level/section fields,
    environment interpolation, and missing required sections.  Section values
    are intentionally declarative and are validated again by the builder.
    """

    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"missing corpus build intent: {source}")
    raw = _read_document(source)
    if not isinstance(raw, dict):
        raise ValueError("corpus build intent must be an object")
    required = {
        "schema_version",
        "artifact",
        "sources",
        "policy",
        "layout",
        "resources",
        "output",
        "execution",
        "reporting",
    }
    unknown = sorted(set(raw) - required)
    missing = sorted(required - set(raw))
    if unknown:
        raise ValueError("unknown corpus build intent fields: " + ", ".join(unknown))
    if missing:
        raise ValueError("missing corpus build intent fields: " + ", ".join(missing))
    if raw["schema_version"] != CORPUS_BUILD_INTENT_SCHEMA_V1:
        raise ValueError(
            "unsupported schema_version; expected " + CORPUS_BUILD_INTENT_SCHEMA_V1
        )
    sections = tuple(required - {"schema_version"})
    for section in sections:
        if not isinstance(raw[section], (dict, list)):
            raise ValueError(
                f"{section} must be an object"
                if section != "sources"
                else "sources must be an array"
            )
    sources_raw = raw["sources"]
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError("sources must be a non-empty array")
    sources = tuple(
        _source(item, base=source.parent, index=index)
        for index, item in enumerate(sources_raw)
    )
    artifact = _section(raw["artifact"], "artifact")
    policy = _section(raw["policy"], "policy")
    layout = _section(raw["layout"], "layout")
    resources = _section(raw["resources"], "resources")
    output = _section(raw["output"], "output")
    execution = _section(raw["execution"], "execution")
    reporting = _section(raw["reporting"], "reporting")
    _check_fields(artifact, {"schema_version", "name"}, "artifact")
    _check_fields(
        policy,
        {
            "normalization",
            "filtering",
            "chunking",
            "deduplication",
            "ordering",
            "tokenizer",
        },
        "policy",
    )
    for field in (
        "normalization",
        "filtering",
        "chunking",
        "deduplication",
        "ordering",
        "tokenizer",
    ):
        if field not in policy:
            raise ValueError(f"policy missing required field: {field}")
    filtering = policy["filtering"]
    chunking = policy["chunking"]
    deduplication = policy["deduplication"]
    if (
        not isinstance(filtering, dict)
        or not isinstance(chunking, dict)
        or not isinstance(deduplication, dict)
    ):
        raise ValueError(
            "policy filtering, chunking, and deduplication must be objects"
        )
    _check_fields(filtering, {"min_chars"}, "policy.filtering")
    _check_fields(chunking, {"name", "max_chars"}, "policy.chunking")
    _check_fields(deduplication, {"enabled"}, "policy.deduplication")
    for field in ("name", "max_chars"):
        if field not in chunking:
            raise ValueError(f"policy.chunking missing required field: {field}")
    if "enabled" not in deduplication:
        raise ValueError("policy.deduplication missing required field: enabled")
    if not isinstance(filtering.get("min_chars", 1), int) or isinstance(
        filtering.get("min_chars", 1), bool
    ):
        raise ValueError("policy.filtering.min_chars must be an integer")
    if not isinstance(chunking["max_chars"], int) or isinstance(
        chunking["max_chars"], bool
    ):
        raise ValueError("policy.chunking.max_chars must be an integer")
    if not isinstance(deduplication["enabled"], bool):
        raise ValueError("policy.deduplication.enabled must be boolean")
    _check_fields(
        layout, {"shard_capacity", "shard_size_examples", "max_shard_bytes"}, "layout"
    )
    if "shard_capacity" not in layout and "shard_size_examples" not in layout:
        raise ValueError("layout requires shard_capacity")
    _check_fields(
        resources,
        {
            "memory_limit",
            "max_memory_bytes",
            "duckdb_memory_limit",
            "worker_count",
            "max_open_files",
            "max_artifact_bytes",
        },
        "resources",
    )
    _check_fields(output, {"artifact_path", "output_dir"}, "output")
    _check_fields(execution, {"resume", "overwrite"}, "execution")
    _check_fields(reporting, {"progress", "format"}, "reporting")
    if "schema_version" not in artifact:
        raise ValueError("artifact missing required field: schema_version")
    if "min_chars" not in filtering:
        raise ValueError("policy.filtering missing required field: min_chars")
    artifact_version = artifact["schema_version"]
    if artifact_version != CORPUS_ARTIFACT_SCHEMA_V2:
        raise ValueError("artifact.schema_version must be " + CORPUS_ARTIFACT_SCHEMA_V2)
    artifact_path = output.get("artifact_path", output.get("output_dir"))
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ValueError("output.artifact_path is required")
    output = dict(output)
    output["artifact_path"] = (
        (source.parent / artifact_path).resolve()
        if not Path(artifact_path).is_absolute()
        else Path(artifact_path)
    )
    for name, value in (
        ("resume", execution.get("resume", False)),
        ("overwrite", execution.get("overwrite", False)),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"execution.{name} must be boolean")
    if execution.get("resume", False) and execution.get("overwrite", False):
        raise ValueError(
            "execution.resume and execution.overwrite are mutually exclusive"
        )
    return CorpusBuildIntent(
        artifact=artifact,
        sources=sources,
        policy=policy,
        layout=layout,
        resources=resources,
        output=output,
        execution=execution,
        reporting=reporting,
        source_path=source,
    )


def selection_authority_payload_v2(
    *,
    corpus_semantic_identity: str,
    tokenizer_binding_digest: str,
    policy: Mapping[str, Any],
    corpus_schema: str = CORPUS_ARTIFACT_SCHEMA_V2,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Return the path-independent v2 production authority projection."""

    if not isinstance(
        corpus_semantic_identity, str
    ) or not corpus_semantic_identity.startswith("sha256:"):
        raise ValueError("corpus semantic identity must be a sha256 digest")
    if not isinstance(
        tokenizer_binding_digest, str
    ) or not tokenizer_binding_digest.startswith("sha256:"):
        raise ValueError("tokenizer binding digest must be a sha256 digest")
    return {
        "schema_version": CORPUS_AUTHORITY_SCHEMA_V2,
        "corpus_schema": corpus_schema,
        "corpus_semantic_identity": corpus_semantic_identity,
        "tokenizer_binding_digest": tokenizer_binding_digest,
        "policy": dict(policy),
        "max_examples": max_examples,
    }


def selection_authority_hash_v2(**kwargs: Any) -> str:
    return sha256(canonical_bytes(selection_authority_payload_v2(**kwargs)))


def _section(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _check_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")


def _read_document(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return json.loads(text, object_pairs_hook=_unique_pairs)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ValueError("YAML corpus intents require PyYAML") from exc

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any, deep: bool = False) -> dict[str, Any]:
        return _unique_pairs(loader.construct_pairs(node, deep=deep))

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml.load(text, Loader=Loader)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


__all__ = [
    "CORPUS_ARTIFACT_SCHEMA_V2",
    "CORPUS_AUTHORITY_SCHEMA_V2",
    "CORPUS_BUILD_INTENT_SCHEMA_V1",
    "CorpusArtifactReference",
    "CorpusBuildIntent",
    "CorpusBuildIntentV2",
    "CorpusSourceSpec",
    "canonical_bytes",
    "load_corpus_build_intent",
    "selection_authority_hash_v2",
    "selection_authority_payload_v2",
    "sha256",
]
