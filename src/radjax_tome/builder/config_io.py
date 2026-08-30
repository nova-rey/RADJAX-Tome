"""Strict serialization boundary for the canonical M5 build intent."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_type_hints

from radjax_tome.builder.config import TomeBuildIntent, validate_tome_build_intent
from radjax_tome.corpora.config import CorpusArtifactReference, CorpusBuildIntentV2


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> Any:
    text = path.read_text()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return json.loads(text, object_pairs_hook=_unique_pairs)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ValueError("YAML build intents require PyYAML") from exc

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any, deep: bool = False) -> dict[str, Any]:
        return _unique_pairs(loader.construct_pairs(node, deep=deep))

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml.load(text, Loader=Loader)


def _dataclass(value: Any, cls: type[Any], *, base: Path, label: str) -> Any:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    declared = {field.name for field in fields(cls)}
    unknown = sorted(set(value) - declared)
    missing = sorted(declared - set(value))
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing {label} fields: {', '.join(missing)}")
    hints = get_type_hints(cls)
    result: dict[str, Any] = {}
    for field in fields(cls):
        item = value[field.name]
        field_type = hints.get(field.name)
        if isinstance(field_type, type) and is_dataclass(field_type):
            item = _dataclass(
                item, field_type, base=base, label=f"{label}.{field.name}"
            )
        elif field.name.endswith("_path") or field.name in {
            "dataset_path",
            "corpus_manifest_path",
            "output_dir",
            "parity_left",
        }:
            if item is not None:
                if not isinstance(item, str):
                    raise ValueError(f"{label}.{field.name} must be a string or null")
                item = (
                    Path(item) if Path(item).is_absolute() else (base / item).resolve()
                )
        result[field.name] = item
    return cls(**result)


def load_tome_build_intent(path: Path) -> TomeBuildIntent:
    source = path.resolve()
    raw = _read(source)
    if (
        isinstance(raw, dict)
        and raw.get("schema_version") == "radjax_tome_build_intent_v2"
    ):
        corpus = raw.get("corpus")
        if not isinstance(corpus, dict):
            raise ValueError("build intent v2 corpus must be an object")
        artifact_path = corpus.get("artifact_path")
        expected_identity = corpus.get("expected_semantic_identity")
        if not isinstance(artifact_path, str) or not isinstance(expected_identity, str):
            raise ValueError(
                "build intent v2 requires corpus artifact_path and "
                "expected_semantic_identity"
            )
        adapted = dict(raw)
        adapted["schema_version"] = "radjax_tome_build_intent_v1"
        adapted["corpus"] = dict(
            corpus, dataset_path=artifact_path, corpus_manifest_path=artifact_path
        )
        return replace(
            load_tome_build_intent_from_raw(source, adapted),
            schema_version="radjax_tome_build_intent_v2",
        )
    return load_tome_build_intent_from_raw(source, raw)
    return load_tome_build_intent_from_raw(source, raw)


def load_tome_build_intent_from_raw(source: Path, raw: Any) -> TomeBuildIntent:
    if not isinstance(raw, dict):
        raise ValueError("build intent must be an object")
    expected = {field.name for field in fields(TomeBuildIntent)}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown:
        raise ValueError("unknown build intent fields: " + ", ".join(unknown))
    if missing:
        raise ValueError("missing build intent fields: " + ", ".join(missing))
    if raw.get("schema_version") != "radjax_tome_build_intent_v1":
        raise ValueError(
            "unsupported schema_version; expected radjax_tome_build_intent_v1"
        )
    intent = _dataclass(raw, TomeBuildIntent, base=source.parent, label="build intent")
    errors = validate_tome_build_intent(intent)
    if errors:
        raise ValueError("invalid Tome build intent: " + "; ".join(errors))
    return intent


def load_tome_build_intent_v2(path: str | Path) -> CorpusBuildIntentV2:
    """Load the explicit path-independent corpus reference projection."""

    source = Path(path).resolve()
    raw = _read(source)
    if not isinstance(raw, dict):
        raise ValueError("build intent v2 must be an object")
    if set(raw) != {"schema_version", "corpus"}:
        raise ValueError("build intent v2 requires only schema_version and corpus")
    if raw["schema_version"] != "radjax_tome_build_intent_v2":
        raise ValueError("unsupported build intent v2 schema_version")
    corpus = raw["corpus"]
    if not isinstance(corpus, dict):
        raise ValueError("build intent v2 corpus must be an object")
    required = {"artifact_path", "expected_semantic_identity", "max_examples"}
    if set(corpus) != required:
        raise ValueError(
            "build intent v2 corpus requires exactly artifact_path, "
            "expected_semantic_identity, and max_examples"
        )
    artifact_path = corpus["artifact_path"]
    expected = corpus["expected_semantic_identity"]
    max_examples = corpus["max_examples"]
    if not isinstance(artifact_path, str):
        raise ValueError("build intent v2 corpus.artifact_path must be a string")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        raise ValueError("build intent v2 expected_semantic_identity must be a digest")
    if max_examples is not None and (
        not isinstance(max_examples, int)
        or isinstance(max_examples, bool)
        or max_examples <= 0
    ):
        raise ValueError(
            "build intent v2 max_examples must be a positive integer or null"
        )
    resolved = Path(artifact_path)
    if not resolved.is_absolute():
        resolved = (source.parent / resolved).resolve()
    return CorpusBuildIntentV2(
        corpus=CorpusArtifactReference(resolved, expected, max_examples),
        source_path=source,
    )
