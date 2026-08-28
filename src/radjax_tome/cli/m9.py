from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from pathlib import Path

from radjax_tome.builder.config import (
    TomeBuildIntent,
    canonical_production_build_intent,
    normalize_production_build_request,
    resolve_tome_build_intent,
    resolved_tome_build_config_payload,
)
from radjax_tome.builder.production import build_production_gpu_tome

_SECTIONS = (
    "teacher",
    "corpus",
    "behavior",
    "corridor_policy",
    "selection",
    "execution",
    "outputs",
    "compatibility",
    "package",
)


def _path(v):
    return None if v is None else Path(v)


def _section(cls, raw, defaults, name):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    unknown = sorted(set(raw) - {f.name for f in fields(cls)})
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")
    data = dict(defaults)
    data.update(raw)
    for f in fields(cls):
        if f.name.endswith("_path") or f.name in {
            "dataset_path",
            "corpus_manifest_path",
            "output_dir",
            "parity_left",
        }:
            if f.name in data:
                data[f.name] = _path(data[f.name])
    return cls(**data)


def load_build_intent(path):
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("YAML build intents require PyYAML") from exc
        raw = yaml.safe_load(path.read_text())
    else:
        raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("build intent must be an object")
    unknown = sorted(set(raw) - set(_SECTIONS) - {"schema_version"})
    if unknown:
        raise ValueError("unknown build intent fields: " + ", ".join(unknown))
    for key in ("teacher", "corpus", "outputs"):
        if key not in raw:
            raise ValueError(f"missing required section: {key}")
    t, c, o = raw["teacher"], raw["corpus"], raw["outputs"]
    if not all(isinstance(x, dict) for x in (t, c, o)):
        raise ValueError("teacher, corpus, and outputs must be objects")
    for key in ("model", "model_provenance_path"):
        if key not in t:
            raise ValueError(f"missing required field: teacher.{key}")
    for key in ("dataset_path", "corpus_manifest_path"):
        if key not in c:
            raise ValueError(f"missing required field: corpus.{key}")
    if "output_dir" not in o:
        raise ValueError("missing required field: outputs.output_dir")
    base = canonical_production_build_intent(
        teacher_model=str(t["model"]),
        tokenizer_id=t.get("tokenizer_id"),
        dataset_path=Path(c["dataset_path"]),
        corpus_manifest_path=Path(c["corpus_manifest_path"]),
        teacher_model_provenance_path=Path(t["model_provenance_path"]),
        output_dir=Path(o["output_dir"]),
    )
    data = asdict(base)
    data.pop("schema_version", None)
    for name in _SECTIONS:
        data[name] = _section(
            type(getattr(base, name)), raw.get(name), data[name], name
        )
    intent = TomeBuildIntent(
        **data, schema_version=raw.get("schema_version", "radjax_tome_build_intent_v1")
    )
    from radjax_tome.builder.config import validate_tome_build_intent

    errors = validate_tome_build_intent(intent)
    if errors:
        raise ValueError("invalid Tome build intent: " + "; ".join(errors))
    return intent


def classify_destination(path):
    if not path.exists():
        return "missing"
    if path.is_file():
        return "file"
    return "nonempty_directory" if any(path.iterdir()) else "empty_directory"


def preflight(intent, *, resume=False, overwrite=False):
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    state = classify_destination(intent.outputs.output_dir)
    if state == "file":
        raise ValueError(f"destination is a file: {intent.outputs.output_dir}")
    if state == "nonempty_directory" and not (resume or overwrite):
        raise ValueError(
            "destination is a nonempty directory; pass --resume or --overwrite"
        )
    if resume and state == "missing":
        raise ValueError("cannot resume a missing destination")
    return {
        "status": "ready",
        "destination": str(intent.outputs.output_dir),
        "classification": state,
        "resume": resume,
        "overwrite": overwrite,
    }


def run_intent(intent, *, resume=False, overwrite=False):
    preflight(intent, resume=resume, overwrite=overwrite)
    intent = replace(
        intent, execution=replace(intent.execution, resume=resume, overwrite=overwrite)
    )
    return build_production_gpu_tome(resolve_tome_build_intent(intent, source="m9_cli"))


def inspect_intent(intent):
    return resolved_tome_build_config_payload(
        normalize_production_build_request(
            resolve_tome_build_intent(intent, source="m9_cli")
        )
    )


def status(path):
    state = classify_destination(path)
    report = path / "production_build_report.json"
    if report.is_file():
        try:
            return {
                "status": "complete",
                "destination": str(path),
                "report": json.loads(report.read_text()),
            }
        except (OSError, ValueError, json.JSONDecodeError):
            return {"status": "invalid", "destination": str(path)}
    return {
        "status": "new" if state == "missing" else "resumable",
        "destination": str(path),
        "classification": state,
    }
