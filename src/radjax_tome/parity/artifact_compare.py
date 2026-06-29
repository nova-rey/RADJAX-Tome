from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SIDECAR_FILES = (
    "metadata.json",
    "vocab_contract.json",
    "teacher_manifest.json",
    "emission_config.json",
    "validation_report.json",
)

ALLOWED_JSON_DIFFERENCES: dict[tuple[str, str], str] = {
    ("metadata.json", "/created_at"): "volatile artifact build timestamp",
    ("metadata.json", "/created_by"): "expected package/module provenance change",
    (
        "metadata.json",
        "/provenance/phase",
    ): "expected migration phase provenance change",
    ("teacher_manifest.json", "/created_at"): "volatile artifact build timestamp",
}


@dataclass(frozen=True)
class JsonDifference:
    file: str
    path: str
    old_value: Any
    new_value: Any
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArrayComparison:
    shard: str
    array: str
    status: str
    old_shape: tuple[int, ...]
    new_shape: tuple[int, ...]
    old_dtype: str
    new_dtype: str
    max_abs_diff: float | None = None
    tolerance_used: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactComparison:
    status: str
    sidecar_comparison: dict[str, Any] = field(default_factory=dict)
    shard_comparison: dict[str, Any] = field(default_factory=dict)
    allowed_differences: tuple[JsonDifference, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_differences"] = [
            item.to_dict() for item in self.allowed_differences
        ]
        return payload


def compare_teacher_textbook_artifacts(
    old_dir: str | Path,
    new_dir: str | Path,
) -> ArtifactComparison:
    old_root = Path(old_dir)
    new_root = Path(new_dir)
    blockers: list[str] = []
    warnings: list[str] = []
    allowed: list[JsonDifference] = []

    sidecar_summary = _compare_sidecars(old_root, new_root, blockers, allowed)
    shard_summary = _compare_shards(old_root, new_root, blockers, warnings)
    _compare_directory_structure(old_root, new_root, blockers)

    return ArtifactComparison(
        status="fail" if blockers else "pass",
        sidecar_comparison=sidecar_summary,
        shard_comparison=shard_summary,
        allowed_differences=tuple(allowed),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def _compare_sidecars(
    old_root: Path,
    new_root: Path,
    blockers: list[str],
    allowed: list[JsonDifference],
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in SIDECAR_FILES:
        old_path = old_root / name
        new_path = new_root / name
        file_summary: dict[str, Any] = {"status": "pass"}
        if not old_path.is_file() or not new_path.is_file():
            if not old_path.is_file():
                blockers.append(f"missing old sidecar: {name}")
            if not new_path.is_file():
                blockers.append(f"missing new sidecar: {name}")
            file_summary["status"] = "fail"
            files[name] = file_summary
            continue
        old_payload = _read_json(old_path, blockers)
        new_payload = _read_json(new_path, blockers)
        if old_payload is None or new_payload is None:
            file_summary["status"] = "fail"
            files[name] = file_summary
            continue
        before = len(blockers)
        _compare_json_value(name, "", old_payload, new_payload, blockers, allowed)
        file_summary["status"] = "fail" if len(blockers) != before else "pass"
        files[name] = file_summary
    return {"files": files}


def _compare_json_value(
    file_name: str,
    path: str,
    old_value: Any,
    new_value: Any,
    blockers: list[str],
    allowed: list[JsonDifference],
) -> None:
    pointer = path or "/"
    if isinstance(old_value, dict) and isinstance(new_value, dict):
        old_keys = set(old_value)
        new_keys = set(new_value)
        for key in sorted(old_keys - new_keys):
            blockers.append(f"{file_name}{_join_pointer(path, key)} missing in new")
        for key in sorted(new_keys - old_keys):
            blockers.append(f"{file_name}{_join_pointer(path, key)} extra in new")
        for key in sorted(old_keys & new_keys):
            _compare_json_value(
                file_name,
                _join_pointer(path, key),
                old_value[key],
                new_value[key],
                blockers,
                allowed,
            )
        return
    if isinstance(old_value, list) and isinstance(new_value, list):
        if len(old_value) != len(new_value):
            blockers.append(
                f"{file_name}{pointer} list length mismatch: "
                f"old={len(old_value)} new={len(new_value)}"
            )
            return
        for index, (old_item, new_item) in enumerate(
            zip(old_value, new_value, strict=True)
        ):
            _compare_json_value(
                file_name,
                _join_pointer(path, str(index)),
                old_item,
                new_item,
                blockers,
                allowed,
            )
        return
    if old_value == new_value:
        return
    reason = ALLOWED_JSON_DIFFERENCES.get((file_name, pointer))
    if reason is not None:
        allowed.append(
            JsonDifference(
                file=file_name,
                path=pointer,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
            )
        )
        return
    blockers.append(
        f"{file_name}{pointer} mismatch: old={old_value!r} new={new_value!r}"
    )


def _compare_shards(
    old_root: Path,
    new_root: Path,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    old_shards = _relative_shards(old_root)
    new_shards = _relative_shards(new_root)
    shard_results: list[dict[str, Any]] = []
    if old_shards != new_shards:
        for name in sorted(set(old_shards) - set(new_shards)):
            blockers.append(f"missing new shard: {name}")
        for name in sorted(set(new_shards) - set(old_shards)):
            blockers.append(f"extra new shard: {name}")
    for shard in sorted(set(old_shards) & set(new_shards)):
        shard_results.append(
            _compare_shard_file(
                old_root / shard,
                new_root / shard,
                shard,
                blockers,
                warnings,
            )
        )
    return {
        "old_shard_count": len(old_shards),
        "new_shard_count": len(new_shards),
        "shards": shard_results,
    }


def _compare_shard_file(
    old_path: Path,
    new_path: Path,
    shard_name: str,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    arrays: list[ArrayComparison] = []
    blocker_count_before = len(blockers)
    with np.load(old_path, allow_pickle=False) as old_loaded:
        with np.load(new_path, allow_pickle=False) as new_loaded:
            old_keys = set(old_loaded.files)
            new_keys = set(new_loaded.files)
            for key in sorted(old_keys - new_keys):
                blockers.append(f"{shard_name} missing new array: {key}")
            for key in sorted(new_keys - old_keys):
                blockers.append(f"{shard_name} extra new array: {key}")
            for key in sorted(old_keys & new_keys):
                arrays.append(
                    _compare_array(
                        shard_name,
                        key,
                        old_loaded[key],
                        new_loaded[key],
                        blockers,
                        warnings,
                    )
                )
    status = (
        "fail"
        if len(blockers) != blocker_count_before
        or any(item.status == "fail" for item in arrays)
        else "pass"
    )
    return {
        "shard": shard_name,
        "status": status,
        "arrays": [item.to_dict() for item in arrays],
    }


def _compare_array(
    shard_name: str,
    key: str,
    old_array: np.ndarray,
    new_array: np.ndarray,
    blockers: list[str],
    warnings: list[str],
) -> ArrayComparison:
    old_shape = tuple(int(part) for part in old_array.shape)
    new_shape = tuple(int(part) for part in new_array.shape)
    old_dtype = str(old_array.dtype)
    new_dtype = str(new_array.dtype)
    if old_shape != new_shape:
        blockers.append(
            f"{shard_name}:{key} shape mismatch: old={old_shape} new={new_shape}"
        )
        return ArrayComparison(
            shard=shard_name,
            array=key,
            status="fail",
            old_shape=old_shape,
            new_shape=new_shape,
            old_dtype=old_dtype,
            new_dtype=new_dtype,
        )
    if old_dtype != new_dtype:
        blockers.append(
            f"{shard_name}:{key} dtype mismatch: old={old_dtype} new={new_dtype}"
        )
        return ArrayComparison(
            shard=shard_name,
            array=key,
            status="fail",
            old_shape=old_shape,
            new_shape=new_shape,
            old_dtype=old_dtype,
            new_dtype=new_dtype,
        )
    if np.array_equal(old_array, new_array):
        return ArrayComparison(
            shard=shard_name,
            array=key,
            status="pass",
            old_shape=old_shape,
            new_shape=new_shape,
            old_dtype=old_dtype,
            new_dtype=new_dtype,
        )
    if np.issubdtype(old_array.dtype, np.floating):
        tolerance = 1e-6 if old_array.dtype == np.float32 else 1e-3
        if np.allclose(old_array, new_array, rtol=0.0, atol=tolerance):
            diff = float(np.max(np.abs(old_array.astype(np.float64) - new_array)))
            warnings.append(
                f"{shard_name}:{key} float values matched within atol={tolerance}"
            )
            return ArrayComparison(
                shard=shard_name,
                array=key,
                status="pass",
                old_shape=old_shape,
                new_shape=new_shape,
                old_dtype=old_dtype,
                new_dtype=new_dtype,
                max_abs_diff=diff,
                tolerance_used=tolerance,
            )
    blockers.append(f"{shard_name}:{key} value mismatch")
    return ArrayComparison(
        shard=shard_name,
        array=key,
        status="fail",
        old_shape=old_shape,
        new_shape=new_shape,
        old_dtype=old_dtype,
        new_dtype=new_dtype,
    )


def _compare_directory_structure(
    old_root: Path,
    new_root: Path,
    blockers: list[str],
) -> None:
    old_files = _relative_files(old_root)
    new_files = _relative_files(new_root)
    for name in sorted(old_files - new_files):
        blockers.append(f"missing new artifact file: {name}")
    for name in sorted(new_files - old_files):
        blockers.append(f"extra new artifact file: {name}")


def _relative_files(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _relative_shards(root: Path) -> set[str]:
    shards = root / "shards"
    if not shards.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in shards.glob("shard-*.npz")
        if path.is_file()
    }


def _read_json(path: Path, blockers: list[str]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"{path.name} invalid JSON: {exc}")
        return None


def _join_pointer(prefix: str, token: str) -> str:
    escaped = token.replace("~", "~0").replace("/", "~1")
    return f"{prefix}/{escaped}" if prefix else f"/{escaped}"
