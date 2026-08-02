"""Read-only, role-bound post-C5 replay guards for private M8A measurements."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any  # noqa: TC003

from .measurement import SelectedPassMeasurementControl

_CHECKPOINT_SCHEMA = "selected_pass_post_c5_checkpoint_v1"
_REQUIRED_POST_C5_ROLES = frozenset(
    {
        "score",
        "corridor",
        "authority",
        "c2",
        "c3",
        "c4",
        "c5",
        "passports",
        "model",
        "tokenizer",
        "corpus",
        "config",
    }
)


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _checkpoint_digest(manifest: Mapping[str, object]) -> str:
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _relative_role_paths(root: Path, role_paths: Mapping[str, Path]) -> dict[str, str]:
    missing = _REQUIRED_POST_C5_ROLES - set(role_paths)
    if missing:
        raise ValueError(
            "post-C5 checkpoint missing required roles: " + ", ".join(sorted(missing))
        )
    unexpected = set(role_paths) - _REQUIRED_POST_C5_ROLES
    if unexpected:
        raise ValueError(
            "post-C5 checkpoint has unsupported roles: " + ", ".join(sorted(unexpected))
        )
    relative: dict[str, str] = {}
    for role, path in role_paths.items():
        resolved = path.resolve()
        try:
            relative_path = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"post-C5 role {role!r} is outside checkpoint root"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"post-C5 role {role!r} evidence is missing")
        relative[role] = relative_path.as_posix()
    return relative


@dataclass(frozen=True)
class ImmutablePostC5Checkpoint:
    """Content-addressed named evidence required before selected-pass replay."""

    root: Path
    file_digests: dict[str, str]
    role_paths: dict[str, str]
    manifest: dict[str, object]
    manifest_path: Path
    digest: str

    @classmethod
    def capture(
        cls,
        root: Path,
        *,
        role_paths: Mapping[str, Path],
        manifest_path: Path,
    ) -> ImmutablePostC5Checkpoint:
        root = root.resolve()
        manifest_path = manifest_path.resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "post-C5 checkpoint manifest must be outside checkpoint root"
            )
        relative_roles = _relative_role_paths(root, role_paths)
        file_digests = _tree_digests(root)
        if any(path not in file_digests for path in relative_roles.values()):
            raise ValueError("post-C5 checkpoint role digest is unavailable")
        manifest: dict[str, object] = {
            "schema_version": _CHECKPOINT_SCHEMA,
            "roles": {
                role: {
                    "path": relative_roles[role],
                    "sha256": file_digests[relative_roles[role]],
                }
                for role in sorted(relative_roles)
            },
            "file_digests": file_digests,
        }
        digest = _checkpoint_digest(manifest)
        document = {**manifest, "checkpoint_digest": digest}
        _write_json_atomic(manifest_path, document)
        return cls(
            root=root,
            file_digests=file_digests,
            role_paths=relative_roles,
            manifest=manifest,
            manifest_path=manifest_path,
            digest=digest,
        )

    def verify_unchanged(self) -> None:
        if _tree_digests(self.root) != self.file_digests:
            raise ValueError(
                "immutable post-C5 checkpoint changed during selected-pass replay"
            )
        roles = self.manifest.get("roles")
        if not isinstance(roles, Mapping) or set(roles) != _REQUIRED_POST_C5_ROLES:
            raise ValueError("post-C5 checkpoint manifest role set is invalid")
        if not self.manifest_path.is_file():
            raise ValueError("post-C5 checkpoint manifest is missing")
        persisted = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if persisted != {**self.manifest, "checkpoint_digest": self.digest}:
            raise ValueError("post-C5 checkpoint manifest changed")

    def prepare_temporary_output(self, output_root: Path) -> None:
        output_root = output_root.resolve()
        if output_root == self.root:
            raise ValueError("measurement output root must not be the checkpoint root")
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError("measurement output root must be fresh")
        output_root.mkdir(parents=True, exist_ok=True)
        # Copy rather than hard-link: a replay write cannot mutate upstream.
        for relative in self.file_digests:
            source = self.root / relative
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_selected_delivery_replay(
    config: Any,
    *,
    checkpoint: ImmutablePostC5Checkpoint,
    control: SelectedPassMeasurementControl,
) -> Any:
    """Invoke the canonical rerun owner; score/selection writers are forbidden."""

    if control.immutable_checkpoint_digest != checkpoint.digest:
        raise ValueError(
            "measurement control checkpoint digest does not match checkpoint"
        )
    control.validate_for_output(config.artifact_dir)
    if config.authoritative_records is None or not config.authoritative_selection:
        raise ValueError(
            "selected-pass replay requires frozen authoritative C5 records"
        )
    validation_started = perf_counter()
    checkpoint.verify_unchanged()
    before_seconds = perf_counter() - validation_started
    from .rerun import run_selected_delivery_rerun

    result = run_selected_delivery_rerun(config, _measurement_control=control)
    validation_started = perf_counter()
    checkpoint.verify_unchanged()
    after_seconds = perf_counter() - validation_started
    diagnostics = config.rerun_metrics.get("selected_pass_execution_v1")
    if isinstance(diagnostics, dict):
        diagnostics["checkpoint_validation"] = {
            "before_seconds": before_seconds,
            "after_seconds": after_seconds,
            "included_in_selected_pass_wall_time": False,
        }
        diagnostics["score_pass_invocation_count"] = 0
        diagnostics["selection_writer_invocation_count"] = 0
    return result
