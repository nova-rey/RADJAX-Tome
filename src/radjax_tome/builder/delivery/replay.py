"""Read-only post-C5 replay guards for private M8A measurements."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .measurement import SelectedPassMeasurementControl


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True)
class ImmutablePostC5Checkpoint:
    """Content-addressed upstream evidence whose files may never be replay outputs."""

    root: Path
    file_digests: dict[str, str]
    digest: str

    @classmethod
    def capture(cls, root: Path) -> ImmutablePostC5Checkpoint:
        root = root.resolve()
        digests = _tree_digests(root)
        body = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
        return cls(
            root=root,
            file_digests=digests,
            digest="sha256:" + hashlib.sha256(body).hexdigest(),
        )

    def verify_unchanged(self) -> None:
        if _tree_digests(self.root) != self.file_digests:
            raise ValueError(
                "immutable post-C5 checkpoint changed during selected-pass replay"
            )

    def prepare_temporary_output(self, output_root: Path) -> None:
        output_root = output_root.resolve()
        if output_root == self.root:
            raise ValueError("measurement output root must not be the checkpoint root")
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError("measurement output root must be fresh")
        output_root.mkdir(parents=True, exist_ok=True)
        # Copy rather than hard-link: an accidental replay write cannot mutate upstream.
        for relative in self.file_digests:
            source = self.root / relative
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


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
