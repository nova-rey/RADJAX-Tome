"""Native Path-B terminal publication of the canonical v4 payload package.

The native state machine still owns score, selection, selected rerun, corridor,
assembly, and legacy validation.  This small terminal adapter runs *after*
those proofs: it projects the completed legacy artifact into the v4 physical
layout at a sibling path.  Keeping the target outside the legacy artifact is
intentional: it prevents the v4 inventory from recursively packaging itself
while retaining the historical v3 tree as a compatibility and resume input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object
from radjax_tome.tome import (
    ShardedTomeV4Result,
    pack_sharded_tome_v4,
    package_legacy_artifact_as_sharded_tome_v4,
)


@dataclass(frozen=True)
class NativeV4Publication:
    """The two deterministic physical forms emitted by a native Path-B run."""

    directory: ShardedTomeV4Result
    archive_path: Path


def native_v4_directory_path(output_dir: Path) -> Path:
    """Return the non-recursive canonical v4 directory adjacent to ``output_dir``."""

    return output_dir.with_name(f"{output_dir.name}.v4")


def native_v4_archive_path(output_dir: Path) -> Path:
    """Return the ordinary native build's distributable deterministic archive."""

    return output_dir.with_name(f"{output_dir.name}.v4.tgz")


def publish_native_path_b_v4(config: Any) -> NativeV4Publication:
    """Publish v4 only after the retained Path-B proof has completed.

    ``package_legacy_artifact_as_sharded_tome_v4`` itself stages and atomically
    promotes the directory.  A failed projection cannot replace a prior v4
    publication, and normal non-resume builds never overwrite an existing
    sibling publication.
    """

    output_dir = Path(config.output_dir)
    directory_path = native_v4_directory_path(output_dir)
    archive_path = native_v4_archive_path(output_dir)
    overwrite = bool(config.overwrite)
    if bool(config.resume) and directory_path.is_dir() and archive_path.is_file():
        # Resume handling validates/reuses staged shard prefixes in the delivery
        # layer.  A completed terminal publication is immutable evidence.
        # Later validation still checks it before a successful terminal report.
        return NativeV4Publication(
            directory=_read_completed_directory(directory_path),
            archive_path=archive_path,
        )
    directory = package_legacy_artifact_as_sharded_tome_v4(
        output_dir,
        directory_path,
        profile="student",
        payload_records_per_shard=config.payload_records_per_shard,
        overwrite=overwrite,
    )
    pack_sharded_tome_v4(
        directory.root,
        archive_path,
        compression="gz",
        overwrite=overwrite,
    )
    return NativeV4Publication(directory=directory, archive_path=archive_path)


def _read_completed_directory(root: Path) -> ShardedTomeV4Result:
    """Recover immutable completed-publication facts for a resume handoff."""

    cover = read_json_object(root / "cover_page.json")
    identity = cover.get("identity")
    layout = read_json_object(root / "selected_exemplars" / "payload-layout.json")
    if not isinstance(identity, dict) or not isinstance(layout, dict):
        raise ValueError(
            "completed native v4 publication is missing identity or layout"
        )
    digest = identity.get("semantic_digest")
    selected_count = layout.get("selected_count")
    shard_index = layout.get("shard_index")
    shard_count = (
        shard_index.get("record_count") if isinstance(shard_index, dict) else None
    )
    if (
        not isinstance(digest, str)
        or not isinstance(selected_count, int)
        or isinstance(selected_count, bool)
        or not isinstance(shard_count, int)
        or isinstance(shard_count, bool)
    ):
        raise ValueError("completed native v4 publication has invalid summary fields")
    return ShardedTomeV4Result(root, digest, selected_count, shard_count)
