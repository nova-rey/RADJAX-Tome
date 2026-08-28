"""Native Path-B terminal publication of the canonical v4 payload package.

The native state machine still owns score, selection, selected rerun, corridor,
assembly, and legacy validation.  This small terminal adapter runs *after*
those proofs: it projects the completed legacy artifact into the v4 physical
layout at a sibling path.  Keeping the target outside the legacy artifact is
intentional: it prevents the v4 inventory from recursively packaging itself
while retaining the historical v3 tree as a compatibility and resume input.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.builder.delivery.staging import (
    V4ShardResumeState,
    complete_v4_shard_staging,
    prepare_v4_shard_staging,
    seal_v4_shard,
)
from radjax_tome.io.json import read_json_object
from radjax_tome.tome.canonical_artifact import derive_tome_semantic_identity
from radjax_tome.tome.payload_sharding_v4 import (
    ShardedTomeV4Result,
    _canonical_bytes,
    _copy_legacy_profile_members,
    _legacy_selected_records,
    _nonselected_payload,
    _write_directory,
    pack_sharded_tome_v4,
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


def native_v4_staging_path(output_dir: Path) -> Path:
    """Return the durable, non-public shard transaction for a native build."""

    return output_dir.with_name(f".{output_dir.name}.v4-shard-staging")


def publish_native_path_b_v4(config: Any) -> NativeV4Publication:
    """Transactionally publish the paved v4 form after retained Path-B proof.

    The legacy artifact is the frozen Path-B evidence input, not the v4
    delivery transaction.  Its selected records are streamed into durable,
    sealed v4 shards first.  A resumed invocation verifies every sealed prefix
    against that source before reuse, writes both physical outputs under
    sibling temporary names, validates them with the pinned portable Contract
    validator, and only then atomically promotes each final member.

    Filesystem rename cannot atomically replace a directory and an archive as
    one object.  The durable shard receipt binds the pair: a process stopped
    before the second rename leaves only validated temporary/final members and
    a later resume reconstructs the same pair from the verified receipt.  A
    completed pair is never accepted on resume without validating both forms.
    """

    output_dir = Path(config.output_dir)
    directory_path = native_v4_directory_path(output_dir)
    archive_path = native_v4_archive_path(output_dir)
    overwrite = bool(config.overwrite)
    if bool(config.resume) and directory_path.is_dir() and archive_path.is_file():
        _validate_contract(directory_path)
        _validate_contract(archive_path)
        return NativeV4Publication(
            directory=_read_completed_directory(directory_path),
            archive_path=archive_path,
        )

    if directory_path.exists() or archive_path.exists():
        if not (bool(config.resume) or overwrite):
            raise ValueError(
                "native v4 publication already exists; use resume or overwrite"
            )
        # A partially promoted pair is not a completed publication.  It is
        # replaced only from the receipt-backed, Contract-validated candidate.
        _remove_publication_path(directory_path)
        _remove_publication_path(archive_path)

    capacity = config.payload_records_per_shard
    source_identity = derive_tome_semantic_identity(output_dir)
    stage = native_v4_staging_path(output_dir)
    state = prepare_v4_shard_staging(
        stage,
        config={
            "source_semantic_identity": source_identity.semantic_digest,
            "profile": "student",
        },
        payload_records_per_shard=capacity,
    )
    _hook(config, "after_v4_staging_prepare")
    state = _stage_native_records(output_dir, state, config)
    complete_v4_shard_staging(state, expected_record_count=state.completed_record_count)
    _hook(config, "after_v4_shard_staging_complete")

    directory_candidate = _directory_candidate_path(directory_path)
    archive_candidate = _archive_candidate_path(archive_path)
    _remove_publication_path(directory_candidate)
    _remove_publication_path(archive_candidate)
    _hook(config, "before_v4_directory_materialization")
    directory = _materialize_directory_from_staged_records(
        output_dir, directory_candidate, state
    )
    _validate_contract(directory.root)
    _hook(config, "before_v4_archive_packing")
    pack_sharded_tome_v4(
        directory.root,
        archive_candidate,
        compression="gz",
        overwrite=False,
    )
    _validate_contract(archive_candidate)
    _hook(config, "before_v4_final_promotion")
    os.replace(directory_candidate, directory_path)
    _hook(config, "after_v4_directory_promotion")
    os.replace(archive_candidate, archive_path)
    return NativeV4Publication(
        directory=ShardedTomeV4Result(
            directory_path,
            directory.semantic_identity_digest,
            directory.selected_count,
            directory.shard_count,
        ),
        archive_path=archive_path,
    )


def _stage_native_records(
    source: Path, state: V4ShardResumeState, config: Any
) -> V4ShardResumeState:
    """Verify a sealed prefix then append only missing source-order records."""

    _verify_sealed_prefix(source, state)
    records = itertools.islice(
        _legacy_selected_records(source), state.completed_record_count, None
    )
    while True:
        first = next(records, None)
        if first is None:
            return state
        encoded = itertools.chain(
            (_canonical_bytes(first),),
            (
                _canonical_bytes(record)
                for record in itertools.islice(
                    records, state.payload_records_per_shard - 1
                )
            ),
        )
        state = seal_v4_shard(state, encoded)
        _hook(config, "after_v4_shard_sealed")


def _verify_sealed_prefix(source: Path, state: V4ShardResumeState) -> None:
    """Reject reuse unless receipt-backed bytes still equal Path-B source order."""

    expected = iter(_legacy_selected_records(source))
    for sealed in state.sealed_shards:
        shard_path = state.stage / sealed.path
        with shard_path.open("rb") as handle:
            for _ in range(sealed.record_count):
                try:
                    record = next(expected)
                except StopIteration as exc:
                    raise ValueError(
                        "v4 staging prefix exceeds native selected payload"
                    ) from exc
                if handle.readline() != _canonical_bytes(record) + b"\n":
                    raise ValueError(
                        "v4 staging prefix no longer matches native payload"
                    )
            if handle.readline():
                raise ValueError("v4 staging shard has unexpected trailing record")


def _materialize_directory_from_staged_records(
    source: Path, destination: Path, state: V4ShardResumeState
) -> ShardedTomeV4Result:
    """Build a candidate only from receipt-verified sealed JSONL shards."""

    source_identity = derive_tome_semantic_identity(source)
    destination.mkdir(parents=True)
    _copy_legacy_profile_members(source, destination, profile="student")
    return _write_directory(
        _staged_records(state),
        destination,
        training_contract=source_identity.training_contract,
        authority=source_identity.authority,
        nonselected_training_payload=_nonselected_payload(source_identity.to_dict()),
        profile="student",
        capacity=state.payload_records_per_shard,
        allow_compact_without_mask=True,
    )


def _staged_records(state: V4ShardResumeState) -> Iterator[dict[str, Any]]:
    """Yield sealed bytes one record at a time; never retain a whole shard."""

    for sealed in state.sealed_shards:
        with (state.stage / sealed.path).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("sealed v4 shard record is invalid")
                yield record


def _validate_contract(path: Path) -> None:
    """Use the pinned portable Contract validator before a public promotion."""

    from radjax_contract.tome import validate_streaming_tome

    result = validate_streaming_tome(path, strict=True)
    if not result.ok:
        raise ValueError(
            "native v4 publication failed Contract validation: "
            + ", ".join(result.errors)
        )


def _directory_candidate_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.candidate")


def _archive_candidate_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.candidate")


def _remove_publication_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _hook(config: Any, event: str) -> None:
    """Invoke a test-only interruption hook when supplied by a caller."""

    callback = getattr(config, "v4_publication_hook", None)
    if callback is not None:
        callback(event)


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
