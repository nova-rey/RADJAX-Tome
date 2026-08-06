"""Private provenance-shape experiment; never imported by production Tome.

The module deliberately distinguishes four boundaries:

* standard package validation detects operational faults in one declared Tome;
* transaction validation is producer-only and never required by a consumer;
* immutable expected-identity comparison models Golden/Contract development use;
* external attestation compares an independently obtained expected identity.

None of those mechanisms proves that a malicious producer, validator, or model
origin claim is honest.  This code is a disposable next-version projection only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import resource
import tarfile
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "radjax_tome_provenance_bakeoff_experimental_vnext"
ATTESTATION_SCHEMA = SCHEMA + ".external_attestation.v1"
RUNS = 3
MATERIAL_REDUCTION = 0.20
NOISE_MULTIPLIER = 2.0


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _configuration(shape: str, capacity: int) -> dict[str, Any]:
    """Private construction settings that a resumed producer must match."""
    return {
        "schema_version": SCHEMA + ".transaction.v1",
        "shape": shape,
        "capacity": capacity,
    }


def _member_path(root: Path, value: Any, *, label: str) -> Path:
    """Resolve a declared member without permitting traversal or absolutes."""
    if not isinstance(value, str):
        raise ValueError(f"invalid {label} pointer")
    member = Path(value)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"invalid {label} pointer")
    path = root / member
    if not path.is_file():
        raise ValueError(f"missing {label} member")
    return path


def _is_public_member(root: Path, path: Path) -> bool:
    return path.is_file() and not any(
        part.startswith(".") for part in path.relative_to(root).parts
    )


@dataclass
class Counters:
    serialization_calls: int = 0
    serialization_bytes: int = 0
    bytes_written: int = 0
    temporary_bytes_written: int = 0
    final_bytes_written: int = 0
    bytes_reread: int = 0
    bytes_rewritten: int = 0
    parse_calls: int = 0
    parsed_bytes: int = 0
    hash_calls: int = 0
    hashed_bytes: int = 0
    journal_operations: int = 0
    shard_seals: int = 0

    def canonical(self, value: Any) -> bytes:
        encoded = _canonical(value)
        self.serialization_calls += 1
        self.serialization_bytes += len(encoded)
        return encoded

    def digest(self, value: bytes) -> str:
        self.hash_calls += 1
        self.hashed_bytes += len(value)
        return _digest(value)

    def digest_file(self, path: Path) -> str:
        """Hash a member without materializing an unbounded index in memory."""
        digest = hashlib.sha256()
        self.hash_calls += 1
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                self.bytes_reread += len(chunk)
                self.hashed_bytes += len(chunk)
        return "sha256:" + digest.hexdigest()

    def write(
        self,
        path: Path,
        value: bytes,
        *,
        surface: str,
        rewrite: bool = False,
    ) -> None:
        if surface not in {"temporary", "final"}:
            raise ValueError("unknown evidence surface")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        self.bytes_written += len(value)
        if surface == "temporary":
            self.temporary_bytes_written += len(value)
        else:
            self.final_bytes_written += len(value)
        if rewrite:
            self.bytes_rewritten += len(value)

    def append(self, path: Path, value: bytes, *, surface: str) -> None:
        if surface not in {"temporary", "final"}:
            raise ValueError("unknown evidence surface")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(value)
        self.bytes_written += len(value)
        if surface == "temporary":
            self.temporary_bytes_written += len(value)
        else:
            self.final_bytes_written += len(value)

    def note_written_file(self, path: Path, *, surface: str) -> None:
        size = path.stat().st_size
        self.bytes_written += size
        if surface == "temporary":
            self.temporary_bytes_written += size
        elif surface == "final":
            self.final_bytes_written += size
        else:
            raise ValueError("unknown evidence surface")

    def read(self, path: Path) -> bytes:
        value = path.read_bytes()
        self.bytes_reread += len(value)
        return value

    def parse(self, value: bytes) -> Any:
        self.parse_calls += 1
        self.parsed_bytes += len(value)
        return json.loads(value)

    def projection(self) -> dict[str, int]:
        return self.__dict__.copy()


class _FramedSequence:
    """One ordered semantic-root hash without retaining every record."""

    def __init__(self, counter: Counters) -> None:
        self._counter = counter
        self._digest = hashlib.sha256()
        prefix = b"radjax-tome-selected-sequence-vnext\x00"
        self._digest.update(prefix)
        counter.hash_calls += 1
        counter.hashed_bytes += len(prefix)
        self.count = 0

    def add_encoded(self, encoded: bytes) -> None:
        frame = len(encoded).to_bytes(8, "big") + encoded
        self._digest.update(frame)
        self._counter.hashed_bytes += len(frame)
        self.count += 1

    def finish(self) -> str:
        return "sha256:" + self._digest.hexdigest()


@dataclass(frozen=True)
class BakeoffResult:
    root: Path
    sequence_digest: str
    semantic_root: str
    archive_digest: str
    counters: dict[str, int]
    construction_seconds: float
    archive_seconds: float
    peak_rss_bytes: int
    configuration: dict[str, Any]


def _logical_id(record: Mapping[str, Any], counter: Counters) -> str:
    return counter.digest(
        counter.canonical(
            {
                "selected_example_id": record["selected_example_id"],
                "selected_position": record["selected_position"],
            }
        )
    )


def _semantic_root(
    *,
    sequence_digest: str,
    authority_digest: str,
    contract_version: str,
    behavioral_policy_identity: str,
    counter: Counters,
) -> str:
    """Bind one standard sequence identity to governed public semantics."""
    return counter.digest(
        counter.canonical(
            {
                "schema_version": SCHEMA + ".semantic_root.v1",
                "sequence_digest": sequence_digest,
                "semantic_authority_identity": authority_digest,
                "contract_version": contract_version,
                "behavioral_policy_identity": behavioral_policy_identity,
            }
        )
    )


def _iter_jsonl(counter: Counters, path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for line in handle:
            counter.bytes_reread += len(line)
            yield counter.parse(line)


def build_projection(
    records: Iterable[Mapping[str, Any]],
    output: Path,
    *,
    authority: Mapping[str, Any],
    capacity: int,
    shape: str,
    contract_version: str = "experimental-contract-vnext",
    behavioral_policy_identity: str = "experimental-behavior-policy-vnext",
) -> BakeoffResult:
    """Build a disposable current-model or standard-candidate projection.

    The iterator is consumed in bounded shards.  The current model alone writes
    the discarded native wrapper and models its reread/rehash/rewrite.  Both
    shapes use the same lean private journal so the comparison isolates the
    disputed staging surfaces rather than duplicating transaction policy.
    """
    if shape not in {"current", "candidate"} or capacity < 1:
        raise ValueError("invalid private bake-off shape or capacity")
    if output.exists():
        raise ValueError("projection output must be fresh")
    started = time.perf_counter()
    counter = Counters()
    authority_digest = counter.digest(counter.canonical(dict(authority)))
    configuration = _configuration(shape, capacity)
    journal_path = output / ".journal.json"
    sealed_log_path = output / ".journal-sealed.jsonl"
    journal = {
        "schema_version": SCHEMA + ".journal.v2",
        "authority_digest": authority_digest,
        "configuration_digest": counter.digest(counter.canonical(configuration)),
        "sealed_log": sealed_log_path.relative_to(output).as_posix(),
        "state": "open",
        "committed_count": 0,
        "committed_end": 0,
    }
    counter.journal_operations += 1
    counter.write(journal_path, counter.canonical(journal), surface="temporary")
    payload_index_path = output / "payload-index.jsonl"
    shard_index_path = output / "shard-index.jsonl"
    counter.write(payload_index_path, b"", surface="final")
    counter.write(shard_index_path, b"", surface="final")

    sequence = _FramedSequence(counter)
    selected_count = 0
    shard_count = 0
    record_iterator = iter(records)
    while chunk := list(itertools.islice(record_iterator, capacity)):
        first = selected_count
        lines: list[bytes] = []
        index_rows: list[dict[str, Any]] = []
        for row, source_record in enumerate(chunk):
            record = dict(source_record)
            encoded = counter.canonical(record)
            sequence.add_encoded(encoded)
            record_digest = counter.digest(encoded)
            if shape == "current":
                native = {
                    "record": record,
                    "payload_hash": counter.digest(
                        counter.canonical({"record": record})
                    ),
                }
                native_path = output / ".native" / f"{first + row:08d}.json"
                counter.write(
                    native_path, counter.canonical(native), surface="temporary"
                )
                reread = counter.read(native_path)
                parsed = counter.parse(reread)
                parsed["payload_hash"] = counter.digest(
                    counter.canonical({"record": parsed["record"]})
                )
                counter.write(
                    native_path,
                    counter.canonical(parsed),
                    surface="temporary",
                    rewrite=True,
                )
            lines.append(encoded + b"\n")
            index_rows.append(
                {
                    "logical_id": _logical_id(record, counter),
                    "selection_index": first + row,
                    "shard_id": shard_count,
                    "row": row,
                    "record_digest": record_digest,
                }
            )
        shard_path = output / "shards" / f"shard-{shard_count:05d}.jsonl"
        shard_bytes = b"".join(lines)
        counter.write(shard_path, shard_bytes, surface="final")
        shard_hash = counter.digest(counter.read(shard_path))
        counter.shard_seals += 1
        entry = {
            "shard_id": shard_count,
            "path": shard_path.relative_to(output).as_posix(),
            "sha256": shard_hash,
            "size_bytes": len(shard_bytes),
            "first": first,
            "count": len(chunk),
        }
        entry_bytes = counter.canonical(entry) + b"\n"
        counter.append(shard_index_path, entry_bytes, surface="final")
        counter.append(sealed_log_path, entry_bytes, surface="temporary")
        journal["committed_count"] = first + len(chunk)
        journal["committed_end"] = first + len(chunk)
        counter.journal_operations += 1
        counter.write(
            journal_path,
            counter.canonical(journal),
            surface="temporary",
            rewrite=True,
        )
        for index_row in index_rows:
            if shape == "current":
                index_row["payload_sha256"] = index_row["record_digest"]
                index_row["payload_semantic_digest"] = index_row["record_digest"]
                index_row["shard_sha256"] = shard_hash
            counter.append(
                payload_index_path,
                counter.canonical(index_row) + b"\n",
                surface="final",
            )
        selected_count += len(chunk)
        shard_count += 1

    sequence_digest = sequence.finish()
    semantic_root = _semantic_root(
        sequence_digest=sequence_digest,
        authority_digest=authority_digest,
        contract_version=contract_version,
        behavioral_policy_identity=behavioral_policy_identity,
        counter=counter,
    )
    layout = {
        "schema_version": SCHEMA + ".public.v2",
        "shape": shape,
        "semantic_authority_identity": authority_digest,
        "contract_version": contract_version,
        "behavioral_policy_identity": behavioral_policy_identity,
        "sequence_digest": sequence_digest,
        "semantic_root": semantic_root,
        "selected_count": selected_count,
        "shard_index": {
            "path": shard_index_path.relative_to(output).as_posix(),
            "sha256": counter.digest_file(shard_index_path),
        },
        "payload_index": {
            "path": payload_index_path.relative_to(output).as_posix(),
            "sha256": counter.digest_file(payload_index_path),
        },
    }
    counter.write(output / "layout.json", counter.canonical(layout), surface="final")
    inventory = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": counter.digest_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if _is_public_member(output, path)
    ]
    counter.write(
        output / "inventory.json", counter.canonical(inventory), surface="final"
    )
    header = {
        "schema_version": SCHEMA + ".manifest.v2",
        "inventory": "inventory.json",
        "inventory_sha256": counter.digest(counter.read(output / "inventory.json")),
        "semantic_root": semantic_root,
    }
    counter.write(
        output / "manifest-header.json", counter.canonical(header), surface="final"
    )
    cover = {
        "schema_version": SCHEMA + ".cover.v2",
        "semantic_authority_identity": authority_digest,
        "contract_version": contract_version,
        "behavioral_policy_identity": behavioral_policy_identity,
        "semantic_root": semantic_root,
        "manifest_header": {
            "path": "manifest-header.json",
            "sha256": counter.digest(counter.read(output / "manifest-header.json")),
        },
    }
    counter.write(output / "cover.json", counter.canonical(cover), surface="final")
    journal["state"] = "complete"
    journal["promotion_marker"] = {
        "semantic_root": semantic_root,
        "sequence_digest": sequence_digest,
        "shard_count": shard_count,
        "selected_count": selected_count,
    }
    counter.journal_operations += 1
    counter.write(
        journal_path,
        counter.canonical(journal),
        surface="temporary",
        rewrite=True,
    )
    archive_started = time.perf_counter()
    archive = output.with_suffix(".tar")
    with tarfile.open(archive, "w") as tar:
        for path in sorted(output.rglob("*")):
            if _is_public_member(output, path):
                tar.add(
                    path, arcname=path.relative_to(output).as_posix(), recursive=False
                )
    archive_seconds = time.perf_counter() - archive_started
    counter.note_written_file(archive, surface="final")
    archive_digest = counter.digest(counter.read(archive))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if os.uname().sysname == "Darwin" else 1024
    )
    return BakeoffResult(
        root=output,
        sequence_digest=sequence_digest,
        semantic_root=semantic_root,
        archive_digest=archive_digest,
        counters=counter.projection(),
        construction_seconds=time.perf_counter() - started,
        archive_seconds=archive_seconds,
        peak_rss_bytes=rss,
        configuration=configuration,
    )


def _public_documents(
    root: Path, counter: Counters
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    cover = counter.parse(counter.read(root / "cover.json"))
    header_path = _member_path(
        root, cover.get("manifest_header", {}).get("path"), label="manifest header"
    )
    header = counter.parse(counter.read(header_path))
    inventory_path = _member_path(root, header.get("inventory"), label="inventory")
    inventory = counter.parse(counter.read(inventory_path))
    layout = counter.parse(counter.read(root / "layout.json"))
    return cover, header, inventory, layout


def validate_standard_projection(
    root: Path, *, authority: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    """Validate one public artifact without touching private producer state."""
    counter = Counters()
    cover, header, inventory, layout = _public_documents(root, counter)
    authority_digest = counter.digest(counter.canonical(dict(authority)))
    if (
        cover.get("semantic_authority_identity") != authority_digest
        or layout.get("semantic_authority_identity") != authority_digest
        or cover.get("semantic_root") != header.get("semantic_root")
        or cover.get("semantic_root") != layout.get("semantic_root")
    ):
        raise ValueError("semantic authority or root mismatch")
    header_path = _member_path(
        root, cover.get("manifest_header", {}).get("path"), label="manifest header"
    )
    inventory_path = _member_path(root, header.get("inventory"), label="inventory")
    if cover["manifest_header"].get("sha256") != counter.digest(
        counter.read(header_path)
    ) or header.get("inventory_sha256") != counter.digest(counter.read(inventory_path)):
        raise ValueError("cover or manifest mismatch")
    inventory_paths = {member["path"] for member in inventory}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if _is_public_member(root, path)
    }
    control_paths = {"cover.json", "manifest-header.json", "inventory.json"}
    if actual_paths - control_paths != inventory_paths:
        raise ValueError("partial or unreceipted public member")
    for member in inventory:
        path = _member_path(root, member.get("path"), label="inventory")
        if member.get("sha256") != counter.digest_file(path):
            raise ValueError(
                f"inventory member mismatch: {member.get('path', '<unknown>')}"
            )
    expected_root = _semantic_root(
        sequence_digest=layout.get("sequence_digest", ""),
        authority_digest=authority_digest,
        contract_version=layout.get("contract_version", ""),
        behavioral_policy_identity=layout.get("behavioral_policy_identity", ""),
        counter=counter,
    )
    if expected_root != cover.get("semantic_root"):
        raise ValueError("semantic-root binding mismatch")
    shard_index_path = _member_path(
        root, layout.get("shard_index", {}).get("path"), label="shard index"
    )
    payload_index_path = _member_path(
        root, layout.get("payload_index", {}).get("path"), label="payload index"
    )
    if layout["shard_index"].get("sha256") != counter.digest_file(
        shard_index_path
    ) or layout["payload_index"].get("sha256") != counter.digest_file(
        payload_index_path
    ):
        raise ValueError("layout index mismatch")
    expected = 0
    sequence = _FramedSequence(counter)
    index_rows = _iter_jsonl(counter, payload_index_path)
    for shard in _iter_jsonl(counter, shard_index_path):
        raw = counter.read(_member_path(root, shard.get("path"), label="shard"))
        if shard.get("sha256") != counter.digest(raw) or shard.get("first") != expected:
            raise ValueError("unsealed, corrupt, or noncontiguous shard")
        rows = [counter.parse(line) for line in raw.splitlines()]
        if len(rows) != shard.get("count"):
            raise ValueError("shard count mismatch")
        for row, record in enumerate(rows):
            try:
                index = next(index_rows)
            except StopIteration as exc:
                raise ValueError("missing payload index row") from exc
            encoded = counter.canonical(record)
            expected_index = {
                "logical_id": _logical_id(record, counter),
                "selection_index": expected + row,
                "shard_id": shard.get("shard_id"),
                "row": row,
                "record_digest": counter.digest(encoded),
            }
            if any(index.get(key) != value for key, value in expected_index.items()):
                raise ValueError("payload index does not bind the shard row")
            sequence.add_encoded(encoded)
        expected += len(rows)
        yield from rows
    if expected != layout.get("selected_count"):
        raise ValueError("payload index count mismatch")
    try:
        next(index_rows)
    except StopIteration:
        pass
    else:
        raise ValueError("excess payload index row")
    if sequence.finish() != layout.get("sequence_digest"):
        raise ValueError("sequence mismatch")


def validate_transaction(
    root: Path,
    *,
    authority: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> None:
    """Validate producer-only resume/promotion state; public consumers skip it."""
    counter = Counters()
    journal = counter.parse(counter.read(root / ".journal.json"))
    authority_digest = counter.digest(counter.canonical(dict(authority)))
    configuration_digest = counter.digest(counter.canonical(dict(configuration)))
    if (
        journal.get("state") != "complete"
        or journal.get("authority_digest") != authority_digest
    ):
        raise ValueError("unsafe or cross-authority journal")
    if journal.get("configuration_digest") != configuration_digest:
        raise ValueError("stale transaction configuration")
    sealed_path = _member_path(root, journal.get("sealed_log"), label="sealed log")
    expected = 0
    sealed_count = 0
    shards = _iter_jsonl(counter, root / "shard-index.jsonl")
    for shard in _iter_jsonl(counter, sealed_path):
        try:
            public_shard = next(shards)
        except StopIteration as exc:
            raise ValueError("unreceipted shard") from exc
        if shard != public_shard:
            raise ValueError("unreceipted shard")
        if shard.get("first") != expected or shard.get("count", 0) < 1:
            raise ValueError("noncontiguous journal range")
        expected += shard["count"]
        sealed_count += 1
    try:
        next(shards)
    except StopIteration:
        pass
    else:
        raise ValueError("unreceipted shard")
    cover = counter.parse(counter.read(root / "cover.json"))
    marker = journal.get("promotion_marker", {})
    if (
        marker.get("semantic_root") != cover.get("semantic_root")
        or marker.get("sequence_digest")
        != counter.parse(counter.read(root / "layout.json")).get("sequence_digest")
        or marker.get("shard_count") != sealed_count
        or marker.get("selected_count") != expected
        or journal.get("committed_count") != expected
        or journal.get("committed_end") != expected
    ):
        raise ValueError("unreceipted shard or incomplete promotion")


def validate_candidate(
    root: Path,
    *,
    authority: Mapping[str, Any],
    configuration: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Compatibility helper for full private-lifecycle experiment validation."""
    if configuration is not None:
        validate_transaction(root, authority=authority, configuration=configuration)
    yield from validate_standard_projection(root, authority=authority)


def compare_immutable_expected_identity(
    root: Path, *, expected_semantic_root: str
) -> None:
    """Model a Golden/Contract comparison against an already trusted identity."""
    cover = json.loads((root / "cover.json").read_text(encoding="utf-8"))
    if cover.get("semantic_root") != expected_semantic_root:
        raise ValueError("immutable expected semantic identity mismatch")


def require_external_attestation(root: Path, *, attestation: Mapping[str, Any]) -> None:
    """Compare a separately obtained expected identity; do not verify signatures.

    The caller is responsible for obtaining and verifying the attestation in a
    distinct trust domain (for example a signed release receipt or transparency
    log).  This local interface intentionally cannot authenticate the issuer.
    """
    cover = json.loads((root / "cover.json").read_text(encoding="utf-8"))
    required = {
        "schema_version": ATTESTATION_SCHEMA,
        "semantic_root": cover.get("semantic_root"),
        "semantic_authority_identity": cover.get("semantic_authority_identity"),
        "contract_version": cover.get("contract_version"),
        "behavioral_policy_identity": cover.get("behavioral_policy_identity"),
    }
    if any(attestation.get(key) != value for key, value in required.items()):
        raise ValueError("external attestation identity mismatch")
    if (
        not isinstance(attestation.get("reference"), str)
        or not attestation["reference"]
    ):
        raise ValueError("external attestation requires an external reference")


def validate_archive(archive: Path, *, expected_digest: str) -> None:
    """Check transport raw identity before extracting an experimental archive."""
    if _file_digest(archive) != expected_digest:
        raise ValueError("archive raw-integrity mismatch")


def summarize(values: list[float]) -> dict[str, float]:
    if len(values) != RUNS:
        raise ValueError("exactly three measurements required")
    ordered = sorted(values)
    return {"median": ordered[1], "spread": ordered[-1] - ordered[0]}


def materially_reduced(baseline: list[float], candidate: list[float]) -> bool:
    left, right = summarize(baseline), summarize(candidate)
    return left["median"] - right["median"] >= max(
        MATERIAL_REDUCTION * left["median"],
        NOISE_MULTIPLIER * ((left["spread"] ** 2 + right["spread"] ** 2) ** 0.5),
    )
