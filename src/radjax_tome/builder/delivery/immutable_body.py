"""Opt-in Contract-governed immutable-body transaction writer.

The body store is content addressed, while the manifest is the semantic
commit point.  Private transaction directories and receipts are never package
inventory members.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_contract.tome.m8g import (
    PROFILE_NAMES,
    CompactBody,
    JournalState,
    _decode_fv3,
    _digest,
    _m8g_fv3,
    body_raw_digest,
    encode_compact_body,
    validate_body_bytes,
    validate_manifest,
    validate_receipt,
)


@dataclass(frozen=True)
class RecoveryAssessment:
    transaction_id: str
    states: tuple[int, ...]
    classification: str
    proposed_action: str
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryPlan:
    transaction_id: str
    classification: str
    actions: tuple[str, ...]


@dataclass(frozen=True)
class PackageFinalizationResult:
    package_path: Path
    package_digest: bytes
    inventory_digest: bytes
    member_count: int
    valid: bool


class ImmutableBodyTransaction:
    """Crash-recoverable, race-resistant writer for one selected exemplar."""

    def __init__(
        self,
        root: Path,
        *,
        profile: str = "producer_evidence",
        fault_boundary: str | None = None,
        configuration_identity: bytes | None = None,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.fault_boundary = fault_boundary
        self.configuration_identity = configuration_identity or _digest(
            b"RDX-M8G-CONFIG-1", profile.encode("utf-8")
        )
        self._reject_symlink_chain(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("bodies", "manifests", ".transactions"):
            path = self.root / name
            self._reject_symlink(path)
            path.mkdir(exist_ok=True)
            if not stat.S_ISDIR(os.lstat(path).st_mode):
                raise ValueError(f"transaction path is not a directory: {name}")

    def commit(
        self,
        body: CompactBody,
        manifest: Mapping[str, Any],
        *,
        canonical_manifest_bytes: bytes,
    ) -> tuple[Path, Path]:
        body_bytes = encode_compact_body(body)
        validate_body_bytes(body_bytes, profile=self.profile)
        closed_manifest = dict(manifest)
        validate_manifest(closed_manifest, body)
        body_digest = body_raw_digest(body_bytes)
        if closed_manifest["body_raw_digest"] != body_digest:
            raise ValueError("manifest body digest does not match body")
        expected_manifest_bytes = _m8g_fv3(closed_manifest)
        if canonical_manifest_bytes != expected_manifest_bytes:
            raise ValueError("manifest bytes do not match validated manifest")

        txid = _digest(
            b"RDX-M8G-TX-1",
            self.configuration_identity
            + body_digest
            + closed_manifest["manifest_semantic_id"],
        ).hex()
        body_path = self.root / "bodies" / f"{body_digest.hex()}.body"
        manifest_path = (
            self.root
            / "manifests"
            / f"{closed_manifest['manifest_semantic_id'].hex()}.manifest"
        )
        lock_path = self.root / ".transactions" / f"{txid}.lock"
        journal_dir = self.root / ".transactions" / txid
        self._preflight_target(body_path, body_bytes)
        self._preflight_target(manifest_path, expected_manifest_bytes)
        self._validate_inventory_file()
        self._reserve(lock_path, txid)
        try:
            self._fault("after_reservation")
            journal_dir.mkdir(exist_ok=True)
            self._fault("after_journal_creation")
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_GENERATING,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            staged_body = journal_dir / "body.tmp"
            # Persist the validated manifest before body publication.  This is
            # private preflight state, not consumer-visible authority, and makes
            # BODY_PROMOTED restartable after a crash.
            staged_manifest = journal_dir / "manifest.tmp"
            self._atomic_write(staged_manifest, expected_manifest_bytes)
            self._atomic_write(staged_body, body_bytes)
            self._fault("after_body_write")
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_WRITTEN,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            validate_body_bytes(staged_body.read_bytes(), profile=self.profile)
            self._fault("after_body_validation")
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_VALIDATED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_DIGESTED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._atomic_write(body_path, body_bytes)
            self._fault("after_body_publication")
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_PROMOTED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.MANIFEST_PROVISIONAL,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.LINKAGE_FINALIZED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.MANIFEST_VALIDATED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._atomic_write(manifest_path, expected_manifest_bytes)
            self._fault("after_manifest_publication")
            manifest_digest = hashlib.sha256(expected_manifest_bytes).digest()
            self._receipt(
                journal_dir,
                txid,
                JournalState.MANIFEST_PROMOTED,
                body_digest,
                manifest_digest,
                closed_manifest,
                len(body_bytes),
            )
            self._bind_inventory(body_path, manifest_path, body_digest, manifest_digest)
            self._fault("after_inventory_mutation")
            package = self._finalize_package(body_path, manifest_path)
            if not package.valid:
                raise ValueError("package finalization failed")
            self._receipt(
                journal_dir,
                txid,
                JournalState.INVENTORY_COMMITTED,
                body_digest,
                manifest_digest,
                closed_manifest,
                len(body_bytes),
            )
            self._fault("after_package_validation")
            self._receipt(
                journal_dir,
                txid,
                JournalState.PACKAGE_VALIDATED,
                body_digest,
                manifest_digest,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.COMMITTED,
                body_digest,
                manifest_digest,
                closed_manifest,
                len(body_bytes),
            )
            staged_body.unlink(missing_ok=True)
            staged_manifest.unlink(missing_ok=True)
            self._fault("during_cleanup")
            return body_path, manifest_path
        finally:
            self._release(lock_path, owner=txid)

    def _fault(self, boundary: str) -> None:
        """Deterministic test-only crash boundary; disabled by default."""
        if self.fault_boundary == boundary:
            raise RuntimeError(f"fault injected at {boundary}")

    def _finalize_package(
        self, body_path: Path, manifest_path: Path
    ) -> PackageFinalizationResult:
        """Build and validate the opt-in transaction package boundary.

        This deliberately does not pretend the transaction root is a complete
        public Tome artifact.  It is a typed, deterministic package of the
        committed body/manifest pair plus the canonical inventory, suitable for
        recovery and archive integrity checks before the enclosing Tome package
        builder consumes it.
        """
        inventory_path = self.root / "inventory.json"
        inventory_bytes = inventory_path.read_bytes()
        inventory_digest = hashlib.sha256(inventory_bytes).digest()
        package_dir = self.root / "packages"
        self._reject_symlink_chain(package_dir)
        package_dir.mkdir(exist_ok=True)
        package_path = package_dir / f"{manifest_path.stem}.tgz"
        temp = package_path.with_name(f".{package_path.name}.tmp")
        self._reject_symlink(temp)
        members = [body_path, manifest_path, inventory_path]
        with temp.open("wb") as output:
            with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", dereference=False
                ) as archive:
                    for member in sorted(members, key=lambda p: p.name):
                        info = archive.gettarinfo(str(member), arcname=member.name)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mode = 0o644
                        with member.open("rb") as handle:
                            archive.addfile(info, handle)
        package_bytes = temp.read_bytes()
        self._atomic_write(package_path, package_bytes)
        temp.unlink(missing_ok=True)
        self._fault("after_archive_construction")
        with tarfile.open(package_path, "r:gz") as archive:
            names = archive.getnames()
            expected = {p.name for p in members}
            if set(names) != expected:
                raise ValueError("package member inventory mismatch")
            inventory = json.loads(inventory_bytes)
            member_map = inventory.get("members", {})
            if manifest_path.name not in member_map:
                raise ValueError("package manifest absent from inventory")
            inventory_member = member_map[manifest_path.name]
            for name in names:
                handle = archive.extractfile(name)
                if handle is None:
                    raise ValueError("package member unreadable")
                data = handle.read()
                if name == body_path.name:
                    expected_digest = inventory_member["body_raw_digest"]
                    actual_digest = body_raw_digest(data).hex()
                elif name == manifest_path.name:
                    expected_digest = inventory_member["manifest_raw_digest"]
                    actual_digest = hashlib.sha256(data).hexdigest()
                else:
                    expected_digest = None
                    actual_digest = None
                if expected_digest is not None and actual_digest != expected_digest:
                    raise ValueError("package member digest mismatch")
        self._fault("after_archive_validation")
        return PackageFinalizationResult(
            package_path=package_path,
            package_digest=hashlib.sha256(package_path.read_bytes()).digest(),
            inventory_digest=inventory_digest,
            member_count=len(members),
            valid=True,
        )

    def _bind_inventory(
        self,
        body_path: Path,
        manifest_path: Path,
        body_digest: bytes,
        manifest_digest: bytes,
    ) -> None:
        inventory_path = self.root / "inventory.json"
        if inventory_path.exists() and (
            stat.S_ISLNK(os.lstat(inventory_path).st_mode)
            or not stat.S_ISREG(os.lstat(inventory_path).st_mode)
        ):
            raise ValueError("inventory path type invalid")
        inventory: dict[str, Any] = {"schema": "m8g_inventory_v1", "members": {}}
        if inventory_path.exists():
            inventory = json.loads(inventory_path.read_text())
            if inventory.get("schema") != "m8g_inventory_v1" or not isinstance(
                inventory.get("members"), dict
            ):
                raise ValueError("inventory schema invalid")
        inventory.setdefault("members", {})[manifest_path.name] = {
            "body": body_path.name,
            "body_raw_digest": body_digest.hex(),
            "manifest_raw_digest": manifest_digest.hex(),
        }
        encoded = json.dumps(inventory, sort_keys=True).encode("utf-8")
        self._atomic_replace(inventory_path, encoded)
        if json.loads(inventory_path.read_text()) != inventory:
            raise ValueError("inventory binding validation failed")

    def _validate_inventory_file(self) -> None:
        path = self.root / "inventory.json"
        if not path.exists():
            return
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError("inventory path type invalid")
        inventory = json.loads(path.read_text())
        if inventory.get("schema") != "m8g_inventory_v1" or not isinstance(
            inventory.get("members"), dict
        ):
            raise ValueError("inventory schema invalid")

    def recover(self) -> list[dict[str, Any]]:
        """Validate journal chains and quarantine incomplete transactions."""
        results = []
        tx_root = self.root / ".transactions"
        for path in sorted(tx_root.iterdir()):
            if path.name.endswith(".lock") or path.name.startswith(".quarantine-"):
                continue
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError("symlink or non-directory transaction entry")
            valid = True
            try:
                receipts = self._receipt_json_paths(path)
            except (OSError, ValueError, TypeError):
                receipts = sorted(path.glob("receipt-*.json"))
                valid = False
            states = []
            for receipt in receipts:
                try:
                    mode = os.lstat(receipt).st_mode
                    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                        raise ValueError("receipt file type invalid")
                    decoded = json.loads(receipt.read_text())
                    for key in (
                        "body_raw_digest",
                        "manifest_raw_digest",
                        "configuration_identity",
                        "semantic_authority_identity",
                        "receipt_digest",
                    ):
                        if isinstance(decoded.get(key), str) and decoded[key]:
                            decoded[key] = bytes.fromhex(decoded[key])
                    validate_receipt(decoded)
                    if decoded["configuration_identity"] != self.configuration_identity:
                        raise ValueError("receipt configuration binding invalid")
                    if decoded["transaction_id"] != path.name:
                        raise ValueError("receipt transaction binding invalid")
                    binary = receipt.with_suffix(".bin")
                    if binary.is_symlink() or binary.read_bytes() != _m8g_fv3(decoded):
                        raise ValueError("receipt binary/json mismatch")
                    if decoded["state"] >= int(JournalState.BODY_PROMOTED):
                        body_file = self._owned_relative(decoded["body_path"])
                        if (
                            body_file.is_symlink()
                            or body_raw_digest(body_file.read_bytes())
                            != decoded["body_raw_digest"]
                        ):
                            raise ValueError("receipt body binding invalid")
                    if decoded["state"] >= int(JournalState.MANIFEST_PROMOTED):
                        manifest_file = self._owned_relative(decoded["manifest_path"])
                        if (
                            manifest_file.is_symlink()
                            or hashlib.sha256(manifest_file.read_bytes()).digest()
                            != decoded["manifest_raw_digest"]
                        ):
                            raise ValueError("receipt manifest binding invalid")
                    states.append(int(decoded["state"]))
                except (OSError, ValueError, KeyError, TypeError):
                    valid = False
                    states.append(None)
            if states != list(range(1, len(states) + 1)):
                valid = False
            if valid and not states:
                for temporary in path.glob("*.tmp"):
                    if temporary.is_symlink() or not stat.S_ISREG(
                        os.lstat(temporary).st_mode
                    ):
                        valid = False
                    else:
                        temporary.unlink()
                if valid:
                    self._release(tx_root / f"{path.name}.lock", owner=path.name)
                    results.append(
                        {
                            "transaction_id": path.name,
                            "receipt_count": 0,
                            "states": [],
                            "status": "restart_ready",
                            "staging": path.name,
                        }
                    )
                    continue
            if valid and states and states[-1] < int(JournalState.BODY_PROMOTED):
                for temporary in path.glob("*.tmp"):
                    if not temporary.is_symlink():
                        temporary.unlink()
                self._release(tx_root / f"{path.name}.lock")
                results.append(
                    {
                        "transaction_id": path.name,
                        "receipt_count": len(receipts),
                        "states": states,
                        "status": "restart_ready",
                        "staging": path.name,
                    }
                )
                continue
            if (
                valid
                and states
                and states[-1] == int(JournalState.BODY_PROMOTED)
                and not (path / "manifest.tmp").exists()
            ):
                body_receipt = json.loads(receipts[-1].read_text())
                body_file = self._owned_relative(body_receipt["body_path"])
                if self._inventory_references_body(body_file.name):
                    valid = False
                elif not body_file.is_symlink():
                    body_file.unlink(missing_ok=True)
                if not valid:
                    quarantine = tx_root / f".quarantine-{path.name}"
                    if not quarantine.exists():
                        os.replace(path, quarantine)
                    results.append(
                        {
                            "transaction_id": path.name,
                            "receipt_count": len(receipts),
                            "states": states,
                            "status": "quarantined",
                            "staging": path.name,
                        }
                    )
                    continue
                self._release(tx_root / f"{path.name}.lock")
                results.append(
                    {
                        "transaction_id": path.name,
                        "receipt_count": len(receipts),
                        "states": states,
                        "status": "restart_ready",
                        "staging": path.name,
                    }
                )
                continue
            if valid and states and states[-1] < int(JournalState.MANIFEST_PROMOTED):
                try:
                    self._resume_manifest(path, receipts, states[-1])
                    results.append(
                        {
                            "transaction_id": path.name,
                            "receipt_count": 12,
                            "states": list(range(1, 13)),
                            "status": "resumed",
                            "staging": path.name,
                        }
                    )
                    continue
                except (OSError, ValueError, KeyError, TypeError):
                    valid = False
            if (
                valid
                and states
                and int(JournalState.MANIFEST_PROMOTED)
                <= states[-1]
                < int(JournalState.COMMITTED)
            ):
                try:
                    self._resume_inventory(path, receipts)
                    results.append(
                        {
                            "transaction_id": path.name,
                            "receipt_count": 12,
                            "states": list(range(1, 13)),
                            "status": "resumed",
                            "staging": path.name,
                        }
                    )
                    continue
                except (OSError, ValueError, KeyError, TypeError):
                    valid = False
            if valid and states == list(range(1, 13)):
                try:
                    inventory = json.loads((self.root / "inventory.json").read_text())
                    if inventory.get("schema") != "m8g_inventory_v1":
                        valid = False
                    manifest_names = {
                        Path(json.loads(item.read_text()).get("manifest_path", "")).name
                        for item in receipts
                        if json.loads(item.read_text()).get("manifest_path")
                    }
                    if not all(
                        name in inventory.get("members", {})
                        for name in manifest_names
                        if name
                    ):
                        valid = False
                    for item in receipts:
                        receipt = json.loads(item.read_text())
                        if receipt.get("state", 0) < int(
                            JournalState.MANIFEST_PROMOTED
                        ):
                            continue
                        name = Path(receipt.get("manifest_path", "")).name
                        member = inventory.get("members", {}).get(name, {})
                        body_digest_text = receipt.get("body_raw_digest")
                        manifest_digest_text = receipt.get("manifest_raw_digest")
                        if isinstance(body_digest_text, bytes):
                            body_digest_text = body_digest_text.hex()
                        if isinstance(manifest_digest_text, bytes):
                            manifest_digest_text = manifest_digest_text.hex()
                        if (
                            set(member)
                            != {"body", "body_raw_digest", "manifest_raw_digest"}
                            or member.get("body")
                            != Path(receipt.get("body_path", "")).name
                            or member.get("body_raw_digest") != body_digest_text
                            or member.get("manifest_raw_digest") != manifest_digest_text
                        ):
                            valid = False
                    if valid:
                        for temporary in path.glob("*.tmp"):
                            if temporary.is_symlink() or not stat.S_ISREG(
                                os.lstat(temporary).st_mode
                            ):
                                valid = False
                            else:
                                temporary.unlink()
                        self._release(tx_root / f"{path.name}.lock", owner=path.name)
                except (OSError, ValueError, TypeError):
                    valid = False
            if valid and states == list(range(1, 13)):
                status = "committed"
            else:
                quarantine = tx_root / f".quarantine-{path.name}"
                if not quarantine.exists():
                    os.replace(path, quarantine)
                status = "quarantined"
            results.append(
                {
                    "transaction_id": path.name,
                    "receipt_count": len(receipts),
                    "states": states,
                    "status": status,
                    "cleanup": "complete" if status == "committed" else "not_performed",
                    "staging": path.name,
                }
            )
        return results

    def inspect(self) -> tuple[RecoveryAssessment, ...]:
        """Read-only assessment of every private transaction journal."""
        assessments: list[RecoveryAssessment] = []
        for path in sorted((self.root / ".transactions").iterdir()):
            if path.name.endswith(".lock") or path.name.startswith(".quarantine-"):
                continue
            try:
                receipts = self._receipt_json_paths(path)
                states = tuple(
                    int(json.loads(p.read_text())["state"]) for p in receipts
                )
                if states == tuple(range(1, 13)):
                    classification, action = (
                        "COMPLETE_NEEDS_CLEANUP",
                        "validate_and_cleanup",
                    )
                elif states and states[-1] >= int(JournalState.MANIFEST_PROMOTED):
                    classification, action = (
                        "RECOVERABLE_RESUME",
                        "bind_inventory_and_package",
                    )
                elif states:
                    classification, action = (
                        "RECOVERABLE_RESTART",
                        "clean_private_stage_and_restart",
                    )
                else:
                    classification, action = "ACTIVE_PARTIAL", "restart"
                assessments.append(
                    RecoveryAssessment(path.name, states, classification, action)
                )
            except (OSError, ValueError, TypeError):
                assessments.append(
                    RecoveryAssessment(path.name, (), "CORRUPT", "quarantine")
                )
        return tuple(assessments)

    def plan(self, assessment: RecoveryAssessment) -> RecoveryPlan:
        actions = {
            "RECOVERABLE_RESTART": (
                "validate_ownership",
                "remove_private_partials",
                "restart_transaction",
            ),
            "RECOVERABLE_RESUME": (
                "validate_body_manifest",
                "bind_inventory",
                "finalize_package",
                "append_receipts",
            ),
            "COMPLETE_NEEDS_CLEANUP": (
                "validate_complete_chain",
                "remove_private_partials",
                "release_reservation",
            ),
            "CORRUPT": ("quarantine_forensic_copy",),
        }.get(assessment.classification, ("validate",))
        return RecoveryPlan(
            assessment.transaction_id, assessment.classification, actions
        )

    def _receipt_json_paths(self, journal_dir: Path) -> list[Path]:
        """Return mirrors after validating binary receipts as authority.

        A missing JSON mirror is regenerated from its binary receipt.  A
        missing binary receipt is unverifiable and is deliberately left for
        the caller to quarantine.
        """
        paths: list[Path] = []
        for binary in sorted(journal_dir.glob("receipt-*.bin")):
            self._reject_symlink(binary)
            payload = binary.read_bytes()
            decoded, end = _decode_fv3(payload, 0)
            if end != len(payload) or not isinstance(decoded, Mapping):
                raise ValueError("receipt binary decode invalid")
            mirror = binary.with_suffix(".json")
            if not mirror.exists():
                serializable = {
                    key: value.hex() if isinstance(value, bytes) else value
                    for key, value in decoded.items()
                }
                self._atomic_write(
                    mirror, json.dumps(serializable, sort_keys=True).encode("utf-8")
                )
            paths.append(mirror)
        json_only = [
            p
            for p in journal_dir.glob("receipt-*.json")
            if not p.with_suffix(".bin").exists()
        ]
        if json_only:
            raise ValueError("receipt binary missing")
        return paths

    def _resume_inventory(self, journal_dir: Path, receipts: list[Path]) -> None:
        latest = json.loads(receipts[-1].read_text())
        body_path = self._owned_relative(latest["body_path"])
        manifest_path = self._owned_relative(latest["manifest_path"])
        body = validate_body_bytes(body_path.read_bytes(), profile=self.profile)
        manifest_bytes = manifest_path.read_bytes()
        decoded, end = _decode_fv3(manifest_bytes, 0)
        if end != len(manifest_bytes) or not isinstance(decoded, Mapping):
            raise ValueError("committed manifest decode invalid")
        manifest = dict(decoded)
        validate_manifest(manifest, body)
        body_digest = body_raw_digest(body_path.read_bytes())
        manifest_digest = hashlib.sha256(manifest_bytes).digest()
        self._bind_inventory(body_path, manifest_path, body_digest, manifest_digest)
        package = self._finalize_package(body_path, manifest_path)
        if not package.valid:
            raise ValueError("package finalization failed")
        txid = latest["transaction_id"]
        for state in (
            JournalState.INVENTORY_COMMITTED,
            JournalState.PACKAGE_VALIDATED,
            JournalState.COMMITTED,
        ):
            if int(state) > int(latest["state"]):
                self._receipt(
                    journal_dir,
                    txid,
                    state,
                    body_digest,
                    manifest_digest,
                    manifest,
                    len(body_path.read_bytes()),
                )

    def _inventory_references_body(self, body_name: str) -> bool:
        path = self.root / "inventory.json"
        if not path.exists():
            return False
        try:
            value = json.loads(path.read_text())
            return any(
                member.get("body") == body_name
                for member in value.get("members", {}).values()
            )
        except (OSError, ValueError, TypeError, AttributeError):
            return True

    def _resume_manifest(
        self, journal_dir: Path, receipts: list[Path], highest: int
    ) -> None:
        latest = json.loads(receipts[-1].read_text())
        body_path = self._owned_relative(latest["body_path"])
        body = validate_body_bytes(body_path.read_bytes(), profile=self.profile)
        staged = journal_dir / "manifest.tmp"
        self._reject_symlink(staged)
        manifest_bytes = staged.read_bytes()
        decoded, end = _decode_fv3(manifest_bytes, 0)
        if end != len(manifest_bytes) or not isinstance(decoded, Mapping):
            raise ValueError("staged manifest decode invalid")
        manifest = dict(decoded)
        validate_manifest(manifest, body)
        target = (
            self.root
            / "manifests"
            / f"{manifest['manifest_semantic_id'].hex()}.manifest"
        )
        self._preflight_target(target, manifest_bytes)
        self._atomic_write(target, manifest_bytes)
        body_digest = body_raw_digest(body_path.read_bytes())
        manifest_digest = hashlib.sha256(manifest_bytes).digest()
        txid = latest["transaction_id"]
        for state in JournalState:
            if (
                state <= JournalState.MANIFEST_PROMOTED
                or state <= JournalState.COMMITTED
            ):
                if int(state) <= highest:
                    continue
                if state == JournalState.MANIFEST_PROMOTED:
                    self._receipt(
                        journal_dir,
                        txid,
                        state,
                        body_digest,
                        manifest_digest,
                        manifest,
                        len(body_path.read_bytes()),
                    )
                    self._bind_inventory(
                        body_path, target, body_digest, manifest_digest
                    )
                elif state == JournalState.INVENTORY_COMMITTED:
                    self._receipt(
                        journal_dir,
                        txid,
                        state,
                        body_digest,
                        manifest_digest,
                        manifest,
                        len(body_path.read_bytes()),
                    )
                elif state == JournalState.PACKAGE_VALIDATED:
                    self._receipt(
                        journal_dir,
                        txid,
                        state,
                        body_digest,
                        manifest_digest,
                        manifest,
                        len(body_path.read_bytes()),
                    )
                elif state == JournalState.COMMITTED:
                    self._receipt(
                        journal_dir,
                        txid,
                        state,
                        body_digest,
                        manifest_digest,
                        manifest,
                        len(body_path.read_bytes()),
                    )
        staged.unlink(missing_ok=True)

    def _receipt(
        self,
        directory: Path,
        txid: str,
        state: JournalState,
        body_digest: bytes,
        manifest_digest: bytes | None,
        manifest: Mapping[str, Any],
        body_size: int,
    ) -> None:
        manifest_name = f"manifests/{manifest['manifest_semantic_id'].hex()}.manifest"
        receipt: dict[str, Any] = {
            "transaction_id": txid,
            "schema_version": "radjax_contract_m8g_v1",
            "profile_code": PROFILE_NAMES[manifest["profile"]],
            "state": int(state),
            "parent_transaction_id": None,
            "body_path": f"bodies/{body_digest.hex()}.body"
            if state >= JournalState.BODY_PROMOTED
            else None,
            "manifest_path": manifest_name
            if state >= JournalState.MANIFEST_PROMOTED
            else None,
            "body_raw_digest": body_digest
            if state >= JournalState.BODY_PROMOTED
            else None,
            "body_size_bytes": body_size
            if state >= JournalState.BODY_PROMOTED
            else None,
            "manifest_raw_digest": manifest_digest,
            "committed_next_state": None
            if state == JournalState.COMMITTED
            else int(state) + 1,
            "configuration_identity": self.configuration_identity,
            "semantic_authority_identity": manifest["selection_authority_id"],
            "receipt_digest": b"",
        }
        receipt["receipt_digest"] = _digest(
            b"RDX-RECEIPT-1",
            _m8g_fv3({k: v for k, v in receipt.items() if k != "receipt_digest"}),
        )
        validate_receipt(receipt)
        self._atomic_write(
            directory / f"receipt-{int(state):02d}.bin", _m8g_fv3(receipt)
        )
        self._fault(f"after_receipt_{int(state):02d}_binary")
        serializable = {
            key: value.hex() if isinstance(value, bytes) else value
            for key, value in receipt.items()
        }
        self._atomic_write(
            directory / f"receipt-{int(state):02d}.json",
            json.dumps(serializable, sort_keys=True).encode("utf-8"),
        )
        self._fault(f"after_receipt_{int(state):02d}_json")

    def _owned_relative(self, value: Any) -> Path:
        if (
            not isinstance(value, str)
            or not value
            or Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise ValueError("receipt path escapes transaction root")
        candidate = self.root / value
        self._reject_symlink_chain(candidate.parent)
        if candidate.is_symlink():
            raise ValueError("receipt resource symlink rejected")
        return candidate

    @staticmethod
    def _reject_symlink(path: Path) -> None:
        if path.exists() and stat.S_ISLNK(os.lstat(path).st_mode):
            raise ValueError(f"symlink path rejected: {path}")

    @staticmethod
    def _reject_symlink_chain(path: Path) -> None:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            current /= component
            if str(current) in {"/tmp", "/var", "/private", "/private/var"}:
                continue
            if current.exists() and stat.S_ISLNK(os.lstat(current).st_mode):
                raise ValueError(f"symlink parent rejected: {current}")

    @staticmethod
    def _preflight_target(path: Path, expected: bytes) -> None:
        ImmutableBodyTransaction._reject_symlink_chain(path.parent)
        if not path.exists():
            return
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(f"unsafe immutable target: {path.name}")
        if path.read_bytes() != expected:
            raise ValueError(f"immutable resource conflict: {path.name}")

    @staticmethod
    def _reserve(path: Path, txid: str) -> None:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ValueError(f"transaction reservation conflict: {txid}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(txid)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _release(path: Path, *, owner: str | None = None) -> None:
        try:
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                return
            if owner is not None and path.read_text(encoding="utf-8") != owner:
                return
            path.unlink()
        except FileNotFoundError:
            return

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        ImmutableBodyTransaction._preflight_target(path, data)
        if path.exists():
            return
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                ImmutableBodyTransaction._preflight_target(path, data)
            else:
                os.unlink(temporary)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.lexists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _atomic_replace(path: Path, data: bytes) -> None:
        ImmutableBodyTransaction._reject_symlink_chain(path.parent)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.lexists(temporary):
                os.unlink(temporary)


__all__ = ["ImmutableBodyTransaction"]
