"""Materialize the explicit v5 generic language/tokenizer binding package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from radjax_tome.io.json import read_json_object

SOURCE_BINDING_PATH = "language_tokenizer_binding_v1.json"
MANIFEST_PATH = "manifests/language_tokenizer_binding_v1.json"


@dataclass(frozen=True)
class LanguageTokenizerBindingV5Materialization:
    """V5 binding facts materialized into an otherwise ordinary package."""

    root: Path
    manifest_path: str
    canonical_binding_digest: str


def materialize_language_tokenizer_binding_v5(
    artifact_root: Path,
    *,
    destination_root: Path,
) -> LanguageTokenizerBindingV5Materialization:
    """Copy only a Contract-admitted source capture into the package staging root.

    The source capture is required to have been written by the tokenizer that
    generated the source payloads.  Package construction additionally proves
    that every source token ID fits the captured exact vocabulary domain.
    """

    source = Path(artifact_root)
    destination = Path(destination_root)
    source_binding = source / SOURCE_BINDING_PATH
    if not source_binding.is_file():
        raise ValueError("native v5 package requires tokenizer binding capture")
    binding = read_json_object(source_binding)
    _validate_source_binding_with_contract(source_binding)
    vocabulary = binding.get("vocabulary")
    if not isinstance(vocabulary, dict) or not isinstance(
        vocabulary.get("vocabulary_size"), int
    ):
        raise ValueError("native v5 source capture has no vocabulary size")
    vocab_size = int(vocabulary["vocabulary_size"])
    _validate_payload_token_domain(source, vocab_size)

    inventory = binding.get("behavior_content_inventory")
    if not isinstance(inventory, list):
        raise ValueError("native v5 source capture has no behavior inventory")
    for item in inventory:
        if not isinstance(item, dict) or not isinstance(
            item.get("inventory_binding"), str
        ):
            raise ValueError("native v5 source capture has an invalid inventory")
        relative = str(item["inventory_binding"])
        if not _safe_relative_path(relative):
            raise ValueError("native v5 source capture has an unsafe inventory path")
        source_resource = source / relative
        if not source_resource.is_file():
            raise ValueError(
                "native v5 source capture is missing an inventory resource"
            )
        destination_resource = destination / relative
        destination_resource.parent.mkdir(parents=True, exist_ok=True)
        destination_resource.write_bytes(source_resource.read_bytes())

    manifest = destination / MANIFEST_PATH
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(source_binding.read_bytes())
    digest = binding.get("canonical_binding_digest")
    if not isinstance(digest, str):
        raise ValueError("native v5 source capture has no canonical binding digest")
    return LanguageTokenizerBindingV5Materialization(
        root=destination,
        manifest_path=MANIFEST_PATH,
        canonical_binding_digest=digest,
    )


def _validate_source_binding_with_contract(binding: Path) -> None:
    """Use the release validator, never a metadata-only local approximation."""

    from radjax_contract.tome import validate_and_resolve_language_tokenizer_binding

    result = validate_and_resolve_language_tokenizer_binding(binding, strict=True)
    if not result.ok:
        codes = ",".join(issue.code for issue in result.issues)
        raise ValueError(f"Contract v5 source binding validation failed: {codes}")


def _validate_payload_token_domain(source: Path, vocabulary_size: int) -> None:
    if vocabulary_size < 1:
        raise ValueError("native v5 vocabulary size must be positive")
    metadata = read_json_object(source / "metadata.json")
    vocab_contract = read_json_object(source / "vocab_contract.json")
    if (
        metadata.get("vocab_size") != vocabulary_size
        or vocab_contract.get("vocab_size") != vocabulary_size
    ):
        raise ValueError("native v5 binding vocabulary disagrees with source payload")
    shards = sorted((source / "shards").glob("shard-*.npz"))
    if not shards:
        raise ValueError("native v5 source payload has no shards")
    for shard_path in shards:
        with np.load(shard_path, allow_pickle=False) as shard:
            if "input_ids" not in shard.files:
                raise ValueError("native v5 source shard has no input_ids")
            input_ids = np.asarray(shard["input_ids"])
        if not np.issubdtype(input_ids.dtype, np.integer):
            raise ValueError("native v5 source input_ids must be integer")
        if input_ids.size and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= vocabulary_size
        ):
            raise ValueError("native v5 source token IDs exceed captured vocabulary")


def _safe_relative_path(value: str) -> bool:
    pure = PurePosixPath(value)
    return (
        bool(value)
        and not pure.is_absolute()
        and ".." not in pure.parts
        and pure.as_posix() == value
    )


__all__ = [
    "LanguageTokenizerBindingV5Materialization",
    "MANIFEST_PATH",
    "SOURCE_BINDING_PATH",
    "materialize_language_tokenizer_binding_v5",
]
