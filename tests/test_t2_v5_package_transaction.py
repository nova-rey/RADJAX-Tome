from __future__ import annotations

import hashlib
import io
import json
import tarfile
from gzip import GzipFile
from pathlib import Path

import numpy as np
import pytest
from radjax_contract.tome import streaming_validation as m7_validation
from radjax_contract.tome import validate_and_resolve_student_consumption
from radjax_contract.tome.streaming_validation import validate_streaming_tome

from radjax_tome.tome import (
    pack_sharded_tome_v4,
    package_legacy_artifact_as_sharded_tome_v4,
    package_tome_artifact,
)
from radjax_tome.tome.archive_compat import safe_extractall
from tests.test_tome_packaging_profiles import _artifact


def _v5_source(tmp_path: Path) -> Path:
    return _artifact(
        tmp_path / "source",
        teacher_backend="smoke_tokenizer",
        vocab_size=512,
        top_k=4,
    )


def _mutate_m7_inner_exemplar(archive_path: Path, destination: Path) -> Path:
    """Rebuild a self-consistent M7 archive with v6-invalid exemplar semantics."""
    root = destination.parent / "mutated-m7"
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extractall(archive, root)
    shard = root / "selected_exemplars/shards/shard-00000.jsonl"
    records = [json.loads(line) for line in shard.read_text().splitlines()]
    records[0]["top_probs"][0] = 0.0
    shard.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))

    def digest(path: Path) -> tuple[str, int]:
        return (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
        )

    shard_digest, shard_size = digest(shard)
    index = root / "selected_exemplars/payload-index.jsonl"
    index_rows = [json.loads(line) for line in index.read_text().splitlines()]
    sequence = m7_validation._SequenceDigest()
    for row, record in zip(index_rows, records, strict=True):
        logical_id, semantic = m7_validation._semantic_record(record)
        row.update(
            {
                "logical_id": logical_id,
                "payload_sha256": m7_validation._canonical(record),
                "payload_semantic_digest": semantic,
                "shard_sha256": shard_digest,
            }
        )
        sequence.add({"logical_id": logical_id, "payload_semantic_digest": semantic})
    sequence_digest = sequence.finish()
    index.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows)
    )
    index_digest, index_size = digest(index)
    shard_index = root / "selected_exemplars/payload-shards.jsonl"
    shard_row = json.loads(shard_index.read_text())
    shard_row.update(
        {
            "sha256": shard_digest,
            "size_bytes": shard_size,
            "semantic_digest": sequence_digest,
        }
    )
    shard_index.write_text(json.dumps(shard_row, sort_keys=True) + "\n")
    shard_index_digest, shard_index_size = digest(shard_index)
    layout = root / "selected_exemplars/payload-layout.json"
    layout_doc = json.loads(layout.read_text())
    layout_doc["sequence_digest"] = sequence_digest
    layout_doc["payload_index"].update(
        {"sha256": index_digest, "size_bytes": index_size}
    )
    layout_doc["shard_index"].update(
        {"sha256": shard_index_digest, "size_bytes": shard_index_size}
    )
    layout.write_text(json.dumps(layout_doc, sort_keys=True))
    cover = root / "cover_page.json"
    cover_doc = json.loads(cover.read_text())
    cover_doc["identity"]["payload_sequence_digest"] = sequence_digest
    cover_doc["identity"]["semantic_digest"] = m7_validation._canonical(
        {
            key: value
            for key, value in cover_doc["identity"].items()
            if key != "semantic_digest"
        }
    )
    cover.write_text(json.dumps(cover_doc, sort_keys=True))
    inventory = root / "manifests/content-manifest-inventory.jsonl"
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    for row in rows:
        row["sha256"], row["size_bytes"] = digest(root / row["path"])
    inventory.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    inventory_digest, inventory_size = digest(inventory)
    header = root / "manifests/content-manifest-header.json"
    header_doc = json.loads(header.read_text())
    header_doc.update(
        {
            "inventory_sha256": inventory_digest,
            "inventory_size_bytes": inventory_size,
            "semantic_identity_digest": cover_doc["identity"]["semantic_digest"],
        }
    )
    header.write_text(json.dumps(header_doc, sort_keys=True))
    header_digest, header_size = digest(header)
    cover_doc["manifests"]["header"].update(
        {"sha256": header_digest, "size_bytes": header_size}
    )
    cover.write_text(json.dumps(cover_doc, sort_keys=True))
    with (
        destination.open("wb") as raw,
        GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gzip,
        tarfile.open(fileobj=gzip, mode="w") as out,
    ):
        ordered = [cover, header, inventory, *(root / row["path"] for row in rows)]
        for path in ordered:
            payload = path.read_bytes()
            member = tarfile.TarInfo(path.relative_to(root).as_posix())
            member.size, member.mtime, member.mode = len(payload), 0, 0o644
            out.addfile(member, io.BytesIO(payload))
    return destination


@pytest.mark.parametrize("archive", ["none", "tgz"])
def test_t2_v5_package_admits_binding_before_directory_or_archive_promotion(
    tmp_path: Path, archive: str
) -> None:
    source = _v5_source(tmp_path)
    output = tmp_path / ("student.tgz" if archive == "tgz" else "student")

    result = package_tome_artifact(
        source,
        output,
        profile="student",
        archive=archive,
    )

    assert result.output_path == output
    if archive == "none":
        admitted = validate_and_resolve_student_consumption(
            output,
            profile_id="native_v3_student_v5",
            strict=True,
        )
        assert admitted.ok, admitted.issues
        assert (output / "manifests/language_tokenizer_binding_v1.json").is_file()
        assert (output / "resources/tokenizer_vocabulary.jsonl").is_file()
        assert (output / "cover_page.json").read_text(encoding="utf-8").find(
            "native_v3_student_v5"
        ) >= 0
    else:
        # Contract v5 admits the safely extracted archive candidate inside the
        # producer transaction before this public archive exists.
        assert output.is_file()


@pytest.mark.parametrize("archive", ["none", "tgz"])
def test_t2_missing_capture_is_a_production_path_no_promotion_failure(
    tmp_path: Path, archive: str
) -> None:
    source = _artifact(tmp_path / "source")
    output = tmp_path / ("student.tgz" if archive == "tgz" else "student")

    with pytest.raises(ValueError, match="requires tokenizer binding capture"):
        package_tome_artifact(source, output, profile="student", archive=archive)

    assert not output.exists()


def test_t2_v4_remains_available_only_by_explicit_historical_selection(
    tmp_path: Path,
) -> None:
    source = _artifact(tmp_path / "source")
    output = tmp_path / "student-v4"

    package_tome_artifact(
        source,
        output,
        profile="student",
        student_contract_profile="v4",
    )

    admitted = validate_and_resolve_student_consumption(
        output,
        profile_id="native_v3_student_v4",
        strict=True,
    )
    assert admitted.ok, admitted.issues


@pytest.mark.parametrize("archive", ["none", "tgz"])
def test_b3_v6_composes_the_v5_binding_without_changing_default(
    tmp_path: Path, archive: str
) -> None:
    source = _v5_source(tmp_path)
    output = tmp_path / ("student-v6.tgz" if archive == "tgz" else "student-v6")

    package_tome_artifact(
        source,
        output,
        profile="student",
        student_contract_profile="v6",
        archive=archive,
    )

    admitted = validate_and_resolve_student_consumption(
        output,
        profile_id="native_v3_student_v6",
        strict=True,
    )
    assert admitted.ok, admitted.issues
    if archive == "none":
        assert (output / "manifests/language_tokenizer_binding_v1.json").is_file()
        assert (output / "manifests/behavioral_resource_binding_v1.json").is_file()
        assert "native_v3_student_v6" in (output / "cover_page.json").read_text(
            encoding="utf-8"
        )


def test_b3_v6_contract_rejects_a_tampered_authority_member(tmp_path: Path) -> None:
    source = _v5_source(tmp_path)
    output = tmp_path / "student-v6"
    package_tome_artifact(
        source, output, profile="student", student_contract_profile="v6"
    )
    input_path = output / "student_consumption/v6/input_ids.npy"
    values = np.load(input_path, allow_pickle=False)
    values[0, 0] = values[0, 0] + 1
    np.save(input_path, values)

    admitted = validate_and_resolve_student_consumption(
        output,
        profile_id="native_v3_student_v6",
        strict=True,
    )
    assert [issue.code for issue in admitted.issues] == [
        "BRC010_RAW_INTEGRITY_MISMATCH",
        "BRC012_REQUIRED_ROLE_MISSING",
    ]


@pytest.mark.parametrize("archive", ["none", "tgz"])
def test_b3_v6_contract_failure_never_promotes_staged_output(
    tmp_path: Path, archive: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import radjax_tome.tome.packaging as packaging

    source = _v5_source(tmp_path)
    output = tmp_path / ("blocked.tgz" if archive == "tgz" else "blocked")
    monkeypatch.setattr(
        packaging,
        "_validate_student_consumption_v6_with_contract",
        lambda _: (_ for _ in ()).throw(ValueError("forced v6 admission failure")),
    )
    with pytest.raises(ValueError, match="forced v6 admission failure"):
        package_tome_artifact(
            source,
            output,
            profile="student",
            archive=archive,
            student_contract_profile="v6",
        )
    assert not output.exists()


@pytest.mark.parametrize("archive", ["none", "tgz"])
def test_b3_v6_binds_the_native_m7_sibling_and_streams_it(
    tmp_path: Path, archive: str
) -> None:
    from radjax_contract.tome import open_verified_student_m7_payload_v6

    source = _v5_source(tmp_path)
    m7_path = source.with_name(f"{source.name}.v4")
    m7_root = (
        m7_path
        if m7_path.is_dir()
        else package_legacy_artifact_as_sharded_tome_v4(source, m7_path).root
    )
    m7_archive = source.with_name(f"{source.name}.v4.tgz")
    if not m7_archive.is_file():
        pack_sharded_tome_v4(m7_root, m7_archive)
    output = tmp_path / ("v6.tgz" if archive == "tgz" else "v6")
    package_tome_artifact(
        source,
        output,
        profile="student",
        archive=archive,
        student_contract_profile="v6",
    )
    admitted = validate_and_resolve_student_consumption(
        output, profile_id="native_v3_student_v6", strict=True
    )
    assert admitted.ok, admitted.issues
    selected = next(
        item
        for item in admitted.descriptor.authority_resources
        if item.role == "selected_exemplar_payload"
    )
    assert selected.encoding == "m7_tome_archive"
    with open_verified_student_m7_payload_v6(
        output, selected.resource_id, strict=True
    ) as reader:
        next(iter(reader))
    assert reader.verification_state == "closed_early"
    with open_verified_student_m7_payload_v6(
        output, selected.resource_id, strict=True
    ) as reader:
        list(reader)
        assert reader.verification_state == "fully_verified"


def test_b3_v6_rejects_self_consistent_inner_m7_exemplar_semantics(
    tmp_path: Path,
) -> None:
    source = _v5_source(tmp_path)
    m7_archive = source.with_name(f"{source.name}.v4.tgz")
    _mutate_m7_inner_exemplar(m7_archive, m7_archive)
    assert validate_streaming_tome(m7_archive).ok

    with pytest.raises(ValueError, match="BRC027_EXEMPLAR_SEMANTICS_INVALID"):
        package_tome_artifact(
            source,
            tmp_path / "rejected",
            profile="student",
            student_contract_profile="v6",
        )
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("archive", ["none", "tgz"])
def test_t2_corrupt_capture_is_a_production_path_no_promotion_failure(
    tmp_path: Path, archive: str
) -> None:
    source = _v5_source(tmp_path)
    vocabulary = source / "resources/tokenizer_vocabulary.jsonl"
    vocabulary.write_bytes(vocabulary.read_bytes() + b"corrupt")
    output = tmp_path / ("student.tgz" if archive == "tgz" else "student")

    with pytest.raises(ValueError, match="Contract v5 source binding validation"):
        package_tome_artifact(source, output, profile="student", archive=archive)

    assert not output.exists()
