from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from radjax_contract.tome import validate_and_resolve_student_consumption

from radjax_tome.tome import (
    pack_sharded_tome_v4,
    package_legacy_artifact_as_sharded_tome_v4,
    package_tome_artifact,
)
from tests.test_tome_packaging_profiles import _artifact


def _v5_source(tmp_path: Path) -> Path:
    return _artifact(
        tmp_path / "source",
        teacher_backend="smoke_tokenizer",
        vocab_size=512,
        top_k=4,
    )


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
