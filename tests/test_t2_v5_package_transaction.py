from __future__ import annotations

from pathlib import Path

import pytest
from radjax_contract.tome import validate_and_resolve_student_consumption

from radjax_tome.tome import package_tome_artifact
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
