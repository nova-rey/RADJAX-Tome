from __future__ import annotations

import hashlib
import json
from pathlib import Path

import radjax_tome.corpora.builder as corpus_builder
from radjax_tome.builder import TeacherTextbookBuildConfig, build_teacher_textbook
from radjax_tome.corpora import (
    CORPUS_BUILD_REPORT_FILENAME,
    CORPUS_JSONL_FILENAME,
    CORPUS_MANIFEST_FILENAME,
    CorpusBuildConfig,
    build_corpus_artifact,
    normalize_text,
    validate_corpus_artifact,
)
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_path: Path, source: Path, **overrides: object) -> Path:
    output = tmp_path / "corpus_out"
    payload = {
        "inputs": (source,),
        "output_dir": output,
        "overwrite": True,
    }
    payload.update(overrides)
    build_corpus_artifact(CorpusBuildConfig(**payload))
    return output


def _rows(output: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output / CORPUS_JSONL_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _manifest(output: Path) -> dict[str, object]:
    return json.loads((output / CORPUS_MANIFEST_FILENAME).read_text(encoding="utf-8"))


def _report(output: Path) -> dict[str, object]:
    return json.loads(
        (output / CORPUS_BUILD_REPORT_FILENAME).read_text(encoding="utf-8")
    )


def _manifest_hash_payload(manifest: dict[str, object]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_hash", "created_at"}
    }
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )


def test_build_corpus_from_local_txt_and_md_files(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    (source / "a.txt").write_text("alpha\n", encoding="utf-8")
    (source / "b.md").write_text("# Beta\n\nbody\n", encoding="utf-8")

    output = _build(tmp_path, source)
    rows = _rows(output)
    manifest = _manifest(output)

    assert [row["example_id"] for row in rows] == [
        "corpus_000000001",
        "corpus_000000002",
    ]
    assert {row["source_relative_path"] for row in rows} == {"a.txt", "b.md"}
    assert manifest["schema_version"] == "corpus_manifest_v1"
    assert manifest["artifact_type"] == "radjax_tome_corpus"
    assert manifest["num_examples"] == 2
    assert manifest["num_sources"] == 2


def test_build_corpus_from_jsonl_text_rows(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        json.dumps({"text": "row one"}) + "\n" + json.dumps({"text": "row two"}) + "\n",
        encoding="utf-8",
    )

    output = _build(tmp_path, source)

    assert [row["text"] for row in _rows(output)] == ["row one", "row two"]


def test_include_and_exclude_globs_and_default_excludes(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    (source / "keep.md").write_text("keep", encoding="utf-8")
    (source / "drop.txt").write_text("drop", encoding="utf-8")
    git_dir = source / ".git"
    git_dir.mkdir()
    (git_dir / "hidden.md").write_text("hidden", encoding="utf-8")

    output = _build(
        tmp_path,
        source,
        include_globs=("**/*.md",),
        exclude_globs=("**/.git/**",),
    )

    rows = _rows(output)
    report = _report(output)
    assert [row["source_relative_path"] for row in rows] == ["keep.md"]
    reasons = {item["excluded_reason"] for item in report["excluded_sources"]}
    assert "include_glob_miss" in reasons
    assert "default_exclude" in reasons


def test_binary_and_unsupported_files_are_skipped_with_reasons(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    (source / "ok.txt").write_text("ok", encoding="utf-8")
    (source / "image.bin").write_bytes(b"\x00\x01")
    (source / "data.csv").write_text("a,b\n", encoding="utf-8")
    (source / "data.json").write_text('{"text": "ambiguous"}', encoding="utf-8")

    output = _build(tmp_path, source)
    report = _report(output)

    assert report["status"] == "warn"
    excluded = {
        item["source_relative_path"]: item["excluded_reason"]
        for item in report["excluded_sources"]
    }
    assert excluded["data.csv"] == "unsupported_file_type"
    assert excluded["data.json"] == "unsupported_file_type"


def test_normalization_and_hashes_are_recorded(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"alpha  \r\nbeta\t\r\n")
    output = _build(tmp_path, source)
    row = _rows(output)[0]
    text = "alpha\nbeta"

    assert normalize_text("alpha  \r\nbeta\t\r\n") == text
    assert row["text"] == text
    assert (
        row["content_hash"]
        == "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    assert (
        row["source_hash"]
        == "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    )


def test_chunking_and_exact_dedup_keep_contiguous_ids(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    (source / "a.txt").write_text("abcdef", encoding="utf-8")
    (source / "b.txt").write_text("abcdef", encoding="utf-8")

    output = _build(tmp_path, source, max_chars=3)
    rows = _rows(output)
    report = _report(output)

    assert [row["text"] for row in rows] == ["abc", "def"]
    assert [row["chunk_index"] for row in rows] == [0, 1]
    assert [row["chunk_count"] for row in rows] == [2, 2]
    assert [row["example_id"] for row in rows] == [
        "corpus_000000001",
        "corpus_000000002",
    ]
    assert report["num_duplicates_removed"] == 2


def test_corpus_and_manifest_hashes_validate(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    output = _build(tmp_path, source)
    manifest = _manifest(output)
    corpus_hash = (
        "sha256:"
        + hashlib.sha256((output / CORPUS_JSONL_FILENAME).read_bytes()).hexdigest()
    )

    assert manifest["corpus_hash"] == corpus_hash
    assert str(manifest["manifest_hash"]).startswith("sha256:")
    assert manifest["created_at"] != "1970-01-01T00:00:00+00:00"
    assert manifest["manifest_hash_policy"] == "exclude_self_hash_and_created_at_v1"
    assert manifest["manifest_hash"] == _manifest_hash_payload(manifest)
    assert _report(output)["manifest_hash_policy"] == (
        "exclude_self_hash_and_created_at_v1"
    )
    validation = validate_corpus_artifact(output)
    assert validation.status == "pass"
    assert validation.corpus_hash == corpus_hash


def test_manifest_hash_excludes_created_at_for_identical_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    monkeypatch.setattr(
        corpus_builder,
        "_utc_created_at",
        lambda: "2026-07-07T00:00:00.000001+00:00",
    )
    build_corpus_artifact(
        CorpusBuildConfig(inputs=(source,), output_dir=first_output, overwrite=True)
    )
    monkeypatch.setattr(
        corpus_builder,
        "_utc_created_at",
        lambda: "2026-07-07T00:00:01.000001+00:00",
    )
    build_corpus_artifact(
        CorpusBuildConfig(inputs=(source,), output_dir=second_output, overwrite=True)
    )

    first = _manifest(first_output)
    second = _manifest(second_output)
    assert first["created_at"] != second["created_at"]
    assert first["corpus_hash"] == second["corpus_hash"]
    assert first["manifest_hash"] == second["manifest_hash"]


def test_corpus_and_manifest_hash_change_for_content_or_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    third_output = tmp_path / "third"

    build_corpus_artifact(
        CorpusBuildConfig(inputs=(source,), output_dir=first_output, overwrite=True)
    )
    source.write_text("beta", encoding="utf-8")
    build_corpus_artifact(
        CorpusBuildConfig(inputs=(source,), output_dir=second_output, overwrite=True)
    )
    source.write_text("alpha", encoding="utf-8")
    build_corpus_artifact(
        CorpusBuildConfig(
            inputs=(source,),
            output_dir=third_output,
            include_globs=("*.txt",),
            overwrite=True,
        )
    )

    first = _manifest(first_output)
    second = _manifest(second_output)
    third = _manifest(third_output)
    assert first["corpus_hash"] != second["corpus_hash"]
    assert first["manifest_hash"] != second["manifest_hash"]
    assert first["corpus_hash"] == third["corpus_hash"]
    assert first["manifest_hash"] != third["manifest_hash"]


def test_corpus_validator_fails_corrupted_content_and_corpus_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    output = _build(tmp_path, source)
    rows = _rows(output)
    rows[0]["content_hash"] = "sha256:bad"
    (output / CORPUS_JSONL_FILENAME).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    validation = validate_corpus_artifact(output)

    assert validation.status == "fail"
    assert any("content_hash mismatch" in blocker for blocker in validation.blockers)
    assert any("corpus_hash" in blocker for blocker in validation.blockers)


def test_corpus_cli_build_inspect_validate(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    (source / "a.md").write_text("# Alpha", encoding="utf-8")
    output = tmp_path / "cli_corpus"

    build = run_cli(
        ROOT,
        "corpus",
        "build",
        "--input",
        str(source),
        "--output",
        str(output),
        "--include",
        "**/*.md",
        "--overwrite",
    )
    inspect = run_cli(ROOT, "corpus", "inspect", "--path", str(output))
    validate = run_cli(ROOT, "corpus", "validate", "--path", str(output))

    assert build.returncode == 0, build.stderr
    assert "status=pass" in build.stdout
    assert "corpus_hash=sha256:" in build.stdout
    assert inspect.returncode == 0, inspect.stderr
    assert "corpus_schema=corpus_manifest_v1" in inspect.stdout
    assert validate.returncode == 0, validate.stderr
    assert "status=pass" in validate.stdout


def test_tome_build_with_corpus_manifest_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    corpus_output = _build(tmp_path, source)
    tome_output = tmp_path / "tome"

    report = build_teacher_textbook(
        TeacherTextbookBuildConfig(
            output_dir=tome_output,
            dataset_path=corpus_output / CORPUS_JSONL_FILENAME,
            corpus_manifest_path=corpus_output / CORPUS_MANIFEST_FILENAME,
            max_examples=1,
            sequence_length=8,
            overwrite=True,
        )
    )

    manifest = _manifest(corpus_output)
    metadata = json.loads((tome_output / "metadata.json").read_text(encoding="utf-8"))
    teacher_manifest = json.loads(
        (tome_output / "teacher_manifest.json").read_text(encoding="utf-8")
    )
    emission_config = json.loads(
        (tome_output / "emission_config.json").read_text(encoding="utf-8")
    )
    cover_page = json.loads(
        (tome_output / "cover_page.json").read_text(encoding="utf-8")
    )

    assert report.status == "pass"
    assert metadata["target_params"]["source_corpus_hash"] == manifest["corpus_hash"]
    assert (
        teacher_manifest["corpus_provenance"]["source_corpus_hash"]
        == (manifest["corpus_hash"])
    )
    assert (
        emission_config["corpus_provenance"]["source_corpus_manifest_hash"]
        == (manifest["manifest_hash"])
    )
    assert cover_page["corpus"]["source_corpus_hash"] == manifest["corpus_hash"]


def test_docs_and_bible_mention_spec_4_1() -> None:
    bible = (ROOT / "bible.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "CORPUS_BUILDER.md").read_text(encoding="utf-8")

    assert "Spec 4.1" in bible
    assert "corpus builder" in bible
    assert "corpus_hash" in docs
    assert "Do not scrape" in docs
    assert "Spec 4.1.1" in bible
    assert ".json is not supported yet" in docs
    assert "exclude_self_hash_and_created_at_v1" in docs
