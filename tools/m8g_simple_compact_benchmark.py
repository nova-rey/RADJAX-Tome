from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import tarfile
import time
from pathlib import Path

from radjax_tome.builder.delivery.modes import compact_payload_for_storage
from radjax_tome.builder.delivery.simple_compact_body import (
    update_compact_linkage,
    write_compact_body_store_from_compact,
    write_compact_body_store_pipelined_from_compact,
)

ROOT = Path("/home/nyx/m8g/published/m8g-current-1k-workload-authoritative-v19")
EVIDENCE = Path(
    "/home/nyx/m8g/evidence/M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1"
)
OUT = Path("/home/nyx/m8g/evidence/M8G_PIPELINED_COMPACT_REPRESENTATION")
MANIFEST = EVIDENCE / "derived_dataset_manifest.json"


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def tree_files(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def archive(root: Path, target: Path) -> int:
    with target.open("wb") as raw:
        import gzip

        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1
        ) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tf:
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        info = tf.gettarinfo(
                            str(path), arcname=path.relative_to(root).as_posix()
                        )
                        with path.open("rb") as handle:
                            tf.addfile(info, handle)
    return target.stat().st_size


def io_snapshot():
    fields = {"read_bytes": 0, "write_bytes": 0}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in fields:
                fields[key] = int(value.strip())
    except OSError:
        pass
    return fields


def phase(fn):
    before = io_snapshot()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    after = io_snapshot()
    return value, {
        "wall_seconds": elapsed,
        "cpu_seconds": (cpu_after.ru_utime + cpu_after.ru_stime)
        - (cpu_before.ru_utime + cpu_before.ru_stime),
        "peak_rss_bytes": int(cpu_after.ru_maxrss) * 1024,
        "bytes_read": after["read_bytes"] - before["read_bytes"],
        "bytes_written": after["write_bytes"] - before["write_bytes"],
    }


def load_inputs():
    manifest = json.loads(MANIFEST.read_text())
    paths = []
    prepared = OUT / "prepared-inputs"
    if prepared.exists():
        shutil.rmtree(prepared)
    legacy_dir = prepared / "legacy"
    compact_dir = prepared / "compact"
    legacy_dir.mkdir(parents=True)
    compact_dir.mkdir(parents=True)
    canonical_hasher = hashlib.sha256()
    linkage_updates = {}
    projection_count = 0
    for item in manifest["records"]:
        path = ROOT / item["relative_path"]
        raw = path.read_bytes()
        if sha(raw) != item["actual_sha256"] or len(raw) != item["actual_size"]:
            raise RuntimeError(f"source digest mismatch: {path}")
        payload = json.loads(raw)["selected_exemplars"][0]
        compact = compact_payload_for_storage(payload)
        projection_count += 1
        canonical_hasher.update(
            json.dumps(compact, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        index = len(paths)
        (legacy_dir / f"{index:04d}.json").write_text(
            json.dumps(payload, separators=(",", ":"))
        )
        (compact_dir / f"{index:04d}.json").write_text(
            json.dumps(compact, separators=(",", ":"))
        )
        linkage_updates[
            (str(payload["selected_example_id"]), int(payload["selected_position"]))
        ] = "linked"
        paths.append(path)
    return {
        "paths": paths,
        "legacy": sorted(legacy_dir.glob("*.json")),
        "compact": sorted(compact_dir.glob("*.json")),
        "linkage_updates": linkage_updates,
        "root": "sha256:" + canonical_hasher.hexdigest(),
        "projection_count": projection_count,
        "prepared": prepared,
    }


def iter_prepared(paths):
    for path in paths:
        yield json.loads(path.read_text())


def run(mode: str, round_no: int, inputs, *, worker_count: int = 2, label: str = "r"):
    sample = OUT / f"{label}{round_no}-{mode}-w{worker_count}"
    if sample.exists():
        shutil.rmtree(sample)
    sample.mkdir(parents=True)
    phases = {}
    pipeline_result = {}
    paths = inputs["paths"]
    logical_root = inputs["root"]
    if mode == "legacy_padded_monolithic":

        def write_legacy():
            path = sample / "selected-exemplars.json"
            with path.open("wb") as handle:
                handle.write(b'{"selected_exemplars":[')
                first = True
                for payload in iter_prepared(inputs["legacy"]):
                    if not first:
                        handle.write(b",")
                    first = False
                    handle.write(json.dumps(payload, separators=(",", ":")).encode())
                handle.write(b"]}")

        _, phases["representation_construction"] = phase(write_legacy)
    else:
        writer = (
            write_compact_body_store_from_compact
            if mode == "compact_k_monolithic"
            else write_compact_body_store_pipelined_from_compact
        )
        if mode == "compact_k_monolithic":
            pipeline_result, phases["representation_construction"] = phase(
                lambda: writer(
                    sample / "compact_body_store", iter_prepared(inputs["compact"])
                )
            )
        else:
            pipeline_result, phases["representation_construction"] = phase(
                lambda: writer(
                    sample / "compact_body_store",
                    iter_prepared(inputs["compact"]),
                    worker_count=worker_count,
                )
            )
    _, phases["initial_staging_publication"] = phase(lambda: None)
    _, phases["validation_hashing"] = phase(
        lambda: sha(b"".join(p.read_bytes() for p in sample.rglob("*") if p.is_file()))
    )
    if mode == "legacy_padded_monolithic":

        def legacy_link():
            path = sample / "selected-exemplars.json"
            temporary = path.with_suffix(".tmp")
            with path.open("rb") as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, path)

        _, phases["linkage_update"] = phase(legacy_link)
    else:
        linkage_counters = {}
        _, phases["linkage_update"] = phase(
            lambda: update_compact_linkage(
                sample / "compact_body_store",
                inputs["linkage_updates"],
                counters=linkage_counters,
            )
        )
    _, phases["post_linkage_reread_rehash_rewrite"] = phase(lambda: None)
    _, phases["inventory"] = phase(
        lambda: {"files": tree_files(sample), "bytes": tree_bytes(sample)}
    )
    archive_path = sample.with_suffix(".tar.gz")
    _, phases["archive"] = phase(lambda: archive(sample, archive_path))
    total = sum(v["wall_seconds"] for v in phases.values())
    report = {
        "mode": mode,
        "round": round_no,
        "logical_evidence_root": logical_root,
        "record_count": len(paths),
        "phases": phases,
        "total_wall_seconds": total,
        "output_bytes": tree_bytes(sample),
        "archive_bytes": archive_path.stat().st_size,
        "output_files": tree_files(sample),
        "body_count": len(
            list((sample / "compact_body_store" / "bodies").glob("*.body"))
        )
        if mode != "legacy_padded_monolithic"
        else 0,
        "metadata_bytes": (sample / "compact_body_store" / "metadata.jsonl")
        .stat()
        .st_size
        if mode != "legacy_padded_monolithic"
        else 0,
        "setup_work_excluded": True,
        "setup_projection_count": inputs["projection_count"],
        "linkage_counters": linkage_counters
        if mode != "legacy_padded_monolithic"
        else {
            "source_payload_reads": len(paths),
            "body_reads": 1,
            "body_hashes": 1,
            "body_rewrites": 1,
        },
        "worker_count": worker_count,
    }
    if mode != "legacy_padded_monolithic":
        report["pipeline_metrics"] = pipeline_result
    (OUT / f"{label}{round_no}-{mode}-w{worker_count}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    order = [
        [
            "legacy_padded_monolithic",
            "compact_k_monolithic",
            "compact_k_immutable_body",
        ],
        [
            "compact_k_monolithic",
            "compact_k_immutable_body",
            "legacy_padded_monolithic",
        ],
        [
            "compact_k_immutable_body",
            "legacy_padded_monolithic",
            "compact_k_monolithic",
        ],
    ]
    reports = []
    for round_no, modes in enumerate(order, 1):
        for mode in modes:
            reports.append(run(mode, round_no, inputs, worker_count=2))
    sweep = []
    for worker_count in (1, 2, 4):
        for repetition in range(1, 4):
            sweep.append(
                run(
                    "compact_k_immutable_body",
                    repetition,
                    inputs,
                    worker_count=worker_count,
                    label=f"sweep-w{worker_count}-",
                )
            )
    (OUT / "raw_three_round_report.json").write_text(
        json.dumps(
            {
                "dataset": "M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1",
                "logical_evidence_root": inputs["root"],
                "setup_projection_count": inputs["projection_count"],
                "setup_work_excluded": True,
                "samples": reports,
                "worker_sweep": sweep,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
