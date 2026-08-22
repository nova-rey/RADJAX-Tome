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
    write_compact_body_store,
)

ROOT = Path("/home/nyx/m8g/published/m8g-current-1k-workload-authoritative-v19")
EVIDENCE = Path("/home/nyx/m8g/evidence/M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1")
OUT = Path("/home/nyx/m8g/evidence/M8G_SIMPLE_COMPACT_K_STORAGE")
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

        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tf:
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        info = tf.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
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
        "cpu_seconds": (cpu_after.ru_utime + cpu_after.ru_stime) - (cpu_before.ru_utime + cpu_before.ru_stime),
        "peak_rss_bytes": int(cpu_after.ru_maxrss) * 1024,
        "bytes_read": after["read_bytes"] - before["read_bytes"],
        "bytes_written": after["write_bytes"] - before["write_bytes"],
    }


def load_inputs():
    manifest = json.loads(MANIFEST.read_text())
    paths = []
    canonical_hasher = hashlib.sha256()
    for item in manifest["records"]:
        path = ROOT / item["relative_path"]
        raw = path.read_bytes()
        if sha(raw) != item["actual_sha256"] or len(raw) != item["actual_size"]:
            raise RuntimeError(f"source digest mismatch: {path}")
        compact = compact_payload_for_storage(json.loads(raw)["selected_exemplars"][0])
        canonical_hasher.update(json.dumps(compact, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        paths.append(path)
    return paths, "sha256:" + canonical_hasher.hexdigest()


def iter_records(paths):
    for path in paths:
        payload = json.loads(path.read_bytes())["selected_exemplars"][0]
        yield payload, compact_payload_for_storage(payload)


def run(mode: str, round_no: int, paths, logical_root: str):
    sample = OUT / f"r{round_no}-{mode}"
    if sample.exists():
        shutil.rmtree(sample)
    sample.mkdir(parents=True)
    phases = {}
    if mode == "legacy_padded_monolithic":
        def write_legacy():
            path = sample / "selected-exemplars.json"
            with path.open("wb") as handle:
                handle.write(b'{"selected_exemplars":[')
                first = True
                for payload, _ in iter_records(paths):
                    if not first:
                        handle.write(b",")
                    first = False
                    handle.write(json.dumps(payload, separators=(",", ":")).encode())
                handle.write(b"]}")
        _, phases["representation_construction"] = phase(write_legacy)
    else:
        _, phases["representation_construction"] = phase(
            lambda: write_compact_body_store(
                sample / "compact_body_store",
                (compact for _, compact in iter_records(paths)),
            )
        )
    _, phases["initial_staging_publication"] = phase(lambda: None)
    _, phases["validation_hashing"] = phase(lambda: sha(b"".join(p.read_bytes() for p in sample.rglob("*") if p.is_file())))
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
        _, phases["linkage_update"] = phase(
            lambda: update_compact_linkage(sample / "compact_body_store", {
                (str(payload["selected_example_id"]), int(payload["selected_position"])): "linked"
                for payload, _ in iter_records(paths)
            })
        )
    _, phases["post_linkage_reread_rehash_rewrite"] = phase(lambda: None)
    _, phases["inventory"] = phase(lambda: {"files": tree_files(sample), "bytes": tree_bytes(sample)})
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
        "body_count": len(list((sample / "compact_body_store" / "bodies").glob("*.body"))) if mode != "legacy_padded_monolithic" else 0,
        "metadata_bytes": (sample / "compact_body_store" / "metadata.jsonl").stat().st_size if mode != "legacy_padded_monolithic" else 0,
    }
    (OUT / f"r{round_no}-{mode}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths, root = load_inputs()
    order = [
        ["legacy_padded_monolithic", "compact_k_monolithic", "compact_k_immutable_body"],
        ["compact_k_monolithic", "compact_k_immutable_body", "legacy_padded_monolithic"],
        ["compact_k_immutable_body", "legacy_padded_monolithic", "compact_k_monolithic"],
    ]
    reports = []
    for round_no, modes in enumerate(order, 1):
        for mode in modes:
            reports.append(run(mode, round_no, paths, root))
    (OUT / "raw_three_round_report.json").write_text(json.dumps({"dataset": "M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1", "logical_evidence_root": root, "samples": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
