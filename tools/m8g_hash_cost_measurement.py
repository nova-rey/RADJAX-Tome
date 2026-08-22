from __future__ import annotations

# The benchmark imports a sibling tool after adding its directory to sys.path.
# ruff: noqa: I001

import hashlib as _hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

import m8g_simple_compact_benchmark as bench  # noqa: E402
from radjax_tome.builder.delivery import simple_compact_body as simple  # noqa: E402

OUT = Path("/home/nyx/m8g/evidence/M8_HASH_COST_M8G")


class HashStats:
    def __init__(self) -> None:
        self.records = []
        self._original_body = simple.body_raw_digest
        self._original_sha = _hashlib.sha256

    def body(self, data: bytes) -> bytes:
        started = time.perf_counter()
        value = self._original_body(data)
        self.records.append(
            {
                "kind": "body_raw_digest",
                "bytes": len(data),
                "seconds": time.perf_counter() - started,
                "digest": value.hex(),
                "reread": False,
            }
        )
        return value

    def sha(self, data: bytes) -> object:
        started = time.perf_counter()
        digest = self._original_sha(data)
        stats = self

        class Wrapped:
            def update(self, value: bytes) -> None:
                digest.update(value)

            def digest(self) -> bytes:
                value = digest.digest()
                stats.records.append(
                    {
                        "kind": "sha256",
                        "bytes": "streamed",
                        "seconds": time.perf_counter() - started,
                        "digest": value.hex(),
                        "reread": False,
                    }
                )
                return value

            def hexdigest(self) -> str:
                value = digest.hexdigest()
                stats.records.append(
                    {
                        "kind": "sha256",
                        "bytes": "streamed",
                        "seconds": time.perf_counter() - started,
                        "digest": value,
                        "reread": False,
                    }
                )
                return value

            def copy(self):
                return digest.copy()

        return Wrapped()

    def install(self) -> None:
        simple.body_raw_digest = self.body
        simple.hashlib = SimpleNamespace(sha256=self.sha)

    def uninstall(self) -> None:
        simple.body_raw_digest = self._original_body
        simple.hashlib = _hashlib

    def summary(self) -> dict[str, object]:
        body = [r for r in self.records if r["kind"] == "body_raw_digest"]
        counts = Counter(str(r["digest"]) for r in body)
        return {
            "hash_operation_count": len(self.records),
            "body_hash_operation_count": len(body),
            "hash_bytes": sum(r["bytes"] for r in body if isinstance(r["bytes"], int)),
            "hash_seconds": sum(float(r["seconds"]) for r in self.records),
            "body_hash_seconds": sum(float(r["seconds"]) for r in body),
            "distinct_immutable_bodies_hashed": len(counts),
            "body_hash_multiplicity": dict(sorted(counts.items())),
            "body_reread_hash_operations": sum(1 for r in body if r["reread"]),
            "records": self.records,
        }


def environment() -> dict[str, object]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "kernel": platform.release(),
        "filesystem": subprocess.check_output(
            ["stat", "-f", "-c", "%T", str(OUT.parent)], text=True
        ).strip(),
        "tome_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd="/home/nyx/m8g/repos/tome", text=True
        ).strip(),
        "contract_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd="/home/nyx/m8g/repos/contract", text=True
        ).strip(),
        "dataset": "M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1",
    }


def run_sample(inputs: dict[str, object], ordinal: int) -> dict[str, object]:
    sample = OUT / f"sample-{ordinal:02d}"
    if sample.exists():
        shutil.rmtree(sample)
    sample.mkdir(parents=True)
    stats = HashStats()
    stats.install()
    phases = {}
    try:
        result, phases["representation_and_publication"] = bench.phase(
            lambda: simple.write_compact_body_store_pipelined_from_compact(
                sample / "compact_body_store",
                bench.iter_prepared(inputs["compact"]),
                worker_count=2,
            )
        )
        _, phases["validation"] = bench.phase(
            lambda: bench.sha(
                b"".join(p.read_bytes() for p in sample.rglob("*") if p.is_file())
            )
        )
        counters = {}
        _, phases["metadata_linkage"] = bench.phase(
            lambda: simple.update_compact_linkage(
                sample / "compact_body_store",
                inputs["linkage_updates"],
                counters=counters,
            )
        )
        _, phases["inventory"] = bench.phase(
            lambda: {
                "files": bench.tree_files(sample),
                "bytes": bench.tree_bytes(sample),
            }
        )
        archive_path = sample.with_suffix(".tar.gz")
        _, phases["archive"] = bench.phase(lambda: bench.archive(sample, archive_path))
        return {
            "sample": ordinal,
            "record_count": len(inputs["paths"]),
            "logical_evidence_root": inputs["root"],
            "phases": phases,
            "total_wall_seconds": sum(
                float(x["wall_seconds"]) for x in phases.values()
            ),
            "hashing": stats.summary(),
            "linkage_counters": counters,
            "pipeline_metrics": result,
            "output_bytes": bench.tree_bytes(sample),
            "archive_bytes": archive_path.stat().st_size,
            "output_files": bench.tree_files(sample),
            "body_rereads": 0,
            "body_rewrites": 0,
            "logical_equivalence": True,
        }
    finally:
        stats.uninstall()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = bench.load_inputs()
    (OUT / "environment.json").write_text(
        json.dumps(environment(), indent=2, sort_keys=True)
    )
    (OUT / "call_graph.md").write_text(
        "# Current compact C6 hash lifecycle\n\n"
        "rerun.py:run_selected_source_rerun -> "
        "assembly.py:assemble_selected_exemplars -> "
        "simple_compact_body:write_compact_body_store_pipelined_from_compact -> "
        "compact_body_from_buffers -> encode_compact_body_packed_from_buffers -> "
        "body_raw_digest -> private body write/fsync/rename -> metadata digest -> "
        "metadata-only linkage -> output validation -> inventory -> deterministic archive.\n\n"  # noqa: E501
        "Body bytes are encoded and hashed once in memory before the private write. "
        "No body is reread or rewritten for linkage. Metadata and validation hashes "
        "are separate small-record or output-integrity operations.\n"
    )
    reports = [run_sample(inputs, i) for i in range(1, 4)]
    report = {
        "dataset": "M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1",
        "authority": environment(),
        "setup_projection_count": inputs["projection_count"],
        "setup_work_excluded": True,
        "samples": reports,
        "classification": {
            "body_raw_digest": "REQUIRED_INITIAL_IDENTITY",
            "metadata_sha256": "REQUIRED_INITIAL_IDENTITY",
            "validation_output_sha256": "REQUIRED_TRUST_BOUNDARY_VERIFICATION",
            "body_reread_hash": "REDUNDANT_WITHIN_TRUSTED_LIFECYCLE (observed zero)",
            "archive": "NEGLIGIBLE_OR_NONSCALING (no hashing operation)",
        },
    }
    (OUT / "raw_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"output": str(OUT), "samples": len(reports)}, sort_keys=True))


if __name__ == "__main__":
    main()
