import hashlib
import json
import math
import platform
import resource
import shutil
import statistics
import tarfile
import time
from pathlib import Path

ROOT = Path("/home/nyx/m8g/published/m8g-current-1k-workload-authoritative-v19")
EVID = Path("/home/nyx/m8g/evidence/M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1")
OUT = Path("/home/nyx/m8g/m8g-derived-benchmark-runs")
MODES = ("legacy_padded_monolithic", "compact_k_monolithic", "compact_k_immutable_body")
ORDER = [
    ("r1", MODES[0]),
    ("r1", MODES[1]),
    ("r1", MODES[2]),
    ("r2", MODES[1]),
    ("r2", MODES[2]),
    ("r2", MODES[0]),
    ("r3", MODES[2]),
    ("r3", MODES[0]),
    ("r3", MODES[1]),
]


def sha(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def fsha(p):
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
            n += len(b)
    return "sha256:" + h.hexdigest(), n


def canon(x):
    return json.dumps(
        x, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def candidates():
    idx = json.loads(
        (
            ROOT / "selection-checkpoint/selected_exemplars/payload_index.json"
        ).read_text()
    )["selected_exemplars"]
    inc = []
    exc = []
    seen = set()
    for i in range(len(idx)):
        p = (
            ROOT
            / "selection-checkpoint/selected_exemplars"
            / f"selected-exemplars-{i:05d}.json"
        )
        base = {
            "index": i,
            "relative_path": p.relative_to(ROOT).as_posix(),
            "expected_payload_hash": idx[i].get("payload_hash"),
            "selected_example_id": idx[i].get("selected_example_id"),
            "selected_position": idx[i].get("selected_position"),
        }
        try:
            st = p.lstat()
            if not p.is_file() or p.is_symlink():
                raise ValueError("not_regular_nofollow")
            if st.st_size == 0:
                raise ValueError("zero_byte")
            raw = p.read_bytes()
            d = json.loads(raw)
            xs = d.get("selected_exemplars") if isinstance(d, dict) else None
            if not isinstance(xs, list) or len(xs) != 1:
                raise ValueError("record_count_invalid")
            x = xs[0]
            req = (
                "selected_example_id",
                "selected_position",
                "effective_top_k",
                "vocab_size",
                "top_token_ids",
                "top_probs",
                "top_log_probs",
                "top_selection_mask",
                "bucket_masses",
            )
            if any(k not in x for k in req):
                raise ValueError("required_field_missing")
            coord = (str(x["selected_example_id"]), int(x["selected_position"]))
            if coord in seen:
                raise ValueError("duplicate_coordinate")
            seen.add(coord)
            v = int(x["vocab_size"])
            k = int(x["effective_top_k"])
            if v <= 0 or k <= 0 or k > v:
                raise ValueError("logical_k_invalid")
            arrs = [
                x["top_token_ids"],
                x["top_probs"],
                x["top_log_probs"],
                x["top_selection_mask"],
            ]
            if any(not isinstance(a, list) or len(a) != v for a in arrs):
                raise ValueError("padded_array_shape_invalid")
            if sum(bool(z) for z in x["top_selection_mask"]) != k:
                raise ValueError("mask_k_mismatch")
            if any(type(z) is not bool for z in x["top_selection_mask"]):
                raise ValueError("mask_type_invalid")
            if any(type(z) is not int or z < 0 or z >= v for z in x["top_token_ids"]):
                raise ValueError("token_id_out_of_bounds")
            if any(
                not finite(z) for a in (x["top_probs"], x["top_log_probs"]) for z in a
            ):
                raise ValueError("nonfinite_probability")
            if any(not finite(z) for z in x["bucket_masses"]):
                raise ValueError("bucket_mass_invalid")
            active = [j for j, z in enumerate(x["top_selection_mask"]) if z]
            logical = dict(x)
            logical["top_token_ids"] = [x["top_token_ids"][j] for j in active]
            logical["top_probs"] = [x["top_probs"][j] for j in active]
            logical["top_log_probs"] = [x["top_log_probs"][j] for j in active]
            logical["top_selection_mask"] = [True] * k
            inc.append(
                {
                    "index": i,
                    "relative_path": base["relative_path"],
                    "actual_size": len(raw),
                    "actual_sha256": sha(raw),
                    "coordinate": {"example_id": coord[0], "position": coord[1]},
                    "logical_k": k,
                    "vocab_size": v,
                    "full_width": k == v,
                    "retained_entries": k,
                    "logical_digest": sha(canon(logical)),
                }
            )
        except Exception as e:
            exc.append({**base, "reason": str(e)})
    return inc, exc


def inv(root):
    out = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink():
            d, n = fsha(p)
            out.append({"path": p.relative_to(root).as_posix(), "size": n, "sha256": d})
    return out


def archive(root):
    a = Path(root) / "archive"
    a.mkdir(exist_ok=True)
    ap = Path(
        shutil.make_archive(str(a / "package"), "gztar", root_dir=str(root / "package"))
    )
    with tarfile.open(ap, "r:gz") as t:
        names = t.getnames()
        assert all(not n.startswith("/") and ".." not in Path(n).parts for n in names)
    return {
        "path": str(ap),
        "size": ap.stat().st_size,
        "sha256": fsha(ap)[0],
        "members": len(names),
    }


def setup(out):
    (out / "metadata.json").write_bytes(
        (ROOT / "selection-checkpoint/metadata.json").read_bytes()
    )
    (out / "shards").mkdir(exist_ok=True)
    shutil.copy2(
        ROOT / "selection-checkpoint/shards/shard-00000.npz",
        out / "shards/shard-00000.npz",
    )


def run(mode, rid, inc, examples):
    from radjax_contract.tome.m8g import (
        _m8g_fv3,
        body_raw_digest,
        encode_compact_body,
        encode_compact_monolithic,
        manifest_semantic_id,
    )

    from radjax_tome.builder.corridor_artifacts import (
        build_corridor_artifacts,
        validate_corridor_artifacts,
    )
    from radjax_tome.builder.delivery.immutable_body import ImmutableBodyTransaction
    from radjax_tome.builder.delivery.modes import compact_body_from_logical_payload
    from radjax_tome.targets.store import TeacherTargetStore

    out = OUT / rid
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    setup(out)
    TeacherTargetStore.open(out)
    records = []
    summaries = []
    for j, e in enumerate(inc):
        x = json.loads((ROOT / e["relative_path"]).read_text())["selected_exemplars"][0]
        x = {
            k: v
            for k, v in x.items()
            if k
            not in ("top_token_ids", "top_probs", "top_log_probs", "top_selection_mask")
        }
        x["_record_index"] = j
        records.append(x)
        summaries.append(x)
    t = time.perf_counter()
    build_corridor_artifacts(
        output_dir=out,
        examples=tuple(examples),
        selected_records=records,
        selected_payloads=summaries,
        delivery_path="two_pass_rerun_selected",
        non_selected_exemplar_payload_retained=False,
        allow_degraded_score_only=True,
    )
    linkage = time.perf_counter() - t
    pkg = out / "package"
    (pkg / "selected_exemplars").mkdir(parents=True)
    shutil.copytree(out / "corridors", pkg / "corridors", dirs_exist_ok=True)
    t = time.perf_counter()
    created = 0
    retained = padded = compact_n = 0
    tx = (
        ImmutableBodyTransaction(out / "m8g_immutable", profile="producer_evidence")
        if mode == "compact_k_immutable_body"
        else None
    )
    for j, e in enumerate(inc):
        x = json.loads((ROOT / e["relative_path"]).read_text())["selected_exemplars"][0]
        k = int(x["effective_top_k"])
        active = [q for q, z in enumerate(x["top_selection_mask"]) if z]
        probs = [x["top_probs"][q] for q in active]
        logical = {
            "selected_example_id": x["selected_example_id"],
            "selected_position": x["selected_position"],
            "vocab_size": x["vocab_size"],
            "num_buckets": x["num_buckets"],
            "top_token_ids": [x["top_token_ids"][q] for q in active],
            "top_probs": probs,
            "top_log_probs": [x["top_log_probs"][q] for q in active],
            "effective_top_k": k,
            "top_mass": x["top_mass"],
            "tail_mass": 1.0 - sum(probs),
            "bucket_masses": x["bucket_masses"],
        }
        body = compact_body_from_logical_payload(
            logical, profile="compact_k_monolithic"
        )
        retained += k
        padded += len(x["top_token_ids"])
        compact_n += len(body.top_token_ids)
        if mode == "legacy_padded_monolithic":
            (
                pkg / "selected_exemplars" / f"selected-exemplars-{j:05d}.json"
            ).write_text(
                json.dumps(
                    {
                        "schema_version": "selected_exemplar_payload_shard_v1",
                        "selected_exemplars": [x],
                    },
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        elif mode == "compact_k_monolithic":
            (
                pkg / "selected_exemplars" / f"selected-exemplars-{j:05d}.cbor"
            ).write_bytes(encode_compact_monolithic(body))
        else:
            bo = compact_body_from_logical_payload(logical, profile="producer_evidence")
            bb = encode_compact_body(bo)
            auth = bytes.fromhex(
                "763c4a47343ea06ab76671f605e8be6d688a2cdf54470e4a2e88a491c607932c"
            )
            man = {
                "schema_version": "selected_exemplar_manifest_v1",
                "profile": "producer_evidence",
                "selected_example_id": x["selected_example_id"],
                "selected_position": int(x["selected_position"]),
                "source_passport_id": (
                    f"{x['selected_example_id']}:{x['selected_position']}"
                ),
                "corridor_mode_id": str(x.get("mode_key")),
                "corridor_fingerprint_id": None,
                "selection_obligation_count": 0,
                "selection_obligations": [],
                "body_semantic_id": bo.semantic_id,
                "body_raw_digest": body_raw_digest(bb),
                "authority_id": auth,
                "selection_authority_id": auth,
                "package_role": "producer_evidence",
            }
            man["manifest_semantic_id"] = manifest_semantic_id(man)
            tx.commit(bo, man, canonical_manifest_bytes=_m8g_fv3(man))
            created += 1
    representation = time.perf_counter() - t
    (pkg / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1",
                "mode": mode,
                "count": len(inc),
            },
            sort_keys=True,
        )
    )
    t = time.perf_counter()
    iv = inv(pkg)
    vr = validate_corridor_artifacts(
        out,
        selected_records=records,
        selected_payloads=summaries,
        expected_selected_count=len(inc),
    )
    validation = time.perf_counter() - t
    t = time.perf_counter()
    ar = archive(out)
    arch = time.perf_counter() - t
    res = {
        "run_id": rid,
        "mode": mode,
        "count": len(inc),
        "linkage_seconds": linkage,
        "representation_seconds": representation,
        "validation_seconds": validation,
        "archive_seconds": arch,
        "total_seconds": linkage + representation + validation + arch,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "inventory_root": sha(canon(iv)),
        "inventory": iv,
        "archive": ar,
        "corridor_valid": vr.ok,
        "corridor_blockers": list(vr.blockers),
        "immutable_bodies_created": created,
        "logical_retained_entries": retained,
        "padded_physical_entries": padded,
        "compact_physical_entries": compact_n,
        "output_root": str(out),
    }
    return res


def main():
    EVID.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    inc, exc = candidates()
    logical_root = sha(canon([e["logical_digest"] for e in inc]))
    manifest = [
        {
            k: e[k]
            for k in (
                "index",
                "relative_path",
                "actual_size",
                "actual_sha256",
                "coordinate",
                "logical_k",
                "vocab_size",
                "full_width",
                "retained_entries",
                "logical_digest",
            )
        }
        for e in inc
    ]
    (EVID / "derived_dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1",
                "source_workload_identity": (
                    "sha256:75c5cf584b1e11b272029dd9e85eb3fe173b6efda2608db7b09b997431140acd"
                ),
                "included_count": len(inc),
                "records": manifest,
            },
            sort_keys=True,
            indent=2,
        )
    )
    (EVID / "derived_dataset_exclusions.json").write_text(
        json.dumps(
            {"excluded_count": len(exc), "records": exc}, sort_keys=True, indent=2
        )
    )
    ks = [e["logical_k"] for e in inc]
    full = sum(e["full_width"] for e in inc)
    (EVID / "derived_dataset_anatomy.json").write_text(
        json.dumps(
            {
                "included_count": len(inc),
                "excluded_count": len(exc),
                "dynamic_k": {
                    "min": min(ks),
                    "median": statistics.median(ks),
                    "max": max(ks),
                },
                "full_width_count": full,
                "full_width_fraction": full / len(inc),
                "retained_entries": sum(ks),
                "padded_physical_entries": sum(e["vocab_size"] for e in inc),
                "input_bytes": sum(e["actual_size"] for e in inc),
                "logical_root": logical_root,
            },
            sort_keys=True,
            indent=2,
        )
    )
    (EVID / "derived_logical_evidence_root.json").write_text(
        json.dumps(
            {
                "dataset": "M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1",
                "logical_evidence_root": logical_root,
                "included_count": len(inc),
            },
            sort_keys=True,
            indent=2,
        )
    )
    from radjax_tome.builder.teacher_textbook import load_text_examples

    examples = tuple(
        load_text_examples(ROOT / "corpus/corpus.jsonl", max_examples=1000)
    )
    results = []
    print("DERIVED", len(inc), logical_root, flush=True)
    for rn, mode in ORDER:
        rid = f"{rn}-{mode}"
        print("START", rid, flush=True)
        res = run(mode, rid, inc, examples)
        results.append(res)
        (EVID / f"{rid}-raw.json").write_text(json.dumps(res, sort_keys=True, indent=2))
        print("DONE", rid, res["total_seconds"], res["corridor_valid"], flush=True)
    (EVID / "raw_nine_sample_report.json").write_text(
        json.dumps(
            {
                "dataset": "M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1",
                "logical_root": logical_root,
                "order": ORDER,
                "results": results,
            },
            sort_keys=True,
            indent=2,
        )
    )
    (EVID / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "tome_commit": "474a0b6ca051c91cdfe9d076c3a4e275a22eba51",
                "contract_commit": "7ae11364b771eb4639dbb10268f9bd71bd3da601",
            },
            sort_keys=True,
            indent=2,
        )
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
