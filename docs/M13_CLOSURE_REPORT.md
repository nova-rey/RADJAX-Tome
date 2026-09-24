# M13 Golden 1K parity and 10K scaling rehearsal

Status: **M13_CLOSED_UNDER_VERSIONED_REFERENCE_ACCEPTANCE**. Historical
Golden parity remains failed; the current-production reference is accepted
under the dated owner decision in docs/M13_OWNER_DECISION_2026-09-24.md.

## Historical record and revised gate

The run used the integrated M12 authority at `97907b3681e49bfc107866d480596981c2eccaa6`, source commit `e3b3ef2`, and Contract `373e3d17060d4ce1c4a0db6065c9289da714bde7`. Lightning recovery selected the evidence-supported stopped `radjax-tome-1` Studio and recovered the exact Golden corpus `sha256:518a5213981de49f52fd3e18a880d73aee31eec86eb6dedaf8b39a3c6f7ab878` with manifest identity `sha256:7357fae12008fc3951f2ebfd819410c5c8d4612f12137d349031cd73a3186494`. No teacher or GPU work ran on Lightning.

The canonical public CLI completed a real-teacher 1K run (1,000 examples, one full teacher pass, 256 selected coordinates, selected rerun batch size 1) and a real-teacher 10K run (10,000 examples, 1,280,000 scored positions, one full teacher pass, 256 selected coordinates, 254 batch-1 selected rerun batches). Both production reports, selection integration, selected delivery, and local validation reports were `pass`; no CUDA OOM or fallback occurred. The 10K run took 3,539.102 s end-to-end, with 2,396.943 s in the score/main pass and 231.010 s in selected delivery. Peak recorded host memory was 7,956,008,960 bytes and selected-rerun peak device memory was 2,899,674,624 bytes.

The frozen Golden fixture itself validates offline (256 coordinates; semantic root `sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`). Golden comparison was not accepted: the existing comparator cannot project the current `compact_k_monolithic` payload shards (`golden capture payload arrays or effective_top_k are invalid`), and the observed 1K selected-coordinate set differed from the fixture (130 missing and 130 extra). No parity claim or tolerance widening is made. The 10K result is preserved as scaling evidence, not as Golden parity.

The CPU postflight package/transfer attempt was stopped after 15 minutes without a receipt; raw outputs remain durable on the task-owned Modal volume and local retrieval. The earlier local validator also correctly rejected the copied report's runtime `/work` dataset path outside its original mount. These are recorded as postflight limitations, not silently treated as passes.

See `evidence/m13_golden_1k_10k/` for recovery inventory, raw run summaries, parity failure, postflight attempt, teardown, and checksums. No main merge or M14 work was performed.

## Owner-approved current-production acceptance
Rey approved a separately versioned current-production reference as M13's
acceptance oracle. The historical Golden fixture, comparator, and failed
reports remain immutable and are not reclassified as passing. The final
policy-aligned retained 1K run is named
M13_CURRENT_PRODUCTION_REFERENCE_V1; its producing SHA is
a1ba7d308a2fe44065e1e05273609469547635d4, its acceptance/evidence SHA is
0c6f5452f824ba2952642115da856801a5af5107 (audited implementation), and its Contract pin remains
373e3d17060d4ce1c4a0db6065c9289da714bde7.

The accepted reference binds the recovered corpus and manifest, teacher and
tokenizer identities, effective Torch/dtype/runtime record, C2 cap 4, C3 cap
10, 128/128 corridor/global budgets, selection configuration hashes, current
semantic roots, and retained raw intermediate digests. The complete retained
output is outside the checkout at
/home/nyx/m8g/m13-recovery/m13-current-production-reference-v1; its file
manifest digest is recorded in
evidence/m13_current_production_reference_v1/current_reference.json.

The historical comparison remains an explicit failure: 128 common, 128
missing, and 128 extra coordinates. Upstream score/feature and configuration
differences are recorded, and the historical producer/runtime/intermediate
score tables are unavailable. M13 acceptance therefore does not assert that
algorithm evolution explains every difference, does not widen tolerances, and
does not claim numerical equivalence, Student quality, or training success.

The accepted 10K run, package/transfer and revalidation receipts, and
controlled interruption/resume receipt remain valid inherited evidence. The
resume probe covers a separate three-record fixture at
after_v4_shard_sealed; it is not a full-10K interruption claim. See
docs/M13_CURRENT_REFERENCE_V1.md and
evidence/m13_current_production_reference_v1/.
