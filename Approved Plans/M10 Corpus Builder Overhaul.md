# M10 — Corpus Builder Overhaul

## Disposition

Implement from `ac23f0ea8a8a474d841e80f787003ac6492409ab` on
`m10/corpus-builder-overhaul`. M10 adds a Tome-owned,
self-describing `radjax_tome_corpus_artifact_v2`, strict
`radjax_tome_corpus_build_intent_v1`, bounded-memory deterministic construction,
verified shards/indexes, restart-safe publication, and the public commands:

```text
radjax-tome corpus build --config CONFIG
radjax-tome corpus validate ARTIFACT
radjax-tome corpus inspect ARTIFACT
radjax-tome --json corpus validate ./corpus.v2
```

No Contract, Student, Golden, M8, score-pass, selected-pass, GPU, or M11–M14
work is authorized.

## Authority and current state

`origin/main` is `ac23f0ea8a8a474d841e80f787003ac6492409ab`, directly descended
from audited M9 `7550ea9bcae23917fdaaee3d7506efaef849c6bf`; accepted M8
`88d6f418e0f39ab2ec61d9047f974f04657e5214` is an ancestor. Contract remains
pinned to `373e3d17060d4ce1c4a0db6065c9289da714bde7`. Existing local worktrees
are dirty or unrelated; use a fresh clean worktree and preserve `.DS_Store` and
`uv.lock`. Package support is Python `>=3.11`, setuptools, argparse, PyYAML,
and DuckDB `1.4.5`. Every commit appends `bible.md`.

## Current corpus and production evidence

`src/radjax_tome/corpora/builder.py` is the list-materializing v1 builder. It
supports text files and JSONL, normalizes CR/LF and trailing whitespace, chunks
by character windows, deduplicates normalized-text SHA-256, emits a v1 manifest
and report, embeds absolute paths, and has no shards, indexes, journal, or safe
resume. `corpora/loaders.py`, `prompts.py`, `splitters.py`, and
`tokenization.py` are compatibility/research paths. Reuse
`corpora/tokenizer.py` and `language_tokenizer_binding.py` without duplicating
fingerprinting.

Current production routes through `builder.config.CorpusIntent`,
`builder.config_io.load_tome_build_intent`,
`production_stages.preflight.validate_required_inputs`,
`builder/teacher_textbook.py`, `builder/backend_textbook.py`, and
`builder/delivery/rerun.py`; several readers parse JSONL independently. M10
adds one verified corpus-input adapter and preserves v1 readers for historical
configs. The fixed 25-field `selection_authority_payload_v1` remains exact.

## Product boundary

The closed M10 adapter set is `local_text_tree_v1` (`.txt`, `.md`, `.markdown`,
`.py`) and `local_jsonl_text_v1` (explicit text field and optional record-ID
field). Direct network/Hugging Face acquisition, arbitrary JSON selectors,
Parquet, crawling, token-window construction, fuzzy deduplication, distributed
execution, and corpus-quality models remain out of scope. Remote data must be
materialized as an immutable local snapshot first.

## Configuration, policy, and identity

Add strict `radjax_tome_corpus_build_intent_v1` with required `artifact`, ordered
`sources`, `policy`, `layout`, `resources`, `output`, `execution`, and
`reporting` sections. Policy explicitly declares normalization, filtering,
`char_window_v1` chunking, exact deduplication, ordering, and tokenizer binding.
JSON/YAML share one loader; duplicate/unknown/missing fields, invalid types,
unsupported versions, environment interpolation, and hidden presets fail.
Paths resolve relative to the config file. Resume/overwrite are typed,
mutually-exclusive operational overrides.

Semantic identity consists of policy, tokenizer binding, ordered source
declarations, and ordered example identities. Operational identity consists of
layout and resource limits. Paths, timestamps, temporary locations, reports,
and journals are excluded.

For build-intent v1, preserve `selection_authority_payload_v1` and all 25 fields
without substitution. For build-intent v2, add an explicit
`selection_authority_payload_v2` containing non-corpus selection semantics,
corpus schema, corpus semantic identity, tokenizer-binding digest, and
`max_examples`; it must exclude `dataset_path`, `manifest_path`,
`artifact_path`, host paths, and shard layout. Moving a validated v2 artifact
therefore cannot change v2 authority.

## Sources, records, filtering, and deduplication

Create immutable `SourceRecord` and `CanonicalCorpusRecord` types. Decode UTF-8
strictly; convert CR/LF; strip per-line trailing whitespace and outer record
whitespace; preserve Unicode code points and controls; reject empty/below-minimum
records; retain current character chunking; and fail malformed JSONL, duplicate
keys, invalid UTF-8, and non-string text. Reports use closed reason codes such
as `CORPUS_FILTER_EMPTY`, `CORPUS_TRANSFORM_CHARACTER_CHUNKED`,
`CORPUS_DUPLICATE_EXACT`, and source parse failures.

Deduplicate globally on normalized UTF-8 bytes using SHA-256 and exact-byte
collision defense in private DuckDB. The winner is the minimum declared-source
ordinal, logical locator, and chunk index. Metadata does not participate;
duplicate provenance is retained. Disabling deduplication is an explicit
semantic policy.

## IDs, ordering, and tokenizer binding

Final order is `(declared_source_ordinal, canonical_logical_locator,
chunk_index)`. Example identity hashes source name, logical locator, chunk
index, and normalized-text digest. Public IDs remain `corpus_{ordinal:09d}`;
overflow fails explicitly. Worker count, enumeration order, shard capacity,
resume point, and absolute path cannot affect identity.

Reuse `TokenizerConfig`, `SmokeTokenizer`, `HFTokenizer`,
`capture_language_tokenizer_binding`, and pinned Contract validation. Capture
the binding from the same tokenizer used by production; store its evidence and
digest in v2. Do not store token IDs or introduce a second special-token rule.
Production verifies this binding before model load or accelerator allocation.

## Artifact and bounded storage

The v2 artifact contains `corpus_cover.json`, `corpus_manifest.json`,
`normalized_intent.json`, `source_manifest.json`,
`language_tokenizer_binding_v1.json`, vocabulary evidence,
`shard_inventory.json`, `shards/records-*.jsonl`,
`indexes/records-*.index.jsonl`, filter/dedup/build reports. Cover inventory is
closed and classifies semantic, provenance, physical-integrity, diagnostic, and
private members. Raw shard/index digests, counts, ranges, offsets, and identities
are cross-checked.

Use uncompressed canonical UTF-8 JSONL shards and matching offset indexes.
`VerifiedCorpusReader` verifies the entire shard and index before yielding any
row from that shard. DuckDB is private, one-threaded, memory-limited, and used
for external deduplication/order. Track live records, bytes, encoder chunks,
spill bytes, open files, and process RSS; prove bounds with synthetic scaling
tests and a subprocess measurement.

## Lifecycle and publication safety

Preflight validates configuration, adapters, dependencies, source/output
overlap, symlinks, destination state, permissions, storage, and resume/overwrite
before output mutation, source opening, tokenizer loading, workers, or progress
writes. Private states are:

```text
PREFLIGHTED → SOURCES_RESOLVED → INGESTING → INGEST_COMPLETE
→ SHARDS_SEALED → VALIDATED → PROMOTION_INTENT → PROMOTED
```

New-directory publication may use same-filesystem no-replace rename plus parent
fsync and may be described as atomic visibility. Replacement of an existing
artifact is not atomic: journal old/new identities, rename the owned old target
to a sibling quarantine, fsync and record `OLD_QUARANTINED`, rename validated
staging into the target, fsync and record `NEW_PROMOTED`, revalidate, commit,
then remove or retain the backup. Recovery deterministically restores or
completes based on journal state and topology. Reports must never claim atomic
old/new visibility. Unrelated paths are never deleted.

## Validation, CLI, and M5/M4 integration

Add typed `CorpusIssue`, `CorpusValidationResult`, and `CorpusInspection`, plus
`load_corpus_build_intent`, `build_corpus_artifact_v2`,
`validate_corpus_artifact_v2`, `inspect_corpus_artifact_v2`, and
`open_verified_corpus`. Extend `tome.artifact_dispatch`; preserve v1 dispatch.

Add the M9 `corpus` group in `cli/mainline.py`. `--json` remains global and
precedes the group; JSON is one stdout document, progress/warnings are stderr.
Legacy corpus flags remain research-only with deprecation. Use existing exit
codes: 0 success, 2 invocation/configuration, 3 unsupported, 4 validation,
5 destination conflict, 7 build failure, 130 interruption, 141 broken pipe,
1 unexpected failure.

Add `radjax_tome_build_intent_v2` with `corpus.artifact_path`,
`expected_semantic_identity`, and `max_examples`. Add
`CorpusArtifactReference`, `ResolvedCorpusInput`, `resolve_corpus_input`,
`open_corpus_input`, and `iter_corpus_examples`. V2 preflight verifies artifact
and tokenizer identity before expensive work; all production readers use the
shared adapter. No v2 identity enters the v1 25-field projection.

## File and commit sequence

Create `corpora/{config,sources,records,dedup,identity,storage,lifecycle,validation}.py`;
adapt `corpora/builder.py`, `corpora/__init__.py`, `builder/config.py`,
`builder/config_io.py`, `production_stages/preflight.py`, production readers,
`tome/artifact_dispatch.py`, `cli/mainline.py`, and `cli/main.py`. Update
`docs/CORPUS_BUILDER.md`, `docs/CLI_GUIDE.md`, `README.md`,
`docs/hydra_disposition.json`, and add focused M10 tests.

Use commits: `M10.1` authority/config; `M10.2` sources/policy/identity;
`M10.3` bounded dedup/shards/indexes; `M10.4` lifecycle/resume/publication;
`M10.5` validation/inspection; `M10.6` CLI and M5/M4 integration; `M10.7`
compatibility/docs/regression; `M10.8` one bounded review correction if needed;
`M10.9` closure evidence. Every commit appends `bible.md`.

## Tests, review, and closure

Test strict config, source ordering, malformed input, Unicode/UTF-8,
normalization, filtering, exact dedup/collision defense, worker/resume/layout
invariance, deterministic shards/indexes, raw-before-yield verification,
bounded memory, symlink/TOCTOU safety, every journal boundary, recoverable
overwrite failure, v1 compatibility, v2 relocation invariance, tokenizer
mismatch, zero-side-effect preflight, global JSON grammar, stdout/stderr
separation, exit codes, and M4/M5 reader convergence.

Run `pytest -q -p no:cacheprovider`, Ruff check/format, compileall,
`git diff --check`, and `python -m build`. A clean Python 3.12 wheel smoke
builds, validates, inspects, repeats, interrupts/resumes, relocates, and feeds
v2 into production preflight and the smallest existing deterministic CPU smoke.
No GPU, Golden regeneration, 10K, or 100K run is required.

Closure is recorded in `docs/M10_CLOSURE_REPORT.md`,
`evidence/m10_corpus_builder/closure.json`, and
`evidence/m10_corpus_builder/SHA256SUMS`, binding commits, Contract pin,
intent/source/tokenizer/policy identities, counts, order, shard/index inventory,
repeat/resume equivalence, memory evidence, CLI snapshots, production proof,
quality gates, reviewer disposition, push, and clean worktree.

## Risks and final recommendation

Stop only for inability to preserve v1 authority, Contract incompatibility,
accepted Golden/M8 semantic conflict, unavailable filesystem durability, or
conflicting history. Do not claim v1/v2 semantic equivalence, path-dependent v2
authority, atomic overwrite, network-source support, fuzzy deduplication, token
materialization authority, M8 performance, or later-milestone completion.

M10_PLAN_READY_FOR_ONE_SHOT_IMPLEMENTATION
