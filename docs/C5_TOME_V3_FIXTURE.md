# Program C: v3 adoption fixture and compatibility proof

This document records the deterministic ordinary-CPU proof fixture for the
opt-in Tome Artifact Contract v3 path. It is adoption evidence, not a default
migration or a production-performance claim.

## Fixture identity

The committed fixture is `tests/fixtures/tome_artifact_v3_smoke/`:

* Contract package `0.9.0`, tag `v0.9.0`, peeled commit
  `1fa43e1aea2e198511db86dafb0aeefa525d48c7`;
* artifact identity `radjax_tome_artifact_contract@3.0.0`;
* profile `selected_exemplar_semantic_profile_v3`;
* four selected records and two final shards (capacity two);
* directory semantic root
  `sha256:f4b765aa0ca3dcd2e551d431e239e7504c5c34ef22c50d9766544a73de934ce2`;
* archive SHA-256
  `sha256:8336d925e520fddf3f59c740cdc7f71c100e76d7aa42aca761e6bc3a1b3585d1`;
* archive size 6493 bytes.

`FIXTURE_RECEIPT.json` is the sanitized machine-readable fixture receipt.
`governed_expectation.json`, `external_attestation.json`, and
`archive_receipt.json` are deliberately separate from the artifact under test.
The archive receipt is a Contract v3 external raw-integrity receipt; the other
two files are comparison/attestation inputs and are never discovered from the
package. The raw source/model inputs and their path-bearing build reports were
retained outside the repository after construction.

## Construction boundary

The fixture was created with the existing canonical production builder on CPU,
using `--artifact-contract-version v3`, the C6 canonical path, four selected
records, and `--payload-records-per-shard 2`. The v3 publisher consumed the
typed finalized selected-record handoff, including already-materialized payload
values retained only for this explicit v3 projection. It did not reread v4,
staging JSON, reports, or other emitted files, and it did not repeat scoring,
selection, or selected-pass execution.

The reproducible command shape (with machine-specific input/output paths
supplied by the fixture builder) was:

```text
PYTHONPATH="$PWD/src:/Users/Cooper/code/RADJAX-Contract/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -c \
'from radjax_tome.builder.production import build_production_gpu_tome; ...' \
--artifact-contract-version v3 --device cpu --payload-records-per-shard 2
```

The ellipsis above denotes only path/configuration arguments and is not a
fixture writer or a semantic expansion rule; the committed artifact and
receipt contain the resulting identities.

The public result is two separately journaled transactions: `artifact.v3/` and
`artifact.v3.tgz`. Both validate and share the semantic root. No claim of
atomic visibility for the directory/archive pair is made. Private journal
records, temporary paths, transaction IDs, timestamps, and recovery state are
not present in either artifact or in the public receipt.

## Proofs exercised

`tests/test_tome_v3_fixture_receipt.py` proves:

* standard validation of directory and `.tgz` transports;
* accepted `.rtome` transport wrapping of the already-promoted v3 directory;
* governed success with the external expected root;
* required external-attestation success and rejection of evidence supplied from
  inside the artifact;
* Contract archive-receipt validation;
* raw shard corruption and cover/graph incoherence rejection;
* pre-yield rejection for corruption in the first shard and before yielding any
  row from a later corrupt shard;
* semantic-root invariance under capacities one and four;
* a fully coherent changed-token replacement passes standalone validation but
  fails governed comparison and the original external attestation.

Existing focused and full Tome tests additionally cover the v2 default, v4/v5/
v6 historical behavior, fixed 25-field authority projection, journal PC39–PC47
state transitions, and the released Contract pin/mirror parity.

The fixture demonstrates adoption correctness and compatibility only. It does
not claim Student training, accelerator execution, M8 batching or performance,
default v3 migration, or truthful model-production origin.
