# M13 current-production reference v1

Status: **accepted for M13 under the dated owner decision**. This reference
does not replace the historical Golden fixture. Historical Golden parity remains
failed and is retained as a separate comparison.

## Identity

| Field | Value |
| --- | --- |
| Reference name | `M13_CURRENT_PRODUCTION_REFERENCE_V1` |
| Reference schema | `radjax_tome.m13_current_production_reference.v1` |
| Producing Tome commit | `a1ba7d308a2fe44065e1e05273609469547635d4` |
| Producing-commit evidence | `evidence/m13_golden_1k_10k_continuation/policy_aligned_provenance.json` |
| Acceptance/evidence commit | `0c6f5452f824ba2952642115da856801a5af5107` |
| Contract commit | `373e3d17060d4ce1c4a0db6065c9289da714bde7` |
| Current artifact semantic root | `sha256:bbd2e6204579d7d6cdf67010eacc4b4d6178e19a6900dbc2219aafe8d80cb182` |
| Canonical Tome semantic identity | `sha256:32d42c02ada3f761e799e7c6b1bb36b6bc51e26919ad308ce5d1ce9584a18175` |
| Artifact tree manifest | `sha256:694d2181cc3da1f3d834ef342fef3df0c02301484e1cdff820cfbd53309611ca` |

The producing SHA is the checked-out source commit recorded by the retained
policy-aligned run. The acceptance/evidence SHA is the later documentation and
evidence seal. A stale temporary harness field named `b5ce585` is not treated
as the producer identity; the retained provenance names `a1ba7d3`.

## Frozen inputs and execution

- Corpus: 1,000 examples from the recovered Golden corpus, corpus hash
  `sha256:518a5213981de49f52fd3e18a880d73aee31eec86eb6dedaf8b39a3c6f7ab878`.
- Corpus manifest:
  `sha256:7357fae12008fc3951f2ebfd819410c5c8d4612f12137d349031cd73a3186494`.
- Model directory:
  `sha256:77c344d305f916b3201311543888f5a95705c3e34de72f62610b9e47afb833d6`.
- Model configuration:
  `sha256:7fe7acd83287d352cd13a7477460025cea0eb3b7271b55812da55bd733c4fcc5`.
- Weights:
  `sha256:e68ec3300bde8aec6fb868b42a6b8ddfdef677a7c38dd279a639278a8b94a46b`.
- Tokenizer:
  `sha256:29aa73c8a911e1f39dddb0200fe8f4817fd369b8c0ce4f25aec6f1157e22690b`.
- Sequence length: `128`; vocabulary size: `262144`.
- Teacher backend: `gpu_torch`; recorded output dtype: `float32`.
- Runtime record: Torch `2.14.0+cu130`; selected teacher batch `1`; selected
  rerun batch `1`; no score pass or reselection was performed for acceptance.
- C2 candidate-pool cap: `4`; C3 corridor-mode cap: `10`.
- Selection integration policy: `corridor_first_global_backfill_v1`.
- Corridor budget: `128`; global budget: `128`; total selected coordinates:
  `256`; shortfall: `0`.

Authority hashes and raw intermediate digests are recorded in
`current_reference.json` and the retained `c6/authority_manifest.json`.

## Durable artifact location

The retained full output is available outside the checkout at:

`/home/nyx/m8g/m13-recovery/m13-current-production-reference-v1`

Its complete file manifest is
`REFERENCE_FILE_SHA256SUMS`, whose digest is recorded above. The original
temporary producer paths in the Modal receipt are retained as provenance only;
they are not the retention location.

## Acceptance checks and limitations

The retained output passed the existing production, C2--C5 selection,
selected-delivery, linkage, validation, and publication checks. The current
coordinate set is internally consistent and has 256 unique coordinates. The
controlled interruption/resume receipt is a separate three-record fixture and
is not a full-1K or full-10K interruption claim.

The historical Golden fixture remains immutable at semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
The final governed comparison recorded 128 common, 128 missing, and 128 extra
coordinates. Upstream score/feature and configuration authorities differ, and
the historical producer/runtime/intermediate score tables are unavailable.
This amendment therefore accepts the separately versioned current reference;
it does not claim historical parity, numerical equivalence, Student quality,
or training success.
