# Configuration reference

Configuration is owned by canonical parsers, not by this page. JSON and YAML
load through the same strict M5 loader; unknown fields, missing required
fields, duplicate keys, and unsupported schema versions fail closed.

## Production intent versions

* radjax_tome_build_intent_v1 is compatibility-supported. Its corpus section
  uses dataset_path, corpus_manifest_path, and max_examples.
* radjax_tome_build_intent_v2 is the current artifact-reference form. Its
  corpus section has exactly artifact_path, expected_semantic_identity, and
  max_examples; all other production sections retain the v1 M5 shape.

Both forms resolve to one canonical production configuration and one
ResolvedCorpusInput. The fixed selection_authority_payload_v1 remains the
exact 25-field authority projection. Corpus semantic identity, policy identity,
source closure, tokenizer-binding digest, and shard inventory are carried in
the broader run/resume authority.

| Section | Important fields | Notes |
| --- | --- | --- |
| teacher | model, tokenizer_id, backend, runtime_mode, provenance path | Resolvable before backend creation |
| corpus | v1 paths or v2 artifact reference, max_examples | v2 identity is checked |
| behavior | target policy, sequence length, vocabulary, top-K/buckets | Representation policy |
| corridor_policy | warning thresholds, side-board caps, full-width policy | Policy bounds |
| selection | delivery path, budgets, replay references | Selection authority |
| execution | batch policy, shard sizes, resume/overwrite | Production presets use batch 1 |
| outputs | workspace and report/manifest paths | Config-relative paths |
| compatibility | historical authority paths | Null unless reproducing authority |
| package | profile, transport, Contract version | Packaging is separate |

Nullable values are intentional only where the parser defines them. Do not infer
defaults from examples. Operational overrides are limited to output, resume,
overwrite, and equivalent lifecycle controls; they do not alter selection
semantics.

Complete examples:
* docs/examples/m9_tome_build_intent.yaml is a complete v1 YAML example.
* docs/examples/m12_tome_build_intent_v2.example.json is a complete v2 JSON
  template with explicit placeholders.
* TUI Save As writes the same canonical document consumed headlessly.

Run safe preflight with:

    radjax-tome build --config ./tome-intent-v2.json --preflight-only

Representative errors are unknown/missing field, schema-version mismatch,
CORPUS_TOKENIZER_BINDING_MISMATCH, semantic-identity mismatch, destination
conflict, incompatible resume identity, and unsupported package profile.
