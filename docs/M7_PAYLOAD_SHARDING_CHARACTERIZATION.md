# M7 Payload Sharding Characterization

## Locked baseline

The canonical Path B producer writes
`selected_exemplars/selected-exemplars-*.json` with schema
`selected_exemplar_payload_shard_v1`. Native streamed delivery currently emits
one selected payload record per file, stages those files under
`.staging-native-c6`, and promotes them only after the complete contiguous
record-index set validates.

The public `StudentTomeReader` and package helpers still expose an eager
`selected_payloads` collection. Package manifests and selected-linkage
validation read each current JSON envelope, even where they retain only scalar
summary data. This is the M7 bounded-memory target; it is not evidence that
the current implementation is already streaming.

## Identity consequence

The v3 identity derivation includes `selected_exemplars/*.json` in its semantic
JSON glob. Regrouping unchanged records across those files therefore changes
the v3 semantic digest. M7 must not reinterpret that historical result.
Canonical shard-size-independent identity requires the approved v4 cover and
semantic-identity-v2 contract, with v3 retained as a native historical format.

## Field census

`builder.delivery._shared._REQUIRED_SELECTED_PAYLOAD_FIELDS` is the complete
pre-M7 producer field census. M7B must classify every field as a required v4
semantic field, an explicitly nonsemantic layout/integrity field, or a
documented opaque extension before any writer migration. No field may be
silently omitted or inferred.

## Protected baseline

M7 changes physical selected-payload layout only. Native Path B orchestration,
late selected-linked corridor ordering, OOM reduction, staging/resume failure
normalization, authority v1/v2, the fixed selection projection, profile
meaning, and the immutable Golden fixture remain unchanged.
