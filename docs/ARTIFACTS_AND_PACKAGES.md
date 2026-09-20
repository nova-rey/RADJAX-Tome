# Artifacts and packages

## Layers

1. Corpus artifact: verified source records, shards/indexes, reports, tokenizer
   binding, duplicate provenance, and a path-independent semantic identity.
2. Producer workspace: Tome cover, manifests, target shards, selection/linkage
   reports, provenance, and publication receipts.
3. Archive/transport: directory or tar transport of an already validated
   workspace. Archive bytes are physical transport, not teaching identity.
4. Student package: a Contract-compatible handoff. Student owns training and
   consumption; Tome does not run it.

Keep corpus, producer artifact, package profile, Contract, and M5 intent schema
versions distinct.

## Identity and integrity

Semantic identity answers which governed evidence is present. Physical integrity
answers whether declared members, offsets, indexes, and digests are intact.
Authority/provenance answers which model, tokenizer, policy, source closure, and
selection decision produced the evidence. External attestation is an optional
comparison against an independently supplied expected identity.

Different archive bytes can carry the same semantic teaching content. Passing
integrity validation does not prove model quality or Student convergence.

## Public lifecycle

    radjax-tome validate ./producer-workspace
    radjax-tome inspect ./producer-workspace
    radjax-tome package ./producer-workspace \
      --output ./student-package --profile student --transport directory
    radjax-tome validate ./student-package

Package creation is no-clobber by default and requires an explicit overwrite
for an owned destination. A synthetic dynamic CPU artifact may validate as a
producer workspace while remaining ineligible for Student because corridor
artifacts are absent; do not fabricate them.

The repository Contract pin is
373e3d17060d4ce1c4a0db6065c9289da714bde7. Student compatibility is established
by the selected Contract profile and validators. This page does not claim a
Student training command or a training run.
