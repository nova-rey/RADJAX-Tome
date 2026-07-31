# M6D Native and Portable Conformance Evidence

The portable validator is the single independent implementation introduced in
M6B. M6D uses it against native writer outputs; it does not create another
portable validator.

Canonical v3 directories, uncompressed `.rtome`, deterministic gzip transport,
and both profile packages must pass native and portable validation. Profile
inventories may differ, but their identity digests must agree.

There is one intentional validation difference: native bundle validation is a
producer-canonicality check and rejects noncanonical tar headers. Portable
consumer validation accepts a safe archive with noncanonical headers and emits
`transport_noncanonical`; its strict mode rejects that warning. Both paths
reject unsafe, corrupt, duplicate, linked, special, missing, extra, and
digest-mismatched members. This difference does not weaken safety or semantic
integrity.
