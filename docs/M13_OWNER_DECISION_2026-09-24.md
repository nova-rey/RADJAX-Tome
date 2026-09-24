# M13 owner decision — versioned current-production reference

Date: 2026-09-24

Rey authorizes replacing the historical Golden coordinate comparison as M13's
hard acceptance gate with the separately versioned
`M13_CURRENT_PRODUCTION_REFERENCE_V1`. This is an acceptance-policy decision,
not a claim that historical parity passed.

The historical Golden fixture, comparator mode, raw reports, and failed
comparisons remain unchanged. The final policy-aligned comparison recorded:

- 128 common coordinates;
- 128 missing coordinates; and
- 128 extra coordinates.

The retained evidence also records upstream score/feature and configuration
differences. The historical producer commit/runtime and complete intermediate
score/C1--C5 tables were not preserved, so full causal reconstruction is not
possible. This decision does not assert that approved algorithm evolution
explains every difference, and it does not change tolerances, force Golden
coordinates, or alter teaching semantics.

M13 acceptance is therefore against the current reference's independently
bound inputs, selection rules, eligibility, quotas, ordering, linkage,
publication, and validation receipts. The historical comparison remains a
failed historical-parity result and must be reported as such anywhere it is
referenced.

M14 may use the current-reference gate only when it records the producing SHA,
reference identity, effective configuration, and current-reference comparison
explicitly. M14 must not silently treat the historical Golden as passed or
replace the immutable fixture.
