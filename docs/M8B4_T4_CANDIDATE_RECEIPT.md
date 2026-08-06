# M8B.4 T4 Candidate Receipt

The private candidate report SHA-256 is
`bf467b4f62537593874acd5c526f21c98072419130536720f786b4211891f608`.
It was collected at `ce4398a7b41c9213e4c26151cc0261db5341004b` against the
M8B.1 report digest
`7b39ce34efe8dbf47022b24ec5052998c4767d3fdca47d8692ea84dcefe8871f`.

The candidate failed: median initial staging was 712.649 seconds versus
236.268 seconds baseline, and selected-pass wall was 842.997 versus 262.905
seconds. Both frozen improvement predicates fail. The implementation was
therefore reverted in `a175aa1`; this receipt records no justified production
optimization. Raw path-bearing evidence remains outside the repository.
