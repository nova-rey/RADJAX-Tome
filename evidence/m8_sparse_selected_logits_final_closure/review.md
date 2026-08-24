# Focused final sparse-logits review

Disposition: approved_with_nonblocking_findings; PASS WITH RESERVATIONS.

Dense and sparse used the same 64-source/71-coordinate workload, batch size 1, tokenizer, model, reducer, compact publication path, and 128-token inputs.

Equivalence correctly failed closed: 4/71 coordinates differ, including one Dynamic-K change (361 versus 421). Logical and body roots differ. The differing fields are authoritative teaching fields; no tolerance was widened.

Timing is reasonably controlled with three runs per mode, no fallback, and identical measured GPU allocation. The order was dense, sparse, sparse, dense, dense, sparse, so order/cache bias remains a nonblocking reservation. Medians are internally consistent: dense wall 24.592561 s versus sparse 24.210041 s (-1.5554%); dense teacher 7.290268 s versus sparse 6.864035 s (-5.8466%). The result is not actionable after equivalence failure.

Peak allocated GPU memory is identical at 1,755,066,368 bytes. Cumulative sparse traffic reduction is not presented as peak-memory savings. Capability handling fails closed for models without explicit logits_to_keep support; a minor residual risk remains for models accepting arbitrary **kwargs, but no fallback occurred in this run.

Final disposition: SPARSE_SELECTED_LOGITS_BEHAVIORALLY_DIFFERENT_REJECTED; M8_CLOSED.
