# M9 CI Reconciliation Handoff

The original M9 worktree is intentionally preserved at
`/home/nyx/radjax-tome`.

- branch: `m9/opinionated-mainline-cli-reconciled`
- clean HEAD before continuation: `1d4910a8530284df7518160b22a63320f217097b`
- dirty files: `bible.md`, `src/radjax_tome/backends/hf_torch.py`

The preserved uncommitted diff was:

```diff
 bible.md                             | 2 ++
 src/radjax_tome/backends/hf_torch.py | 3 +--

diff --git a/src/radjax_tome/backends/hf_torch.py b/src/radjax_tome/backends/hf_torch.py
@@
     "dense_logits",
     "topk_with_tail_v0",
     "cascaded_soft_labels_v1",
-    "dynamic_cascaded_soft_labels_v1",
     "corridor_exemplar_v1",
@@
-                "and cascaded_soft_labels_v1, dynamic_cascaded_soft_labels_v1, corridor_exemplar_v1"
+                "and cascaded_soft_labels_v1, corridor_exemplar_v1"
```

The matching `bible.md` addition describes this as an HF capability-admission
correction. It was not copied into this continuation blindly. Focused testing
confirmed it makes the unsupported dynamic-policy test pass, but accepted M8
also fails that test; it is therefore a pre-existing compatibility correction,
not an M9-introduced regression. It remains excluded from this branch pending
an explicit in-scope decision.
