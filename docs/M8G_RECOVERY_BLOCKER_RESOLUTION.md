# M8G recovery blocker resolution

| Finding | Corrective surface | Evidence |
| --- | --- | --- |
| Valid pre-body journals were quarantined | `recover()` cleans private partials and returns `restart_ready`; empty journals are restartable | recovery matrix: early boundaries |
| BODY_PROMOTED lacked manifest recovery | `commit()` pre-stages `manifest.tmp`; `_resume_manifest()` validates and promotes it | state 5–8 fault cases |
| State 9 could skip inventory | `_resume_inventory()` validates the pair, idempotently binds inventory, then appends states 10–12 | post-manifest fault cases |
| Binary/JSON authority gap | binary FV3 receipts are decoded and validated; missing mirrors are regenerated; JSON-only journals quarantine | receipt matrix |
| Archive overwrite/digest gap | archive uses exclusive publication and validates body/manifest bytes against inventory digests | conflicting archive test |
| Configuration identity alias | explicit/profile-derived configuration identity is receipt-checked and included in tx identity | configuration swap test |
| Crash coverage was absent | opt-in named fault boundaries and 46 individually collected recovery tests | `test_immutable_body_recovery_matrix.py` |

The transaction archive is the opt-in M8G resource archive for the body store;
it is not substituted for the full canonical Tome producer package, whose
cover/resource requirements remain enforced by the existing package layer.
