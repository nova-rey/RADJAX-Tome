# Native-v3 Student semantic declarations v3

`row_range_declaration` is a required semantic declaration. Its consumption
declaration is exactly `{"kind":"row_range_declaration"}`; its delivered
JSON body is interpreted by the v3 resolver as the authoritative row-range
statement and cross-checked against the selected resource counts.

`delivery_receipt` is a required semantic declaration. Its consumption
declaration is exactly `{"kind":"delivery_receipt"}`. Its delivered JSON is
the closed `native_v3_student_consumption_delivery_receipt_v2` body with one
of `one_pass_full` or `two_pass_rerun_selected`, both named-array encodings,
and source roles in this exact order: `native_v3_mode_assignments`,
`native_v3_score_shards`. It contains no physical source path.

`authority_reference` is a required semantic declaration. Its consumption
declaration is exactly `{"kind":"authority_reference"}`. Its delivered JSON
is the closed `native_v3_student_consumption_authority_reference_v1` body:
`selection_integration_config_hash`, either or both score-pass authority hash
forms, and optional `delivery_authority_hash`. This preserves authority v1/v2
compatibility without changing the selected v3 profile.
