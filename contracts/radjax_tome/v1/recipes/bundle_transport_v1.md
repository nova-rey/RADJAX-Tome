# Bundle Transport v1

Semantic identity is independent of wrapping. Producers must emit root-level,
sorted regular-file tar members with `mtime=0`, `uid=0`, `gid=0`, empty owner
names, and mode `0644`; gzip uses `mtime=0` and an empty original filename.

Consumers always reject unsafe or corrupt transport: absolute or traversal
paths, duplicate members, links, special files, unsupported formats, missing
or extra manifest members, and digest or size mismatches. Safe archives whose
container metadata is noncanonical are accepted with
`transport_noncanonical`; strict conformance mode rejects that warning.
