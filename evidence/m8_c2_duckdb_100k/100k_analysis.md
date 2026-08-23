# M8 C2 100K DuckDB store-only scale result

The vectorized DuckDB store completed the streamed 100,000-source fixture:

- 12,800,000 candidates generated;
- 10,661,700 eligible memberships;
- 46 nonempty pools;
- one rank materialization;
- 46 reserve queries;
- 2,677 bounded cursor fetches;
- 10,661,700 rows streamed;
- zero OFFSET queries.

The complete measured wall time was 560.62 seconds (9m20.6s), with 785.49 seconds of process CPU time. Generation plus ingestion/ranking took 521.92 seconds; ordered reserve streaming and digest validation took 38.25 seconds. Peak observed database file size was 1,468,280,832 bytes and peak spill usage was 1,701,871,616 bytes. Scratch use stayed below 25 GiB and free disk stayed above 91 GB.

Structural and canonical-order checks passed. Each pool was streamed once, rank ordering was monotonic, pool digests were recorded, and the aggregate ranked root was:

`sha256:3e99131dbe3b03c114d39df042b6cfa273aaf722d53c538336e5b30714e00a72`

The process RSS high-water was 1,610,735,616 bytes, approximately 1.5001 GiB: just above the stated 1.5 GiB hard boundary and above the preferred 1 GiB target. The run completed without sustained host failure, but this means the strict memory acceptance condition is not met. The in-process safety monitor also observed a transient boundary crossing and exited; the external run completed safely and cleanup succeeded.

No production code changed. No C3, C4, C5, artifact writing, GPU, Modal, teacher, tokenizer, or Contract work was performed. The database and spill scratch were deleted after the report was written.
