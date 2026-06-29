# RADJAX-Tome Bible

## 2026-06-29 — Tome Builder migration scaffold

Moved the Tome Builder / TeacherTextbook builder from the historical `qrwkv-xla`
repo into `RADJAX-Tome` with only the minimum required producer-side support code.
The historical repo remains read-only. This phase preserves existing builder
behavior and does not yet implement the new `cover_page.json` Tome contract.
