# Focused independent review

Disposition: BLOCKED for adoption.

The governed smoke failed before publication acceptance at corpus_000000662 position 6: expected entropy 0.84765625, observed 0.85546875, delta 0.0078125 versus tolerance 0.00390625. The top token ID matched, but no sparse-truncated output or T4 timing sample is acceptable.

The reviewer found prefix accounting internally consistent, confirmed original-policy tokenization before slicing, and found deterministic per-batch union/remapping. Focused tests pass, but synthetic tokenizer tests do not independently prove real EOS/left-padding behavior.
