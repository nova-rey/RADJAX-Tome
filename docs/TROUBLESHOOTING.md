# Troubleshooting

Use the smallest safe diagnostic first. Preserve destinations and reports when
a command fails.

| Symptom | Likely cause | Safe diagnostic | Repair or stop condition |
| --- | --- | --- | --- |
| unknown field or schema-version error | wrong intent dialect or typo | build --config ... --preflight-only | use a complete v1/v2 example |
| CORPUS_TOKENIZER_BINDING_MISMATCH | corpus and production tokenizer differ | inspect corpus binding and teacher provenance | bind the actual tokenizer; do not edit a digest |
| semantic identity mismatch | artifact is not the declared corpus | corpus inspect and compare identity | point intent at the verified artifact |
| destination exists or is unrelated | unsafe overwrite/resume | inspect ownership | new path or explicit overwrite for an owned path |
| interrupted build | durable boundary incomplete | inspect status/journal and validate | resume only when lifecycle says compatible |
| package profile rejected | producer lacks required members | validate and inspect workspace | use correct profile or complete real corridor selection |
| missing TUI dependency | textual extra absent | radjax-tome tui --help | install the tui extra; headless CLI remains supported |
| non-TTY or small terminal | unsafe interactive layout | use help and headless config | run canonical CLI; do not script keystrokes |
| JSON mixed with logs | --json was placed after command | put --json before command | parse stdout; diagnostics stay on stderr |
| optional HF import failure | teacher extra absent | radjax-tome doctor | install pinned extra or use CPU fixture |
| validation passes but quality is poor | integrity is not quality | inspect provenance and governed comparison | investigate model/data authority |

Do not retry blindly when an error names authority mismatch, destination
conflict, or unsupported package. Preserve the JSON result and workspace.
