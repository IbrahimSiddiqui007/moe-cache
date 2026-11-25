# Benchmarks: runs, claims and the board

What this folder is for: every number we publish (README, `docs/results/benchmark-card.html`, release notes) is a **claim** that points at the **runs** that produced it,
and whether it still holds is **computed**, not remembered.

| File | What it is |
|---|---|
| [BOARD.md](BOARD.md) | **Start here.** Generated: one row per machine, model, setting and arm with the newest number, its date, its status and its claim |
| [JOURNAL.md](JOURNAL.md) | Generated: one line per run set, newest first |
| [runs/](runs/) | One record per run set (`DATE_MACHINE_SHORTID.json`), append-only, never edited: the raw evidence |
| [claims/](claims/) | One file per published number: the statement in plain words, the runs behind it, what it is valid for |

Status of a claim: **current**, **stale** (the moe-cache version or the llama.cpp pin moved past what the claim is valid for; the number may still hold, nobody has
re-measured), **superseded** (a newer claim covers the same cell) or **withdrawn** (written by hand, with the reason). A value nobody measured is `not recorded`.

Runs that were converted from [docs/results/archive/results-windows-desktop.md](../docs/results/archive/results-windows-desktop.md) (everything before the record format existed) say so in their `notes`: only the numbers written
down at the time are in them, and VRAM, process RAM and CPU are `not recorded`. `docs/results/archive/results-windows-desktop.md` stays as the numbered history up to section 79.

Commands (all `python3 bin/moe_cache.py bench-record ...`; the format and the rules are in [docs/reference/benchmark-template.md](../docs/reference/benchmark-template.md), the schema in
[docs/reference/bench-record.schema.json](../docs/reference/bench-record.schema.json)):

| Command | Does |
|---|---|
| `ingest RECORD.json [--machine pc]` | copy a record into `runs/` under its canonical name |
| `claim add ID --run RUNID --vs ARM --title ... --statement ...` | create a claim; `claim show ID` and `claim list` print them with their status |
| `board [--check]` | write `BOARD.md` and `JOURNAL.md` (`--check`: exit 1 if out of date) |
| `lint [--warn]` | every performance number in README.md and the card needs `<!-- claim:ID -->` of a current claim |
| `replay ID [--run] [--record NEW.json]` | the commands that reproduce a claim; compare a new record with it inside a noise band |

Adding a result: run the test with `bench-report` or `bench-record assemble`, `ingest` the record, `claim add` for the sentence you want to publish, `board`, then put
`<!-- claim:ID -->` next to the number where you publish it.

## reference.csv (stored stock / default results)

`benchmarks/reference.csv` holds one row per measured reference arm (stock llama.cpp, or our launcher default) so the test chain does not measure it again every slot (rule R16). The key is model, quant, kit, llama build, machine, context, plan hash, protocol and apps open/closed; a row younger than 14 days with the same key is reused by the test chain (not shipped; its spec key `reuse_reference: true`), and a stock arm for a key that has a row is refused unless the spec gives a `remeasure_reason`. Rows are written by the test chain (not shipped) from a finished item, never by hand. The file starts empty (header only): no reference number exists yet. A row from one run is labelled "forecast (one run)" (R17).

Columns: `key` is the key fields joined with `|`; `plan_hash` is a short hash of what the arm **declares** about its plan (launcher flags, server flags, env words, engine, always-plugin), not of the plan the launcher resolves: two rows with the same hash were asked for the same thing, and `tools/plan_check.py` (run through `conversation_bench_pair.sh --plan-check`) compares the resolved plans (-np, -ub, KV types, --n-cpu-moe, load method and the rest) of two arms before a pair runs. `instrument` says what measured the number (R17), for example "the test chain (not shipped)" or "bench-record from the launcher's own timings"; an empty value is written `not recorded`. `source` is the run the numbers came from. `date`, `n_runs` and `spread_pct` say how old and how firm the number is; with fewer than 2 runs the `label` is "forecast (one run)".
