# Benchmarks: runs, claims and the board

What this folder is for: every number we publish (README, `docs/benchmark_card.html`, release notes) is a **claim** that points at the **runs** that produced it,
and whether it still holds is **computed**, not remembered.

| File | What it is |
|---|---|
| [BOARD.md](BOARD.md) | **Start here.** Generated: one row per machine, model, setting and arm with the newest number, its date, its status and its claim |
| [JOURNAL.md](JOURNAL.md) | Generated: one line per run set, newest first |
| [runs/](runs/) | One record per run set (`DATE_MACHINE_SHORTID.json`), append-only, never edited: the raw evidence |
| [claims/](claims/) | One file per published number: the statement in plain words, the runs behind it, what it is valid for |

Status of a claim: **current**, **stale** (the moe-cache version or the llama.cpp pin moved past what the claim is valid for; the number may still hold, nobody has
re-measured), **superseded** (a newer claim covers the same cell) or **withdrawn** (written by hand, with the reason). A value nobody measured is `not recorded`.

Runs that were converted from [docs/results-archive/RESULTS.md](../docs/results-archive/RESULTS.md) (everything before the record format existed) say so in their `notes`: only the numbers written
down at the time are in them, and VRAM, process RAM and CPU are `not recorded`. `docs/results-archive/RESULTS.md` stays as the numbered history up to section 79.

Commands (all `python3 bin/moe_cache.py bench-record ...`; the format and the rules are in [docs/BENCHMARK_TEMPLATE.md](../docs/BENCHMARK_TEMPLATE.md), the schema in
[docs/bench_record.schema.json](../docs/bench_record.schema.json)):

| Command | Does |
|---|---|
| `ingest RECORD.json [--machine pc]` | copy a record into `runs/` under its canonical name |
| `claim add ID --run RUNID --vs ARM --title ... --statement ...` | create a claim; `claim show ID` and `claim list` print them with their status |
| `board [--check]` | write `BOARD.md` and `JOURNAL.md` (`--check`: exit 1 if out of date) |
| `lint [--warn]` | every performance number in README.md and the card needs `<!-- claim:ID -->` of a current claim |
| `replay ID [--run] [--record NEW.json]` | the commands that reproduce a claim; compare a new record with it inside a noise band |

Adding a result: run the test with `bench-report` or `bench-record assemble`, `ingest` the record, `claim add` for the sentence you want to publish, `board`, then put
`<!-- claim:ID -->` next to the number where you publish it.
