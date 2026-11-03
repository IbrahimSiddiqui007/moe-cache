# Benchmarks: one folder per test day

Each test day is one folder, `benchmarks/YYYY-MM-DD/`:

```
benchmarks/
  README.md              this page: how to read a day, and the index
  2026-11-03/
    README.md            the day page: hardware, method, results, what each flag did, limitations
    records/*.json       the raw records (one per run set), when the day was measured with bench-record
```

`docs/RESULTS.md` stays as the numbered history, sections 1-79. **New results go here**, one day folder per date.

## How to read a day page

1. The first lines say the machine, whether the day was clean (PC restarted, baseline guard) and what the headline is. A page that was converted from an old RESULTS section says so in bold, and lists which metrics were never recorded.
2. **Hardware**, **Software and versions** and **Method** say what ran and how (order, runs, cold or warm file cache, prompts).
3. **What changed since the last run** links the previous day.
4. **Results per model and config** has one table per model and config with the per-run numbers: prompt and decode tok/s, VRAM, process RAM, CPU, GPU expert layers, context, `-ub`. **What each flag did** has the A/B tables.
5. **Limitations** lists everything not measured. `not recorded` means nobody measured it; it is never a guess. A number that we computed from other numbers says "my arithmetic".
6. **Reproduce** has the commands and **Raw data** links the `records/` files.

## The record format

- [docs/BENCHMARK_TEMPLATE.md](../docs/BENCHMARK_TEMPLATE.md): the fields, how each is collected on Linux and Windows, the day-page template.
- [docs/bench_record.schema.json](../docs/bench_record.schema.json): the record as JSON Schema.
- Draft a page from the records of a day: `python3 bin/moe_cache.py bench-record day benchmarks/YYYY-MM-DD --title "Title" --out benchmarks/YYYY-MM-DD/README.md`. Then fill in the lines marked *write by hand*.

## Index (newest first)

| Date | Machine | What was tested | Headline result | Page |
|---|---|---|---|---|
| 2026-11-03 | Windows 11 desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5) | v0.3.5 against v0.3.4 on the official b11966 server at 34k / 64k / 128k; 256k `-ub`; 122B and GLM guards; planner rule; Windows priorities; `-ub 512` decode check; decode-tier kill check | prompts +38 % to +42 %, decode -6 % to -10 % after long prompts; decode tier killed | [2026-11-03](2026-11-03/README.md) (converted, resources not recorded) |
| 2026-11-02 | same | next-release defaults against v0.3.4 with patched server (superseded); 64k placement follow-up; 64k default check | prompts +35 % to +56 %, 64k decode -15 % traced to 40 instead of 39 CPU layers | [2026-11-02](2026-11-02/README.md) (converted, resources not recorded) |
