# Quality gate for the approximate mode

Written 2026-10-29 **before any approximate item was measured**. The thresholds below are fixed; an item that misses one is not kept, and a threshold is changed
only with a written reason and the owner's say-so (the table is the contract, not the measurement).

Exact mode (no `--fast`) keeps the token-identity tests (`tests/identity.sh`, `tests/kl_batch.sh`, CI job `identity`). The approximate items cannot be token-identical,
so they pass this gate instead: *the meaning has to match* (owner's rule 2026-10-29).

## What is measured (`tools/quality_gate.py`)

| Part | How | Against |
|---|---|---|
| KL divergence and top-1 agreement | `llama-perplexity`, 3 chunks x 512 tokens, `-b 8 -ub 8` (every step goes through the executor's small-batch path), `--kl-divergence`; the method of RESULTS 25.3 / R11 | stock llama.cpp, f16 KV, same model, same offload |
| Task score | the 202 exact-answer tasks of `tools/quality_tasks.json`, greedy, through `llama-server` (`tools/quality_yardstick.py run --tasks-only`) | the same tasks on the stock server |
| Loops (IQ2 failure mode) | the yardstick's repeat detector on the task answers | stock |

## Levels and thresholds

A level = the settings of one approximate package. A run passes when **all three** limits hold.

| Level | Settings on top of exact | Mean KLD at most | Same top token at least | Tasks right, at least stock minus | Based on (measured earlier) |
|---|---|---|---|---|---|
| `exact` | none (plugin only) | 0.00001 | 100 % | 0 (identical) | RESULTS 25.3, CI identity |
| `kv` | q8_0 KV cache (`-ctk q8_0 -ctv q8_0`) | 0.015 | 95 % | 2 | KL 0.0045 on its own, 0.012 on Vulkan (RESULTS 53, R11) |
| `fast` | route skip + q8_0 KV (today's `--fast`) | 0.040 | 90 % | 4 | KL 0.025, top-1 93.6 % (25.3) |
| `fewer` | `fast` + the lowest-ranked 2 experts dropped on CPU-held layers (B3) | 0.060 | 88 % | 5 | none: new; Q2_0 is already 80.6 % top-1 vs Q8_0 (Strata), errors stack |

Rules that apply to every level:

- No level may raise the number of looping task answers by more than 2.
- Thresholds are for the **target model** (Flash-Next Q2_0, PC queue). KL depends on the model and on the text, so the numbers on another model are not comparable.
- CI uses Granite 1B-A400M, which is far more sensitive than a 100 B model (few experts, 4 bits, tiny). There the gate is a **smoke guard only** (`--smoke`):
  mean KLD at most 0.25, same top token at least 70 %, tasks within 10, and `exact` must stay at KLD 0. It proves that every level runs, that
  the plugin really was active and that nothing went wrong by an order of magnitude; it does not prove quality.
- A speed gain only counts if the level passes. A level that is faster in every ABBA pair at 4k, 64k and 128k **and** passes is kept.
- The default stays exact until the owner decides with these numbers.
