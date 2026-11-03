# ENGINE-001: helper-thread prefetch for fit-in-RAM speed (design check, 2026-11-03)

Verdict: **not built. Nothing in the idea differs from what was already measured and killed.** Recorded so the row can be closed.

## What the idea says
A helper thread reads ahead so a model that fits in RAM decodes faster. The owner's note: CPU and GPU memory bandwidth do not add up on the shared bus, so this was left open as the lead.

## What exists and what was measured

| Earlier work | What it did | Result | Source |
|---|---|---|---|
| `MOE_CACHE_LOOKAHEAD=1` (in the plugin, `src/mc_state_io.inl`, helper reader threads) | while layer N computes, guess layer N+1's experts from the router weights and read them | -12 % to -74 % | RESULTS.md (X1), archive/README.md |
| Block prefetch, `archive/plugin/gpu_prefetch.patch` | upload the next block while the current one computes | 34k: 598/593 off, 598/597 on; 128k: 414 vs 418 (no gain) | archive/README.md |
| Competitor-style helper thread (Strata `adapt_async_default.hpp`) | same as look-ahead | -5.5 % on CPU-bound boxes; their own best 32 GB run set `STRATA_LOOKAHEAD=0` | deepdive_strata_maya_20261029.md SM12 |
| Startup fill (`preload()`, `src/mc_state_profile.inl`) | when every CPU expert fits the cache, all are read in layer order before the first token | in the plugin, on by default | code |
| Lookahead in other engines | BigMoeOnEdge: "prefetchers and route-ahead lost to the device cache"; llama.cpp #28766: gain only with 5 tokens of look-ahead | - | scout10_20261030.md |

## Every form a helper thread could take for a fit-in-RAM model

| Form | New versus the killed ones? | Why it cannot gain | Exactness risk | Time to build |
|---|---|---|---|---|
| Read next layer's experts early | no, this is `MOE_CACHE_LOOKAHEAD` | measured loss | none | built |
| Warm the experts that are not in RAM yet | no, the startup fill does it before the first token | after the fill nothing is left to warm; the fill is synchronous so it costs no decode time | none | built |
| Touch resident pages from a helper so they stay mapped | partly (it is the view-mode idea) | with everything resident there is no eviction, so no page faults to hide; with a tight cache it is look-ahead again | none | 1 day |
| Helper thread that copies weights while the CPU computes, to "use" idle bandwidth | yes | CPU and GPU share one bus: a copy that reads the same bytes the dot products read competes with them; the decode step is bandwidth bound, so any extra read slows it | none | 2 days |
| A second reader helping the CPU matrix product (more bandwidth threads) | no, `-t` / `MOE_CACHE_EXEC_THREADS` is that, and spin gave -0.8 to -2 % (N30) | the bus is already full at the tuned thread count | none | built |

## Decision
Kill rule met on paper: the only unmeasured form (copy helper) cannot read fewer bytes than the compute, and decode of a RAM-resident model is limited by bytes read. No code, no switch. Ledger state: `killed (no new form)`. Retest only if a machine shows a decode step with the memory bus clearly not full (a measured bandwidth below ~70 % of the copy benchmark `tests/exec_bw_bench.cpp` at the tuned thread count) while experts are all resident.
