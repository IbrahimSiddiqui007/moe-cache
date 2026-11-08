# GPU expert cache ("bookshelf in VRAM"): design and kill check (2026-11-08)

Branch `claude/gpu-expert-cache`. Replaces the "not built" status of `GPU_TIER_DESIGN.md` for the bank part. All numbers below are replay or arithmetic unless marked measured.

## 1. What already exists (read first)

CACHE-011 `--hybrid` / `MOE_CACHE_HYBRID` (src/mc_hybrid.inl) is already a per-expert GPU bank: filled from the usage profile (global ranking by count), the GPU computes the picks that hit, the CPU computes the misses at the same time,
and `hybrid_adapt` swaps the most-missed experts for the coldest resident expert of the same layer (defaults every 8 tokens, 16 swaps, decay 0.97, shadow "duel" against the profile bank). The bank is freed during long prompts and reloaded.
So a per-expert GPU cache with live swaps is NOT a new feature: the open question is why it measured slower on the PC (hybrid 21.8 vs plugin 28 tok/s, Q2_0, 34k), and whether a better rule or lower per-layer cost turns that around.

## 2. What differs from the old negative evidence

| Design | Where misses go | Sync | Result | Same as the new idea? |
|---|---|---|---|---|
| llama.cpp `--moe-cache-mib` | uploaded over PCIe, then GPU computes | every layer, uploads on the critical path | 9 vs 18 tok/s at 4 GiB, crash at 8/12 GiB | no: the new idea never uploads a miss, CPU computes it |
| plugin `--hybrid` (today) | CPU computes from RAM, in parallel | one fresh GPU graph, allocation, small copies, wait and read-back per layer per token | 21.8 vs 28 tok/s (PC, Q2_0); laptop 17.6 vs 12.5 (+41 %, IQ2_XS, slow CPU) | this IS it. Difference left: bank size, swap rule, per-layer cost |

## 3. Kill check 1: hit rate (replay, `tools/gpu_tier_replay.py`, Qwen3.6-35B trace, 40 layers x 256 experts, top-8, 10 requests x 299 decode tokens, leave-one-request-out profile)

Flash-Next has no full-layer trace (only a profile). Its profile is MORE skewed than the proxy's trace (top 12.5 % of experts = 62.6 % of picks, vs 43.7 %), so the proxy is the cautious column.

| Bank (share of all experts) | whole layers (today's `--n-cpu-moe` VRAM) | fixed, profile (today's hybrid start) | fixed per-layer ranges | rule today (every 4 tokens, 96 swaps) | decayed counts, per layer (every 4, 24 swaps, x0.7) | every token, 24 swaps | oracle per-layer |
|---|---|---|---|---|---|---|---|
| 6.25 % | 5.0 % | 23.3 % | 23.2 % | 44.2 % | 43.4 % | 46.4 % | 46.5 % |
| 12.5 % (= the PC's 6 GPU layers, about 9 GiB at IQ3_XXS) | 12.5 % | 36.3 % | 36.4 % | 60.7 % | 58.1 % | 62.0 % | 64.1 % |
| 19 % | 17.5 % | 46.6 % | 46.2 % | 70.5 % | 67.8 % | 72.2 % | 75.5 % |

Findings: (a) same VRAM as whole layers, hot experts serve 3-5x the picks; (b) live swaps are the big lever (36 -> 58-62 %); per-layer slot ranges and decayed counts add nothing over today's rule (ours already ranks by count); (c) swap cost at 12.5 %: 5.9-15.7 experts/token = 18-48 MiB/token (3.08 MiB each) = 1.5-4 ms at 12 GB/s if not overlapped. Caveat: 300-token requests; a 65k chat drifts more (needs a real long trace).

## 4. Kill check 2: decode time (arithmetic, PC Flash-Next IQ3_XXS: 20 tok/s = 50 ms, 48 layers, CPU holds 42)

The share of the token that is CPU expert work (f) is NOT measured on the PC (guess 40-60 %: 1.3 GB per token over about 50 GB/s DDR5 = 26 ms). Per-layer cost o = extra ms per hybrid layer per token (graph, allocation, copies, sync, read-back).

| f (CPU expert share) | bank hit | upload ms | gain if o = 0 | largest o per layer that still gives +15 % | gain at o = 0.25 ms/layer |
|---|---|---|---|---|---|
| 40 % | 58 % | 1.5 | +22 % | 0.05 ms | -6 % |
| 50 % | 58 % | 1.5 | +30 % | 0.10 ms | -1 % |
| 50 % | 61 % | 4.0 | +25 % | 0.07 ms | -4 % |
| 60 % | 58 % | 1.5 | +39 % | 0.16 ms | +4 % |

The measured hybrid deficit (28 -> 21.8 tok/s = +10 ms/token over 48 layers) implies o of roughly 0.2 ms/layer (it also had a worse bank), above every break-even in the table. **Verdict: hit rate passes (3-5x whole layers); +15 % decode is reachable only if per-layer cost falls to about 0.05-0.15 ms, i.e. by half or more. Today's code does not get there. Not a clean pass, not a clear kill.**

## 5. Plan if it goes on (each step has its own stop gate)

| Step | What | Gate | Agent h | CI h | Machine min |
|---|---|---|---|---|---|
| 1 | Per-layer timers in `hybrid_group` (build, copies, wait, read-back), on a small MoE that fits (LLaDA-MoE 7B on the laptop; Flash-Next only if the laptop GPU is free) | measure o; kill if o < 0.05 ms already (then the loss is elsewhere) or the loss is the CPU side | 2 | 0 | 30 laptop |
| 2 | Cut o: persistent per-layer graph for n_tok = 1 (built once, reused, ids and activation copied into fixed buffers, one stream, one event instead of sync), swap upload on a copy stream between tokens | o <= 0.15 ms | 5-8 | 0.5 | 60 laptop |
| 3 | Defaults: adapt every 4 / 24-96 swaps behind a new switch, bank sized from planner VRAM (give back GPU layers 1:1), prompts free the bank as today | replay + laptop identity | 2 | 0.5 | 30 laptop |
| 4 | PC chain item, Vulkan, A B B A with warm-up, 65k chat | default on only if >= +10 % paired | 1 | 1 | 120 PC |
| Total | | | 10-13 | 2 | 120 PC + 120 laptop |

Identity: GPU and CPU kernels differ in the last float bits, so text is "same meaning" not byte-exact (hybrid is documented not token-identical); check by 4 prompts at temperature 0, count identical tokens, and the KL gate.

Open risks: Vulkan sync cost may differ from CUDA; the planner and bank compete for the same VRAM (needs a trade rule); the f split on the PC is unmeasured (one phase-timer run on the PC would settle it, 10 PC min).

## 6. Update 2026-11-08 evening: step 1 and 2 measured (laptop, RTX 4060 8 GB, CUDA, llama.cpp 7fe450e)

Per layer step, microseconds, from the plugin's own phase timers (`MOE_CACHE_STATS=1`, one line at exit). Flash-Next IQ3_XXS, bank 2.2-3.0 GiB (7 % of experts, 41-54 % of picks served by the bank), CPU limited to 6 cores, so the CPU-miss time is far larger than on the PC.

| Phase (us per layer step) | base | MOE_CACHE_HYBRID_ASYNC=1 | MOE_CACHE_HYBRID_PERSIST=1 |
|---|---|---|---|
| scan, build graph | 32-39 | 28-29 | 35-39 |
| input copies | 42-81 | 13-14 | 21 |
| launch | 47-62 | 42-46 | 18 |
| GPU wait | 27-43 | 78-84 | 113 |
| read-back | 71-89 | 11 | 12 |
| Sum without CPU misses | 192-311 | 144-155 | 199 (same run: base 311) |
| Extra GPU memory | 0 | 0 | +580 MiB (47 graph buffers) |

- The base overhead of 0.19-0.31 ms per layer matches the 0.2 ms implied by the PC loss. Async cuts it by about 55-100 us per layer (3-5 ms per token over 47 layers) for no VRAM; persistent graphs cut more but cost 580 MiB and one run died with a CUDA out-of-memory abort in the CUDA graph instantiate at a 3 GiB bank. PERSIST therefore stays an experiment.
- Token identity at temperature 0 (4 prompts x 48 tokens, Flash-Next, 41 % bank hits, live swaps): async == base and persist == base; hybrid also equals hybrid-off over these 192 tokens. Granite (100 % hits): identical.
- Speed on the laptop is NOT a verdict: rounds of the same configuration vary 4-14 tok/s (SSD, thermals, 6 cores); the verdict is the PC chain item (07_gpu_tier, W P H A B B A H P).
- Replay of today's swap setting (every 8 tokens, 16 swaps): 51 % at a 12.5 % bank, vs 58 % (every 4, 24 swaps) and 61 % (every 4, 96 swaps).

## 7. Update 2026-11-08 night: PC verdict and the profile / swap-rate replay

PC (Vulkan, IQ3_XXS p64k, 16 GB): the hybrid bank loses to the plugin default in both PC items (decode -9.6 % in item 40, -29 % in item 55 with the same code). The diagnostic (item 55) measured the bank hit at 27.3 % with the shipped swap setting, a per-layer overhead of 0.25-0.29 ms (copies about 0, launch 28-91, GPU wait 86-94, read-back 100-120 us), and CPU misses at about 122 us per pick. ASYNC is not faster on Vulkan.

Replay on a real Flash-Next decode trace (laptop, 8 prompts, 1,398 tokens, `tools/gpu_tier_flash_replay.py`), bank of 4,104 experts (16.7 %, the PC's size), first 161 tokens from a cold bank:

| Bank choice and swap rule | Bank hit |
|---|---|
| shipped profile, fixed | 43.9 % |
| decode-only profile (other 7 requests), fixed | 40.5 % |
| shipped profile + shipped swaps (every 8 tokens, 16 swaps) | 52.6 % |
| decode-only + every 4 tokens, 24 swaps | 56.7 % |
| decode-only + every 4 tokens, 96 swaps | 66.8 % |
| PC, shipped setting (measured) | 27.3 % |

- A decode-only profile does not beat the shipped one on this trace, so the prompt-heavy profile counting (every group is counted, prompts included) is not what limits the bank hit.
- Faster swap rounds are the lever (+5 to +16 points) and already exist as `MOE_CACHE_HYBRID_ADAPT=N,K`.
- The replay overshoots the PC by about 25 points, so the replay is not trusted for the long chat until a trace of the real chat is replayed (PC item 09_gpu_bank_hit records it).
