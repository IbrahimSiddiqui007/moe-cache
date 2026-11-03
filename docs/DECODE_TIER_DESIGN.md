# Decode tier: getting the GPU layers back for decode without losing the big prompt chunk (design spike, 2026-11-03)

**Status:** design spike plus a default-off prototype (`MOE_CACHE_DECODE_TIER`), compiled but **never run on a GPU or a model** (laptop only; the PC kill check below is the first real test). Every number below is either quoted with its source or marked as my arithmetic or guess.

## The problem in one paragraph
v0.3.5 reads long prompts about 1.4x faster (`-ub 2048`) but decodes 6-10 % slower after them, because the bigger chunk needs a bigger compute buffer and 2 fewer expert layers fit on the 16 GB card (RESULTS 79: 64k, 39-40 CPU layers; `-ub 512` with 38 CPU layers decodes 26.6 / 26.9 against 23.7 / 22.9 tok/s, i.e. +15 % for 2 layers, and v0.3.4 decoded 27.4 / 27.0). The wish: prompt with `-ub 2048` **and** decode with 2 more GPU layers.

## Short answers
| Question | Answer |
|---|---|
| 1. Where does the VRAM go, how much idles in decode? | About 2.2 GiB of 16 is idle in decode and can be lent by the plugin alone (the block buffer 1.1 GiB + planner slack). A further ~1.6 GiB (the compute buffer, which llama.cpp reserves for a 2048-token prompt) is idle too but only a llama.cpp patch can lend it. Table below. |
| 2. Why did single-token batches give wrong output (T25)? | **Not found by reading the code**, and it was never retested after 2026-10-15 although the block code was rewritten several times since. The GPU kernels themselves are fine for one token: the existing `--hybrid` bank has decoded single tokens on the GPU since 0.2.x (RESULTS 33, 61). So the tier does not go through `gpu_segment()` at all; it reuses the hybrid path (`hybrid_group`). A built-in check mode compares GPU and CPU per layer (prototype). |
| 3. Best design? | **C, built on the existing hybrid bank (whole layers, 2 of them), exact-by-check, behind a switch.** Plugin only, no llama.cpp patch. Expected decode **+5 to +14 %, planning value +10 %** (a guess: the tier keeps the GPU round trip the native layers do not have). B+C and D come after C is proven or dead. A cheaper idea, F1 (revive the archived output-rows patch 0042 and let the planner use the smaller buffer), has a 10-minute PC check and might give the same layers in both phases; run it in the same session. |
| 4. Prototype? | Yes: 146 added lines in existing plugin files plus a 45-line header and a model-free test. PC plan at the end (about 30 minutes). |

## 1. Where the VRAM goes at 64k, `-ub 2048`, 40 CPU layers (16 GiB card)
There is no PC log of the whole budget; this is the planner's own arithmetic (it chose 8 GPU layers, `n_cpu_moe` 40). "Back-solved" means: the planner gave 8 layers, so the rest must add up to that.

| Item | MiB | Source | In decode |
|---|---|---|---|
| 8 GPU expert layers (0.66 GiB = 676 MiB each) | 5,408 | task numbers; `planner/gguf_plan.py:780` (layers = room // bytes per layer) | needed |
| Compute buffer for `-ub 2048` at 64k | 1,750 | `planner/gguf_plan.py:493` (512: 690, 1024: 920, 2048: 1,750) | decode graph needs about 1/2048 of it (my guess, ~100 MiB); **~1,650 idle, held by llama.cpp** |
| Long-prompt block buffer (`c->galloc`) | 1,107 measured for a 704 MiB layer at 2048 (comment at `gguf_plan.py:~737`); the planner keeps 1,498 free (`block_for`, `gguf_plan.py:741`: 1.6 x layer + 416) | `src/mc_backend.inl:463` (the buffer an earlier block allocated is "held" for the next one); only `hybrid_reload` frees it (`src/mc_hybrid.inl:203-206`) | **idle, 1,107 held**; the 391 MiB the planner kept beyond it are free too |
| Left over after dividing by whole layers | 0 to 675, say ~290 | rounding in `gguf_plan.py:780` | free |
| Reserve for other programs and growth | 477 (0.5 GB) | `tune(reserve_gb=0.5)`; raised by `desk_reserve_gb` when the desktop holds memory | keep |
| Non-expert weights + KV (q8, 27.5 KB/token = 1,760 MiB at 64k) + recurrent state + driver | ~5,500-6,200, **back-solved** assuming ~15 GiB free at start | | needed |

What is idle in decode and who can lend it:
| Piece | MiB | Who can lend it |
|---|---|---|
| Block buffer held after the first block | 1,107 | the plugin (free the gallocr; `hybrid_reload` already does) |
| Planner slack + whole-layer remainder | ~390 + ~290 | nobody needs to: it is simply free |
| **Total the plugin can use** | **~1,800 (range 1,500-2,200)** | with 512 MiB kept free: ~1,300-1,700 = **2 layers (1,352)** fit in the whole range, a 3rd (2,028) does not |
| Compute buffer (logits for 2048 rows, attention scratch) | ~1,650 | only a llama.cpp change (B, F1) |

So the plugin alone lends exactly about the 2 layers that `-ub 512` would have bought. That matches the planner: the 512 arm's reserve is 690 + 1,329 against 1,750 + 1,498 for 2048, a difference of 1,229 MiB = 1.8 layers.

## 2. The single-token bug (T25)
What is known:
- Recorded once: CHANGELOG 0.2.3.3 ("`MOE_CACHE_GPU_MIN_TOKENS=1` gave one repeated token ... Granite 3.1"), `src/mc_state.inl:638-645` refuses 1. `src/mc_exec.inl:225` and `:703` still test `gpu_min_tokens == 1` (dead code now). No cause was ever written down; no later retest exists in RESULTS.
- Since then the block was changed a lot (pinned read-back N22, split N23, `GPU_SUM`, pack I4, async input copies N15 with `be_sync`, `src/mc_compute.inl:263`). The defect may be gone or may still be there.
- The same single-token GPU matmul (`MUL_MAT_ID` + GLU on a cloned graph) is what `hybrid_group` runs for every banked layer in every decode step: GLM-5.3-Flash +9.7 %, Flash-Next IQ2_XS 8 GB card 17.6 tok/s, KL 0.0043 with no lost answers (RESULTS 33, 61 "B11/B12/R3"). So the kernels, the cloned graph and the readback are fine for 1 token.

So the cause is in what `gpu_segment()` does that `hybrid_group()` does not. Candidates, most likely first (all unproven):
1. The executor is switched off when the minimum is 1 (`mc_exec.inl:225`), so decode went through llama.cpp's own CPU path with the block bolted on; the ids/activation hand-over for the *first* graph of a split (inputs copied from the GPU backend asynchronously) was different from the executor path. `hybrid_group` runs only inside `exec_run_graph`, after `be_sync`.
2. The packing of used experts (`mc_backend.inl:402-453`) for a 1-token batch: 8 of 40 experts used, ids renumbered; `hybrid_group` uses fixed slots instead.
3. The shared pinned read-back buffer `rb_buf` (`mc_backend.inl:1029-1043`) reused every layer without an event between layers (only a synchronize); correct if the synchronize is complete, but never tested at 1 token per call.

Fixable? Probably, but it is **not on the path** of the tier. To find it: run `gpu_segment` for one-token groups in a check mode like the prototype's `tier_check` (GPU result against the executor's, per layer). Estimate 1-2 days to find, fix size unknown (guess: under 50 lines). Only worth doing if C works and we want the block's pack/upload path in decode (a *dynamic* tier), which this design does not need.

## 3. Designs compared
Ceiling: +15 % for 2 layers (measured, RESULTS 79). The tier keeps a GPU round trip per tier layer per token that a native layer does not have. Evidence on its size: Qwen3.6 hybrid on a laptop lost to llama.cpp's own GPU cache because of "graph, activation upload, sync, copy back, 40 layers per token" (RESULTS 33); the 122B on the PC lost 17 % with a bank in 47 layers (RESULTS B11/B12), i.e. about 0.2 ms per layer net. With only 2 tier layers that is 0.4-0.6 ms of a 37 ms token, about 1.5 %, against 2.7 ms saved per layer (the +15 % / 2 layers = 5.4 ms). The unknown is how much of the 2.7 ms is the CPU work and misses (goes away) and how much is the plugin split's fixed cost (stays). That is what the kill check measures.

| Design | What | Decode gain at 64k (guess) | Prompt cost | Effort | Risk (exactness) | Kill check (PC) |
|---|---|---|---|---|---|---|
| **C** decode tier, whole layers, hybrid path | After the prompt: free the block buffer, upload all experts of K chosen layers into the room, run their expert matmuls on the GPU in decode (`hybrid_group`, no CPU part, no expert reads for those layers). A long prompt drops the bank (existing `hybrid_drop`), decode loads it again. | **+5 to +14 %, planning +10 %** (recovers v0.3.4's decode) | none in the prompt itself; each prompt that needs the room pays one drop (free) and each decode start one reload of 1.35 GiB (estimate 0.3-1 s; short follow-up turns may not drop at all, because the block only uploads the experts the turn uses) | prototype done; productising 4-7 days (see below) | Not token-identical to *today*: layers 38-39 change from CPU to GPU kernels, exactly as moving them with `--n-cpu-moe` would. Against stock llama.cpp **with the same two layers on the GPU** it should be identical (same kernels, same inputs; ids and weights unchanged, slot = expert), but the CPU-side sum and the weights multiply stay on the CPU. Proven only by the check mode and an identity run | 30 min, see section 4 |
| **B+C** smaller compute buffer in decode (llama.cpp patch re-reserves) plus C | The scheduler is rebuilt for a small ubatch when all slots decode and again for the prompt chunk | up to +25 % over v0.3.5 if 4 layers fit (~1.6 GiB more; my guess) | each switch re-reserves (`llama-fit-params` builds these graphs in about 3 s each), twice per request; hurts agent turns | 5-8 days, rebases with every pin change | high: batch splitter, MTP/draft graphs, checkpoints, the planner and the plugin assume a fixed ubatch | only after C works; check how long `sched_reserve` takes on the PC first (2 min) |
| **D** layers chosen by misses (launcher `-ot`) | The 8 GPU layers are the 8 most-missed instead of the last 8 | +1-3 % | none | 1 day | exact against stock with the same `-ot` | 25 min A/B at 64k |
| **F1** output-rows patch 0042 + planner | `LLAMA_N_OUTPUTS_MAX=8` (archived, `archive/llama.cpp/0042-n-outputs-max.patch`) stops llama.cpp reserving 2048 logits rows; the planner then reserves less | 0 to +13 %: the logits are 82 % of the buffer on LLaDA (`patches/llama.cpp/README.md:112`); on Flash-Next at 64k the share is unknown (the table says 1,750 MiB in total) | none; same layers in **both** phases | 1-2 days (restore patch, change `_COMPUTE_MIB`, PC measure) | none for the numbers; a batch that wants more than N output rows grows the buffer at that moment (can fail on a full card) | **10 min, no benchmark**: server log "compute buffer size" at 64k `-ub 2048` with and without the variable. It was archived because prompts did not speed up (the freed memory was never turned into layers) |
| F2 per-token cost of GPU vs CPU for one token | see the paragraph above | | | | | the prototype's stats line |
| E / T2 (earlier) | smaller `-ub` with more layers | no: chunk size matters (RESULTS 78) | | | | done |

### Why C reuses the hybrid bank
`src/mc_hybrid.inl` is already a decode-time GPU consumer with the phase behaviour wanted here: `hybrid_drop` (`:25`) frees the bank when a long-prompt block finds no room (`mc_compute.inl:594`), and the next decode step calls `hybrid_reload` (`:203`), which frees the block buffer first and loads the bank again. It was built for a different purpose (hot experts across all layers, approximate, loses when the bank touches many layers: each touched layer pays a round trip, the 122B lost 17 %). A bank of **whole layers** touches only K layers, so the round trips are K, not 40+, and every pick hits, so the CPU does nothing for them. The old verdict "GPU hot tier tried and lost" (`docs/ENGINE_DECISION.md`, RESULTS B11/B12) was about hot *experts* in every layer; it does not cover this.

Other reasons the archived items do not apply: `GPU_KEEP_PIN` kept tensors between *prompt* blocks (no gain, uploads were not the bottleneck, `archive/README.md`); the layer-wise prefill and block prefetch were prompt-side too.

### Choosing the layers
Which two? The decode cost of a CPU layer is mostly fixed work, not misses: with a cache holding 45 % of the experts, a static estimate from `profiles/qwen4exp-48x512.usage` (my calculation, one generic profile, not this prompt) gives per-layer miss rates between 3.8 % and 5.9 % (mean 4.8 %), so the two most-missed layers carry 6.0 % of all CPU-layer misses against 4.7 % for layers 38-39 and 5.0 % for an average pair (at 30 % and 60 % caches the same: 5.9 / 4.4 and 6.6 / 4.4 against 5.0). That is a 1.2-1.4x spread, which is why D is worth only +1-3 % and why the first PC test uses layers 38 and 39 (the same two a smaller `-ub` would have put on the GPU), so the comparison with the +15 % ceiling is like for like.

## 4. Prototype (default off, 146 added lines plus a 45-line header)
Switches (all default off; with `MOE_CACHE_DECODE_TIER` unset the code paths are the old ones):
| Switch | Meaning |
|---|---|
| `MOE_CACHE_DECODE_TIER=38,39` | layers whose experts are kept whole on the GPU during decode (explicit list for the first test; the layers must be CPU layers, i.e. below `--n-cpu-moe`) |
| `MOE_CACHE_DECODE_TIER_KEEP_MIB=512` | free VRAM that stays free (default 512); layers that do not fit are left out, the log says which |
| `MOE_CACHE_DECODE_TIER_CHECK=1` | for those layers also run the CPU executor, compare, print, and **use the CPU result** (text unchanged). A rounding difference shows as about 1e-3 relative, a T25-like defect as 1e-1 or more |

Where: `src/mc_tier.h` (layer parsing and fit, no ggml), `tier_build` / `tier_check` and the hook in `hybrid_init` in `src/mc_hybrid.inl`, three small edits in `src/mc_compute.inl` (skip reading experts for a layer the bank served in full; load what is missing if a tier group falls back to the executor; keep the executor when the tier is asked for), two members in `src/mc_backend.inl`. Test: `tests/tier_test.cpp` (model-free: parsing, fit, edge cases; `g++ -std=c++17 -O1 -Wall -Werror -Isrc tests/tier_test.cpp -o /tmp/tier_test && /tmp/tier_test`; not wired into `.github` because CI files belong to the pin change). The plugin builds without warnings against both the ggml 0.25.1 and 0.26 headers (`build.sh --portable`). **Not run with a model or a GPU.**

What it does not do (on purpose): choose layers by itself, drop the bank for tiny prompts differently, evict the tier layers' experts from the RAM cache after the switch (they stay resident until normal eviction; the first token of each tier layer still reads its picks), Linux/CUDA checks, MTP/speculative graphs (the hybrid path was not checked with them either).

### PC test plan (Windows PC, Flash-Next Q2_0, about 30 minutes, load once, model already in the file cache)
Use the launcher as in RESULTS 79 with `--ctx 65536` (so the planner picks 40 CPU layers) and a 34k-token prompt (`C:\dl\long30k.txt`, 33,979 tokens) followed by a 256-token answer, greedy. The decode loss is a property of the placement, not of the prompt length, so the shorter prompt is enough for a kill check. Add `MOE_CACHE_STATS=1` to every arm.

| # | Arm | Time | What to read |
|---|---|---|---|
| 0 | `MOE_CACHE_DECODE_TIER=38,39 MOE_CACHE_DECODE_TIER_CHECK=1`, a 2k prompt, 128 tokens | 3 min | log line `decode tier: layers 38,39 on the GPU ... 1352 MiB`; `decode tier check` lines. **Stop if "groups above 0.05" is not 0** (wrong output, T25 again) or the line says no layer fits (then send the free-VRAM figure) |
| 1 | BASE: defaults (`-ub 2048`, 40 CPU layers) | 4 min | prompt tok/s, decode tok/s |
| 2 | TIER: same + `MOE_CACHE_DECODE_TIER=38,39` | 4 min | same, plus the `hybrid:` stats line (hits should be 100 %), the reload time line |
| 3 | TIER again | 4 min | |
| 4 | BASE again | 4 min | |
| 5 (optional) | `--ub 512` (38 CPU layers), same prompt | 5 min | the +15 % ceiling on the same boot |
| 6 (optional) | F1: server log only, `LLAMA_N_OUTPUTS_MAX=8` with the restored patch 0042 vs without, `compute buffer size` at 64k `-ub 2048` | 10 min | does the buffer fall below 1,750 MiB, and by how much |

Verdict rule: **keep** C if TIER decode is at least +6 % over BASE in both pairs (40 % of the ceiling) and prompt tok/s is within 3 %; **kill** if under +3 % or check mode shows mismatches. Between: send the numbers. Order for the evening: 0, then 1-4 (ABBA as above), then 5 and 6 if time is left.

## 5. Recommendation and effort
| Step | What | Days | Confidence it pays |
|---|---|---|---|
| 1 | PC session above (arms 0-4, plus F1 log) | 0.1 PC evening | decides everything |
| 2 | If C passes: productise (choose layers from the planner: the K = layers freed by the block room, last CPU layers first, optionally by decode misses; drop policy for short turns; evict the tier layers' experts from the RAM cache after the switch; stats line; docs; a plugin identity run against stock with the same layers; model-free test in CI) | **4-7** | about 50 % that C reaches +6 % or more on the PC |
| 3 | If C is dead and F1's buffer shrinks by 0.9 GiB or more: F1 | 1-2 | 50 % (depends on the logits share) |
| 4 | D (launcher) | 1 | +1-3 % only; build after C or F1 either way |
| 5 | B+C | 5-8 more | not before steps 1-2 succeed |

**Honest estimate:** the old 1-2 weeks was for a dynamic tier through the GPU block and the T25 fix. Because the hybrid bank already does the hard part (drop for prompts, reload for decode, single-token GPU path), the realistic path is **about 1 PC evening to decide and, if it works, 4-7 working days to ship C (point estimate 5), confidence about 50 %** that it delivers at least +6 % decode at 64k. The largest unknown is the fixed cost of a tier layer inside the plugin split (not measurable on the laptop). Exactness is "same as stock with those layers on the GPU", not "same as today"; say so in the release notes.

## Facts I could not settle here
- The per-layer decode cost split (CPU work vs split overhead vs misses): needs the PC (stats line and the arms above).
- The size of the logits share of the 64k compute buffer on Flash-Next (the table gives 1,750 MiB; the vocabulary figure in `patches/llama.cpp/README.md:112` is the owner's number and would give 1.9 GiB, more than the whole buffer, so one of the two is off).
- Whether T25 still happens (nobody ran MIN_TOKENS=1 since 2026-10-15).
- Free VRAM in decode: my 1.5-2.2 GiB range comes from the planner's rules, not a PC log.
