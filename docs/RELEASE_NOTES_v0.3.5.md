# moe-cache v0.3.5 (2026-11-03)

Faster long prompts. On a 16 GB card prompts of 34k-128k tokens are read about 1.4x faster than in v0.3.4 (decode after a long prompt is 6-10 % slower). Bundled llama.cpp is still the official, unmodified b11966.

## Measured (Windows 11, Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB DDR5, Qwen3.8-Flash-Next Q2_0)

| Context | Prompt tok/s, v0.3.4 | Prompt tok/s, v0.3.5 | Change | Decode after, v0.3.4 | Decode after, v0.3.5 | Change |
|---|---|---|---|---|---|---|
| 34k | 408.6 / 403.1 | 577.0 / 572.6 | **+42 %** | 28.43 / 28.59 | 27.04 / 26.38 | -6 % |
| 64k | 313.8 / 314.1 | 421.3 / 448.3 | **+38 %** | 27.42 / 27.04 | 25.69 / 23.44 | -10 % |
| 128k | 304.8 / 305.5 | 420.7 / 421.9 | **+38 %** | not measured (the answer ended after 2 tokens) | not measured | — |

Clean run 2026-11-03 02:30-03:25: PC restarted at 02:03, baseline guard PASS 49.4 tok/s, both versions on the official llama.cpp b11966 server, order new old old new, two runs per cell (first / second). Decode after a long prompt is 6-10 % lower than v0.3.4: the bigger prompt chunk takes GPU memory, so 2 fewer expert layers fit on the GPU (39-40 instead of 37-38 on the CPU at 34k-64k). A later release aims to give that memory back after the prompt. Other models with the new chunk size (same night, not paired with v0.3.4): Qwen3.5-122B at 64k prompts +24 % (545 vs 438 tok/s against -ub 1024); GLM-5.3-Flash at 30k prompts +62 % (73 vs 45), decode 2.88 vs 3.06 (-6 %).

## What changed

| Area | Change |
|---|---|
| Long prompts | A long-prompt block also does the router-weight sum on the GPU, so about a tenth of the data comes back to the CPU (`MOE_CACHE_GPU_SUM=0` turns it off) |
| Launcher | Above 32k context the micro-batch is 2048 and more GPU memory is left free (at 256k: prompts +26 %, 334 vs 264 tok/s). A card filled to the edge was the largest slowdown found on long prompts. gpt-oss stays at 1024 because its decode fell by a third |
| Claude Code | `connect claude` also sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0`, so a changing block at the start of the prompt no longer makes the server read everything again each turn |
| Windows warning | The launcher and the control page name programs that use the GPU (the EA app was measured: prompts up to 23 % slower, decode 8-14 % slower) |
| Control page | Cancel works right after "Share my numbers". New "Other programs" list |
| Bug fixes | 19 from an independent audit, among them: `--port`, `-c` and `-ngl` after `--` are seen by the launcher; parent cgroup memory limits count; `MOE_CACHE_DISABLE=0` no longer disables the plugin; a second `moe-cache.bat` cannot share the control page's port; a crash at 128k in the split GPU block; a server that hung when the launcher's stderr closed |
| Off by default (prototypes) | Peer registry for several servers (`--peers`), second-card long-prompt blocks (`MOE_CACHE_GPU_DEV2`), planner rule (`MOE_CACHE_PLAN_RULE`) |
| Patches | `patches/llama.cpp` holds a small patch set for people who build llama.cpp themselves. All patches are off unless their switch is set, and none is in the bundled build. Tested losers were moved to `archive/` |
| Docs | Every doc checked for stale numbers and dates. The benchmark card shows each row's test details (GPU layers, context, runs, builds, date, source section); GLM is now the paired 2.0x (the earlier 2.7x compared against a stock run on another drive). New `docs/CODING_STANDARDS.md`, `docs/CONTROL_PAGE_CHECK.md`, a warning about apps that draw on the GPU |

## Not in this release
- Patch 0022 (faster Vulkan expert counting, about +5 % on prompts in two PC pairs). The same change was merged upstream on 2026-11-02 (llama.cpp PR 31266). It arrives with the next pin move to an official release that contains it, with no patched build needed.

## Known gaps
- Decode after a long prompt is 6-10 % slower than v0.3.4 (see above).
- The benchmark card's rows were measured 2026-10-27/28; VRAM use was not recorded for them, and RAM only for some. They will be re-measured with full resource numbers.
