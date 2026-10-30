# Configuration (environment variables)

`moe-cache-server` and the control page set these for you. Moved here from the README on 2026-10-30, unchanged.


| Variable | Meaning |
|---|---|
| `MOE_CACHE_SIZE_GIB` | cache size in GiB, or `auto` (size from the memory limit) |
| `MOE_CACHE_RAM_GIB`, `MOE_CACHE_MARGIN_GIB` | override the detected memory limit; memory to keep free (default 1 GiB) |
| `MOE_CACHE_GGUF` | model file (split models: the first part) |
| `MOE_CACHE_PROFILE` / `MOE_CACHE_WARM` | learn expert usage here and warm-start from it / warm start only |
| `MOE_CACHE_STATS=1` | print hit rate, reads and evictions at exit |
| `MOE_CACHE_EXEC_FUSE=0` | turn off the fused gate+up executor job (on by default since 0.3.2: decode +0.3-2.6 % in every PC pair, same text; RESULTS 73) |
| `MOE_CACHE_IDLE_REBALANCE=0` | turn off the idle cache refill (on since 0.3.3: refills the RAM cache while an agent pauses; every PC pair won) |
| `MOE_CACHE_POLICY=classlru` | frequency-aware cache replacement instead of LRU (opt-in; on the miss-heavy 122B +1.3-2.9 % decode in every pair, on Flash-Next within noise; RESULTS 73) |
| `MOE_CACHE_STATS_SIGNAL=1` | print one `moe-cache-stats {json}` line of the running counters on request: Linux `kill -USR1 <pid>`, Windows set the named event `Local\moe-cache-stats-<pid>` |
| `MOE_CACHE_DISABLE=1` | load nothing: the server behaves exactly like stock |
| `MOE_CACHE_EXEC=0` | switch the expert executor off (default on) |
| `MOE_CACHE_EXEC_ECORES=N` | let the executor use N efficiency cores as well (the launcher sets up to 4 on hybrid Intel CPUs; `--no-ecores` to disable) |
| `MOE_CACHE_EXEC_THREADS=N`, `MOE_CACHE_EXEC_READERS=N` | compute threads (default: what llama.cpp asks for) and background reader threads (default 4) |
| `MOE_CACHE_EXEC_CHECK=1` | recompute every expert op the stock way and compare (slow; for testing) |
| `MOE_CACHE_FAST_KERNELS=0` | use llama.cpp's own Q2_0 kernel instead of the plugin's |
| `MOE_CACHE_IQ3_KERNEL=1` / `=0` | use the plugin's IQ3_XXS gather kernel (bit-identical) on any AVX2 CPU / turn it off; off by default on AMD, where gathers are slow (Zen 4: about 0.6x llama.cpp's own function) |
| `MOE_CACHE_ROUTE=skip` (`moe-cache-server --fast`; `--fast` also sets a q8_0 KV cache: KL 0.0045 alone on the laptop (CUDA), **0.0120 on the PC (Vulkan)**, `--fast` as shipped 0.0145 there; kept in `--fast` by the owner's choice, RESULTS 61 R11) | opt-in approximate routing: skip a low-ranked expert that is not in RAM. About +10-12 % when the model is much bigger than RAM; about 1 top token in 40-50 differs from stock (KL 0.005-0.012; RESULTS section 31). Off by default |
| `MOE_CACHE_HYBRID=MiB\|auto` (`moe-cache-server --hybrid`) | opt-in, experimental: a bank of the most used experts (from the usage profile) in free GPU memory computes those picks while the CPU computes the rest. Flash-Next on 8 GB: 13.1 -> 17.4 tok/s; KL 0.0043 vs exact (about 1 top token in 60 differs). Needs a usage profile (the first session has none). The bank adapts to the session (`MOE_CACHE_HYBRID_ADAPT=N,K`, default 8,16; 0 = keep the profile choice). For models whose experts are slow on the CPU (IQ / Q2_0 quants); with Q4_K-class models such as Qwen3.6 it is slower than `--gpu-cache` (8.7 vs 15.7 tok/s at 12 GB). Off by default |
| `moe-cache-server --fast=kv` | **the default at `--ctx 131072` and above** (`--exact` keeps the f16 KV); otherwise opt-in: only the q8_0 KV cache of `--fast`, every expert computed. Passed the quality gate at 4k context (`docs/QUALITY_GATE.md`, level kv: KL 0.0079, top-1 96.7 %, tasks 190 = stock on Flash-Next Q2_0). At ctx 262144 the freed KV buys more GPU layers and `-ub 1024`: prefill 2x, decode +11-13 % at 64k (RESULTS 72). Off by default |
| `MOE_CACHE_FEWER=N` (`moe-cache-server --fast=fewer` sets 2 and `--fast`) | opt-in approximate: on the CPU-held layers, decode and small batches never compute the N lowest-ranked experts of a token (at least half are kept). Passes the gate (level fewer: KL 0.034, top-1 92.0 %, tasks 192 vs 190); faster at 4k (+2-6 %), **unproven at 64k-256k** (decode pairs lost their values). Off by default |
| `MOE_CACHE_ROUTE_DROP=N`, `MOE_CACHE_ROUTE_KEEP=N` | with `ROUTE=skip`: how many of the lowest-ranked picks may be skipped (default 2; `--fast=2` sets 4, with at least half of the picks always kept) |
| `MOE_CACHE_COALESCE_MIB=N` | merge neighbouring expert reads of a long prompt into reads of up to N MiB (default 16; 0 = off). Flash-Next prefill +14-24 % |
| `MOE_CACHE_SCAN_TOKENS=N` | prompt batches of N+ tokens load experts at the evict-first end, so the cache from before a long prompt survives it (default 512; 0 = off). Flash-Next prefill +27 % |
| starter profiles (`moe-cache-server --no-starter` to skip) | with no usage profile yet, the launcher starts from the shipped one for the model family (`profiles/`): first request 2.1x faster on Flash-Next (RESULTS 53) |
| `MOE_CACHE_GPU_PACK=0` | turn off packing: by default a long-prompt block uploads only the experts its tokens use (exact; 70-token prompts 1.8x faster on Qwen3.6, RESULTS 52) |
| `MOE_CACHE_HOST_RESERVE_MIB` | memory kept free for llama-server's prompt cache (`--cache-ram`); the launcher sets it |
| `MOE_CACHE_GUARD=1` | tests only: reading an evicted expert crashes instead of returning zeros (small models only) |
