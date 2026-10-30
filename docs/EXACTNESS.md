# Same tokens as stock: how moe-cache checks it

moe-cache runs MoE models bigger than your RAM inside an **unmodified** llama-server. Its default promise is simple: the answer is the one stock
llama.cpp gives for the same model, settings and layer split. This page lists how that is checked, on every change, and where the promise has limits.
Numbers and logs: [RESULTS.md](RESULTS.md); the approximate modes: [QUALITY_GATE.md](QUALITY_GATE.md).

## What "the same" means

| Mode | Promise | How it is checked |
|---|---|---|
| default, context below 128k | **token-identical** to stock with the same layer split | greedy decoding, stock and plugin servers, every generated token compared (`tests/identity.sh`) |
| default at `--ctx 131072` and above | q8_0 KV cache (`--fast=kv`): same meaning, not the same bytes | quality gate (KL divergence, top-1, 202 tasks), measured at 4k context; `--exact` keeps f16 and identity |
| `--fast`, `--fast=2`, `--fast=fewer`, `--hybrid` | approximate, opt-in only | quality gate with limits fixed before measuring ([QUALITY_GATE.md](QUALITY_GATE.md)) |

"Same layer split" matters: with long prompts the amount of work that runs on the GPU can change from run to run with the free VRAM, and a layer
computed on the GPU instead of the CPU can change the last digits and then a token, in stock llama.cpp too. The identity tests fix the split.

## Checked on every change (CI)

Three llama.cpp versions in every run: two pinned commits (ggml 0.25 / backend API 2 and ggml 0.26 / API 3) and **llama.cpp master of that day**,
so an upstream change that breaks the plugin shows up before a release does (Linux). Windows CI uses the official llama.cpp release zip.

| Check | What it catches |
|---|---|
| Identity with the cache forced to evict (0.2 GiB) | experts read back from the SSD wrongly, or a stale expert used |
| Identity on long prompts (512+ token batches) | the long-prompt GPU block and scan-resistant loading |
| Executor forced on, every expert op **byte-checked** against llama.cpp's own result | any arithmetic difference in the plugin's CPU kernels (`mismatches 0` required) |
| Logits at batch 8, KL divergence against stock | differences that greedy tokens would hide |
| Different thread counts for prompts and decode | the switch between two thread pools |
| Cache smaller than one operation (1 MiB) | the fallbacks when memory is very short |
| Background reads deliberately delayed | a missing wait: the plugin computing with an expert still loading |
| Guard mode (evicted pages made inaccessible) | an unintercepted read crashes instead of silently reading zeros |
| Each read's memory poisoned while in flight | reading a neighbour's half-written page (seen on Windows) |
| Repeated expert ids, two slots in parallel | the crash a GPU top-k with NaN produced on the PC |
| Idle release (memory given back while idle) | tokens unchanged after the cache was emptied and refilled |
| Noise floor: plugin loaded but disabled | proves the comparison itself is deterministic |
| **Mutant builds**: copies of the plugin with a known bug put back (a missing wait, an old read path, a wrong op number) | each must **fail** its test: proves the tests can see the bug |
| Sanitizers (address, thread) | memory errors and data races |
| Windows: the release kit unzipped as a user gets it, the cache running, identity with the file-view cache | the Windows path, from the zip |

## Checked on real hardware

The Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB) runs paired A/B tests for every speed change: A, B, B, A, and the text of every run
is compared (A vs A, B vs B, A vs B). A change is kept only when it is faster in every pair and the text stays identical (RESULTS 59-73).

## Check it yourself

`python bin/moe_cache.py bench-report MODEL.gguf` runs stock and moe-cache on your machine and prints `tokens IDENTICAL` or `DIFFERENT`
(DIFFERENT can be normal when the two used different GPU/CPU splits; both plans are in the report). `tests/identity.sh MODEL.gguf` is the strict test
with a fixed split. Please report what you see: [Benchmark report](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=benchmark_report.yml).

## Limits

| Limit | Note |
|---|---|
| The identity tests in CI use small models on CPU runners | big models are checked on one PC and by users' reports |
| The quality gate for the 128k+ default ran at 4k context | a long-context loop test is part of the PC runs |
| Different llama.cpp builds or backends (CUDA, Vulkan, HIP) give different digits | compare against stock on the same build |
