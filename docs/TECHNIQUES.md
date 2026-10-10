# Techniques used in moe-cache: what each does, the evidence, the downside

Audit of 2026-10-10. "Default" is what is in the code, not what the docs say. Numbers come from [RESULTS](RESULTS.md).

| Technique | Default | What it does | Evidence | Downside / risk | Verdict |
|---|---|---|---|---|---|
| Residency cache (core) | on | experts keep stock addresses; the cache decides which have physical pages; miss = O_DIRECT read from the GGUF, evict = `madvise(DONTNEED)` | 2.1x to 12x stock when the model does not fit; token-identical on 7 architectures | needs O_DIRECT on the file system; one tested llama.cpp/ggml version; uses an internal ggml header (upgrade risk); 3 to 14 % slower than stock when everything fits (CPU-only small-model test) | keep |
| LRU eviction | on | evict the least recently used expert | replay of 13,464 real steps: LRU 84.2 %, 2Q 83.6 %, LFU 84.2 %, LRU+pinned 84.8 %, optimum 92.5 % | cannot beat the offline optimum by more than about 8 points without a predictor | keep |
| Epoch pinning of a group | on | experts of the op being run cannot be evicted | by design; "cache too small for one op" aborts the server when the limit is tiny | abort instead of graceful fallback | keep, document |
| Automatic cache sizing | on | sizes from the cgroup limit, keeps process anonymous memory under limit minus 1 GiB, re-checks after every 256 MiB read | `tests/limit.sh` PASS on 5 models | reads `/proc` and cgroup v2; cannot see memory other processes take later; margin costs speed (1.5 GiB cost 12 to 15 % on gpt-oss-20b) | keep |
| Read straight into place | auto when file > 85 % of the limit | O_DIRECT read lands at the expert's address, no bounce copy | +22 % on gpt-oss-120b; none when the model fits | alignment shifts and 12 KiB padding per expert tensor | keep |
| Sibling batch load | on | loads gate, up and down experts of a layer together | +28 % on gpt-oss-120b (paired); no effect on models that group them | none seen | keep |
| Reader threads | 6 total | parallel direct reads | 12 and 24 threads: no change | batches are small | keep |
| Preload everything when it fits | on, no profile | load all experts in order at start | closed a 13 % gap to pinned stock (35.5 vs 36.0) | start-up reads the whole model (about 6 s for 15 GiB) and commits the RAM early | keep |
| Usage profile / warm start | on with the launcher | counts per expert, saved at exit, loaded next run | first request 28 vs 16 tok/s; KAT gap 14 % to 1.5 % | a wrong-domain profile is worse than none; the fingerprint is tensor names and sizes, so fine-tunes of one architecture share it | keep |
| Automatic CPU repack | auto: on only without a GPU | same weight layout stock uses on CPU-only machines | CPU-only identity 3/4 to 4/4 | slower than repack off in the 3 GB OLMoE cell (13.7 vs 16.6), not explained | keep, investigate |
| GPU for long prompts | on, batches of 32+ tokens | runs the expert block of big batches on the GPU | 42 to 166 prompt tok/s | does not read llama.cpp's `--no-op-offload`, so with that flag stock and plugin can differ; uploads whole tensors (up to 8.8 GB per prompt) | keep, document |
| CPU pinning (hybrid Intel) | on | one thread per performance core | +15 to +22 % | only Intel hybrid CPUs; strict pinning hurts when other heavy processes run | keep |
| Planner placement | with the launcher | chooses `--n-cpu-moe` from free VRAM | +60 % over all-on-CPU (16.2 to 25.9) | NVIDIA and AMD discrete GPUs only | keep |
| Lookahead prefetch | off | predicts next layer's experts from router weights | -25 %, -12 %, -74 % on three models | wasted reads on SSD-bound models | off, code stays |
| L3 helper thread | off | pulls later tensors of a group into L3 | -19 % | contention | off |
| Huge pages | off | `MADV_HUGEPAGE` on the arena | no gain | none | off |
| n-gram speculative decoding | not used | llama.cpp feature | flat to -25 % on MoE | verifying several tokens wakes extra experts | do not use |
| CPU+GPU bandwidth aggregation | not built | | shared memory bus: no add | | closed |
| GPU hot tier | not built | | simulation +7 % at 4 GB, about +22 % at 12 GB, not token-identical | open defect in the one-token GPU path | deferred |
| Fast mode (skip small experts) | not built | | 25 % fewer reads for 9 % of the output weight (Qwen3.6 trace) | changes tokens | idea, opt-in only |
| Compressed RAM cache | not built | | zstd on 1 GiB of IQ2_XS expert data: 97.4 % of original size | no | rejected |
