# Next-block weight prefetch (PL05, `MOE_CACHE_GPU_PREFETCH=1`, default off)

Status: design written before the code. Nothing here is measured on the PC. Numbers marked "from the brief" come from the 2026-11-02 Flash-Next 34k run (-ub 2048, GPU_SUM on, about 60 s).

## Precheck: can it help at all?

Per block, from the brief's sums (34k prompt, -ub 2048: about 17 micro-batches x 37 CPU-side layers = about 630 blocks, arithmetic, not a count):

| Part | Run total | Per block |
|---|---|---|
| upload (whole expert tensors, about 0.5 GB) | about 23 s | about 37 ms |
| compute | about 8 s | about 13 ms |
| read-back | about 1.6 s | about 2.5 ms |
| rest (gaps: llama.cpp's own attention work, graph-thread CPU work, setup) | about 27 s | about 43 ms |

The prefetch window of the design below is the gap after a block (the buffer is in use until the block's compute ends). Gap (about 43 ms) is only about 15 % longer than the upload (about 37 ms). So:

| Question | Answer |
|---|---|
| Upper bound | all of the upload hidden: up to about 23 s of 60 s. Only if the gap is entirely free for the bus and the helper copies at the full rate |
| Realistic | half to two thirds, 9-15 s (GUESS: the gap contains graph-thread CPU work and GPU waits, which leave the bus free, but it also contains the block's own small transfers and lock waits) |
| Kill line | if the PC counters (below) show the gap per block is shorter than the upload per block, or the wait at block start is more than half of the old upload time, stop and archive the switch |
| Not a good result | the bus is already near its limit during the upload (23.7-26.7 GB/s on PCIe 4.0, PREFILL-170). Prefetch does not make the bus faster; it moves the same bytes into time the bus is idle. Idle bus time is 60 % of the run, so there is room |

So the precheck does not kill the idea. It is a close call per block (gap about equal to upload), which is why the measured counters matter more than this arithmetic.

## Design

| Point | Choice | Why |
|---|---|---|
| Which experts | whole tensors (gate, up, down) of the next block | at -ub 2048 every expert is used (the plugin's own packing switches itself off when all are used, `pack_used.empty()`), so the whole tensor is exactly what the block uploads. A predictor (recall 68 %) could only upload less than all, and the missing 32 % would then be uploaded by the block anyway; it helps only for partial-use blocks, which are short prompts that the packing already handles. Not built |
| Which block is next | learned: the plugin records, per block, which weight tensors the block before it used (`next_of`). Layer order repeats every micro-batch, so from the second micro-batch on the guess is right. No routing needed | needs no model knowledge. First micro-batch and any odd order simply miss |
| A wrong guess | costs time only. The block builds its graph from the real tensors. A tensor found in the prefetch buffer (same source pointer, type and shape) is placed there with `ggml_backend_tensor_alloc` (the way `held` copies are), anything else is uploaded as today. If the helper failed, the block uploads the tensor itself into the same place | output cannot change: the same bytes land in the same kind of buffer |
| Thread | one helper thread that only copies (file view -> `ggml_backend_tensor_set` into the prefetch buffer, 32 MiB slices). The graph thread never waits for it except at the start of the block it feeds | the graph thread has no idle time of its own to spend between blocks: the gap is llama.cpp's scheduler running other backends, which the plugin cannot hook. A helper is the only way to use it |
| Is `tensor_set` from a second thread safe on Vulkan | at the pin (b11966, `ggml-vulkan-buffers.cpp:577-615`): a write to a device-local buffer takes `device->mutex`, uses the transfer queue's own command pool and `device->fence`, all under that mutex; a write to a host-visible (resizable BAR) buffer is a plain `memcpy`. So two threads are serialised, not corrupted. The old comment at `MOE_CACHE_GPU_UPLOAD_THREADS` (audit round 4) describes an older pin. Older or newer builds: the switch only starts for ggml builds in `mc_tested.h` | read from the source; not tested on Vulkan here (the laptop has CUDA). CUDA's `tensor_set` uses `cudaStreamPerThread`, thread-safe |
| Same thread, between blocks | rejected. The Vulkan backend has no "submit and return" for a copy (`ggml_vk_synchronize` always waits for its fence, `ggml-vulkan.cpp:13540-13600`), and an async copy needs a pinned source, which would add a copy on the graph thread | |
| VRAM | one extra block of weights (3 tensors, about 0.5-0.7 GB). Allocated once, only when free GPU memory is at least that plus 768 MiB (512 MiB spare plus the plugin's 256 MiB headroom). Freed and not tried again for 64 graphs when free memory falls below 512 MiB. A block's own need is reduced by the bytes that already sit in the prefetch buffer, so the extra buffer does not push a block off the card | a too-full card is known to make blocks far slower (53 s vs 8 s compute, WDDM moving memory), so "no room" means do nothing |
| When it does nothing | partial-use (packed) blocks, split blocks, templates (`MOE_CACHE_GPU_TMPL`), `MOE_CACHE_GPU_KEEP_MIB`, the second card (`MOE_CACHE_GPU_DEV2`), a backend that is not Vulkan or CUDA, an untested ggml build, no room |
| Windows | the helper waits for the readers' flags (`wait_experts`) and applies the same "all experts resident or mapped" check as the block's own upload; if it fails, nothing is copied and the block does its own path (which keeps it on the CPU). The check-then-copy window is longer than for the block itself (about one gap); the file view keeps `mapped` pages readable, which is what the existing rule relies on |
| Self-check | after 32 guesses with fewer than half used, the switch turns itself off for the process and says so once |
| Counters (with `MOE_CACHE_SEG_TIMES=1`) | blocks prefetched / used, bytes, helper busy time, wait of the graph thread at block start, extra VRAM |

## Not built, and why

| Idea | Reason |
|---|---|
| Two prefetch buffers (fill during the block's compute and read-back too) | adds 13 + 2.5 ms of window per block (about 40 % more) for a second block of VRAM. Retest only if the first version wins and the card has room |
| Predictor for partial blocks | see above |
| Async copy from the graph thread | see above |

## What the PC run needs

See the final report of the session. In short: Flash-Next 34k, `MOE_CACHE_SEG_TIMES=1`, A B B A with and without `MOE_CACHE_GPU_PREFETCH=1`, same prompt, nothing else running; plus `tests/identity.sh`.
