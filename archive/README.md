# Archive: built, tested, did not win

Code that lost or gained nothing in a paired run leaves the program and stays here, so it can come back if the situation changes. Each file is a patch that holds the code and where it goes.

## Restore
Plugin: `git apply archive/plugin/NAME.patch` from the top of the repo. llama.cpp: copy the file back to `patches/llama.cpp/`. Later changes to the same lines may need a small manual fix.

## What is here
Results: Flash-Next Q2_0, 34k prompt, -ub 1024, RX 9070 XT + 7950X, Windows, 2026-11-01, one baseline against each arm in mirrored order. Baseline 252.1 / 251.9 tok/s prompt, decode about 27.7. Keep-or-drop runs, not chart numbers.

| File | Switch | What it did | Prompt tok/s | Decode | Why archived | Retest if |
|---|---|---|---|---|---|---|
| plugin/gpu_keep_pin.patch | `MOE_CACHE_GPU_KEEP_PIN=1` with `MOE_CACHE_GPU_KEEP_MIB=N` (in `src/mc_backend.inl`) | keeps the first expert tensors on the GPU between blocks, so they are not uploaded again | 251.3 / 251.5 | 27.6 | no gain | much more free VRAM, or uploads become the largest cost |
| plugin/gpu_whole_runs.patch | `MOE_CACHE_GPU_WHOLE_RUNS=N` (`src/mc_backend.inl`) | uploads a whole expert tensor in runs, each after its own wait | 247.4 / 249.9 | 27.7 | no gain | cold-disk waits inside uploads become large |
| plugin/gpu_rb_pipe.patch | `MOE_CACHE_GPU_RB_PIPE=N` (`src/mc_backend.inl`) | reads a block's result back in slices, overlapping host copy and transfer | 247.5 / 250.8 | 27.4 / 26.9 | no gain, decode lower | read-back becomes the largest cost |
| plugin/timer_1ms.patch | `MOE_CACHE_TIMER_1MS=1` (`src/platform.h`, `src/mc_state.inl`) | asks Windows for a 1 ms timer tick | 249.4 / 250.2 vs 248.7 / 250.7 | 27.4 vs 27.3 | no gain | a wait is shown to depend on the tick |
| llama.cpp/0001-vulkan-fence-spin-iters.patch | `GGML_VK_FENCE_SPIN_ITERS=N` (`ggml-vulkan.cpp`) | polls the GPU fence more often in the last 20 % of a graph | 247.4 / 247.2 | 27.8 / 28.0 | prompts about 2 % slower | never expected to gain much; another GPU or driver |

## Notes on 0001
`ggml_vk_wait_for_fence` waits for the last 20 % of a graph by polling `getFenceStatus` with 1000 `pause` instructions between polls. The patch made the count an env value (N blocks of 10 pauses, unset = 100 = stock). Only the poll interval changes, so output cannot change. Tested at N = 10.
