moe-cache v0.3.4 runs MoE models bigger than your RAM through llama.cpp's `llama-server`. This release makes long prompts faster, adds a way to keep using your PC while a model runs, and fixes bugs.

## What is new in v0.3.4

| Area | Change |
|---|---|
| Long prompts | Plugin outputs go straight to the GPU instead of through llama.cpp's slowest copy path. Measured in paired runs on Flash-Next with identical text; details in docs/RESULTS.md section 76. Turn it off with `MOE_CACHE_HOST_OUT=0` |
| Leave room for me | `--leave-room` and a control-page box keep the PC usable while a model runs |
| Windows | Without the file views (not the default), experts are read through the file cache |
| Security and bug fixes | The API key no longer follows a redirect to another host; `--project` state identity with q8_0 KV; saved agent conversations; orphaned `bench-report` and `learn` servers; `[`, `*`, `?` in model paths; clean-up after a failed file reopen or view mapping |
| Experiments | Small changes from an optimization audit. The ones not proven exact are off by default behind `MOE_CACHE_*` switches and are untested. Nothing here is a promised gain |

## What is not measured

| Item | State |
|---|---|
| Benchmark chart and README tables | not re-run for this release: they show older, slower long-prompt numbers |
| The long-prompt change | measured on one model at about 34k and 64k tokens. Other models and deeper contexts are not measured, and decode right after a 64k prompt may be slightly lower (one reading) |
| The release on the developer's PC | checked by CI, including the Windows kit on a GitHub runner, but not run on the developer's PC |

Downloads, checking the hash and unblocking the unsigned kit work as in docs/INSTALL.md.
