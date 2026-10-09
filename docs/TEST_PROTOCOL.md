# Standard test protocol (STP-1)

Why: results used to be collected with different settings for different models, so numbers could not be compared and kept "getting bigger". Every number
that goes into README or RESULTS from now on comes from this protocol (or says exactly how it differs).

## What is fixed
| Item | Value |
|---|---|
| Machine state | start at or below 58 C, background CPU below 25 % of one core (the tool waits for the CPU to cool and flags the run if not), nothing else heavy running |
| Threads | pinned to the performance cores on hybrid Intel CPUs (the default) |
| Context | 8192, flash attention on, KV cache f16 |
| Workload | 4 fixed prompts, 100 generated tokens each, temperature 0, seed 42 |
| Metric | steady generation speed = mean of requests 2 to 4 (request 1 is the warm-up), in tokens/s |
| Repeats | stock and moe-cache each run twice, second time in reverse order; the table shows the mean of the two |
| Noise | run-to-run variation on SSD-bound models is about +-10 %: differences below 10 % are not claimed |
| Memory | every run in its own cgroup with a hard limit and swap off (the tool does this) |
| Identity | token ids of stock and moe-cache are compared on every cell and must match |

## The two placements ("profiles")
| Profile | Meaning | `n_cpu_moe` in the tool |
|---|---|---|
| **A: CPU experts** | all experts stay on the CPU; the GPU only does attention and the shared parts (a floor that any machine can match) | `all` |
| **B: GPU-assisted** | as many whole layers of experts as fit in the free VRAM go to the GPU, the rest stay on the CPU; identical for stock and moe-cache | `fit` (computed from the GGUF header and free VRAM) |

## The two memory limits
12 GB and 24 GB (what is left for the model on a 16 GB and a 32 GB machine). Together with the model size this covers "does not fit", "partly fits" and "fits".

## How to run
```bash
bin/moe-cache-bench --no-browser &                       # the benchmark server
bench/standard.py MODEL.gguf [MODEL2.gguf ...]            # both profiles, both limits, stock and moe-cache, both orders
bench/standard.py MODEL.gguf --limits 24 --profiles gpu   # one cell
```
Each cell is also saved as a normal result (charts, deep metrics) in the GUI.

## Standard model set
Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B, KAT-Coder-35B-A3B (30-35B class); Qwen3.5-122B-A10B at IQ2_XXS / IQ3_XXS / IQ4_XS (100-150B class, one model at three bit widths); gpt-oss-120b (MXFP4 only: its expert rows cannot be quantized lower).
Experiments are developed on the smaller models and confirmed on the large ones; only the final confirmation is run on the biggest model.
