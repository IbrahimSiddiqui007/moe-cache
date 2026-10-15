# Design: the per-layer expert scheduler (P1; implementation plan for experiment E4)

Status: **design only**, written 2026-10-15. Builds on MULTI_DEVICE_DESIGN.md (the idea and the limits) and on the shipped `--hybrid` code
(`src/mc_hybrid.inl`: GPU bank + CPU executor in parallel, measured 13.1 -> 17.6 tok/s on Flash-Next, KL 0.0043). Nothing here is built.
Opt-in and approximate (different processors round differently): judged by the quality gate below, never the exact default.

## 1. What changes compared with `--hybrid`

`hybrid_group()` splits one layer's picks two ways: picks whose expert is in the GPU bank go to the GPU, all others to the CPU executor, both run at once.
The scheduler makes that split N-way and cost-driven:

| | `--hybrid` today | scheduler |
|---|---|---|
| Who computes a pick | bank hit -> GPU, else CPU | any **lane** that can (see 2), chosen by earliest finish |
| GPU for misses | never (no upload) | yes, through an upload lane, when uploading + computing finishes before the CPU would |
| iGPU / NPU | not used | lanes if E1-E3 / E5 say they pay |
| Costs | none (fixed rule) | learned online per lane and weight type |
| Placement (what lives in VRAM) | profile bank + adaptive swaps | same mechanism, with a "pays back within H tokens" rule |

Two decisions stay separate (MULTI_DEVICE_DESIGN): **placement** changes slowly (moving weights costs a read or an upload); **execution** is chosen per layer
and only moves activations.

## 2. Lanes

| Lane | Can compute pick (t, e) when | Fetch cost | Compute | Launch cost per layer |
|---|---|---|---|---|
| `cpu` | always (after the expert is in RAM) | SSD read if not resident (shared reader pool) | executor kernels (`exec_one_op`) | ~0 (pool already running) |
| `gpu_bank` | e is in the VRAM bank | 0 | MUL_MAT_ID on the bank (as `hybrid_group`) | graph build + submit + wait (~0.1-0.3 ms, measure) |
| `gpu_up` | e is in RAM (or read first) | PCIe upload of e's gate/up/down (9-11 GB/s pageable; more if I1 pins the staging buffer) | MUL_MAT_ID on a staging tensor | same submit as `gpu_bank` if both run in one graph |
| `igpu` | E1-E3 passed; e in RAM and the iGPU can map it (Vulkan host import) | 0 with zero-copy, else a copy | Vulkan MUL_MAT_ID | E1's measured submit-and-wait (stop if ~3 ms) |
| `npu` | E5 passed; prompts only (batches) | copy | OpenVINO | large; prompts only |

`gpu_bank` and `gpu_up` share one GPU graph per layer (bank tensors and staging tensors as two MUL_MAT_ID sources), so there is one submit.

## 3. The assignment rule (per layer, per step)

Input: the picks of this layer (n_tok x n_used ids), which experts are resident where, and the cost table.

1. For each distinct expert e picked in the layer: the lanes that can take it and `cost(e, lane) = fetch(e, lane) + n_cols(e) * compute(type, lane)`
   (n_cols = tokens that picked e: an expert's rows are read once for all its columns, as the executor does today).
2. Sort the experts by their cheapest cost, largest first (longest-processing-time first).
3. Give each to the lane that would **finish it earliest**: `busy(lane) + cost(e, lane)`, adding `launch(lane)` if the lane has nothing yet this layer.
   A lane whose launch cost is larger than the work it would get is skipped.
4. Reads: experts assigned to `cpu` or `gpu_up` that are not in RAM are queued to the reader pool first (largest fetch first), as `prepare_group` does now.

About 10 experts and 2-4 lanes: a few hundred operations, under a microsecond. With only `cpu` and `gpu_bank` available the rule gives exactly today's
`--hybrid` split (bank hits are cheapest on the GPU), so `--hybrid` becomes the two-lane special case and its numbers are the regression floor.

## 4. Costs, learned online

| Cost | How it is measured | Kept as |
|---|---|---|
| `compute(type, lane)` | wall time of the lane's work in this layer / (rows x columns), from timestamps already taken in `exec_run_graph` / `hybrid_group` | EWMA (alpha 0.1) per (weight type, lane) |
| `fetch(path, bytes)` | reader-pool read time per job (exists: `ns_read`), upload time per staging copy | EWMA per path, as time = a + bytes / bandwidth |
| `launch(lane)` | submit-to-done time of an empty or tiny graph, measured at start, then from real layers | EWMA |

Start values come from one calibration pass at start (one layer's worth of each kernel per type, ~50 ms), so the first tokens are not scheduled blind.
Costs adapt to heat and to other programs; a lane whose real finish is more than 2x its estimate for 32 layers in a row is dropped for the session (with one line).

## 5. Execution order inside a layer

1. Build and submit the GPU graph (bank picks + uploaded picks) asynchronously: `ggml_backend_graph_compute_async` (as `hybrid_group`).
2. Submit the iGPU graph if it has work.
3. The CPU executor computes its picks (the caller thread joins the pool).
4. Wait for the GPU / iGPU, copy their result rows back into the layer's output (`[n_embd, n_used, n_tok]`, one row per pick), as `hybrid_group` does
   for bank hits today. Picks are independent until the weighted sum, which llama.cpp does after this block, so the merge is a row copy.

## 6. Placement: what lives in VRAM

The bank keeps `hybrid_adapt`'s mechanism (profile start, decayed hit scores, shadow duel against the profile bank), with one added rule for the upload lane:
an expert that keeps being uploaded is swapped into the bank when `uploads_saved_per_token x upload_cost x H > one bank swap` (H = 64 tokens to start),
and only into the slot whose score is lowest. Nothing else moves weights; execution never triggers a move.

## 7. Quality gate (opt-in flag `--sched`, judged like `--hybrid`)

| Check | Pass |
|---|---|
| KL divergence vs exact (`kl_run.sh`, 20 chunks prose + code, base `kl_hyb/base.kld`) | <= `--hybrid`'s 0.0043 x 1.5, and far below the quant yardstick 0.243 (RESULTS 43) |
| Same top token | >= 98 % |
| Answer bank (`q_run.sh`, 26 checkable answers) and code tasks (10) | same pass count as exact |
| Speed (paired, clean-run gate) | >= +10 % over `--hybrid` on Flash-Next at 24 GB, else not shipped (E4's stop rule) |

## 8. Code plan

| Step | Where | What | Test |
|---|---|---|---|
| S1 | new `src/mc_sched.inl` | lane table, cost EWMAs, calibration pass, the assignment function (pure, no ggml) | unit test with synthetic costs (`tests/sched_test.cpp`, runs in CI without a GPU): the two-lane case reproduces `--hybrid`'s split; LPT bound |
| S2 | `mc_hybrid.inl` | `hybrid_group` takes a per-pick lane map instead of `hit[]`; staging tensors for `gpu_up` picks next to the bank tensors | identity of the two-lane case vs current `--hybrid` (same tokens) |
| S3 | `mc_compute.inl` | `MOE_CACHE_SCHED=1` (launcher `--sched`) routes executor graphs through the scheduler | KL + answers (gate above) on Flash-Next |
| S4 | counters | per lane: picks, ms, estimated vs real finish; one summary line | visible in the stats line |
| S5 | iGPU lane | second GPU backend (Vulkan device index), only after E1-E3 | Claw: E3 numbers vs this lane |

Order: S1 -> S2 -> S4 -> S3 on the laptop (dGPU + CPU only: the `gpu_up` lane is the new part), then S5 on the Claw if E1-E3 pass.
Effort: S1-S4 about 2 days; S5 about 1 day plus Claw time.

## 9. Kill rules (stop early)

- `gpu_up` loses to the CPU for single misses on this laptop if PCIe uploads stay at 9-11 GB/s: the worked example in MULTI_DEVICE_DESIGN gives a split
  only when 3+ experts miss per layer. If the calibrated costs say the upload lane never wins on Flash-Next, stop after S1 (no code in the hot path).
- The whole scheduler needs a compute-bound model (Flash-Next yes; K-quant models are bandwidth-bound and gain nothing, RESULTS 34 and 42).
- If I1 (pinned uploads) is dropped, `gpu_up` loses its main lever; check I1's kill test first.
