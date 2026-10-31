# moe-cache: optimization audit and experiment backlog

Reviewed 31 October 2026 (Asia/Dubai). Repository: [IbrahimSiddiqui007/moe-cache](https://github.com/IbrahimSiddiqui007/moe-cache). Commit: `526c804610cf835fdfcccd379fd7072d24e1aa2f` (v0.3.3). Remote main was fetched and matched this commit. Production source and settings were not edited.

## Direct assessment

There is remaining software headroom. The most concrete unmeasured opportunities are eliminating discarded work in split GPU blocks, compacting readback, preparing shared gate/up inputs once, reusing hot-path metadata/graphs, coalescing async reads and dispatching compute by actual readiness. Bigger opportunities require changes to llama.cpp scheduling, GPU kernels or memory ownership. A full engine rewrite is not required to test the first group.

This is a catalog of **124 candidate optimization experiments**, not 124 verified speedups. It includes small targets, conditional integration work and major research directions. Several may produce zero gain or regressions. A finite source review cannot establish every possible future optimization or a guaranteed final token rate. No new GPU/model throughput was measured in this environment; assigning a 1–2% gain to every row would be invented.

The fixed baseline is the same model, quantization, requested context, quality settings and physical machine. No hardware purchases, overclocking or shorter answers are counted as optimization. Compiler/runtime targeting of capabilities the existing machine already has is allowed. Workload reuse is labeled separately from fresh-prefill/decode kernel throughput. Approximation is outside the 124-item strict optimization inventory and is listed separately at the end.

## What was reviewed and checked

- Production performance paths: all plugin module roles, graph compute, executor/pool, GPU block/hybrid paths, cache admission/eviction, read/pack/view paths, profiles/rebalance, launcher/planners, warm-state persistence and HTTP/UI forwarding. The source tree has about 17,105 lines across the enumerated production C++ headers/includes and Python planner/bin files; that is a scope count, not a claim every line was formally verified.
- Repository benchmark history, especially RESULTS 59–74, PERFORMANCE_ANALYSIS, configuration and release/build/CI paths. Old profiles describe different builds/settings and must not be treated as the current timing breakdown.
- Related primary sources: Strata, Maya, KTransformers, oMLX, Fiddler, the Cintu07 research prototype and Peregrine. Previous source inspection in this session covered selected Strata cache/prefill files, Maya fused RDNA expert code and the research prototype's scheduler. External performance claims are not measurements of moe-cache.
- Fresh checks: both bundled ggml header sets passed warning-as-error C++ syntax checks; args, tested-ggml and rebalance C++ tests passed; 15 Python suites passed. The unmodified UI suite stopped at `/proc/PID/task/PID/children`, unavailable in this runtime. It was not marked a full pass. No GPU run, full-model identity test, new hardware profile or sanitizer run was performed this round.

## Read this inventory correctly

**Priority 1:** first experiments to run because the mechanism is visible or the payoff could be substantial. **Priority 2:** next candidates or workload-dependent tuning. **Priority 3:** research, backend/upstream integration or lower-confidence small targets. Priority is an engineering judgment, not a percentage forecast.

**Evidence M:** the current code contains the stated repeated work/decision; the speed benefit is unmeasured. **Evidence C:** the basic technique is already present or only helps if a condition is met; the row is a refinement/integration test, not credit for the existing feature. **Evidence H:** design hypothesis requiring a prototype.

**Boundary E:** intended to preserve exact arithmetic and bytes; it still needs strict regression tests. **V:** mathematically equivalent intent, but changes to batch shape, placement, dispatch or kernel can change rounding, so bit identity is not assumed. **U:** needs substantial upstream/backend/scheduler ownership; numerical equivalence still has to be tested. **W:** request/session efficiency rather than faster raw kernel throughput. All categories must preserve correctness; these letters are not permission to accept wrong answers.

Every item below has a source anchor, mechanism, and acceptance measurement. Unless explicitly identified as an already recorded result, **its gain is unknown**. Overlapping candidates must not be added together.

## Inventory at a glance

| Area | Candidates | Main opportunity |
|---|---:|---|
| Prefill/GPU | 24 | Less discarded work, fewer copies, more overlap and reuse |
| Decode/CPU | 27 | Runnable work scheduling, less setup and synchronization |
| Storage/OS | 17 | Fewer read/VM calls and better demand completion latency |
| Cache/residency | 20 | More avoided critical-path I/O per memory byte |
| Planning/speculation | 11 | Choose the fastest complete request plan |
| Upstream/backends | 10 | Kernels, graph submission and memory ownership |
| Serving/reuse | 9 | Avoid recomputation and foreground stalls |
| Build/runtime | 6 | Small instruction, dispatch and allocation savings |
| **Total** | **124** | **Experiments, not guaranteed cumulative gains** |

## Concrete findings worth testing first

1. **Split GPU work is not compact.** In `gpu_segment`, excluded routes map to expert slot zero, the graph retains the original output shape, every partition executes it, and only matching rows are copied back. For 1,024 tokens with eight picks, three partitions can compute 24,576 output slots to retain 8,192. This is a structural work-count illustration, not a measured 3x slowdown or 3x whole-prompt gain. P01/P02 target arithmetic and transfer waste separately. Shape compaction may select a different kernel and therefore needs V-level validation.
2. **Async reads miss the existing synchronous coalescing path.** `prepare_group(async)` publishes jobs directly to the reader queue. `flush_reads` contains the coalescer. I01 can extend it while preserving completion and protected-page rules.
3. **Gate/up fusion shares a launch, not all preparation.** `exec_run_graph` calls `exec_prep_op` twice before the fused launch. Identical activation preparation can be shared only when all conversion/layout conditions match. D03 is small; historical setup accounted for under 1% of pool thread time, so it must not be sold as a huge gain.
4. **Workers can claim an unready expert and spin.** The pool initially orders ready tasks first, which already helps; it cannot reorder later asynchronous completions once workers have claimed pending chunks. D01 is the next scheduling step, not the discovery of resident-first CPU work.
5. **Metadata and clocks recur on hot paths.** Task maps/vectors, fused task lists and GPU graph descriptors are rebuilt; `exec_run_prepared` always times its launch; `n_runs += times` is an atomic RMW even when adding zero. These are real code mechanisms but may be far below 1% of full inference time individually.
6. **GPU buffers are reused, but that is not a persistent tagged expert cache.** P11 proposes skipping uploads only when a verified model/layer/expert/version still occupies the same GPU slot. A small cache may have little cross-block reuse because the graph traverses many layers before returning to one.

Source anchors for these mechanisms: [src/mc_backend.inl:524](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L524); [src/mc_state_lru.inl:405](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L405); [src/mc_compute.inl:53](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_compute.inl#L53); [src/mc_pool.inl:76](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L76); [src/mc_exec.inl:558](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L558); [src/mc_pool.inl:190](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L190); [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197).

## P. Prefill and GPU transfers (24 experiments)

### P01. Compact split-block routes

Priority 1 · Evidence M · Boundary V · [src/mc_backend.inl:524](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L524)

Each expert partition currently maps excluded picks to slot zero, computes a full-shaped graph, and discards excluded output rows. Gather only that partition's active routes, execute compact tensors, and scatter results back.

**Keep only if:** Count useful versus computed rows and GPU time with 2/3/4 partitions; compare logits because changed shapes can select different kernels.

### P02. Compact split-block readback

Priority 1 · Evidence M · Boundary E · [src/mc_backend.inl:474](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L474)

Each partition reads a full output tensor back even though only its own expert rows survive. Read/scatter only useful contiguous spans, or compact them on the GPU first.

**Keep only if:** D2H bytes, copy calls and wall time; byte-check every kept output, including duplicate ids and the last uneven partition.

### P03. Resident-first upload order

Priority 1 · Evidence M · Boundary E · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Packing enumerates expert ids in ascending order. Partition resident/ready experts ahead of pending reads, and consistently rebuild the id-to-slot mapping.

**Keep only if:** Time blocked in wait_experts, upload throughput and TTFT; require identical selected weight bytes and outputs.

### P04. Double-buffer upload and compute

Priority 1 · Evidence H · Boundary E · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Existing GPU_PIPE overlaps SSD fill with uploads, but the sub-block loop uploads, computes and synchronizes serially. Use two bounded device/staging slots and completion events.

**Keep only if:** Transfer/compute overlap on a timeline, peak memory and full request time; never overwrite a buffer still used by the GPU.

### P05. Pinned upload staging

Priority 1 · Evidence M · Boundary E · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Pinned readback already exists; upload sources can still be mapped/pageable memory. Stage selected weight spans in a bounded backend-host buffer before asynchronous copies.

**Keep only if:** H2D bandwidth and total time including the extra host copy; reject when staging hurts or consumes valuable expert RAM.

### P06. Parallel host packing

Priority 2 · Evidence M · Boundary E · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Noncontiguous input packing and host staging copies happen on the submission thread. Move large copies to a bounded helper pool while preserving ownership.

**Keep only if:** Submission-thread time, memory bandwidth competition and TTFT; no additional work for small inputs.

### P07. Cache GPU graph templates

Priority 1 · Evidence M · Boundary E · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

gpu_segment creates a context, tensor clones, maps and graph each call. Cache immutable templates by shapes, types, operations, device and allocation generation; refresh dynamic inputs.

**Keep only if:** Allocations/graph-build time per block and TTFT; invalidate on model unload, layout changes and buffer resizing.

### P08. Bucket reusable graph shapes

Priority 2 · Evidence H · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Expert counts and final chunks create shape variants. Compare a few reusable shape buckets against rebuilding exact-sized graphs; bound padding overhead.

**Keep only if:** Graph-cache hit rate versus wasted arithmetic; numerical checks for each dispatch bucket.

### P09. Upload shared inputs once per partition loop

Priority 1 · Evidence M · Boundary E · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

The loop uploads activation/id inputs again for every partition. Inputs that are invariant can stay in a separate persistent buffer; only partition-local ids change.

**Keep only if:** Per-block H2D bytes and calls; prove the graph allocator does not alias or overwrite retained inputs.

### P10. Interleave gate/up/down readiness

Priority 2 · Evidence M · Boundary E · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Uploads are traversed tensor by tensor. A gate slice may be ready while the matching up/down slices are not. Schedule copy groups around expert readiness and consuming operations.

**Keep only if:** Dependency stalls and end-to-end block time; respect the GLU and down-projection dependencies.

### P11. Retain GPU experts across blocks

Priority 2 · Evidence H · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

The allocation is reused but uploaded weights are not managed as a persistent tagged working set. Keep a bounded layer/expert/type/version slot directory and skip repeated uploads when reuse exists.

**Keep only if:** Actual inter-block reuse and avoided H2D bytes after charging extra VRAM; do not assume consecutive micro-batches visit the same layer consecutively.

### P12. Select partitions by measured cost

Priority 1 · Evidence M · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

The default split chooses the first memory-fitting number of expert parts, not the fastest one. Sweep partition size with transfer, compute and readback costs.

**Keep only if:** Total block time and peak memory across prompt lengths; a larger fitting block is not automatically faster.

### P13. Learn the CPU/GPU crossover

Priority 1 · Evidence M · Boundary V · [bin/moe_cache.py:1645](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L1645)

The plugin uses a token threshold (normally 32, 64 in approximate modes). Include unique experts, rows per expert, transfer volume and kernel type in the decision.

**Keep only if:** TTFT for 1/4/8/16/31/32/33/64/128/256-token tails, without hiding numerical placement changes.

### P14. Avoid expensive prompt-tail offload

Priority 2 · Evidence M · Boundary V · [bin/moe_cache.py:1645](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L1645)

Checkpoint-created short tail batches can trigger an entire weight upload. Extend the existing stock-only short-batch policy to measured plugin/model cases.

**Keep only if:** Follow-up request latency including checkpoints; verify recurrent state and same-shape exactness.

### P15. Group GPU work by populated expert tiles

Priority 2 · Evidence H · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Packing already removes unused experts, but it does not by itself ensure kernels execute only populated token/expert tiles. Explore sorted row lists and grouped kernels.

**Keep only if:** Useful FLOPs, tile occupancy, routing-sort overhead and TTFT; upstream/backend kernel work may be required.

### P16. Extend supported GPU block windows

Priority 2 · Evidence M · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

The block currently accepts MUL_MAT_ID and GLU nodes only. Bias/clamp/scale variants can force a CPU fallback. Add explicitly validated pointwise operations and model patterns.

**Keep only if:** Count avoided fallbacks by model and compare complete block outputs; never silently swallow an unsupported operation.

### P17. Balance partitions by active routes

Priority 2 · Evidence H · Boundary V · [src/mc_backend.inl:524](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L524)

Equal expert counts can yield unequal token work. Partition using route counts plus byte and residency costs rather than expert count alone.

**Keep only if:** Per-partition busy/idle distribution and total time; keep deterministic output scatter and reduction order.

### P18. Replace broad synchronization with events

Priority 2 · Evidence M · Boundary E · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Several paths synchronize the GPU backend around a block. Track input readiness, weight readiness and output completion separately where the backend API supports it.

**Keep only if:** Timeline bubbles, synchronization count and TTFT; maintain cross-backend ordering on every supported load method.

### P19. Alias nonoverlapping scratch lifetimes

Priority 3 · Evidence H · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Attention/MoE and temporary packing buffers can have disjoint lifetimes. Reuse memory to allow larger useful batches without increasing the memory cap.

**Keep only if:** Peak allocated VRAM and full TTFT; scheduler/allocator changes are needed beyond plugin-owned buffers.

### P20. Lend decode-cache memory to prefill

Priority 3 · Evidence H · Boundary V · [src/mc_hybrid.inl:326](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_hybrid.inl#L326)

The hybrid bank is dropped only after a failed allocation. Plan a reversible prefill workspace loan before allocation and restore only the valuable decode entries.

**Keep only if:** TTFT plus refill plus first decode latency; preserve the full context and memory cap.

### P21. Layerwise reuse across micro-batches

Priority 3 · Evidence H · Boundary U · [src/mc_compute.inl:200](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_compute.inl#L200)

Moe-cache already groups experts inside a micro-batch. Reuse one layer's transferred experts across more prompt chunks before releasing them; this requires model/scheduler integration and recurrent-state care.

**Keep only if:** Expert bytes per prompt token, intermediate-state memory, causality and end-to-end TTFT.

### P22. Prefetch exact upcoming PLE rows

Priority 2 · Evidence H · Boundary U · [docs/RESULTS.md:2982](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/docs/RESULTS.md#L2982)

Historical profiling identified model lookup-table faults outside the plugin. If token ids make future row addresses knowable, read those rows ahead of use.

**Keep only if:** Lookup-attributed faults and wall time; the old profile attributes only about 3% of that whole prompt to the identified longest PLE gaps.

### P23. Batch PLE projection computation

Priority 3 · Evidence H · Boundary V · [docs/RESULTS.md:2982](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/docs/RESULTS.md#L2982)

For supported lookup-table architectures, replace token-at-a-time projection work with batched operations where dependencies allow, following the upstream model path.

**Keep only if:** Lookup/projection time plus total TTFT; batch arithmetic may change floating-point results and is not automatically bit-exact.

### P24. Keep routing metadata on the GPU longer

Priority 3 · Evidence H · Boundary U · [src/mc_compute.inl:200](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_compute.inl#L200)

CPU-controlled caching needs router ids back at split boundaries. GPU-side route compaction and resident-slot lookup could remove selected readbacks, while a host path handles actual misses.

**Keep only if:** D2H synchronization latency, GPU route overhead and miss handling; substantial scheduler/backend work, not a simple flag.

## D. Decode executor and CPU kernels (27 experiments)

### D01. Completion-driven runnable queue

Priority 1 · Evidence M · Boundary E · [src/mc_pool.inl:76](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L76)

A worker claims a task then spins on that expert's loading flag. Publish only ready tasks, adding delayed tasks from reader completions; avoid head-of-line stalls.

**Keep only if:** Worker useful time, runnable queue overhead, CPU energy and per-token wall latency under deliberately reordered read completions.

### D02. Pipeline each expert through its block

Priority 2 · Evidence H · Boundary E · [src/mc_compute.inl:53](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_compute.inl#L53)

Operations are completed globally before dependent operations start. Allow an expert's gate/up, activation and down to progress when its own prerequisites finish.

**Keep only if:** Global barrier count and token wall time; keep the original per-output arithmetic and final expert reduction order.

### D03. Share gate/up activation preparation

Priority 1 · Evidence M · Boundary E · [src/mc_exec.inl:382](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L382)

A fused gate/up pair still calls exec_prep_op twice and independently quantizes/prepares inputs. Share only when activation address/strides, quant type and prepared-kernel format all match.

**Keep only if:** Preparation time, bytes converted and full token latency; historical setup was under 1% of pool thread time, so expect a small target.

### D04. Fuse mixed-quant gate/up jobs

Priority 2 · Evidence M · Boundary E · [src/mc_compute.inl:53](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_compute.inl#L53)

Current fusion requires matching types and dot functions. Give each task its own kernel descriptor so independent mixed-quant gate/up operations can share a wake/barrier.

**Keep only if:** Additional fusion coverage versus indirect-call cost; byte-check each output with its own stock kernel.

### D05. Fuse independent mixed-shape jobs

Priority 3 · Evidence H · Boundary E · [src/mc_pool.inl:190](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L190)

The pool's row count and chunks-per-task are global. Per-task row geometry could combine otherwise independent expert jobs with different output dimensions.

**Keep only if:** Saved launches versus task-index overhead; do not fuse dependencies or allow row ranges outside their tensors.

### D06. Tune row chunk size per shape

Priority 1 · Evidence C · Boundary E · [src/mc_pool.inl:190](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L190)

128 rows already won over 32 on the desktop. Keep that result; test per-quant/per-row-count/per-thread-count choices rather than replacing it with one new constant.

**Keep only if:** Blocked randomized paired runs; include tiny ops, uneven tails and memory-resident versus miss-heavy cases.

### D07. Tune active compute workers

Priority 1 · Evidence H · Boundary E · [src/mc_pool.inl:190](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L190)

Many tiny expert ops may have fewer useful chunks than workers, while too many readers/compute workers compete for bandwidth. Use measured shape/work-based worker counts.

**Keep only if:** Token latency and CPU time; avoid changing the surrounding stock backend's numeric thread behavior without validation.

### D08. Separate contended pool counters

Priority 2 · Evidence M · Boundary E · [src/mc_pool.inl:76](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L76)

Shared next/done/inside/generation fields can cause cache-line traffic. Pad or partition the hot counters after checking actual layout.

**Keep only if:** Cache-to-cache traffic and token latency; padding is a hypothesis and can increase footprint.

### D09. Claim multiple row chunks at once

Priority 2 · Evidence M · Boundary E · [src/mc_pool.inl:76](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L76)

Each row chunk performs a shared fetch_add. Compare small block claims or worker-local ranges plus work stealing.

**Keep only if:** Atomic operations per op, load balance and token latency; use small claims when expert reads are uneven.

### D10. Publish completion in fewer updates

Priority 2 · Evidence M · Boundary E · [src/mc_pool.inl:76](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L76)

Every row chunk increments the shared done counter. Aggregate completed counts locally and publish safely in batches or at worker exit.

**Keep only if:** Counter contention and tail latency; prove caller completion cannot race job reuse or late-arriving workers.

### D11. Reuse task and fused-job storage

Priority 1 · Evidence M · Boundary E · [src/mc_exec.inl:382](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L382)

all/tasks/late/task_of and fused both vectors are rebuilt for each op. Use bounded per-backend scratch storage sized for observed maxima.

**Keep only if:** Allocator calls and metadata time per token; preserve reentrancy and two-op buffer lifetime.

### D12. Sparse generation-stamped task maps

Priority 2 · Evidence M · Boundary E · [src/mc_exec.inl:382](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L382)

task_of initializes one entry per model expert despite decode touching only top-k. Keep generation stamps or a small sparse map for touched ids.

**Keep only if:** Initialization bytes and op time for 128/256/512/1024-expert models; handle duplicate ids and generation wrap.

### D13. Make detailed timers optional

Priority 1 · Evidence M · Boundary E · [src/mc_exec.inl:558](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L558)

exec_run_prepared takes timestamps and records per-type runtime even when exec-times is off. Separate essential summary counters from optional detailed timing.

**Keep only if:** Clock calls per token and full latency with equal observability settings; retain a diagnostic mode.

### D14. Remove atomic add-of-zero

Priority 1 · Evidence M · Boundary E · [src/mc_pool.inl:190](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L190)

n_runs += times still performs an atomic read-modify-write when timing is disabled. Guard the update with if(times).

**Keep only if:** Generated assembly and op latency; tiny benefit, no inflated token/s claim.

### D15. Aggregate nonessential statistics locally

Priority 2 · Evidence M · Boundary E · [src/mc_state_io.inl:3](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L3)

Many read/compute counters are atomics updated in hot paths. Use thread-local deltas merged for snapshots where exact instantaneous totals are unnecessary.

**Keep only if:** Atomic traffic and runtime with/without snapshots; memory-budget and completion synchronization must remain authoritative.

### D16. Reduce repeated crash-context stores

Priority 2 · Evidence M · Boundary E · [src/mc_pool.inl:76](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L76)

The innermost row/column loop rewrites TLS task/row/column/destination metadata. Hoist invariant fields; keep useful fault attribution without unnecessary repeated stores.

**Keep only if:** Disassembly, instructions per dot and token latency; preserve crash reporting and never remove bounds checks to win a benchmark.

### D17. Skip empty pool launches

Priority 1 · Evidence M · Boundary E · [src/mc_pool.inl:190](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L190)

A zero-task job can still publish/wake a generation. Return early after validating zero work where hybrid/approximate masks or empty routes produce it.

**Keep only if:** Count zero-work jobs and avoided wakeups; outputs for intentionally skipped picks must already be initialized correctly.

### D18. Cache executable graph subsets

Priority 1 · Evidence M · Boundary E · [src/mc_exec.inl:258](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L258)

exec_run_nodes builds a keep vector and scans/mutates/restores the whole graph for each small CPU batch. Reuse safe subset graph plans or compact node arrays.

**Keep only if:** Graph bookkeeping and small-op CPU time; invalidate dynamic graphs and retain view dependencies.

### D19. Reuse routing/duplicate analysis per ids tensor

Priority 1 · Evidence M · Boundary E · [src/mc_exec.inl:298](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L298)

The same ids drive gate/up/down, but duplicate/tiled decisions are repeated. Compute a validated routing descriptor once per group and reuse it.

**Keep only if:** Repeated scans/sorts per group and token latency; duplicate ids and forced tiled kernels remain supported.

### D20. Blocked multi-column expert dots

Priority 2 · Evidence C · Boundary V · [src/mc_pool.inl:76](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L76)

The executor already groups up to eight columns, but calls a one-row dot separately per column. Decode a weight block once into registers and update several columns while preserving each column's accumulation.

**Keep only if:** 2/4/8-column verification speed and full MTP latency; keep the stock tiled crossover and compare logits at that boundary.

### D21. Specialize common dot dimensions

Priority 2 · Evidence H · Boundary E · [src/fast_q2_0.h:200](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/fast_q2_0.h#L200)

Common expert widths permit fixed-trip-count loops, fewer indirect branches and tuned prefetch distances. Dispatch only validated hot shapes and retain a generic tail.

**Keep only if:** Streaming, cache-cold kernel speed plus whole-model time; preserve upstream accumulation lanes and operation order.

### D22. Cache a faster lossless hot-weight layout

Priority 3 · Evidence H · Boundary V · [src/mc_exec.inl:382](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L382)

Frequently used resident experts may justify a predecoded/interleaved layout that removes repeated unpack work. Charge its larger memory footprint against the expert budget.

**Keep only if:** Total tokens/s after cache-hit loss and conversion time; never assume a cache-hot microkernel gain survives SSD misses.

### D23. Calibrate kernel selection on actual workload shapes

Priority 2 · Evidence C · Boundary E · [src/fast_iq3.h:80](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/fast_iq3.h#L80)

Current self-checks establish numeric compatibility, while vendor rules choose speed. Add a short streaming-aware calibration among compatible kernels and persist it per CPU/build/type.

**Keep only if:** Whole-layer streaming performance and startup cost; retain the current AMD gather rejection unless a different implementation wins.

### D24. Compact hybrid GPU hit routes

Priority 2 · Evidence M · Boundary V · [src/mc_hybrid.inl:326](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_hybrid.inl#L326)

Hybrid builds a GPU graph for all picks with miss picks mapped to slot zero, then replaces only GPU-hit outputs. Execute only real hit routes and transfer only their results.

**Keep only if:** GPU work/readback per actual hit and mixed CPU/GPU wall time; enforce the approximate-placement quality boundary separately from strict mode.

### D25. Separate ready flags from frequently mutated metadata

Priority 2 · Evidence M · Boundary E · [src/mc_pool.inl:76](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L76)

Workers poll loading flags stored beside replacement/usage metadata. Separate completion state from frequently written cache fields to reduce false sharing after measuring actual layout.

**Keep only if:** Cache-line invalidations and wait overhead; preserve acquire/release publication and pin the node storage for reader lifetime.

### D26. Multi-row expert microkernels

Priority 2 · Evidence H · Boundary E · [src/mc_pool.inl:76](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_pool.inl#L76)

The inner loop calls one dot per output row. A two/four-row kernel can keep activation blocks in registers and amortize call/unpack setup while maintaining a separate stock-ordered accumulator per row.

**Keep only if:** Streaming layer speed and full decode latency, including tails; kernel-only cache-hot gains are insufficient.

### D27. Direct exact pointwise executor operations

Priority 2 · Evidence H · Boundary V · [src/mc_compute.inl:53](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_compute.inl#L53)

Small activation/scale nodes still go through the stock CPU graph path between matrix products. Implement only proven hot pointwise patterns directly or fuse their dispatch, retaining the same scalar/vector semantics.

**Keep only if:** Small-op dispatch time, full decode and byte checks including nonfinite values and model-specific activations.

## I. Storage, read queues and operating-system overhead (17 experiments)

### I01. Coalesce asynchronous demand reads

Priority 1 · Evidence M · Boundary E · [src/mc_state_lru.inl:405](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L405)

The synchronous flush path merges adjacent reads, but the async path queues individual jobs. Add capped merges with a completion fan-out for every constituent expert.

**Keep only if:** Read calls/bytes, queue latency and token time; preserve per-expert ready flags and avoid large merges delaying first useful work.

### I02. Prioritize demand over advisory I/O

Priority 1 · Evidence M · Boundary E · [src/mc_state_io.inl:3](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L3)

Demand, sibling, warmup and predicted reads compete for the same device even when using different queues/threads. Apply demand-first service and bounded advisory bandwidth.

**Keep only if:** Demand p95 read latency and tokens/s; avoid starving necessary down projections or progress-critical prefill.

### I03. Cancel stale queued prefetches

Priority 1 · Evidence H · Boundary E · [src/mc_state_io.inl:194](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L194)

Queued predictions can arrive after the consumer has passed. Tag them with model, token/layer deadline and generation; cancel before submission when stale.

**Keep only if:** Useful/on-time/late/unused bytes, demand latency and no writes into recycled slots.

### I04. Tune read concurrency dynamically

Priority 1 · Evidence C · Boundary E · [src/mc_state_io.inl:3](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L3)

Reader counts are fixed defaults. Adapt outstanding bytes and requests to measured read latency, size and compute pressure rather than blindly adding threads.

**Keep only if:** Device bandwidth, p99 demand latency, CPU usage and full token time under both view and private paths.

### I05. Linux io_uring backend

Priority 2 · Evidence H · Boundary E · [src/platform.h:217](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/platform.h#L217)

pread workers issue blocking reads. A bounded io_uring implementation can batch submission and consume completions without one blocking thread per request.

**Keep only if:** Same-destination byte checks, partial-read/failure handling, syscall counts and total latency; test SQPOLL separately, not as a free gain.

### I06. Windows IOCP backend

Priority 2 · Evidence H · Boundary E · [src/platform.h:217](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/platform.h#L217)

ReadFile is opened overlapped but each worker immediately waits for its one request. IOCP can keep several requests outstanding and dispatch by completion.

**Keep only if:** Private-memory path latency and CPU cost; default mapped views need their own evaluation and may not benefit.

### I07. Batch mapped-view prefetch calls

Priority 1 · Evidence M · Boundary E · [src/platform.h:403](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/platform.h#L403)

file_view_load issues one prefetch/advice plus population/touch sequence per expert. Group compatible ready ranges into bounded Windows range arrays or merged Linux advice.

**Keep only if:** OS calls and first-use fault time; mark ready only after every required page is accessible.

### I08. Put the view owner directly in read jobs

Priority 1 · Evidence M · Boundary E · [src/mc_state_io.inl:671](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L671)

read_job linearly scans all mapped views before deciding its path. Cache the validated view/path descriptor in the job or tensor metadata.

**Keep only if:** Instructions/read and wall latency across split files; invalidate on unload and do not confuse private/repacked destinations.

### I09. Cache or share partial boundary-page reads

Priority 2 · Evidence M · Boundary E · [src/mc_state_io.inl:671](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L671)

Direct reads may fetch first and last partial pages separately for nearby experts. Reuse immutable file boundary pages or coalesce safe reads into bounce storage.

**Keep only if:** Extra page-read calls and bytes; never directly overwrite a live neighbor's shared page, a known past correctness failure.

### I10. Avoid duplicate memory commits

Priority 2 · Evidence M · Boundary E · [src/platform.h:293](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/platform.h#L293)

Private Windows loads commit during admission and again during read/copy. Track committed coverage under a safe ownership invariant and skip redundant calls.

**Keep only if:** VirtualAlloc calls, commit failures and token time; verify the pages cannot be decommitted between admission and completion.

### I11. Batch compatible evictions

Priority 2 · Evidence M · Boundary E · [src/mc_state_lru.inl:101](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L101)

Each victim issues a decommit/view-drop call. Collect adjacent unprotected ranges and release them in batches outside long lock-held sections where safe.

**Keep only if:** VM calls, lock duration and reclaim effectiveness; protect shared boundary pages and in-flight readers.

### I12. Time-budget memory polling

Priority 1 · Evidence M · Boundary E · [src/mc_state_profile.inl:263](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_profile.inl#L263)

Auto tuning polls memory by group/read counts; async graphs check more frequently. Cache short-lived readings with immediate low-memory triggers and conservative reserves.

**Keep only if:** Memory-query cost per token and peak memory; do not trade away the hard cap for speed.

### I13. Coordinate reader and compute affinity

Priority 2 · Evidence H · Boundary E · [src/mc_state_io.inl:3](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L3)

Reader helpers are not jointly allocated with the compute pool. Avoid placing all work on the same execution resources when it increases completion latency.

**Keep only if:** Full request latency, memory bandwidth and CPU energy on the unchanged machine; no hardware upgrade is involved.

### I14. Deterministic prefill read-ahead

Priority 2 · Evidence H · Boundary U · [src/mc_compute.inl:200](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_compute.inl#L200)

When the next block is known to use essentially all experts, a bounded streaming lookahead may outperform speculative decode routing prediction.

**Keep only if:** Read-ahead usefulness, overlap and cache damage; do not claim future sparse routes are known before their authoritative router executes.

### I15. Enable prefetch only when it saves time

Priority 1 · Evidence C · Boundary E · [src/mc_state_io.inl:271](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L271)

The router predictor exists and has mixed recorded results. Gate it by per-layer on-time benefit, wasted bytes and demand queue pressure; learn breadth/depth rather than always prefetching.

**Keep only if:** Saved demand stall minus predictor/eviction/I/O cost; untouched routing preserves output only if residency synchronization remains correct.

### I16. Remove the pack's extra scatter copy

Priority 3 · Evidence M · Boundary V · [src/mc_state_pack.inl:273](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_pack.inl#L273)

A pack read currently lands in a temporary block and memcpy scatters gate/up/down into separate addresses. A slot layout or sidecar compatible with compute could read directly into owned final storage.

**Keep only if:** Read size benefit after conversion, scatter bandwidth, commit and cache-capacity costs; the current pack already lost with default views.

### I17. Order warm-start reads independently of retention rank

Priority 2 · Evidence M · Boundary E · [src/mc_state_profile.inl:178](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_profile.inl#L178)

Warm-start candidates are selected by utility and then read by do_reads, not the synchronous coalescing flush path. Preserve desired LRU rank separately while sorting/merging physical reads where useful.

**Keep only if:** First-request readiness including preload and first decode; no change to the selected expert set, profile semantics or memory cap.

## C. Cache policy, residency and memory efficiency (20 experiments)

### C01. Optimize stall cost per cache byte

Priority 2 · Evidence H · Boundary E · [src/mc_state_lru.inl:77](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L77)

ClassLRU considers age/frequency, not measured reload latency and expert size. Rank retention by expected avoided critical-path milliseconds per byte.

**Keep only if:** Trace replay then live missed-stall time; byte hit rate alone is not the objective.

### C02. Admission filter for one-use experts

Priority 2 · Evidence H · Boundary E · [src/mc_state_lru.inl:205](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L205)

LRU/ClassLRU admit demand misses. A small probation/TinyLFU-style admission policy can keep one-use scans from displacing valuable decode experts.

**Keep only if:** Mixed scan/chat traces, cache misses and token latency; admitted or bypassed experts must still be computed exactly.

### C03. Separate prefill and decode residency goals

Priority 2 · Evidence C · Boundary E · [src/mc_state_lru.inl:205](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L205)

Scan resistance already exists. Extend it to an explicit protected decode segment and bounded prefill streaming region with a measured transition.

**Keep only if:** Prefill time plus the first 32/128 decode tokens; existing scan resistance is not a new gain.

### C04. Soft per-layer cache quotas

Priority 2 · Evidence H · Boundary E · [src/mc_state_lru.inl:77](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L77)

A global cache can concentrate capacity in layers with many popular experts while one critical layer repeatedly stalls. Allocate marginal bytes using per-layer latency/miss curves.

**Keep only if:** Whole-token time at fixed total memory; compare against unrestricted global policy and permit quota borrowing.

### C05. Manage co-used tensor slices as a unit

Priority 2 · Evidence C · Boundary E · [src/mc_state_lru.inl:205](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L205)

Sibling loading already fetches gate/up/down. Admission/eviction can still operate on individual slices; compare expert-level utility and partially resident states.

**Keep only if:** Useful complete experts, premature slice evictions and saved stalls; keep fine-grained mode where heterogeneous tensors make it better.

### C06. Workload-conditioned warm profiles

Priority 2 · Evidence C · Boundary E · [src/mc_state_profile.inl:178](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_profile.inl#L178)

Profiles can already be named, but selection is mostly explicit. Combine a bounded request/session signature with existing profiles, retaining a global fallback.

**Keep only if:** Cold/first-turn latency after charging classification and profile storage; do not mix incompatible models or leak user content into telemetry.

### C07. Warm likely decode experts at prefill end

Priority 2 · Evidence H · Boundary E · [src/mc_state_rebalance.inl:64](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_rebalance.inl#L64)

Idle refill waits for a pause. Use the final prompt chunks as a cheap signal to arrange the first decode working set before the phase switch, only with spare bandwidth.

**Keep only if:** First-token and first-32-token latency including warmup; reject extra work on saturated SSDs.

### C08. Budget idle refill by expected reuse

Priority 2 · Evidence C · Boundary E · [src/mc_state_rebalance.inl:64](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_rebalance.inl#L64)

Refill is already on by default. Replace fixed byte/gain settings with bounded benefit estimates and cancellation at request arrival.

**Keep only if:** Next-turn wall latency, background bytes and energy; do not count idle refill as free if it competes with other activity.

### C09. Use cheaper replacement bookkeeping

Priority 2 · Evidence H · Boundary E · [src/mc_state_lru.inl:77](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L77)

Linked-list updates and class aging run under the cache mutex. Compare CLOCK/SIEVE or sampled victims to LRU/ClassLRU when bookkeeping is measurable.

**Keep only if:** Metadata CPU time and equal-memory miss/stall curves; a cheaper policy can lose more in SSD traffic than it saves in CPU work.

### C10. Prioritize sibling slices by consumption time

Priority 2 · Evidence C · Boundary E · [src/mc_state_lru.inl:205](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L205)

Sibling loading is useful but can read down weights before an urgent gate/up slice elsewhere. Retain batching while ordering by earliest consuming dependency.

**Keep only if:** Gate readiness, down readiness and queue contention; avoid canceling siblings that become imminent.

### C11. Choose view versus private path by evidence

Priority 2 · Evidence C · Boundary E · [src/mc_state_io.inl:671](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L671)

Both mapped-view and private/direct implementations exist. Select only after representative measurements of page faults, commit, copies and pressure on the current OS/backend.

**Keep only if:** Cold/warm full request time within the same memory limit; Windows view gains must not be assumed for every OS/model.

### C12. Bypass cache admission for selected scans

Priority 2 · Evidence C · Boundary E · [src/mc_state_lru.inl:405](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L405)

Long prefills can stream one-use experts without promoting them into decode residency, using bounded scratch and the existing protected cache.

**Keep only if:** Avoided pollution minus extra copies/reads; this is distinct from current cold-end insertion and must retain every required expert.

### C13. Reconcile logical residency with real page residency

Priority 2 · Evidence M · Boundary E · [src/platform.h:403](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/platform.h#L403)

A mapped expert marked resident may still be reclaimed by the OS. Sample actual fault/residency behavior and tune protection/population instead of trusting logical hit rate.

**Keep only if:** Compute-time major faults, physical memory and latency; avoid expensive residency scans on every token.

### C14. Losslessly compress cold in-RAM experts

Priority 3 · Evidence H · Boundary E · [src/mc_state_io.inl:671](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L671)

Only if measured quantized-byte compressibility and decompression cost justify it, keep a cold compressed tier and expand into bounded compute slots.

**Keep only if:** Net avoided SSD time minus decompression and capacity costs; low-bit GGUF may compress poorly, so this can correctly be rejected quickly.

### C15. Store the owning tensor index in cache metadata

Priority 2 · Evidence M · Boundary E · [src/mc_state_lru.inl:101](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L101)

Each evict performs a binary search over tensors to recover the node owner. A stable owner index or compact range directory can remove that work for frequent evictions.

**Keep only if:** Metadata cost per eviction and token time; maintain stable indices through model unload/reload and account for metadata bytes.

### C16. Release protection at last actual consumer

Priority 3 · Evidence M · Boundary U · [src/mc_state_lru.inl:205](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_lru.inl#L205)

Protection lasts for the whole graph, which safely prevents prior corruption but can restrict eviction longer than necessary. A validated last-use/reference-count plan could free completed groups earlier.

**Keep only if:** Lower peak required residency and stalled admissions; delayed readers and all asynchronous GPU consumers must be finished before unprotecting.

### C17. NUMA-aware first-touch and allocation

Priority 3 · Evidence H · Boundary E · [src/mc_state_io.inl:671](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L671)

On an existing multi-NUMA system only, place reader-filled expert pages near their compute consumers and partition work to reduce remote traffic. This is a software placement experiment, not a hardware upgrade.

**Keep only if:** Local/remote bytes and token latency on that same machine; skip on a single-NUMA machine and avoid duplicating weights beyond the cap.

### C18. Cheap statistical prefetch before router prediction

Priority 2 · Evidence H · Boundary E · [src/mc_state_io.inl:271](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L271)

Use a small per-layer co-activation/transition predictor when it is cheaper than evaluating the next router; select or combine predictors by on-time stall savings, not recall alone.

**Keep only if:** Prediction CPU time, on-time demand bytes saved and full decode latency; never substitute the predicted experts for the authoritative route.

### C19. Reduce the existing predictor's own work

Priority 2 · Evidence M · Boundary E · [src/mc_state_io.inl:271](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_io.inl#L271)

The next-router predictor rescales activations and loops across router rows. Cache norm-ratio constants, reuse scratch and compare blocked/vectorized router matvec variants before increasing prediction depth.

**Keep only if:** Predictor time and net saved demand stalls; changed predictions are allowed only as advisory I/O, with intact correctness synchronization.

### C20. Choose warm-start amount by incoming phase

Priority 1 · Evidence C · Boundary E · [src/mc_state_profile.inl:178](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_state_profile.inl#L178)

preload runs on the first graph and uses a fixed/profile-based fraction. A long cold prefill may reread or displace much of a startup fill. Compare phase-aware defer/partial preload with immediate warm fill.

**Keep only if:** Cold TTFT plus first decode tokens, all startup bytes included; retain the current warm policy when it wins follow-up use.

## L. Planning and speculative execution (11 experiments)

### L01. Per-model compute-buffer calibration

Priority 1 · Evidence M · Boundary V · [planner/gguf_plan.py:385](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/planner/gguf_plan.py#L385)

The compute table is measured mainly on Flash-Next and estimates other architectures. Persist conservative measured buffers by model layout, backend, build, KV type, context and batch.

**Keep only if:** Avoided GPU fallbacks and selected plan speed; retain headroom for changing external VRAM use.

### L02. Optimize end-to-end plan cost

Priority 1 · Evidence M · Boundary V · [planner/gguf_plan.py:419](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/planner/gguf_plan.py#L419)

The planner walks 2048/1024/512 and selects a fitting choice rather than minimizing measured request time. Score prompt length, expected output length, copies, cache misses and decode placement jointly.

**Keep only if:** Full request latency at fixed quality and memory; evaluate short chat separately from long documents.

### L03. Cache validated autotuning results

Priority 2 · Evidence H · Boundary V · [planner/gguf_plan.py:419](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/planner/gguf_plan.py#L419)

Store a small performance frontier by model/build/backend rather than rerunning every probe at every start. Invalidate changed drivers, code, model layout and memory conditions.

**Keep only if:** Saved startup cost and stable near-best choices; never reuse a stale unsafe memory plan.

### L04. Choose static GPU layers by measured benefit

Priority 2 · Evidence C · Boundary V · [planner/gguf_plan.py:419](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/planner/gguf_plan.py#L419)

The simple CPU-layer count uses average bytes. Explore supported tensor overrides/per-layer placement to maximize avoided CPU work or reads per VRAM byte.

**Keep only if:** Total decode/prefill time including extra split/transfer boundaries; placement changes need logits/quality tests.

### L05. Allocate RAM between prompt cache and experts adaptively

Priority 2 · Evidence C · Boundary W · [bin/moe_cache.py:1645](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L1645)

The launcher already caps host prompt cache and disables it under small limits. Compare marginal prefix-reuse savings with marginal expert-cache stalls instead of a fixed split.

**Keep only if:** Real multi-turn request time at equal peak memory; do not take RAM from expert caching for a prefix that is never reused.

### L06. Grow context storage only as required

Priority 3 · Evidence H · Boundary U · [planner/gguf_plan.py:419](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/planner/gguf_plan.py#L419)

Reserving for the maximum context can reduce expert placement before the context is used. Paged/on-demand KV and compute-buffer policies can defer cost without shortening the requested window.

**Keep only if:** Early-turn decode, long-context growth cost and memory cap; requires compatible upstream allocation and recurrent-state handling.

### L07. Plan MTP and expert residency together

Priority 1 · Evidence C · Boundary V · [bin/moe_cache.py:1153](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L1153)

MTP consumes weights, KV and buffers that may shrink the expert cache. Include draft memory and induced reads in the plan, not just a model-wide on/off rule.

**Keep only if:** Milliseconds per accepted target token including prefill and draft loading; never claim acceptance alone proves a gain.

### L08. Adapt draft length online

Priority 2 · Evidence C · Boundary V · [bin/moe_cache.py:1153](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L1153)

A fixed draft length of two misses workload variation. Try bounded 0/1/2/4-length choices based on measured verification time and acceptance, with explicit fallback.

**Keep only if:** Accepted target tokens per wall second on prose/code/agent tasks; use a correct verifier and distinguish distributional exactness from identical RNG traces.

### L09. Deduplicate expert work across verification

Priority 2 · Evidence H · Boundary V · [src/mc_exec.inl:382](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L382)

Multi-token verification already groups weight use per expert; extend queued-read deduplication and shared weight decoding across the draft's union of required experts.

**Keep only if:** Actual distinct experts/bytes per accepted token; no speculative answer may bypass target verification.

### L10. Tune verification/executor crossover together

Priority 2 · Evidence M · Boundary V · [src/mc_exec.inl:298](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L298)

The executor stops at eight columns and stock can switch to tiled arithmetic at eight rows per expert; Vulkan has its own batch dispatch boundaries. A larger draft can abruptly change paths.

**Keep only if:** Latency sweeps around every boundary, repeated-id cases and stock-compatible logits; do not just raise EXEC_MAXCOL.

### L11. Prompt/history lookup speculation without a draft model

Priority 2 · Evidence H · Boundary V · [bin/moe_cache.py:1153](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L1153)

Where the pinned llama-server supports it, use repeated prompt/history token sequences as a lightweight proposal source instead of loading an MTP/draft head. Verify every proposal with the target.

**Keep only if:** Accepted tokens per millisecond after verification and expert I/O, on repetitive code versus prose; correct sampling/greedy semantics and a zero-gain fallback.

## U. Upstream/backend integration opportunities (10 experiments)

### U01. Attention kernel shape tuning

Priority 3 · Evidence H · Boundary V · [docs/PERFORMANCE_ANALYSIS.md:1](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/docs/PERFORMANCE_ANALYSIS.md#L1)

At long depth, tune existing attention tile/subgroup sizes, vectorization and intermediate traffic for the current backend and KV layout, without dropping tokens.

**Keep only if:** Kernel time and full token time at 4k/32k/128k/256k; Flash-Next's historical attention growth was a small share of decode, unlike some other architectures.

### U02. Verify upstream expert-id hoisting is actually used

Priority 2 · Evidence C · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

llama.cpp PR 28501 already raised the Vulkan expert limit to 1024 before the bundled release. Audit dispatch coverage and preserve that path in compact/padded plugin graphs; do not count the merged patch as new headroom.

**Keep only if:** Shader dispatch evidence and ids-scan work for 512/1024 experts; test before calling any missing path a bug.

### U03. Fused quantized GPU expert kernels

Priority 3 · Evidence H · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Maya/Strata explore fused dequantization, matrix instructions and populated tiles. Prototype only the measured hot quant/shape rather than importing an entire engine.

**Keep only if:** Whole-prompt time including routing/packing; compare numerical changes, and keep portable fallback paths.

### U04. Fuse GPU gate/up and activation

Priority 3 · Evidence H · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Current cloned graph leaves kernel fusion to the backend. A backend operation can reuse the input, fuse gate/up/activation and reduce launches/intermediate memory traffic.

**Keep only if:** Kernel count, VRAM traffic and TTFT; retain scale, bias, activation and rounding semantics.

### U05. Capture/replay stable backend command graphs

Priority 3 · Evidence H · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

Plugin metadata reuse does not necessarily eliminate backend command submission overhead. Reuse compiled pipelines/command buffers or capture stable compute graphs where supported.

**Keep only if:** CPU launch overhead and steady decode/short-prefill time; update dynamic ids and addresses correctly and invalidate on allocation changes.

### U06. Improve stock tiled fallback for small MoE batches

Priority 3 · Evidence H · Boundary V · [src/mc_exec.inl:298](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L298)

The plugin deliberately returns repeated/high-row cases to upstream tiled kernels. Optimize that path or expose it to a ready scheduler instead of disabling the correctness guard.

**Keep only if:** Tiled-op latency, forced-tiled tests and exact reference comparisons; leave arithmetic-compatible fallback intact.

### U07. Overlap independent shared and routed branches

Priority 3 · Evidence H · Boundary U · [src/mc_compute.inl:200](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_compute.inl#L200)

Some models have a shared expert branch independent of routed experts until a final sum. A scheduler could overlap them instead of serializing the whole layer.

**Keep only if:** Critical-path timeline and final outputs; only genuinely independent graph branches qualify.

### U08. Adapt CPU/GPU work by measured completion time

Priority 3 · Evidence C · Boundary V · [src/mc_hybrid.inl:100](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_hybrid.inl#L100)

Hybrid already runs GPU hits alongside CPU misses and adapts the bank. Add a cost model that accounts for transfer, CPU bandwidth competition and bank reloads instead of optimizing hit count alone.

**Keep only if:** Total layer/request time, swaps and refill cost; existing hybrid placement changes arithmetic relative to stock.

### U09. Lend inactive output-head/dense-weight memory

Priority 3 · Evidence H · Boundary U · [planner/gguf_plan.py:419](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/planner/gguf_plan.py#L419)

In suitable architectures, temporarily stream weights whose allocation is idle during a prompt phase and lend the memory to larger prefill chunks, restoring them before decode.

**Keep only if:** TTFT plus all restores and next-token time; difficult with unmodified llama.cpp ownership, not a plugin-only promise.

### U10. Evaluate zero-copy host access for tiny transfers

Priority 3 · Evidence H · Boundary V · [src/mc_backend.inl:197](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_backend.inl#L197)

For backends exposing appropriate host buffers, small activations/route metadata may be cheaper to read directly than to stage and synchronize. Large expert matrices may be much worse.

**Keep only if:** Latency by transfer size and end-to-end impact on the unchanged machine; do not replace bulk copies based on a small-buffer microbenchmark.

## S. Serving, prompt reuse and agent latency (9 experiments)

### S01. Stable exact-prefix serialization

Priority 1 · Evidence C · Boundary W · [bin/moe_cache_ui.py:1623](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache_ui.py#L1623)

Keep unchanged system/tool/repository content byte-stable across turns so llama-server can reuse a prefix. Avoid needless template/tokenization differences; never remove required user content.

**Keep only if:** Tokens actually recomputed and request TTFT; report as avoided prefill, not a higher fresh-prefill kernel rate.

### S02. Explicit prefix/slot affinity

Priority 2 · Evidence C · Boundary W · [bin/moe_cache.py:2145](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L2145)

The server already matches slots by prompt. Expose or measure affinity for independent agents so a short request does not displace a valuable long conversation.

**Keep only if:** Cache-hit prefix tokens, evictions and multi-agent request time; memory cap and per-client isolation remain unchanged.

### S03. Tune checkpoint retention by workload

Priority 2 · Evidence C · Boundary W · [bin/moe_cache.py:1645](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L1645)

Eight recurrent checkpoints and host reserves are defaults, not universal optima. Retain points that avoid the most real reprocessing per byte and save cost.

**Keep only if:** Editing-in-the-middle agent traces, peak memory and latency; every required recurrent state must be restorable.

### S04. Incremental or nonblocking slot persistence

Priority 3 · Evidence M · Boundary U · [bin/moe_cache.py:2132](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L2132)

Warm saves already have size/change-aware delays, but each save rewrites a full slot and can block the server. Upstream snapshot/copy-on-write or delta persistence could reduce pauses.

**Keep only if:** Foreground p99 latency, saved bytes and crash recovery; requires upstream state-format/lifetime support.

### S05. Tune idle release/unload from reuse evidence

Priority 2 · Evidence C · Boundary W · [bin/moe_cache_on_demand.py:248](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache_on_demand.py#L248)

Idle release saves memory but can force expert refill or model restart on the next request. Choose timeouts from actual tool-pause distributions with explicit memory limits.

**Keep only if:** Complete agent-session time and background energy; improves repeated requests rather than a cold decode kernel.

### S06. Reuse local proxy connections

Priority 2 · Evidence M · Boundary W · [bin/moe_cache_on_demand.py:459](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache_on_demand.py#L459)

The on-demand proxy creates a fresh backend HTTP connection per request. A bounded keepalive pool can remove repeated setup for many short agent calls.

**Keep only if:** Client TTFT, CPU/request and connection churn; drain canceled/failed streams safely before reuse and preserve authentication.

### S07. Flush complete SSE events efficiently

Priority 2 · Evidence M · Boundary W · [bin/moe_cache_ui.py:2028](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache_ui.py#L2028)

The UI proxy flushes each received line, including SSE separators. Forward complete events or readily available chunks without delaying the first event; inspect TCP_NODELAY behavior.

**Keep only if:** Client-visible inter-token latency and syscalls, separately from server tokens/s; never buffer a whole answer to improve a benchmark.

### S08. Cache compatible tokenization/template results

Priority 3 · Evidence H · Boundary U · [bin/moe_cache_ui.py:1623](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache_ui.py#L1623)

Long identical prefixes can repeat tokenizer/template work before inference. Cache by tokenizer/model/template/content identity or use upstream reuse rather than retokenizing unchanged bytes.

**Keep only if:** Pre-inference CPU time and actual TTFT; template changes, special tokens and model identity must invalidate results.

### S09. Restore only the warm session the request can use

Priority 2 · Evidence M · Boundary W · [bin/moe_cache.py:2145](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/bin/moe_cache.py#L2145)

Startup restores the two largest matching saved slots before knowing the incoming conversation. A request-aware or lazy restore can avoid loading irrelevant multi-GB states.

**Keep only if:** Cold-start TTFT plus later reuse and memory; fix shard/model state identity first and never restore a mismatched state.

## B. Build and small runtime overhead (6 experiments)

### B01. Compare strict O3 with current O2

Priority 2 · Evidence M · Boundary V · [build.sh:32](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/build.sh#L32)

build.sh uses O2. Test O3 for the plugin and relevant backend kernels with the same floating-point policy; inspect code size and vectorization.

**Keep only if:** Whole-model paired runs plus byte/logit checks; no fast-math, unsafe reassociation or assumed positive result.

### B02. Profile-guided optimization

Priority 2 · Evidence H · Boundary E · [build.sh:32](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/build.sh#L32)

Train PGO on real mixtures of decode, misses, prefill, fallback and idle transitions to improve branches, inlining and hot/cold layout.

**Keep only if:** Out-of-training workload latency and build reproducibility; keep exact arithmetic and safe error paths.

### B03. Measure LTO rather than assume it helps

Priority 3 · Evidence H · Boundary E · [build.sh:32](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/build.sh#L32)

The plugin is already largely one translation unit. LTO may still help selected boundaries/linkage but has less obvious opportunity than a multi-file program.

**Keep only if:** Binary/assembly differences and end-to-end timing; reject a no-op or compile-cost-only change.

### B04. Ship measured CPU multiversion dispatch

Priority 3 · Evidence C · Boundary V · [build.sh:32](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/build.sh#L32)

Portable builds should retain compatibility while validated existing-CPU variants optimize hot metadata/kernels. Dispatch by proven compatibility and speed rather than advertising a wider ISA alone.

**Keep only if:** Streaming benchmarks and complete runs per variant; this is software targeting of existing hardware, not new hardware.

### B05. Resolve hot indirect calls and metadata lookups once

Priority 2 · Evidence M · Boundary E · [src/mc_buffer.inl:52](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_buffer.inl#L52)

Type-trait, buffer-context registry and kernel function lookups can be cached in validated operation descriptors rather than rediscovered repeatedly; keep public ggml buffer APIs.

**Keep only if:** Indirect calls/lock contention per token and wall time; preserve ABI checks, unload invalidation and dynamic backend support.

### B06. Compare allocator/page policies for persistent scratch

Priority 3 · Evidence H · Boundary E · [src/mc_exec.inl:382](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/src/mc_exec.inl#L382)

Bounded arenas, explicit alignment and selective huge-page advice can reduce scratch allocation/TLB overhead without increasing live expert data. Do not pin the entire model.

**Keep only if:** Allocator/TLB cost, resident memory and whole request latency; huge pages and locked memory can worsen eviction, so keep an evidence-based fallback.

## Already implemented: do not count these as new gains

| Existing technique | Current evidence | What remains |
|---|---|---|
| Resident-first CPU executor | `mc_exec.inl` partitions ready and late tasks | D01 handles readiness changes after that initial partition |
| Persistent CPU worker pool | `mc_pool.inl` | Reduce useful-work granularity and completion overhead; do not merely create another pool |
| Gate/up single pool job | `exec_run_prepared`, default on | Share compatible preparation and extend validated fusion coverage |
| Row chunks of 128 | RESULTS 64 addendum: about +3.9% over the old 32-row setting on the desktop | D06 tests other shapes; the already-shipped +3.9% cannot be claimed again |
| Small-batch weight reuse | Multiple token columns grouped per expert, up to eight | Register-blocked multi-column kernels, with tiled-path boundaries respected |
| Q2_0 SIMD and prepared column sums | Runtime compatibility checks choose matching kernels | Any further kernel must win streaming end-to-end, not just cached dots |
| IQ3 gather and optional AVX-512 paths | Self-check and vendor gating exist | A different implementation/calibration, not simply enabling a flag that already lost |
| Async input readback | `be_cpy_tensor_async`, default | Finer dependencies where the backend can express them |
| SSD fill overlapped with GPU uploads | `MOE_CACHE_GPU_PIPE`, default | Overlap upload with compute and avoid ordered waits |
| Pinned result readback | `MOE_CACHE_GPU_READBACK`, default | Compact useful rows and reduce bytes; not a second claim for pinned readback |
| Upload only used experts | `MOE_CACHE_GPU_PACK`, default | Packing does not automatically compact every routed-output row |
| Split GPU blocks to fit memory | Expert partitions already exist | Remove work/readback discarded by those partitions |
| Synchronous adjacent-read coalescing | `flush_reads`, capped | Async completion-aware coalescing and mapped-view batching |
| Scan-resistant admission | Long prompts load at the cold end | Explicit phase budgets and measured decode transition |
| File-view cache | Windows and Linux paths exist | Avoid redundant page-management work and monitor real faults |
| Sibling gate/up/down loading | Default on | Priority and admission refinements |
| Profiles, starter profiles and warm sessions | Existing launcher/plugin mechanisms | Workload-aware choices and less foreground persistence cost |
| Memory/commit-aware sizing | Existing tuner, cap and fallback mechanisms | Lower safe polling/VM overhead; not removal of limits |
| Micro-batch/placement planning | Existing joint memory plan | Optimize measured latency, not just fit |
| Stock bypass for models that fit | Existing launcher decision | Preserve it unless a specific plugin path demonstrably wins |
| MTP and hybrid GPU/CPU experts | Existing, model/workload dependent | Joint cost model and lower dummy work/transfer costs |
| ClassLRU | Opt-in; RESULTS 73: +1.3–2.9% on tested 122B pairs, mixed on Flash-Next | Broader measurement before changing defaults |
| Idle cache refill | Default since v0.3.3 | Better utility/cancellation and phase-transition policy |

The above percentages are historical repository results with their original conditions. They are not new measurements from this audit and are not all applicable to the same model/configuration.

## Negative results and traps to retain

| Attempt | Recorded outcome / limit | Audit consequence |
|---|---|---|
| Longer executor spin (N30) | Roughly 0.8–2% slower decode with 14–23% more CPU time, despite far fewer parks | Do not equate fewer sleeps with higher throughput; D01 is a different scheduling change |
| Current gather / AVX-512 IQ kernels on Zen 4 | RESULTS 55: tested variants about 0.56–0.77x stock kernel speed | Do not turn them on as a free upgrade; new kernels need different evidence |
| Current expert pack with default file views | RESULTS 74: reader 1.11→1.17 GB/s but decode roughly 15.6–16.1→14.5–15.1 tok/s | More I/O bandwidth did not mean a faster model; keep pack opt-in |
| Pack in private-memory mode | Large gains in a much slower private-path condition | Does not prove an improvement over the existing default views |
| N20 faster Q2_0 microkernel | Cached kernel win, no clear PC decode win | Memory/whole-model effects can erase instruction-level gains |
| Router lookahead plus executor on tested 122B (D5) | One pair won and the other lost; not kept | Prefetch must be gated by useful on-time stalls avoided |
| MTP on Flash-Next with the small-VRAM laptop setting | Earlier result about 17% slower than no MTP | Draft memory and verification I/O can outweigh accepted tokens |
| Larger prompt micro-batches | Can reduce the number of streamed passes, but can also consume the room needed for useful GPU placement or trigger slower kernels | Benchmark the full plan; no universal `-ub 8192` recommendation |
| More GPU layers | Can squeeze prefill workspace, prompt buffers, MTP or KV, and cause CPU fallback | More offload is not an invariant improvement |
| Stronger working-set locking / full-arena pinning | May hurt Windows commit/VRAM availability or OS reclaim | Pin bounded staging only when total request time improves |
| Smaller GPU expert caches | Historical 500 MiB cache was much slower than static placement on the laptop | Do not assume an additional tier is useful at every capacity |
| Turning off required waits, duplicate-id checks or page-boundary protections | Prior bugs produced wrong bytes, corrupt routing and crashes | Never call correctness removal an optimization |
| Strata grouped GEMM experiment | Its published prefill notes report an isolated kernel win but an engine-level loss | Imported techniques must be retested in moe-cache, not credited from a paper/README |

Sources: pinned [RESULTS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/docs/RESULTS.md), [PERFORMANCE_ANALYSIS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/526c804610cf835fdfcccd379fd7072d24e1aa2f/docs/PERFORMANCE_ANALYSIS.md), and the Strata prefill source listed below. Individual timing rows have differing cleanliness, builds and baselines; use each row's caveats.

## How far can pure optimization go?

There is no defensible universal percentage from this static review. The present code has avoidable work, so it is not justified to declare it fully optimized. It is equally unjustified to promise 2x or 3x more whole-model throughput from 124 candidates. The ceiling is workload-specific and moves as bottlenecks are removed.

### Small gains compound only when they are real and independent

Ten independent 1% throughput wins yield `1.01^10 = 1.1046`, about **10.5%**, not 10 guaranteed percentage points. Twenty yield about **22.0%**. Ten independent 2% wins yield about **21.9%**. In practice these changes share bottlenecks: eliminating a wait can reduce the value of a later queue tweak, and bigger batches can undo a cache gain. Re-measure the cumulative branch after every merge.

For a stage taking fraction `f` of request wall time, speeding it up by `s` yields:

`whole-request speedup = 1 / ((1 - f) + f / s)`

| Stage share of wall time | Make that stage 2x faster | Remove that stage entirely (unrealistic limit) |
|---:|---:|---:|
| 1% | 1.005x (+0.5%) | 1.010x (+1.0%) |
| 10% | 1.053x (+5.3%) | 1.111x (+11.1%) |
| 30% | 1.176x (+17.6%) | 1.429x (+42.9%) |
| 60% | 1.429x (+42.9%) | 2.500x (+150%) |
| 80% | 1.667x (+66.7%) | 5.000x (+400%) |

These are arithmetic scenarios, not estimates that a particular listed stage occupies those fractions in the current build.

### Decode ceiling

Use actual expert bytes and critical-path wall measurements. A lower bound includes at least the largest of the non-overlappable limits: selected weight bytes divided by sustainable memory bandwidth, genuinely cold bytes divided by effective storage bandwidth, required operations divided by usable compute, mandatory transfers, and dependency/synchronization time. A pipeline can hide some transfers, but it cannot make them free or bypass causal layer dependencies.

Historical repository measurements put a simple memory read around 50–55 GB/s while a measured expert phase was nearer 26 GB/s. That is a lead to investigate, not proof the entire decoder can double: arithmetic, short operations, reads, cache misses and synchronization are included differently. The newer gate/up fusion already removes part of the historical overhead.

Likewise, the reported 36% busy / 32% read wait / 32% barrier split is **summed pool thread time**, not a partition of token wall time. Taking `1/0.36` and promising 2.78x decode would be wrong. A worker spinning may overlap useful work on another worker or I/O.

### Prefill ceiling

For a streaming prompt, a useful first approximation is:

`number of passes × cold expert bytes per pass / effective storage throughput`, plus the nonhidden transfer, compute, attention and orchestration costs.

Larger effective batches can reduce passes; layer reuse can reduce bytes per pass; compact split execution can remove dummy rows; staging and asynchronous execution can overlap phases. Those can be larger wins than scalar instruction cleanup when the current path repeatedly moves or computes unnecessary data.

In an ideal *fully serial* three-stage load/upload/compute pipeline with stage times `a,b,c`, a fully overlapped pipeline approaches `max(a,b,c)` instead of `a+b+c`, after startup/drain. Balanced stages would permit a 3x **pipeline-stage** improvement in that artificial baseline. Moe-cache already overlaps some phases; real transfers contend for memory, finite buffers consume cache, and attention remains. This is not a 3x prediction for moe-cache.

The old pressured 34k run attributed 61% of its interval to the plugin and 39% to gaps. Even deleting the plugin's work entirely would cap that exact old interval around 2.56x, and doubling it would give about 1.44x, assuming nonoverlapping wall-time attribution. It was a noisy, older run, so those numbers must not be used as the current ceiling. New tracing should resolve what the gaps actually contain before treating all of them as removable scheduler overhead.

### Reuse ceiling

If a 100k-token prompt reuses a validated 95k-token prefix, only the suffix may need fresh prompt computation. The resulting TTFT reduction can be manyfold. It is real application efficiency, but it is not a 20x improvement in fresh-prefill kernels. Keep prefix-reuse results separate from uncached prompt throughput.

**Practical judgment:** pursue the visible waste first, then measure the new roofline. Substantial gains are possible in a path that currently performs repeated transfers, dummy rows or avoidable fallbacks; a genuinely bandwidth-saturated warm decode will eventually have only small software headroom. This audit establishes experiment targets, not a measured remaining percentage.

## Twelve measurement steps that make 1–2% wins credible

1. **Freeze the baseline:** model/shard hashes, quant, llama.cpp/plugin commits, compiler flags, backend, driver, all runtime flags, context and requested/generated token counts. Fix the split for strict exactness tests. Do not silently compare different drives or quality levels.
2. **Measure complete latency:** uncached TTFT, cached TTFT, prompt evaluation time, first-32-token decode, steady decode, full agent-session wall time and p50/p95/p99 token gaps. Report throughput and latency separately.
3. **Measure wall-stage dependencies:** CPU/GPU timelines with spans for preparation, waits, reads, H2D, compute, D2H, stock attention/lookup kernels and inter-split gaps. Avoid adding overlapping worker durations as if they were wall time.
4. **Measure actual data movement:** demand/prefetch SSD bytes, H2D/D2H bytes, useful rows, discarded rows, coalescing sizes, first-use faults and in-flight versus fully ready hits.
5. **Separate regimes:** cold storage/cold process, warm process/cold selected experts, warm cache; short versus long context; one request versus concurrent agents; memory-resident versus miss-heavy models. A win in one regime is not universal.
6. **Randomize paired runs:** use repeated randomized A/B blocks or balanced ABBA with identical prompts and output lengths. Keep workload order and cache treatment explicit. A lone fastest run does not establish 1%.
7. **Let variance determine run count:** use a pilot to estimate paired variance and collect enough independent blocks for confidence intervals narrow enough to resolve the desired gain. For tiny effects this may take dozens of pairs, not one or two. Do not pretend every token in one run is an independent machine restart.
8. **Use stable work:** score decode over enough tokens, preserve full answers and separately flag EOS/truncation. Account for repeated prompt contents and prefix-cache hits; avoid timing a faster but shorter answer.
9. **Preserve correctness:** kernel bytes, expert-op bytes, fixed-plan greedy tokens, logits/KL where arithmetic paths change, duplicates, partial reads, delayed completion, shared boundary pages, eviction and unload/reload. Test long-context recurrent checkpoints as well as a 4k task set.
10. **Validate resource tradeoffs:** fixed peak RAM/VRAM limits, background I/O, CPU seconds and joules/token where available. More spinning or a larger hidden cache can fake a throughput win while worsening efficiency.
11. **Measure outside the profiler too:** detailed timers and GPU performance loggers can synchronize and change scheduling. Keep an instrumented diagnosis run and an uninstrumented timing run with identical semantics.
12. **Recheck interactions:** compare baseline, each change alone and the combined branch. Maintain a small representative held-out prompt suite and a regression guard; no re-crediting already-shipped gains.

### Suggested experiment matrix

| Axis | Minimum useful coverage |
|---|---|
| Model/quant | Flash-Next Q2_0, IQ3_XXS; one K-quant/MXFP4 model; a many-expert miss-heavy case |
| Fresh prompt lengths | 32, 128, 512, 4k, 32k, 128k where model/context permits |
| Decode depths | 4k, 32k, 128k; compare the same configured context when isolating depth |
| Decode work | Fixed 256–1024 generated tokens for timing; separately retain real EOS behavior |
| Expert residency | High hit rate, working set over RAM, and very small cache for correctness stress |
| GPU block mode | Unsplit, 2/3/4-way split and forced CPU fallback |
| Verification batches | 1,2,4,7,8,9,16,32 tokens where supported, with per-expert row counts recorded |
| Agents | Single conversation, edit in middle, two agents, idle/tool pauses and a restart/restore |

The matrix is a menu for targeted experiments, not an instruction to run every cross-product after every small edit. Use mechanism-specific tests first, then representative full runs and the relevant stress cases.

## Implementation sequence

**Wave 0 — accurate baseline and invariants.** Establish the measurement protocol and fix the previously confirmed state-identity/key/redirect defects before relying on persisted model-specific plans. Those security/state findings were documented in the prior audit; their edge-case repros were not rerun in this performance round, so the passing security suite is not proof they disappeared.

**Wave 1 — small local work removal.** D03, D11–D14, D17–D19, I08, C15 and B05. Most changes are localized and suitable for strict operation-byte tests. Keep individual effects even below 1% if repeated paired measurements support them and maintenance cost stays low.

**Wave 2 — avoid bulk wasted work.** P01–P05, P09, I01/I07, L01/L02 and C20. First diagnose where split mode is actually used; P01/P02 do not help unsplit blocks. P01 can change kernel shapes, so it needs stronger numerical validation than merely rearranging a copy.

**Wave 3 — completion-driven execution.** D01/D02, I02–I04 and C16. Define explicit ownership states and consumer lifetimes before reworking flags/queues. This is where speed changes can reintroduce previously fixed use-before-load bugs if invariants are not preserved.

**Wave 4 — policy and full-request savings.** C01–C10/C18/C19, L05/L07–L11, S01–S09. Use trace replay to reject weak policies cheaply, then live workloads. Do not count a better prefix hit or a shorter restore as an increase in raw decode token throughput.

**Wave 5 — backend/upstream work.** P19–P24, U01–U10 and supported compiler/kernel experiments. Build separate prototypes and keep an ordinary llama.cpp/plugin path. A successful upstream backend patch can benefit moe-cache without requiring its own tokenizer/server/model engine.

## Adjacent speed/quality tradeoffs (not pure unchanged-quality optimization)

These remain useful to test if the owner explicitly wants a quality/performance frontier, but they must not inflate the 124 candidate count or the exact-mode speedup:

1. Lower KV precision such as q8/q4, beyond the currently documented q8 default conditions.
2. Fewer experts per token or cache-aware route dropping; existing `--fast`/`--fast=fewer` already do variants.
3. Substituting a resident expert for the actual routed expert.
4. Stronger/mixed weight quantization or lower-precision cached copies of selected tensors.
5. Attention-token eviction, KV sparsification or compressed approximate attention that discards information.
6. Early exit through layers or confidence-based skipping of model computation.
7. Prompt summarization, truncation or reducing the requested context.
8. Shorter reasoning/output budgets or a different faster model.

Lossless layout changes, caching exact prior work and correct target-verified speculative proposals are different: they can preserve the intended computation/distribution, subject to implementation tests. Mathematical equivalence is still not automatic bit identity when floating-point order or dispatch changes.

## Related-project evidence and what it actually supports

| Primary source | Transferable lesson | Limit |
|---|---|---|
| [Strata prefill breakdown](https://github.com/Niko1221/Strata/blob/main/bench/results/2026-09-28-prefill-speed/README.md) | Scratch reuse, larger useful chunks, helper-thread staging, PLE prefetch and streaming can compound. Some steps change arithmetic, others only movement. | Different engine, NVIDIA/64 GB test; its combined speedup is not a moe-cache prediction. |
| [oMLX resident-first prefill PR 4313](https://github.com/jundot/omlx/pull/4313) | Consuming resident experts before admitting misses can reduce rereads; chunk boundaries also matter. | Its reported 10–61% prompt gains are on Apple hardware and specific residency conditions, not this plugin. |
| [oMLX overlap/cache PR 4240](https://github.com/jundot/omlx/pull/4240) | Read-ahead and phase-dependent memory borrowing are worth isolating as separate experiments. | Merged implementation; emulated memory-constrained setup and limited runs, not proof of equivalent gains elsewhere. |
| [KTransformers layerwise prefill](https://ktransformers.net/en/docs/optimization-techniques/layerwise-prefill) | Larger GPU work units amortize transfers in long prompts, shifting the bottleneck toward movement. | Requires extra workspace and scheduling control; not every native quant/model/backend path is supported. |
| [Borrowed Prefill](https://github.com/05yuki/kt-prefill-lend) | Temporarily lending inactive weight allocations can enlarge useful prompt chunks. | A specialized KT/SGLang/CUDA implementation; ownership and restore costs are not automatically accessible to a ggml plugin. |
| [KTransformers MESH PR 2003](https://github.com/kvcache-ai/ktransformers/pull/2003) | Explicit residency, async completion, I/O validation and phase-specific cache modes. | Open experimental PR, with AMX/NUMA-specific manual validation; not a production guarantee. |
| [Maya fused RDNA expert source](https://github.com/mw00/project-maya/blob/main/src/prefill/moe_fused_rdna4.cu) | Quant-aware matrix instructions and populated tiles deserve backend prototypes on compatible existing GPUs. | Native pack layouts and arithmetic semantics differ; no direct performance comparison was run. |
| [Fiddler](https://github.com/efeslab/fiddler) | Sometimes moving small activations to CPU compute is cheaper than transferring whole expert weights. | Old proof-of-concept, different model/precision; moe-cache's CPU offload and hybrid path already embody parts of this idea. |
| [Cintu07/strata](https://github.com/Cintu07/strata) | Useful reference implementations of admission, layout and expert-centric read planning. | Separate from Niko1221/Strata; published dramatic prefill ratios are synthetic scheduling/I/O tests, not full LLM results. |
| [Peregrine](https://github.com/s-b-repo/peregrine) | Completion-owned scheduling, deadline-aware prefetch and predictor scoring are candidate design patterns. | Research repository; CUDA lane includes host-gated/stub caveats. Do not treat its feature list as verified comparable throughput. |
| [llama.cpp Vulkan PR 28501](https://github.com/ggml-org/llama.cpp/pull/28501) | Expert-count dispatch limits can silently disable efficient paths. | Already merged before the bundled b11966 baseline; verify coverage, do not re-credit the old reported gain. |
| [llama.cpp Vulkan issue 25356](https://github.com/ggml-org/llama.cpp/issues/25356) | Sweep batch dispatch boundaries and rows per expert rather than assuming smooth scaling. | A reported gfx1151/model-specific cliff, closed as not planned; it does not establish the same bug on the current 9070 XT build. |
| [llama.cpp speculative server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | Prompt/history lookup proposal modes provide alternatives to a memory-heavy draft head. | Check the actual pinned binary's help/support and verifier semantics before enabling or tuning a mode. |

## Review limits and final judgment

This audit identifies source mechanisms and research directions. It does not assert that 124 independent wins exist or that every byte of every dependency was audited. Vendored headers/tests/docs were considered for compatibility and evidence; GPU drivers, compiler implementations, full llama.cpp internals and external engines were not formally audited. The source anchors identify the baseline inspected; external main-branch links are mutable.

The strongest near-term case is to **remove measured unnecessary work before adding complicated predictions**. The small changes are worth retaining when they survive whole-request timing, but their sum must be measured. The largest prefill upside is likely in reducing repeated weight movement, discarded output work and memory-driven fallbacks. The largest decode questions are readiness scheduling, per-expert dependencies, useful memory bandwidth and cache miss cost. The exact remaining multiple is unknown until those questions have wall-time measurements on the current build.
