# Layer-wise prefill (old name P21): design

Status: design written before the code (2026-11-01). The result section at the end is filled in after the laptop checks. Switch: `LLAMA_LAYERWISE_PREFILL=N` (patch `0050-layerwise-prefill`, default off, stock when unset). Everything below about llama.cpp is read from the pinned b11966 (`ff0d396e3`) sources; file:line are in that tree.

## 1. The idea in one paragraph

Today a prompt of 34k tokens is cut into chunks (ubatches, `-ub 1024`). Each chunk goes through all layers, so the experts of layer L are read from RAM/disk and pushed to the GPU once per chunk: 34 times for 34k tokens. With the switch, N chunks form a group. The group goes through layer 0 (chunk 1, chunk 2, ... chunk N), then through layer 1, and so on. The experts of a layer are needed N times in a row, so they are moved once per group, not once per chunk.

## 2. Why the result is the same (causality, per part of qwen4exp)

Order inside one layer stays the stock order (chunk 1, then 2, ...). Only the order of the layers against the chunks changes. A layer reads two kinds of things: the hidden state of its own chunk (comes from the layer below, same chunk) and per-layer state written by earlier chunks at the same layer. Nothing reads state of a different layer, except through the hidden state. So the result of (chunk k, layer L) depends on: (chunk k, layer L-1) and (chunks 1..k-1, layer L). Both are done before (k, L) runs in layer-major order. Checked per part:

| Part (qwen4exp) | What chunk k at layer L reads | Why layer-major order keeps it exact | Source |
|---|---|---|---|
| Hidden state (hc-wide residual, `n_embd*4` floats per token) | Output of chunk k at layer L-1 | Held per chunk between layers (section 5) | `qwen4exp.cpp:497-550` |
| Full attention KV (`llama_kv_cache`) | K/V rows of chunks before k at layer L | Layer L of chunk j<k ran before layer L of chunk k; K/V rows are per layer. The mask uses cell positions, so the extra cells of later chunks (all chunks of the group are placed up front) are masked out | `llama-kv-cache.cpp:1092` (`apply_ubatch`), `set_input_kq_mask` |
| Indexer keys and pooled keys (QSA, `llama_memory_hybrid_idx`) | Pooled keys of complete pools of chunks before k, written by `scatter_pooled` at the same layer | Pools are per layer; chunk k only re-pools the pools it touches (`kpool_build_state`), the older ones are read from rows earlier chunks wrote at this layer | `llama-memory-hybrid-idx.cpp:625-713`, `qwen4exp.cpp:816-870` |
| Linear-attention (gated delta net) state and conv state (`llama_memory_recurrent`) | State after chunk k-1 at layer L | One state per layer and sequence, updated in place; same order inside the layer | `llama-memory-recurrent.cpp:514-720` |
| Recurrent state source ids (`src0`, `rs_z`) | Where the first chunk takes its start state from | NOT repeatable: `find_slot` rewrites them, and `s_copy()` resets `rs_idx`. Saved per chunk when the chunk is placed, replayed later | `llama-memory-recurrent.cpp:1361` |
| PLE (per-layer embeddings, `ple_emb`) | Tokens of chunk k, history in the recurrent cache | Built from the chunk's tokens in every segment graph; the history is the recurrent state above | `qwen4exp.cpp:469-475` |
| Positions, token ids, output ids | The chunk's own ubatch | Set from that chunk's ubatch for each graph | `llama-context.cpp:1525-1590` |
| MoE routing and experts | Only the chunk's own tokens | Token-wise; no cross-chunk dependency | `build_layer_ffn` |
| Logits | Last layer of the last chunk (output ids) | Taken from the last-layer pass of each chunk | `llama-context.cpp:2060-2100` |

Refused (stock path is used, a line says why, once): any other architecture; more than one sequence in the batch; embeddings output; SWA caches; sampler offload; `N < 2`; a batch with a single chunk.

Results are the same in meaning, not bit-identical: the attention reads a different `n_kv` window (padding differs only by masked cells) and graphs of other shapes may round differently. This is checked by the laptop logit comparison (section 8).

## 3. Approach: two options, one picked

| | A. Second build path: one graph per layer per chunk | B. Re-enter the scheduler by node range |
|---|---|---|
| How | The qwen4exp graph builder gets a layer range [l0, l1) in `llm_graph_params`. The segment graph reads the hidden state from an input tensor (unless l0 = 0) and exposes it as an output (unless l1 = n_layer). | Build the full graph per chunk as today, compute only the nodes of layer L, keep the graph alive |
| Scheduler | Unchanged: `ggml_backend_sched_alloc_graph` per segment | `ggml_backend_sched` has no "compute nodes a..b of the graph last allocated" call; every sub-range needs its own split plan and the compute buffer holds one graph at a time, so the N live full graphs cannot share it |
| Memory contexts | Need per-chunk state snapshots (section 4), the same in both | Same |
| Cost | Graph build + alloc per (layer, chunk): about 40 x more, 40 x smaller graphs | Same count of sub-computes, plus N live graphs |
| Risk | Touches qwen4exp.cpp (range, inputs that no layer uses), llama-context.cpp (loop), 4 memory-context classes | Needs ggml sched changes: far more risky |

**Picked: A.** It reuses everything (build, alloc, set_inputs, compute, logits path), and the plugin sees ordinary graphs.

## 4. How the graph runs

```
decode(batch):                                   (LLAMA_LAYERWISE_PREFILL=N, qwen4exp, 1 seq, >= 2 chunks)
  mctx = memory->init_batch(...)                 as today: all chunks planned
  while chunks left:
    group = next N chunks                        (a short last group is fine)
    for each chunk k in group:                   in stock order
        mctx->apply()                            places the chunk in the caches
        mctx->lw_save()                          snapshot of what apply() set (n_kv, head, s_copy ids, k-pool layout + state)
        mctx->next()
    for layer L in 0..n_layer-1:
        for each chunk k in group:
            mctx->lw_load(k)                     go back to the chunk's state; no apply()
            n_outputs = outputs of chunk k
            build graph [L, L+1) ; alloc ; set_inputs (reads boundary[k]) ; compute
            boundary[k] <- output of the graph   (skipped after the last layer)
            if L is the last layer: logits / embeddings of chunk k, as in the stock loop
```

- **Segment graph.** `qwen4exp::graph` builds only layers [l0, l1). If l0 > 0 the hidden state (`n_embd*hc` x n_tokens, F32) is an input tensor `llm_graph_input_lw`, filled from the chunk's boundary buffer. If l1 < n_layer the hidden state is `ggml_set_output` and copied to the boundary buffer after the compute. Token embedding and `hc_init` only in the first segment; output cropping, head mixer and logits only in the last.
- **Unused inputs.** A one-layer graph may not use an input the whole model uses (a recurrent layer has no use for the KV index inputs and the reverse). `set_inputs` crashes on an input that was not allocated (comment at `llama-context.cpp:1566`), so the segment builder adds every always-set input (K/V index, `s_copy`, k-pool inputs) to the graph with `ggml_build_forward_expand`, as the MTP graph already does for `s_copy` (`qwen4exp.cpp:625`).
- **A group cannot be bigger than one `llama_decode` call.** `llama_decode` takes at most `-b` tokens, so a group is at most `-b / -ub` chunks. The launcher sets `-b max(ub, 2048)`: at `-ub 1024` that is 2 chunks. A PC run needs `-b 8192` (N = 8 at `-ub 1024`). The server already feeds the prompt in `-b` sized batches, so nothing else changes.
- **No graph reuse.** `can_reuse` stays off in this mode (each segment differs in layer and `n_kv`).
- **Memory-context snapshots.** `apply()` is not repeatable (it removes and re-adds cells, `find_slot` rewrites the state-source ids), so each chunk is placed once, in order, and what the graph needs afterwards is saved: the KV and indexer `n_kv`; the recurrent `head`, `n`, `rs_z` and the `s_copy` values; the k-pool state of the chunk (which pools are new) and a copy of the k-pool layout at that time (so `pool_cells`, the pool mask and the number of pools are exactly the stock ones). `lw_load(k)` puts them back. The default `lw_save/lw_load` of `llama_memory_context_i` refuse, so any other memory type falls back to stock.

## 5. Memory

Boundary tensors are held per chunk of the current group only: `N x ub x n_embd x hc x 4` bytes. Flash-Next has `hc = 4` (hyper-connection count; the residual is four streams wide), so the tensor is four times the 2560-wide one in the brief.

| N (chunks per group) | ub 1024 | ub 2048 |
|---|---|---|
| 4 | 168 MB | 336 MB |
| 8 | 336 MB | 671 MB |
| 16 | 671 MB | 1.34 GB |
| whole 34k prompt at once (not allowed) | 1.39 GB | 1.39 GB |
| whole 128k prompt at once (not allowed) | 5.4 GB | 5.4 GB |

(`n_embd = 2560`, 10240 floats = 40 KiB per token.) The 28 GB host cap is not touched at any sane N. Groups, not the whole prompt, are why. The compute buffer is that of a one-layer graph (smaller than stock). The chunk's own inputs (positions, k-pool inputs, mask) are rebuilt per segment, nothing is kept for all chunks. Cost on the PCIe link: the boundary goes down and up once per (layer, chunk): `2 x 40 KiB x tokens x layers` = about 110 GB at 34k (about 5 s at 20 GB/s). A later variant keeps the boundary buffers in GPU memory and removes it; not in the first version.

## 6. What cuts the upload, and the plugin

The plugin already has the right tool: `MOE_CACHE_GPU_KEEP_MIB=N` (idea P11, `mc_backend.inl:117,771`) keeps the whole expert tensors uploaded for a block on the GPU, least recently used out. It never paid in stock order: between two uses of layer L comes a pass over all the other layers, so the keeper must hold the whole model. In layer-major order the next use of layer L is the next chunk, so a keeper of **one layer** (Flash-Next Q2_0: about 517 MB per block, from 651 GB / 1258 blocks) hits N-1 times out of N. Chunks that use only some experts go the packed path (`pack_weights`), which the keeper skips; at ub 1024 and 256 experts almost every expert is used, so the whole-tensor path is taken (the same case the 651 GB figure comes from).

So the first version needs **no plugin change**: run with `LLAMA_LAYERWISE_PREFILL=8 MOE_CACHE_GPU_KEEP_MIB=1200` (about two layers, so the next layer's tensors can be placed before the old ones go). A small plugin switch to size the keeper by itself is a follow-up once the graph part is measured. GPU_SUM (`mc_backend.inl:670`) works per block of a graph; the segment graph still holds the expert nodes and the weighted sum after them (same layer), so nothing changes. Not measured.

## 7. Arithmetic for the best case (not measured; inputs from the 34k Flash-Next Q2_0 run: 651 GB, 29 s upload, 1258 blocks, 137 s wall)

Upload volume = number of groups x layers with experts x bytes per layer. `layers x bytes` = 1258 blocks x 517 MB / 34 chunks = 19.1 GB per pass over the model.

| Prompt | Chunks (ub 1024) | N | Groups | Upload now | Upload layer-wise | Saved at 22 GB/s (651 GB / 29 s) | Boundary traffic (cost) |
|---|---|---|---|---|---|---|---|
| 34k | 34 | 4 | 9 | 651 GB | 172 GB | 22 s | 5.5 s |
| 34k | 34 | 8 | 5 | 651 GB | 96 GB | 25 s | 5.5 s |
| 34k | 34 | 34 | 1 | 651 GB | 19 GB | 28 s | 5.5 s |
| 128k | 128 | 8 | 16 | about 2.4 TB (extrapolated, same per-chunk volume) | 306 GB | about 95 s | about 21 s |
| 128k | 128 | 16 | 8 | about 2.4 TB | 153 GB | about 100 s | about 21 s |

So at 34k the best case is a saving of 22-28 s of upload for a cost of about 5 s of boundary copies (137 s wall: at most about 16 %), more if the gaps between blocks and the disk reads also fall with fewer expert reads (not measured; the mmap reads of a model bigger than RAM are repeated per chunk today, so the disk saving may be larger than the PCIe one). At 128k the saving is larger in seconds. Not counted: the extra graph builds (1360 segments at 34k instead of 34 graphs, each small; about 3 ms each is 4 s, a guess), and 128k PC memory pressure of the keeper (GPU headroom, the blocks already split at 128k).

## 8. Checks planned on the laptop

1. Tiny qwen4exp with random weights from `test-llama-archs --save-models` (no real qwen4exp model is on the laptop: `work/dl-tiny` has only a dense stories260K, `work/zoo` only a 21 GB qwen3.6 file, other arches). A test program decodes the same 200-400 tokens with `-ub 32`: stock, then `LLAMA_LAYERWISE_PREFILL=2,3,4`, on the CPU backend, and compares last-row logits and every row's logits (NMSE and argmax).
2. Same with CUDA (laptop RTX 4060) if the build is done.
3. A refusal check: another architecture, two sequences.
4. Not on the laptop: Flash-Next with the plugin (the PC test in the final report).

## 9. Result

Laptop (Linux, RTX 4060 8 GB, 16 threads), patch 0050 on the pin, tiny random qwen4exp from `test-llama-archs` (8 layers: layers 3 and 7 full attention, the rest linear attention; QSA indexer with k-pool 4 and top-k 32 so it selects among 75+ pools; 8 experts, 2 used; vocabulary 128). `tests/layerwise_check.sh`: six prompt shapes (300-1000 random tokens; chunk sizes 17, 30, 50, 64, of which 17 and 30 are not multiples of the pool size 4 so pools straddle chunks; logits of every token or of the last one), groups of N chunks, every logit compared with the stock order.

| Backend | N | Result |
|---|---|---|
| CPU | 4 | all six shapes: max\|diff\| 0, same best token on every row (2,600 rows) |
| CPU + moe-cache plugin (cache mode, executor) | 4 | max\|diff\| 0, and equal to the run without the plugin |
| CUDA, all layers on the GPU | 4 and 8 | max\|diff\| 0, same best token everywhere |
| CUDA, 3 of 9 layers on the GPU (scheduler splits) | 4 | max\|diff\| 0 |
| N = 1 | - | refused with one warning, stock result |

A first CUDA run gave NaN and wrong tokens: the copy of the hidden state out of a segment used the buffer-level `ggml_backend_tensor_get`, which does not wait for the GPU backend's compute stream. Fixed with a `ggml_backend_sched_synchronize` before it. The check caught it; the CPU run could not.

Cost of the machinery on the tiny model (per segment graph, 80 segments): build 0.07 ms, scheduler alloc 0.05 ms (CPU) to 0.12 ms (CUDA), set inputs 0.04-0.07 ms, copy of the hidden state 0.03-0.1 ms. So 1400 segments at 34k cost well under a second here. A real layer graph has more nodes than this one and a real model has far larger tensors, so this is a lower bound, not a prediction; the decode log prints the same breakdown for every `llama_decode` (`layer-wise prefill: N segment graphs, build ... ms`), so the PC run measures it.

**What was not checked, and why:**

| Not checked | Why |
|---|---|
| Real Flash-Next (or any real qwen4exp) output | no such model on the laptop; PC test below |
| Upload volume with the plugin's GPU block and `MOE_CACHE_GPU_KEEP_MIB` | the tiny model's expert block has `VIEW` nodes (fused gate-up tensor), which `gpu_segment` refuses (all 64 blocks fell back to the CPU, with and without the switch), so no upload is measured; PC test |
| Speed | tiny model; nothing here predicts the PC |
| 128k memory behaviour | the group holds `N x ub x 40 KiB`; not run at that size |
| Upstream expert cache (`--moe-cache-mib`) with segments | not run |
| Other architectures | refused by design (qwen3.x hybrid, glm5-next have the same k-pool memory and could follow with the same `lw_*` hooks; not started) |

**What blocks a real win, honestly:** nothing found in the graph or memory part: it runs and matches. The open question is only cost against gain on the PC: the keeper must hold the layer (`MOE_CACHE_GPU_KEEP_MIB` of about two layers), `-b` must be at least N x `-ub`, the hidden state crosses PCIe twice per layer and chunk, and `set_input_kpool` now runs once per QSA layer and chunk instead of once per chunk (a fill of `n_pool x n_tokens` values: about 8 M at 34k, 33 M at 128k per call; a guess of a few ms each, so up to a few tens of seconds at 128k; if the log shows it, cache the filled inputs per chunk).
