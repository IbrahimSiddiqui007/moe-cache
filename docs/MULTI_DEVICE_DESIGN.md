# Design: computing experts on every processor (dGPU, iGPU, NPU, CPU)

Status: **design only.** Written 2026-10-14. Nothing here is built or measured; the experiments that decide whether it gets built are E1-E5 below.
Approximate by nature (different processors round differently), so it is judged by a quality check (KL divergence, same top token, answer comparison), not by token identity.

## The idea

A MoE layer picks k experts per token. The experts are independent until their outputs are summed, so they can be computed on different processors at the same time.
Today: the dGPU computes the experts it holds (llama.cpp's GPU expert cache or static layers), the CPU computes everything else. The idea is to also use the iGPU
and the NPU where they exist, and to choose per layer who computes what.

## Two separate decisions

| Decision | Question | Changes how often |
|---|---|---|
| **Placement** | where an expert's weights live: VRAM, RAM, or only on the SSD | slowly; moving weights costs an upload or a read, so only when a move clearly pays over many tokens |
| **Execution** | which processor computes a picked expert in this layer | every layer; free to change, because it moves only the activations (a few KB) |

The iGPU and the NPU share RAM with the CPU: they are execution choices, not storage tiers.

## The decision rule: finish together, not "fastest for each expert"

The processors run in parallel, so what counts is when the **last** one finishes the layer. Sending every expert to the fastest processor leaves the others idle.

For each picked expert e and processor d:
`cost(e, d) = fetch(e, d) + compute(e, d)`, where fetch is 0 if the weights are already where d can read them, a PCIe upload for the dGPU, an SSD read if the expert is not in RAM.
Each processor also has a fixed `launch(d)` per layer (submit and wait).

Per layer: sort the picked experts by their cheapest cost, largest first, and give each to the processor that would **finish it earliest**, counting what that
processor already has in this layer plus its launch cost if it has nothing yet. A processor whose launch cost is larger than the work it would take is not used.
With about 10 experts and 2-4 processors this is a few hundred operations: well under a microsecond.

Costs are learned online: a running average of compute time per (processor, weight type) and of fetch time per (path, size), so the rule adapts to heat,
to other programs and to the model.

Worked example (Flash-Next, one layer, measured costs): 5 missing experts, CPU about 0.12 ms each, dGPU upload about 0.15 ms each.
All uploaded: about 0.75 ms. All on the CPU: about 0.6 ms. Split 3 CPU / 2 dGPU in parallel: about 0.36 ms.

## Limits that decide whether it can win

| Limit | Known |
|---|---|
| Memory bus | CPU, iGPU and NPU share it; CPU + GPU together measured 40 GB/s total (DDR4-3200). It helps only models limited by compute, not by bandwidth: Flash-Next (99 ms compute vs 32 ms reads per token, about 8 of 40 GB/s used) yes, Qwen3.6 K-quants no. |
| Power and heat | The iGPU shares the package power with the CPU; on a laptop at its limit it may take clock speed away from the CPU. |
| Launch cost | A whole layer has about 1.5 ms. A reported (unverified) test on an Arc 140T lost about 3 ms per CPU/iGPU handoff; if that holds, the iGPU is out. |
| Zero-copy | The iGPU only reads the plugin's RAM cache directly if Vulkan maps that memory (host-memory import); otherwise every expert is copied. |
| NPU | llama.cpp's OpenVINO backend (preview) supports FP16/Q8_0/Q4 only: no IQ2/Q2_0. NPUs suit batches (prompts) better than one-token decode. |

## Experiments, in order (each with a stop rule)

| # | Experiment | Machine | Stop if |
|---|---|---|---|
| E1 | iGPU kernel speed for expert-sized IQ2_S / IQ2_XXS / IQ3_XXS / Q2_0 matrix-vector products, and one submit-and-wait | MSI Claw (Arc 140V), then the laptop (UHD) | iGPU under ~20 % of the CPU, or a handoff near 3 ms |
| E2 | CPU kernel speed alone vs with the iGPU busy | Claw, laptop | the CPU loses more than the iGPU adds |
| E3 | Whole expert layers on the iGPU with `-ot` (no code) | Claw | no gain over the same layers on the CPU |
| E4 | The scheduler above inside the plugin (dGPU hits, CPU / iGPU / dGPU-upload for misses); implementation plan: SCHEDULER_DESIGN.md | Claw, laptop | quality check fails, or gain under 10 % |
| E5 | NPU through OpenVINO with a Q4_K MoE, prompts first | Claw | slower than the CPU |

Related: GPU_TIER_DESIGN.md (the dGPU half, frozen as its own project; its hybrid hit/miss part is E4), ENGINE_DECISION.md.
