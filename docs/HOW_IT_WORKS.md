# How moe-cache works

Moved here from the README on 2026-10-30, unchanged.


In short, moe-cache manages the CPU-side experts itself:

- keeps the **hot experts in RAM** and reads the rest from your SSD with large, parallel direct reads;
- **sizes its cache from your real memory limit** and stays under it (no out-of-memory kills);
- **learns which experts you use** and warm-starts from that next time (`moe-cache-learn` builds a profile in a minute);
- produces **token-identical output** to stock llama.cpp (checked on every model in [RESULTS](RESULTS.md); the Hunyuan test was short only);
- works with speculative decoding (MTP; it helps when the model or most of it fits: Qwen3.6 at 12 GB 21.7 -> 24.0 tok/s, but it was 17 % slower on Qwen3.8-Flash-Next with an 8 GB card), split GGUF files, MXFP4 / K-quants / IQ4_XS / Q5_0, CPU-only machines and Vulkan GPUs;
- runs long prompts on the GPU like stock llama.cpp does;
- **expert executor** (on by default): while missing experts load from the SSD it already computes the ones in RAM, and on hybrid Intel CPUs it can use the efficiency cores too. Results are bit-identical (every output was compared byte for byte with llama.cpp's own on the test models);
- **faster Q2_0 kernel**: llama.cpp has no SIMD Q2_0 dot product on x86; the plugin supplies one (bit-identical, about 3x faster), which matters for models whose down projections are Q2_0 (for example Qwen3.8-Flash-Next).

The path of one operation:

```
 llama.cpp builds the graph of the model
          |
  ggml scheduler hands each operation to a backend
          |
   +------+------------------+
   |  GPU backend            |  attention, norms, shared layers, the experts you put on the GPU
   |  CPU backend            |  everything else
   |  moe-cache backend  <-----+  the CPU-side expert matrix multiplications:
   +-------------------------+    1. look at which experts the router picked
                                  2. make sure those experts are in RAM (read the missing ones from the SSD, in the background)
                                  3. compute the experts that are ready while the others still load (several threads, optionally efficiency cores)
                                  4. give back the memory of experts nobody needs (LRU)
                                  5. use llama.cpp's own dot-product functions, so every number is identical
```

Each expert keeps its normal address inside one large reserved block of memory; moe-cache decides which experts have real memory behind them.
It plugs into **ggml**, not into llama.cpp's model code, so a model that llama.cpp runs through its expert multiplication should work; the architectures that were actually tested are listed in [COMPATIBILITY](COMPATIBILITY.md) and [RESULTS](RESULTS.md).

## The patched llama.cpp build

Alongside the plugin there is room for a small set of changes to llama.cpp itself (`patches/llama.cpp/`), applied to the exact release the Windows kit bundles. It is optional: the plugin runs on stock llama.cpp, and every change has its own switch that is off until a paired run shows it helps. At the moment the set is empty; the patched build is a manual CI job and is not what the releases ship.
