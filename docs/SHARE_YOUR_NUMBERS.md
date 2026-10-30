# Share your numbers

One command measures stock llama.cpp and moe-cache on your machine and writes a report you paste into a GitHub issue. It takes 5-20 minutes, only
measures (the report itself downloads, uploads and changes nothing; you need a model file, see "Get a model" in the README), and the report holds no prompts, answers, user name, host name or folder names.

## Windows (10 / 11)

| Step | What to do |
|---|---|
| 1 | **llama.cpp**: included in the kit from v0.3.3 (`llama\` folder, Vulkan). Older kit or another GPU backend: from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), `llama-bNNNNN-bin-win-vulkan-x64.zip` for AMD, Intel and most NVIDIA cards (tested: b11669, and b11966 or newer). AMD: use Vulkan, not HIP (HIP was 2.4x slower on the test PC). NVIDIA: the CUDA zip should work but is untested on Windows. Unzip it to `C:\llama` |
| 2 | **Python 3.8 or newer** (tick "Add python.exe to PATH") and the **moe-cache kit**: right-click the zip > Properties > Unblock, then unzip it |
| 3 | **A model bigger than your RAM.** When the model fits, the launcher runs stock llama.cpp and the report would only say so. `python bin\moe_cache.py pick` suggests one for your machine and prints the download command (`pip install -U huggingface_hub` first); the README has the table. Models are 20-120 GB: keep the size plus about 10 GB of disk free |
| 4 | **Leave the PC alone while it runs**: close the browser and games (they hold GPU memory and slow long prompts), plug in a laptop, no other big program |
| 5 | `python bin\moe_cache.py doctor D:\models\MODEL.gguf` and fix what it lists (AMD: driver 32.0.32015.7029 or newer) |
| 6 | `python bin\moe_cache.py bench-report D:\models\MODEL.gguf --flush-from D:\other-big-files` (no folder of other big files? leave `--flush-from` out; the report then says WARM) |
| 7 | Open a [Benchmark report issue](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=benchmark_report.yml), paste the `moe-cache-report-....md` file, tick the privacy box |

`--flush-from` names a folder with other big files (other models, videos, about as much as your RAM). Reading them pushes the model out of
Windows' file cache, so stock and moe-cache both start cold, as in the README tables. Without it the report says **WARM**: still useful, but not
comparable with the README rows.

## Linux

`./install.sh` (or your own llama.cpp built with `-DBUILD_SHARED_LIBS=ON`), then `python3 bin/moe_cache.py doctor MODEL.gguf` and
`bin/moe-cache-bench-report MODEL.gguf`. The cold start needs nothing extra on Linux (no root, no systemd-run).

## What the report says

| Line | Meaning |
|---|---|
| decode tok/s | answer speed (128 tokens, greedy); the main number |
| prompt tok/s | how fast a ~2000-token prompt is read (agents and long documents depend on it) |
| time to first token | wait before the answer starts |
| cache hit rate | share of experts found in RAM; the rest were read from the drive |
| tokens IDENTICAL / DIFFERENT | whether stock and moe-cache wrote the same answer. DIFFERENT can be normal when the two used different GPU/CPU splits (both plans are in the report); report it anyway |
| COLD / WARM | whether the model was pushed out of the file cache before each run |
| Backend | the GPU backend of your llama.cpp (Vulkan, CUDA, ROCm, CPU only); please correct it in the issue if it says UNKNOWN |

The runs go stock, moe-cache, moe-cache, stock (`--pairs 2` repeats that), so both sides see the same warm-up. Coding agents (OpenCode, Hermes, Claude Code): add `--scenario agent` for a 3-turn conversation with 5 s pauses (a tool running); the report then shows
the wait for the first token of every turn. Options: `--ram GB` (cache size),
`--ctx`, `--tokens`, `--prompt-tokens`, `--fast` (measures an approximate mode; identity is then n/a), `--force-plugin` (measure a model that fits).

Answers look wrong, loop, or the speed is far from the README? See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) before you open an issue.
