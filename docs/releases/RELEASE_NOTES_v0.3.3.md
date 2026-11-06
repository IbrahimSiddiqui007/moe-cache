moe-cache v0.3.3 runs MoE models bigger than your RAM through llama.cpp's `llama-server`, with the same answers as plain llama.cpp (token-identical below 128k context with the same layer split; see [docs/EXACTNESS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/docs/EXACTNESS.md)). **v0.3.3 is one download**: the Windows kit and the Linux `-ggml0.26` tarball now include the official, unmodified llama.cpp build, so you no longer fetch llama.cpp separately. Everything from [v0.3.2](https://github.com/IbrahimSiddiqui007/moe-cache/releases/tag/v0.3.2) is in here too.

Please try it and share your numbers: the control page has a **Share my numbers** button (or run `python bin/moe_cache.py bench-report MODEL.gguf`), and the [Benchmark report](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=benchmark_report.yml) issue form takes the result.

## What is new in v0.3.3

| Area | Change |
|---|---|
| One download | The Windows kit carries llama.cpp b11966 (Vulkan) in its `llama\` folder and the Linux `-ggml0.26` tarball carries the b11966 Linux build in `llama/`: official, unmodified, sha256-checked when the release is built, llama.cpp's MIT licence next to it. The launcher, `check`, the picker and the control page find it by themselves; your own `llama-server` (`LLAMA_SERVER`, PATH) still wins. The API-2 Linux tarball (old ggml 0.25) has none |
| Agents | The RAM cache refills itself while an agent pauses for a tool (on by default). PC, recorded OpenCode session with its pauses x4, paired runs: decode 12.6 -> 13.0 and 13.2 -> 13.5 tok/s, the waits for follow-up answers 26.3 -> 26.1 and 32.0 -> 23.9 s. It only changes which experts sit in RAM, so answers cannot change. `MOE_CACHE_IDLE_REBALANCE=0` turns it off |
| Share my numbers | A button on the control page runs the benchmark report for the selected model and shows the text to paste into an issue; nothing is uploaded |
| New models in the picker | gpt-oss-120b MXFP4 (63 GB) and DeepSeek-V4.1-Flash IQ1_M + Engram (231 GB, needs llama.cpp b11870 or newer, chat only for now). Both are marked "untested on this class of machine" until we have measured them |
| Under the hood | One model identity for saved sessions, usage profiles, reports and the page (an old profile for another file is ignored with a message); one shared launch plan behind the command line, the control page, the dry run and on-demand; a CI row for llama.cpp b11966 |

## Downloads

| File | For |
|---|---|
| `moe-cache-v0.3.3-windows-x64.zip` | Windows 10/11 x64. **llama.cpp included** (b11966, Vulkan: AMD, Intel and NVIDIA cards). Unzip, double-click `moe-cache.bat`. Read `README_WINDOWS.txt` first |
| `moe-cache-v0.3.3-ggml0.26-linux-x86_64.tar.gz` | Linux x86-64. **llama.cpp included** (b11966). Also works with your own llama.cpp from commit 23b0202a1 on (ggml 0.26 and newer) |
| `moe-cache-v0.3.3-linux-x86_64.tar.gz` | Linux x86-64 with an older llama.cpp (ggml 0.24 / 0.25, up to commit 7fe450e19); bring your own llama.cpp |

Check a download: `gh attestation verify <file> -R IbrahimSiddiqui007/moe-cache` (`SHA256SUMS` and `SHA256SUMS-windows` only detect corruption).

## Windows: "unknown publisher"

The kit is not code-signed, so Windows may warn. Before you extract the zip: right-click it > **Properties** > tick **Unblock** > **OK**. If SmartScreen says "Windows protected your PC": **More info** > **Run anyway**. Check that your zip is the one CI built: `Get-FileHash moe-cache-v0.3.3-windows-x64.zip` must equal its line in `SHA256SUMS-windows`.

## Known issues

| Issue | Note |
|---|---|
| Bigger downloads | the Windows kit is about 38 MB and the `-ggml0.26` tarball about 36 MB (was 2-3 MB), because llama.cpp is inside |
| Not byte-identical at 128k+ by default | the q8_0 KV cache there passed the quality gate at 4k context; `--exact` keeps f16 |
| Tested hardware | one Windows desktop (Ryzen 9 7950X, RX 9070 XT 16 GB, 32 GB) plus CI runners; NVIDIA and Intel GPUs on Windows untested |
| New picker models | gpt-oss-120b and DeepSeek V4.1 are not measured yet on our PC |

Full list: [CHANGELOG.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/CHANGELOG.md).
