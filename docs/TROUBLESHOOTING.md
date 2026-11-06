# Troubleshooting

Commands are given for both systems. On Linux use `python3 bin/moe_cache.py ...`; on Windows use `python bin\moe_cache.py ...`. Everything below behaves the same on both.

Still stuck? Open a [Bug report](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=bug_report.yml) and paste the output of `doctor`.

## Answers look wrong, misspelled or loop

moe-cache is exact by default: with the same layer split its output is token-identical to stock llama.cpp (`tests/identity.sh`, CI on Linux and Windows). With long
prompts, how much work runs on the GPU can change from run to run with the free VRAM, which can change the last digits and then a token (stock llama.cpp too).
So when answers look wrong, first find out
whether stock llama.cpp does the same (last row of the table). Go down the table in this order; each row is a cheap check.

| # | Check | Why it matters | How |
|---|---|---|---|
| 1 | **Does stock llama-server do the same?** | If stock gives the same text, the cause is the model file, the settings or the build, not moe-cache | Run the same prompt, same flags, with plain `llama-server` (no plugin). Use temperature 0 and a fixed seed (`--seed 42`) |
| 2 | **Temperature-0 twice** | The same prompt at temperature 0 must give identical text twice. If it does not, suspect unstable hardware (a RAM overclock such as XMP / EXPO at its limit) or a broken build. This is general knowledge, not something we measured | Send the same request twice with `"temperature": 0, "seed": 42`, compare the two answers |
| 3 | **Sampling settings** | Low-bit quants (IQ2, IQ3, Q2) tend to loop on long answers without the model card's settings (temperature, top-p, top-k, min-p, presence penalty). The launcher passes them for Qwen3 / 3.5 / 3.6 / 3.8 and GLM models; plain `llama-server` does not, and your client may send its own values (they win) | The launcher prints the settings it passes. `--no-sampling-defaults` turns them off. Set the card's values in your client if you start `llama-server` yourself |
| 4 | **KV cache type** | q8_0 KV is approximate (passed the quality gate at 4k context: KL 0.0079, top-1 96.7 %, tasks = stock, Flash-Next Q2_0). At `--ctx 131072` and above it is the default. q4_0 KV is not tested by us | Look at the launcher's start lines or your `-ctk` / `-ctv` flags. `--exact` keeps llama.cpp's f16 KV (and costs VRAM and long-prompt speed) |
| 5 | **An approximate mode** | `--fast`, `--fast=2`, `--fast=fewer` and `--hybrid` change the output on purpose and only when you ask. `fewer` is unproven at 64k-256k ([QUALITY_GATE.md](QUALITY_GATE.md)) | Start without them (`--exact` also forces exact output) and compare |
| 6 | **The model file** | A broken or truncated download, or a different upload of the same quant, gives odd text | Compare the sha256 of your file with the one on the model's Hugging Face page ("Files" > the file > "Copy SHA256"): Linux `sha256sum MODEL.gguf`, Windows `certutil -hashfile MODEL.gguf SHA256` (PowerShell: `Get-FileHash MODEL.gguf`). Download again if it differs |
| 7 | **llama.cpp build, backend, driver** | An old build may lack the model's fixes; CUDA, Vulkan and HIP are different code. AMD on Windows: a Vulkan driver older than 32.0.32015.7029 crashes on Flash-Next checkpoints | `doctor` lists build, backend and driver and says what to fix. Update llama.cpp (tested: b11669, and b11966 or newer) and the GPU driver |
| 8 | **Context too small for the conversation** | When the chat grows past `--ctx`, llama.cpp drops or shifts old text and the model can lose the thread or repeat itself | Raise `--ctx`, or start a new chat. Long contexts cost VRAM: see the README table for tested sizes |

### Check whether the model loops: `tools/loop_check.py`

`loop_check.py` sends 20 long-answer prompts to a running llama-server and counts repeated word sequences. It needs only Python, no extra packages.

1. Start the server as you normally do (`bin/moe-cache-server MODEL.gguf` on Linux, `python bin\moe_cache.py server MODEL.gguf` on Windows).
2. In a second terminal:

| System | Command |
|---|---|
| Linux | `python3 tools/loop_check.py --url http://127.0.0.1:8080 --label mine --json loop_mine.json` |
| Windows | `python tools\loop_check.py --url http://127.0.0.1:8080 --label mine --json loop_mine.json` |

It sends the prompts **without** sampling fields, so the server's own defaults are what gets measured. To compare, run it again against a server started with
`--no-sampling-defaults` (same seed, `--seed` on the script defaults to 1). It takes a while: 20 answers of up to 1500 tokens each.

What the output means:

| Output | Meaning |
|---|---|
| repeat share | share of 8-word sequences that already appeared earlier in the same answer. Normal text is low; a high share means repetition |
| `LOOP` | the answer ends in a repeat (more than half of the last 200 words repeat) or it hit the token limit while repeating |
| summary line | loops / answers, plus median and max repeat share |

Thinking text is counted with the answer, because loops happen there too. A few `LOOP` answers with the sampling defaults off and none with them on means sampling was the cause (row 3). `LOOP` in
both, and also on stock llama-server, points at the file or the quant (rows 6 and 1). `LOOP` only with moe-cache: please open a bug report and attach the JSON file.

Our own task runs (202 exact-answer tasks) answer in 128 tokens or less, which is too short to catch long-answer loops; that is why this script exists.

## Slow

| Symptom | Cause | What to do |
|---|---|---|
| Not faster than stock, or the launcher says it started stock | The model fits in your RAM, so stock llama.cpp already runs from the file cache with the same kernels. moe-cache helps when memory is short or the model is larger than RAM | Use stock llama.cpp (README, "When to use something else"). |
| Answers slow down sharply, or very slow prompts, while a browser or game is open | They hold GPU memory, so the card is too full | Close them, or add `--gpu-margin 2` (GiB of VRAM left free; the default is about 0.5) |
| Speed drops with the EA app open (or another game launcher or program that keeps drawing on the GPU) | While such an app draws, it keeps the GPU's graphics engine busy and our work waits behind it. It costs both prompts and decode, even with free VRAM. An EA app sitting idle cost nothing in our test | Close it while the model runs; if it stays in the system tray, quit it from there. The launcher warns when it sees the EA app, and the control page's **Other programs** block lists what is using the GPU right now |
| The PC stutters or feels frozen while a model runs | By default the server takes almost all RAM (total minus 4 GiB), the GPU and every core: your own programs are pushed out | Tick **Leave room for me** on the control page or add `--leave-room` (8 GiB or 25 % of RAM, 2 GB of GPU memory and 2 cores stay free, lower priority; the model gets slower). If the CPU has its own graphics, plug the monitor into the motherboard |
| Misses (experts read from the drive) are slow | The model sits on a SATA SSD, hard disk or USB drive | Run `doctor`: its "drive speed" line times random reads and warns when the drive is slow. Move the model to an NVMe drive |
| Very long prompts take minutes | Reading a 63k-token prompt takes about 2.5 minutes on our Windows PC with v0.3.5 (about 440 tok/s; it took about 290 s in October, RESULTS 64, §78), and a 222k-token prompt at 256k context about 11 minutes (334 tok/s, 2026-11-03). A long prompt is a wait even at that speed | Keep the context as short as the task allows. Closing the browser helps long prompts too (34k: 109-110 vs 67-68 tok/s with a browser open, RESULTS 71) |

## Does not start

1. Run `doctor` on your model: Linux `python3 bin/moe_cache.py doctor MODEL.gguf`, Windows `python bin\moe_cache.py doctor D:\models\MODEL.gguf`. Every line says OK, WARN or FAIL and what to do.
2. Fix the FAIL lines first (typical ones: a llama.cpp build without shared ggml libraries, an AMD driver that is too old, not enough free VRAM or RAM).
3. Still failing: open a [Bug report](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=bug_report.yml) and paste the doctor output and the last lines of the server log.

## Crashes on Windows with other programs open

Windows promises memory to programs out of its **commit** budget (RAM + page file, "Committed" in Task Manager > Performance > Memory). When that budget is
used up, new allocations fail even though RAM still shows as free, and llama-server can stop mid-request. The launcher prints the budget at start
(`moe-cache: Windows memory: RAM ... GB free; commit ... GB free of ... GB`), shrinks its own prompt cache to fit, and warns when it is still too tight.

| Fix | How |
|---|---|
| Close big programs | browsers with many tabs, games, other model servers |
| Let the page file grow | Windows: System > About > Advanced system settings > Performance > Advanced > Virtual memory: "Automatically manage paging file size" (or a size of at least 16 GB) |
| Use less GPU memory | `--gpu-margin 2` (Vulkan's GPU memory is charged to commit too) |

## Vulkan "out of memory" with GPU memory free (AMD on Windows)

| Symptom | Try |
|---|---|
| `vkQueueSubmit` / `ErrorOutOfDeviceMemory` although the GPU shows several GB free (reported on AMD Windows PCs without Resizable BAR, llama.cpp #30888) | turn on Resizable BAR / "Smart Access Memory" in the BIOS and Adrenalin, or start once with `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` (Windows: `set GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`, then start again) and compare |
| the GPU also drives your screen and answers slow down | `--gpu-margin 2` (GiB kept free) |

## A thinking model writes for a very long time

Qwen3 / GLM thinking models can think for thousands of tokens before they answer. llama-server can cap it: add `-- --reasoning-budget N` (tokens; llama.cpp
#30333, #31093), or use the model's no-think mode for coding agents. This changes how long the model thinks, not the plugin.


## Known problems from llama.cpp and the platforms (one line each)

The causes below come from the llama.cpp issue pages and from our own code; the issue numbers are llama.cpp's. We have not hit all of them ourselves.

| Problem | Cause | What to do |
|---|---|---|
| Very slow reads under WSL2 or a cloud VM (17-35x slower in llama.cpp discussions #30333 and #30409; OLMoE test 0.19 vs 3.2-3.4 tok/s) | A disk read-ahead of `read_ahead_kb=8192` pulls in 8 MiB for every small read | Lower the read-ahead of the disk the model is on (`echo 128 \| sudo tee /sys/block/sdX/queue/read_ahead_kb`, `sdX` = that disk). The report is about llama.cpp's file-mapped loading; moe-cache reads missed experts itself with large direct reads |
| Memory runs out, or the process is killed, while a CPU-only model loads, especially under a memory limit | With no GPU in use, llama.cpp repacks the weights into private memory: a full copy that cannot be streamed from the file. moe-cache does what stock does here (`MOE_CACHE_REPACK=auto`) and does not read repacked experts directly (`src/mc_state_profile.inl`) | Add `-- --no-repack` to the server arguments, or set `MOE_CACHE_REPACK=off` ([CONFIGURATION.md](CONFIGURATION.md)) |
| llama.cpp's own GPU cache option aborts at load on some AMD cards (RX 6900 XT on Windows, Radeon Pro VII; llama.cpp #30697; fixes #31114 and #31139 were still open on 2026-11-01) | Its cache bank is larger than the card's single-buffer limit and the card lacks 64-bit indexing | Do not use `--moe-cache-mib` on these cards (next section) |
| The first load is slow on an APU or handheld (for example MSI Claw) when llama.cpp loads without mmap (`--load-mode none`; older builds `--no-mmap`) | The whole model file is read once at load, and one full expert tensor sits in a temporary buffer (`src/mc_buffer.inl`) | Expect the longer first load; keep enough free RAM for one expert tensor on top of the cache |
| The server grows or gets slower after hours | Upstream reports memory growth in llama-server with speculative decoding (#31392: draft-mtp scheduler rebuilt per request). moe-cache's own cache is capped (`MOE_CACHE_SIZE_GIB`); llama-server's own growth is not | Restart the server when its memory keeps growing; try without `--mtp` to see whether it stops |
| Empty or missing final answers from a thinking model | Thinking models can spend the whole reply budget on thinking (see the section below) | Raise the reply limit in the client, or cap the thinking with `-- --reasoning-budget N` |

## Do not combine llama.cpp's `--moe-cache-mib` with moe-cache's cache or with a draft model

`--moe-cache-mib` is llama.cpp's own GPU expert cache, not ours.

| Case | What happens | Do |
|---|---|---|
| the plugin plus `--moe-cache-mib` (without `--gpu-cache`) | measured 29.6 tok/s against 38.8 without it (Qwen3-Coder-30B, laptop, [RESULTS_OLD_LAPTOP.md](results-archive/RESULTS_OLD_LAPTOP.md), table "plugin + `--moe-cache-mib 5000`"): the plugin's buffer is not a host buffer, so the upstream cache finds no layer to cache | Leave it off. For a GPU cache on top of ours use `--gpu-cache` (Linux, preload method only) |
| `--moe-cache-mib` on AMD cards under Vulkan | aborts at load on some cards (table above) | Leave it off on AMD |
| a draft model (`-md` / `--model-draft`) together with it | reported broken upstream (llama.cpp #31497); we have not tested a draft model with that cache | Use only one of the two |
