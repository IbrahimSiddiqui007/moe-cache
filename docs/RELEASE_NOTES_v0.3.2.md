moe-cache v0.3.2 runs MoE models bigger than your RAM through llama.cpp's `llama-server`. By default the answers mean the same as stock llama.cpp's: below 128k context the output is identical to stock with the same layer split (CI); at 128k and above the KV cache is stored as q8_0 (`--fast=kv`), which passed the quality gate at 4k context (`docs/QUALITY_GATE.md`) and `--exact` turns off. **On Windows it is faster than stock** on every model we measured; the **Windows kit** is experimental.

**v0.3.2 is the first release for public testing.** Please try it on your Windows or Linux machine: `python bin/moe_cache.py bench-report MODEL.gguf` (a model bigger than your RAM) measures stock and moe-cache and writes a report to paste into a [Benchmark report issue](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=benchmark_report.yml); the recipe is [docs/SHARE_YOUR_NUMBERS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/docs/SHARE_YOUR_NUMBERS.md). Problems: the Bug report template, with what `doctor` prints.

## What is new in v0.3.2

| Area | Change |
|---|---|
| Long prompts with a browser open | When the GPU also draws the desktop, the desktop and a browser hold VRAM; the planner's GPU margin now grows with what other programs hold, and the long-prompt GPU block runs in parts when VRAM is short. Flash-Next Q2_0 with a browser open: prompts at 34k tokens 109-110 instead of 67-68 tok/s, at 128k 79-95 instead of 67-75 (PC, RESULTS 71) |
| Faster prompts | The long-prompt GPU block reads back into pinned memory inside its loop (N22, +2-5 %) and fills the expert cache while the block uploads (N23, +8-11 %); tokens identical |
| Faster decode | Gate and up of each layer now run as one executor job (one barrier fewer): +0.3-2.6 % decode in every PC pair, same text. Opt-in: `MOE_CACHE_POLICY=classlru`, a frequency-aware cache that helped the miss-heavy 122B (+1.3-2.9 %) |
| Default Flash-Next file | **IQ3_XXS** (3-bit: reads best, the most tasks right, popular for coding) is now the picker's and the docs' default on a 16 GB GPU; Q2_0 stays as the faster choice (about 10 % more tokens per second, e.g. for long thinking). Smaller GPUs keep Q2_0 |
| 128k+ context | The KV cache is q8_0 by default there (`--fast=kv`): prompts about 2x faster at 256k on a 16 GB card (the smaller KV leaves room for a bigger GPU plan); quality gate at 4k context: KL 0.0079 against stock, top-1 96.7 %, 202-task score equal to stock. Not on CPU-only machines or with `--stock`; the control page's Exact mode keeps f16. `--exact` keeps llama.cpp's f16 KV |
| Quality gate | `docs/QUALITY_GATE.md`, `tools/quality_gate.py`: every approximate mode is measured against stock (KL divergence, top-1, the 202 tasks) with thresholds fixed before measuring. New opt-in `--fast=fewer` (decode +2-6 % at 4k, KL 0.034; unproven at 64k-256k) |
| Control page | Readable selected and primary buttons in both themes; "Which model?" is a card with your machine, one recommendation and its numbers; MTP draft-head files are no longer listed as models; "Switch to <name>" in the top bar and a menu on every model (right-click, Shift+F10 or ⋮: Start, Switch, Stop, Plan / Why, Show in folder, Copy path); a "llama.cpp not found" dialog that remembers the path |
| Newer llama.cpp | Fixed a crash at start with llama.cpp builds after b12000 (ggml changed a struct layout without a version bump), and b11966+ (ggml 0.28) now gets the plugin instead of a "DLL not found" message |
| Finding llama-server | One search for the launcher, `check`, the picker and the page: `LLAMA_SERVER` (on Windows also from the registry, where `setx` writes it), the page's saved path, PATH, `C:\llama*`, your user folder, Downloads; the newest build a shipped plugin supports wins, one log line names it |
| Share your numbers | `bench-report` (Windows and Linux): stock vs moe-cache on your machine, a privacy-safe report for a GitHub issue; issue templates for benchmarks and bugs |
| Windows memory crash fixed | with a browser and other programs open, Windows' commit (RAM + page file) could run out while RAM looked free and llama-server stopped. The launcher now reads the commit headroom, sizes the GPU plan and its prompt cache to fit, prints the numbers at start and warns when it is still tight; `doctor` shows it too |
| Coding agents | `connect opencode` sets a 30-minute reply timeout for windows of 64k+ (OpenCode's default is 5 minutes); `bench-report --scenario agent` measures turn-by-turn waits; docs/AGENTS.md says what keeps prompt reuse working |
| Checks you can read | [docs/EXACTNESS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/docs/EXACTNESS.md): what "same tokens as stock" means and every check behind it. `moe_cache.py update --check` says whether a newer moe-cache or llama.cpp is out (downloads nothing) |
| GPU out of memory at load | The launcher measures free VRAM again, plans again and starts once more instead of stopping |
| Model picker | Says when the disk is too small for the recommended file, and finds models you already have as Ollama blobs |
| `doctor` | `check` under a second name; both end with an overview table (system, CPU, RAM, GPU and driver, disk, llama-server, plugin, model) and a "what to fix" list |
| Sleep | The machine stays awake while the server runs (Windows power request, Linux `systemd-inhibit`); `--allow-sleep` turns it off |
| CI | Every model link the picker can recommend is checked against Hugging Face |

## Experimental, off by default

| Option | What it does | Turn on |
|---|---|---|
| Expert pack | `moe_cache.py pack MODEL.gguf` writes a side file with each expert in one block, so a missing expert is one read instead of three. Exact. It helps only when the file-view cache is off (`MOE_CACHE_WIN_VIEW=0` / `MOE_CACHE_LINUX_VIEW=0`, private memory): there it read experts 3x faster on the PC. With the default file views, reads are already as fast and the pack gains nothing | `MOE_CACHE_PACK=1` |
| Idle cache refill | while an agent pauses (a tool running), the plugin refills its RAM cache toward the experts recent turns used | `MOE_CACHE_IDLE_REBALANCE=1` |
| Frequency-aware cache | ClassLRU replacement instead of LRU: helped the miss-heavy 122B (+1.3-2.9 % decode), within noise on Flash-Next | `MOE_CACHE_POLICY=classlru` |

## In progress: diffusion MoE models

`moe-cache-diffusion` (`python bin/moe_cache.py diffusion`) runs llama.cpp's `llama-diffusion-cli` with the plugin, the cache size and the memory cap. LLaDA-MoE-7B passed: at a 3 GB limit each step is 14-16 % faster than stock and the text is identical. RND1 30B and the GPU-block tests come after v0.3.2; until then treat diffusion as a preview. Official Windows llama.cpp zips do not contain `llama-diffusion-cli` (`--cli PATH`).

Everything below is what v0.3.0 and v0.3.1 brought, plus the speed numbers measured on v0.3.2.

## Downloads

| File | For |
|---|---|
| `moe-cache-v0.3.2-windows-x64.zip` | Windows 10/11 x64 with the official llama.cpp release (Vulkan, CPU; tested b11669 and b11966). Unzip, double-click `moe-cache.bat`. Read `README_WINDOWS.txt` first |
| `moe-cache-v0.3.2-linux-x86_64.tar.gz` | Linux x86-64, llama.cpp 7fe450e19 (ggml 0.25) |
| `moe-cache-v0.3.2-ggml0.26-linux-x86_64.tar.gz` | Linux x86-64, llama.cpp 23b0202a1 or newer (ggml 0.26 and newer) |

Check a download: `gh attestation verify <file> -R IbrahimSiddiqui007/moe-cache` (`SHA256SUMS` and `SHA256SUMS-windows` only detect corruption).

## Windows: "unknown publisher"

The kit is not code-signed (a certificate costs money every year; the code is open source and the zip is built by GitHub Actions from the tagged commit), so Windows may warn. Before you extract the zip: right-click it > **Properties** > tick **Unblock** > **OK**. If SmartScreen says "Windows protected your PC": **More info** > **Run anyway**. Check that your zip is the one CI built: `Get-FileHash moe-cache-v0.3.2-windows-x64.zip` in PowerShell must equal its line in `SHA256SUMS-windows` (or `gh attestation verify` as above).

## Windows quick start

| Step | What to do |
|---|---|
| 1 | Unpack llama.cpp's `llama-bNNNNN-bin-win-vulkan-x64.zip` to `C:\llama` |
| 2 | Install Python 3.8 or newer (tick "Add python.exe to PATH") |
| 3 | Unzip the kit and double-click `moe-cache.bat`; pick a model on the page and press Start |
| 4 | Coding agents: start with `--agent`, then `python bin\moe_cache.py connect opencode --write` (or `hermes`). Or `python bin\moe_cache.py on-demand MODEL.gguf`: the agents start the model when they need it |

## Speed on Windows

One desktop: Ryzen 9 7950X, 32 GB DDR5, RX 9070 XT 16 GB, Windows 11, llama.cpp Vulkan (ggml 0.27). Every run cold (Windows' file cache emptied first), stock and moe-cache in the order stock, moe-cache, moe-cache, stock, launcher defaults, a 256-token greedy reply. Details: RESULTS 59-63 in [docs/RESULTS.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/docs/RESULTS.md).

| Model (file size; all bigger than the 32 GB RAM) | stock llama.cpp | moe-cache | |
|---|---|---|---|
| **Qwen3.8-Flash-Next Q2_0 (66 GB), the faster file** | 18.4 tok/s; prompts 121 tok/s | **22.2 tok/s; prompts 281 tok/s** | +20 %; prompts 2.33x |
| Qwen3.8-Flash-Next IQ2_XS (68 GB) | 14.9 | **21.9** | +47 % |
| Qwen3.8-Flash-Next IQ3_XXS (76 GB) | 15.1 | **19.9** | +31 % |
| Qwen3.8-Flash-Next IQ3_S (84 GB) | 12.9 | **17.3** | +34 % |
| Qwen3.5-122B UD-IQ2_XXS (36.6 GB) | 16.0 | **19.7** | +24 % |
| GLM-5.3-Flash UD-IQ2_XXS (102 GB) | 2.2 | **4.4**; **6.9 with `--mtp`** | 2.0x; 3.1x |

The text is byte-identical to stock on every model checked (Flash-Next IQ2_XS and IQ3_S, the 122B, GLM, also with MTP).

**Which Flash-Next file**: **IQ3_XXS** is the default since 0.3.2 (193 of 202 test tasks right, reads best, popular for coding); **Q2_0** when you want speed (about 10 % faster decode, prompts 11-19 % faster than IQ2_XS, 191/202), the **Coder IQ1_M** file for coding agents (an OpenCode session 55.5 s instead of 73.4 s). IQ3_S is not worth it. **GLM-5.3-Flash**: put the model on your fastest drive (+4-13 %) and keep the launcher's default layers; `--mtp` helps chat (+10-23 %) but not code.

**Coding agents** (recorded OpenCode and Hermes Agent sessions replayed against the server, Flash-Next IQ2_XS):

| Session | stock | moe-cache | |
|---|---|---|---|
| OpenCode (5 requests) | 175 s | **73 s** | 2.4x |
| Hermes Agent (7 requests) | 164 s | **82 s** | 2.0x |
| both at once on one server | 336 s | **145 s** | 2.3x |
| OpenCode on the Coder IQ1_M file | | **55.5 s** | |

The gain is in the prompts: the first token comes about 3x sooner, and so does the first token of each follow-up turn. A model that fits in RAM (Qwen3.6-35B) gets 5-9 % shorter sessions; `--mtp` raises its decode from 43.5 to 72 tok/s.

These numbers come from one PC. Your speed depends on RAM, the SSD and the GPU. If the model fits in RAM, the launcher runs stock llama.cpp (that is as fast or faster then).

## What v0.3.0 and v0.3.1 brought

| Area | Change |
|---|---|
| Windows speed | Windows counted the cache's private memory, and Vulkan's GPU memory, against RAM + page file, which held the cache at 8-11 GiB and resized it hundreds of times per request. The cache now keeps experts in **read-only views of the model file**, which cost nothing there; a hard working-set minimum keeps them in RAM. Same cache logic as Linux. `MOE_CACHE_WIN_VIEW=0` goes back to the old way |
| Windows memory | No Windows job memory limit with the file view (it only capped llama.cpp's GPU memory: `--ram 16` failed to load, now it runs at 92 % of the 24 GB speed). Another program that needs 8-14 GB while the server runs gets it in 1-4 s, no freeze |
| Idle RAM | `on-demand --idle-release` (default 5 min): while the agents are quiet the plugin gives the cache's RAM back (PC: 6.8 -> 25 GB free) and the server keeps running; the next request pays ~3 s instead of a ~40 s restart |
| Windows fixes | A program with the plugin could hang at exit and could not be killed (seen with llama-perplexity): fixed. Wrong output and a crash at the second request with models bigger than RAM (Qwen3.5-122B) fixed. The plugin releases the model-file pages llama.cpp's Windows loader leaves in memory (about 13 GB with 12 GPU expert layers). No false "plugin disabled itself" warning for a user without the working-set privilege |
| Launcher | **Starts about 6x sooner**: the model's header is read once and fast (a Flash-Next dry run 17 s -> 2-3 s on the PC; every on-demand model start gained the same). on-demand opens its port at once (was 16.5 s; agents that connect early wait instead of being refused). `--gpu-cache` on Vulkan is capped to what the driver allows (it aborted the load). A micro-batch you choose (`--ub N`) now goes through the tuner, so long prompts keep their GPU block: 1.5-2.3x faster prompts at 34k tokens (the default `--ub auto` was not affected). `--ram` is GB everywhere. AMD and Intel GPUs are found on Windows |
| `check` | Measures the model's drive (4 s of random reads past the file cache) and warns below 1.0 GB/s (SATA, hard disk, USB) |
| Agents | `--agent` (65k context, tool calls, warm sessions after a restart), `connect opencode\|hermes --write`. **`on-demand`**: an always-on listener that starts the model at the first request and stops it after 15 idle minutes, with **several models** (`--add`; the request's model picks one, one runs at a time) and start at login |
| Control page | `moe-cache.bat` (Windows) or `moe-cache-server --ui`: a **three-pane** page: chats, agent sessions and models on the left, its own chat in the middle, live speed, cache, RAM/VRAM, warnings and the log on the right. It says why it runs stock or moe-cache, and the Quick benchmark compares both on your PC in one click |
| Speed options | Flash-Next IQ3_XXS is the recommended file since 0.3.2 (3-bit, 193/202 tasks), Q2_0 the faster one (above). `--mtp` on GLM-5.3-Flash now works with the cache (3.1x stock on chat). `--fast=2` for memory far below the model size; starter usage profiles for Flash-Next, GLM-5.3-Flash, Qwen3.5-122B and Qwen3.6 |
| Executor kernels | The plugin's Q2_0 and IQ3_XXS executor kernels now run on Windows x86-64 too, bit-identical. The Q2_0 one matches llama.cpp's own AVX2 function (#30447) bit for bit (same rule on both systems: at start the plugin checks llama.cpp's Q2_0 function bit for bit and uses the matching kernel, or leaves it alone). No speed change at default settings (bandwidth-bound) |
| Quality | 202 exact-answer tasks (`tools/quality_tasks.json`): Flash-Next IQ3_XXS 193 right, IQ3_S 192, Q2_0 191, Coder IQ1_M 189, IQ2_XS 187 (differences not statistically proven) |
| AMD Vulkan decode | The plugin reads its split's inputs from the GPU asynchronously: decode +12 % (Flash-Next Q2_0, 22.7 -> 25.6 tok/s after a 34k prompt; every ABBA pair, text identical). `MOE_CACHE_ASYNC_IN=0` goes back |
| Long context | 64k-256k: the tuner sizes the KV cache from the model header, and past the measured contexts scales the compute buffer instead of a flat guess; the launcher says when `--fast` would fit more layers. Flash-Next Q2_0: 64k prompt 1.8x stock |
| Memory cap | The whole server stays at or below physical RAM minus max(4 GiB, 12.5 %) on Linux and Windows (`--max-ram GB` changes it); kept by planning, never by stopping a run |
| Linux | The expert cache lives in read-only mappings of the model file too (as on Windows): steady decode +4.7 %, first request +49 %, tokens identical. `MOE_CACHE_LINUX_VIEW=0` goes back |
| Sampling | The launcher passes the model card's sampling settings for Qwen3/3.5/3.6/3.8 and GLM; `--no-sampling-defaults` turns it off |
| Launcher | `--version`; finds `C:\llama\llama-server.exe` (Windows) or install.sh's build (Linux) without `--llama-server` |
| **Model picker** | `moe_cache.py pick` and the control page read your machine (RAM, VRAM, free disk, SSD or not, CPU) and ask what it is for (agents and coding, chat, long documents), then recommend one model, file, context and flags and can use it (`--apply`; `connect` picks it up). It warns but never blocks, its memory estimate includes the chosen context, and it says "untested" where we have no measurement. README: "Which model for my machine?" |
| Agent setup | `connect hermes\|opencode --write` keeps a `.bak` of your own config (also a local Ollama or LM Studio setup) and prints what it replaced; the control page reads back `opencode.jsonc`. Tested end to end on Windows and Linux with Hermes and OpenCode (two-turn coding tasks; follow-up turns reuse the prompt and process only the new tokens) |
| Stock runs | A model that fits in RAM + VRAM (e.g. Nemotron 3.5 Bolt) runs stock llama-server without needing the plugin build, and short requests stay on the CPU when expert layers are there (`GGML_OP_OFFLOAD_MIN_BATCH=512`): a tiny request 1.32 -> 0.6 s, answers identical |
| Fixes from a full code audit | The tuner reserved too little GPU memory for `-ub 2048` at 128k+ contexts (long prompts lost their GPU block); the control page's port scan could send its API key to another local server; the on-demand listener read request bodies before checking the key; Linux idle release did nothing (now 1656 -> 967 MiB); a second Ctrl+C during a prompt could hang the exit; two NVIDIA GPUs were treated as none; the release job's Windows kit build is locked down like the Linux one; about a dozen smaller ones (CHANGELOG) |
| CI | The Windows runner unzips the kit and runs `check --run` and `on-demand` from it, as a user would. Releases are published only after identity and Windows CI passed on the tagged commit |

Full list: [CHANGELOG.md](https://github.com/IbrahimSiddiqui007/moe-cache/blob/main/CHANGELOG.md).

## Known limits

| Limit | Note |
|---|---|
| Windows is experimental | measured on one desktop (Ryzen + RX 9070 XT) and tested on a GitHub Windows runner (CPU); NVIDIA and Intel GPUs on Windows are untested |
| Page file | keep it system-managed or 14 GB or more: the cache costs nothing there now, but Vulkan's GPU memory does |
| Working-set privilege | the hard minimum that keeps experts in RAM needs SeIncreaseWorkingSetPrivilege (a normal desktop login has it); without it the cache still runs at full speed, but Windows may trim it under memory pressure |
| Flash-Next on AMD, driver older than 32.0.32015.7029 | context checkpoints are off (the driver crashes); update Adrenalin |
| Replacing llama.cpp's own CPU functions outside the executor, `--gpu-cache` | Linux only (they need `LD_PRELOAD`); the executor's Q2_0 and IQ3_XXS kernels run on Windows too |
| HIP / ROCm on Windows | loads and gives the same answers, but is about 2.4x slower than Vulkan on the RX 9070 XT: use the Vulkan build |
| Nemotron 3.5 Bolt | needs llama.cpp b11966 or newer; the picker says so when your llama-server is older |

## Known issues

| Issue | Note |
|---|---|
| Unsigned Windows zip | see "unknown publisher" above |
| Not byte-identical at 128k+ by default | the q8_0 KV cache there passed the quality gate at 4k context, but does not give the same bytes as stock; `--exact` keeps f16 |
| Diffusion | preview: LLaDA-MoE-7B tested, RND1 30B not yet |
| Tested hardware | Windows: one desktop plus a CPU-only GitHub runner. Linux: Intel + NVIDIA, CPU-only, Vulkan; a user's RX 9070 XT. macOS no |
| Models measured | Qwen3.8-Flash-Next (5 files), Qwen3.5-122B, GLM-5.3-Flash, Qwen3.6-35B, Nemotron 3.5 Bolt 74B (agents), Granite 1B (CI). Others untested |
| Not tested | winget / scoop install (manifests are in the repo, not submitted); Hermes / OpenCode Desktop on Windows; fresh Ubuntu `install.sh` with apt; Windows 10 |
| Speed numbers | one PC, cold runs, mostly single series |
