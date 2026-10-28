# Changelog

## 0.0.0.TBD (2026-10-28)

### Agents
- **Warm-session saves scale with the file** (idea N27): a save rewrites the whole slot (~1.7 GB at 64k, ~7 GB at 256k on Flash-Next) and llama-server answers nothing while it writes. The saver now waits 20 s + 20 s per GB of idle time, and a conversation that grew only a little since its last save waits 3 min + 30 s per GB (at most 10 min); the older file restores exactly (llama-server processes only the newer tokens). tests/warm_test.py (fake server) in both CIs.

### Docs
- **Flash-Next file recommendation: GSQ-RCO Q2_0 is the default** (66.4 GB; PC: 22.2 tok/s, prompts 281 / 287 tok/s at 3k / 34k tokens, +5 % decode and +11-19 % prompts vs IQ2_XS, 191/202 tasks); IQ3_XXS when accuracy matters (193/202); ISTA GSQ-RCO Coder IQ1_M for coding agents (OpenCode session 55.5 vs 73.4 s); IQ2_XS superseded; IQ3_S not worth it; quality differences not proven (RESULTS 63). README, docs/WINDOWS.md, docs/AGENTS.md, benchmark card.

### Windows and AMD
- **Windows: the expert cache lives in read-only views of the model file** (idea I47). Its memory was private and Windows charges private
  memory to commit (RAM + page file), as it does Vulkan's GPU allocations: on a 32 GB PC with a 16 GB card the cache was held to 8-11 GiB and
  resized hundreds of times per request, while stock llama.cpp's mmap used file pages that cost no commit. File-view pages cost no commit; a
  hard working-set minimum keeps the cached experts in RAM, and an evicted expert stays in the file cache. Which experts are cached, prefetch and
  the executor are unchanged (same as Linux). PC cold ABBA vs stock, launcher defaults: Flash-Next IQ3_XXS **19.9 vs 15.1 tok/s**, IQ2_XS
  **21.9 vs 14.9** (RESULTS 59). `MOE_CACHE_WIN_VIEW=0` goes back to private memory.
- **Windows kit**: `moe-cache-vX-windows-x64.zip` with both plugin DLLs (API 2 / API 3), the launcher, the control page, a one-click `moe-cache.bat`
  (`--shortcut` adds a Start-menu entry) and `README_WINDOWS.txt`. Built by `tools/make_windows_kit.sh` from a clean export; CI unzips it outside the
  repository and runs `check --run` from it; the release workflow attests it and attaches it to the release.
- **Fixed: wrong output and a crash at the second request with models bigger than RAM on Windows** (PC, Qwen3.5-122B IQ2 from a fast NVMe
  drive). An expert read straight into place covered the memory pages it shares with its neighbours; on Windows a page under such a read briefly
  holds other data, so the executor sometimes read garbage in a neighbour's first or last rows: wrong text ("3333..."), NaN router scores, and a
  GPU top-k that then listed one expert several times per token, which overflowed an executor task and crashed a pool worker. Reads now write only
  the expert's own bytes, an expert listed more than once can no longer overflow a task (and every task is checked before it runs), and
  `MOE_CACHE_EXEC_CHECK=1` reports where a mismatch comes from. Confirmed on the PC: text identical to stock, no crash, 0 executor mismatches.
- **Windows memory commit**: when Windows refuses memory (RAM + page file full, error 1455) the cache gives experts back and shrinks instead of
  reading into pages that do not exist ("read failed ... error 998"); a range past the end of its reservation (error 487) is clamped instead of
  retried page by page; the start budget counts what llama-server really uses and keeps max(2 GiB, 8 % of the commit limit) of system-wide commit
  free (GPU memory on Windows counts there too: GLM-5.3 + MTP started at 4.3 instead of 18.9 GiB and no longer starves the whole PC). The crash
  report names the executor's op, row and output, and no longer mistakes a small address for an expert.
- CI and release: checks that could not fail now can (the T7 grep, the DLL import list, "executor ran" in every executor-forced step); the release
  workflow runs with pipefail and publishes a tag only when identity and windows passed on that commit.
- New CI steps: repeated expert ids (`tests/dup_ids.sh`, `MOE_CACHE_TEST_DUP_IDS`) and reads poisoned while in flight
  (`MOE_CACHE_TEST_POISON_READS`), each with a mutant that must fail.
- **Tested on a Windows 11 desktop** (Ryzen 9 7950X + RX 9070 XT 16 GB, official llama.cpp b11669 Vulkan, driver 32.0.23034.4; RESULTS 54-55): identity 4/4 on
  Granite, Qwen3.6 and Qwen3.5-122B (bigger than RAM); executor byte check 0 mismatches; Flash-Next through the launcher and a chat. Steps:
  [docs/WINDOWS.md](docs/WINDOWS.md); status in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md). Windows stays **experimental**.
- Launcher finds the GPU on Windows with AMD and Intel cards (the planner knew only nvidia-smi and Linux sysfs, so it saw no GPU and kept every expert on
  the CPU): it reads `llama-server --list-devices` and picks the adapter with the most dedicated VRAM in the registry, not the first listed (the iGPU).
- **Windows: the plugin releases the model-file pages llama.cpp leaves in llama-server's working set** (I37). llama.cpp's Windows loader never unmaps the
  file pages of the GPU tensors, so ~13 GB stayed counted as in use with 12 GPU expert layers and the cache shrank. It looked like AMD's driver taking RAM for
  VRAM; it is not. The log says "released N GiB of model-file pages from the working set"; the cache measures the process by its private working set.
  GPU expert layers are planned as on other GPUs (an earlier "all experts on the CPU" rule for AMD is gone: Flash-Next cold decode 12.9-14.9 tok/s all on the
  CPU vs 14.9-15.0 with 12 GPU expert layers, no cache shrink).
- **A small page file limits the cache on Windows** (every allocation needs commit): 6.9 GB page file and 16 GB committed by other programs, GLM's cache
  16.1 -> 9.6 GiB; 14 GB page file, 14.9 GiB and 1.52 -> 2.02 tok/s. The plugin warns near the commit limit.
- **`--ram` is GB everywhere**: it reached the plugin and the planner as GiB, 7 % over the Windows job limit the launcher sets (`--ram 20`: commit failed,
  Windows error 1455, the server died).
- Windows: llama-server's exit code 1 (e.g. a model that fails to load) is reported as a failure, not as a quiet stop, in the launcher and the page.
- **Flash-Next (`qwen4exp`) on an AMD Vulkan GPU under Windows: `--ctx-checkpoints 0`.** Saving a context checkpoint crashes `amdvlk64.dll` (stock llama-server
  too); the launcher turns checkpoints off with a message that names the driver version. `learn` does the same. Price: a conversation cannot jump back to an earlier point.
- **Fixed: Windows crash in the long-prompt GPU block** (GLM-5.3-Flash, 102 GB, desktop): the block uploaded whole expert tensors including evicted
  (decommitted) experts. Evicted experts are never uploaded now; confirmed on the PC (RESULTS 54). The launcher warns when the cache shrinks below half.
- Windows robustness: UTF-8 (non-ASCII) paths, `--cache-ram` from the command line, efficiency-core lookahead workers with the AVX2 router dot, processor-group note.
  Profile saves work on Windows; `replace_file` reports errors; an iGPU-only machine reports its VRAM.
- `check` on Windows reports the pinning result and knows llama.cpp b11669; `build.sh --windows` honors `BUILD_DIR` (an API 2 and an API 3 DLL side by side) and the
  launcher finds the DLL that matches the server's ggml, also in `build\win*` and the kit's `ggml-moe-cache-api2.dll` / `-api3.dll`; `learn` picks the library the way `server` does.
- The start-up line "tested with ggml X only" lists every tested version of the build's API (API 3: 0.26.0 and 0.27.0, API 2: 0.25.1); it warned wrongly on the
  Windows release b11669 (ggml 0.27.0). CI fails if the warning shows on a pinned version.
- `--gpu-cache` on Vulkan is capped so llama.cpp's GPU expert cache loads: llama.cpp keeps each expert tensor's cache slots in one bank tensor, and
  Vulkan refuses a tensor over the device's limit (maxBufferSize / maxStorageBufferRange), which aborted the load ("pre-allocated tensor (moe_cache.0...)",
  PC: `auto` planned 8909 MiB, the limit allowed about 7700). The launcher reads the limit from the Vulkan driver (Linux and Windows) and works out
  the largest size from the model's expert layout (planner/gguf_plan.py gpu_cache_cap_mib).
- **Q2_0 on current llama.cpp (b11669, master), Linux and Windows: an exact faster kernel** (idea N20). llama.cpp now has its own AVX2 Q2_0 dot
  (#30447), so the plugin's old kernels (exact to the plain C loop of older pins) stepped aside. The executor now has a version with the same
  integer lanes and multiply-add order as #30447 that takes the activation's lane sums from a column prepared once instead of in every row's dot
  (1.25x per dot, bit-identical). At start the plugin compares the CPU library's Q2_0 function bit for bit with the plain loop and with #30447 and
  uses the matching kernel, or leaves llama.cpp's alone. The executor kernels were fenced to Linux with the LD_PRELOAD override (which stays
  Linux-only); they now run on Windows too. Windows CI checks the kernels as built for Windows (tests/kernel_check.cpp).
- The launcher warns when `--n-cpu-moe N` puts more expert layers on the GPU than planned while MTP is on: the draft context needs GPU memory
  too and llama-server aborted at load on GLM (RESULTS 62). The control page's measured speeds have Flash-Next Q2_0 and Coder IQ1_M, and the
  122B row is RESULTS 61's re-measure (+24 %); a file gets the row of the closest size.
- Long contexts (64k-256k): the tuner sizes the KV cache from the model header (attention layers, KV heads, key/value lengths, the indexer key, the
  recurrent state per slot) instead of a flat 40 KB per token, which over-reserved 3-5 GiB of VRAM at 256k (idea N26). Flash-Next at 64k: 2.13 GiB,
  as measured. At 256k on a 16 GB card Flash-Next keeps 2 more expert layers on the GPU, the 122B 3 more. Layouts it does not model (MLA, GLM)
  keep the flat estimate.
- The executor hands its pool 128 weight rows per chunk instead of 32: decode +3.9 % on the PC (Flash-Next Q2_0, every ABBA pair, text identical; idea N25).
  `MOE_CACHE_EXEC_CHUNK=N` sets it.
- At 128k+ context the launcher says in one line when `--fast` (q8_0 KV cache) would put 2+ more expert layers on the GPU, or let the context fit at
  all, and by how much; exact mode stays exact (f16 KV, identical to stock; owner's decision, idea N28).
- `MOE_CACHE_STATS` reports `pool_parks` (executor workers that fell asleep); `MOE_CACHE_POOL_SPIN_US=N` keeps them spinning N us after their last job (test knob for idea N30; default unchanged until the PC ABBA).
- Tuner at 128k-256k: llama-server's measured GPU buffers on the PC (pc-bench): the indexer cache is 2 x key_length per token (KV estimate now
  within 1 MiB of llama-server at 256k: 6144 + 1536 + 450 MiB), compute buffers at 256k (512: 1403, 1024: 2983 MiB). -ub 2048 is not planned
  at 128k+ (slower than 1024 on long prompts). At 256k on 16 GB, -ub 1024 does not fit next to the long-prompt block's reserve, so 512 stays
  until a PC ABBA compares the two (N29).
- **Linux: the expert cache lives in read-only mappings of the model file too** (as on Windows): steady decode +4.7 % in every pair, the first
  request +49 % (page-cache read-ahead loads faster than the private O_DIRECT reads), tokens identical (RESULTS 66). `MOE_CACHE_LINUX_VIEW=0` goes back.
- **Hard memory cap, Linux and Windows alike**: the whole server (cache + llama.cpp's host buffers, KV, compute) stays at or below physical RAM
  minus max(4 GiB, 12.5 %) = 28 GiB on a 32 GiB machine, on top of the free-memory logic (the lower wins), so Windows does not run out of
  memory and Linux does not OOM. `--max-ram GB` / `MOE_CACHE_MAX_RAM_GB` change it (0 = off); `--ram` cannot go above it; the cap is printed
  at start (launcher and plugin). The Windows working-set minimum follows the capped cache budget. The cap is kept by planning only, never by
  stopping a running server.
- `tools/win_flush_cache.ps1`: empties Windows' file cache before a timed run by reading other big files (no admin, read-only), for fair stock-vs-plugin timings.
- `tests/kernel_variants.cpp` builds and runs on Windows; skips AVX-VNNI variants on CPUs without them (Zen 4 crashed with an illegal instruction).

### Agent mode, Hermes Agent and OpenCode
- on-demand: an earlier idle stage, `--idle-release MIN` (default 5, 0 = off): the plugin gives the expert cache's RAM back while the server idles and the server keeps running, so the next request needs no restart (N3; PC: available RAM 6.8 -> 25 GB, next request +0.5 s instead of a 34 s start). Windows: the hard working-set minimum is dropped and the file-view pages move to standby; Linux: the experts are evicted. The plugin listens for SIGUSR2 / the event `Local\moe-cache-idle-<pid>`.
- **`moe_cache.py on-demand`: agents work without starting a server by hand** (as with Ollama). A small always-on listener holds the agent port; the
  first request starts the launcher in agent mode for the remembered model, requests (streaming included) pass through, and the model stops after 15 idle
  minutes (`--idle`). The API key and the localhost-only CORS rule are checked in front, so a web page cannot wake the model; `/v1/models` and `/props`
  answer without starting it. `--install-autostart` starts it at login (systemd --user unit; Windows Startup-folder shortcut, no admin). `connect
  hermes|opencode` now defaults to the on-demand port. [docs/AGENTS.md](docs/AGENTS.md#without-starting-a-server-by-hand-on-demand)
- on-demand opens its port at once: it bound the port only after the launcher's dry run had checked the options (16.5 s on the Windows PC), so an
  agent that connected meanwhile was refused. Now that agent waits and is answered when the check is done. The free-port check before it took 1 s
  on Windows (a refused local connect is retried until the timeout); its timeout is now 0.25 s.
- The launcher reads a model's GGUF header faster and once per launch: skipping the tokenizer's string array asked the file size once per
  entry (~150k system calls), and the header was read 4+ times per start. On the Windows PC this was most of the 16-21 s a dry run took (and every
  on-demand model start pays it again).
- **`--agent`** (RESULTS 51): context 65536, `--jinja` tool calls, stable `--alias`, at most 8 context checkpoints per slot with a 1 GiB reserve, config blocks printed
  when ready; `--api-key` for other machines. **`connect hermes|opencode [--write]`** writes the agent's config (Hermes through `hermes config set`, OpenCode by merging
  `opencode.json`, old file kept as `.bak`). OpenCode 1.19.4 and Hermes Agent v0.22.5 fixed files through moe-cache on Linux ([docs/AGENTS.md](docs/AGENTS.md),
  including their GUIs on Linux and Windows).
- **Warm agent sessions**: every idle conversation of 2k+ tokens is saved with its checkpoints, one file per slot, and the two longest are restored at the next start
  (first turn after a restart 9.0 s / 6.2 s instead of ~45 s / 23 s; `--no-warm` turns it off). RESULTS 51-52.
- **Long prompts upload only the experts they use** (I4, `MOE_CACHE_GPU_PACK=0` turns it off; exact): ~70-token prompts 1.62 -> 0.90 s, ~170 tokens 1.74 -> 1.12 s,
  which is what agent follow-up turns look like. RESULTS 52.
- Tuner: the long-prompt block reserve includes the plugin's 256 MiB headroom; at `--ctx 8192` Flash-Next prompts no longer fall back to the CPU (prefill ~20 -> 93-135 tok/s). RESULTS 52.
- Agent benchmark numbers (laptop, Qwen3.6, 16 GB): OpenCode task 154-183 s stock vs 97-114 s with moe-cache; Hermes 476-646 s stock vs 116 s. (Preliminary; I30.)

### Control page (GUI)
- One stock-or-plugin rule: `moe_cache.decide()` is used by the launcher and the page, so the page's verdict cannot differ from what the launcher
  runs. New launcher option `--stock` (stock llama-server whatever the model's size, to compare). On the page: **Run as** Auto / moe-cache / Stock,
  and a **Stock vs moe-cache** table from the Quick benchmark runs (newest run of each, same model and settings, with the speed-up).
- Control page: **MTP** Auto / On / Off (the launcher's `--mtp` / `--no-mtp`); the plan's Why list gives the launcher's own MTP line
  (`mtp_decide()`), Status shows "+ MTP"; MTP benchmark runs are compared only with MTP runs. docs/GUI.md has screenshots.
- `moe-cache-server --ui` or `python bin\moe_cache.py ui`: a local page at http://127.0.0.1:8070 on Linux and Windows. Models with the planner's verdict and reasons, start and stop,
  live tok/s, hit rate, SSD reads, RAM and VRAM (also for AMD/Intel on Windows), the log, llama-server's chat page; Connect OpenCode / Hermes buttons with a preview;
  warnings read from the log; measured speeds from RESULTS next to the verdict; warm-session state. `bin\moe-cache-ui.bat` starts it with one click on Windows. See [docs/GUI.md](docs/GUI.md).
  The VRAM setting defaults to auto (it used the free VRAM of the moment the page loaded).

### `--fast`, profiles and speed
- **`--fast` also uses a q8_0 KV cache** (KL 0.0045 alone); in agent mode Flash-Next gets `-ub 1024` (prefill ~50 -> ~78 tok/s).
- **`--fast=2`**: a stronger opt-in level for memory far below the model (`MOE_CACHE_ROUTE_DROP` 4): Flash-Next at 12 GB 13.7 vs 9.6 tok/s, KL 0.022, same answers on 6
  questions; no gain at 24 GB. RESULTS 52-53.
- **Starter usage profiles** (`profiles/`, `--no-starter` to skip) for Flash-Next, GLM-5.3-Flash, Qwen3.5-122B and Qwen3.6: first request 5.96/6.18 -> 13.70/11.81 tok/s on Flash-Next.
- Long prompts keep part of the cache (I3, below): Flash-Next prefill +27 %.
- Windows, clean run (RESULTS 55, Flash-Next, all experts on the CPU, file cache flushed before every run): plugin decode ~12.2 tok/s steady vs stock 12.1-16.1, prefill 2.2x stock (120 vs 53-57).
  The 122B and GLM tables of that section are still being filled.

### Kernels
- `MOE_CACHE_IQ3_KERNEL=1` forces the bit-identical IQ3_XXS gather kernel on any AVX2 CPU, `=0` turns it off (GitHub issue #1: RX 9070 XT + Ryzen 7 7700 on Linux,
  Flash-Next 15.9 -> 18.8 tok/s). **Measured on a Zen 4 7950X: it is 0.58-0.64x of llama.cpp's own function, and the AVX-512 kernels (bit-exact) 0.60-0.77x, so the gate
  stays Intel-only by default.** The plugin's own CPU kernels (Q2_0, IQ3_XXS gather, AVX-512) are Linux-only (they need `LD_PRELOAD`).
- `state_t` methods split into three include files (`.text` and `.rodata` unchanged).

### Quality yardstick
- `tools/quality_tasks.json`: 202 original exact-answer tasks (arithmetic and word problems, unit conversions, Python output, strings, logic, dates, facts, tool-call JSON; the 20 old ones included), every answer computed or brute-forced by `tools/make_quality_tasks.py`. `quality_yardstick.py run --tasks tools/quality_tasks.json --tasks-only` runs them alone (no open prompts): 5-10 minutes on Flash-Next instead of 8-14.

### Known limits
- **Windows: a small page file limits the cache** (above): let Windows manage it, or make it 14 GB or more.
- **Flash-Next on AMD under Windows runs without context checkpoints with drivers older than 32.0.32015.7029** (driver crash, above).
- **HIP / ROCm release on Windows**: lists no device without AMD's HIP SDK; untested.
- **The plugin's CPU kernels are Linux-only**; on Windows the executor uses llama.cpp's own kernels. `--gpu-cache` is not available on Windows (needs `LD_PRELOAD`).
- Windows speed numbers are from one desktop; plugin decode is about equal to stock there (models bigger than RAM), prefill 1.5-2.2x. Intel GPUs on Windows: untested.

### Details of earlier entries in this section

### Control page: security review and Quick benchmark
- Control page, security review: a per-run token (in the address the launcher opens, then an HttpOnly SameSite=Strict cookie; required on every
  `/api` call), `X-Frame-Options: DENY` + CSP `frame-ancestors 'none'`, `Sec-Fetch-Site` and `Origin` checked on GET too, request bodies capped at
  64 KiB and a handler timeout, numbers range-checked (no nan / inf), the API key passed to the launcher in `LLAMA_API_KEY` (not argv),
  `/api/command` is POST, CSV cells that would run as formulas are quoted, `moe-cache-ui.bat` starts programs by full path from its own folder and
  escapes `'` in `--shortcut`.
- Control page: **Quick benchmark** for the running server (warm-up, 2 x 256-token replies, one ~3.5k-token prompt; decode, prefill and the
  cache hit rate), appended to `~/.cache/moe-cache/bench.csv` and shown next to earlier runs, so stock and moe-cache can be compared on your machine.
- Control page, step 2: Connect OpenCode / Hermes with a preview and what was written; live VRAM on Windows with AMD/Intel GPUs (the launcher's
  GPU pick + registry); warnings from the log (cache shrank, plugin disabled, why llama-server stopped) on the page; measured speeds from RESULTS next
  to the verdict for the models measured before; quant from the GGUF header (Ollama); warm-session state in the Agents tab; `bin\moe-cache-ui.bat`
  for Windows (one click: finds Python, the plugin DLL and llama-server; `--shortcut` adds a Start-menu entry); the page shows what the launcher
  decided (incl. the Windows/AMD rules: experts on the CPU, `--ctx-checkpoints 0`) and predicts those rules in the plan; `tests/ui_live.py` runs the page with Granite in CI (Linux and Windows).
- **Control page, step 1** (GUI1; `moe-cache-server --ui`, or `moe_cache.py ui`; [docs/GUI.md](docs/GUI.md)): a local page at
  http://127.0.0.1:8070, standard-library Python on Linux and Windows. Lists the GGUF models in `~/models` (`C:\models` on Windows), added folders and
  Ollama's store with the planner's verdict and its reasons (fits in RAM: stock / does not fit: moe-cache / all experts on the GPU; idea I33), starts
  and stops the server through the launcher, and shows live tok/s, cache hit rate, SSD reads, RAM and VRAM, llama-server's chat page and the log.
  `tests/ui_test.py` checks it without a model (CI on Linux and Windows).
- **Long prompts keep part of the cache** (`MOE_CACHE_SCAN_TOKENS`, default 512, 0 = off; idea I3): prompt batches of 512+ tokens load experts at the
  evict-first end of the cache, so what was cached before the prompt survives from chunk to chunk (with plain LRU a prompt bigger than the cache kept
  nothing). Flash-Next at 24 GB, paired: prefill 72.7 -> 92.4 tok/s (+27 %), cache hits during prompts 44 % -> 78 %, SSD reads 242 -> 103 GB; decode
  unchanged; output unchanged (only the eviction order differs). The same idea lost in October (X8, -25 %) before reads were merged. RESULTS 48.
- Launcher (found on the MSI Claw): llama-server now dies with the launcher (Windows job object, Linux parent-death signal; a closed or killed launcher
  left it running on its port), and when the plugin was meant to load but llama-server comes up without any `moe-cache:` line, the launcher says so
  loudly (it reported the plugin as loaded while llama-server silently ran stock).
- **Windows** (first real run, MSI Claw, llama.cpp b11669, RESULTS 49): the DLL did not load; `build.sh --windows` now imports the backend registry from
  `ggml.dll` (not `ggml-base.dll`) and links the MinGW threads runtime statically. Launcher on Windows: memory from GlobalMemoryStatusEx (no /proc there),
  the tuner's "no VRAM information" no longer overrides the planner (integrated GPUs), `check` says why the plugin did not load, the dry-run line is
  PowerShell syntax, the URL is 127.0.0.1 (Windows tried ::1 first), "profile dir will be created" instead of a false warning. `windows_test.ps1` names a
  missing DLL dependency instead of pointing at the API.
- **Windows: cache mode no longer crashes in the first graph** (MSI Claw round 2, RESULTS 50): the plugin's buffers are reserved address space only, and
  on Windows a reserved page faults until committed; the scheduler's compute buffer was never committed. Non-expert tensors are now committed when they
  are placed (once per 64 KiB chunk), and a commit range that runs past the end of its reservation (the last expert of a buffer plus one page)
  now commits the pages inside instead of nothing (found by the new Windows CI from a crash dump). Linux was never affected (untouched pages appear
  on first touch there).
- Windows: `--no-repack` on llama-server is now seen by the plugin (it repacked anyway on CPU-only machines, so its output differed from stock's).
  `MOE_CACHE_STATS_SIGNAL=1` works on Windows (named event `Local\moe-cache-stats-<pid>`); the snapshot includes executor counters.
- **CI on Windows** (`.github/workflows/windows.yml`): the DLL cross-built as `build.sh --windows`, the official llama.cpp b11669 CPU release,
  `windows_test.ps1` with forced eviction (4/4) and with the executor forced on and byte-checked, the platform tests natively, and the launcher.
  The sanitizer workflow's ThreadSanitizer job now runs (plugin loaded by path into a GGML_BACKEND_DL build): 4/4, no reports in plugin code.
  The identity workflow also runs the official loading method (GGML_BACKEND_PATH into a GGML_BACKEND_DL build).
- **`--gpu-cache` is refused where it gave wrong tokens** (mini PC, T7): with the plugin loaded through `GGML_BACKEND_PATH` (install.sh, Vulkan, official
  and all Windows builds) the hook that loads experts before llama.cpp's GPU cache uploads them cannot run. The plugin now ignores
  `MOE_CACHE_GPU_CACHE=1` there with a message, and the launcher falls back to static GPU expert layers. Builds that link their backends (LD_PRELOAD)
  are unchanged.
- The long-prompt GPU path uses an integrated GPU when there is no discrete one (Vega 8 via Vulkan: prompt 363 -> 569-597 tok/s).
- **Core pinning on Windows** (Claw finding 5): on hybrid CPUs the launcher pins llama-server to the performance cores and gives the executor
  efficiency-core workers, from the cores' efficiency class, as on Linux. Not yet run on real Windows.
- **`moe-cache-check MODEL --run`**: also starts llama-server with the plugin and a 0.25 GiB cache for one short request and reports whether the
  cache really computed, or why the server stopped (idea I26; the plain check said "plugin loads: yes" on the Claw while cache mode crashed).
- Launcher: says why llama-server stopped (exit code or signal, e.g. `0xC0000005 (access violation)`, and the last plugin line); planner errors give the
  real reason; the API mismatch warning works on Windows; a missing `MOE_CACHE_LIB` file is named; `check` uses GB throughout; the stock dry run has no
  empty `env`.
- Tools: `install.sh --vulkan` warns when Vulkan sees no real GPU (with the `render` group hint); `wait_cool.sh` works without lm-sensors on AMD;
  the friend test's kernel check finds `libggml-cpu-<variant>.so` and skips the GPU-cache case where it cannot work; the log says `layers all`.
- CI: sanitizer workflow (ThreadSanitizer / AddressSanitizer, `.github/workflows/sanitize.yml`, push to branch `sanitize` to run); long-prompt identity step.

## 0.2.3.5 (2026-10-15)
- **No abort when the cache is smaller than one op** (a fixed `MOE_CACHE_SIZE_GIB` below the experts one step needs; `auto` never gets there): the experts
  that do not fit are mapped straight from the model file, the way stock llama.cpp loads them (file cache, reclaimable, so the size still holds), or, where
  that is not possible (executor's background reads, repacked weights, Windows), loaded over the size for that step. One warning, and a count in the
  summary line. Two CI steps run with a 1 MiB cache (mapping with guard mode on; executor forced on) and must be 4/4 identical.
- Removed five switches whose experiments lost (all were off by default): `MOE_CACHE_L3PREFETCH` (-19 %), `MOE_CACHE_THP` (no gain),
  `MOE_CACHE_SCAN_TOKENS` (-25 % prefill), `MOE_CACHE_PREFAULT` (no gain), `MOE_CACHE_PROMPT_PREFETCH` (no gain). Setting them now does nothing.
- The plugin source is split into ten parts (`src/mc_*.inl`: common, pool, state, buffer, backend, exec, hybrid, compute, device, interpose), included
  by `src/moe_cache_plugin.cpp` in order: still one translation unit, so build commands do not change; the compiled code is byte-identical (`.text` and `.rodata`).
- **Launcher: stock llama-server when the model fits** in the memory the server may use (file + 1.5 GB <= `--ram`, the cgroup limit or available RAM):
  stock keeps the whole file in the page cache, warm across restarts (Qwen3.6-35B at 24 GB: stock 42.9 vs plugin 36.6 warm / 33.4 cold tok/s, RESULTS 44).
  `--fast` and `--hybrid` keep the plugin; `--always-plugin` forces it.
- **Launcher: no prompt-cache reserve below 16 GB** (`--cache-ram 0` instead of 2048): the 2 GiB reserve took a large share of the expert cache at small
  limits (Qwen3.6-35B at 12 GB: 16.9 tok/s with 2048 MiB, 19.2 with 512, 22.6 with 0, stock 6.9; this was the unexplained drop from RESULTS 2b's 23.5, audit
  E02). A single chat keeps reusing its slot; only switching between conversations loses the host prompt cache. Pass `--cache-ram` to choose.
- RESULTS 43: quant yardstick for `--fast` (T8): the Q2_0 file of Flash-Next is KL 0.243 from IQ2_XS (88.9 % same top token); `--fast` is 0.005-0.012.
- **Python launcher** (T15): `bin/moe_cache.py server|check|learn|project`, standard library only, also for Windows (plugin loaded with `GGML_BACKEND_PATH`).
  The bash scripts `bin/moe-cache-*` now call it; same command lines (`--dry-run` identical to the bash launcher in 10 flag combinations on Flash-Next and
  GLM). Fixed on the way: `--gpu-margin` was ignored (now passed to the tuner; GiB; default unchanged), `--vram` meant GiB for the planner but GB for the tuner
  (now GiB for both), `moe-cache-learn` refuses an occupied port and fails when its server stops or no new profile appears, project save/restore report
  HTTP errors and accept `--api-key`, the check counts every shard of a split model and knows 23b0202a1, and among several plugin builds for the same API the
  newest is used (it was the first by name, which picked old test builds). CI runs a launcher dry run and the check.
- Fixes from the static audit of 2026-10-15 (handover/STATIC_AUDIT_20261015_1324.md):
  - **`--fast` and `--hybrid` were silently inactive on K-quant models in 0.2.3.4** (the K-quant speed filter switched the executor off, and both modes
    live in it; output stayed exact). An explicitly requested mode now keeps the executor.
  - Lookahead (`MOE_CACHE_LOOKAHEAD`) set and cleared the "still loading" flag with plain stores while the executor reads it without the lock: atomic now.
  - The hybrid bank's GPU memory is freed with the backend; a freed model's experts are forgotten (no later eviction touches memory that was released).
  - Experts are checked byte for byte (first 4 KiB) against `MOE_CACHE_GGUF`, not only by name and size: another model with the same layout no longer
    gets the wrong weights (that tensor loads the normal way, with a warning).
  - Generic reads of expert memory (`get_tensor`) load the expert first; the executor pool cannot let a late worker read a job slot being refilled.
  - Profiles are written to a per-process temporary file and only replace the old one when every write succeeded.
  - The tuner also finds AMD GPU memory (it overrode the planner's AMD answer with "no GPU"). `install.sh` installs the newest release (it pinned
    v0.1.2); the release workflow checks that. `tools/windows_test.ps1` now turns the cache on (`-CacheGiB`, default 0.5) and fails when it is not on.
  - Tests: `identity.sh` fails when no prompt produced at least 8 tokens; two CI identity steps that could not fail now gate; the T45 test now reaches
    the step-aside wait (Q8_0 copy of the tiny gpt-oss, routing on with every pick kept, so output stays exact) and a mutation build without that wait
    must fail it. `tools/bank_replay.py` follows the live bank policy. RESULTS 41/42 wording qualified.

## 0.2.3.4 (2026-10-15)
- **Speed fix for K-quant models that do not fit in RAM** (in releases 0.2.0-0.2.3.3): the executor only takes graphs whose expert weights are IQ types or
  Q2_0; K-quants and legacy types go llama.cpp's own way, whose kernels are faster. Qwen3.6-35B Q4_K_M at 12 GB: 7.8 -> 16.6 tok/s, identical to stock.
- `MOE_CACHE_LOOKAHEAD=2` (experimental): the lookahead prefetch together with the executor (exact; no reliable gain on GLM-5.3-Flash, RESULTS 41).
- `MOE_CACHE_TEST_READ_DELAY_US=N` (tests only): background expert reads start N us late, so a code path that does not wait for an expert still loading
  reads stale data every time instead of only by bad luck. Checked on gpt-oss-20b: the build without the T45 fix fails 0/4, the fixed build passes 4/4.
- The plugin's Q2_0 kernels step aside when llama.cpp has its own (llama.cpp master, ggml-org/llama.cpp#30447: 96 ns per 2560-weight dot vs the plugin's
  126-238 ns, and adding up in another order). Checked once at start by comparing the library's function with the plain C loop; on 23b0202a1 nothing changes.
- The prompt-cache reserve line says where the number came from (launcher, `LLAMA_ARG_CACHE_RAM` or the command line).

## 0.2.3.3 (2026-10-15)
- **Exactness fix (race, in releases v0.2.0-v0.2.3.2):** when gate, up and down come in separate calls (gpt-oss), the experts read in the background
  for the next call could be used by llama.cpp's own path before they were complete (the executor stepped aside because they looked resident).
  gpt-oss-20b at a 1 GiB cache: 0/4 -> 4/4 identical. RESULTS section 36 (also: 7 more models identical on llama.cpp 23b0202a1).
- `MOE_CACHE_GPU_MIN_TOKENS=1` gave one repeated token (single-token batches on the GPU; Granite 3.1, T25); 1 is now treated as 2 with a warning.
- **Prompt reads merged** (`MOE_CACHE_COALESCE_MIB`, default 16, 0 = off): neighbouring expert reads of a long prompt become one read of up to 16 MiB.
  Flash-Next prompts read at 3.4 instead of 2.5 GB/s; prefill with the default launcher 74-81 -> 90-103 tok/s (RESULTS section 38).
- **Prompt speed fix for the default launcher** (`--gpu-cache off`): static expert layers and `-ub` now come from one tuner decision that keeps room for the
  long-prompt GPU block at the chosen micro-batch (1.6x a layer at `-ub 2048`). Flash-Next on 8 GB: prefill 18 -> 64-76 tok/s (RESULTS section 35).
- Launcher: `-b` follows a `-ub` above 2048 (it was fixed at 2048, which capped `--ub 4096`).
- `--hybrid` and `--fast` keep prompt batches under 64 tokens on the CPU (`MOE_CACHE_GPU_MIN_TOKENS=64`): short chat prompts 4.6 -> 3.3 s to the first token.
  The exact default keeps 32, as llama.cpp.
- `MOE_CACHE_PROMPT_PREFETCH=1` (experimental, off): read the next layer's experts in the background during long prompts; no gain measured.
- RESULTS section 34: the hybrid on GLM-5.3-Flash (bank too small on 8 GB) and on Qwen3.6 (slower than `--gpu-cache` for Q4_K experts).
- **Exactness fix (race, batches of 8 tokens):** in executor graphs the expert reads are asynchronous; the ops handed back to llama.cpp (tiled case, entry below)
  did not wait for them and could read an expert that was still loading. They wait now. Seen as logit differences at batch 8 with evictions (KL 0.004-0.006, Flash-Next
  and Qwen3-Coder); never in a release (it came with the tiled hand-back). New `tests/kl_batch.sh` (llama-perplexity logits at batch 8, stock vs plugin, must be KL 0)
  and a CI step that runs it with forced eviction and the forced executor (RESULTS section 31).
- **`moe-cache-server --hybrid`** (opt-in, experimental): sets `MOE_CACHE_HYBRID=auto`, keeps all experts on the CPU side and replaces `--gpu-cache`. Quality on Flash-Next:
  KL 0.0043 vs exact with 57 % of the picks on the GPU, 98.4 % same top token (`--fast` at 24 GB: 0.0053); RESULTS section 31.
- Hybrid: when a long prompt's GPU block needs the memory the bank holds, the bank is freed and loaded again when decoding starts. Prefill with `--hybrid`:
  17 -> 66-75 tok/s (`--gpu-cache auto`: 60-64), about 1 s per long prompt for the reload (RESULTS section 32).
- Hybrid: adaptive bank (`MOE_CACHE_HYBRID_ADAPT=N,K`, default 8,16, 0 = off): every N tokens up to K of the most missed experts replace the coldest expert of their
  layer's bank. Mixed-topic session: bank hits 20 -> 39 %, decode 13.1 -> 14.4 tok/s; when the profile already matches the prompts exactly it costs a little
  (51 -> 47 % hits, 17.6 -> 16.2 tok/s, one run). `tools/bank_replay.py` replays a `MOE_CACHE_TRACE_IDS` recording offline (RESULTS section 33).
- Hybrid: the adaptive bank switches itself (`MOE_CACHE_HYBRID_DUEL=W`, default 32 tokens): it compares itself with what the profile's own bank would have
  served and stops swapping (bringing profile experts back) while the profile does better. Matching profile 17.6 (fixed) -> 17.05 tok/s; drifting session
  12.8 -> 14.9 tok/s (+16.5 %) (RESULTS section 33.1).
- `--fast` quality re-measured without the race: KL 0.0053 at 24 GB, 0.0115 at 6 GiB (it was 0.025); about 1 top token in 40-50 differs.
- **Exactness fix (GLM-5.3-Flash):** llama.cpp 23b0202a1 computes an expert with a separate tiled routine once 8 or more tokens of a batch picked it, and that adds up
  in another order than the one-row kernel the executor uses. The executor now leaves exactly those ops to llama.cpp. Before: 3 of 12,348 ops differed in the last bit
  and one answer changed at its first token (3/4); after: 0 mismatches, 4/4, multi-token batches still on. New switch `MOE_CACHE_EXEC_MAXCOL=N` (1-8, default 8).
- `MOE_CACHE_EXEC_CHECK=1` now prints where the first mismatches are (weight type, tokens, first differing value).
- Faster IQ3_XXS dot product in the executor (gather-based, bit-identical, 1.24x on the dot product; Intel CPUs only). GLM-5.3-Flash's down projections are IQ3_XXS.
- The automatic cache size uses the smaller of the memory limit and what the machine has available (a 28 GB limit on a laptop with 25 GB free was OOM-killed).
- The executor finds llama.cpp's CPU kernels in builds with separately loaded backends (official releases, Vulkan, Windows), where it used to switch itself off.
  Checked on a Vulkan build (llama.cpp 7fe450e19, `GGML_BACKEND_PATH`): executor on, Flash-Next 4/4 identical to stock (RESULTS section 27).
- `tests/identity.sh`: `IDENT_KEEP=DIR` keeps both server logs and token files.
- **Experimental `MOE_CACHE_HYBRID=MiB|auto`** (off by default, not token-identical): a GPU bank of the most used experts; the GPU computes the picks it holds while
  the CPU executor computes the rest, in parallel, with nothing uploaded per token. Flash-Next on an 8 GB card: 13.1 -> 17.4 tok/s (RESULTS section 30).
- RESULTS section 26: GLM-5.3-Flash (101.8 GB) at 24 GB: 0.32 -> 1.7-1.9 tok/s, identical; 2.0-2.3 with `--fast`. docs/MULTI_DEVICE_DESIGN.md (design only).

## 0.2.3.2 (2026-10-14)
- `moe-cache-server --fast`: opt-in approximate routing (`MOE_CACHE_ROUTE=skip`), about +10-12 % when the model is much bigger than RAM, output no longer identical
  to stock. The default stays exact; the launcher suggests `--fast` when the model is more than twice the memory it may use.
- The launcher tells the plugin how much memory llama-server's prompt cache needs (`MOE_CACHE_HOST_RESERVE_MIB`) instead of the plugin reading the command line
  (kept as a fallback).
- `MOE_CACHE_GUARD=1` (tests only): an evicted expert's pages become inaccessible, so a read the plugin did not intercept crashes instead of seeing zeros.
- README: a "When to use something else" table (stock llama.cpp when the model fits, Strata for Flash-Next / Qwen3.6 on 12 GB+ VRAM and 32 GB+ RAM), a table of
  which features need which loading method, the Flash-Next row, MTP and Vulkan claims qualified. QUICKSTART: 2x-12x instead of 2x-7x. BACKENDS: the OpenVINO NPU backend.
- ROADMAP: llama.cpp version policy (API 3 first-class, API 2 bug fixes only, and when API 2 is dropped).
- **The benchmark tool moved to its own repository**, [moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark) (`bin/moe-cache-bench`, `bench/*.py`, BENCHMARKS.md, TEST_PROTOCOL.md).
  It finds moe-cache through `MOE_CACHE_DIR` or a folder next to it. The shared tuner moved from `bench/gguf_plan.py` to `planner/gguf_plan.py`.
- ROADMAP: llama.cpp API 2 stays in maintenance mode (bug fixes only) although its drop condition is met.

## 0.2.3.1 (2026-10-14)
- **Fix: CPU weight repacking now follows llama-server's `--no-repack` / `-nr` / `LLAMA_ARG_REPACK=0`.** On CPU-only machines the plugin repacked expert weights even when the server
  was started without repacking, so its kernels added up in another order than stock and tokens differed late in the answer (found by the new CI job).
- **CI: `identity` workflow** (GitHub runner, no GPU): llama.cpp built CPU-only at both pins, Granite 3.1 1B-A400M (sha256-checked), stock vs plugin with the cache forced
  to evict (about 39k evictions), with the executor forced on and byte-checked (6,708 ops, 0 mismatches), a noise-floor run and a no-eviction run; 4/4 identical on API 2 and API 3.
  `tests/identity.sh` now prints the first diverging token per prompt.

## 0.2.3 (2026-10-12)
- **Fix: long-prompt processing with the GPU expert cache.** The plugin's long-prompt GPU block needs about one layer of free VRAM; with a full-size cache it silently
  stayed on the CPU (prefill 24 tok/s) or aborted the server. The plugin now checks free VRAM and warns once; the tuner leaves room for the block
  (Flash-Next: cache 1800 instead of 2459 MiB at ctx 8192; prefill 24 -> 56 tok/s, decode about -5 %). RESULTS.md section 23.
- **Saved contexts**: `moe-cache-server --project NAME` (own usage profile plus restore of the saved prompt state at start) and `bin/moe-cache-project save|restore|list`.
  Restoring a 2.7k-token context took 0.25 s instead of 56 s with identical output (RESULTS.md section 24.2).
- `bench/coding_session.py --sections N` warms a context in growing prefixes so an edit in the middle reuses everything before the nearest checkpoint
  (886 tokens processed instead of 5,945; section 24.1). The context builder now respects its token budget (about 2 characters per token).
- The launcher picks the plugin build that matches the llama-server's ggml API (`build*/` next to each other) and warns on a mismatch; plugin builds carry a
  readable `moe-cache-api=N` tag.
- Executor counters by weight type with `MOE_CACHE_STATS=1` (time, ops, expert data per type).
- `tools/wait_cool.sh` (the clean-run gate as one command); the benchmark tools' default start temperature is now 65 C.
- `tools/trace_replay.py` (cache policies and next-layer prediction from a `MOE_CACHE_TRACE_IDS` recording); `bench/quality.py codegen` (10 small code tasks run against assertions).
- RESULTS.md sections 24 (prompt reuse, saved contexts, kernel time by type, cache policy, predictor recall) and 25 (quantisations, thinking, KL divergence of the approximate modes).

## 0.2.2 (2026-10-11)
- Opt-in **approximate routing** (`MOE_CACHE_ROUTE=skip|subst`, `MOE_CACHE_ROUTE_KEEP`): low-ranked picks that are not in RAM are dropped or replaced by a resident expert.
  +10-12 % at a small cache on Flash-Next with no measured loss on the checkable prompts (RESULTS.md section 21). Off by default.
- One-line capability report in the server log (`moe-cache: active: ...`): load method, executor and why it is off, kernels, GPU-cache mode.
- Shared tuner (`bench/gguf_plan.py: tune`) for micro-batch, GPU cache or static layers; `moe-cache-server --ub auto`.
- Fixes from the review: disk lookup table not counted as GPU weight in the benchmark planner, identity compares every repeat, manual `--moe-cache-mib` sets the plugin's
  host-buffer mode, MB/MiB units, quality-harness false passes. New `bench/coding_session.py`, `bench/quality.py judge-calibrate`.
- First coding-session measurement (RESULTS.md section 22): a repeated 20k-token context is reused (about 790 s cold, about 50 s per later question), but an edit in the
  middle of the context re-processes all of it. Timings were not temperature-gated.
- **Plugin + llama.cpp's GPU expert cache** (`MOE_CACHE_GPU_CACHE=1` with `--moe-cache-mib N`, or `moe-cache-server --gpu-cache auto`): the plugin makes experts resident
  right before llama.cpp uploads them. Identical to stock + cache; +12 % over static GPU layers on Flash-Next (4 rounds).
- **Small-batch executor**: the executor now takes up to 8 tokens per step (prompt batches, speculative verification), exact (0 byte mismatches in 57,669 compared ops).
- **Named profiles**: `moe-cache-server --profile coding`, `moe-cache-learn --name coding`.
- **Benchmark tool**: finds your models (Ollama, llama.cpp cache, Hugging Face, LM Studio, folders), runs many in a queue, redesigned page, Serve card.
- **Quality harness** `bench/quality.py` (compare, longctx, soak).
- **Testers' kit**: AVX-512 IQ2_XXS / IQ3_XXS kernels (opt-in `MOE_CACHE_AVX512=1`, self-checked, untested on real AVX-512 hardware), `tools/friend_test.sh`,
  `docs/BACKENDS.md` (ROCm / SYCL / Vulkan build notes, untested), `docs/FRIEND_TEST.md`.
- Opt-in `MOE_CACHE_PREFAULT=1` (no gain measured on Flash-Next), `MOE_CACHE_EXEC=2` (force the executor, for tests).
- New measurement tools: `tests/read_bench.cpp`, `tests/h2d_bench.cu`, `tests/sparsity_probe.cpp`, `tests/kernel_variants.cpp`.
- Analysis: `docs/PERFORMANCE_ANALYSIS.md`.
- Prompt processing: `-ub 2048` measured +58 % at a 6k prompt (short context); long contexts need a smaller `-ub` because the compute buffer grows with the context.
  Opt-in `MOE_CACHE_SCAN_TOKENS=N` (scan-resistant loading of prompt batches: -25 % prefill, +31 % first decode after a prompt; off by default).
- **Long-session memory safety**: llama-server's host prompt cache (`--cache-ram`, 8 GiB by default) is now accounted for. A 24 GB limit was OOM-killed at 128k context
  when a second request started. The plugin reads `--cache-ram` from the command line and keeps that much free; `moe-cache-server` and the benchmark tool cap it at 2 GiB
  unless you set it.

## 0.2.1 (2026-10-11)
- **Second supported llama.cpp**: commit `23b0202a1` (ggml 0.26.0, backend API 3; GLM-5.3-Flash and MTP support are in it) next to `7fe450e19` (ggml 0.25.1, API 2). Each needs its own plugin build: the release has one tarball per version (`...-ggml0.26-...` is the new one); from source use `GGML_API=3 ./build.sh`.
- A plugin built for one backend API now refuses to register in a ggml of the other API (it prints why and the model loads normally) instead of crashing.
- `install.sh`: `LLAMA_SHA=23b0202a1...` builds llama.cpp at the new commit and the matching plugin.
- Token identity vs stock re-checked on the new commit: Granite 3B and 1B, OLMoE 7B, Qwen3.6-35B, Qwen3-Coder-30B, a second qwen35moe model, Qwen3.8-Flash-Next IQ2_XS (4 prompts x 48 tokens each, all identical). Not re-run there: gpt-oss, Qwen3-Next 80B, DeepSeek-V2-Lite, Hunyuan.
- New tests: `tests/kernel_variants.cpp` (IQ2_XXS / IQ3_XXS dot product candidates, bit-exactness and timing) and `tests/read_bench.cpp` (O_DIRECT read speed: pread, pre-faulted pages, io_uring batches).
- Measurements in `docs/RESULTS.md` section 15, including llama.cpp's own new `--moe-cache-mib` (GPU cache of host experts): it does not combine with this plugin yet.

## 0.2.0 (2026-10-11)
- **Expert executor** (default on; steps aside when everything is resident and the weights are bandwidth-bound): background reader threads, work-sharing compute pool that starts on experts already in RAM, optional efficiency-core workers (`MOE_CACHE_EXEC_*`). Bit-identical to llama.cpp's output; `MOE_CACHE_EXEC_CHECK=1` compares every op byte for byte.
- **Fast Q2_0 x Q8_0 kernel** (AVX2, bit-identical, about 3x; the executor uses a second, permuted-activation version that is about 6x): llama.cpp has no SIMD version on x86. `MOE_CACHE_FAST_KERNELS=0` switches it off. `tests/kernel_check.cpp` verifies it.
- `bench/run_matrix.py`: benchmark many models, placements and memory limits and print a publishable table.
- Planner: built-in GGUF header reader (new quantisation types), split models, AMD VRAM, n-gram table not counted as GPU weights.
- Security and privacy: benchmark GUI token, Host/Origin checks and config validation; shareable reports without prompts or paths; private result store; hardened build flags; validated profile loading; graceful handling of a wrong `MOE_CACHE_GGUF`.
- Tests: identity test fails when the plugin is not active; limit test cleans up its own server; port-busy checks.
- Release: built by GitHub Actions from the tagged commit, with a build attestation; tarball includes all of `src/`.

## 0.1.4
- Benchmark tool documented for your own models and prompts; area 3 GUI fixes.

## 0.1.3
- First release built by GitHub Actions; attestation.

## 0.1.2
- Sibling batch load, read-into-place, automatic pinning, preload-all, benchmark GUI, installer.

## 0.1.1
- First prebuilt plugin.
