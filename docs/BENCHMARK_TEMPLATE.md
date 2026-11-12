# Benchmark template: one record for every result

Every number we publish (our Windows PC, our Linux laptop, and anyone who sends a report) carries the same fields, so results can be compared and
re-run. The rule: **a value that was not recorded is written `not recorded`, never guessed.** A result with `not recorded` in a required field is
still shown, but it is not a chart number until it is measured again.

| Piece | File | What it is |
|---|---|---|
| This page | `docs/BENCHMARK_TEMPLATE.md` | the fields, how each is collected on Linux and Windows, the recipes, runs / claims / board |
| Schema | `docs/bench_record.schema.json` | the record as JSON Schema (required fields, "number or `not recorded`") |
| Collector | `bin/moe_cache_record.py` (also `python bin/moe_cache.py bench-record ...`) | collects the machine, samples a run (VRAM, process RAM, free RAM, CPU), builds, validates and renders the record; standard library only, Linux and Windows |
| Report | `python bin/moe_cache.py bench-report MODEL.gguf` | stock against moe-cache in ABBA order; since this template its report carries the record (card lines + JSON) |
| Runs, claims, board | `benchmarks/`, `bin/moe_cache_claims.py` | every published number is a claim on recorded runs; status computed; `BOARD.md` is what is true now |
| Issue form | `.github/ISSUE_TEMPLATE/benchmark_report.yml` | where a tester pastes it |

**Required** below means: the key must be in the record (the validator refuses a record without it), and its value is a real measurement or
`not recorded`. **Optional** keys may be absent.

## 1. Hardware (`hardware`)

| Field (key) | Req | Linux | Windows |
|---|---|---|---|
| CPU model (`cpu_model`) | yes | `/proc/cpuinfo` "model name" | PowerShell `Get-CimInstance Win32_Processor` |
| CPU cores / threads (`cpu_cores`, `cpu_threads`) | yes | `/proc/cpuinfo` physical id + core id; logical count | `Win32_Processor` NumberOfCores / NumberOfLogicalProcessors |
| RAM size (`ram_gib`) | yes | `/proc/meminfo` MemTotal | `GlobalMemoryStatusEx` |
| RAM type, speed (`ram_type`, `ram_speed_mts`) | yes | `dmidecode -t memory` (needs root; otherwise `not recorded`: run `sudo dmidecode -t memory` and add the two values by hand) | `Win32_PhysicalMemory` SMBIOSMemoryType, ConfiguredClockSpeed |
| GPU model (`gpu_model`) | yes | `nvidia-smi`, else `llama-server --list-devices` | `llama-server --list-devices` |
| VRAM total (`vram_mib`) | yes | `nvidia-smi`, else `--list-devices` | `--list-devices` (WMI's AdapterRAM stops at 4 GB, so it is not used) |
| GPU driver (`gpu_driver`) | yes | `nvidia-smi` driver_version; AMD: `not recorded` unless `vulkaninfo --summary` shows driverInfo | `Win32_VideoController` DriverVersion (AMD Adrenalin: also note the Adrenalin number in the issue) |
| OS, build (`os`, `os_build`) | yes | `platform` (kernel) | `platform` (e.g. 10.0.26100) |
| Storage: model, interface, drive (`storage.*`) | yes | `findmnt` + `lsblk` MODEL, TRAN of the disk holding the model file | `Get-Partition -DriveLetter X \| Get-Disk` FriendlyName, BusType (NVMe / SATA / USB) |
| Power plan (`power_plan`) | yes | `powerprofilesctl get` and the cpufreq governor | `powercfg /getactivescheme` |

The drive that holds the model matters more than the CPU for the cache's misses: always say which drive (`storage.drive`).

## 2. Software (`software`)

| Field | Req | How |
|---|---|---|
| moe-cache version (`moe_cache_version`) | yes | `VERSION` of `bin/moe_cache.py` |
| git commit (`moe_cache_commit`) | yes | `git rev-parse` in the checkout; a release kit has no `.git`: `not recorded`, so name the kit's file name in the issue |
| plugin file + sha256 prefix (`plugin_file`, `plugin_sha256_prefix`) | yes | sha256 (12 hex) of `libggml-moe-cache.so` / `ggml-moe-cache*.dll` found next to the launcher; `not recorded` if the launcher runs stock |
| llama.cpp build, commit (`llama_build`, `llama_commit`) | yes | `llama-server --version` ("version: 11966 (ff0d396e3)") |
| backend (`llama_backend`) | yes | `llama-server --list-devices`: Vulkan, ROCm-HIP, CUDA, SYCL, Metal, CPU only. AMD testers: say Vulkan or ROCm |
| Python (`python`) | yes | `platform.python_version()` |

## 3. Model (`model`)

| Field | Req | How |
|---|---|---|
| file name (`file`) | yes | base name only (no folders; the report scrubs paths) |
| quant (`quant`) | yes | GGUF general.file_type, else the experts' dominant tensor type |
| size (`size_gb`) | yes | sum of the shards |
| sha256 prefix (`sha256_head16MiB`) | yes | sha256 of the first 16 MiB (16 hex shown); with the size it identifies the file quickly |
| full sha256 prefix (`sha256_full`) | optional | `bench-record collect --hash-model` (reads the whole file: minutes on a 60 GB model) |
| source URL (`source_url`) | yes | the Hugging Face page of the file (`--model-url`); `not recorded` if you do not know it |

## 4. Settings (`settings`)

| Field | Req | How |
|---|---|---|
| full command line (`command_line`) | yes | the launcher's own plan: `moe_cache.py server MODEL ... --plan-json` ("command_line"), including its `env MOE_CACHE_...=` prefix. bench-report records it |
| environment (`environment`) | yes | every `MOE_CACHE_*`, `GGML_*`, `LLAMA_*` variable set in the process or in the command line's `env` prefix (`LLAMA_API_KEY` is never copied) |
| context, `-ub`, `-b` (`ctx`, `ubatch`, `batch`) | yes | parsed from the command line (`-c`, `-ub`, `-b`) |
| `--n-cpu-moe` and GPU expert layers (`n_cpu_moe`, `gpu_expert_layers`) | yes | `gpu_expert_layers` = model layers (llama-server log `n_layer`) minus `n_cpu_moe`; `not recorded` when the log did not show the layer count |
| KV type (`kv_type_k`, `kv_type_v`) | yes | `-ctk` / `-ctv`; `not recorded` = llama.cpp's default (f16) was used |
| MTP, `--fast` (`mtp`, `fast_flag`) | yes | yes / no, from the command line |
| threads (`threads`) | yes | `-t`; `not recorded` = llama.cpp default |
| stock arm's command line (`stock_command_line`) | optional | the `--stock` plan (bench-report) |
| `-ngl`, `n_layer`, flash attention | optional | command line / log |

## 5. Conditions (`conditions`)

| Field | Req | How |
|---|---|---|
| uptime (`uptime_h`) | yes | `/proc/uptime`; Windows `GetTickCount64`. Chart numbers need a restart within the last hours: write the restart time in the claim's statement |
| baseline check (`baseline_check`) | yes | the stock llama.cpp speed check before the run, e.g. "PASS 49.4 tok/s at 12:29" (`--baseline "..."`); `not recorded` if none was run |
| other programs on the GPU/CPU (`other_programs`) | yes | text you give (`--other-programs "browser closed, nothing else"`); the numbers below show what the machine was doing |
| file cache (`file_cache`) | yes | cold or warm (bench-report drops the model from the cache before each run unless `--warm`) |
| CPU busy before, free RAM before, VRAM in use before | optional | sampled for one second before the run (system-wide) |
| temperatures (`cpu_temp_c`, `gpu_temp_c`) | optional | Linux: hwmon k10temp/coretemp, `nvidia-smi`; **Windows: not collected** (no standard source without a vendor tool); write it by hand if you have one |

### Environment block (`runs[].environment`, `conditions.environment`; optional, schema 1 unchanged)

A published number is **real-use: measured on a working desktop**, so it says what else was running. The samplers of `bench-report` and `conversation_bench` (the `Sampler` of `bin/moe_cache_record.py`) take a snapshot at the start of each run, one every 30 s, and one at the end; the block holds the snapshot with the highest RAM in use (`basis`) and all three in `sampled_at` (`start`, `peak`, `end`). `conditions.environment` is the block of the busiest run. GB are GiB.

| Field | Meaning |
|---|---|
| `ram_total_gb`, `ram_used_total_gb` | installed RAM; total minus available (Windows: available includes standby; Linux: `MemAvailable` counts clean page cache as available) |
| `ram_moe_cache_gb` | llama-server and its launcher: working set (Windows) or `VmRSS` (Linux), i.e. private plus the mapped model pages it holds |
| `ram_rest_of_system_gb`, `top_programs` | used minus moe-cache; the five biggest programs as `[name, GB]` (one program's processes added up; the list is from the top processes gputaskman or `/proc` report, so it can be a lower bound) |
| `vram_total_gb`, `vram_per_process` | card size; `[name, GB, spill GB]` with moe-cache first, then the biggest others. Spill is the shared (system RAM) memory of that process; `not recorded` when the source gives none |
| `commit_free_gb` | Windows: commit limit minus commit charge; Linux: `CommitLimit - Committed_AS` (advisory when overcommit is on) |
| `source` | `gputaskman --json` (Windows PC `C:\tools\gputaskman\...`, laptop `gtm`), or `own counters (no gputaskman)`: then `top_programs` is `not recorded` on Windows and the VRAM list is moe-cache and "other programs" |

Rendering: `bench-record card RECORD --as env` prints the one-line text, the card's detail lines, the RESULTS section, `claim show` and the board's *Working desktop* column carry it:
`working desktop (at peak): 31 GB RAM, moe-cache 9.0 GB, other programs 8.4 GB (Claude 0.8, Dia 0.4, Code 0.3...), GPU: moe-cache 10.9 GB, desktop 1.8 GB`. A run without the block prints `Environment: not recorded` in `claim show` and no line elsewhere. For runs the chain harness makes outside `Sampler`, `bench-record env-from-gtm FILE.gtm.jsonl --start T --end T [--line]` builds the same block from chain's `gtm_sampler.ps1` lines for one run's time window.

## 6. Protocol (`protocol`)

| Field | Req | How |
|---|---|---|
| arms (`arms`) | yes | stock, moe-cache (and e.g. MTP variants, named in the run's `arm`) |
| order (`order`) | yes | `ABBA` (stock, moe-cache, moe-cache, stock, per pair) or `single` |
| runs per arm (`runs_per_arm`) | yes | bench-report `--pairs N` gives 2N |
| prompt set + sha256 (`prompt_set`, `prompt_sha256`) | yes | a name (`bench-report/chat-v1`) and the sha256 prefix of the prompt text; the prompt is not published, the hash proves two runs used the same one |
| prompt length (`prompt_tokens`) | yes | from the server's timings |
| `n_predict`, temperature, seed, sampler | yes | `--tokens`, 0, 42, greedy |

## 7. Metrics per run (`runs[]`)

| Field (key) | Req | Linux | Windows |
|---|---|---|---|
| prompt tok/s, decode tok/s (`prompt_tps`, `decode_tps`) | yes | `timings` of the server's answer | same |
| time to first token, wall time (`ttft_s`, `wall_s`) | yes | `timings.prompt_ms`; clock | same |
| VRAM used peak (`vram_used_peak_mib`, `vram_source`) | yes | `nvidia-smi --query-compute-apps` (the server's own share), else `nvidia-smi memory.used`, else amdgpu `mem_info_vram_used` (whole card); last fallback: llama-server's buffer-size log lines (model + KV + compute on the GPU; an estimate, labelled) | `typeperf "\GPU Process Memory(*)\Dedicated Usage"` (the server's own share), else `\GPU Adapter Memory(*)\Dedicated Usage` (whole card), else the log lines. Counters need Windows 10 1709+ and the WDDM driver (all current AMD/NVIDIA/Intel) |
| whole-card VRAM peak (`vram_system_peak_mib`) | optional | as above | as above |
| process RAM peak (`process_ram_peak_mib`) | yes | RSS of llama-server and its children, peak of 0.5 s samples and `VmHWM`. **Includes the mapped model file**, so a memory-mapped run can show far more than the cache's own memory | working set (`GetProcessMemoryInfo`) |
| process private RAM peak (`process_private_peak_mib`) | optional | `RssAnon` (anonymous memory: the cache, buffers) | private bytes |
| system RAM free minimum (`system_ram_free_min_mib`) | yes | lowest `MemAvailable` | lowest `ullAvailPhys` |
| CPU % average and peak (`cpu_avg_pct`, `cpu_peak_pct`) | yes | CPU time of the process tree per 0.5 s interval, as % of **all logical CPUs** (100 = every CPU busy); average over the run, highest interval | same, `GetProcessTimes` |
| system CPU % average, peak (`system_cpu_*`) | optional | `/proc/stat` | `GetSystemTimes`; shows other programs competing |
| cache hit % (`cache_hit_pct`) | optional | plugin stats (`MOE_CACHE_STATS_SIGNAL`) | same |
| disk read (`disk_read_gb`) | optional | plugin stats `read_bytes` | same |
| GPU blocks / segments (`gpu_blocks`) | optional | only if the plugin reports them; otherwise `not recorded` | same |
| text hash (`text_sha256`) | yes | sha256 prefix of the answer | same |

Sampling is every 0.5 s (VRAM every 2 s; on Windows each `typeperf` call takes about a second), so a peak shorter than that is missed. CPU and RAM sampling
costs well under 1 % of one CPU. The sampler measures the launcher's process tree (the launcher and its llama-server child).

## 8. Outputs (`outputs`)

`text_sha256_by_arm` (hash per arm) and `identical_to_baseline`: `yes`, `no` or `not checked` (approximate modes, or an arm that failed). A `no` is reported, not hidden.

## 9. Reproduce (`reproduce`)

`commands` (the exact command that was run; for bench-report the line starts `python bin/moe_cache.py bench-report <model file> ...`), `kit` (release page or branch and
commit), `model_url`, `prompt_file`. With these and the record's settings a second person can run the same thing.

## Recipes

### Pick a model the launcher does not run as stock

The launcher runs plain llama.cpp when the model fits your RAM and all its experts fit on the card, and then there is nothing to compare (a model that fits in RAM but keeps expert layers on the CPU runs through the plugin by default; `--stock` gives plain llama.cpp for the comparison). Pick a file **bigger than your free RAM** (rule of thumb:
size at least 1.2 x RAM, e.g. 40 GB or more on 32 GB):

```bash
python3 bin/moe_cache.py pick          # recommends a model for your RAM / VRAM / disk and prints the download command
python3 bin/moe_cache.py server MODEL.gguf --plan-json | grep -o '"engine": *"[a-z]*"'     # must say "plugin"
```

On Windows: `python bin\moe_cache.py server D:\models\MODEL.gguf --plan-json | findstr engine`. `"stock"` means it fits: pick a bigger model (or `--force-plugin` for a
forced run, which the report then states).

### Windows (10 / 11), normal PowerShell, no administrator

```powershell
# 1. llama.cpp is in the kit (llama\ folder, Vulkan). AMD: Vulkan, not HIP. Python 3.8+ with "Add to PATH".
# 2. close the browser and games; plug in a laptop; restart the PC first for chart-quality numbers
python bin\moe_cache.py doctor D:\models\MODEL.gguf
# 3. the run: 2 ABBA pairs, cold file cache (--flush-from = a folder of other big files, about as large as your RAM)
python bin\moe_cache.py bench-report D:\models\MODEL.gguf --pairs 2 --flush-from D:\other-big-files --out $env:USERPROFILE\moe-reports `
    --baseline "PASS 49.4 tok/s" --other-programs "nothing" --model-url https://huggingface.co/ORG/REPO
# --baseline / --other-programs / --model-url are text you give; leave them out and the report says "not recorded"
```

Windows notes: the report names the backend (Vulkan or ROCm) from `--list-devices`; VRAM comes from the GPU performance counters (`typeperf`); temperatures are not collected.

### Linux

```bash
./install.sh                                  # or your own llama.cpp built with -DBUILD_SHARED_LIBS=ON
python3 bin/moe_cache.py doctor MODEL.gguf
python3 bin/moe_cache.py bench-report MODEL.gguf --pairs 2 --out ~/moe-reports --other-programs "nothing" --model-url https://huggingface.co/ORG/REPO
sudo dmidecode -t memory | grep -E "Type:|Speed:"     # optional: RAM type and speed, which need root
```

### Paste it into the issue

1. The report is `moe-cache-report-YYYYMMDD-HHMM.md` (and `.json`) in the `--out` folder. It holds the numbers, the four card lines and the whole record in a collapsed JSON block; no prompts, answers, user name, host name or folder names.
2. Open the [Benchmark report form](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=benchmark_report.yml), paste the whole `.md` into **The report**, fill the few key fields (backend, GPU and driver, OS, what else was running), tick the privacy box.
3. Check the gaps list at the end of the report: every field that says `not recorded` can still be added as a comment.

### Our own runs (PC harness, laptop scripts)

A harness that measures tok/s itself keeps its own loop and adds the sampler and the record:

```bash
# per run: start the server, then watch its process while the harness sends the prompts
python3 bin/moe_cache.py bench-record monitor --pid 12345 --seconds 120 --out mon_A1.json     # or: monitor --out mon.json -- <the command>
# at the end: runs.json = [{"arm": "stock", "decode_tps": 28.4, "prompt_tps": 408.6, "ttft_s": 4.9, "text": "...", "monitor": "mon_A1.json"}, ...]
python3 bin/moe_cache.py bench-record assemble --runs runs.json --model MODEL.gguf --llama-server LS --backend Vulkan --order ABBA \
    --command-line "env MOE_CACHE_SIZE_GIB=auto llama-server -m ... -c 65536 --n-cpu-moe 39 -ub 2048" --prompt-file prompt.txt --prompt-tokens 34000 --n-predict 128 \
    --cache cold --baseline "PASS 49.4 tok/s" --other-programs "none" --model-url https://huggingface.co/... --out record.json
python3 bin/moe_cache.py bench-record validate record.json            # schema check, and how many fields say "not recorded"
python3 bin/moe_cache.py bench-record card record.json --as details --section 80    # the four lines for docs/benchmark_card.html
python3 bin/moe_cache.py bench-record card record.json --as results --section 80 --title "Title (Windows PC, date)"   # the old-style RESULTS section
python3 bin/moe_cache.py bench-record ingest record.json --machine pc      # into benchmarks/runs/; then `claim add`, `board` (see "Runs, claims and the board")
```

From Python: `from moe_cache_record import Sampler, run_entry, assemble`; `Sampler(pid).start()` ... `Sampler.stop()` returns the monitor dict that `run_entry` takes.
Run `collect --out env.json` once per session and pass `--env env.json` to `assemble` so the machine facts are read once.

## The benchmark card's detail lines

`bench-record card FILE --as details` prints the four lines of a row in `docs/benchmark_card.html` (field `d`):

```
GPU expert layers: 9 (n-cpu-moe 39) · -ub 2048
ctx 65536 · prompt 34000 tokens · 2 runs each, order ABBA · cold file cache
VRAM peak 7.8 GiB · process RAM peak 9.0 GiB (system free min 4.2 GiB) · CPU 22.5 % avg, 61.0 % peak
launcher abc123def, DLL 0123456 · llama.cpp 11966 (ff0d396e3, Vulkan) · 11-03 · §80
```

A row whose line says `not recorded` is a row that still has to be measured (or the field edited in by hand with the reason).

## Runs, claims and the board (`benchmarks/`)

The problem this solves: a number gets published (README, the benchmark card, release notes), the code or llama.cpp moves on, and nobody can tell which published
numbers still hold. So every published number is a **claim** that points at the **runs** behind it, and its status is computed.

```
benchmarks/
  runs/DATE_MACHINE_SHORTID.json   one record (this page's format) per run set; append-only, never edited. SHORTID = 6 hex of the content hash
  claims/CLAIM-ID.md               a number we publish: statement, runs behind it, cells, what it is valid for
  BOARD.md                         generated: machine x model x setting x arm -> newest number, date, status, claim
  JOURNAL.md                       generated: one line per run set, newest first
```

**Run**: what a harness measured, as a record. `bench-record ingest RECORD.json [--machine pc]` copies it into `runs/` under its canonical name (the machine name
defaults to the GPU). A record that does not validate is refused; the same record again changes nothing. An old result with only a few numbers written down is
ingested as a record whose other fields say `not recorded`, and its `notes` say where it was converted from. Never edit a run file: measure again and ingest the new one.

**Claim**: `bench-record claim add ID --run RUNID [--run ...] [--vs ARM] --title "the published sentence" --statement "conditions, order, caveats, source" [--valid-moe MIN-MAX] [--valid-llama MIN-MAX]`.
It writes `claims/ID.md`: a front matter block (`runs`, `cells`, `valid_moe_cache`, `valid_llama_builds`, `withdrawn`) and the plain-words statement. A **cell** is
one arm and one metric (`prompt_tps` or `decode_tps`) of one run; its value is **computed** as the mean of that arm's runs, never typed, so the claim cannot drift
from its runs. `--vs ARM` adds the change against that arm. Validity defaults to the version and llama.cpp build of the run. `claim show ID` prints the claim with its
computed values, spread and gaps; `claim list` lists all with their status.

**Status is computed**, in this order:

| Status | When |
|---|---|
| withdrawn | `withdrawn:` has a reason (the only status typed by hand; use it when a run was bad, e.g. the PC slowed down) |
| superseded | every cell of the claim is covered by a newer claim (same machine, model, setting, arm and metric) |
| stale | the moe-cache `VERSION` is above `valid_moe_cache`, the llama.cpp pin (`LLAMA_TAG` in `tools/make_windows_kit.sh`) is above `valid_llama_builds`, or the validity was not recorded |
| current | none of the above |

Stale does not mean wrong: it means nobody re-measured since the code or llama.cpp moved. Two claims of the same date with different numbers for one cell are
listed under **Conflicts** on the board.

**Board**: `bench-record board` writes `BOARD.md` and `JOURNAL.md`; `board --check` exits 1 when they are out of date. `BOARD.md` is the short "what is true now" page.

**Lint**: `bench-record lint [FILES]` checks `README.md` and `docs/benchmark_card.html`. A *performance number* is a `N tok/s` value, a signed percent (`+42 %`) or an
`Nx` / `N×` multiplier. Every line that shows one needs the marker `<!-- claim:ID -->` somewhere on the same line (in a table row, put it inside the row; in
Markdown, put it at the end of the line), and the claim must be *current*. It reports unclaimed lines, unknown claim ids and claims that are not current; code
fences and `<script>` / `<style>` blocks are skipped. Default exit 1 on a problem; `--warn` prints and exits 0 (what `quick.yml` runs until every current number has a claim).

**Replay**: `bench-record replay ID` prints the conditions to match and the exact commands of the claim's runs (`reproduce.commands`; `not recorded` for a converted run).
`--run` executes those commands. `--record NEW.json` compares a new record with the claim, cell by cell, inside a noise band: the largest of 5 % (`--band`), the
claim's own first/second spread and the new record's spread. Exit 1 when a cell is outside its band; then measure again and add a newer claim, or withdraw the old one.
`--arm-map CLAIMARM=NEWARM` maps arm names when the new record calls them something else (e.g. `v0.3.5=moe-cache`).

## Old RESULTS.md sections (up to 79)

Sections 1-79 of `docs/results-archive/RESULTS.md` keep their text; where they lack a field, the field is not recorded. `bench-record card FILE --as results --section N --title "..."`
still prints the old-style section for anyone who wants one in `RESULTS.md`.
