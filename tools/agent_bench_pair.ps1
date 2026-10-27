# agent_bench_pair.ps1: the Windows counterpart of tools/agent_bench_pair.sh (I30 paired runs; PowerShell 5).
# Stock llama-server vs moe-cache `--agent`, replaying the recorded agent sessions (tests\agent_sessions\*.json) with tools\agent_bench.py, ABBA-style.
# Each run: the model pushed out of Windows' file cache (tools\win_flush_cache.ps1: reads other big files, no admin), server started and, at the end,
# stopped by PID; health wait; a quiet-CPU gate; replay OpenCode then Hermes; one CSV row per replay, printed and appended to OutDir\results.csv.
# Stock = the llama-server command the launcher prints for --agent --dry-run (computed once), without the plugin or any key.
# moe-cache = the launcher with --agent --always-plugin (the stored API key is required by the server and sent by the replay) and an empty
# --slot-save-path (no warm session to restore). The script stops if the mc server log has no "moe-cache: active:" line.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\agent_bench_pair.ps1 -OutDir C:\dl\i30 -Model C:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf `
#       -Alias qwen36 -Ram 20 -FlushOther "C:\models\a.gguf;C:\models\b.gguf" [-Order "stock,mc,mc,stock"]
# -Ram: GB for the launcher (the plugin run also gets a Windows job memory limit of that size); -FlushOther: OTHER model files/folders totalling more than
# the machine's RAM (never the model under test). Environment: none needed (the plugin DLL is -Lib). Windows has no cgroup for the stock run: use a
# model bigger than the RAM (the comparison is then memory-limited on both sides by the machine itself).
# -Mtp auto|on|off: the launcher's MTP choice (I43), for both kinds (stock gets the same arguments): auto = on with --agent when the model has an
# MTP head and llama-server has draft-mtp; on = --mtp (stops if not possible); off = --no-mtp. Use a separate -OutDir per setting.
param([Parameter(Mandatory = $true)][string]$OutDir, [Parameter(Mandatory = $true)][string]$Model, [Parameter(Mandatory = $true)][string]$Alias,
      [Parameter(Mandatory = $true)][int]$Ram, [Parameter(Mandatory = $true)][string]$FlushOther, [string]$Order = "stock,mc,mc,stock", [int]$Port = 8099,
      [string]$LlamaServer = "C:\llama-vk\llama-server.exe", [string]$Lib = "C:\moe-cache-kit\ggml-moe-cache-api3.dll", [string]$Repo = "C:\moe-cache",
      [string]$Sessions = "", [ValidateSet("auto", "on", "off")][string]$Mtp = "auto")
$ErrorActionPreference = "Continue"
if (-not $Sessions) { $Sessions = Join-Path $Repo "tests\agent_sessions" }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$csv = Join-Path $OutDir "results.csv"; $prog = Join-Path $OutDir "progress.log"
function Log([string]$m) {
  $l = "{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $m
  $l
  for ($i = 0; $i -lt 5; $i++) { try { Add-Content -Path $prog -Value $l -ErrorAction Stop; break } catch { Start-Sleep -Milliseconds 300 } }
}
function Add-Csv([string]$line) {
  for ($i = 0; $i -lt 5; $i++) { try { Add-Content -Path $csv -Value $line -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 300 } }
}
function Stop-Tree($p) {
  if (-not $p) { return }
  foreach ($c in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)")) {
    foreach ($g in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($c.ProcessId)")) { Stop-Process -Id $g.ProcessId -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue
  }
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
function Quote-Args([string[]]$a) { return (($a | Where-Object { $_ -ne "" } | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } }) -join " ") }
function Port-Busy() { try { $null = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/health" -TimeoutSec 2; return $true } catch { return $false } }

$launcher = Join-Path $Repo "bin\moe_cache.py"; $bench = Join-Path $Repo "tools\agent_bench.py"; $flush = Join-Path $Repo "tools\win_flush_cache.ps1"
foreach ($f in @($launcher, $bench, $flush, $Model, $LlamaServer, $Lib)) { if (-not (Test-Path $f)) { Log "missing: $f"; exit 1 } }
$kinds = @($Order -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($k in $kinds) { if ($k -ne "stock" -and $k -ne "mc") { Log "-Order words are stock and mc, got '$k'"; exit 1 } }
if (-not (Test-Path $csv)) { Add-Csv "run,kind,session,requests,task_s,pauses_s,first_ttft_s,followup_ttft_median_s,followup_ttft_sum_s,prompt_processed,prompt_reused,generated,decode_tps_median,errors" }

# the stock command: what the launcher prints for --agent (computed once; dry run, nothing starts)
$env:MOE_CACHE_LIB = $Lib
$mtpFlag = @(switch ($Mtp) { "on" { "--mtp" } "off" { "--no-mtp" } default { } })
$dry = @(& python $launcher server $Model --agent --ram $Ram --port $Port --llama-server $LlamaServer --no-warm --dry-run @mtpFlag "--" --alias $Alias 2>&1 | ForEach-Object { "$_" })
$m = @($dry | Where-Object { $_ -match "llama-server\.exe" })
if ($m.Count -eq 0) { Log "could not get the launcher's command (dry run printed no llama-server line)"; $dry | ForEach-Object { Log "  $_" }; exit 1 }
$stockArgs = (($m[$m.Count - 1] -replace '^.*llama-server\.exe"?\s+', '') -replace "'", '"')
Log "stock arguments (-Mtp $Mtp): $stockArgs"
$dry | Where-Object { $_ -match "MTP" } | ForEach-Object { Log "  $_" }
$kl = @(& python $launcher key 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ } | ForEach-Object { $_.Trim() })
$key = $kl[$kl.Count - 1]     # the stored per-user key (made on first use); never logged

$k = 0
foreach ($kind in $kinds) {
  $k++; $tag = "$k-$kind"
  if (Port-Busy) { Log "port $Port is in use (another server?): stop"; exit 1 }
  $global:LASTEXITCODE = 0     # win_flush_cache.ps1 calls exit only on failure
  $out = & $flush -Target $Model -Other ($FlushOther -split ";" | Where-Object { $_ }) *>&1 | ForEach-Object { "$_" }      # *>&1: the script reports with Write-Host
  $fc = $LASTEXITCODE
  Log "run $k/$($kinds.Count) ($kind): $($out -join ' ')"
  if ($fc -ne 0) { Log "the file cache was not flushed enough (the other files are smaller than the RAM): stop"; exit 1 }
  $slots = Join-Path $OutDir "slots-$tag"; New-Item -ItemType Directory -Force $slots | Out-Null
  $so = Join-Path $OutDir "server-$tag.out.log"; $se = Join-Path $OutDir "server-$tag.err.log"
  if ($kind -eq "mc") {
    $env:MOE_CACHE_LIB = $Lib; $env:MOE_CACHE_STATS = "1"
    $a = @($launcher, "server", $Model, "--agent", "--always-plugin", "--ram", "$Ram", "--port", "$Port", "--llama-server", $LlamaServer) + $mtpFlag + @("--", "--alias", $Alias, "--slot-save-path", $slots)
    $p = Start-Process -FilePath python -ArgumentList (Quote-Args $a) -PassThru -RedirectStandardOutput $so -RedirectStandardError $se -WindowStyle Hidden
    Remove-Item Env:MOE_CACHE_STATS -ErrorAction SilentlyContinue
  } else {
    Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $LlamaServer -ArgumentList $stockArgs -PassThru -RedirectStandardOutput $so -RedirectStandardError $se -WindowStyle Hidden
  }
  $ready = $false; $t0 = Get-Date
  while (((Get-Date) - $t0).TotalSeconds -lt 1800) {
    Start-Sleep 2
    if ($p.HasExited) { break }
    try { if ((Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/health" -TimeoutSec 3).StatusCode -eq 200) { $ready = $true; break } } catch {}
  }
  if (-not $ready) { Log "run $k ($kind): the server did not become ready (see $se)"; Stop-Tree $p; exit 1 }
  if ($kind -eq "mc") {
    # "cache mode" is printed at load; "active:" only at the first computed graph, which with --no-warmup comes with the first request (checked after the replays)
    $cm = @(Get-Content $so, $se -ErrorAction SilentlyContinue | Where-Object { $_ -match "moe-cache: cache mode" })
    if ($cm.Count -eq 0) { Log "run $k (mc): no 'moe-cache: cache mode' line in the server log (plugin not active): stop"; Stop-Tree $p; exit 1 }
  }
  Start-Sleep 15
  # quiet-CPU gate (Windows has no temperature sensor to read here): three samples in a row at or below 15 % load, up to 5 minutes
  $quiet = 0; $g0 = Get-Date; $load = 0
  while ($quiet -lt 3 -and ((Get-Date) - $g0).TotalSeconds -lt 300) {
    $load = (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average
    if ($load -le 15) { $quiet++ } else { $quiet = 0 }
    Start-Sleep 5
  }
  Log "run $k ($kind): server up, CPU load $load % ($(if ($quiet -ge 3) { 'quiet' } else { 'NOT quiet after 5 min' }))"
  foreach ($s in @("opencode_stats", "hermes_stats")) {
    $json = Join-Path $OutDir "$tag-$s.json"; $txt = Join-Path $OutDir "$tag-$s.txt"
    if ($kind -eq "mc") { $env:LLAMA_API_KEY = $key } else { Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue }
    & python $bench replay (Join-Path $Sessions "$s.json") --url "http://127.0.0.1:$Port" --label "$kind run $k" --json $json 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $txt -Encoding utf8
    Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue
    $lines = @(Get-Content $txt -ErrorAction SilentlyContinue | Where-Object { $_.Trim() })
    Log ("run $k ($kind) $s : " + $(if ($lines.Count) { $lines[$lines.Count - 1] } else { "no output" }))
    if (Test-Path $json) {
      try {
        $r = Get-Content $json -Raw | ConvertFrom-Json; $su = $r.summary
        $row = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}" -f $k, $kind, $s, $su.requests, $su.task_s, $su.pauses_s, $su.first_ttft_s, `
               $su.followup_ttft_median_s, $su.followup_ttft_sum_s, $su.prompt_processed, $su.prompt_reused, $su.generated, $su.decode_tps_median, $su.errors
        Add-Csv $row; Log "  csv: $row"
      } catch { Log "  (could not read $json : $_)" }
    } else { Log "  (no result file for ${s}: the replay failed, see ${txt})" }
  }
  if ($kind -eq "mc") {
    $active = @(Get-Content $so, $se -ErrorAction SilentlyContinue | Where-Object { $_ -match "moe-cache: active:" })
    if ($active.Count -eq 0) { Log "run $k (mc): no 'moe-cache: active:' line after the replays (the plugin computed nothing): stop"; Stop-Tree $p; exit 1 }
    Log ("  " + $active[0].Substring(0, [math]::Min(200, $active[0].Length)))
  }
  Stop-Tree $p
  Start-Sleep 5
}
Log "done: $csv  (python tools\agent_bench.py table $OutDir\*.json for the side-by-side table)"
