# conversation_bench_pair.ps1 (TEST-699): the Windows driver for tools\conversation_bench.py: stock llama-server against moe-cache, ABBA, one multi-turn
# conversation per run that fills the context (default 65,536 tokens), each run on a cold file cache. PowerShell 5. NOT run by the author on a PC yet: read
# docs\CONVERSATION_BENCH.md, run it once with -Ctx 8192 -MaxTurns 4 to see it work, then the real run.
# Same shape as tools\agent_bench_pair.ps1: the model pushed out of Windows' file cache before every run (tools\win_flush_cache.ps1), the server started and
# stopped by PID, a health wait, a quiet-CPU gate, then `conversation_bench.py run`; at the end `conversation_bench.py assemble` writes ONE bench record.
# Stock = the llama-server command the launcher prints for --agent --dry-run (computed once). moe-cache = the launcher with --agent --always-plugin and
# MOE_CACHE_STATS_SIGNAL=1, so the plugin's hits_dec / misses_dec are read per turn from the server's stderr log.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\conversation_bench_pair.ps1 -OutDir C:\dl\conv -Model D:\models\Qwen3.8-Flash-Next-IQ3_XXS.gguf `
#       -Alias flash -Ram 24 -FlushOther "D:\other-models" [-Ctx 65536] [-MaxTokens 128] [-Order "stock,mc,mc,stock"] [-MaxTurns 0]
param([Parameter(Mandatory = $true)][string]$OutDir, [Parameter(Mandatory = $true)][string]$Model, [Parameter(Mandatory = $true)][string]$Alias,
      [Parameter(Mandatory = $true)][int]$Ram, [Parameter(Mandatory = $true)][string]$FlushOther, [int]$Ctx = 65536, [int]$MaxTokens = 128,
      [string]$Order = "stock,mc,mc,stock", [int]$MaxTurns = 0, [int]$Port = 8099, [string]$LlamaServer = "C:\llama-vk\llama-server.exe",
      [string]$Lib = "C:\moe-cache-kit\ggml-moe-cache-api3.dll", [string]$Repo = "C:\moe-cache", [string]$Baseline = "", [string]$OtherPrograms = "")
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$prog = Join-Path $OutDir "progress.log"
function Log([string]$m) {
  $l = "{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $m
  $l
  for ($i = 0; $i -lt 5; $i++) { try { Add-Content -Path $prog -Value $l -ErrorAction Stop; break } catch { Start-Sleep -Milliseconds 300 } }
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

$launcher = Join-Path $Repo "bin\moe_cache.py"; $bench = Join-Path $Repo "tools\conversation_bench.py"; $flush = Join-Path $Repo "tools\win_flush_cache.ps1"
foreach ($f in @($launcher, $bench, $flush, $Model, $LlamaServer, $Lib)) { if (-not (Test-Path $f)) { Log "missing: $f"; exit 1 } }
$kinds = @($Order -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($k in $kinds) { if ($k -ne "stock" -and $k -ne "mc") { Log "-Order words are stock and mc, got '$k'"; exit 1 } }

# the stock command: what the launcher prints for --agent --ctx N (computed once; dry run, nothing starts)
$env:MOE_CACHE_LIB = $Lib
$dry = @(& python $launcher server $Model --agent --ctx $Ctx --ram $Ram --port $Port --llama-server $LlamaServer --no-warm --dry-run "--" --alias $Alias 2>&1 | ForEach-Object { "$_" })
$m = @($dry | Where-Object { $_ -match "llama-server\.exe" })
if ($m.Count -eq 0) { Log "could not get the launcher's command (dry run printed no llama-server line)"; $dry | ForEach-Object { Log "  $_" }; exit 1 }
$stockArgs = (($m[$m.Count - 1] -replace '^.*llama-server\.exe"?\s+', '') -replace "'", '"')
Log "stock arguments: $stockArgs"
# the moe-cache command: the same dry run with the plugin (its llama-server line shows the GPU/CPU layer plan the launcher chose for the plugin arm)
$env:MOE_CACHE_LIB = $Lib
$dryMc = @(& python $launcher server $Model --agent --ctx $Ctx --always-plugin --ram $Ram --port $Port --llama-server $LlamaServer --no-warm --dry-run "--" --alias $Alias 2>&1 | ForEach-Object { "$_" })
$mm = @($dryMc | Where-Object { $_ -match "llama-server\.exe" })
$mcArgs = if ($mm.Count -gt 0) { (($mm[$mm.Count - 1] -replace '^.*llama-server\.exe"?\s+', '') -replace "'", '"') } else { "" }
Log "moe-cache arguments: $mcArgs"
$kl = @(& python $launcher key 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ } | ForEach-Object { $_.Trim() })
$key = $kl[$kl.Count - 1]     # the stored per-user key (made on first use); never logged

$runs = @(); $k = 0
foreach ($kind in $kinds) {
  $k++; $tag = "$k-$kind"
  if (Port-Busy) { Log "port $Port is in use (another server?): stop"; exit 1 }
  $global:LASTEXITCODE = 0
  $out = & $flush -Target $Model -Other ($FlushOther -split ";" | Where-Object { $_ }) *>&1 | ForEach-Object { "$_" }
  $fc = $LASTEXITCODE
  Log "run $k/$($kinds.Count) ($kind): $($out -join ' ')"
  if ($fc -ne 0) { Log "the file cache was not flushed enough (the other files are smaller than the RAM): stop"; exit 1 }
  $slots = Join-Path $OutDir "slots-$tag"; New-Item -ItemType Directory -Force $slots | Out-Null
  $so = Join-Path $OutDir "server-$tag.out.log"; $se = Join-Path $OutDir "server-$tag.err.log"
  if ($kind -eq "mc") {
    $env:MOE_CACHE_LIB = $Lib; $env:MOE_CACHE_STATS_SIGNAL = "1"
    $a = @($launcher, "server", $Model, "--agent", "--ctx", "$Ctx", "--always-plugin", "--ram", "$Ram", "--port", "$Port", "--llama-server", $LlamaServer, "--no-warm") + @("--", "--alias", $Alias, "--slot-save-path", $slots)
    $p = Start-Process -FilePath python -ArgumentList (Quote-Args $a) -PassThru -RedirectStandardOutput $so -RedirectStandardError $se -WindowStyle Hidden
    Remove-Item Env:MOE_CACHE_STATS_SIGNAL -ErrorAction SilentlyContinue
  } else {
    Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $LlamaServer -ArgumentList $stockArgs -PassThru -RedirectStandardOutput $so -RedirectStandardError $se -WindowStyle Hidden
  }
  $ready = $false; $t0 = Get-Date; $loadS = 0
  while (((Get-Date) - $t0).TotalSeconds -lt 1800) {
    Start-Sleep 2
    if ($p.HasExited) { break }
    try { if ((Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/health" -TimeoutSec 3).StatusCode -eq 200) { $ready = $true; $loadS = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1); break } } catch {}
  }
  if (-not $ready) { Log "run $k ($kind): the server did not become ready (see $se)"; Stop-Tree $p; exit 1 }
  if ($kind -eq "mc") {
    $cm = @(Get-Content $so, $se -ErrorAction SilentlyContinue | Where-Object { $_ -match "moe-cache: cache mode" })
    if ($cm.Count -eq 0) { Log "run $k (mc): no 'moe-cache: cache mode' line in the server log (plugin not active): stop"; Stop-Tree $p; exit 1 }
  }
  Start-Sleep 15
  $quiet = 0; $g0 = Get-Date; $load = 0
  while ($quiet -lt 3 -and ((Get-Date) - $g0).TotalSeconds -lt 300) {
    $load = (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average
    if ($load -le 15) { $quiet++ } else { $quiet = 0 }
    Start-Sleep 5
  }
  Log "run $k ($kind): server up, CPU load $load % ($(if ($quiet -ge 3) { 'quiet' } else { 'NOT quiet after 5 min' }))"
  $json = Join-Path $OutDir "$tag.json"; $txt = Join-Path $OutDir "$tag.txt"
  $ba = @($bench, "run", "--url", "http://127.0.0.1:$Port", "--arm", $(if ($kind -eq "mc") { "moe-cache" } else { "stock" }), "--ctx", "$Ctx", "--max-tokens", "$MaxTokens", "--out", $json)
  if ($MaxTurns -gt 0) { $ba += @("--max-turns", "$MaxTurns") }
  # every run: resource peaks (VRAM, process RAM, free RAM, CPU) sampled by the runner on llama-server's pid, the run's own command line (GPU layers, ctx, ub),
  # llama-server's log (layer counts, buffer sizes) and the load time measured above
  $llama = @(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Sort-Object CreationDate -Descending)
  $cmdText = if ($kind -eq "mc") { "env MOE_CACHE_LIB=$Lib MOE_CACHE_STATS_SIGNAL=1 llama-server $mcArgs" } else { "llama-server $stockArgs" }
  $cmdFile = Join-Path $OutDir "cmd-$tag.txt"; Set-Content -Path $cmdFile -Value $cmdText -Encoding UTF8
  $ba += @("--command-line", "@$cmdFile", "--server-log", $se, "--load-s", "$loadS")
  if ($llama.Count -gt 0) { $ba += @("--monitor-pid", "$($llama[0].ProcessId)") }
  if ($kind -eq "mc") {
    if ($llama.Count -gt 0) { $ba += @("--stats-pid", "$($llama[0].ProcessId)", "--stats-log", $se) }
    $env:LLAMA_API_KEY = $key
  }
  & python @ba 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $txt -Encoding utf8
  Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue
  $lines = @(Get-Content $txt -ErrorAction SilentlyContinue | Where-Object { $_.Trim() })
  Log ("run $k ($kind): " + $(if ($lines.Count) { $lines[$lines.Count - 1] } else { "no output" }))
  if (Test-Path $json) { $runs += $json } else { Log "run ${k}: no result file (see $txt)" }
  if ($kind -eq "mc") {
    $active = @(Get-Content $so, $se -ErrorAction SilentlyContinue | Where-Object { $_ -match "moe-cache: active:" })
    if ($active.Count -eq 0) { Log "run $k (mc): no 'moe-cache: active:' line after the run (the plugin computed nothing): stop"; Stop-Tree $p; exit 1 }
  }
  Stop-Tree $p
  Start-Sleep 5
}
if ($runs.Count -eq $kinds.Count) {
  $rec = Join-Path $OutDir "record.json"
  $aa = @($bench, "assemble") + $runs + @("--out", $rec, "--model", $Model, "--llama-server", $LlamaServer, "--cache", "cold")
  if ($Baseline) { $aa += @("--baseline", $Baseline) }
  if ($OtherPrograms) { $aa += @("--other-programs", $OtherPrograms) }
  & python @aa 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $OutDir "tables.md") | ForEach-Object { Log $_ }
  Log "done: $rec and $(Join-Path $OutDir 'tables.md')    (then: python bin\moe_cache.py bench-record ingest $rec --machine pc)"
} else { Log "done, but not every run produced a result file: no record written" }
