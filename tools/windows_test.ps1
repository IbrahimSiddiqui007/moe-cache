# moe-cache on Windows: UNTESTED port. This script answers one question: does the plugin load into an official Windows llama-server and give the same tokens as stock?
#   powershell -ExecutionPolicy Bypass -File windows_test.ps1 -LlamaDir C:\llama -Model D:\models\model.gguf -Dll C:\moe\ggml-moe-cache-api3.dll
# -LlamaDir : folder with llama-server.exe (an official release zip, CUDA build if you have an NVIDIA GPU)
# -Model    : a GGUF that FITS IN RAM for the first test (Windows has no memory-limit scope here: a model bigger than RAM can freeze the machine)
# -Dll      : ggml-moe-cache-api2.dll (llama.cpp 7fe450e19 / ggml 0.25.x) or ggml-moe-cache-api3.dll (llama.cpp 23b0202a1 and newer ggml 0.26.x)
# -CacheGiB : expert cache size for the plugin run (default 0.5: far below the model, so experts are evicted and read again; "auto" sizes it from RAM)
param(
  [Parameter(Mandatory = $true)][string]$LlamaDir,
  [Parameter(Mandatory = $true)][string]$Model,
  [Parameter(Mandatory = $true)][string]$Dll,
  [int]$Port = 8099, [int]$Tokens = 32, [int]$Ctx = 4096, [int]$NCpuMoe = 999, [string]$CacheGiB = "0.5"
)
$ErrorActionPreference = "Stop"
$srv = Join-Path $LlamaDir "llama-server.exe"
if (-not (Test-Path $srv)) { throw "llama-server.exe not found in $LlamaDir" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
if (-not (Test-Path $Dll)) { throw "plugin dll not found: $Dll" }
$prompts = @("Explain what a hash table is in two sentences.", "Write a Python function that reverses a string.",
             "What is the capital of Australia and why was it chosen?", "List three uses of a binary search.")
$out = Join-Path $env:TEMP "moe-cache-test"; New-Item -ItemType Directory -Force $out | Out-Null

function Run-One([string]$label, [bool]$plugin) {
  $log = Join-Path $out "$label.err.txt"; $std = Join-Path $out "$label.out.txt"
  # MOE_CACHE_SIZE_GIB turns the cache on: without it the plugin only passes the weights through and the test would prove nothing
  if ($plugin) { $env:GGML_BACKEND_PATH = (Resolve-Path $Dll).Path; $env:MOE_CACHE_GGUF = (Resolve-Path $Model).Path; $env:MOE_CACHE_STATS = "1"; $env:MOE_CACHE_SIZE_GIB = $CacheGiB }
  else { foreach ($v in "GGML_BACKEND_PATH", "MOE_CACHE_GGUF", "MOE_CACHE_STATS", "MOE_CACHE_SIZE_GIB") { Remove-Item "Env:$v" -ErrorAction SilentlyContinue } }
  $sargs = @("-m", $Model, "-ngl", "99", "--n-cpu-moe", "$NCpuMoe", "-c", "$Ctx", "-fa", "on", "--no-warmup", "--port", "$Port", "--seed", "42")
  $p = Start-Process -FilePath $srv -ArgumentList $sargs -PassThru -RedirectStandardError $log -RedirectStandardOutput $std -WindowStyle Hidden
  $up = $false
  for ($i = 0; $i -lt 600; $i++) {
    if ($p.HasExited) { break }
    try { if ((Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/health" -TimeoutSec 2).StatusCode -eq 200) { $up = $true; break } } catch {}
    Start-Sleep -Seconds 2
  }
  if (-not $up) { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; Get-Content $log -Tail 15; throw "$label : server did not start (see $log)" }
  $texts = @(); $tps = @()
  foreach ($q in $prompts) {
    $body = @{ prompt = $q; n_predict = $Tokens; temperature = 0; seed = 42; cache_prompt = $false } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 3600
    $texts += $r.content; $tps += [math]::Round($r.timings.predicted_per_second, 2)
  }
  Stop-Process -Id $p.Id -Force; Start-Sleep -Seconds 3
  $active = (Select-String -Path $log -Pattern "moe-cache:" -SimpleMatch | Select-Object -First 6 | ForEach-Object { $_.Line }) -join "`n"
  $cache = [bool](Select-String -Path $log -Pattern "moe-cache: cache mode" -SimpleMatch -Quiet)
  $disabled = [bool](Select-String -Path $log -Pattern "plugin disabled" -SimpleMatch -Quiet)
  $stats = (Select-String -Path $log -Pattern "moe-cache: graphs=" -SimpleMatch | Select-Object -Last 1 | ForEach-Object { $_.Line })
  return @{ texts = $texts; tps = $tps; active = $active; cache = $cache; disabled = $disabled; stats = $stats }
}

Write-Host "1/2 stock server"; $a = Run-One "stock" $false
Write-Host "2/2 server with the plugin"; $b = Run-One "plugin" $true
Write-Host "---- plugin log lines (must show that the plugin loaded):"; Write-Host $b.active
if (-not $b.active) { Write-Host "FAIL: no 'moe-cache:' line in the plugin run, the DLL was not loaded (check API: $out\plugin.err.txt)"; exit 2 }
if ($b.disabled -or -not $b.cache) { Write-Host "FAIL: the plugin loaded but the cache was not on (no 'moe-cache: cache mode' line, or 'plugin disabled'): this run proves nothing"; exit 2 }
if ($b.stats) { Write-Host "cache: $($b.stats)" }
if (-not ($a.texts | Where-Object { $_.Length -gt 0 })) { Write-Host "FAIL: every stock answer is empty, so the comparison exercised no decoding"; exit 2 }
$same = 0; for ($i = 0; $i -lt $prompts.Count; $i++) { if ($a.texts[$i] -ceq $b.texts[$i]) { $same++ } }
Write-Host ("decode tok/s stock : " + ($a.tps -join ", ")); Write-Host ("decode tok/s plugin: " + ($b.tps -join ", "))
Write-Host "identical prompts: $same/$($prompts.Count)"
if ($same -eq $prompts.Count) { Write-Host "PASS" } else { Write-Host "FAIL"; exit 1 }
Write-Host "Please send back: $out\plugin.err.txt, the lines above, and your llama.cpp release name."
