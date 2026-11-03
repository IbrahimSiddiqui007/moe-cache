# moe-cache on Windows: UNTESTED port. This script answers one question: does the plugin load into an official Windows llama-server and give the same tokens as stock?
#   powershell -ExecutionPolicy Bypass -File windows_test.ps1 -LlamaDir C:\llama -Model D:\models\model.gguf -Dll C:\moe\ggml-moe-cache-api3.dll
# -LlamaDir : folder with llama-server.exe (an official release zip, CUDA build if you have an NVIDIA GPU)
# -Model    : a GGUF that FITS IN RAM for the first test. The script refuses a model bigger than the memory cap (below) unless -AllowBigModel is given
# -Dll      : ggml-moe-cache-api2.dll (llama.cpp 7fe450e19 / ggml 0.25.x) or ggml-moe-cache-api3.dll (llama.cpp 23b0202a1 and newer ggml 0.26.x)
# -CacheGiB : expert cache size for the plugin run (default 0.5: far below the model, so experts are evicted and read again; "auto" sizes it from RAM)
# -MemLimitGB : cap on the memory llama-server may commit, in decimal GB (WINDOWS-082). Default 0 = the launcher's rule: RAM minus max(4 GiB, 12.5 %), i.e. 28 GiB
#              on a 32 GiB PC (MOE_CACHE_MAX_RAM_GB overrides, as in the launcher). It is a Windows job object around each llama-server this script starts, so a
#              runaway process is stopped by Windows (its allocation fails) instead of freezing the PC. -1 = no limit. The job limit counts committed memory, not
#              the file cache behind a memory-mapped model.
# -AllowBigModel : run a model whose file is bigger than that cap anyway (it will page; close other programs)
# -Extra    : more llama-server arguments for both runs, e.g. "--no-repack" (CPU-only machines: repacked weights switch the expert executor off)
param(
  [Parameter(Mandatory = $true)][string]$LlamaDir,
  [Parameter(Mandatory = $true)][string]$Model,
  [Parameter(Mandatory = $true)][string]$Dll,
  [int]$Port = 8099, [int]$Tokens = 32, [int]$Ctx = 4096, [int]$NCpuMoe = 999, [string]$CacheGiB = "0.5", [string]$Extra = "", [double]$MemLimitGB = 0, [switch]$AllowBigModel
)
$ErrorActionPreference = "Stop"
$srv = Join-Path $LlamaDir "llama-server.exe"
if (-not (Test-Path $srv)) { throw "llama-server.exe not found in $LlamaDir" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
if (-not (Test-Path $Dll)) { throw "plugin dll not found: $Dll" }
# ---- memory cap (WINDOWS-082): the same rule as the launcher's mem_cap(): RAM minus max(4 GiB, 12.5 %); MOE_CACHE_MAX_RAM_GB overrides (0 = no cap)
$ramBytes = [double](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$capBytes = $ramBytes - [math]::Max(4GB, 0.125 * $ramBytes)
if ($env:MOE_CACHE_MAX_RAM_GB) { $o = 0.0; if ([double]::TryParse($env:MOE_CACHE_MAX_RAM_GB, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$o) -and $o -ge 0) { $capBytes = $o * 1e9 } }
if ($MemLimitGB -gt 0) { $capBytes = $MemLimitGB * 1e9 } elseif ($MemLimitGB -lt 0) { $capBytes = 0 }
$modelBytes = (Get-Item $Model).Length
if ($capBytes -gt 0 -and $modelBytes -gt $capBytes -and -not $AllowBigModel) {
  throw ("the model file is {0:N1} GB but the memory cap is {1:N1} GB (RAM {2:N1} GB minus max(4 GiB, 12.5 %)). A model bigger than the cap can freeze this PC: use a smaller model for this identity test, or pass -AllowBigModel (or -MemLimitGB N) if you accept that" -f ($modelBytes / 1e9), ($capBytes / 1e9), ($ramBytes / 1e9))
}
if ($capBytes -gt 0) { Write-Host ("memory cap for llama-server: {0:N1} GB (job object; RAM {1:N1} GB, model {2:N1} GB)" -f ($capBytes / 1e9), ($ramBytes / 1e9), ($modelBytes / 1e9)) } else { Write-Host "memory cap: none (-MemLimitGB -1 or MOE_CACHE_MAX_RAM_GB=0)" }
if ($capBytes -gt 0) {
  # a Windows job object with JOB_OBJECT_LIMIT_JOB_MEMORY around each llama-server (the Python launcher does the same through ctypes, bin/moe_cache.py)
  if (-not ("MoeJob" -as [type])) {
    Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public static class MoeJob {
  [StructLayout(LayoutKind.Sequential)] struct BASIC { public long PerProcessUserTimeLimit, PerJobUserTimeLimit; public uint LimitFlags; public UIntPtr MinWs, MaxWs; public uint ActiveProcessLimit; public UIntPtr Affinity; public uint Priority, Sched; }
  [StructLayout(LayoutKind.Sequential)] struct IOC { public ulong a, b, c, d, e, f; }
  [StructLayout(LayoutKind.Sequential)] struct EXT { public BASIC Basic; public IOC Io; public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed; }
  [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr CreateJobObjectW(IntPtr a, string n);
  [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetInformationJobObject(IntPtr j, int c, ref EXT i, int n);
  [DllImport("kernel32.dll", SetLastError = true)] static extern bool QueryInformationJobObject(IntPtr j, int c, ref EXT i, int n, IntPtr r);
  [DllImport("kernel32.dll", SetLastError = true)] static extern bool AssignProcessToJobObject(IntPtr j, IntPtr p);
  [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
  // Limits the process (and its children) to `bytes` of committed memory and kills it when the job handle closes. Returns the limit read back, 0 on failure.
  public static ulong Assign(IntPtr process, ulong bytes, out IntPtr job) {
    job = CreateJobObjectW(IntPtr.Zero, null);
    if (job == IntPtr.Zero) return 0;
    EXT e = new EXT(); e.Basic.LimitFlags = 0x2000 | 0x200; e.JobMemoryLimit = new UIntPtr(bytes);   // KILL_ON_JOB_CLOSE | JOB_MEMORY
    if (!SetInformationJobObject(job, 9, ref e, Marshal.SizeOf(typeof(EXT)))) return 0;
    if (!AssignProcessToJobObject(job, process)) return 0;
    EXT back = new EXT();
    if (!QueryInformationJobObject(job, 9, ref back, Marshal.SizeOf(typeof(EXT)), IntPtr.Zero)) return 0;
    return back.JobMemoryLimit.ToUInt64();
  }
  public static void Close(IntPtr job) { if (job != IntPtr.Zero) CloseHandle(job); }
}
"@
  }
}
$prompts = @("Explain what a hash table is in two sentences.", "Write a Python function that reverses a string.",
             "What is the capital of Australia and why was it chosen?", "List three uses of a binary search.")
$out = Join-Path $env:TEMP "moe-cache-test"; New-Item -ItemType Directory -Force $out | Out-Null

# a server left on the fixed port by an earlier run would answer /health and this run would test THAT process (AUD-056): stop if anything holds the port
function Assert-PortFree {
  for ($i = 0; $i -lt 10; $i++) {
    try { $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port); $l.Start(); $l.Stop(); return } catch { Start-Sleep -Seconds 1 }
  }
  throw "port $Port is already in use (a llama-server from an earlier run? stop it, or pass -Port): this test would talk to that process"
}
$envNames = "GGML_BACKEND_PATH", "MOE_CACHE_GGUF", "MOE_CACHE_STATS", "MOE_CACHE_STATS_SIGNAL", "MOE_CACHE_SIZE_GIB"
$savedEnv = @{}; foreach ($v in $envNames) { $savedEnv[$v] = [Environment]::GetEnvironmentVariable($v, "Process") }

function Run-One([string]$label, [bool]$plugin) {
  Assert-PortFree
  $log = Join-Path $out "$label.err.txt"; $std = Join-Path $out "$label.out.txt"
  # MOE_CACHE_SIZE_GIB turns the cache on: without it the plugin only passes the weights through and the test would prove nothing
  if ($plugin) { $env:GGML_BACKEND_PATH = (Resolve-Path $Dll).Path; $env:MOE_CACHE_GGUF = (Resolve-Path $Model).Path; $env:MOE_CACHE_STATS = "1"; $env:MOE_CACHE_STATS_SIGNAL = "1"; $env:MOE_CACHE_SIZE_GIB = $CacheGiB }
  else { foreach ($v in $envNames) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue } }
  $sargs = @("-m", "`"$Model`"", "-ngl", "99", "--n-cpu-moe", "$NCpuMoe", "-c", "$Ctx", "-fa", "on", "--no-warmup", "--port", "$Port", "--seed", "42")
  if ($Extra) { $sargs += ($Extra -split " " | Where-Object { $_ }) }
  $p = Start-Process -FilePath $srv -ArgumentList $sargs -PassThru -RedirectStandardError $log -RedirectStandardOutput $std -WindowStyle Hidden
  $job = [IntPtr]::Zero
  if ($capBytes -gt 0) {
    $lim = [MoeJob]::Assign($p.Handle, [uint64]$capBytes, [ref]$job)
    if ($lim -eq 0) { Write-Host "warning: the memory cap could not be set on $label (another job object around this shell? the run continues without it)" }
    else { Write-Host ("  {0}: job memory limit {1:N1} GB read back" -f $label, ($lim / 1e9)) }
  }
  try {
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
    if ($plugin) {
      # the server is killed below, so the plugin's summary at exit never appears: ask for a snapshot first (named event, see src/mc_state.inl)
      try { $ev = [System.Threading.EventWaitHandle]::OpenExisting("Local\moe-cache-stats-$($p.Id)"); [void]$ev.Set(); Start-Sleep -Seconds 2 } catch {}
    }
  } finally {
    # any exception above (health never came, a request failed) must not leave llama-server on the port
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    if ($job -ne [IntPtr]::Zero) { [MoeJob]::Close($job) }
  }
  Start-Sleep -Seconds 3      # the finally above stopped the server; give Windows a moment to release the port and the log files
  $active = (Select-String -Path $log -Pattern "moe-cache:" -SimpleMatch | Select-Object -First 6 | ForEach-Object { $_.Line }) -join "`n"
  $cache = [bool](Select-String -Path $log -Pattern "moe-cache: cache mode" -SimpleMatch -Quiet)
  $disabled = [bool](Select-String -Path $log -Pattern "plugin disabled" -SimpleMatch -Quiet)
  $stats = (Select-String -Path $log -Pattern "moe-cache-stats {" -SimpleMatch | Select-Object -Last 1 | ForEach-Object {
    $l = $_.Line; $f = @{}
    foreach ($k in "hits", "misses", "read_bytes", "evictions", "exec_ops", "exec_checked", "exec_mismatch") {
      if ($l -match ('"' + $k + '":(\d+)')) { $f[$k] = [int64]$Matches[1] } else { $f[$k] = "?" }
    }
    $rd = if ($f.read_bytes -is [int64]) { [math]::Round($f.read_bytes / 1MB) } else { "?" }
    "hits $($f.hits), misses $($f.misses), read $rd MiB, evictions $($f.evictions), executor ops $($f.exec_ops), checked $($f.exec_checked), mismatches $($f.exec_mismatch)" })
  return @{ texts = $texts; tps = $tps; active = $active; cache = $cache; disabled = $disabled; stats = $stats }
}

try {
  Write-Host "1/2 stock server"; $a = Run-One "stock" $false
  Write-Host "2/2 server with the plugin"; $b = Run-One "plugin" $true
} finally {
  foreach ($v in $envNames) { [Environment]::SetEnvironmentVariable($v, $savedEnv[$v], "Process") }      # the caller's session gets its own values back
}
Write-Host "---- plugin log lines (must show that the plugin loaded):"; Write-Host $b.active
if (-not $b.active) {
  $lf = Select-String -Path (Join-Path $out "plugin.err.txt") -Pattern "failed to load" -SimpleMatch | Select-Object -First 1 | ForEach-Object { $_.Line }
  if ($lf) { Write-Host "FAIL: llama.cpp could not load the DLL ($lf). An empty reason on Windows usually means a missing dependency DLL; check with: dumpbin /dependents (or objdump -p) on the DLL" }
  else { Write-Host "FAIL: no 'moe-cache:' line in the plugin run, the DLL was not loaded (wrong ggml API? see $out\plugin.err.txt)" }
  exit 2
}
if ($b.disabled -or -not $b.cache) { Write-Host "FAIL: the plugin loaded but the cache was not on (no 'moe-cache: cache mode' line, or 'plugin disabled'): this run proves nothing"; exit 2 }
if ($b.stats) { Write-Host "cache: $($b.stats)" }
if (-not ($a.texts | Where-Object { $_.Length -gt 0 })) { Write-Host "FAIL: every stock answer is empty, so the comparison exercised no decoding"; exit 2 }
$same = 0; for ($i = 0; $i -lt $prompts.Count; $i++) { if ($a.texts[$i] -ceq $b.texts[$i]) { $same++ } }
Write-Host ("decode tok/s stock : " + ($a.tps -join ", ")); Write-Host ("decode tok/s plugin: " + ($b.tps -join ", "))
Write-Host "identical prompts: $same/$($prompts.Count)"
if ($same -eq $prompts.Count) { Write-Host "PASS" } else { Write-Host "FAIL"; exit 1 }
Write-Host "Please send back: $out\plugin.err.txt, the lines above, and your llama.cpp release name."
