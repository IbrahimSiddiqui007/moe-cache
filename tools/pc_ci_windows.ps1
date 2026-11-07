# The Windows half of tools/pc_ci.sh: runs the steps of the `windows` job of .github/workflows/windows.yml on this PC, for one commit that
# pc_ci.sh has already copied here (-Work) together with the DLL, test programs and kit it cross-built on the laptop (-BuildDir, -KitZip).
# Windows PowerShell 5.1 compatible. Starts only from pc_ci.sh, which has checked that no PC chain runs.
#
# Groups:  tests  = no model is loaded (default)
#          model  = starts llama-server with the small Granite model (-Model switch); the whole group is limited to -ModelMinutes minutes
# Left to GitHub on purpose (they change the owner's real PATH / Start menu / Startup folder / HKLM): the `installer` job, `--shortcut`, `--install-autostart`,
# the crash-dump registry key. Writes only below -Work and -CacheDir.
param(
  [Parameter(Mandatory = $true)][string]$Work,
  [Parameter(Mandatory = $true)][string]$BuildDir,
  [Parameter(Mandatory = $true)][string]$KitZip,
  [Parameter(Mandatory = $true)][string]$CacheDir,
  [switch]$Model,
  [int]$ModelMinutes = 15
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# pinned exactly as in windows.yml (LLAMA_RELEASE, RELEASE-008) and the Granite file of the workflow
$LlamaRelease = "b12064"
$LlamaZipName = "llama-$LlamaRelease-bin-win-cpu-x64.zip"
$LlamaZipSha = "c1ced7358438e19c2af6debefe188280ba39ac0242352719cd30d931ca8eec73"
$GraniteUrl = "https://huggingface.co/bartowski/granite-3.1-1b-a400m-instruct-GGUF/resolve/main/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf"
$GraniteSha = "3a2ec1c2a78cb29d901e29bbf5162dcd03381e13803d2cbdcff838d4d08142eb"

# the guard of the next run reads this: a live pid here means a run is active
Set-Content -Encoding ascii (Join-Path (Split-Path (Split-Path $Work)) "running.lock") $PID
$RunTemp = Join-Path $Work "runner_temp"
$TestTemp = Join-Path $Work "tmp"
New-Item -ItemType Directory -Force $RunTemp, $TestTemp | Out-Null
$LogFile = Join-Path $Work "pc_ci.log"
$ResultFile = Join-Path $Work "results.tsv"
Set-Content -Encoding ascii $ResultFile "step`tgroup`tresult`tseconds"
Set-Content -Encoding ascii $LogFile ""
$env:TEMP = $TestTemp          # windows_test.ps1 and the tests write below the work folder, not the owner's %TEMP%
$env:TMP = $TestTemp

$Granite = Join-Path $CacheDir "granite.gguf"
$LlamaDir = Join-Path $CacheDir "llama-$LlamaRelease"
$LlamaServer = Join-Path $LlamaDir "llama-server.exe"
$PluginDll = Join-Path $BuildDir "ggml-moe-cache.dll"
$ModelWatchdog = $null
$ModelStart = $null
$ModelSeconds = $ModelMinutes * 60
$AnyFail = $false

# environment variables a step may set; restored after every step so one step cannot leak into the next
$EnvNames = @("MOE_CACHE_LIB", "MOE_CACHE_EXEC", "MOE_CACHE_EXEC_CHECK", "MOE_CACHE_WIN_VIEW", "XDG_CACHE_HOME", "APPDATA_UNUSED", "LLAMA_SERVER", "REQUIRE_NODE")

function Write-CiLog([string]$Text) { Add-Content -Encoding utf8 -Path $LogFile -Value $Text }

function Assert-ExitCode([string]$What) {
  if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

function Assert-LogMatch([string]$Path, [string]$Pattern, [string]$Message) {
  if (-not (Select-String -Path $Path -Pattern $Pattern -Quiet)) { throw $Message }
}

function Start-ModelWatchdog {
  # a watchdog for the model group, started with its first step: after -ModelMinutes it stops any llama-server of ours (by path, never by name alone)
  $script:ModelWatchdog = Start-Job -ArgumentList $ModelSeconds, $CacheDir, $Work -ScriptBlock {
    param($Seconds, $Cache, $WorkDir)
    Start-Sleep -Seconds ($Seconds + 60)
    foreach ($server in (Get-Process -Name "llama-server" -ErrorAction SilentlyContinue)) {
      if ($server.Path -and ($server.Path.StartsWith($Cache, [StringComparison]::OrdinalIgnoreCase) -or $server.Path.StartsWith($WorkDir, [StringComparison]::OrdinalIgnoreCase))) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
      }
    }
  }
}

function Invoke-CiStep {
  param([string]$Name, [string]$Group, [scriptblock]$Body)
  if ($Group -eq "model" -and -not $Model) {
    Add-Content -Encoding ascii $ResultFile "$Name`t$Group`tskipped (no -Model)`t0"
    return
  }
  if ($Group -eq "model") {
    if (-not $ModelStart) { $script:ModelStart = Get-Date; Start-ModelWatchdog }
    if (((Get-Date) - $ModelStart).TotalSeconds -gt $ModelSeconds) {
      Add-Content -Encoding ascii $ResultFile "$Name`t$Group`tFAIL (model time limit $ModelMinutes min)`t0"
      $script:AnyFail = $true
      return
    }
  }
  $saved = @{}
  foreach ($n in $EnvNames) { $saved[$n] = [Environment]::GetEnvironmentVariable($n, "Process") }
  Write-CiLog "=== $Name ($Group) $(Get-Date -Format s)"
  Write-Host "[pc_ci] $Name"
  $t0 = Get-Date
  $result = "pass"
  try {
    Push-Location $Work
    & $Body 2>&1 | ForEach-Object { Write-CiLog ("$_") }
  } catch {
    $result = "FAIL"
    Write-CiLog "FAILED: $($_.Exception.Message)"
    $script:AnyFail = $true
  } finally {
    Pop-Location
    foreach ($n in $EnvNames) { [Environment]::SetEnvironmentVariable($n, $saved[$n], "Process") }
  }
  $sec = [int]((Get-Date) - $t0).TotalSeconds
  Add-Content -Encoding ascii $ResultFile "$Name`t$Group`t$result`t$sec"
}

# ---------------------------------------------------------------- tests group (no model is loaded)
Invoke-CiStep "kernel_check (Q2_0 kernels as built for Windows)" "tests" {
  & (Join-Path $BuildDir "kernel_check.exe") | Tee-Object (Join-Path $RunTemp "kernel_check.txt")
  if ($LASTEXITCODE -ne 0) { throw "kernel_check failed ($LASTEXITCODE)" }
  Assert-LogMatch (Join-Path $RunTemp "kernel_check.txt") "permuted-activation kernel: 0 mismatches" "permuted kernel mismatches"
}

Invoke-CiStep "platform layer natively" "tests" {
  Push-Location $RunTemp
  try {
    foreach ($exe in "platform_test.exe", "open_hint_test.exe", "file_view_load_test.exe") {
      & (Join-Path $BuildDir $exe)
      if ($LASTEXITCODE -ne 0) { throw "$exe failed ($LASTEXITCODE)" }
    }
  } finally { Pop-Location }
}

Invoke-CiStep "fetch llama.cpp $LlamaRelease (CPU) and Granite (sha256-checked, cached)" "tests" {
  New-Item -ItemType Directory -Force $CacheDir | Out-Null
  $zip = Join-Path $CacheDir $LlamaZipName
  if (-not (Test-Path $zip)) {
    Invoke-WebRequest -UseBasicParsing "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaRelease/$LlamaZipName" -OutFile $zip
  }
  $zipHash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
  if ($zipHash -ne $LlamaZipSha) { throw "${LlamaZipName}: wrong sha256 $zipHash (file kept in $CacheDir for a look)" }
  if (-not (Test-Path $LlamaServer)) {
    $tmpExtract = Join-Path $CacheDir "llama-$LlamaRelease.extracting"
    if (Test-Path $tmpExtract) { Move-Item $tmpExtract (Join-Path $CacheDir ("llama-old-" + (Get-Date -Format yyyyMMddHHmmss))) }
    Expand-Archive $zip -DestinationPath $tmpExtract
    $found = Get-ChildItem -Recurse $tmpExtract -Filter llama-server.exe | Select-Object -First 1
    if (-not $found) { throw "no llama-server.exe in $LlamaZipName" }
    Move-Item $found.DirectoryName $LlamaDir
  }
  if (-not (Test-Path $Granite)) {
    $part = "$Granite.part"
    Invoke-WebRequest -UseBasicParsing $GraniteUrl -OutFile $part
    Move-Item $part $Granite
  }
  $gh = (Get-FileHash $Granite -Algorithm SHA256).Hash.ToLower()
  if ($gh -ne $GraniteSha) { throw "granite.gguf: wrong sha256 $gh" }
  "llama: $LlamaDir"; "granite: $Granite"
}

Invoke-CiStep "launcher without a model run (dry runs, stock without plugin, core detection)" "tests" {
  $env:MOE_CACHE_LIB = $PluginDll
  python bin/moe_cache.py server $Granite --llama-server $LlamaServer --ram 4 --always-plugin --gpu-cache 1000 --dry-run 2>&1 | Tee-Object (Join-Path $RunTemp "launcher_dry.txt")
  Assert-LogMatch (Join-Path $RunTemp "launcher_dry.txt") "GGML_BACKEND_PATH" "dry run: no GGML_BACKEND_PATH"
  if (Select-String (Join-Path $RunTemp "launcher_dry.txt") -Pattern "moe-cache-mib" -SimpleMatch -Quiet) { throw "dry run: --gpu-cache must be dropped on path loading" }
  $env:MOE_CACHE_LIB = Join-Path $Work "no-such-dir\ggml-moe-cache.dll"
  $stockOut = Join-Path $RunTemp "launcher_stock_nolib.txt"
  python bin/moe_cache.py server $Granite --llama-server $LlamaServer --ram 8 --dry-run 2>&1 | Out-File $stockOut
  $rc = $LASTEXITCODE
  $env:MOE_CACHE_LIB = $PluginDll
  if ($rc -ne 0 -or -not (Select-String $stockOut -Pattern "running stock llama-server" -SimpleMatch -Quiet) -or (Select-String $stockOut -Pattern "GGML_BACKEND_PATH" -SimpleMatch -Quiet)) { Get-Content $stockOut; throw "a model that fits must run stock without a plugin build" }
  $pyCode = "import importlib.util as u, sys; sys.argv=['x']; s=u.spec_from_file_location('m','bin/moe_cache.py'); m=u.module_from_spec(s); s.loader.exec_module(m); c=m.win_cores(); print('cores (efficiency class, cpus):', c); print('hybrid masks:', m.hybrid_masks(c)); sys.exit(0 if c else 1)"
  python -c $pyCode
  if ($LASTEXITCODE -ne 0) { throw "win_cores: no CPU cores found" }
}

Invoke-CiStep "python tests (picker, connect, catalogue, parts, multigpu, node, control page)" "tests" {
  $env:MOE_CACHE_LIB = $PluginDll
  foreach ($t in "picker_test", "connect_agents_test", "connect_check_test", "models_catalogue_test", "model_parts_test", "multigpu_test", "require_node_test") {
    python "tests/$t.py"
    Assert-ExitCode $t
  }
  $env:REQUIRE_NODE = "1"
  python tests/ui_test.py
  Assert-ExitCode "ui_test"
}

Invoke-CiStep "keep-awake, doctor, desk margin, oom retry, diffusion tests" "tests" {
  foreach ($t in "awake_test", "doctor_test", "desk_margin_test", "oom_retry_test", "diffusion_test") {
    python "tests/$t.py"
    Assert-ExitCode $t
  }
}

Invoke-CiStep "on-demand (fake server), warm session, security tests" "tests" {
  $env:MOE_CACHE_LIB = $PluginDll
  foreach ($t in "on_demand_test", "warm_test", "security_test") {
    python "tests/$t.py"
    Assert-ExitCode $t
  }
}

Invoke-CiStep "bench report tests (no server)" "tests" {
  python tests/bench_report_test.py
  Assert-ExitCode "bench_report_test"
  python tests/bench_record_test.py
  Assert-ExitCode "bench_record_test"
}

Invoke-CiStep "one-click start (moe-cache-ui.bat finds Python, DLL and llama-server; page answers)" "tests" {
  $dll3 = Join-Path $Work "build\win-api3"
  New-Item -ItemType Directory -Force $dll3 | Out-Null
  Copy-Item $PluginDll (Join-Path $dll3 "ggml-moe-cache.dll") -Force
  $env:LLAMA_SERVER = $LlamaServer
  [Environment]::SetEnvironmentVariable("MOE_CACHE_LIB", $null, "Process")
  $batOut = Join-Path $RunTemp "ui_bat.out"; $batErr = Join-Path $RunTemp "ui_bat.err"
  $t0 = Get-Date
  $p = Start-Process cmd -ArgumentList "/c", "bin\moe-cache-ui.bat", "--no-browser", "--ui-port", "8078" -RedirectStandardOutput $batOut -RedirectStandardError $batErr -PassThru -NoNewWindow
  $tok = $null; $sys = $null
  for ($n = 0; $n -lt 480 -and -not $tok -and -not $p.HasExited; $n++) {
    Start-Sleep -Milliseconds 500
    $m = Select-String $batOut -Pattern '\?t=([A-Za-z0-9_-]+)' -ErrorAction SilentlyContinue
    if ($m) { $tok = $m.Matches[0].Groups[1].Value }
  }
  for ($n = 0; $n -lt 120 -and $tok -and -not $sys; $n++) {
    try { $sys = Invoke-RestMethod http://127.0.0.1:8078/api/system -Headers @{ "X-MoeCache-Token" = $tok } } catch { Start-Sleep -Milliseconds 500 }
  }
  "page after $([int]((Get-Date) - $t0).TotalSeconds) s"
  taskkill /T /F /PID $p.Id | Out-Null
  Get-Content $batOut, $batErr
  if (-not $sys) { throw "the page did not come up" }
  if (-not ($sys.plugin -like "*build\win-api3\ggml-moe-cache.dll")) { throw "the .bat did not find the DLL in build\win-api3" }
  if ($sys.llama_server -ne $env:LLAMA_SERVER) { throw "the page does not use LLAMA_SERVER" }
  # the CI step's --shortcut part is left to GitHub: it writes a Start-menu entry
}

# ---------------------------------------------------------------- model group (Granite only)
Invoke-CiStep "kit as a user gets it: unzip, check --run, bundled llama, on-demand, control page, pick" "model" {
  $kitRoot = Join-Path $RunTemp "kit"
  if (Test-Path $kitRoot) { Move-Item $kitRoot (Join-Path $RunTemp ("kit-old-" + (Get-Date -Format HHmmss))) }
  Expand-Archive $KitZip -DestinationPath $kitRoot
  $k = (Get-ChildItem $kitRoot -Directory | Select-Object -First 1).FullName
  [Environment]::SetEnvironmentVariable("MOE_CACHE_LIB", $null, "Process")
  Push-Location $k
  try {
    $checkOut = Join-Path $RunTemp "kit_check.txt"
    python bin\moe_cache.py check $Granite --llama-server $LlamaServer --run | Tee-Object $checkOut
    if (-not (Select-String $checkOut -Pattern "ggml-moe-cache-api3.dll" -SimpleMatch -Quiet)) { throw "kit: check did not pick the kit's API 3 DLL" }
    if (-not (Test-Path (Join-Path $k "llama\llama-server.exe"))) { throw "kit: no bundled llama\llama-server.exe" }
    Assert-LogMatch $checkOut "plugin loads .*yes" "kit: the plugin does not load"
    Assert-LogMatch $checkOut "cache runs .*yes" "kit: the cache did not compute"
    $env:XDG_CACHE_HOME = Join-Path $RunTemp "kit_xdg"
    $ui = (python bin\moe_cache.py ui --help 2>&1) -join "`n"; $uirc = $LASTEXITCODE
    $pk = (python bin\moe_cache.py pick --json --use chat --llama-server $LlamaServer 2>&1) -join "`n"; $pkrc = $LASTEXITCODE
    if ($uirc -ne 0 -or $ui -notmatch "ui-port") { throw "kit: the control page does not start from the kit ($uirc)" }
    if ($pkrc -ne 0 -or $pk -notmatch '"recommendations"') { throw "kit: pick does not run from the kit ($pkrc)" }
    $od = (python bin\moe_cache.py on-demand --status 2>&1) -join "`n"
    if ($od -notmatch "No model chosen yet") { throw "kit: on-demand does not run from the kit: $od" }
  } finally { Pop-Location }
}

Invoke-CiStep "bench-report from the kit (real servers, Granite, warm)" "model" {
  $kitDirs = Get-ChildItem (Join-Path $RunTemp "kit") -Directory
  if (-not $kitDirs) { throw "the kit step did not unpack the kit" }
  $k = ($kitDirs | Select-Object -First 1).FullName
  [Environment]::SetEnvironmentVariable("MOE_CACHE_LIB", $null, "Process")
  $benchOut = Join-Path $RunTemp "bench out"
  Push-Location $k
  try {
    python bin\moe_cache.py bench-report $Granite --llama-server $LlamaServer --force-plugin --warm --tokens 16 --prompt-tokens 300 --ctx 2048 --out $benchOut 2>&1 | Tee-Object (Join-Path $RunTemp "bench_report.txt")
    $rc = $LASTEXITCODE
  } finally { Pop-Location }
  $md = Get-Content (Get-ChildItem "$benchOut\*.md" | Select-Object -First 1).FullName -Raw
  if ($rc -ne 0) { throw "bench-report exit $rc" }
  if ($md -notmatch "Checks: tokens IDENTICAL; plugin active yes") { throw "bench-report: tokens not identical or the plugin was not active" }
  if ($md -notmatch "\*\*Backend\*\* \| \*\*CPU only\*\*") { throw "bench-report: backend line wrong for the CPU release" }
}

Invoke-CiStep "launcher check --run and real start with --ram and --agent (port 8090)" "model" {
  $env:MOE_CACHE_LIB = $PluginDll
  python bin/moe_cache.py check $Granite --llama-server $LlamaServer --run | Tee-Object (Join-Path $RunTemp "launcher_check.txt")
  Assert-LogMatch (Join-Path $RunTemp "launcher_check.txt") "plugin loads .*yes" "check: the plugin does not load"
  Assert-LogMatch (Join-Path $RunTemp "launcher_check.txt") "cache runs .*yes" "check --run: the cache did not compute"
  $errFile = Join-Path $RunTemp "launcher_real.err"; $outFile = Join-Path $RunTemp "launcher_real.out"
  $p = Start-Process python -ArgumentList "bin/moe_cache.py", "server", $Granite, "--llama-server", $LlamaServer, "--ram", "3", "--always-plugin", "--agent", "--ctx", "4096", "--port", "8090" -RedirectStandardError $errFile -RedirectStandardOutput $outFile -PassThru -NoNewWindow
  $ok = $false
  for ($n = 0; $n -lt 300; $n++) { try { if ((Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8090/health).StatusCode -eq 200) { $ok = $true; break } } catch {} ; Start-Sleep 1 }
  $chatBody = '{"model":"granite","messages":[{"role":"user","content":"Say hi"}],"max_tokens":8,"temperature":0}'
  $key = (python bin/moe_cache.py key | Out-String).Trim()
  if (-not $key) { throw "agent mode server but moe_cache.py key printed nothing" }
  if ($ok) { (Invoke-WebRequest -UseBasicParsing -Method Post -ContentType application/json -Headers @{ Authorization = "Bearer $key" } -Body $chatBody http://127.0.0.1:8090/v1/chat/completions).Content }
  python bin/moe_cache.py connect opencode --url http://127.0.0.1:8090 --write --config (Join-Path $RunTemp "opencode.json")
  & taskkill /T /F /PID $p.Id | Out-Null
  Start-Sleep 3
  if (-not $ok) { Get-Content $errFile -Tail 30; throw "the launcher's server never became ready" }
  if (Select-String $errFile -Pattern "Windows job memory limit" -SimpleMatch -Quiet) { throw "--ram set a job memory limit although the file-view cache is on" }
  $env:MOE_CACHE_WIN_VIEW = "0"
  python bin/moe_cache.py server $Granite --llama-server $LlamaServer --ram 3 --always-plugin --dry-run 2>&1 | Out-File (Join-Path $RunTemp "launcher_joblimit.txt")
  if (-not (Select-String (Join-Path $RunTemp "launcher_joblimit.txt") -Pattern "would set a Windows job memory limit of 3.0 GB" -SimpleMatch -Quiet)) { throw "MOE_CACHE_WIN_VIEW=0 --ram did not plan a job memory limit" }
  if (-not (Select-String $errFile -Pattern "cache mode" -SimpleMatch -Quiet)) { throw "the plugin's cache was not on" }
  if (-not (Select-String $errFile, $outFile -Pattern "Windows memory: RAM .* GB free; commit .* GB free of" -Quiet)) { throw "the launcher did not report Windows commit headroom (XA-3)" }
  if (-not (Select-String (Join-Path $RunTemp "opencode.json") -Pattern '"tool_call": true' -SimpleMatch -Quiet)) { throw "connect opencode wrote no model entry" }
}

Invoke-CiStep "control page with the model (ui_live)" "model" {
  $env:MOE_CACHE_LIB = $PluginDll
  python tests/ui_live.py $Granite --llama-server $LlamaServer
  Assert-ExitCode "ui_live"
}

Invoke-CiStep "on-demand live with Granite" "model" {
  $env:MOE_CACHE_LIB = $PluginDll
  python tests/on_demand_live.py $Granite --llama-server $LlamaServer
  Assert-ExitCode "on_demand_live"
  # the CI step's --install-autostart part is left to GitHub: it writes to the Startup folder
}

function Invoke-WindowsTestScript {
  param([string]$Tokens, [string]$Extra)
  $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "tools/windows_test.ps1", "-LlamaDir", $LlamaDir, "-Model", $Granite, "-Dll", $PluginDll, "-CacheGiB", "0.2", "-Tokens", $Tokens, "-Ctx", "2048")
  if ($Extra) { $argList += @("-Extra", $Extra) }
  & powershell @argList
}

Invoke-CiStep "identity: cache on, forced to evict (0.2 GiB)" "model" {
  Invoke-WindowsTestScript "24" ""
  if ($LASTEXITCODE -ne 0) { throw "windows_test.ps1 failed ($LASTEXITCODE)" }
}

Invoke-CiStep "identity: private-memory cache (MOE_CACHE_WIN_VIEW=0)" "model" {
  $env:MOE_CACHE_WIN_VIEW = "0"
  Invoke-WindowsTestScript "24" ""
  if ($LASTEXITCODE -ne 0) { throw "windows_test.ps1 (private memory) failed ($LASTEXITCODE)" }
  if (Select-String (Join-Path $env:TEMP "moe-cache-test\plugin.err.txt") -Pattern "read-only views of the model file" -Quiet) { throw "MOE_CACHE_WIN_VIEW=0 did not turn the file view off" }
}

Invoke-CiStep "identity: file-view cache, executor byte-checked" "model" {
  $env:MOE_CACHE_EXEC = "2"; $env:MOE_CACHE_EXEC_CHECK = "1"
  Invoke-WindowsTestScript "48" "--no-repack"
  if ($LASTEXITCODE -ne 0) { throw "windows_test.ps1 (file view) failed ($LASTEXITCODE)" }
  $log = Join-Path $env:TEMP "moe-cache-test\plugin.err.txt"
  if (-not (Select-String $log -Pattern "experts are kept in read-only views of the model file" -Quiet)) { throw "the file-view cache was not used" }
  $ev = @(Select-String $log -Pattern '"evictions":(\d+)' | ForEach-Object { [int64]$_.Matches[0].Groups[1].Value })
  if ($ev.Count -eq 0 -or ($ev | Measure-Object -Maximum).Maximum -eq 0) { throw "no evictions: the 0.2 GiB cache did not cycle experts through the views (stats lines found: $($ev.Count))" }
  $x = Select-String $log -Pattern '"exec_checked":(\d+),"exec_mismatch":(\d+)' | Select-Object -Last 1
  if (-not $x -or [int64]$x.Matches[0].Groups[1].Value -eq 0 -or [int64]$x.Matches[0].Groups[2].Value -ne 0) { throw "executor byte check on the views: none checked or mismatches ($($x.Line))" }
}

Invoke-CiStep "identity: executor forced on and byte-checked" "model" {
  $env:MOE_CACHE_EXEC = "2"; $env:MOE_CACHE_EXEC_CHECK = "1"
  Invoke-WindowsTestScript "24" "--no-repack"
  if ($LASTEXITCODE -ne 0) { throw "windows_test.ps1 (executor) failed ($LASTEXITCODE)" }
  $log = Join-Path $env:TEMP "moe-cache-test\plugin.err.txt"
  if (Select-String $log -Pattern "CPU weight repacking on" -SimpleMatch -Quiet) { throw "the plugin repacked although llama-server had --no-repack" }
  if (Select-String $log -Pattern "tested with ggml" -SimpleMatch -Quiet) { throw "the plugin calls a ggml that is untested (MOE_CACHE_TESTED_GGML)" }
  if (-not (Select-String $log -Pattern "executor=on" -Quiet)) { throw "the executor did not run" }
  $m = Select-String $log -Pattern '"exec_checked":(\d+),"exec_mismatch":(\d+)' | Select-Object -Last 1
  if (-not $m -or [int64]$m.Matches[0].Groups[1].Value -eq 0 -or [int64]$m.Matches[0].Groups[2].Value -ne 0) { throw "executor byte check: none checked or mismatches ($($m.Line))" }
}

if ($ModelWatchdog) { Stop-Job $ModelWatchdog -ErrorAction SilentlyContinue; Remove-Job $ModelWatchdog -Force -ErrorAction SilentlyContinue }
Copy-Item (Join-Path $env:TEMP "moe-cache-test\*.txt") $RunTemp -ErrorAction SilentlyContinue
$stepCount = @(Get-Content $ResultFile).Count - 1
Add-Content -Encoding ascii $ResultFile "pc_ci finished`t$stepCount steps`t$(if ($AnyFail) { 'FAIL' } else { 'pass' })`t0"
if ($AnyFail) { exit 1 } else { exit 0 }
