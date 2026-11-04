# No-model check of the desktop app's NSIS installer on a real Windows PC. Run with: powershell -NoProfile -ExecutionPolicy Bypass -File pc_install_check.ps1 -Setup X -CheckPy Y
# Installs silently into its own folder (-Work\app), checks the layout, runs the page check with the INSTALLED Python and kit (no model), starts the
# installed app exe (headless session: it may not get a window, that is recorded), kills it and checks the Python child ends by itself, uninstalls
# silently, and checks the installer's shortcuts and registry entry are gone. Never deletes: leftovers are MOVED to D:\to_delete.
param(
    [Parameter(Mandatory)][string]$Setup,
    [Parameter(Mandatory)][string]$CheckPy,
    [string]$Work = "D:\desktop_checks"
)
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force $Work | Out-Null
$log = Join-Path $Work "result.txt"
"" | Out-File $log -Encoding utf8
function Say($m) { Write-Host $m; $m | Out-File $log -Append -Encoding utf8 }
$fails = 0
function Check($ok, $what) { if ($ok) { Say "ok   $what" } else { Say "FAIL $what"; $script:fails++ } }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$app = Join-Path $Work "app"
if (Test-Path $app) { New-Item -ItemType Directory -Force "D:\to_delete" | Out-Null; Move-Item $app "D:\to_delete\desktop_app_before_$stamp" }
Say ("start " + (Get-Date) + "  setup " + $Setup + "  size " + (Get-Item $Setup).Length)
Say ("sha256 " + (Get-FileHash $Setup -Algorithm SHA256).Hash)
$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$uninstKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall"
$before = @(Get-ChildItem $startMenu -Recurse -Filter "*moe*" -ErrorAction SilentlyContinue).Count
Check ($before -eq 0) "no moe-cache shortcut in the Start menu before the install ($before)"

# 1 silent install (NSIS: /S silent, /D= install folder, must be last and unquoted)
$p = Start-Process -FilePath $Setup -ArgumentList "/S", "/D=$app" -Wait -PassThru
Check ($p.ExitCode -eq 0) "silent install exit code $($p.ExitCode)"
Start-Sleep -Seconds 2
$exe = Join-Path $app "moe-cache.exe"
Check (Test-Path $exe) "moe-cache.exe installed"
$kit = Join-Path $app "moe-cache"
$py = Join-Path $app "python\python.exe"
foreach ($rel in @("app_entry.py", "bin\moe_cache.py", "ui\index.html", "models.json", "ggml-moe-cache-api3.dll", "llama\llama-server.exe")) {
    Check (Test-Path (Join-Path $kit $rel)) "kit has $rel"
}
Check (Test-Path $py) "bundled python.exe"
Check (Test-Path (Join-Path $app "uninstall.exe")) "uninstall.exe present"
$sc = @(Get-ChildItem $startMenu -Recurse -Filter "*moe*" -ErrorAction SilentlyContinue).Count
Say "info Start menu entries after install: $sc"
$key = Get-ChildItem $uninstKey -ErrorAction SilentlyContinue | Where-Object { (Get-ItemProperty $_.PSPath).DisplayName -like "*moe-cache*" }
Say ("info registry uninstall entry: " + ($(if ($key) { $key.PSChildName } else { "none" })))
Say ("info installed size MB: " + [math]::Round((Get-ChildItem $app -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1))

# 2 the page check with the installed Python and kit (no model)
Say "--- page check (installed python, installed kit)"
$o = & $py $CheckPy $kit $py 2>&1 | Out-String
Say $o
if ($LASTEXITCODE -ne 0) { $fails++; Say "FAIL page check exit $LASTEXITCODE" }

# 3 the real app exe: start, see whether it keeps running and starts the page, kill it, the page process must end by itself
Say "--- app exe (headless session: a window may be impossible; recorded either way)"
$beforeIds = @(Get-Process -Name python, llama-server -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
$ap = Start-Process -FilePath $exe -PassThru
Start-Sleep -Seconds 20
$alive = -not $ap.HasExited
Say "info app exe alive after 20 s: $alive (exit code if ended: $(if ($ap.HasExited) { $ap.ExitCode } else { 'running' }))"
$kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($ap.Id)" | Where-Object { $_.Name -like "python*" })
Say "info python children of the app: $($kids.Count)"
if ($alive) {
    Stop-Process -Id $ap.Id -Force
    foreach ($k in $kids) {
        $t = 0
        while ((Get-Process -Id $k.ProcessId -ErrorAction SilentlyContinue) -and $t -lt 25) { Start-Sleep -Seconds 1; $t++ }
        Check (-not (Get-Process -Id $k.ProcessId -ErrorAction SilentlyContinue)) "python child $($k.ProcessId) ended by itself $t s after the app was killed"
    }
    if ($kids.Count -eq 0) { Say "info the app ran but started no python child (page not started in this session)" }
}
$left = @(Get-Process -Name python, llama-server -ErrorAction SilentlyContinue | Where-Object { $beforeIds -notcontains $_.Id })
Check ($left.Count -eq 0) "no new python / llama-server processes left ($($left.Count))"

# 4 silent uninstall (_?= runs it in place and waits)
Say "--- uninstall"
$un = Join-Path $app "uninstall.exe"
$u = Start-Process -FilePath $un -ArgumentList "/S", "_?=$app" -Wait -PassThru
Check ($u.ExitCode -eq 0) "silent uninstall exit code $($u.ExitCode)"
Start-Sleep -Seconds 2
Check (-not (Test-Path $exe)) "moe-cache.exe gone"
Check (-not (Test-Path $kit)) "kit folder gone"
$after = @(Get-ChildItem $startMenu -Recurse -Filter "*moe*" -ErrorAction SilentlyContinue).Count
Check ($after -eq 0) "no moe-cache shortcut left in the Start menu ($after)"
$key2 = Get-ChildItem $uninstKey -ErrorAction SilentlyContinue | Where-Object { (Get-ItemProperty $_.PSPath).DisplayName -like "*moe-cache*" }
Check (-not $key2) "registry uninstall entry gone"
$rest = if (Test-Path $app) { @(Get-ChildItem $app -Recurse -Force -ErrorAction SilentlyContinue) } else { @() }
Say "info files left in the install folder: $($rest.Count) ($(($rest | Select-Object -First 5 | ForEach-Object { $_.Name }) -join ', '))"
if (Test-Path $app) { New-Item -ItemType Directory -Force "D:\to_delete" | Out-Null; Move-Item $app "D:\to_delete\desktop_app_$stamp"; Say "info leftovers moved to D:\to_delete\desktop_app_$stamp" }
Say ("end " + (Get-Date) + "  failures: $fails")
exit $(if ($fails -eq 0) { 0 } else { 1 })
