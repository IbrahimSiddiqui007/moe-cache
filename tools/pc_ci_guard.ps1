# Run by tools/pc_ci.sh on the PC before anything else: refuses (exit 3) while a PC chain, a llama-server or another pc_ci run is active,
# and moves old pc_ci work folders to D:\to_delete (nothing is deleted on the PC; the owner empties that folder).
param(
  [string]$Root = "C:\dl\pc_ci",
  [int]$KeepFolders = 3
)

$busy = @()

$chainPidFile = "C:\dl\afternoon\runner.pid"
if (Test-Path $chainPidFile) {
  $chainPid = (Get-Content $chainPidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($chainPid -match '^\d+$' -and (Get-Process -Id ([int]$chainPid) -ErrorAction SilentlyContinue)) { $busy += "chain runner (pid $chainPid, $chainPidFile) is running" }
}

foreach ($server in (Get-Process -Name "llama-server" -ErrorAction SilentlyContinue)) { $busy += "llama-server is running (pid $($server.Id))" }

$lockFile = Join-Path $Root "running.lock"
if (Test-Path $lockFile) {
  $lockPid = (Get-Content $lockFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($lockPid -match '^\d+$' -and (Get-Process -Id ([int]$lockPid) -ErrorAction SilentlyContinue)) { $busy += "another pc_ci run is active (pid $lockPid)" }
}

if ($busy.Count -gt 0) {
  foreach ($line in $busy) { Write-Host "REFUSED: $line" }
  exit 3
}

$workRoot = Join-Path $Root "work"
$holding = "D:\to_delete\pc_ci"
if (Test-Path $workRoot) {
  $old = @(Get-ChildItem $workRoot -Directory | Sort-Object Name -Descending | Select-Object -Skip $KeepFolders)
  if ($old.Count -gt 0) {
    New-Item -ItemType Directory -Force $holding | Out-Null
    foreach ($dir in $old) {
      $target = Join-Path $holding ($dir.Name + "_" + (Get-Date -Format HHmmss))
      Move-Item $dir.FullName $target
      Write-Host "moved old work folder to $target"
    }
  }
}
Write-Host "PC is free"
exit 0
