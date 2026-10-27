# Push a model out of Windows' file cache before a benchmark run, without admin rights and without changing anything on disk.
# It only READS other big files (more than the machine's RAM in total), so the standby cache no longer holds the target.
# Stock llama-server benefits from a warm file cache, the plugin's unbuffered reads do not: flush before every run for a fair stock-vs-plugin comparison.
#
#   powershell -ExecutionPolicy Bypass -File tools\win_flush_cache.ps1 -Target D:\models\a.gguf -Other "D:\models|E:\more\b.gguf"
#     -Target  the model(s) you are about to run: never read here, and skipped if they also lie in an -Other folder
#     -Other   big files and/or folders (searched recursively, files of 100 MB or more) to read instead
#     -Factor  read this multiple of the RAM (default 1.15; the cache cannot hold more than the RAM)
#   Prints GB read and seconds. Writes and deletes nothing. Exit code 1 if the other files were too small to cover the RAM (the target may still be cached).
param(
  [string[]]$Target = @(),
  [Parameter(Mandatory = $true)][string[]]$Other,
  [double]$Factor = 1.15
)
$ErrorActionPreference = "Stop"
# "-File" passes a list as separate words that do not bind to [string[]]: callers join several paths with "|" (never part of a Windows path)
$Target = @($Target | ForEach-Object { $_ -split '\|' } | Where-Object { $_ })
$Other = @($Other | ForEach-Object { $_ -split '\|' } | Where-Object { $_ })
$ram = [double](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$want = $ram * $Factor
$skip = @{}
foreach ($t in $Target) { $skip[(Resolve-Path -LiteralPath $t).ProviderPath.ToLowerInvariant()] = $true }
$files = @()
foreach ($o in $Other) {
  $p = (Resolve-Path -LiteralPath $o).ProviderPath
  if (Test-Path -LiteralPath $p -PathType Container) { $files += Get-ChildItem -LiteralPath $p -Recurse -File | Where-Object { $_.Length -ge 100MB } }
  else { $files += Get-Item -LiteralPath $p }
}
$files = $files | Where-Object { -not $skip.ContainsKey($_.FullName.ToLowerInvariant()) } | Sort-Object FullName -Unique
$avail = ($files | Measure-Object Length -Sum).Sum
if ($avail -lt $want) {
  Write-Host ("WARNING: the other files total {0:N1} GB, the machine has {1:N1} GB RAM: the target may still be partly cached. Add more files." -f ($avail / 1e9), ($ram / 1e9))
}
$t0 = Get-Date; $n = 0.0
$buf = New-Object byte[] (16MB)
foreach ($f in $files) {
  if ($n -ge $want) { break }
  $fs = [IO.File]::Open($f.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
  try {
    while ($n -lt $want -and ($r = $fs.Read($buf, 0, $buf.Length)) -gt 0) { $n += $r }
  } finally { $fs.Close() }
}
$sec = ((Get-Date) - $t0).TotalSeconds
Write-Host ("flushed file cache: read {0:N1} GB of other files in {1:N0} s (RAM {2:N1} GB, target {3})" -f ($n / 1e9), $sec, ($ram / 1e9), $(if ($Target.Count) { $Target -join ", " } else { "none given" }))
if ($n -lt $want) { exit 1 }
