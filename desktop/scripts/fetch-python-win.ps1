# Windows: download the python.org embeddable zip into src-tauri/resources/python/ and check its SHA-256.
# Usage: powershell -File scripts\fetch-python-win.ps1 -Version 3.12.8 -Sha256 <SHA-256 of the zip> (the workflow pins the 3.12.8 hash)
param([Parameter(Mandatory)][string]$Version, [Parameter(Mandatory)][string]$Sha256)
$ErrorActionPreference = "Stop"
$dst = Join-Path $PSScriptRoot "..\src-tauri\resources\python"
$zipDir = if ($env:PY_ZIP_DIR) { $env:PY_ZIP_DIR } else { $env:TEMP }   # CI restores PY_ZIP_DIR with actions/cache
New-Item -ItemType Directory -Force $zipDir | Out-Null
$zip = Join-Path $zipDir "python-$Version-embed-amd64.zip"
# retried with backoff, skipped when the cached zip already has the pinned sha256; a failure names the host
& (Join-Path $PSScriptRoot "..\..\tools\ci\fetch.ps1") -Url "https://www.python.org/ftp/python/$Version/python-$Version-embed-amd64.zip" -Out $zip -Sha256 $Sha256
if ($LASTEXITCODE -ne 0) { throw "download host down or wrong sha256 for python-$Version-embed-amd64.zip" }
if ((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $Sha256.ToLower()) { throw "SHA-256 of $zip does not match" }
Remove-Item $dst -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive $zip -DestinationPath $dst
# the embeddable build ignores PYTHONPATH and the script folder unless python3NN._pth allows them: the launcher imports its sibling files
Get-ChildItem $dst -Filter "python*._pth" | ForEach-Object { Add-Content $_.FullName "`n..\moe-cache\bin`n..\moe-cache\planner" }
Write-Host "Python $Version in $dst"
