# Windows: download the python.org embeddable zip into src-tauri/resources/python/ and check its SHA-256.
# Usage: powershell -File scripts\fetch-python-win.ps1 -Version 3.12.8 -Sha256 <SHA-256 of the zip> (the workflow pins the 3.12.8 hash)
param([Parameter(Mandatory)][string]$Version, [Parameter(Mandatory)][string]$Sha256)
$ErrorActionPreference = "Stop"
$dst = Join-Path $PSScriptRoot "..\src-tauri\resources\python"
$zip = Join-Path $env:TEMP "python-$Version-embed-amd64.zip"
Invoke-WebRequest "https://www.python.org/ftp/python/$Version/python-$Version-embed-amd64.zip" -OutFile $zip
if ((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $Sha256.ToLower()) { throw "SHA-256 of $zip does not match" }
Remove-Item $dst -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive $zip -DestinationPath $dst
# the embeddable build ignores PYTHONPATH and the script folder unless python3NN._pth allows them: the launcher imports its sibling files
Get-ChildItem $dst -Filter "python*._pth" | ForEach-Object { Add-Content $_.FullName "`n..\moe-cache\bin`n..\moe-cache\planner" }
Write-Host "Python $Version in $dst"
