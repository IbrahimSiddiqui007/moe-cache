# Download one file for CI (the PowerShell twin of tools/ci/fetch.sh): keep OUT when it already has the right sha256 (an actions/cache hit),
# else retry with backoff and check the sha256. A failure names the host: "download host down" is not a code failure.
#   tools/ci/fetch.ps1 -Url URL -Out FILE [-Sha256 hex] [-Tries 6] [-Wait 15]
param([Parameter(Mandatory)][string]$Url, [Parameter(Mandatory)][string]$Out, [string]$Sha256 = "", [int]$Tries = 6, [int]$Wait = 15)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$hostName = ([uri]$Url).Host
function Test-Pinned([string]$f) {
  if (-not (Test-Path $f) -or (Get-Item $f).Length -eq 0) { return $false }
  return ($Sha256 -eq "" -or (Get-FileHash $f -Algorithm SHA256).Hash.ToLower() -eq $Sha256.ToLower())
}
if (Test-Pinned $Out) { Write-Output "ci_fetch: $Out already here, not downloading"; exit 0 }
# github.com/.../releases/download/TAG/NAME answers 504 in an outage while api.github.com still serves the asset (2026-11-08): with a token (GITHUB_TOKEN / GH_TOKEN)
# the asset is looked up by tag + name and fetched from the API first; the plain URL is the fallback.
function Get-ApiUrl {
  $tok = if ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } else { $env:GH_TOKEN }
  if (-not $tok -or $Url -notmatch '^https://github\.com/([^/]+/[^/]+)/releases/download/([^/]+)/([^/?]+)$') { return $null }
  $repo, $tag, $name = $Matches[1], $Matches[2], $Matches[3]
  try {
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/tags/$tag" -Headers @{ Authorization = "Bearer $tok"; Accept = "application/vnd.github+json" } -TimeoutSec 60
    $a = $rel.assets | Where-Object { $_.name -eq $name } | Select-Object -First 1
    if ($a) { return "https://api.github.com/repos/$repo/releases/assets/$($a.id)" }
  } catch { }
  return $null
}
$why = ""
for ($i = 1; $i -le $Tries; $i++) {
  Remove-Item "$Out.part" -ErrorAction SilentlyContinue
  try {
    $got = $false
    $api = Get-ApiUrl
    if ($api) {
      try {
        $tok = if ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } else { $env:GH_TOKEN }
        Invoke-WebRequest -UseBasicParsing -Uri $api -OutFile "$Out.part" -TimeoutSec 900 -Headers @{ Authorization = "Bearer $tok"; Accept = "application/octet-stream" }
        $got = $true; Write-Output "ci_fetch: got it through the GitHub API ($api)"
      } catch { Write-Output "ci_fetch: API route failed ($($_.Exception.Message)), trying the plain URL" }
    }
    if (-not $got) { Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile "$Out.part" -TimeoutSec 900 }
    if (Test-Pinned "$Out.part") { Move-Item -Force "$Out.part" $Out; Write-Output "ci_fetch: $Url ok (try $i)"; exit 0 }
    $why = "sha256 mismatch: got $((Get-FileHash "$Out.part" -Algorithm SHA256).Hash.ToLower()), pinned $Sha256"
  } catch { $why = "$($_.Exception.Message) (host $hostName)" }
  Write-Output "ci_fetch: try $i of $Tries failed: $why"
  if ($i -lt $Tries) { Start-Sleep -Seconds $Wait; $Wait = [Math]::Min($Wait * 2, 240) }
}
Remove-Item "$Out.part" -ErrorAction SilentlyContinue
if ($why -like "sha256*") { Write-Output "::error::$Url downloaded but the file is not the pinned one ($why). This is NOT a host outage: the asset changed or the pin is wrong." }
else { Write-Output "::error::download host down: $Url ($why) after $Tries tries. Not a code failure: re-run when $hostName is back; a cache hit skips this step." }
exit 1
