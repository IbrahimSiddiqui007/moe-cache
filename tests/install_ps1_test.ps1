# install.ps1 on Windows, no Pester: a fake release (zip + SHA256SUMS-windows) in a folder, the real user PATH in the registry (saved and restored).
# Checks: a wrong hash is refused, a zip with a path that leaves the folder is refused, a good install writes the kit, the command and the PATH entry,
# a second run is idempotent, an update says what it replaced, -Uninstall removes exactly what was added (a PATH entry that was already there stays).
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\install_ps1_test.ps1
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$installer = Join-Path (Split-Path -Parent $here) 'install.ps1'
$T = Join-Path ([IO.Path]::GetTempPath()) ('moe-install-ps1-' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $T | Out-Null
$hostExe = (Get-Process -Id $PID).Path
$fails = 0
function Check([bool]$ok, [string]$what) { if ($ok) { Write-Host "  ok   $what" } else { Write-Host "  FAIL $what"; Write-Host ('       last installer output: ' + $script:lastOut); $script:fails++ } }

$py = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $py) { $py = Get-Command py -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1 }
if (-not $py) { Write-Host 'no Python on this machine: cannot run the install test'; exit 1 }
$maker = Join-Path $T 'make_release.py'
@'
import hashlib, os, sys, zipfile
d, ver, mode = sys.argv[1:4]
os.makedirs(d, exist_ok=True)
n = "moe-cache-v%s-windows-x64" % ver
z = os.path.join(d, n + ".zip")
src = 'import sys\nprint("moe-cache %s" if sys.argv[1:2] == ["--version"] else "args: " + " ".join(sys.argv[1:]))\n' % ver
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    f.writestr(n + "/bin/moe_cache.py", src)
    f.writestr(n + "/LICENSE", "x")
    if mode == "slip":
        f.writestr(n + "/../../evil.txt", "x")
h = hashlib.sha256(open(z, "rb").read()).hexdigest()
open(os.path.join(d, "SHA256SUMS-windows"), "w").write(("0" * 64 if mode == "bad" else h) + "  " + n + ".zip\n")
'@ | Set-Content -LiteralPath $maker -Encoding ASCII
function New-Release([string]$name, [string]$ver, [string]$mode) {
    $d = Join-Path $T $name
    & $py.Source $maker $d $ver $mode
    if ($LASTEXITCODE -ne 0) { throw 'could not build the fake release' }
    return $d
}
function Run-Installer([string]$rel, [string]$ver, [string[]]$extra) {
    $out = & $hostExe -NoProfile -ExecutionPolicy Bypass -File $installer -Root $script:root -BaseUrl $rel -Version $ver @extra 2>&1 | Out-String
    $script:lastOut = $out
    return @{ Code = $LASTEXITCODE; Out = $out }
}
function Count-Entry([string]$dir) { return @((Get-UserPath) -split ';' | Where-Object { $_.TrimEnd('\') -ieq $dir.TrimEnd('\') }).Count }
function Get-UserPath { $k = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment'); try { return [string]$k.GetValue('Path', '', 'DoNotExpandEnvironmentNames') } finally { $k.Close() } }
function Set-UserPath([string]$v) { $k = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true); try { $k.SetValue('Path', $v, 'ExpandString') } finally { $k.Close() } }

$origPath = Get-UserPath
try {
    $script:root = Join-Path $T 'root'; $bin = Join-Path $root 'bin'; $kit = Join-Path $root 'kit'; $shim = Join-Path $bin 'moe-cache.cmd'

    $r = Run-Installer (New-Release 'bad' '9.9.1' 'bad') '9.9.1' @()
    Check ($r.Code -ne 0 -and $r.Out -match 'REFUSED' -and $r.Out -match 'sha256') 'a wrong hash is refused'
    Check (-not (Test-Path $kit) -and -not (Test-Path $shim) -and (Count-Entry $bin) -eq 0) '...and nothing was installed or added to the PATH'

    $r = Run-Installer (New-Release 'slip' '9.9.1' 'slip') '9.9.1' @()
    Check ($r.Code -ne 0 -and $r.Out -match 'REFUSED' -and -not (Test-Path (Join-Path $T 'evil.txt')) -and -not (Test-Path $kit)) 'a zip with a path that leaves the folder is refused'

    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @('-DryRun')
    Check ($r.Code -eq 0 -and $r.Out -match 'would download' -and -not (Test-Path $kit) -and -not (Test-Path $bin) -and -not (Test-Path (Join-Path $root 'kit.part'))) '-DryRun prints the plan and installs nothing'

    $good1 = Join-Path $T 'good1'
    $r = Run-Installer $good1 '9.9.1' @()
    Check ($r.Code -eq 0 -and $r.Out -match 'sha256 matches' -and $r.Out -match 'installed: moe-cache 9.9.1') 'install: hash checked, version printed'
    Check ((Test-Path (Join-Path $kit 'bin\moe_cache.py')) -and (Test-Path $shim)) '...kit and command written'
    Check ((Count-Entry $bin) -eq 1) '...the bin folder is on the user PATH, once'
    Check ($r.Out -match 'undo:' -and $r.Out -match '-Uninstall') '...it says how to undo'
    $o = (& $shim launch --x 2>&1 | Out-String).Trim()
    Check ($o -eq 'args: launch --x') "the moe-cache command runs the kit's moe_cache.py with its arguments ($o)"

    $r = Run-Installer $good1 '9.9.1' @()
    Check ($r.Code -eq 0 -and $r.Out -match 'already installed' -and (Count-Entry $bin) -eq 1) 'second run: idempotent, no second PATH entry'

    $r = Run-Installer (New-Release 'good2' '9.9.2' 'ok') '9.9.2' @()
    Check ($r.Code -eq 0 -and $r.Out -match 'installed: moe-cache 9.9.2 \(was 9.9.1\)') 'update in place says what it replaced'

    # ---- failure midway (AUD-050): a failed update keeps the 9.9.2 install whole, a failed first install leaves nothing
    function Snapshot { $k = if (Test-Path $kit) { (Get-ChildItem -LiteralPath $kit -Recurse -Force | ForEach-Object { $_.FullName.Substring($kit.Length) }) -join '|' } else { '' }
        return ($k + '#' + (@(Get-ChildItem -LiteralPath $root -Force -ErrorAction SilentlyContinue | ForEach-Object Name) -join ',') + '#' + (@(Get-ChildItem -LiteralPath $bin -Force -ErrorAction SilentlyContinue | ForEach-Object Name) -join ',') +
            '#' + $(if (Test-Path $shim) { Get-Content -LiteralPath $shim -Raw } else { '' }) + '#' + $(if (Test-Path (Join-Path $root 'install.json')) { Get-Content -LiteralPath (Join-Path $root 'install.json') -Raw } else { '' }) + '#' + (Get-UserPath)) }
    $good3 = New-Release 'good3' '9.9.3' 'ok'
    $before = Snapshot
    foreach ($fp in 'unpack', 'swap', 'shim', 'record', 'path') {
        $env:MOE_CACHE_INSTALL_FAIL_AT = $fp
        try { $r = Run-Installer $good3 '9.9.3' @() } finally { Remove-Item Env:\MOE_CACHE_INSTALL_FAIL_AT -ErrorAction SilentlyContinue }
        Check ($r.Code -ne 0 -and $r.Out -match "injected at $fp" -and (Snapshot) -eq $before) "update fails at '$fp': the 9.9.2 install is exactly as before (no kit.part / kit.old, same command, record and PATH)"
        $o = (& $shim --version 2>&1 | Out-String).Trim()
        Check ($o -eq 'moe-cache 9.9.2') '...and the command still runs 9.9.2'
    }
    # a locked file in the kit: the swap fails at once and nothing has changed
    $lock = [IO.File]::Open((Join-Path $kit 'bin\moe_cache.py'), 'Open', 'Read', 'None')
    try { $r = Run-Installer $good3 '9.9.3' @() } finally { $lock.Dispose() }
    Check ($r.Code -ne 0 -and (Snapshot) -eq $before) 'a locked file in the old kit: the update fails whole, the old install is untouched'
    $r = Run-Installer $good3 '9.9.3' @()
    Check ($r.Code -eq 0 -and $r.Out -match 'installed: moe-cache 9.9.3 \(was 9.9.2\)' -and -not (Test-Path (Join-Path $root 'kit.old')) -and -not (Test-Path (Join-Path $root 'kit.part'))) 'the update after the failures works and leaves no kit.old / kit.part'
    # a kit without the completeness marker is not trusted as "already installed"
    Remove-Item -LiteralPath (Join-Path $kit '.moe-cache-complete') -Force
    $r = Run-Installer $good3 '9.9.3' @()
    Check ($r.Code -eq 0 -and $r.Out -notmatch 'already installed' -and (Test-Path (Join-Path $kit '.moe-cache-complete'))) 'a kit without its completeness marker is reinstalled, not "repaired"'
    $r = Run-Installer $good3 '9.9.3' @('-Uninstall')
    Check ($r.Code -eq 0 -and -not (Test-Path $root)) 'uninstall leaves no empty folders'
    foreach ($fp in 'unpack', 'swap', 'shim', 'record', 'path') {
        $env:MOE_CACHE_INSTALL_FAIL_AT = $fp
        try { $r = Run-Installer $good3 '9.9.3' @() } finally { Remove-Item Env:\MOE_CACHE_INSTALL_FAIL_AT -ErrorAction SilentlyContinue }
        Check ($r.Code -ne 0 -and -not (Test-Path $root) -and (Count-Entry $bin) -eq 0) "first install failing at '$fp' leaves nothing (no folder, no command, no PATH entry)"
    }
    $r = Run-Installer $good3 '9.9.3' @()
    Check ($r.Code -eq 0) '...and the next run installs normally'
    # a command of the user's: refused before anything is downloaded
    Remove-Item -LiteralPath $shim -Force; 'echo mine' | Set-Content $shim
    $r = Run-Installer (New-Release 'good4' '9.9.4' 'ok') '9.9.4' @()
    Check ($r.Code -ne 0 -and $r.Out -match 'not ours' -and (Get-Content $shim -Raw) -match 'echo mine' -and (Get-Content (Join-Path $root 'install.json') -Raw) -match '9.9.3') 'a moe-cache.cmd that is not ours is refused before the update starts'
    Remove-Item -LiteralPath $shim -Force
    $r = Run-Installer $good3 '9.9.3' @('-Force')
    Check ($r.Code -eq 0) 'reinstall with -Force over a repaired state'
    # a -Root that already has a kit folder of the user's: left alone (AUD-052)
    $kroot = Join-Path $T 'kroot'; New-Item -ItemType Directory -Force -Path (Join-Path $kroot 'kit') | Out-Null; 'mine' | Set-Content (Join-Path $kroot 'kit\mine.txt')
    $rr = & $hostExe -NoProfile -ExecutionPolicy Bypass -File $installer -Root $kroot -BaseUrl $good3 -Version 9.9.3 2>&1 | Out-String
    Check ($LASTEXITCODE -ne 0 -and $rr -match 'not created by this installer' -and (Test-Path (Join-Path $kroot 'kit\mine.txt'))) 'a -Root with a kit folder of the user'"'"'s: refused, the folder stays'
    # non-ASCII -Root (AUD-051): install.json is read back, uninstall removes the PATH entry
    $uroot = Join-Path $T ('Jos' + [char]0xE9 + '-' + [char]0x0416)
    $ubin = Join-Path $uroot 'bin'
    $rr = & $hostExe -NoProfile -ExecutionPolicy Bypass -File $installer -Root $uroot -BaseUrl $good3 -Version 9.9.3 2>&1 | Out-String
    Check ((Count-Entry $ubin) -eq 1) 'non-ASCII -Root: installed and on the PATH'
    $rr = & $hostExe -NoProfile -ExecutionPolicy Bypass -File $installer -Root $uroot -Uninstall 2>&1 | Out-String
    Check ((Count-Entry $ubin) -eq 0 -and -not (Test-Path $uroot)) '...-Uninstall removes the PATH entry and the folder'
    # back to the 9.9.2-based flow below
    $r = Run-Installer $good3 '9.9.3' @('-Uninstall')
    $r = Run-Installer $good1 '9.9.1' @()
    $r = Run-Installer (New-Release 'good2b' '9.9.2' 'ok') '9.9.2' @()

    'not ours' | Set-Content (Join-Path $root 'my-notes.txt')
    $r = Run-Installer $good1 '9.9.2' @('-Uninstall')
    Check ($r.Code -eq 0 -and -not (Test-Path $kit) -and -not (Test-Path $shim) -and -not (Test-Path (Join-Path $root 'install.json'))) '-Uninstall removes kit, command and record'
    Check ((Count-Entry $bin) -eq 0 -and ((@((Get-UserPath) -split ';' | Where-Object { $_ }) -join ';') -eq (@($origPath -split ';' | Where-Object { $_ }) -join ';'))) '...and the PATH is exactly as before'
    Check ((Test-Path (Join-Path $root 'my-notes.txt'))) "...a file of the user's in the same folder stays"
    Remove-Item (Join-Path $root 'my-notes.txt')

    # a PATH entry that was already there is not ours to remove
    Set-UserPath (($origPath.TrimEnd(';') + ';' + $bin).TrimStart(';'))
    $r = Run-Installer $good1 '9.9.1' @()
    Check ($r.Code -eq 0 -and (Count-Entry $bin) -eq 1) 'a PATH entry that was already there is not added twice'
    $r = Run-Installer $good1 '9.9.1' @('-Uninstall')
    Check ($r.Code -eq 0 -and (Count-Entry $bin) -eq 1) '...and -Uninstall leaves it (this installer did not add it)'
    Set-UserPath $origPath

    $r = Run-Installer $good1 '9.9.1' @('-Uninstall')
    Check ($r.Code -eq 0 -and $r.Out -match 'nothing to remove') 'uninstall twice is harmless'
} finally {
    Set-UserPath $origPath
    Remove-Item -LiteralPath $T -Recurse -Force -ErrorAction SilentlyContinue
}
if ($fails) { Write-Host "FAIL ($fails failed)"; exit 1 }
Write-Host 'PASS (0 failed)'
