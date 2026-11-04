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
    for extra in ("moe-cache.exe", "ggml-moe-cache-api3.dll", "llama/llama-server.exe"):
        if mode != "nodll" or extra != "ggml-moe-cache-api3.dll":
            f.writestr(n + "/" + extra, "x")
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

# RELEASE-686: a fake gh.cmd first on PATH (FAKE_GH_AUTH / FAKE_GH_VERIFY = exit codes). Without a verified build attestation the installer stops unless -YesUnverified / MOE_CACHE_ALLOW_UNVERIFIED=1.
$fakeGh = Join-Path $T 'fakegh'; New-Item -ItemType Directory -Force -Path $fakeGh | Out-Null
Set-Content -LiteralPath (Join-Path $fakeGh 'gh.cmd') -Encoding ASCII -Value @('@echo off', 'if "%1 %2"=="auth status" exit /b %FAKE_GH_AUTH%', 'if "%1 %2"=="attestation verify" if defined FAKE_GH_LOG echo %*>>"%FAKE_GH_LOG%"', 'if "%1 %2"=="attestation verify" if defined FAKE_GH_MSG echo %FAKE_GH_MSG% 1>&2', 'if "%1 %2"=="attestation verify" exit /b %FAKE_GH_VERIFY%', 'exit /b 1')
$env:FAKE_GH_AUTH = '0'; $env:FAKE_GH_VERIFY = '0'
$origProcPath = $env:PATH
$env:PATH = "$fakeGh;$origProcPath"
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

    # ---- RELEASE-686: the build attestation
    $env:FAKE_GH_VERIFY = '1'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @()
    Check ($r.Code -ne 0 -and $r.Out -match 'REFUSED' -and $r.Out -match 'NOT checked: who built' -and $r.Out -match 'checked:\s+sha256' -and $r.Out -match 'YesUnverified' -and -not (Test-Path $kit) -and -not (Test-Path $shim)) 'gh verify fails: refused, says what was and was not checked, nothing installed'
    $env:FAKE_GH_VERIFY = '0'; $env:FAKE_GH_AUTH = '1'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @()
    Check ($r.Code -ne 0 -and $r.Out -match 'REFUSED' -and $r.Out -match 'not logged in' -and -not (Test-Path $kit)) 'gh not logged in: refused, nothing installed'
    $env:FAKE_GH_AUTH = '0'
    $env:PATH = (($origProcPath -split ';') | Where-Object { $_ -and -not (Test-Path (Join-Path $_ 'gh.exe')) -and -not (Test-Path (Join-Path $_ 'gh.cmd')) }) -join ';'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @()
    Check ($r.Code -ne 0 -and $r.Out -match 'REFUSED' -and $r.Out -match 'is not installed' -and -not (Test-Path $kit)) 'gh not installed: refused, nothing installed'
    $r = Run-Installer (New-Release 'bad' '9.9.1' 'bad') '9.9.1' @('-YesUnverified')
    Check ($r.Code -ne 0 -and $r.Out -match 'REFUSED' -and $r.Out -match 'sha256' -and -not (Test-Path $kit)) '-YesUnverified never skips the sha256: a wrong hash is still refused'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @('-YesUnverified')
    Check ($r.Code -eq 0 -and $r.Out -match 'WARNING: continuing without the build attestation' -and $r.Out -match 'NOT checked' -and (Test-Path (Join-Path $kit 'bin\moe_cache.py'))) '-YesUnverified with no gh: installs, with a warning that says what was not checked'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @('-Uninstall')
    Check ($r.Code -eq 0 -and -not (Test-Path $kit)) '...(uninstalled again)'
    $env:FAKE_GH_VERIFY = '1'; $env:MOE_CACHE_ALLOW_UNVERIFIED = '1'; $env:PATH = "$fakeGh;$origProcPath"
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @()
    Check ($r.Code -ne 0 -and $r.Out -match 'REFUSED' -and $r.Out -match 'FAILED' -and $r.Out -match 'never override' -and -not (Test-Path $kit) -and -not (Test-Path $shim)) 'MOE_CACHE_ALLOW_UNVERIFIED=1 does NOT override a verification that ran and FAILED'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @('-YesUnverified')
    Check ($r.Code -ne 0 -and $r.Out -match 'never override' -and -not (Test-Path $kit)) '-YesUnverified does NOT override a failed verification either'
    $env:FAKE_GH_MSG = 'dial tcp: lookup api.github.com: no such host'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @()
    Check ($r.Code -eq 0 -and $r.Out -match 'WARNING' -and $r.Out -match 'could not reach GitHub' -and (Test-Path (Join-Path $kit 'bin\moe_cache.py'))) 'a verify that could not run (GitHub unreachable) + MOE_CACHE_ALLOW_UNVERIFIED=1: installs, with a warning'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @('-Uninstall')
    Check ($r.Code -eq 0 -and -not (Test-Path $kit)) '...(uninstalled again)'
    Remove-Item Env:\MOE_CACHE_ALLOW_UNVERIFIED
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @()
    Check ($r.Code -ne 0 -and $r.Out -match 'REFUSED' -and $r.Out -match 'could not reach GitHub' -and -not (Test-Path $kit)) 'GitHub unreachable without the flag: refused'
    Remove-Item Env:\FAKE_GH_MSG
    # the pinned signer workflow
    $env:FAKE_GH_VERIFY = '0'; $env:FAKE_GH_LOG = Join-Path $T 'gh_args.log'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @()
    Check ($r.Code -eq 0 -and (Get-Content -LiteralPath $env:FAKE_GH_LOG -Raw) -match '--signer-workflow IbrahimSiddiqui007/moe-cache/.github/workflows/release.yml' -and (Get-Content -LiteralPath $env:FAKE_GH_LOG -Raw) -match '-R IbrahimSiddiqui007/moe-cache') 'gh attestation verify is called with -R and the pinned --signer-workflow'
    Remove-Item Env:\FAKE_GH_LOG
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @('-Uninstall')
    Check ($r.Code -eq 0 -and -not (Test-Path $kit)) '...(uninstalled again)'
    # required files after the unpack (antivirus may have removed one)
    $r = Run-Installer (New-Release 'nodll' '9.9.1' 'nodll') '9.9.1' @()
    Check ($r.Code -ne 0 -and $r.Out -match 'REFUSED' -and $r.Out -match 'ggml-moe-cache-api3.dll' -and $r.Out -match 'antivirus' -and -not (Test-Path $kit) -and -not (Test-Path (Join-Path $root 'kit.part'))) 'a kit missing a required file (the plugin DLL) is refused, names the file and mentions the antivirus'
    # the shim finds Python through a folder with a non-ASCII (Arabic) name: the where.exe lookup under such a user name (Critic B3; not reproducible on the PC, kept as a regression check)
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @()
    $arab = -join ([char[]](0x0645, 0x062C, 0x0644, 0x062F)); $junc = Join-Path $T ("py_" + $arab)
    cmd /c mklink /J "$junc" (Split-Path $py.Source) | Out-Null
    $savedPath = $env:PATH; $env:PATH = "$junc;$env:SystemRoot\System32"
    try { $o = (& cmd /c "`"$shim`" launch --x" 2>&1 | Out-String).Trim() } finally { $env:PATH = $savedPath }
    Check ($o -eq 'args: launch --x') "the moe-cache command finds Python through a folder with an Arabic name ($o)"
    cmd /c rmdir "$junc" | Out-Null
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @('-Uninstall')
    Check ($r.Code -eq 0 -and -not (Test-Path $kit)) '...(uninstalled again)'
    $env:FAKE_GH_VERIFY = '0'
    $r = Run-Installer (New-Release 'good1' '9.9.1' 'ok') '9.9.1' @('-DryRun')
    Check ($r.Code -eq 0 -and $r.Out -match 'would download') '-DryRun does not need gh'

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
    # a locked file in the old kit. Windows decides: a folder holding an open file may refuse the rename (then nothing has changed) or may be
    # renamed anyway (then the new kit is in place whole and the old one waits as kit.old until it can be deleted). Never a half state.
    $lock = [IO.File]::Open((Join-Path $kit 'bin\moe_cache.py'), 'Open', 'Read', 'None')
    try { $r = Run-Installer $good3 '9.9.3' @() } finally { $lock.Dispose() }
    $o = (& $shim --version 2>&1 | Out-String).Trim()
    $whole = ($r.Code -ne 0 -and (Snapshot) -eq $before -and $o -eq 'moe-cache 9.9.2') -or ($r.Code -eq 0 -and $o -eq 'moe-cache 9.9.3' -and (Test-Path (Join-Path $kit '.moe-cache-complete')) -and -not (Test-Path (Join-Path $root 'kit.part')))
    if (-not $whole) { $script:lastOut += "`n--- code $($r.Code); version '$o'; before/after differ: " + ((Snapshot) -ne $before) + "`n--- before: $before`n--- after:  $(Snapshot)" }
    Check $whole 'a locked file in the old kit: the update either fails with the old install untouched or completes whole (never half)'
    $r = Run-Installer (Join-Path $T 'good2') '9.9.2' @('-Force')       # back to a clean 9.9.2 (the lock is released; a leftover kit.old is removed now)
    Check ($r.Code -eq 0 -and -not (Test-Path (Join-Path $root 'kit.old'))) '...with the lock released, the next run installs and leaves no kit.old'
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
