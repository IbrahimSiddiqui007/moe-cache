# moe-cache installer for Windows (PowerShell 5.1 or newer). No administrator rights, no telemetry, nothing outside your own profile.
#
#   One line:    irm https://github.com/IbrahimSiddiqui007/moe-cache/releases/latest/download/install.ps1 | iex
#   Read first:  irm https://github.com/IbrahimSiddiqui007/moe-cache/releases/latest/download/install.ps1 -OutFile install.ps1
#                notepad install.ps1        (read it)
#                powershell -ExecutionPolicy Bypass -File .\install.ps1
#
# What it does:
#   1. checks that Python 3.8 or newer is installed (it never installs Python for you) and says how to get it if not
#   2. downloads the Windows zip of this release and SHA256SUMS-windows, and REFUSES (installs nothing) when the hash does not match
#   3. unpacks it to %LOCALAPPDATA%\moe-cache\kit (llama.cpp is inside the zip)
#   4. writes a small command, %LOCALAPPDATA%\moe-cache\bin\moe-cache.cmd, and adds that bin folder to YOUR user PATH (never the machine PATH)
#   5. prints the version and how to undo it
# Run it again to update in place (same version: it only repairs the command and the PATH entry; -Force reinstalls).
#
#   -Uninstall   remove exactly what it added: the kit folder, the command, the PATH entry (only when this installer added it), the record.
#                Your models, your settings (%LOCALAPPDATA%\... moe-cache cache) and every other file stay.
#   -DryRun      print what it would do and stop
#   -Version X   install release X (default: the release this file belongs to; run from a checkout: the newest)
#   -Root DIR    install somewhere else (default %LOCALAPPDATA%\moe-cache)
#   -NoPath      do not touch the PATH
#   -BaseUrl U   where the release files are (a folder or an https address; default this release's GitHub download address)
# With `| iex` no flags can be given: use the read-first form above, or  & ([scriptblock]::Create((irm URL))) -Uninstall
[CmdletBinding()]
param(
    [string]$Version = '@VERSION@',
    [string]$Root = '',
    [string]$BaseUrl = '',
    [switch]$Uninstall,
    [switch]$DryRun,
    [switch]$NoPath,
    [switch]$Force
)

$Repo = 'IbrahimSiddiqui007/moe-cache'
$Marker = 'rem moe-cache command written by install.ps1'

function Say([string]$m) { Write-Host $m }

function Get-UserPathRaw {
    $k = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment')
    try { return [string]$k.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames) } finally { $k.Close() }
}

function Set-UserPathRaw([string]$value) {
    $k = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
    try { $k.SetValue('Path', $value, [Microsoft.Win32.RegistryValueKind]::ExpandString) } finally { $k.Close() }
    # tell running programs (new terminals) that the environment changed; failure only means "open a new terminal"
    try {
        if (-not ('MoeCache.NativeEnv' -as [type])) {
            Add-Type -Namespace MoeCache -Name NativeEnv -MemberDefinition '[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError=true, CharSet=System.Runtime.InteropServices.CharSet.Auto)] public static extern System.IntPtr SendMessageTimeout(System.IntPtr h, uint m, System.UIntPtr w, string l, uint f, uint t, out System.UIntPtr r);'
        }
        $r = [UIntPtr]::Zero
        [void][MoeCache.NativeEnv]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 3000, [ref]$r)
    } catch { }
}

function Split-PathList([string]$p) { return @($p -split ';' | Where-Object { $_ -ne '' }) }
function Same-Dir([string]$a, [string]$b) { return ($a.TrimEnd('\') -ieq $b.TrimEnd('\')) }

function Find-Python {
    # MOE_CACHE_PYTHON, then the py launcher, then a python.exe that really runs (the one Windows ships without Python only opens the Store)
    $cands = @()
    if ($env:MOE_CACHE_PYTHON) { $cands += , @($env:MOE_CACHE_PYTHON) }
    foreach ($n in 'py', 'python') {
        $c = Get-Command $n -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($c) { if ($n -eq 'py') { $cands += , @($c.Source, '-3') } else { $cands += , @($c.Source) } }
    }
    foreach ($c in $cands) {
        try {
            $exe = $c[0]; $rest = @($c | Select-Object -Skip 1)
            $out = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2]); sys.exit(0 if sys.version_info >= (3, 8) else 3)" 2>$null
            if ($LASTEXITCODE -eq 0) { return @{ Exe = $exe; Args = $rest; Version = ($out | Select-Object -First 1) } }
        } catch { }
    }
    return $null
}

function Get-File([string]$base, [string]$name, [string]$dest) {
    if (Test-Path -LiteralPath $base -PathType Container) { Copy-Item -LiteralPath (Join-Path $base $name) -Destination $dest -ErrorAction Stop; return }
    $u = $base.TrimEnd('/') + '/' + $name
    try { Invoke-WebRequest -Uri $u -OutFile $dest -UseBasicParsing -ErrorAction Stop } catch { throw "could not download $u ($($_.Exception.Message))" }
}

function Expand-Kit([string]$zipPath, [string]$stage) {
    # every entry must sit under one top folder (the kit folder) and stay inside $stage; the top folder is dropped
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $stageFull = [IO.Path]::GetFullPath($stage).TrimEnd('\') + '\'
    $zip = [IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        $n = 0
        foreach ($e in $zip.Entries) {
            $parts = @($e.FullName.Replace('\', '/').Split('/') | Where-Object { $_ -ne '' })
            if ($parts.Count -lt 1) { continue }
            if ($parts -contains '..' -or $e.FullName -match '^[A-Za-z]:' -or $e.FullName.StartsWith('/') -or $e.FullName.StartsWith('\')) { throw "REFUSED: the zip has an unsafe path ($($e.FullName))" }
            if ($parts.Count -lt 2) { if ($e.Name -eq '') { continue } else { throw "REFUSED: the zip has a file outside its folder ($($e.FullName))" } }
            $sub = ($parts | Select-Object -Skip 1) -join '\'
            $dest = [IO.Path]::GetFullPath((Join-Path $stage $sub))
            if (-not $dest.StartsWith($stageFull, [StringComparison]::OrdinalIgnoreCase)) { throw "REFUSED: the zip has a path that leaves the folder ($($e.FullName))" }
            if ($e.Name -eq '') { [void](New-Item -ItemType Directory -Force -Path $dest); continue }
            [void](New-Item -ItemType Directory -Force -Path (Split-Path $dest))
            [IO.Compression.ZipFileExtensions]::ExtractToFile($e, $dest, $true)
            $n++
        }
        if ($n -eq 0) { throw 'REFUSED: the zip is empty' }
    } finally { $zip.Dispose() }
}

function Install-MoeCache {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'      # Invoke-WebRequest is many times faster in Windows PowerShell without its progress bar
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }
    if (-not $Root) {
        if (-not $env:LOCALAPPDATA) { throw 'LOCALAPPDATA is not set: pass -Root DIR' }
        $Root = Join-Path $env:LOCALAPPDATA 'moe-cache'
    }
    $Root = [IO.Path]::GetFullPath($Root)
    $kit = Join-Path $Root 'kit'; $bin = Join-Path $Root 'bin'; $shim = Join-Path $bin 'moe-cache.cmd'; $rec = Join-Path $Root 'install.json'
    $old = $null
    if (Test-Path -LiteralPath $rec) { try { $old = Get-Content -LiteralPath $rec -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $old = $null } }

    if ($Uninstall) {
        if (-not $old) { Say "moe-cache: nothing to remove: $rec does not exist (this installer added nothing under $Root)"; return 0 }
        if ($DryRun) { Say "install.ps1: would remove $kit, $shim, the PATH entry $bin$(if ($old.added_path) { '' } else { ' (not added by us: stays)' }) and $rec"; return 0 }
        if (Test-Path -LiteralPath $kit) { Remove-Item -LiteralPath $kit -Recurse -Force; Say "removed $kit" }
        if ((Test-Path -LiteralPath $shim) -and ((Get-Content -LiteralPath $shim -Raw) -like "*$Marker*")) { Remove-Item -LiteralPath $shim -Force; Say "removed $shim" }
        if ($old.added_path -and -not $NoPath) {
            $list = Split-PathList (Get-UserPathRaw)
            $keep = @($list | Where-Object { -not (Same-Dir $_ $old.bin) })
            if ($keep.Count -ne $list.Count) { Set-UserPathRaw ($keep -join ';'); Say "removed $($old.bin) from your user PATH" }
            $env:Path = ((Split-PathList $env:Path) | Where-Object { -not (Same-Dir $_ $old.bin) }) -join ';'
        }
        Remove-Item -LiteralPath $rec -Force
        foreach ($d in $bin, $Root) { if ((Test-Path -LiteralPath $d) -and -not (Get-ChildItem -LiteralPath $d -Force)) { Remove-Item -LiteralPath $d -Force } }
        Say 'moe-cache was removed. Kept: your models and your settings (the moe-cache folder under your user cache), and any other file you put in these folders.'
        return 0
    }

    $py = Find-Python
    if (-not $py) {
        Say 'Python 3.8 or newer is needed and was not found (a python.exe that only opens the Microsoft Store does not count).'
        Say '  Get it with:   winget install Python.Python.3.12'
        Say '  or from:       https://www.python.org/downloads/   (tick "Add python.exe to PATH")'
        Say 'Then open a NEW terminal and run this installer again. Nothing was installed.'
        return 1
    }
    if (-not $BaseUrl -and $env:MOE_CACHE_BASE_URL) { $BaseUrl = $env:MOE_CACHE_BASE_URL }
    if ($Version -like '@*@') {       # run from a checkout, not from a release: take the newest release (pre-releases included)
        try { $rels = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases?per_page=10" -Headers @{ 'User-Agent' = 'moe-cache-install' } } catch { throw "could not ask GitHub for the newest release ($($_.Exception.Message)); pass -Version X.Y.Z" }
        $Version = ($rels | Where-Object { -not $_.draft } | Select-Object -First 1).tag_name
        if (-not $Version) { throw 'GitHub lists no release; pass -Version X.Y.Z' }
    }
    $Version = $Version.TrimStart('v')
    if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9]+)*$') { throw "odd version '$Version'" }
    $name = "moe-cache-v$Version-windows-x64"
    if (-not $BaseUrl) { $BaseUrl = "https://github.com/$Repo/releases/download/v$Version" }
    Say "moe-cache $Version for Windows: installing into $Root (Python $($py.Version) found)"
    if ($DryRun) { Say "install.ps1: would download $BaseUrl/$name.zip and SHA256SUMS-windows, check the hash, unpack to $kit, write $shim$(if (-not $NoPath) { " and add $bin to your user PATH" })"; return 0 }

    # preflight: what can refuse runs before anything is downloaded or touched
    if ((Test-Path -LiteralPath $shim) -and -not ((Get-Content -LiteralPath $shim -Raw) -like "*$Marker*")) { throw "$shim exists and is not ours: left alone. Nothing was installed." }
    $failAt = $env:MOE_CACHE_INSTALL_FAIL_AT      # tests only: stop at a named step to prove the rollback
    function Fail-At($step) { if ($failAt -eq $step) { throw "test failure injected at $step" } }
    $kitOld = Join-Path $Root 'kit.old'; $stage = Join-Path $Root 'kit.part'
    $same = $old -and $old.version -eq $Version -and (Test-Path -LiteralPath (Join-Path $kit 'bin\moe_cache.py')) -and (Test-Path -LiteralPath (Join-Path $kit '.moe-cache-complete')) -and -not $Force      # the marker is written last into the staged kit: a kit without it is not trusted
    $swapped = $false; $shimTouched = $false; $oldShimText = $null; $recWritten = $false; $pathApplied = $false; $oldRecText = $null
    if (Test-Path -LiteralPath $shim) { $oldShimText = [IO.File]::ReadAllText($shim) }
    if (Test-Path -LiteralPath $rec) { $oldRecText = [IO.File]::ReadAllText($rec) }
    try {
    if ($same) {
        Say "moe-cache $Version is already installed here: repairing the command and the PATH entry only (-Force reinstalls)"
    } else {
        $tmp = Join-Path ([IO.Path]::GetTempPath()) ('moe-cache-install-' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
        [void](New-Item -ItemType Directory -Force -Path $tmp)
        try {
            Say '1/3 download (nothing is installed until the hash matches)'
            Get-File $BaseUrl 'SHA256SUMS-windows' (Join-Path $tmp 'SHA256SUMS-windows')
            Get-File $BaseUrl "$name.zip" (Join-Path $tmp "$name.zip")
            $want = $null
            foreach ($line in Get-Content -LiteralPath (Join-Path $tmp 'SHA256SUMS-windows')) {
                if ($line -match '^\s*([0-9A-Fa-f]{64})\s+\*?(\S+)\s*$' -and $Matches[2] -eq "$name.zip") { $want = $Matches[1]; break }
            }
            if (-not $want) { throw "REFUSED: SHA256SUMS-windows has no line for $name.zip. Nothing was installed." }
            $got = (Get-FileHash -LiteralPath (Join-Path $tmp "$name.zip") -Algorithm SHA256).Hash
            if ($got -ine $want) { throw "REFUSED: the download's sha256 ($got) is not the one in SHA256SUMS-windows ($want). Nothing was installed." }
            Say "sha256 matches SHA256SUMS-windows ($want)"
            Say '2/3 unpack (into a temporary folder; the old kit stays until the new one is complete)'
            if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
            [void](New-Item -ItemType Directory -Force -Path $stage)
            try { Expand-Kit (Join-Path $tmp "$name.zip") $stage } catch { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue; throw }
            if (-not (Test-Path -LiteralPath (Join-Path $stage 'bin\moe_cache.py'))) { Remove-Item -LiteralPath $stage -Recurse -Force; throw 'REFUSED: the zip has no bin\moe_cache.py. Nothing was installed.' }
            [IO.File]::WriteAllText((Join-Path $stage '.moe-cache-complete'), "$Version`r`n")
            Fail-At 'unpack'
            # swap by renames: kit -> kit.old (fails at once when a running llama-server / python holds a file in it, and then nothing has changed),
            # kit.part -> kit; the old kit is deleted last, after the command and the record are written
            try {
                if (Test-Path -LiteralPath $kitOld) { Remove-Item -LiteralPath $kitOld -Recurse -Force }
                if (Test-Path -LiteralPath $kit) { Move-Item -LiteralPath $kit -Destination $kitOld }
                try { Move-Item -LiteralPath $stage -Destination $kit } catch { if ((Test-Path -LiteralPath $kitOld) -and -not (Test-Path -LiteralPath $kit)) { Move-Item -LiteralPath $kitOld -Destination $kit }; throw }
                $swapped = $true
            } catch {
                throw "could not replace $kit ($($_.Exception.Message)). Close a running moe-cache server or llama-server and run this again. The previous install is untouched."
            }
            Fail-At 'swap'
        } finally { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue; if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue } }
    }

    Say '3/3 the moe-cache command'
    [void](New-Item -ItemType Directory -Force -Path $bin)
    $cmd = @(
        '@echo off',
        ($Marker + ' (remove: install.ps1 -Uninstall)'),
        'setlocal',
        'set "KIT=%~dp0..\kit"',
        'set "PY=%MOE_CACHE_PYTHON%"',
        'if defined PY goto run',
        'for /f "delims=" %%P in (''%SystemRoot%\System32\where.exe $PATH:py.exe 2^>nul'') do if not defined PY (',
        '  "%%~fP" -3 -c "import sys" >nul 2>nul && set "PY="%%~fP" -3"',
        ')',
        'if defined PY goto run',
        'for /f "delims=" %%P in (''%SystemRoot%\System32\where.exe $PATH:python.exe 2^>nul'') do if not defined PY (',
        '  "%%~fP" -c "import sys" >nul 2>nul && set "PY="%%~fP""',
        ')',
        'if defined PY goto run',
        'echo Python 3.8 or newer is needed: https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)',
        'exit /b 1',
        ':run',
        '%PY% "%KIT%\bin\moe_cache.py" %*',
        'exit /b %errorlevel%'
    ) -join "`r`n"
    $shimTouched = $true
    [IO.File]::WriteAllText($shim, $cmd + "`r`n", (New-Object Text.ASCIIEncoding))
    Fail-At 'shim'

    $added = $false
    if ($old -and $old.added_path) { $added = $true }       # an earlier run added it: still ours
    $needPath = $false
    if (-not $NoPath) {
        $list = Split-PathList (Get-UserPathRaw)
        if (-not ($list | Where-Object { Same-Dir $_ $bin })) { $needPath = $true; $added = $true }
    }
    $json = [ordered]@{ version = $Version; root = $Root; kit = $kit; bin = $bin; command = $shim; added_path = $added }
    $recWritten = $true
    [IO.File]::WriteAllText($rec, ($json | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
    Fail-At 'record'
    if ($needPath) { $pathApplied = $true; Set-UserPathRaw ((@($list) + $bin) -join ';'); Say "added $bin to your user PATH (new terminals see it)" }
    if (-not $NoPath -and -not ((Split-PathList $env:Path) | Where-Object { Same-Dir $_ $bin })) { $env:Path = $env:Path.TrimEnd(';') + ';' + $bin }
    Fail-At 'path'
    } catch {
        # roll back to the previous install: the old kit, command and record come back, what this run added goes
        $msg = $_.Exception.Message
        try {
            if ($pathApplied) { Set-UserPathRaw ((Split-PathList (Get-UserPathRaw) | Where-Object { -not (Same-Dir $_ $bin) }) -join ';') }
            if ($recWritten) { if ($null -ne $oldRecText) { [IO.File]::WriteAllText($rec, $oldRecText, (New-Object Text.UTF8Encoding($false))) } else { Remove-Item -LiteralPath $rec -Force -ErrorAction SilentlyContinue } }
            if ($shimTouched) { if ($null -ne $oldShimText) { [IO.File]::WriteAllText($shim, $oldShimText, (New-Object Text.ASCIIEncoding)) } else { Remove-Item -LiteralPath $shim -Force -ErrorAction SilentlyContinue } }
            if ($swapped) {
                if (Test-Path -LiteralPath $kit) { Remove-Item -LiteralPath $kit -Recurse -Force }
                if (Test-Path -LiteralPath $kitOld) { Move-Item -LiteralPath $kitOld -Destination $kit }
            }
            if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue }
            foreach ($d in $bin, $Root) { if ((Test-Path -LiteralPath $d) -and -not (Get-ChildItem -LiteralPath $d -Force)) { Remove-Item -LiteralPath $d -Force -ErrorAction SilentlyContinue } }
        } catch { $msg += " (and the rollback hit: $($_.Exception.Message))" }
        throw "$msg`nStopped: the previous install (if any) was put back and the half-done files were removed."
    }
    if (Test-Path -LiteralPath $kitOld) { Remove-Item -LiteralPath $kitOld -Recurse -Force -ErrorAction SilentlyContinue }      # last, best effort

    $shown = (& $shim --version 2>&1 | Select-Object -First 1)
    Say ''
    Say "installed: $shown$(if ($old -and $old.version -ne $Version) { " (was $($old.version))" })"
    Say "  files:   $kit"
    Say "  command: $shim   (folder on your user PATH: $(if ($added) { 'added by this installer' } else { 'was already there' }))"
    Say '  undo:    powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall      (removes exactly the above; your models stay)'
    Say ('           or, without the file:  & ([scriptblock]::Create((irm https://github.com/' + $Repo + '/releases/latest/download/install.ps1))) -Uninstall')
    Say ''
    Say 'Next:  moe-cache launch     (a guided start: checks this PC, suggests a model, starts the server)'
    Say '       (a terminal that was already open needs to be closed and opened again to find the command)'
    return 0
}

$rc = 1
try { $rc = Install-MoeCache } catch { Write-Host ("moe-cache install: " + $_.Exception.Message) -ForegroundColor Red; $rc = 1 }
if ($PSCommandPath) { exit $rc }      # run as a file: the exit code. Under `| iex` we must not close the user's terminal.
