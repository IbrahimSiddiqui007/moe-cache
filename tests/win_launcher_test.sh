#!/usr/bin/env bash
# Model-free check of tools/win_launcher.c (AUD-003): moe-cache.exe must hand its own redirected/piped stdin, stdout and stderr to the batch file.
# Needs x86_64-w64-mingw32-gcc to compile; needs wine to run. Without mingw it SKIPS (exit 0, visible message); with mingw but no wine it only compiles.
# On the Windows PC the same check by hand: `moe-cache.exe --version > f.txt` must put the version in f.txt, `echo y | moe-cache.exe ...` must see stdin.
set -u
cd "$(dirname "$0")/.."
CC="${MINGW_CC:-x86_64-w64-mingw32-gcc}"
# source check that does not need mingw: the three std handles must be passed on (wine passes them through even without this, real Windows does not)
for w in STARTF_USESTDHANDLES hStdInput hStdOutput hStdError GetStdHandle HANDLE_FLAG_INHERIT; do
  grep -q "$w" tools/win_launcher.c || { echo "FAIL: tools/win_launcher.c does not use $w (std handles are not passed to cmd.exe)"; exit 1; }
done
echo "ok: win_launcher.c passes the std handles on"
if ! command -v "$CC" >/dev/null 2>&1; then echo "SKIP: win_launcher_test needs $CC (mingw-w64); nothing checked"; exit 0; fi
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
"$CC" -x c -O2 -Wall -Werror -municode -static -DMC_VERSION='"9.9.9"' -o "$T/moe-cache.exe" tools/win_launcher.c || { echo "FAIL: win_launcher.c does not compile"; exit 1; }
echo "ok: win_launcher.c compiles"
if ! command -v wine >/dev/null 2>&1; then echo "SKIP: wine not found, redirect behaviour not run (compile only)"; exit 0; fi
# a stand-in kit: the batch file prints a line to stdout, a line to stderr, and echoes what it reads from stdin
printf '@echo off\r\necho OUT-%%1\r\necho ERR-%%1 1>&2\r\nset /p L=\r\necho IN-%%L%%\r\n' > "$T/moe-cache.bat"
export WINEPREFIX="$T/prefix" WINEDEBUG=-all WINEDLLOVERRIDES="mscoree,mshtml="
cd "$T"
# WINDOWS-675: --version and --help print and exit without starting the batch file
wine ./moe-cache.exe --version 2>/dev/null | tr -d '\r' | grep -q '^moe-cache 9.9.9$' || { echo "FAIL: moe-cache.exe --version"; exit 1; }
echo "ok: moe-cache.exe --version prints and exits"
echo hello | wine ./moe-cache.exe xyz > out.txt 2> err.txt
rc=$?
fail=0
grep -q 'OUT-xyz' out.txt || { echo "FAIL: stdout redirect lost (out.txt: $(tr -d '\r' < out.txt | head -c 200))"; fail=1; }
grep -q 'IN-hello' out.txt || { echo "FAIL: piped stdin not seen by the batch file"; fail=1; }
grep -q 'ERR-xyz' err.txt || { echo "FAIL: stderr redirect lost"; fail=1; }
[ "$rc" = 0 ] || { echo "FAIL: exit code $rc"; fail=1; }
[ "$fail" = 0 ] && echo "ok: stdin, stdout and stderr reach moe-cache.bat"
exit $fail
