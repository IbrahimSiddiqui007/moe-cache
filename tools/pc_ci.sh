#!/bin/bash
# Daily-work CI on the owner's Windows PC instead of GitHub: tools/pc_ci.sh REF [--model] [--no-pc]
#   REF       branch, tag or SHA of this repository (pushes nothing; the exact commit is exported with git archive)
#   --model   also run the steps that load the small Granite model (what windows.yml's identity steps do), max 15 min
#   --no-pc   only the laptop part (cross-build of the DLL, kit and compile checks), nothing is sent to the PC
# What runs where (see docs/internal/PC_CI.md):
#   laptop  = windows.yml's `dll` job: mingw-w64 cross-build of the plugin DLL, the Windows test programs and the kit (GitHub does this on Linux too)
#   PC      = windows.yml's `windows` job: the tests on real Windows (tools/pc_ci_windows.ps1); `--model` adds the Granite steps
# Not covered: identity.yml (Linux, builds llama.cpp three times) and windows.yml's `installer` job (changes the real PATH / Start menu): GitHub only.
# Refuses while a PC chain (C:\dl\afternoon\runner.pid) or any llama-server runs. Output: one table (step, result, minutes) and the log folder; exit 0 only if nothing failed.
# Env: PC_HOST (default ibrrum@100.122.164.85), PC_CI_LOGS (default ~/.cache/pc_ci)
set -uo pipefail
cd "$(dirname "$0")/.."
REF=""; MODEL=0; NOPC=0
for a in "$@"; do case "$a" in --model) MODEL=1 ;; --no-pc) NOPC=1 ;; -h|--help) sed -n 2,12p "$0"; exit 0 ;; -*) echo "unknown option $a" >&2; exit 2 ;; *) REF="$a" ;; esac; done
[ -n "$REF" ] || { echo "usage: tools/pc_ci.sh REF [--model] [--no-pc]" >&2; exit 2; }
SHA=$(git rev-parse --verify -q "$REF^{commit}" || git rev-parse --verify -q "origin/$REF^{commit}") || { echo "pc_ci: unknown ref $REF (fetch it first; nothing is fetched or pushed here)" >&2; exit 2; }
PC="${PC_HOST:-ibrrum@100.122.164.85}"
STAMP=$(date +%Y%m%d_%H%M%S); NAME="${STAMP}_${SHA:0:10}"
LOGDIR="${PC_CI_LOGS:-$HOME/.cache/pc_ci}/$NAME"; mkdir -p "$LOGDIR"
ROWS="$LOGDIR/results.tsv"; : > "$ROWS"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 "$PC")
pcssh() { "${SSH[@]}" "$@" 2> >(grep -v -e '^\*\*' >&2); }
FAIL=0
row() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$ROWS"; [ "$3" = pass ] || [[ "$3" == skipped* ]] || FAIL=1; }
step() {      # step NAME COMMAND...: runs, times, logs to $LOGDIR/laptop.log
  local name="$1"; shift; local t0=$SECONDS
  echo "[pc_ci] $name"; echo "=== $name" >> "$LOGDIR/laptop.log"
  if "$@" >> "$LOGDIR/laptop.log" 2>&1; then row "$name" laptop pass $((SECONDS - t0)); else row "$name" laptop FAIL $((SECONDS - t0)); return 1; fi
}
summary() {
  echo; echo "commit $SHA ($REF)"; echo "log folder: $LOGDIR"
  { printf 'step\twhere\tresult\tminutes\n'; awk -F'\t' '{printf "%s\t%s\t%s\t%.2f\n",$1,$2,$3,$4/60}' "$ROWS"; } | column -t -s $'\t'
  if [ $FAIL = 0 ]; then echo "ALL PASS$([ $MODEL = 0 ] && grep -q 'skipped' "$ROWS" && echo ' (model steps skipped: run with --model for the identity steps)')"; else echo "FAILED (see the log folder)"; fi
  exit $FAIL
}

# ---- 0. PC free? (before any work, so a busy PC costs nothing)
TOOLS_WIN='C:\dl\pc_ci\tools'
if [ $NOPC = 0 ]; then
  echo "[pc_ci] checking that the PC is free"
  printf 'New-Item -ItemType Directory -Force C:\\dl\\pc_ci\\tools,C:\\dl\\pc_ci\\in,C:\\dl\\pc_ci\\work,C:\\dl\\pc_ci\\cache | Out-Null\n' | pcssh 'powershell -NoProfile -Command -' >> "$LOGDIR/laptop.log" || { row "reach PC" laptop FAIL 0; summary; }
  scp -q -o BatchMode=yes tools/pc_ci_guard.ps1 tools/pc_ci_windows.ps1 "$PC:C:/dl/pc_ci/tools/" 2>>"$LOGDIR/laptop.log" || { row "copy scripts to PC" laptop FAIL 0; summary; }
  t0=$SECONDS
  if ! pcssh "powershell -NoProfile -ExecutionPolicy Bypass -File $TOOLS_WIN\\pc_ci_guard.ps1" | tee -a "$LOGDIR/laptop.log"; then :; fi
  if ! grep -q "PC is free" "$LOGDIR/laptop.log"; then row "PC free check" laptop "FAIL (PC busy: see log)" $((SECONDS - t0)); summary; fi
  row "PC free check" laptop pass $((SECONDS - t0))
fi

# ---- 1. laptop: the `dll` job of windows.yml, from a clean export of exactly this commit
B=$(mktemp -d /tmp/pc_ci.XXXXXX); SRC="$B/src"
cleanup() { git worktree remove --force "$SRC" >/dev/null 2>&1; rm -rf "$B"; }
trap cleanup EXIT
step "export commit" git worktree add --detach "$SRC" "$SHA" || summary
if command -v x86_64-w64-mingw32ucrt-g++ >/dev/null; then export MINGW_CXX=x86_64-w64-mingw32ucrt-g++; else export MINGW_CXX=x86_64-w64-mingw32-g++; fi
echo "compiler: $($MINGW_CXX --version | head -1)" >> "$LOGDIR/laptop.log"
dll_job() {
  cd "$SRC"
  set -e
  $MINGW_CXX -std=c++17 -Wall -Werror -Wno-unused-function -Ithird_party/ggml-v3/include -Ithird_party/ggml-v3/src -fsyntax-only src/moe_cache_plugin.cpp
  GGML_API=3 MOE_CACHE_CXXFLAGS="-g" ./build.sh --windows
  x86_64-w64-mingw32-objdump -p build/win/ggml-moe-cache.dll | grep "DLL Name" | tee build/win/imports_dlls.txt
  grep -q -i "KERNEL32" build/win/imports_dlls.txt
  if grep -v -i -E "KERNEL32|api-ms-win-crt|msvcrt|ggml-base.dll|ggml.dll|USER32|ADVAPI32" build/win/imports_dlls.txt; then echo "FAIL: the DLL imports a DLL that is not a system or ggml DLL"; exit 1; fi
  # the test programs are built in parallel (16 cores)
  $MINGW_CXX -std=c++17 -O2 -Isrc tests/platform_test.cpp -o build/win/platform_test.exe -static &
  $MINGW_CXX -std=c++17 -O2 -Isrc tests/open_hint_test.cpp -o build/win/open_hint_test.exe -static &
  $MINGW_CXX -std=c++17 -O2 -Wall -Werror -Wno-unused-function -DMOE_CACHE_TEST_INPAGE_ERROR -Isrc tests/file_view_load_test.cpp -o build/win/file_view_load_test.exe -static &
  $MINGW_CXX -std=c++17 -O2 -Wall -Werror -Isrc tests/win_read_bench.cpp -o build/win/win_read_bench.exe -static &
  $MINGW_CXX -std=c++17 -O2 -Isrc tests/kernel_check.cpp -o build/win/kernel_check.exe -static &
  wait -n; wait
  for f in platform_test open_hint_test file_view_load_test win_read_bench kernel_check; do [ -f build/win/$f.exe ] || { echo "FAIL: $f.exe not built"; exit 1; }; done
}
step "dll + test programs (mingw cross-build, warnings are errors)" bash -c "$(declare -f dll_job); MINGW_CXX=$MINGW_CXX SRC=$SRC; dll_job" || summary
kit_job() {
  cd "$SRC"; mkdir -p "$HOME/.cache/pc_ci"
  LLAMA_ZIP_CACHE="$HOME/.cache/pc_ci" tools/make_windows_kit.sh 0.0.0-ci
}
step "windows kit (zip, launcher modules import check)" bash -c "$(declare -f kit_job); SRC=$SRC; kit_job" || summary

# ---- 2. ship to the PC and run the windows job there
if [ $NOPC = 1 ]; then summary; fi
WORK="C:\\dl\\pc_ci\\work\\$NAME"; WORKU="C:/dl/pc_ci/work/$NAME"; INU="C:/dl/pc_ci/in"
t0=$SECONDS
ship() {
  set -e
  git archive --format=tar "$SHA" > "$B/repo.tar"
  ( cd "$SRC/build/win" && tar -cf "$B/build.tar" ggml-moe-cache.dll platform_test.exe open_hint_test.exe file_view_load_test.exe win_read_bench.exe kernel_check.exe )
  cp "$SRC/dist/moe-cache-v0.0.0-ci-windows-x64.zip" "$B/kit.zip"
  scp -q -o BatchMode=yes "$B/repo.tar" "$B/build.tar" "$B/kit.zip" "$PC:$INU/"
  printf '$w = "%s"\nNew-Item -ItemType Directory -Force "$w\\build" | Out-Null\ntar -xf C:\\dl\\pc_ci\\in\\repo.tar -C $w\ntar -xf C:\\dl\\pc_ci\\in\\build.tar -C "$w\\build"\nCopy-Item C:\\dl\\pc_ci\\in\\kit.zip "$w\\kit.zip"\nif (-not (Test-Path "$w\\bin\\moe_cache.py")) { exit 1 }\n' "$WORK" | pcssh 'powershell -NoProfile -Command -'
}
step "ship commit, DLL and kit to the PC" bash -c "$(declare -f ship pcssh); SHA=$SHA B=$B SRC=$SRC PC=$PC INU=$INU WORK='$WORK'; SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 $PC); ship" || summary

echo "[pc_ci] running the windows steps on the PC$([ $MODEL = 1 ] && echo ' (with the Granite model steps)')"
PCARGS="-Work $WORK -BuildDir $WORK\\build -KitZip $WORK\\kit.zip -CacheDir C:\\dl\\pc_ci\\cache"
[ $MODEL = 1 ] && PCARGS="$PCARGS -Model"
pcssh "powershell -NoProfile -ExecutionPolicy Bypass -File $TOOLS_WIN\\pc_ci_windows.ps1 $PCARGS" 2>&1 | tee "$LOGDIR/pc_console.log" >/dev/null
scp -q -o BatchMode=yes "$PC:$WORKU/results.tsv" "$LOGDIR/pc_results.tsv" 2>>"$LOGDIR/laptop.log"
scp -q -o BatchMode=yes "$PC:$WORKU/pc_ci.log" "$LOGDIR/pc_ci.log" 2>>"$LOGDIR/laptop.log"
if [ -s "$LOGDIR/pc_results.tsv" ]; then
  # the PC script ends with a "pc_ci finished" row; without it (ssh dropped, script died) the run is incomplete and must not pass
  grep -q -P '^pc_ci finished\t' "$LOGDIR/pc_results.tsv" || row "PC run did not finish (no end marker: connection lost?)" PC FAIL $((SECONDS - t0))
  tail -n +2 "$LOGDIR/pc_results.tsv" | tr -d '\r' | while IFS=$'\t' read -r s g r sec; do printf '%s\t%s\t%s\t%s\n' "$s" "PC/$g" "$r" "$sec"; done >> "$ROWS"
  grep -q -P '\tFAIL' <(tail -n +2 "$LOGDIR/pc_results.tsv" | tr -d '\r') && FAIL=1
else
  row "PC steps (no results came back)" PC FAIL $((SECONDS - t0))
fi
echo "PC work folder (kept): $WORK"
summary
