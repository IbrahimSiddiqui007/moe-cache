#!/bin/bash
# Build the Windows kit: tools/make_windows_kit.sh VERSION   (e.g. 0.2.3.6). Needs mingw-w64 (x86_64-w64-mingw32-g++, posix threads) and python3.
# Output: dist/moe-cache-vVERSION-windows-x64.zip + dist/SHA256SUMS-windows. Nothing is published (the windows workflow keeps it as an artifact).
# Like make_release.sh, it is built from a clean export of the last commit (git archive HEAD), never from the working tree.
# Kit layout (the launcher's HERE is the kit folder, so it finds the DLLs there; the .bat looks one folder up from bin\):
#   moe-cache.bat                 one click: the control page (moe-cache.bat --shortcut adds it to the Start menu)
#   moe-cache.exe                 the same as a program (for winget/scoop; it runs moe-cache.bat from its own folder, also through a symlink)
#   README_WINDOWS.txt            what to install and the first commands
#   ggml-moe-cache-api3.dll       for llama.cpp with ggml 0.26 / 0.27 (b11669 and newer); -api2.dll for ggml 0.25
#   bin\ planner\ ui\ tools\ profiles\ docs\ LICENSE   the launcher, the page and what they read (Python 3.8+, standard library only)
set -e
cd "$(dirname "$0")/.."
V="$1"; [ -n "$V" ] || { echo "usage: tools/make_windows_kit.sh VERSION"; exit 1; }
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?([.-][A-Za-z0-9]+)?$ ]] || { echo "VERSION must look like 0.2.3.6 (got '$V')"; exit 1; }
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git checkout: cannot build a clean export"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || echo "warning: uncommitted changes are NOT part of the kit (only the last commit is)"
X="$(mktemp -d)"; trap 'rm -rf "$X"' EXIT
git archive HEAD | tar -x -C "$X"
N=moe-cache-v$V-windows-x64; K="$X/kit/$N"; mkdir -p "$K"
( cd "$X" && BUILD_DIR=build/win-api2 ./build.sh --windows && GGML_API=3 BUILD_DIR=build/win-api3 ./build.sh --windows )
cp "$X/build/win-api2/ggml-moe-cache.dll" "$K/ggml-moe-cache-api2.dll"
cp "$X/build/win-api3/ggml-moe-cache.dll" "$K/ggml-moe-cache-api3.dll"
# moe-cache.exe: runs moe-cache.bat from its own real folder (winget's portable installs need an .exe and start it through a symlink)
"${MINGW_CXX:-x86_64-w64-mingw32-g++}" -x c -O2 -Wall -municode -static -o "$K/moe-cache.exe" "$X/tools/win_launcher.c"   # MINGW_CXX as build.sh
# only system DLLs and llama.cpp's own ggml DLLs may be imported (libwinpthread-1.dll kept it from loading on the Claw)
for d in "$K"/ggml-moe-cache-api*.dll "$K/moe-cache.exe"; do
  i=$(x86_64-w64-mingw32-objdump -p "$d" | grep "DLL Name") || { echo "FAIL: no import list for $d"; exit 1; }
  if echo "$i" | grep -v -i -E "KERNEL32|api-ms-win-crt|msvcrt|ggml-base.dll|ggml.dll|USER32|ADVAPI32"; then echo "FAIL: $d imports a non-system DLL"; exit 1; fi
done
mkdir -p "$K/bin" "$K/tools" "$K/docs"
cp "$X"/bin/*.py "$X"/bin/moe-cache-ui.bat "$K/bin/"     # every launcher module (v0.3.0 left out moe_cache_pick.py: the control page did not start)
cp -r "$X/planner" "$X/ui" "$X/profiles" "$K/"
cp "$X"/tools/*.ps1 "$X"/tools/learn_prompts.txt "$X"/tools/agent_bench.py "$X"/tools/loop_check.py "$K/tools/"
cp "$X"/docs/WINDOWS.md "$X"/docs/WINDOWS_TEST.md "$X"/docs/GUI.md "$X"/docs/AGENTS.md "$X"/docs/COMPATIBILITY.md "$X"/docs/SHARE_YOUR_NUMBERS.md "$X"/docs/TROUBLESHOOTING.md "$K/docs/"   # WINDOWS.md links WINDOWS_TEST.md
cp "$X/LICENSE" "$X/CHANGELOG.md" "$K/"
# every module of the kit's bin\ imports from there alone (nothing from the repository): catches a launcher file left out of the copy above
( cd "$K/bin" && XDG_CACHE_HOME="$X/xdg" python3 -I -c "import sys; sys.path.insert(0, '.'); import moe_cache, moe_cache_ui, moe_cache_on_demand, moe_cache_pick, moe_cache_pack" ) \
  || { echo "FAIL: a launcher module does not import from the kit's bin\\ (a file missing from the copy?)"; exit 1; }
for m in "$K"/bin/*.py; do grep -oE "^\s*(import|from) moe_cache[a-z_]*" "$m"; done | awk '{print $NF}' | sort -u | while read -r mod; do
  [ -f "$K/bin/$mod.py" ] || { echo "FAIL: $mod.py is imported but not in the kit"; exit 1; }
done
sed "s/@VERSION@/$V/g; s/@COMMIT@/$(git rev-parse --short HEAD)/g" "$X/tools/README_WINDOWS.txt" | sed 's/$/\r/' > "$K/README_WINDOWS.txt"
printf '@echo off\r\nrem moe-cache control page: double-click. "moe-cache.bat --shortcut" adds it to the Start menu.\r\ncall "%%~dp0bin\\moe-cache-ui.bat" %%*\r\n' > "$K/moe-cache.bat"
find "$K" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p dist
python3 - "$X/kit" "$N" "$PWD/dist/$N.zip" <<'EOF'
import os, sys, zipfile
root, name, out = sys.argv[1:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d, _, files in sorted(os.walk(os.path.join(root, name))):
        for f in sorted(files):
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, root))
EOF
( cd dist && sha256sum "$N.zip" > SHA256SUMS-windows && cat SHA256SUMS-windows )
echo "built from commit $(git rev-parse --short HEAD) -> dist/$N.zip"
