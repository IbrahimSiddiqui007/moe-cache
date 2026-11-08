#!/bin/bash
# Put moe-cache's PATCHED llama.cpp build (CI artifacts of .github/workflows/llama_patched.yml) into a kit folder: DEST/llama/.
#   tools/bundle_llama_patched.sh DEST SRC_DIR linux|win [TAG]
#     DEST    the kit's folder; the build goes to DEST/llama/
#     SRC_DIR the unpacked CI build: linux = llama-server + the .so files (symlinks kept); win = llama-server.exe, the CPU/ggml DLLs and ggml-vulkan.dll
#     TAG     the llama.cpp base the patches apply to (default b12207)
# Writes next to the server: PATCHES-moe-cache.txt (one line, read by the launcher and printed at start-up), README-moe-cache.txt, LICENSE-llama.cpp.txt.
# Every patch in patches/llama.cpp/ is built in and OFF; each has its own environment switch (patches/llama.cpp/README.md).
# LLAMA_LICENSE_FILE=PATH copies that licence file instead of fetching llama.cpp's from GitHub (tests, offline builds).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
D="$1"; S="$2"; P="$3"; TAG="${4:-b12207}"
[ -n "$D" ] && [ -d "$S" ] && { [ "$P" = linux ] || [ "$P" = win ]; } || { sed -n 2,9p "$0"; exit 1; }
EXE=llama-server; [ "$P" = win ] && EXE=llama-server.exe
[ -f "$S/$EXE" ] || { echo "FAIL: no $EXE in $S"; exit 1; }
[ "$P" = linux ] || [ -f "$S/ggml-vulkan.dll" ] || { echo "FAIL: no ggml-vulkan.dll in $S (the Vulkan artifact was not unpacked next to the server)"; exit 1; }
NUMS="$(cd "$HERE/patches/llama.cpp" && ls -1 [0-9]*.patch | sed 's/-.*//' | paste -sd, -)"
[ -n "$NUMS" ] || { echo "FAIL: no patches in patches/llama.cpp"; exit 1; }
LINE="llama.cpp $TAG + moe-cache patches $NUMS built in, all off"
rm -rf "$D/llama"; mkdir -p "$D/llama"
cp -a "$S"/. "$D/llama/"
[ "$P" = linux ] && chmod +x "$D/llama/llama-server"
if [ -n "$LLAMA_LICENSE_FILE" ]; then cp "$LLAMA_LICENSE_FILE" "$D/llama/LICENSE-llama.cpp.txt"
else "$HERE/tools/ci/fetch.sh" "https://raw.githubusercontent.com/ggml-org/llama.cpp/$TAG/LICENSE" "$D/llama/LICENSE-llama.cpp.txt" || exit 1; fi    # retried with backoff
NL='\n'; [ "$P" = win ] && NL='\r\n'
printf '%s%b' "$LINE" "$NL" > "$D/llama/PATCHES-moe-cache.txt"
{
  printf 'llama.cpp %s with moe-cache patches (%s), built from the official source by GitHub Actions (.github/workflows/llama_patched.yml).%b' "$TAG" "$NUMS" "$NL"
  printf 'The patches are in patches/llama.cpp/ of the moe-cache repository; the build is otherwise unmodified. Licence: MIT (LICENSE-llama.cpp.txt).%b' "$NL"
  printf 'Every patch is built in but OFF: each has its own environment switch and does nothing until it is set (patches/llama.cpp/README.md). With none set the server behaves as the official build.%b' "$NL"
  printf 'To use the official build instead, download it from https://github.com/ggml-org/llama.cpp/releases/tag/%s and point LLAMA_SERVER at its llama-server. moe-cache finds this one here otherwise.%b' "$TAG" "$NL"
  if [ "$P" = linux ]; then printf 'The Vulkan build needs the system Vulkan loader and a driver: sudo apt install libvulkan1 mesa-vulkan-drivers (Ubuntu / Debian). Without a GPU it runs on the CPU.\n'; fi
} > "$D/llama/README-moe-cache.txt"
echo "bundled patched llama.cpp ($LINE) in $D/llama"
