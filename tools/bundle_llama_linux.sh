#!/bin/bash
# Put the official, unmodified llama.cpp Linux build into a release tarball folder (the Linux twin of the Windows kit's llama\ folder).
#   tools/bundle_llama_linux.sh DEST TAG ASSET SHA256 [CACHE_DIR]
#     DEST   the tarball's folder; the build goes to DEST/llama/ (llama-server, the ggml libraries, LICENSE-llama.cpp.txt, README-moe-cache.txt)
#     TAG    the llama.cpp release, e.g. b11966
#     ASSET  the release asset, e.g. llama-b11966-bin-ubuntu-vulkan-x64.tar.gz
#     SHA256 the asset's digest from `gh api repos/ggml-org/llama.cpp/releases/tags/TAG` (the download must match it)
#     CACHE_DIR where the asset is kept (default: $LLAMA_TGZ_CACHE or /tmp); an asset already there is not downloaded again
# LLAMA_LICENSE_FILE=PATH copies that licence file instead of fetching llama.cpp's from GitHub (tests, offline builds).
set -e
D="$1"; TAG="$2"; ASSET="$3"; SHA="$4"; CACHE="${5:-${LLAMA_TGZ_CACHE:-/tmp}}"
[ -n "$D" ] && [ -n "$TAG" ] && [ -n "$ASSET" ] && [ -n "$SHA" ] || { sed -n 2,11p "$0"; exit 1; }
mkdir -p "$CACHE"; Z="$CACHE/$ASSET"
[ -f "$Z" ] || curl -fsSL -o "$Z" "https://github.com/ggml-org/llama.cpp/releases/download/$TAG/$ASSET"
echo "$SHA  $Z" | sha256sum -c - >/dev/null || { echo "FAIL: $ASSET sha256 does not match the release asset"; exit 1; }
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
tar --no-same-owner -C "$T" -xzf "$Z"
S="$(find "$T" -type f -name llama-server | head -n 1)"
[ -n "$S" ] || { echo "FAIL: no llama-server in $ASSET"; exit 1; }
rm -rf "$D/llama"; mkdir -p "$D/llama"
cp -a "$(dirname "$S")"/. "$D/llama/"        # the whole folder as shipped (symlinks kept): llama-server finds its libraries next to itself
chmod +x "$D/llama/llama-server"
if [ -n "$LLAMA_LICENSE_FILE" ]; then cp "$LLAMA_LICENSE_FILE" "$D/llama/LICENSE-llama.cpp.txt"
else curl -fsSL -o "$D/llama/LICENSE-llama.cpp.txt" "https://raw.githubusercontent.com/ggml-org/llama.cpp/$TAG/LICENSE"; fi
printf 'Official llama.cpp %s build for Linux x86-64, unmodified (%s, sha256 %s), from\nhttps://github.com/ggml-org/llama.cpp/releases/tag/%s\nLicence: MIT (LICENSE-llama.cpp.txt). moe-cache finds it here; set LLAMA_SERVER to use another build.\nThe Vulkan build needs the system Vulkan loader and a driver: sudo apt install libvulkan1 mesa-vulkan-drivers (Ubuntu / Debian). Without a GPU it runs on the CPU.\n' \
  "$TAG" "$ASSET" "$SHA" "$TAG" > "$D/llama/README-moe-cache.txt"
echo "bundled llama.cpp $TAG ($ASSET) in $D/llama"
