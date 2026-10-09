#!/bin/bash
# Build the plugin. It needs a C++17 compiler and the directory that contains libggml-base.so (it comes with your llama.cpp install).
#   GGML_LIB_DIR   directory with libggml-base.so (default: searched in common places and next to llama-server)
#   --portable     build a library for the prebuilt release: no link to libggml-base (symbols come from the llama-server that loads it) and no glibc 2.38 symbols.
# The ggml headers are included in third_party/, so no llama.cpp source is needed. They match ggml 0.25.1 (llama.cpp commit 7fe450e19).
set -e
cd "$(dirname "$0")"
if [ "$1" = "--portable" ]; then
  mkdir -p build
  gcc -O2 -fPIC -std=gnu99 -U_GNU_SOURCE -c src/compat_glibc.c -o build/compat_glibc.o 2>/dev/null || gcc -O2 -fPIC -std=c99 -c src/compat_glibc.c -o build/compat_glibc.o
  g++ -O2 -fPIC -shared -std=c++17 -Wall -Wno-unused-function -Ithird_party/ggml/include -Ithird_party/ggml/src src/moe_cache_plugin.cpp build/compat_glibc.o \
      -o build/libggml-moe-cache.so -lpthread -Wl,-z,undefs -Wl,--as-needed
  echo "built portable build/libggml-moe-cache.so"
  exit 0
fi
LIBS="${GGML_LIB_DIR:-}"
if [ -z "$LIBS" ]; then
  LS="$(command -v "${LLAMA_SERVER:-llama-server}" 2>/dev/null || true)"
  for d in ${LS:+"$(dirname "$(readlink -f "$LS")")"} ${LS:+"$(dirname "$(readlink -f "$LS")")/../lib"} /usr/lib/llama.cpp-cuda /usr/local/lib /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu; do
    if ls "$d"/libggml-base.so* >/dev/null 2>&1; then LIBS="$d"; break; fi
  done
fi
[ -n "$LIBS" ] || { echo "libggml-base.so not found. Set GGML_LIB_DIR to the directory of your llama.cpp libraries."; exit 1; }
BASE="$(ls "$LIBS"/libggml-base.so "$LIBS"/libggml-base.so.* 2>/dev/null | head -1)"
mkdir -p build
echo "using libggml-base from $LIBS"
g++ -O2 -fPIC -shared -std=c++17 -Wall -Wno-unused-function -Ithird_party/ggml/include -Ithird_party/ggml/src src/moe_cache_plugin.cpp -o build/libggml-moe-cache.so -lpthread "$BASE" -Wl,-rpath,"$LIBS"
echo "built build/libggml-moe-cache.so"
