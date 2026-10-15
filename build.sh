#!/bin/bash
# Build the plugin. It needs a C++17 compiler and the directory that contains libggml-base.so (it comes with your llama.cpp install).
#   GGML_LIB_DIR   directory with libggml-base.so (default: searched in common places and next to llama-server)
#   --portable     build a library for the prebuilt release: no link to libggml-base (symbols come from the llama-server that loads it) and no glibc 2.38 symbols.
#   --windows      cross-compile ggml-moe-cache.dll with mingw-w64 (x86_64-w64-mingw32-g++, dlltool). Load it with GGML_BACKEND_PATH. UNTESTED on real Windows.
# The ggml headers are included in third_party/, so no llama.cpp source is needed. They match ggml 0.25.1 (llama.cpp commit 7fe450e19).
#   GGML_API=3     build against the ggml 0.26 headers (backend API 3, llama.cpp commit 23b0202a1) instead of 0.25.1 (API 2)
#   BUILD_DIR      output directory (default build; use another one to keep an API 2 and an API 3 build side by side)
set -e
cd "$(dirname "$0")"
HDR=third_party/ggml; OUT="${BUILD_DIR:-build}"
if [ "${GGML_API:-2}" = 3 ]; then HDR=third_party/ggml-v3; fi
if [ "$1" = "--windows" ]; then
  M=x86_64-w64-mingw32; mkdir -p build/win
  $M-g++ -O2 -std=c++17 -Wall -Wno-unused-function -I$HDR/include -I$HDR/src -c src/moe_cache_plugin.cpp -o build/win/plugin.o
  # ggml's functions live in two DLLs: the backend registry (ggml-backend-reg.cpp) in ggml.dll, everything else in ggml-base.dll. Importing all of
  # them from ggml-base.dll made Windows refuse to load the plugin (found on the MSI Claw, 2026-10-15). Same 16 names in ggml 0.25.1 and 0.26.0.
  REG=" ggml_backend_dev_by_name ggml_backend_dev_by_type ggml_backend_dev_count ggml_backend_dev_get ggml_backend_device_register ggml_backend_init_best ggml_backend_init_by_name ggml_backend_init_by_type ggml_backend_load ggml_backend_load_all ggml_backend_load_all_from_path ggml_backend_reg_by_name ggml_backend_reg_count ggml_backend_reg_get ggml_backend_register ggml_backend_unload "
  $M-nm -u build/win/plugin.o | awk '$1=="U" && ($2 ~ /^ggml_/ || $2 ~ /^gguf_/) {print $2}' | sort -u > build/win/imports.txt
  ( echo "LIBRARY ggml-base.dll"; echo EXPORTS; while read -r f; do case "$REG" in *" $f "*) ;; *) echo "$f";; esac; done < build/win/imports.txt ) > build/win/ggml-base.def
  ( echo "LIBRARY ggml.dll"; echo EXPORTS; while read -r f; do case "$REG" in *" $f "*) echo "$f";; esac; done < build/win/imports.txt ) > build/win/ggml.def
  $M-dlltool -d build/win/ggml-base.def -l build/win/libggml-base.dll.a -D ggml-base.dll
  $M-dlltool -d build/win/ggml.def -l build/win/libggml.dll.a -D ggml.dll
  # -static links the mingw threads runtime in: a DLL that needs libwinpthread-1.dll does not load on a normal Windows machine (found on the Claw)
  $M-g++ -shared -o build/win/ggml-moe-cache.dll build/win/plugin.o build/win/libggml-base.dll.a build/win/libggml.dll.a -static -static-libgcc -static-libstdc++ -Wl,--exclude-all-symbols
  echo "built build/win/ggml-moe-cache.dll (not tested on real Windows)"
  exit 0
fi
# MOE_CACHE_CXXFLAGS: extra compiler flags (CI uses it for a test-only mutant build)
HARDEN="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2 -fstack-protector-strong"   # same on every compiler, not only on distros that turn it on
LDHARDEN="-Wl,-z,relro,-z,now"
if [ "$1" = "--portable" ]; then
  mkdir -p $OUT
  gcc -O2 -fPIC -std=gnu99 -U_GNU_SOURCE -c src/compat_glibc.c -o $OUT/compat_glibc.o 2>/dev/null || gcc -O2 -fPIC -std=c99 -c src/compat_glibc.c -o $OUT/compat_glibc.o
  g++ -O2 $HARDEN ${MOE_CACHE_CXXFLAGS:-} -fPIC -shared -std=c++17 -Wall -Wno-unused-function -I$HDR/include -I$HDR/src src/moe_cache_plugin.cpp $OUT/compat_glibc.o \
      -o $OUT/libggml-moe-cache.so -lpthread -ldl -Wl,-z,undefs -Wl,--as-needed $LDHARDEN
  echo "built portable $OUT/libggml-moe-cache.so"
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
mkdir -p $OUT
echo "using libggml-base from $LIBS"
g++ -O2 $HARDEN ${MOE_CACHE_CXXFLAGS:-} -fPIC -shared -std=c++17 -Wall -Wno-unused-function -I$HDR/include -I$HDR/src src/moe_cache_plugin.cpp -o $OUT/libggml-moe-cache.so -lpthread -ldl "$BASE" -Wl,-rpath,"$LIBS" $LDHARDEN
echo "built $OUT/libggml-moe-cache.so"
