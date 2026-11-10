#!/bin/bash
# build_top.sh: the native Monitor `moe-cache-top` (R15-03, phase 0 spike): src/top/main.cpp + the vendored FTXUI 7.0.3 sources (third_party/ftxui, sha256 pinned in
# third_party/THIRD_PARTY.txt; nothing is fetched at build time).
#   ./build_top.sh              Linux: bin/moe-cache-top   (g++, C++17, -O2, glibc dynamic)
#   ./build_top.sh --windows    Windows .exe with the mingw UCRT cross compiler (x86_64-w64-mingw32-g++, static): bin/moe-cache-top.exe
#   JOBS=4 ./build_top.sh       parallel compile jobs (default 4)
cd "$(dirname "$0")" || exit 1
set -e
CXX=${CXX:-g++}; OUT=bin/moe-cache-top; EXTRA="-pthread"; LIBS=""
if [ "${1:-}" = "--windows" ]; then
  CXX=${CXX_WIN:-x86_64-w64-mingw32-g++}; OUT=bin/moe-cache-top.exe; EXTRA="-static -static-libgcc -static-libstdc++ -DUNICODE -D_UNICODE -DNOMINMAX -D_WIN32_WINNT=0x0A00"; LIBS="-lws2_32"
fi
J=${JOBS:-4}; B=build/top; if [ "$OUT" = bin/moe-cache-top.exe ]; then B=build/top-win; fi
mkdir -p "$B" bin
FL="-std=c++17 -O2 -Ithird_party/ftxui/include -Ithird_party/ftxui/src $EXTRA"
SRC=$(find third_party/ftxui/src -name '*.cpp' | sort)
objs=""; todo="$B/todo.txt"; : > "$todo"
for f in $SRC src/top/main.cpp; do
  o="$B/$(echo "$f" | tr '/' '_').o"; objs="$objs $o"
  if [ ! -f "$o" ] || [ "$f" -nt "$o" ]; then echo "$CXX $FL -c $f -o $o" >> "$todo"; fi
done
xargs -P "$J" -I{} sh -c '{}' < "$todo"
$CXX $FL $objs -o "$OUT" $LIBS
strip "$OUT" 2>/dev/null || ${CXX%g++}strip "$OUT" 2>/dev/null || true
ls -l "$OUT"
