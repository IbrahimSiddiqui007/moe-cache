#!/bin/bash
# PACKAGING-023: build.sh refuses a GGML_API other than 2 or 3 and picks the matching header set. Model-free and compiler-free:
# fake gcc/g++ in PATH record their arguments (--portable path), nothing is built.
cd "$(dirname "$0")/.." || exit 1
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir "$T/bin"
for c in gcc g++; do printf '#!/bin/sh\necho "$@" >> "$FAKE_LOG"\n' > "$T/bin/$c"; chmod +x "$T/bin/$c"; done
fail=0
ok() { echo "  ok   $1"; }
bad() { echo "  FAIL $1"; fail=1; }
run() { : > "$T/log"; FAKE_LOG="$T/log" PATH="$T/bin:$PATH" BUILD_DIR="$T/out" "$@" > "$T/stdout" 2> "$T/stderr"; }

for v in 4 1 0 abc 23; do
  if GGML_API=$v run ./build.sh --portable; then bad "GGML_API=$v is accepted"
  elif grep -q "GGML_API must be 2 or 3" "$T/stderr" && [ ! -s "$T/log" ]; then ok "GGML_API=$v: error, compiler never called"
  else bad "GGML_API=$v: wrong failure ($(cat "$T/stderr"))"; fi
done
if GGML_API= run ./build.sh --portable; then ok "GGML_API empty = default (2)"; else bad "empty GGML_API refused: $(cat "$T/stderr")"; fi
if run ./build.sh --portable && grep -q "third_party/ggml/include" "$T/log" && ! grep -q "ggml-v3" "$T/log"; then ok "unset: API 2 headers"; else bad "unset: not API 2 headers"; fi
if GGML_API=2 run ./build.sh --portable && grep -q "third_party/ggml/include" "$T/log"; then ok "GGML_API=2: API 2 headers"; else bad "GGML_API=2"; fi
if GGML_API=3 run ./build.sh --portable && grep -q "third_party/ggml-v3/include" "$T/log"; then ok "GGML_API=3: API 3 headers"; else bad "GGML_API=3"; fi
[ "$fail" = 0 ] && echo "build_api_test: all passed"
exit $fail
