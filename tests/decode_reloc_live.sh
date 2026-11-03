#!/bin/bash
# Tiny-model check of llama.cpp patch 0070 (DECODE-630); needs a CUDA build of the patched llama.cpp and a small MoE GGUF. Not in the quick set.
#   tests/decode_reloc_live.sh LLAMA_CPP_DIR BUILD_DIR MODEL.gguf
# Runs tests/decode_reloc_live.cpp with the switch unset and set (and with a second context) and compares the token ids.
set -e
SRC="${1:?llama.cpp dir}"; BLD="${2:?build dir}"; MODEL="${3:?model}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(mktemp -d)"
g++ -std=c++17 -O1 -Wall -I"$SRC/include" -I"$SRC/ggml/include" "$HERE/decode_reloc_live.cpp" -L"$BLD/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$BLD/bin" -o "$OUT/live"
export LLAMA_DECODE_RELOC_UB=32 LLAMA_DECODE_RELOC_AFTER=8 LLAMA_DECODE_RELOC_KEEP_MIB=200
env -u LLAMA_DECODE_RELOC "$OUT/live" "$MODEL" > "$OUT/off.txt" 2> "$OUT/off.err"
LLAMA_DECODE_RELOC=1 "$OUT/live" "$MODEL" > "$OUT/on.txt" 2> "$OUT/on.err"
LLAMA_DECODE_RELOC=1 "$OUT/live" "$MODEL" twoctx > "$OUT/two.txt" 2> "$OUT/two.err"
fail=0
chk() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
chk "switch unset: no patch log line" "! grep -q reloc_ '$OUT/off.err'"
chk "switch on: layers moved" "grep -q 'layers of experts moved to the GPU' '$OUT/on.err'"
chk "switch on: moved back for the big chunk" "grep -q 'back to the prompt layout' '$OUT/on.err'"
chk "tokens equal with the switch on and off (prompt, decode, restore, move back, decode)" "cmp -s '$OUT/off.txt' '$OUT/on.txt'"
chk "slot save/restore in the decode phase reproduces the tokens (on)" "grep -q 'restore_matches: yes' '$OUT/on.txt'"
chk "second context: move refused" "grep -q 'another context' '$OUT/two.err' && ! grep -q 'layers of experts moved' '$OUT/two.err'"
chk "second context: tokens unchanged" "cmp -s '$OUT/off.txt' '$OUT/two.txt'"
echo "logs in $OUT"
exit $fail
