#!/bin/bash
# Tiny-model check of llama.cpp patch 0070 (DECODE-630); needs a CUDA build of the patched llama.cpp and a small MoE GGUF. Not in the quick set.
#   tests/decode_reloc_live.sh LLAMA_CPP_DIR BUILD_DIR MODEL.gguf
# Runs tests/decode_reloc_live.cpp with the switch unset and set (and with a second context) and compares the token ids. Also:
#   reserve failure: LLAMA_DECODE_RELOC_FAIL=1 (first reserve throws) and =2 (reserve after the move throws): the run goes on, the layers are back, tokens equal
#   late second context: created after the move; the first context moves back at its next batch; no crash, tokens equal
#   long greedy identity: 2400 tokens, 4 prompts (700 and 1500 tokens arrive in the decode phase), off vs on; identical yes/no and the first differing token are reported (a difference is not a failure)
set -e
export LC_ALL=C
SRC="${1:?llama.cpp dir}"; BLD="${2:?build dir}"; MODEL="${3:?model}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(mktemp -d)"
g++ -std=c++17 -O1 -Wall -I"$SRC/include" -I"$SRC/ggml/include" "$HERE/decode_reloc_live.cpp" -L"$BLD/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$BLD/bin" -o "$OUT/live"
export LLAMA_DECODE_RELOC_UB=32 LLAMA_DECODE_RELOC_AFTER=8 LLAMA_DECODE_RELOC_KEEP_MIB=200
env -u LLAMA_DECODE_RELOC "$OUT/live" "$MODEL" > "$OUT/off.txt" 2> "$OUT/off.err"
LLAMA_DECODE_RELOC=1 "$OUT/live" "$MODEL" > "$OUT/on.txt" 2> "$OUT/on.err"
LLAMA_DECODE_RELOC=1 "$OUT/live" "$MODEL" twoctx > "$OUT/two.txt" 2> "$OUT/two.err"
# a run that may fail: keep its exit code in $OUT/NAME.rc, do not stop the script
run() { local n="$1"; shift; set +e; "$@" > "$OUT/$n.txt" 2> "$OUT/$n.err"; echo $? > "$OUT/$n.rc"; set -e; }
run f1 env LLAMA_DECODE_RELOC=1 LLAMA_DECODE_RELOC_FAIL=1 "$OUT/live" "$MODEL"
run f2 env LLAMA_DECODE_RELOC=1 LLAMA_DECODE_RELOC_FAIL=2 "$OUT/live" "$MODEL"
run l_off env -u LLAMA_DECODE_RELOC "$OUT/live" "$MODEL" late2
run l_on env LLAMA_DECODE_RELOC=1 "$OUT/live" "$MODEL" late2
run g_off env -u LLAMA_DECODE_RELOC -u LLAMA_DECODE_RELOC_UB -u LLAMA_DECODE_RELOC_AFTER -u LLAMA_DECODE_RELOC_KEEP_MIB "$OUT/live" "$MODEL" long
run g_on env -u LLAMA_DECODE_RELOC_UB -u LLAMA_DECODE_RELOC_AFTER -u LLAMA_DECODE_RELOC_KEEP_MIB LLAMA_DECODE_RELOC=1 "$OUT/live" "$MODEL" long
fail=0
chk() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
chk "switch unset: no patch log line" "! grep -q reloc_ '$OUT/off.err'"
chk "switch on: layers moved" "grep -q 'layers of experts moved to the GPU' '$OUT/on.err'"
chk "switch on: moved back for the big chunk" "grep -q 'back to the prompt layout' '$OUT/on.err'"
chk "tokens equal with the switch on and off (prompt, decode, restore, move back, decode)" "cmp -s '$OUT/off.txt' '$OUT/on.txt'"
chk "slot save/restore in the decode phase reproduces the tokens (on)" "grep -q 'restore_matches: yes' '$OUT/on.txt'"
chk "second context: move refused" "grep -q 'another context' '$OUT/two.err' && ! grep -q 'layers of experts moved' '$OUT/two.err'"
chk "second context: tokens unchanged" "cmp -s '$OUT/off.txt' '$OUT/two.txt'"
# test 1: reserve failure
chk "fail=1: first reserve throws, the log says why" "grep -q 'reserve with ubatch 32 failed (LLAMA_DECODE_RELOC_FAIL=1' '$OUT/f1.err' && ! grep -q 'layers of experts moved' '$OUT/f1.err'"
chk "fail=1: process finished (exit 0), tokens equal to the switch off" "[ \"\$(cat $OUT/f1.rc)\" = 0 ] && cmp -s '$OUT/off.txt' '$OUT/f1.txt'"
chk "fail=2: reserve after the move throws, moved back, the log says why" "! grep -q 'layers of experts moved' '$OUT/f2.err' && grep -q 'reserve after the move failed (LLAMA_DECODE_RELOC_FAIL=2' '$OUT/f2.err' && grep -q 'expert tensors back in their original' '$OUT/f2.err'"
chk "fail=2: process finished (exit 0), tokens equal to the switch off" "[ \"\$(cat $OUT/f2.rc)\" = 0 ] && cmp -s '$OUT/off.txt' '$OUT/f2.txt'"
chk "fail=1/2: no second try (one failure line each)" "[ \"\$(grep -c 'failed (LLAMA_DECODE_RELOC_FAIL' '$OUT/f1.err')\" = 1 ] && [ \"\$(grep -c 'failed (LLAMA_DECODE_RELOC_FAIL' '$OUT/f2.err')\" = 1 ]"
# test 2: second context created after the move
chk "late 2nd context: layers were moved first, then the first context moved back (log)" "grep -q 'layers of experts moved' '$OUT/l_on.err' && grep -q 'another context was created, moving back' '$OUT/l_on.err'"
chk "late 2nd context: no crash (exit 0) with the switch on and off" "[ \"\$(cat $OUT/l_on.rc)\" = 0 ] && [ \"\$(cat $OUT/l_off.rc)\" = 0 ]"
chk "late 2nd context: tokens of both contexts equal to the switch off" "cmp -s '$OUT/l_off.txt' '$OUT/l_on.txt'"
# test 3: long greedy identity
ntok=$(grep -E '^P[0-9]:' "$OUT/g_on.txt" | wc -w); ntok=$((ntok - 4))
chk "long: at least 2000 tokens generated ($ntok), exit 0" "[ $ntok -ge 2000 ] && [ \"\$(cat $OUT/g_on.rc)\" = 0 ] && [ \"\$(cat $OUT/g_off.rc)\" = 0 ]"
chk "long: layers moved at least twice and moved back at least twice (700 and 1500 token batches in the decode phase)" "[ \$(grep -c 'layers of experts moved' '$OUT/g_on.err') -ge 2 ] && [ \$(grep -c 'back to the prompt layout' '$OUT/g_on.err') -ge 2 ]"
if cmp -s "$OUT/g_off.txt" "$OUT/g_on.txt"; then echo "ok   long: text and token ids byte-identical, switch off vs on"
else
  # owner's rule: semantic similarity, not byte identity (CPU -> GPU expert matmuls may change floats). A difference is reported, not a failure; a crash,
  # garbage or a wrong restore is (the checks above and the fail=1/2, late2 checks).
  echo "DIFF long: not byte-identical, switch off vs on (reported, not a failure; compare the text below by eye)"
  python3 -I - "$OUT/g_off.txt" "$OUT/g_on.txt" <<'PY'
import sys
a = open(sys.argv[1]).read().splitlines(); b = open(sys.argv[2]).read().splitlines()
for la, lb in zip(a, b):
    if la != lb:
        name = la.split(":")[0]
        if name.startswith("T"):
            print("  first differing text char in", name, "at index", next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb))))
        else:
            ta, tb = la.split()[1:], lb.split()[1:]
            i = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            print("  first differing token in", name, "at generated index", i, ": off", ta[i:i+1], "on", tb[i:i+1], "; tokens differing in this prompt:", sum(x != y for x, y in zip(ta, tb)), "of", len(ta))
        break
else:
    print("  line counts differ:", len(a), len(b))
PY
fi
echo "logs in $OUT"
exit $fail
