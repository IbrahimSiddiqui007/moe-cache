#!/bin/bash
# Tiny-model check of llama.cpp patch 0070 (DECODE-630); needs a CUDA build of the patched llama.cpp and a small MoE GGUF. Not in the quick set.
#   tests/decode_reloc_live.sh LLAMA_CPP_DIR BUILD_DIR MODEL.gguf
# Env: DECODE_RELOC_EXPECT=moved (default) the switch-on run must really move layers (a machine with a GPU, e.g. the laptop);
#      DECODE_RELOC_EXPECT=any   CI without a real GPU (llama_patched.yml): checks that need a move are skipped, with a line saying so, when none happened;
#      DECODE_RELOC_SKIP_LONG=1  leave out the 2400-token run (a software GPU such as lavapipe is too slow for it);
#      DECODE_RELOC_PLUGIN=LIB   also run with the moe-cache plugin loaded (GGML_BACKEND_PATH) and a 2 MB cache so experts are evicted: the experts of the host layers
#                                (DECODE_RELOC_PLUGIN_CACHE_GIB, default 0.002) are then in the plugin's MOE_CACHE buffer (not "host" memory) and must still move, through get_tensor, with the same tokens;
#      DECODE_RELOC_EXE_DIR=DIR  where the test program is put (default: a temporary folder); a build with dynamic backends (GGML_BACKEND_DL) finds them next to the program.
# Runs tests/decode_reloc_live.cpp with the switch unset and set (and with a second context) and compares the token ids. Also:
#   reserve failure: LLAMA_DECODE_RELOC_FAIL=1 (first reserve throws) and =2 (reserve after the move throws): the run goes on, the layers are back, tokens equal
#   late second context: created after the move; the first context moves back at its next batch; no crash, tokens equal
#   long greedy identity: 2400 tokens, 4 prompts (700 and 1500 tokens arrive in the decode phase), off vs on; identical yes/no and the first differing token are reported (a difference is not a failure)
set -e
export LC_ALL=C
SRC="${1:?llama.cpp dir}"; BLD="${2:?build dir}"; MODEL="${3:?model}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(mktemp -d)"; EXE="${DECODE_RELOC_EXE_DIR:-$OUT}"; EXPECT="${DECODE_RELOC_EXPECT:-moved}"
g++ -std=c++17 -O1 -Wall -I"$SRC/include" -I"$SRC/ggml/include" "$HERE/decode_reloc_live.cpp" -L"$BLD/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$BLD/bin" -o "$EXE/live"
export LLAMA_DECODE_RELOC_UB=32 LLAMA_DECODE_RELOC_AFTER=8 LLAMA_DECODE_RELOC_KEEP_MIB=200
env -u LLAMA_DECODE_RELOC "$EXE/live" "$MODEL" > "$OUT/off.txt" 2> "$OUT/off.err"
LLAMA_DECODE_RELOC=1 "$EXE/live" "$MODEL" > "$OUT/on.txt" 2> "$OUT/on.err"
LLAMA_DECODE_RELOC=1 "$EXE/live" "$MODEL" twoctx > "$OUT/two.txt" 2> "$OUT/two.err"
# a run that may fail: keep its exit code in $OUT/NAME.rc, do not stop the script
run() { local n="$1"; shift; set +e; "$@" > "$OUT/$n.txt" 2> "$OUT/$n.err"; echo $? > "$OUT/$n.rc"; set -e; }
run f1 env LLAMA_DECODE_RELOC=1 LLAMA_DECODE_RELOC_FAIL=1 "$EXE/live" "$MODEL"
run f2 env LLAMA_DECODE_RELOC=1 LLAMA_DECODE_RELOC_FAIL=2 "$EXE/live" "$MODEL"
run l_off env -u LLAMA_DECODE_RELOC "$EXE/live" "$MODEL" late2
run l_on env LLAMA_DECODE_RELOC=1 "$EXE/live" "$MODEL" late2
[ -n "${DECODE_RELOC_SKIP_LONG:-}" ] || run g_off env -u LLAMA_DECODE_RELOC -u LLAMA_DECODE_RELOC_UB -u LLAMA_DECODE_RELOC_AFTER -u LLAMA_DECODE_RELOC_KEEP_MIB "$EXE/live" "$MODEL" long
[ -n "${DECODE_RELOC_SKIP_LONG:-}" ] || run g_on env -u LLAMA_DECODE_RELOC_UB -u LLAMA_DECODE_RELOC_AFTER -u LLAMA_DECODE_RELOC_KEEP_MIB LLAMA_DECODE_RELOC=1 "$EXE/live" "$MODEL" long
# diagnostics: the memory figure is the real one (or says it was not queried); LLAMA_DECODE_RELOC_FAKE_FREE_MIB (test only) pretends the device has little free memory
run f3 env LLAMA_DECODE_RELOC=1 LLAMA_DECODE_RELOC_FAKE_FREE_MIB=50 "$EXE/live" "$MODEL"
if [ -n "${DECODE_RELOC_PLUGIN:-}" ]; then
  PENV=(GGML_BACKEND_PATH="$DECODE_RELOC_PLUGIN" MOE_CACHE_GGUF="$MODEL" MOE_CACHE_SIZE_GIB="${DECODE_RELOC_PLUGIN_CACHE_GIB:-0.002}" MOE_CACHE_STATS=1)
  run p_off env -u LLAMA_DECODE_RELOC "${PENV[@]}" "$EXE/live" "$MODEL"
  run p_on env LLAMA_DECODE_RELOC=1 "${PENV[@]}" "$EXE/live" "$MODEL"
  run pl_off env -u LLAMA_DECODE_RELOC "${PENV[@]}" "$EXE/live" "$MODEL" late2
  run pl_on env LLAMA_DECODE_RELOC=1 "${PENV[@]}" "$EXE/live" "$MODEL" late2
fi
# shrink only (LLAMA_DECODE_RELOC_MAX_LAYERS=0): the small micro-batch during decode, nothing moves, the big one comes back for a big chunk
run sh_on env LLAMA_DECODE_RELOC=1 LLAMA_DECODE_RELOC_MAX_LAYERS=0 "$EXE/live" "$MODEL"
[ -z "${DECODE_RELOC_PLUGIN:-}" ] || run shp_on env LLAMA_DECODE_RELOC=1 LLAMA_DECODE_RELOC_MAX_LAYERS=0 "${PENV[@]}" "$EXE/live" "$MODEL"
fail=0
MOVED=0; grep -q 'layers of experts moved to the GPU' "$OUT/on.err" && MOVED=1
chkm() { if [ "$MOVED" = 1 ]; then chk "$@"; else echo "skip $1 (no layer could move on this machine: $(grep -h 'no layer fits\|not on a GPU' "$OUT/on.err" | head -1 | cut -c1-120))"; fi; }
chk() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
chk "switch unset: no patch log line" "! grep -q reloc_ '$OUT/off.err'"
[ "$EXPECT" != moved ] || chk "EXPECT=moved: the switch-on run moved layers" "[ $MOVED = 1 ]"
chkm "switch on: layers moved" "grep -q 'layers of experts moved to the GPU' '$OUT/on.err'"
chkm "switch on: moved back for the big chunk" "grep -q 'back to the prompt layout' '$OUT/on.err'"
chk "tokens equal with the switch on and off (prompt, decode, restore, move back, decode)" "cmp -s '$OUT/off.txt' '$OUT/on.txt'"
chk "slot save/restore in the decode phase reproduces the tokens (on)" "grep -q 'restore_matches: yes' '$OUT/on.txt'"
chk "second context: move refused" "grep -q 'another context' '$OUT/two.err' && ! grep -q 'layers of experts moved' '$OUT/two.err'"
chk "second context: tokens unchanged" "cmp -s '$OUT/off.txt' '$OUT/two.txt'"
# shrink only: works on any machine (no move needed)
shrink_ok() { awk '/shrink only/ { for (i = 1; i <= NF; i++) if ($i == "buffers") { a = $(i+1); b = $(i+4) } } END { exit !(a > 0 && b + 0 < a + 0) }' "$1"; }
chk "shrink only (MAX_LAYERS=0): the small micro-batch is reserved and the compute buffers get smaller, nothing moves, moved back for the big chunk" "[ \"\$(cat $OUT/sh_on.rc)\" = 0 ] && grep -q 'shrink only (no layer moves)' '$OUT/sh_on.err' && ! grep -q 'layers of experts moved' '$OUT/sh_on.err' && grep -q 'back to the prompt layout' '$OUT/sh_on.err' && cmp -s '$OUT/off.txt' '$OUT/sh_on.txt'"
chk "shrink only: the logged compute buffers shrank (INFO line: A MiB -> B MiB with B < A)" "$(declare -f shrink_ok); shrink_ok '$OUT/sh_on.err'"
if [ -n "${DECODE_RELOC_PLUGIN:-}" ]; then
  chk "shrink only with the plugin loaded (experts evicted): exit 0, tokens equal to the switch off" "[ \"\$(cat $OUT/shp_on.rc)\" = 0 ] && grep -q 'shrink only (no layer moves)' '$OUT/shp_on.err' && cmp -s '$OUT/p_off.txt' '$OUT/shp_on.txt'"
fi
# diagnostics
if [ "$MOVED" = 1 ]; then
  chk "no layer fits: the log names the memory figure and the reasons (fake free memory 50 MiB)" "[ \"\$(cat $OUT/f3.rc)\" = 0 ] && grep -q 'no layer fits (50 MiB free of 50 MiB' '$OUT/f3.err' && grep -q 'too big for the free memory' '$OUT/f3.err' && cmp -s '$OUT/off.txt' '$OUT/f3.txt'"
else
  chk "no layer fits (no GPU here): the log says the memory was not queried and counts the layers not on a GPU" "[ \"\$(cat $OUT/f3.rc)\" = 0 ] && grep -q 'device memory not queried' '$OUT/f3.err' && grep -Eq '[1-9][0-9]* not on a GPU' '$OUT/f3.err' && cmp -s '$OUT/off.txt' '$OUT/f3.txt'"
fi
if [ -n "${DECODE_RELOC_PLUGIN:-}" ]; then
  chk "plugin: the experts are in the MOE_CACHE buffer and the cache evicted (2 MB cache)" "grep -q 'buffer type overridden to MOE_CACHE' '$OUT/p_on.err' && grep -Eq 'evictions=[1-9]' '$OUT/p_on.err'"
  chk "plugin: exit 0, tokens equal with the switch on and off (experts evicted while the layers move)" "[ \"\$(cat $OUT/p_on.rc)\" = 0 ] && [ \"\$(cat $OUT/p_off.rc)\" = 0 ] && cmp -s '$OUT/p_off.txt' '$OUT/p_on.txt'"
  chkm "plugin: layers moved, read through the moe-cache buffer" "grep -q 'of them read through the moe-cache buffer' '$OUT/p_on.err' && ! grep -q 'of them read through the moe-cache buffer' '$OUT/p_off.err'"
  chk "plugin, second context after the move: no crash, tokens equal to the switch off" "[ \"\$(cat $OUT/pl_on.rc)\" = 0 ] && cmp -s '$OUT/pl_off.txt' '$OUT/pl_on.txt'"
fi
# test 1: reserve failure
chk "fail=1: first reserve throws, the log says why" "grep -q 'reserve with ubatch 32 failed (LLAMA_DECODE_RELOC_FAIL=1' '$OUT/f1.err' && ! grep -q 'layers of experts moved' '$OUT/f1.err'"
chk "fail=1: process finished (exit 0), tokens equal to the switch off" "[ \"\$(cat $OUT/f1.rc)\" = 0 ] && cmp -s '$OUT/off.txt' '$OUT/f1.txt'"
chkm "fail=2: reserve after the move throws, moved back, the log says why" "! grep -q 'layers of experts moved' '$OUT/f2.err' && grep -q 'reserve after the move failed (LLAMA_DECODE_RELOC_FAIL=2' '$OUT/f2.err' && grep -q 'expert tensors back in their original' '$OUT/f2.err'"
chk "fail=2: process finished (exit 0), tokens equal to the switch off" "[ \"\$(cat $OUT/f2.rc)\" = 0 ] && cmp -s '$OUT/off.txt' '$OUT/f2.txt'"
chk "fail=1: no second try (one failure line)" "[ \"\$(grep -c 'failed (LLAMA_DECODE_RELOC_FAIL' '$OUT/f1.err')\" = 1 ]"
chkm "fail=2: no second try (one failure line)" "[ \"\$(grep -c 'failed (LLAMA_DECODE_RELOC_FAIL' '$OUT/f2.err')\" = 1 ]"
# test 2: second context created after the move
chkm "late 2nd context: layers were moved first, then the first context moved back (log)" "grep -q 'layers of experts moved' '$OUT/l_on.err' && grep -q 'another context was created, moving back' '$OUT/l_on.err'"
chk "late 2nd context: no crash (exit 0) with the switch on and off" "[ \"\$(cat $OUT/l_on.rc)\" = 0 ] && [ \"\$(cat $OUT/l_off.rc)\" = 0 ]"
chk "late 2nd context: tokens of both contexts equal to the switch off" "cmp -s '$OUT/l_off.txt' '$OUT/l_on.txt'"
# test 3: long greedy identity
if [ -n "${DECODE_RELOC_SKIP_LONG:-}" ]; then echo "skip long greedy identity (DECODE_RELOC_SKIP_LONG)"; else
ntok=$(grep -E '^P[0-9]:' "$OUT/g_on.txt" | wc -w); ntok=$((ntok - 4))
chk "long: at least 2000 tokens generated ($ntok), exit 0" "[ $ntok -ge 2000 ] && [ \"\$(cat $OUT/g_on.rc)\" = 0 ] && [ \"\$(cat $OUT/g_off.rc)\" = 0 ]"
chkm "long: layers moved at least twice and moved back at least twice (700 and 1500 token batches in the decode phase)" "[ \$(grep -c 'layers of experts moved' '$OUT/g_on.err') -ge 2 ] && [ \$(grep -c 'back to the prompt layout' '$OUT/g_on.err') -ge 2 ]"
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
fi
echo "logs in $OUT"
exit $fail
