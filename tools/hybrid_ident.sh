#!/bin/bash
# hybrid_ident.sh MODEL.gguf [N_TOKENS] [common ENV=VAL ...]: the same 4 prompts at temperature 0 / seed 42 through the plugin with --hybrid in three modes
# (base, MOE_CACHE_HYBRID_ASYNC=1, MOE_CACHE_HYBRID_PERSIST=1) and compares the token ids. The three must be identical (only the way data is copied differs).
# Needs a usage profile: PROFILE_SRC=FILE (every mode gets its own fresh copy, because a run rewrites its profile at exit).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; N="${2:-100}"; shift 2 || shift $#
LS="${LLAMA_SERVER:-llama-server}"; PORT=${PORT:-8096}
. "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$LS" "${MOE_CACHE_LIB:-$(moe_cache_pick_lib "$LS" "$HERE")}" auto
TMP=$(mktemp -d); trap 'kill $SPID 2>/dev/null || true; rm -rf "$TMP"' EXIT
for mode in off base async persist; do
  X=(); [ $mode = off ] && X=(MOE_CACHE_HYBRID=0); [ $mode = async ] && X=(MOE_CACHE_HYBRID_ASYNC=1); [ $mode = persist ] && X=(MOE_CACHE_HYBRID_PERSIST=1)
  cp "$PROFILE_SRC" "$TMP/$mode.usage"
  env "${MOE_CACHE_LOAD[@]}" MOE_CACHE_PROFILE="$TMP/$mode.usage" MOE_CACHE_SIZE_GIB=auto MOE_CACHE_GGUF="$MODEL" MOE_CACHE_STATS=1 MOE_CACHE_HYBRID=${MOE_CACHE_HYBRID:-auto} "$@" "${X[@]}" \
    "$LS" -m "$MODEL" -ngl 99 --n-cpu-moe ${NCMOE:-999} -c ${CTX:-4096} -np 1 -fa on --load-mode mmap --no-warmup --port $PORT --seed 42 > "$TMP/$mode.log" 2>&1 & SPID=$!
  for i in $(seq 600); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && break; sleep 2; done
  python3 "$HERE/tests/ident.py" "$TMP/$mode.json" $PORT $N
  kill -INT $SPID; wait $SPID 2>/dev/null || true
  echo "$mode: $(grep -o 'picks computed on the GPU from its bank ([0-9.]*%)' "$TMP/$mode.log" | tail -1)"
done
ok=1
for mode in async persist; do cmp -s "$TMP/base.json" "$TMP/$mode.json" && echo "IDENTICAL: $mode == base" || { echo "DIFFERENT: $mode != base"; ok=0; }; done
python3 - "$TMP" <<'P'
import json, sys
d = sys.argv[1]
o, b = json.load(open(d + "/off.json")), json.load(open(d + "/base.json"))
def first_diff(x, y):
    return next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
print("tokens per prompt:", [len(x) for x in b])
print("hybrid vs hybrid off, tokens equal before the first difference (informational; GPU and CPU round differently):", [first_diff(x, y) for x, y in zip(o, b)])
P
[ $ok = 1 ]
