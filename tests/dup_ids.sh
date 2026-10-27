#!/bin/bash
# Repeated expert ids: a token that lists the same expert more than once (a GPU top-k fed NaN router scores does this). Before the fix the executor
# wrote past its per-expert task (more than 8 columns) and a pool worker then wrote through a garbage pointer (Windows crash at request 2, 2026-10-27).
#   tests/dup_ids.sh MODEL.gguf [DUP]      (DUP_CACHE_GIB: cache size, default 0.2; IDENT_KEEP=DIR keeps the log)
# Runs only the server with the plugin (stock never sees the repeated ids, so there is nothing to compare tokens with): two slots, the prompts sent two
# at a time (decode batches of 2 tokens, prompt tails of a few tokens), MOE_CACHE_TEST_DUP_IDS=DUP (default 5: picks 1-4 of every token repeat pick 0, so a 1-token step gives the executor 5 columns of one expert and a 2-token batch 10, more than a task holds)
# and MOE_CACHE_EXEC_CHECK=1 (every op the executor runs is recomputed by llama.cpp and compared byte for byte). PASS: the server answered every prompt,
# is still alive, the executor ran ops and none differed.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; DUP="${2:-5}"; PORT=8099
LS="${LLAMA_SERVER:-llama-server}"
[ -f "$MODEL" ] || { echo "usage: tests/dup_ids.sh MODEL.gguf [DUP]"; exit 1; }
TMP=$(mktemp -d); trap 'kill $SPID 2>/dev/null || true; [ -n "$IDENT_KEEP" ] && mkdir -p "$IDENT_KEEP" && cp "$TMP"/* "$IDENT_KEEP"/ 2>/dev/null; rm -rf "$TMP"' EXIT
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/health"; then echo "port $PORT is already in use (a leftover server?)"; exit 1; fi
. "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$LS" "${MOE_CACHE_LIB:-$HERE/build/libggml-moe-cache.so}" "${METHOD:-auto}"
env "${MOE_CACHE_LOAD[@]}" MOE_CACHE_GGUF="$MODEL" MOE_CACHE_SIZE_GIB="${DUP_CACHE_GIB:-0.2}" MOE_CACHE_EXEC=2 MOE_CACHE_EXEC_CHECK=1 MOE_CACHE_STATS=1 MOE_CACHE_TEST_DUP_IDS="$DUP" \
  "$LS" -m "$MODEL" ${DEVICE:+--device $DEVICE} -ngl 99 --n-cpu-moe 999 -c 4096 -np 2 -fa on --load-mode mmap --no-warmup --no-repack --port $PORT --seed 42 \
  > "$TMP/plugin.log" 2>&1 & SPID=$!
for i in $(seq 150); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && break
  kill -0 $SPID 2>/dev/null || { echo "FAIL: the server died while starting"; tail -5 "$TMP/plugin.log"; exit 1; }
  sleep 2
done
ok=1
IDENT_PAR=1 timeout 600 python3 "$HERE/tests/ident.py" "$TMP/plugin.json" $PORT 24 || { echo "FAIL: the prompts did not all get an answer"; ok=0; }
kill -0 $SPID 2>/dev/null || { echo "FAIL: the server is gone (crashed)"; ok=0; }
kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true
grep "moe-cache:" "$TMP/plugin.log" | grep -vE "^moe-cache-stats" | tail -4
line=$(grep -E "moe-cache: executor ran [0-9]+ expert ops itself, checked" "$TMP/plugin.log" | tail -1)
echo "$line"
echo "$line" | grep -qE "executor ran [1-9][0-9]* expert ops itself, checked [1-9][0-9]*, mismatches 0" || { echo "FAIL: the executor did not run, or differed from llama.cpp"; ok=0; }
[ $ok = 1 ] && echo PASS || { echo FAIL; exit 1; }
