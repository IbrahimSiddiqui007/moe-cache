#!/bin/bash
# Token-identity test: the same model with and without the plugin must produce the same tokens (temperature 0, seed 42).
#   tests/identity.sh MODEL.gguf [--llama-server PATH] [--n-cpu-moe N] [--ram GB] [--tokens N] [--long]
# --long uses long prompts (hundreds of tokens), which exercise the batched / GPU-assisted path.
# Starts the stock server (no plugin), then the server with the plugin, on port 8099, and compares 4 prompts.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; shift || true
LS="${LLAMA_SERVER:-llama-server}"; NCMOE=999; RAM=""; N=100; PORT=8099
while [ $# -gt 0 ]; do case "$1" in --llama-server) LS=$2; shift 2;; --n-cpu-moe) NCMOE=$2; shift 2;; --ram) RAM=$2; shift 2;; --tokens) N=$2; shift 2;; --long) export IDENT_LONG=1; shift;; *) echo "unknown: $1"; exit 1;; esac; done
[ -f "$MODEL" ] || { echo "usage: tests/identity.sh MODEL.gguf [options]"; exit 1; }
TMP=$(mktemp -d); trap 'kill $SPID 2>/dev/null || true; rm -rf "$TMP"' EXIT
COMMON=(-m "$MODEL" ${DEVICE:+--device $DEVICE} -ngl 99 --n-cpu-moe "$NCMOE" -c 8192 -fa on --load-mode mmap --no-warmup --port $PORT --seed 42)
wait_up() { for i in $(seq 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && return 0; sleep 2; done; echo "server did not start"; return 1; }
echo "1/2 stock server"; "$LS" "${COMMON[@]}" > "$TMP/stock.log" 2>&1 & SPID=$!; wait_up; python3 "$HERE/tests/ident.py" "$TMP/stock.json" $PORT $N; kill $SPID; wait $SPID 2>/dev/null || true
echo "2/2 server with the plugin"
. "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$LS" "${MOE_CACHE_LIB:-$HERE/build/libggml-moe-cache.so}" "${METHOD:-auto}"
ENVV=("${MOE_CACHE_LOAD[@]}" MOE_CACHE_SIZE_GIB=auto MOE_CACHE_GGUF="$MODEL"); [ -n "$RAM" ] && ENVV+=(MOE_CACHE_RAM_GIB="$RAM")
env "${ENVV[@]}" "$LS" "${COMMON[@]}" > "$TMP/plugin.log" 2>&1 & SPID=$!; wait_up; python3 "$HERE/tests/ident.py" "$TMP/plugin.json" $PORT $N; kill $SPID; wait $SPID 2>/dev/null || true
python3 - "$TMP/stock.json" "$TMP/plugin.json" <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok = sum(x == y for x, y in zip(a, b))
print(f"identical prompts: {ok}/{len(a)}"); print("PASS" if ok == len(a) else "FAIL"); sys.exit(0 if ok == len(a) else 1)
PY
