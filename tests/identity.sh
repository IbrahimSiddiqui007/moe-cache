#!/bin/bash
# Token-identity test: the same model with and without the plugin must produce the same tokens (temperature 0, seed 42).
#   tests/identity.sh MODEL.gguf [--llama-server PATH] [--n-cpu-moe N] [--ram GB] [--tokens N] [--long] [--ctx N] [--np N] [--fill TOKENS] [--parallel] [--limit GB]
# --ctx N sets the context size (default 8192); --fill T pads every prompt to about T tokens (long-context test, needs --ctx > T);
# --limit GB runs both servers in their own memory-limited scope (systemd-run, swap off): use it for models bigger than your RAM, so a runaway
#   server is killed alone instead of taking the desktop session down.
# --np N starts the server with N slots; --parallel sends the prompts two at a time (use with --np 2).
# IDENT_ALLOW_INACTIVE=1 skips the 'plugin was active' check (noise-floor run with MOE_CACHE_DISABLE=1).
# --long uses long prompts (hundreds of tokens), which exercise the batched / GPU-assisted path.
# Starts the stock server (no plugin), then the server with the plugin, on port 8099, and compares 4 prompts.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; shift || true
LS="${LLAMA_SERVER:-llama-server}"; NCMOE=999; RAM=""; N=100; PORT=8099; CTX=8192; NP=1; LIMIT=""
while [ $# -gt 0 ]; do case "$1" in --llama-server) LS=$2; shift 2;; --n-cpu-moe) NCMOE=$2; shift 2;; --ram) RAM=$2; shift 2;; --tokens) N=$2; shift 2;; --long) export IDENT_LONG=1; shift;; --ctx) CTX=$2; shift 2;; --np) NP=$2; shift 2;; --limit) LIMIT=$2; shift 2;; --fill) export IDENT_FILL=$2; shift 2;; --parallel) export IDENT_PAR=1; shift;; *) echo "unknown: $1"; exit 1;; esac; done
[ -f "$MODEL" ] || { echo "usage: tests/identity.sh MODEL.gguf [options]"; exit 1; }
PRE=(); [ -n "$LIMIT" ] && PRE=(systemd-run --user --scope -q -p MemoryMax=${LIMIT}G -p MemorySwapMax=0)
# IDENT_KEEP=DIR keeps both server logs and token files there (speed, plugin messages) instead of deleting them
TMP=$(mktemp -d); trap 'kill $SPID 2>/dev/null || true; [ -n "$IDENT_KEEP" ] && mkdir -p "$IDENT_KEEP" && cp "$TMP"/* "$IDENT_KEEP"/ 2>/dev/null; rm -rf "$TMP"' EXIT
COMMON=(-m "$MODEL" ${DEVICE:+--device $DEVICE} -ngl 99 --n-cpu-moe "$NCMOE" -c "$CTX" -np "$NP" -fa on --load-mode mmap --no-warmup --port $PORT --seed 42)
# IDENT_EXTRA="--moe-cache-mib 2000": more llama-server arguments for both servers (e.g. llama.cpp's GPU expert cache)
# IDENT_PLUGIN_ENV="MOE_CACHE_GPU_CACHE=1 MOE_CACHE_AVX512=1": more environment for the plugin server only
[ -n "$IDENT_EXTRA" ] && COMMON+=($IDENT_EXTRA)
wait_up() { for i in $(seq 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && return 0; sleep 2; done; echo "server did not start"; return 1; }
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/health"; then echo "port $PORT is already in use (a leftover server?): stop it first, the comparison would be wrong"; exit 1; fi
echo "1/2 stock server"; "${PRE[@]}" "$LS" "${COMMON[@]}" > "$TMP/stock.log" 2>&1 & SPID=$!; wait_up; python3 "$HERE/tests/ident.py" "$TMP/stock.json" $PORT $N; kill $SPID; wait $SPID 2>/dev/null || true
echo "2/2 server with the plugin"
. "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$LS" "${MOE_CACHE_LIB:-$(moe_cache_pick_lib "$LS" "$HERE")}" "${METHOD:-auto}"
ENVV=("${MOE_CACHE_LOAD[@]}" MOE_CACHE_SIZE_GIB=auto MOE_CACHE_GGUF="$MODEL"); [ -n "$RAM" ] && ENVV+=(MOE_CACHE_RAM_GIB="$RAM")
[ -n "$IDENT_PLUGIN_ENV" ] && ENVV+=($IDENT_PLUGIN_ENV)
"${PRE[@]}" env "${ENVV[@]}" "$LS" "${COMMON[@]}" > "$TMP/plugin.log" 2>&1 & SPID=$!; wait_up; python3 "$HERE/tests/ident.py" "$TMP/plugin.json" $PORT $N
if [ -n "$IDENT_IDLE" ]; then
  # IDENT_IDLE=1: idle release (N3, what on-demand sends after its idle time): the server's resident memory must drop, and the same prompts
  # afterwards must give the same tokens (audit 2026-10-29 N1: with the Linux file view it gave nothing back but said it did)
  rss() { awk '/^(RssFile|RssAnon):/{s+=$2} END{print s}' /proc/$SPID/status; }
  R0=$(rss); kill -USR2 $SPID; sleep 3; R1=$(rss)
  echo "idle release: resident $((R0 / 1024)) MiB -> $((R1 / 1024)) MiB"; grep "moe-cache: idle:" "$TMP/plugin.log" | tail -1
  if ! grep -q "moe-cache: idle:" "$TMP/plugin.log" || [ "$R1" -gt $((R0 * 9 / 10)) ]; then echo "FAIL: the idle release gave back less than 10 % of the server's memory"; exit 3; fi
  python3 "$HERE/tests/ident.py" "$TMP/plugin_idle.json" $PORT $N
  cmp -s "$TMP/plugin.json" "$TMP/plugin_idle.json" || { echo "FAIL: tokens after the idle release differ from before it"; exit 3; }
  echo "idle release: same tokens after it"
fi
kill $SPID; wait $SPID 2>/dev/null || true
if [ -z "$IDENT_ALLOW_INACTIVE" ] && { grep -q "moe-cache: .*plugin disabled" "$TMP/plugin.log" || ! grep -qE "moe-cache: (cache mode|pass-through mode)" "$TMP/plugin.log"; }; then
  echo "FAIL: the plugin was not active in the second run (no 'moe-cache: cache mode' line in the server log), so this comparison proves nothing."
  grep "moe-cache:" "$TMP/plugin.log" | tail -5; exit 2
fi
[ -n "$MOE_CACHE_STATS" ] && grep "moe-cache:" "$TMP/plugin.log" | tail -6
python3 - "$TMP/stock.json" "$TMP/plugin.json" "$N" <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
# answers that end at once (end-of-text as the first token) compare equal without exercising any decoding: demand some real output
need = min(8, int(sys.argv[3]))
print(f"generated tokens per prompt (stock): {[len(x) for x in a]}")
if not a or max(len(x) for x in a) < need:
    print(f"FAIL: no prompt produced {need} or more tokens, so this comparison exercised (almost) no decoding"); sys.exit(1)
ok = sum(x == y for x, y in zip(a, b))
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        d = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
        print(f"prompt {i}: first different token at position {d} of {len(x)} (stock {x[d] if d < len(x) else None}, plugin {y[d] if d < len(y) else None})")
print(f"identical prompts: {ok}/{len(a)}"); print("PASS" if ok == len(a) else "FAIL"); sys.exit(0 if ok == len(a) else 1)
PY
