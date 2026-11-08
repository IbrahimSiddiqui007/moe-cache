#!/bin/bash
# hybrid_probe.sh MODEL.gguf [N_TOKENS] [ENV=VAL ...]: one server with the plugin, hybrid bank on, one 2-round completion at temperature 0; prints decode tok/s and the
# plugin's hybrid lines (phase timers per layer step). Extra ENV=VAL arguments go to the plugin server (e.g. MOE_CACHE_HYBRID=2000 MOE_CACHE_HYBRID_ADAPT=4,24).
# Run it through handover/tools/cool_run.sh when it takes more than 2 minutes. Never judge speed with timers on if a run is a speed run: timers cost ~1 us per layer step.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; N="${2:-128}"; shift 2 || shift $#
LS="${LLAMA_SERVER:-llama-server}"; PORT=${PORT:-8097}; CTX=${CTX:-4096}
. "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$LS" "${MOE_CACHE_LIB:-$(moe_cache_pick_lib "$LS" "$HERE")}" auto
TMP=$(mktemp -d); trap 'kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true; rm -rf "$TMP"' EXIT
env "${MOE_CACHE_LOAD[@]}" MOE_CACHE_SIZE_GIB=auto MOE_CACHE_GGUF="$MODEL" MOE_CACHE_STATS=1 MOE_CACHE_HYBRID=${MOE_CACHE_HYBRID:-auto} "$@" \
  "$LS" -m "$MODEL" -ngl 99 --n-cpu-moe ${NCMOE:-999} -c $CTX -np 1 -fa on --load-mode mmap --no-warmup --port $PORT --seed 42 > "$TMP/log" 2>&1 & SPID=$!
for i in $(seq 600); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && break; sleep 2; done
python3 - "$PORT" "$N" <<'P'
import json, sys, urllib.request
port, n = sys.argv[1], int(sys.argv[2])
for r in range(2):
    body = {"prompt": "Explain how a hash table handles collisions, with examples.", "n_predict": n, "temperature": 0, "seed": 42, "cache_prompt": False}
    q = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(q, timeout=3000))["timings"]
    print(f"round {r}: decode {t['predicted_per_second']:.2f} tok/s ({t['predicted_n']} tokens)")
P
kill -INT $SPID 2>/dev/null; wait $SPID 2>/dev/null || true; SPID=
grep -E "moe-cache:" "$TMP/log" | grep -v "^$" | cut -c1-700 | tail -${TAILN:-25}
