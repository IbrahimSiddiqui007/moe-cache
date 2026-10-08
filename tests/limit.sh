#!/bin/bash
# Memory-limit test: run a model inside a hard memory limit (systemd cgroup, no swap) with moe-cache, send short and long prompts, and check that
# the server survives and answers. PASS = no out-of-memory kill. Needs systemd (user session).
#   tests/limit.sh MODEL.gguf --limit GB [--llama-server PATH] [--n-cpu-moe N] [--ctx N]
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; shift || true
LS="${LLAMA_SERVER:-llama-server}"; LIMIT=12; NCMOE=999; CTX=8192; PORT=8093
while [ $# -gt 0 ]; do case "$1" in --limit) LIMIT=$2; shift 2;; --llama-server) LS=$2; shift 2;; --n-cpu-moe) NCMOE=$2; shift 2;; --ctx) CTX=$2; shift 2;; *) echo "unknown: $1"; exit 1;; esac; done
[ -f "$MODEL" ] || { echo "usage: tests/limit.sh MODEL.gguf --limit GB [options]"; exit 1; }
. "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$LS" "${MOE_CACHE_LIB:-$HERE/build/libggml-moe-cache.so}" "${METHOD:-auto}"
python3 -c "import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)" "$MODEL"
LOG="$(mktemp)"
(systemd-run --user --scope -q -p MemoryMax=${LIMIT}G -p MemorySwapMax=0 env "${MOE_CACHE_LOAD[@]}" MOE_CACHE_SIZE_GIB=auto MOE_CACHE_GGUF="$MODEL" MOE_CACHE_STATS=1 \
  "$LS" -m "$MODEL" -ngl 99 --n-cpu-moe "$NCMOE" -c "$CTX" -fa on --load-mode mmap --no-warmup -np 1 --port $PORT --seed 42 > "$LOG" 2>&1 &)
for i in $(seq 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && break; sleep 2; done
python3 - "$PORT" "$HERE" <<'PY'
import json, sys, time, urllib.request, pathlib
port, here = sys.argv[1], pathlib.Path(sys.argv[2])
txt = (here / "README.md").read_text().splitlines(True)
tests = [("short prompt", "Write a Python function that merges overlapping intervals.", 64),
         ("long prompt (~600 tokens)", "Summarize this text in two sentences:\n\n" + "".join(txt[0:45]), 32),
         ("short prompt again", "Explain how a hash table handles collisions.", 64)]
ok = True
for name, p, n in tests:
    t0 = time.time()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps({"prompt": p, "n_predict": n, "temperature": 0, "cache_prompt": False}).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=900))["timings"]
        print(f"  {name}: ok, {d['predicted_per_second']:.1f} tok/s generation, {d['prompt_per_second']:.1f} tok/s prompt, {time.time()-t0:.0f} s")
    except Exception as e:
        print(f"  {name}: FAILED ({type(e).__name__}) - the server was probably killed"); ok = False; break
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
RC=$?; pkill -x "$(basename "$(readlink -f "$(command -v "$LS")")")" 2>/dev/null || true; sleep 2
grep -h "moe-cache: graphs" "$LOG" | sed 's/.*hit/  cache: hit/' | cut -c1-170 | tail -1; rm -f "$LOG"; exit $RC
