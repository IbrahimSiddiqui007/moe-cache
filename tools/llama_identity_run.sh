#!/bin/bash
# Greedy answers of one llama-server (temperature 0, fixed seed, CPU) for three short prompts, written to OUT as JSON lines (text + token ids). Used by
# .github/workflows/llama_patched.yml to prove the patched build answers like the official one, with every patch switch off and with the switches on:
#   tools/llama_identity_run.sh SERVER_BIN MODEL.gguf PORT OUT [VAR=value ...]     (the VAR=value pairs are set for the server only)
# -lv 4 makes the server print llama.cpp's own start-up lines (the 0040 diagnostic is one of them). The same flags every time; compare two OUT files with diff.
set -e
SERVER="$1"; MODEL="$2"; PORT="$3"; OUT="$4"; shift 4 || { sed -n 2,6p "$0"; exit 1; }
PY="$(command -v python3 || command -v python)"
LOG="$OUT.log"
env "$@" "$SERVER" -m "$MODEL" -ngl 0 --device none -t 4 -b 512 -ub 512 -c 4096 -np 1 --port "$PORT" --no-warmup -lv 4 >"$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
ok=0
for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  kill -0 $PID 2>/dev/null || break
  sleep 1
done
[ $ok = 1 ] || { echo "FAIL: the server never became ready"; tail -40 "$LOG"; exit 1; }
"$PY" - "$PORT" "$OUT" <<'PYEOF'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
prompts = ["The quick brown fox jumps over the lazy dog. The quick brown fox",
           "Explain in two sentences why the sky is blue.",
           "def fib(n):\n    if n < 2:\n        return n\n    return"]
with open(out, "w") as f:
    for p in prompts:
        body = json.dumps({"prompt": p, "n_predict": 100, "temperature": 0, "seed": 1, "cache_prompt": False, "return_tokens": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        if len(r.get("content", "")) < 20:
            sys.exit("FAIL: too little text for prompt %r" % p)
        f.write(json.dumps({"prompt": p, "content": r["content"], "tokens": r.get("tokens")}) + "\n")
PYEOF
echo "answers written to $OUT"
