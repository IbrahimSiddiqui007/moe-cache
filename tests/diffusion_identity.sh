#!/bin/bash
# S1 of the diffusion plan: llama-diffusion-cli with and without the plugin must print the same text (confidence remasking, temperature 0 = deterministic),
# and the plugin must not be slower below the model size. Both arms run in their own memory-limited scope (systemd-run, swap off).
#   tests/diffusion_identity.sh MODEL.gguf OUTDIR [--cli PATH] [--limit GB] [--ram GB] [--ub 256] [--steps 64] [--block 32] [--prompt TEXT] [--lib PATH] [--ngl N] [--order stock|plugin]
# Output: OUTDIR/{stock,plugin}.{txt,log}; prints time per step, distinct experts per layer per step (plugin stats) and IDENTICAL / DIFFERENT.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; OUT="$2"; shift 2 || { echo "usage: $0 MODEL OUTDIR [options]"; exit 1; }
CLI=llama-diffusion-cli; LIMIT=3; RAM=""; UB=256; STEPS=64; BLOCK=32; LIB=""; NGL=0; ORDER=stock
PROMPT="Write a short Python function that returns the n-th Fibonacci number, then explain how it works."
while [ $# -gt 0 ]; do case "$1" in --cli) CLI=$2; shift 2;; --limit) LIMIT=$2; shift 2;; --ram) RAM=$2; shift 2;; --ub) UB=$2; shift 2;; --steps) STEPS=$2; shift 2;;
  --block) BLOCK=$2; shift 2;; --prompt) PROMPT=$2; shift 2;; --lib) LIB=$2; shift 2;; --ngl) NGL=$2; shift 2;; --order) ORDER=$2; shift 2;; *) echo "unknown: $1"; exit 1;; esac; done
[ -n "$RAM" ] || RAM=$(python3 -c "print(round($LIMIT*0.55,2))")
mkdir -p "$OUT"
. "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$CLI" "${LIB:-$(moe_cache_pick_lib "$CLI" "$HERE")}" auto
ARGS=(-m "$MODEL" -ngl "$NGL" -ub "$UB" -b "$UB" -c "$UB" -fa on -lm mmap -s 42 --temp 0 -p "$PROMPT" --diffusion-steps "$STEPS" --diffusion-block-length "$BLOCK")
PRE=(systemd-run --user --scope -q -p MemoryMax=${LIMIT}G -p MemorySwapMax=0)
run() {
  dd if="$MODEL" iflag=nocache count=0 status=none 2>/dev/null || true
  if [ "$1" = stock ]; then "${PRE[@]}" env -u LD_PRELOAD -u GGML_BACKEND_PATH "$CLI" "${ARGS[@]}" > /dev/null 2> "$OUT/stock.log" || echo "stock run failed (exit $?)"
  else "${PRE[@]}" env "${MOE_CACHE_LOAD[@]}" MOE_CACHE_SIZE_GIB=auto MOE_CACHE_GGUF="$MODEL" MOE_CACHE_RAM_GIB="$RAM" MOE_CACHE_STATS=1 "$CLI" "${ARGS[@]}" > /dev/null 2> "$OUT/plugin.log" || echo "plugin run failed (exit $?)"; fi
}
if [ "$ORDER" = stock ]; then run stock; run plugin; else run plugin; run stock; fi
for a in stock plugin; do echo "$a: $(grep -h 'time per step' $OUT/$a.log $OUT/$a.txt 2>/dev/null | tail -1)"; done
grep -h "moe-cache: graphs=" "$OUT/plugin.log" | tail -1 | python3 -c "
import re,sys
l=sys.stdin.read(); m=re.search(r'graphs=(\d+) groups=(\d+) ops=(\d+) hits=(\d+) misses=(\d+).*read=([\d.]+) MB',l)
if m:
    g,gr,o,h,mi,mb=map(float,m.groups()); print(f'plugin: {int(gr)} layer-steps, distinct experts per layer-step {(h+mi)/gr:.1f}, hit {100*h/max(h+mi,1):.1f} %, read {mb/1000:.2f} GB in total')"
# the generated text is what the CLI logs after its "total time" line (it goes to stderr); drop the log prefix and the plugin's own lines
text() { awk '/total time:/ {on=1; next} on' "$OUT/$1.log" | sed -E 's/^[0-9.]+ [A-Z] //' | grep -v "^moe-cache:"; }
text stock > "$OUT/stock.txt"; text plugin > "$OUT/plugin.txt"
if cmp -s "$OUT/stock.txt" "$OUT/plugin.txt" && [ -s "$OUT/stock.txt" ]; then echo "IDENTICAL text ($(wc -c < "$OUT/stock.txt") bytes)"; else echo "DIFFERENT text (or empty)"; exit 3; fi
