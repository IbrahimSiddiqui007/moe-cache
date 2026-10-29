#!/bin/bash
# S0 of the diffusion plan: a diffusion step is a no-cache prefill of the whole canvas, so time prefill passes of canvas size, stock vs plugin,
# under a memory limit below the model size. llama-bench -p N starts every repetition with an empty KV cache (no carry).
#   tests/diffusion_proxy.sh MODEL.gguf OUTDIR [--bench PATH] [--limit GB] [--ram GB] [--ncmoe N] [--sizes "128 256 512 1024"] [--reps 2] [--lib PATH]
# Each size runs once with stock llama-bench and once with the plugin, in their own systemd scope (MemoryMax, no swap); the order alternates per
# size; the model's page cache is dropped (dd nocache) before every run so neither arm is helped by the other. Output: OUTDIR/<arm>_p<N>.json
# plus plugin_p<N>.log (the plugin's end-of-run line: graphs, hits, misses, read MB) and summary.tsv.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; OUT="$2"; shift 2 || { echo "usage: $0 MODEL OUTDIR [options]"; exit 1; }
BENCH=llama-bench; LIMIT=24; RAM=""; NCMOE=999; SIZES="128 256 512 1024"; REPS=2; LIB=""; THREADS=""; MASK=""
while [ $# -gt 0 ]; do case "$1" in --bench) BENCH=$2; shift 2;; --limit) LIMIT=$2; shift 2;; --ram) RAM=$2; shift 2;; --ncmoe) NCMOE=$2; shift 2;;
  --sizes) SIZES=$2; shift 2;; --reps) REPS=$2; shift 2;; --lib) LIB=$2; shift 2;; --threads) THREADS=$2; shift 2;; --mask) MASK=$2; shift 2;; *) echo "unknown: $1"; exit 1;; esac; done
[ -f "$MODEL" ] || { echo "no such model: $MODEL"; exit 1; }
[ -n "$RAM" ] || RAM=$(python3 -c "print(round($LIMIT*0.93-1.5,2))")     # the plugin's cache stays under the scope limit with room for llama.cpp's own buffers
mkdir -p "$OUT"
. "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$BENCH" "${LIB:-$(moe_cache_pick_lib "$BENCH" "$HERE")}" auto
ARGS=(-m "$MODEL" -ngl 99 --n-cpu-moe "$NCMOE" -fa on -lm mmap -n 0 -r "$REPS" -o json)
[ -n "$THREADS" ] && ARGS+=(-t "$THREADS"); [ -n "$MASK" ] && ARGS+=(-C "$MASK" --cpu-strict 1)
PRE=(systemd-run --user --scope -q -p MemoryMax=${LIMIT}G -p MemorySwapMax=0)
run() {   # run ARM P
  local arm=$1 p=$2 f="$OUT/${1}_p$2"
  dd if="$MODEL" iflag=nocache count=0 status=none 2>/dev/null || true
  local t0=$(date +%s)
  if [ "$arm" = stock ]; then
    "${PRE[@]}" env -u LD_PRELOAD -u GGML_BACKEND_PATH "$BENCH" "${ARGS[@]}" -p "$p" -b "$p" -ub "$p" > "$f.json" 2> "$f.log" || echo "FAILED $arm p=$p"
  else
    "${PRE[@]}" env "${MOE_CACHE_LOAD[@]}" MOE_CACHE_SIZE_GIB=auto MOE_CACHE_GGUF="$MODEL" MOE_CACHE_RAM_GIB="$RAM" MOE_CACHE_STATS=1 "$BENCH" "${ARGS[@]}" -p "$p" -b "$p" -ub "$p" > "$f.json" 2> "$f.log" || echo "FAILED $arm p=$p"
  fi
  echo "$arm p=$p done in $(( $(date +%s) - t0 )) s"
}
i=0
for p in $SIZES; do
  if [ $((i % 2)) = 0 ]; then run stock "$p"; run plugin "$p"; else run plugin "$p"; run stock "$p"; fi
  i=$((i + 1))
done
python3 - "$OUT" $SIZES <<'PY'
import json, re, sys
out, sizes = sys.argv[1], sys.argv[2:]
rows = [["p", "arm", "tok/s", "pass_s", "stdev_ts", "graphs", "groups", "distinct_per_group", "read_GB_per_graph"]]
for p in sizes:
    for arm in ("stock", "plugin"):
        try:
            j = json.load(open(f"{out}/{arm}_p{p}.json"))
            r = j[0]; ts = r["avg_ts"]; sd = r["stddev_ts"]
        except Exception:
            rows.append([p, arm, "-", "-", "-", "-", "-", "-", "-"]); continue
        g = grp = dist = rd = "-"
        if arm == "plugin":
            for line in open(f"{out}/plugin_p{p}.log", errors="replace"):
                m = re.search(r"graphs=(\d+) groups=(\d+) ops=(\d+) hits=(\d+) misses=(\d+).*read=([\d.]+) MB", line)
                if m:
                    g, grp, _, h, mi, mb = m.groups(); g, grp = int(g), int(grp)
                    dist = f"{(int(h) + int(mi)) / max(grp, 1):.1f}"; rd = f"{float(mb) / 1000 / max(g, 1):.2f}"
        rows.append([p, arm, f"{ts:.1f}", f"{int(p) / ts:.2f}", f"{sd:.1f}", g, grp, dist, rd])
open(f"{out}/summary.tsv", "w").write("\n".join("\t".join(map(str, r)) for r in rows) + "\n")
print("\n".join("\t".join(map(str, r)) for r in rows))
PY
