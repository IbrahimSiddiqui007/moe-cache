#!/bin/bash
# agent_bench_pair.sh OUTDIR LIMIT_GB RAM_GB MODEL ALIAS ORDER...   (ORDER words: stock | mc; e.g. stock mc mc stock)
# I30 paired runs on Linux: stock llama-server vs moe-cache --agent, replaying the recorded agent sessions (tests/agent_sessions/*.json).
# Each run: the model's pages dropped from the page cache (checked: stops if more than 1 GB stays cached, see RESULTS 54), server started under a
# hard cgroup limit (systemd-run, no swap), /health + 15 s, clean-run gate (tools/wait_cool.sh 65), replay OpenCode then Hermes, cgroup memory
# events, stop (by its systemd unit, never by a process-name pattern).
# Stock = the llama-server command the launcher prints for --agent --dry-run (computed once, with the GPU idle), without the plugin's environment.
# moe-cache = the launcher with --agent --always-plugin and an empty --slot-save-path (no warm session to restore); stops if the plugin is not active.
# Environment: LLAMA_SERVER (llama-server binary), MOE_CACHE_LIB (plugin library), SESSIONS (folder of recorded sessions).
set -u
O=$1; LIM=$2; RAM=$3; M=$4; AL=$5; shift 5; mkdir -p "$O"; P=${PORT:-8099}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LS=${LLAMA_SERVER:-llama-server}; LIB=${MOE_CACHE_LIB:-$HERE/build/libggml-moe-cache.so}; SES=${SESSIONS:-$HERE/tests/agent_sessions}
MFILES=$(ls "$(dirname "$M")"/$(basename "$M" | sed -E 's/-00001-of-/-*-of-/'))
evict() { python3 - "$@" <<'PY'
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
PY
}
resident_gb() { python3 - "$@" <<'PY'
import ctypes, mmap, os, sys
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]; libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
tot = 0
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY); n = os.fstat(fd).st_size; pg = mmap.PAGESIZE; v = (ctypes.c_ubyte * ((n + pg - 1) // pg))()
    a = libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0); libc.mincore(a, n, v); tot += sum(x & 1 for x in v) * pg
    libc.munmap(a, n); os.close(fd)
print(f"{tot / 1e9:.2f}")
PY
}
log() { echo "$(date +%T) $*" | tee -a "$O/progress.log"; }
CMD=$(MOE_CACHE_LIB=$LIB python3 "$HERE/bin/moe_cache.py" server "$M" --agent --ram "$RAM" --port $P --llama-server "$LS" --no-warm --dry-run -- --alias "$AL" 2>/dev/null | tail -1)
CMD=$LS${CMD#*"$LS"}; log "stock command: $CMD"
k=0
for kind in "$@"; do
  k=$((k + 1)); tag=$k-$kind; U=i30-$tag-$$
  if curl -s -o /dev/null --max-time 2 http://127.0.0.1:$P/health; then log "port $P is in use (another server?): stop"; exit 1; fi
  evict $MFILES; RES=$(resident_gb $MFILES); log "run $k/$# ($kind): model in page cache after eviction: $RES GB"
  awk "BEGIN{exit !($RES > 1.0)}" && { log "eviction failed (the file is mapped or pinned elsewhere): stop"; exit 1; }
  mkdir -p "$O/slots-$tag"
  if [ "$kind" = mc ]; then
    MOE_CACHE_LIB=$LIB MOE_CACHE_STATS=1 systemd-run --user --scope -q --unit=$U -p MemoryMax=${LIM}G -p MemorySwapMax=0 \
      python3 "$HERE/bin/moe_cache.py" server "$M" --agent --always-plugin --ram "$RAM" --port $P --llama-server "$LS" -- --alias "$AL" --slot-save-path "$O/slots-$tag" > "$O/server-$tag.log" 2>&1 &
  else
    echo "stock: $CMD" > "$O/server-$tag.log"
    systemd-run --user --scope -q --unit=$U -p MemoryMax=${LIM}G -p MemorySwapMax=0 $CMD >> "$O/server-$tag.log" 2>&1 &
  fi
  for i in $(seq 900); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health)" = 200 ] && break; sleep 2; done
  # "cache mode" is printed at load; "active:" only at the first computed graph, which with --no-warmup comes with the first request (checked after the replays)
  if [ "$kind" = mc ] && ! grep -q "moe-cache: cache mode" "$O/server-$tag.log"; then
    log "run $k/$# (mc): no 'moe-cache: cache mode' line in the server log (plugin not active): stop"; systemctl --user stop $U.scope; exit 1; fi
  sleep 15; G=$(bash "$HERE/tools/wait_cool.sh" 65 1800)
  log "run $k/$# ($kind): server up, $G"
  for s in opencode_stats hermes_stats; do
    python3 "$HERE/tools/agent_bench.py" replay "$SES/$s.json" --url http://127.0.0.1:$P --label "$kind run $k" --json "$O/$tag-$s.json" > "$O/$tag-$s.txt" 2>&1
    log "run $k/$# ($kind) $s: $(tail -1 "$O/$tag-$s.txt" | cut -c1-230)"
  done
  if [ "$kind" = mc ] && ! grep -q "moe-cache: active:" "$O/server-$tag.log"; then
    log "run $k/$# (mc): no 'moe-cache: active:' line after the replays (the plugin computed nothing): stop"; systemctl --user stop $U.scope; exit 1; fi
  CG=$(systemctl --user show -p ControlGroup --value $U.scope)
  log "run $k/$# ($kind): memory.events $(tr '\n' ' ' < /sys/fs/cgroup$CG/memory.events) peak $(( $(cat /sys/fs/cgroup$CG/memory.peak 2>/dev/null || echo 0) / 1048576 )) MiB"
  systemctl --user stop $U.scope; sleep 5
done
