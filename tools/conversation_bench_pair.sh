#!/bin/bash
# conversation_bench_pair.sh: the Linux twin of conversation_bench_pair.ps1. A real multi-turn conversation that fills the context (default 65536), played
# through the moe-cache LAUNCHER in two arms, A B B A, one server at a time, then ONE bench record (conversation_bench.py assemble) and a spread table.
# Arm A and arm B differ only by what you give them: launcher arguments (--args-X), environment (--env-X) or a different launcher checkout (--root-X).
# Same arguments on both arms = the noise floor (the A-vs-A spread). Laptop runs go through the heat guard:
#   handover/tools/cool_run.sh -c 16 -m 28 -- tools/conversation_bench_pair.sh --out DIR --model M.gguf --llama-server LS --ctx 65536
#
# usage: conversation_bench_pair.sh --out DIR --model FILE.gguf --llama-server BIN [options]
#   --ctx N              context to fill (65536)            --port P            server port (8099)
#   --order "A,B,B,A"    arms in the order they run         --label-a / --label-b   names in the record (A, B)
#   --root-a DIR / --root-b DIR   launcher checkout per arm (default: the checkout this script is in); two code versions = two worktrees
#   --args-a "ARGS" / --args-b "ARGS"   launcher arguments after the model, split on spaces (e.g. "--load-mode none", or "--n-cpu-moe 24 -- --alias x")
#   --env-a "K=V K=V" / --env-b "K=V"   environment of that arm's launcher (MOE_CACHE_STATS_SIGNAL=1 also gives the per-turn plugin hit rate)
#   --cache-seed DIR     an XDG_CACHE_HOME to copy into every arm's own cache first (arms start from the same planner records); default: empty cache per arm
#   --max-turns N  --max-tokens N  --legacy   passed to conversation_bench.py run
#   --bench-args "ARGS"  more arguments for conversation_bench.py run (split on spaces)
#   --prep               before the arms, start each distinct arm setup once, alone, until ready (the launcher's auto-tune runs and stores its plan there), stop it, and
#                        give every arm of that setup a copy of that cache: no arm pays for or varies in the tuning walk. Without it every arm tunes in its own empty cache
#                        (unless --cache-seed has the records). Costs one extra load per distinct setup.
#   --no-evict           do not push the model out of the page cache before each arm   --no-cool   skip the quiet gate (tools/wait_cool.sh 65)
#   --settle-s N         seconds to wait after ready, before the conversation (15)
#   --ready-s N          seconds to wait for /health (1800)
# Every arm: its own XDG_CACHE_HOME ($OUT/cache-N-label), the server started in its own process group, health wait, `conversation_bench.py run` with the
# server's pid (VRAM, RAM, CPU peaks), an independent nvidia-smi / MemAvailable sampler, a clean stop (SIGTERM, then SIGKILL) and a check that nothing of it
# is left (port free, no process in its group). An EXIT/INT/TERM trap does the same, so a stopped run leaves no llama-server behind.
# Outputs in DIR: N-label.json per run, N-label.txt, server-N-label.log, cmd-N-label.txt, peaks-N-label.json, record.json, tables.md, progress.log.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT=""; MODEL=""; LS="${LLAMA_SERVER:-}"; CTX=65536; PORT=8099; ORDER="A,B,B,A"; LA=A; LB=B; RA="$HERE"; RB="$HERE"
AA=""; AB=""; EA=""; EB=""; SEED=""; MT=""; MX=""; LEG=0; BA=""; EVICT=1; COOL=1; PREP=0; READY=1800; SETTLE=15
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2;; --model) MODEL=$2; shift 2;; --llama-server) LS=$2; shift 2;; --ctx) CTX=$2; shift 2;; --port) PORT=$2; shift 2;;
    --order) ORDER=$2; shift 2;; --label-a) LA=$2; shift 2;; --label-b) LB=$2; shift 2;; --root-a) RA=$2; shift 2;; --root-b) RB=$2; shift 2;;
    --args-a) AA=$2; shift 2;; --args-b) AB=$2; shift 2;; --env-a) EA=$2; shift 2;; --env-b) EB=$2; shift 2;; --cache-seed) SEED=$2; shift 2;;
    --max-turns) MT=$2; shift 2;; --max-tokens) MX=$2; shift 2;; --legacy) LEG=1; shift;; --bench-args) BA=$2; shift 2;;
    --prep) PREP=1; shift;; --no-evict) EVICT=0; shift;; --no-cool) COOL=0; shift;; --ready-s) READY=$2; shift 2;; --settle-s) SETTLE=$2; shift 2;;
    -h|--help) sed -n '2,32p' "$0"; exit 0;; *) echo "unknown argument: $1 (see --help)" >&2; exit 2;;
  esac
done
[ -n "$OUT" ] && [ -n "$MODEL" ] && [ -n "$LS" ] || { echo "need --out, --model and --llama-server (or LLAMA_SERVER); see --help" >&2; exit 2; }
for f in "$MODEL" "$LS" "$RA/bin/moe_cache.py" "$RB/bin/moe_cache.py"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
IFS=, read -ra KINDS <<< "$ORDER"
[ "${#KINDS[@]}" -gt 0 ] || { echo "--order is empty" >&2; exit 2; }
for k in "${KINDS[@]}"; do [ "$k" = A ] || [ "$k" = B ] || { echo "--order words are A and B, got '$k'" >&2; exit 2; }; done
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
log() { echo "$(date +%T) $*" | tee -a "$OUT/progress.log"; }

SPID=""; SAMP=""
# the arm is its own session (setsid): any live (non-zombie) process in that session still belongs to it
group_alive() { ps -o stat= --sid "$SPID" 2>/dev/null | grep -qv '^Z'; }
# stop_arm: SIGTERM to the arm's process group (the launcher stops its server cleanly), SIGKILL after 40 s; then whatever is left in the group is killed by PID
stop_arm() {
  [ -n "$SAMP" ] && { kill "$SAMP" 2>/dev/null; wait "$SAMP" 2>/dev/null; SAMP=""; }
  [ -n "$SPID" ] || return 0
  kill -TERM -- "-$SPID" 2>/dev/null      # SIGTERM, not SIGINT: a background job of a script starts with SIGINT ignored, and the launcher handles SIGTERM
  for _ in $(seq 40); do group_alive || break; sleep 1; done
  if group_alive; then log "the arm's processes did not stop on SIGTERM: SIGKILL to the group"; kill -KILL -- "-$SPID" 2>/dev/null; sleep 1; fi
  wait "$SPID" 2>/dev/null
  group_alive && log "STILL RUNNING after SIGKILL: $(ps -o pid=,args= --sid "$SPID" | cut -c1-100 | tr '\n' ';')"
  SPID=""
}
trap 'stop_arm' EXIT
trap 'log "interrupted: stopping the server"; stop_arm; exit 130' INT TERM

port_pid() { ss -ltnH "sport = :$PORT" -p 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
port_busy() { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null)" != 000 ] || [ -n "$(port_pid)" ]; }
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
# sampler: peak VRAM used (whole card, nvidia-smi, MiB) and lowest MemAvailable (MiB), every second, written to a json file when stopped (SIGTERM)
sampler() {
  python3 - "$1" <<'PY' &
import json, signal, subprocess, sys, time
out = sys.argv[1]; st = {"vram_mib_peak": None, "vram_mib_start": None, "mem_available_mib_min": None, "mem_available_mib_start": None, "samples": 0}
stop = []
signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
def vram():
    try: return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=5, stderr=subprocess.DEVNULL).split()[0])
    except Exception: return None
def avail():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"): return int(l.split()[1]) // 1024
    except Exception: return None
def dump():
    json.dump(st, open(out, "w"))
while not stop:
    v, a = vram(), avail(); st["samples"] += 1
    if v is not None:
        st["vram_mib_peak"] = v if st["vram_mib_peak"] is None else max(st["vram_mib_peak"], v); st["vram_mib_start"] = st["vram_mib_start"] if st["vram_mib_start"] is not None else v
    if a is not None:
        st["mem_available_mib_min"] = a if st["mem_available_mib_min"] is None else min(st["mem_available_mib_min"], a); st["mem_available_mib_start"] = st["mem_available_mib_start"] if st["mem_available_mib_start"] is not None else a
    if st["samples"] % 5 == 1: dump()
    time.sleep(1)
dump()
PY
  SAMP=$!
}

mapfile -t MFILES < <(compgen -G "$(dirname "$MODEL")/$(basename "$MODEL" | sed -E 's/-00001-of-/-*-of-/')")
[ "${#MFILES[@]}" -gt 0 ] || MFILES=("$MODEL")
log "conversation pair: model $MODEL, ctx $CTX, order $ORDER, A=$LA [root $RA] args '$AA' env '$EA', B=$LB [root $RB] args '$AB' env '$EB'"
log "llama-server $LS; host $(hostname); labels for the record: machine laptop, one machine, forecast"
if port_busy; then log "port $PORT is in use (another server?): stop"; exit 1; fi

arm_setup() {   # sets LAB ROOT ARGS ENVS for arm kind A or B
  if [ "$1" = A ]; then LAB=$LA; ROOT=$RA; ARGS=$AA; ENVS=$EA; else LAB=$LB; ROOT=$RB; ARGS=$AB; ENVS=$EB; fi
}
# start_server CACHE LOGFILE CMDFILE: starts the launcher (own session, own XDG_CACHE_HOME), waits for /health; sets SPID and LOADS; stops everything and exits 1 when it never gets ready
start_server() {
  local cache=$1 logf=$2 cmdf=$3 t0 ready=0 AARR EARR
  read -ra AARR <<< "$ARGS"; read -ra EARR <<< "$ENVS"
  CMD=(env XDG_CACHE_HOME="$cache" "${EARR[@]}" python3 "$ROOT/bin/moe_cache.py" server "$MODEL" --llama-server "$LS" --ctx "$CTX" --port "$PORT" "${AARR[@]}")
  printf '%q ' "${CMD[@]}" > "$cmdf"; echo >> "$cmdf"
  log "  ${CMD[*]}"
  t0=$(date +%s.%N)
  setsid "${CMD[@]}" > "$logf" 2>&1 &
  SPID=$!
  for _ in $(seq $((READY / 2))); do
    kill -0 "$SPID" 2>/dev/null || break
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = 200 ] && { ready=1; break; }
    sleep 2
  done
  if [ "$ready" != 1 ]; then log "  the server did not become ready (see $logf)"; stop_arm; exit 1; fi
  LOADS=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $t0}")
}
evict_model() {
  [ "$EVICT" = 1 ] || return 0
  evict "${MFILES[@]}"; RES=$(resident_gb "${MFILES[@]}"); log "$1: model in page cache after eviction: $RES GB"
  awk "BEGIN{exit !($RES > 1.0)}" && log "WARNING: eviction left $RES GB cached (the arm starts warmer than a cold start; A B B A balances it)"
}

# --prep: every distinct arm setup starts once, alone, in a cache of its own (seeded from --cache-seed) until it is ready, then stops: the launcher's auto-tune and
# probe records are stored there, and every arm of that setup starts from a copy of it. Arms with identical setups share one prep.
declare -A PREPDIR
if [ "$PREP" = 1 ]; then
  for kind in A B; do
    case " ${KINDS[*]} " in *" $kind "*) ;; *) continue;; esac
    arm_setup "$kind"; key="$ROOT|$ARGS|$ENVS"; prior=""
    for k2 in A B; do [ "$k2" != "$kind" ] && [ -n "${PREPDIR[$k2]:-}" ] && { arm_setup "$k2"; [ "$ROOT|$ARGS|$ENVS" = "$key" ] && prior="${PREPDIR[$k2]}"; }; done
    arm_setup "$kind"
    if [ -n "$prior" ]; then PREPDIR[$kind]=$prior; log "prep $LAB: same setup as the other arm, sharing $prior"; continue; fi
    pc="$OUT/cache-prep-$LAB"; rm -rf "$pc"; mkdir -p "$pc"
    [ -n "$SEED" ] && { cp -a "$SEED"/. "$pc"/ || { log "cache seed not copied: stop"; exit 1; }; }
    if port_busy; then log "prep $LAB: port $PORT is in use: stop"; exit 1; fi
    evict_model "prep $LAB"; log "prep $LAB: starting once to let the launcher tune and store its plan"
    start_server "$pc" "$OUT/server-prep-$LAB.log" "$OUT/cmd-prep-$LAB.txt"
    log "prep $LAB: ready after ${LOADS} s (the stored plan is in $pc)"
    stop_arm; [ -n "$(port_pid)" ] && { log "prep $LAB: port $PORT still held after the stop: stop"; exit 1; }
    PREPDIR[$kind]=$pc
  done
fi

RUNS=(); n=0
for kind in "${KINDS[@]}"; do
  n=$((n + 1)); arm_setup "$kind"
  tag="$n-$LAB"; CACHE="$OUT/cache-$tag"; rm -rf "$CACHE"; mkdir -p "$CACHE"
  src="$SEED"; [ "$PREP" = 1 ] && src="${PREPDIR[$kind]}"
  if [ -n "$src" ]; then cp -a "$src"/. "$CACHE"/ || { log "cache seed not copied: stop"; exit 1; }; fi
  if port_busy; then log "run $n/${#KINDS[@]} ($LAB): port $PORT is in use: stop"; exit 1; fi
  evict_model "run $n/${#KINDS[@]} ($LAB)"
  log "run $n/${#KINDS[@]} ($LAB):"
  sampler "$OUT/peaks-$tag.json"
  start_server "$CACHE" "$OUT/server-$tag.log" "$OUT/cmd-$tag.txt"
  LPID=$(port_pid); log "run $n ($LAB): ready after ${LOADS} s, llama-server pid ${LPID:-unknown}"
  sleep "$SETTLE"
  if [ "$COOL" = 1 ] && [ -x "$HERE/tools/wait_cool.sh" ]; then G=$(bash "$HERE/tools/wait_cool.sh" 65 1800); log "run $n ($LAB): $G"; fi
  BARGS=(run --url "http://127.0.0.1:$PORT" --arm "$LAB" --ctx "$CTX" --out "$OUT/$tag.json" --command-line "@$OUT/cmd-$tag.txt" --server-log "$OUT/server-$tag.log" --load-s "$LOADS")
  [ -n "$LPID" ] && BARGS+=(--monitor-pid "$LPID")
  case " $ENVS " in *" MOE_CACHE_STATS_SIGNAL=1 "*) [ -n "$LPID" ] && BARGS+=(--stats-pid "$LPID" --stats-log "$OUT/server-$tag.log");; esac
  [ -n "$MX" ] && BARGS+=(--max-tokens "$MX"); [ -n "$MT" ] && BARGS+=(--max-turns "$MT"); [ "$LEG" = 1 ] && BARGS+=(--legacy)
  read -ra BAARR <<< "$BA"
  python3 "$HERE/tools/conversation_bench.py" "${BARGS[@]}" "${BAARR[@]}" > "$OUT/$tag.txt" 2>&1
  rc=$?
  log "run $n ($LAB): exit $rc: $(tail -1 "$OUT/$tag.txt" | cut -c1-200)"
  stop_arm
  [ -n "$(port_pid)" ] && { log "run $n ($LAB): port $PORT still held after the stop: stop"; exit 1; }
  if [ -f "$OUT/$tag.json" ]; then RUNS+=("$OUT/$tag.json"); else log "run $n: no result file (see $OUT/$tag.txt)"; fi
  sleep 3
done

if [ "${#RUNS[@]}" = "${#KINDS[@]}" ]; then
  python3 "$HERE/tools/conversation_bench.py" assemble "${RUNS[@]}" --out "$OUT/record.json" --model "$MODEL" --llama-server "$LS" --cache cold 2>&1 | tee "$OUT/tables.md"
  python3 "$HERE/tools/conversation_bench.py" spread "$OUT/record.json" 2>&1 | tee -a "$OUT/tables.md"
  log "done: $OUT/record.json, $OUT/tables.md  (peaks: $OUT/peaks-*.json)"
else
  log "done, but not every run produced a result file: no record written"; exit 1
fi
