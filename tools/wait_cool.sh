#!/bin/bash
# wait_cool.sh [MAX_C] [TIMEOUT_S]: block until the CPU package temperature has been at or below MAX_C (default 65) for three samples in a row
# and the 1-minute load average is below 1.5; exit 1 on timeout. For scripts that time things: the clean-run rule as one command.
MAX=${1:-65}; TMO=${2:-3600}; ok=0; end=$((SECONDS + TMO))
temp() {
  local t; t=$(sensors 2>/dev/null | grep -m1 -E "Package id 0|Tctl|Tdie" | grep -o '+[0-9]*' | head -1 | tr -d +)
  if [ -z "$t" ]; then for z in /sys/class/thermal/thermal_zone*/temp; do [ -r "$z" ] && { t=$(( $(cat "$z") / 1000 )); break; }; done; fi
  echo "${t:-99}"
}
while [ $SECONDS -lt $end ]; do
  t=$(temp); l=$(cut -d' ' -f1 /proc/loadavg)
  if [ "$t" -le "$MAX" ] && awk "BEGIN{exit !($l < 1.5)}"; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -ge 3 ] && { echo "cool: ${t} C, load $l"; exit 0; }
  sleep 10
done
echo "timeout: still ${t} C, load $l"; exit 1
