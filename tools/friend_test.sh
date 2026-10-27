#!/bin/bash
# moe-cache friend test: finds out what works on YOUR machine and writes one report file you can send back.
#   tools/friend_test.sh [--llama-server PATH] [--model MODEL.gguf] [--plugin-lib FILE] [--out FILE]
#
# What it does (nothing is uploaded; the report holds hardware names, tool versions and test results, no prompts, no file contents; your home
# directory is replaced by ~):
#   1  machine: CPU model and instruction sets (AVX2, AVX-VNNI, AVX-512), RAM, GPU devices that your llama-server can use (CUDA / ROCm-HIP / SYCL / Vulkan)
#   2  kernels: bit-exactness and speed of the IQ2_XXS / IQ3_XXS dot products against your llama.cpp, including the AVX-512 versions (needs g++)
#   3  with --model: token identity of stock llama.cpp vs the plugin, then the same with llama.cpp's GPU expert cache, and with the AVX-512 kernels
#
# --model should be a small MoE model that fits in RAM (a 1-4 GB one such as OLMoE or Granite MoE is plenty).
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LS="${LLAMA_SERVER:-$(command -v llama-server || true)}"; MODEL=""; LIB=""; OUT="$PWD/moe-cache-friend-report-$(date +%Y%m%d-%H%M).txt"
while [ $# -gt 0 ]; do case "$1" in
  --llama-server) LS=$2; shift 2;; --model) MODEL=$2; shift 2;; --plugin-lib) LIB=$2; shift 2;; --out) OUT=$2; shift 2;;
  -h|--help) sed -n 2,14p "$0"; exit 0;; *) echo "unknown argument: $1"; exit 1;; esac; done
[ -x "$LS" ] || { echo "llama-server not found: pass --llama-server /path/to/llama-server"; exit 1; }
LS="$(readlink -f "$LS")"
TD="$(mktemp -d)"; TMP="$TD/report"; trap 'rm -rf "$TD"' EXIT     # one private folder: no fixed names next to a file in /tmp
say() { echo "$@" | tee -a "$TMP"; }
run() { echo "\$ $*" >> "$TMP"; "$@" 2>&1 | tee -a "$TMP"; }

say "moe-cache friend test $(date -u +%Y-%m-%dT%H:%MZ)"
say "== 1. machine"
say "kernel: $(uname -sr)   distro: $( . /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"
say "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')   threads: $(nproc)"
say "cpu flags: $(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|fma|f16c|avx_vnni|avx512f|avx512bw|avx512vl|avx512dq|avx512_vnni|avx512vbmi|amx_int8|amx_bf16)$' | tr '\n' ' ')"
say "ram: $(awk '/MemTotal/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
say "llama-server: $LS"
run "$LS" --version
say "devices llama-server can use:"
run "$LS" --list-devices
LD_PATH="$(ldd "$LS" 2>/dev/null | awk '/libggml-cpu|libggml-base/ {print $3}' | head -1)"
say "ggml libraries: ${LD_PATH:-not linked (backends are loaded as plugins; kernel test uses the folder of llama-server)}"
LIBDIR="$(dirname "${LD_PATH:-$LS}")"

say ""
say "== 2. kernels (IQ2_XXS / IQ3_XXS dot products vs llama.cpp)"
if command -v g++ >/dev/null; then
  if g++ -O2 -std=c++17 -I"$HERE/src" "$HERE/tests/kernel_variants.cpp" -o "$TMP.kv" -ldl 2>"$TMP.kverr"; then
    run env -u LD_PRELOAD "$TMP.kv" "$LIBDIR"
    rm -f "$TMP.kv"
  else
    say "could not compile tests/kernel_variants.cpp:"; head -5 "$TMP.kverr" | tee -a "$TMP"
  fi
  rm -f "$TMP.kverr"
else
  say "g++ not found: skipped"
fi

if [ -n "$MODEL" ]; then
  [ -f "$MODEL" ] || { say "model file not found: $MODEL"; MODEL=""; }
fi
if [ -n "$MODEL" ]; then
  [ -n "$LIB" ] || LIB="$(ls "$HERE"/build/libggml-moe-cache.so "$HERE"/lib/libggml-moe-cache.so 2>/dev/null | head -1)"
  say ""
  say "== 3. token identity ($(basename "$MODEL"))"
  if [ -z "$LIB" ] || [ ! -f "$LIB" ]; then
    say "plugin library not found: build it with ./build.sh (GGML_API=3 ./build.sh for llama.cpp 23b0202a1 or newer) or pass --plugin-lib"
  else
    export LLAMA_SERVER="$LS" MOE_CACHE_LIB="$LIB"
    say "-- stock vs plugin"
    run "$HERE/tests/identity.sh" "$MODEL" --tokens 32 | tail -4
    say "-- stock + llama.cpp GPU expert cache vs plugin + the same cache (needs a GPU backend)"
    . "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$LS" "$LIB" auto 2>/dev/null
    if [ "${MOE_CACHE_METHOD:-}" = path ]; then
      # the plugin can feed llama.cpp's GPU cache only when preloaded (LD_PRELOAD); loaded as a file it refuses the mode (T7)
      say "skipped: this llama-server loads its backends as files (GGML_BACKEND_PATH), where the plugin does not support llama.cpp's GPU expert cache"
    else
      IDENT_EXTRA="--moe-cache-mib 1000" IDENT_PLUGIN_ENV="MOE_CACHE_GPU_CACHE=1" run "$HERE/tests/identity.sh" "$MODEL" --tokens 32 | tail -4
    fi
    if grep -q -E 'avx512(f|bw)' /proc/cpuinfo; then
      say "-- stock vs plugin with the AVX-512 IQ kernels (only useful for IQ2_XXS / IQ3_XXS models)"
      IDENT_PLUGIN_ENV="MOE_CACHE_AVX512=1" run "$HERE/tests/identity.sh" "$MODEL" --tokens 32 | tail -4
    fi
  fi
fi

# the report must not contain the user's name or paths
r=$(<"$TMP"); r=${r//"$HOME"/\~}       # literal replacement: a HOME with # . & would break a sed pattern or leave the real path in
printf '%s\n' "$r" | sed -e "s#/home/[^/ ]*#/home/USER#g" > "$OUT"
echo
echo "Report written to: $OUT"
echo "Please send that file back. Thank you!"
