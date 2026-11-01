#!/bin/bash
# Logit check for LLAMA_LAYERWISE_PREFILL (patches/llama.cpp/0050): stock order against layer-wise order on a tiny random qwen4exp model.
#   tests/layerwise_check.sh LLAMA_CPP_DIR [N=4] [n_gpu_layers=0]
# LLAMA_CPP_DIR: a llama.cpp checkout at the pin with 0050 applied and built (build/bin/test-llama-archs and libllama in LLAMA_CPP_DIR/build;
#   BUILD_DIR=build-cuda for another build directory). No real qwen4exp model is needed: test-llama-archs writes a 68 MB random one (8 layers,
#   linear and full attention, QSA indexer with a top-k of 32 so it really selects, 8 experts).
# For every configuration the same random prompt is decoded twice (one process each: the switch is read once) and all logits are compared.
# Exact is expected on the CPU (max|diff| 0); a GPU may round differently, so there NMSE below 1e-6 and the same best token everywhere is the pass.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LC="$(cd "${1:?usage: layerwise_check.sh LLAMA_CPP_DIR [N] [n_gpu_layers]}" && pwd)"
N=${2:-4}; NGL=${3:-0}; B="$LC/${BUILD_DIR:-build}"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
(cd "$W" && LLAMA_TEST_QWEN4EXP_LAYERS=8 LLAMA_TEST_QWEN4EXP_CTX=2048 LLAMA_TEST_QWEN4EXP_TOPK=32 "$B/bin/test-llama-archs" -a qwen4exp -o . >/dev/null 2>&1)
M="$W/qwen4exp-moe.gguf"; [ -f "$M" ] || { echo "FAIL: no model written"; exit 1; }
g++ -O2 -std=c++17 -I"$LC/include" -I"$LC/ggml/include" "$HERE/layerwise_logits.cpp" -L"$B/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin" -o "$W/lwl"
fail=0
# n_tokens ub batch all|last: batch/ub chunks per decode call; ub 30 and 17 are not multiples of the k-pool size (4), so pools straddle chunks
for cfg in "301 30 250 all" "301 30 250 last" "500 50 400 last" "500 17 170 all" "1000 64 640 last" "700 64 640 all"; do
  "$W/lwl" run "$M" "$W/a.bin" $cfg "$NGL" >/dev/null 2>&1
  LLAMA_LAYERWISE_PREFILL=$N "$W/lwl" run "$M" "$W/b.bin" $cfg "$NGL" >"$W/b.log" 2>&1
  grep -q "layer-wise prefill on" "$W/b.log" || { echo "FAIL [$cfg]: the layer-wise path did not run"; fail=1; continue; }
  echo -n "[$cfg] N=$N ngl=$NGL: "; "$W/lwl" cmp "$W/a.bin" "$W/b.bin" || fail=1
done
# refused cases must still give the stock answer: one chunk per batch (nothing to group) and N=1
LLAMA_LAYERWISE_PREFILL=1 "$W/lwl" run "$M" "$W/c.bin" 300 32 256 all "$NGL" >"$W/c.log" 2>&1
grep -q "ignored" "$W/c.log" || { echo "FAIL: N=1 was not refused"; fail=1; }
"$W/lwl" run "$M" "$W/a.bin" 300 32 256 all "$NGL" >/dev/null 2>&1
echo -n "[N=1 refused]: "; "$W/lwl" cmp "$W/a.bin" "$W/c.bin" || fail=1
[ $fail = 0 ] && echo "PASS" || { echo "FAIL"; exit 1; }
