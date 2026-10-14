#!/bin/bash
# Logit-identity test at small batches: llama-perplexity with batch 8 (every step goes through the executor's multi-token path) on stock and with the
# plugin, then the KL divergence between them. Exact means mean KLD 0 and the same top token everywhere. Token tests (identity.sh) decode one token
# at a time and process long prompts in big batches, so they never compare batches of 2-8 tokens; this one does.
#   tests/kl_batch.sh MODEL.gguf [--chunks N] [--limit GB]     (LLAMA_PERPLEXITY=path, MOE_CACHE_LIB=plugin, KL_PLUGIN_ENV="VAR=VALUE ...", KL_EXTRA="args for both")
# Exit status 0 = identical, 1 = different, 2 = the plugin was not active.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; shift || true
P="${LLAMA_PERPLEXITY:-llama-perplexity}"; CHUNKS=4; LIMIT=""
while [ $# -gt 0 ]; do case "$1" in --chunks) CHUNKS=$2; shift 2;; --limit) LIMIT=$2; shift 2;; *) echo "unknown: $1"; exit 1;; esac; done
[ -f "$MODEL" ] || { echo "usage: tests/kl_batch.sh MODEL.gguf [--chunks N] [--limit GB]"; exit 1; }
PRE=(); [ -n "$LIMIT" ] && PRE=(systemd-run --user --scope -q -p MemoryMax=${LIMIT}G -p MemorySwapMax=0)
TMP=$(mktemp -d); trap '[ -n "$KL_KEEP" ] && mkdir -p "$KL_KEEP" && cp "$TMP"/*.txt "$KL_KEEP"/ 2>/dev/null; rm -rf "$TMP"' EXIT
cat "$HERE"/README.md "$HERE"/docs/*.md "$HERE"/src/moe_cache_plugin.cpp > "$TMP/text.txt"     # prose and code, more than enough tokens
COMMON=(-m "$MODEL" -f "$TMP/text.txt" -c 512 --chunks "$CHUNKS" -b 8 -ub 8 -ngl 99 --n-cpu-moe 999 --kl-divergence-base "$TMP/base.kld")
[ -n "$KL_EXTRA" ] && COMMON+=($KL_EXTRA)
echo "1/2 stock"; "${PRE[@]}" "$P" "${COMMON[@]}" > "$TMP/stock.txt" 2>&1
. "$HERE/bin/_moe_cache_env.sh"; moe_cache_load_env "$P" "${MOE_CACHE_LIB:-$HERE/build/libggml-moe-cache.so}" "${METHOD:-auto}"
ENVV=("${MOE_CACHE_LOAD[@]}" MOE_CACHE_GGUF="$MODEL"); [ -n "$KL_PLUGIN_ENV" ] && ENVV+=($KL_PLUGIN_ENV)
echo "2/2 with the plugin"; "${PRE[@]}" env "${ENVV[@]}" "$P" "${COMMON[@]}" --kl-divergence > "$TMP/plugin.txt" 2>&1
if ! grep -qE "moe-cache: (cache mode|pass-through mode)" "$TMP/plugin.txt"; then
  echo "FAIL: the plugin was not active in the second run"; grep "moe-cache:" "$TMP/plugin.txt" | tail -5; exit 2
fi
grep "moe-cache:" "$TMP/plugin.txt" | grep -E "executor ran|hit |active:" || true
grep -E "Mean +KLD|Maximum KLD|Same top p" "$TMP/plugin.txt"
# the base logits are stored rounded (16 bits), which leaves a maximum KLD of about 1e-4 and a mean of +-1e-6 between identical runs
KLD=$(grep -m1 -oE "Mean +KLD: +-?[0-9.]+" "$TMP/plugin.txt" | grep -oE -- "-?[0-9.]+$")
if grep -qE "Same top p: +100\.000" "$TMP/plugin.txt" && awk -v k="${KLD:-1}" 'BEGIN { exit !(k <= 1e-5 && k >= -1e-5) }'; then echo "PASS: same logits as stock at batch 8"; exit 0; fi
echo "FAIL: the plugin's logits differ from stock at batch 8"; exit 1
