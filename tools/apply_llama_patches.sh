#!/bin/bash
# Apply patches/llama.cpp/*.patch, in name order, to a llama.cpp checkout pinned at the base commit.
#   tools/apply_llama_patches.sh LLAMA_CPP_DIR
# Env: LLAMA_PATCH_DIR (default patches/llama.cpp), LLAMA_BASE_COMMIT (default b11966).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${1:?usage: apply_llama_patches.sh LLAMA_CPP_DIR}"
PDIR="${LLAMA_PATCH_DIR:-$HERE/patches/llama.cpp}"
BASE="${LLAMA_BASE_COMMIT:-ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1}"
git -C "$DIR" rev-parse --git-dir >/dev/null 2>&1 || { echo "FAIL: $DIR is not a git checkout"; exit 1; }
HEAD_SHA="$(git -C "$DIR" rev-parse HEAD)"
[ "$HEAD_SHA" = "$BASE" ] || { echo "FAIL: $DIR is at $HEAD_SHA, the patches need $BASE"; exit 1; }
git -C "$DIR" diff --quiet HEAD || { echo "FAIL: $DIR has uncommitted changes"; exit 1; }
n=0
for p in $(ls "$PDIR"/*.patch 2>/dev/null | sort); do
  echo "check $(basename "$p")"
  git -C "$DIR" apply --check "$p" || { echo "FAIL: $(basename "$p") does not apply to $BASE"; exit 1; }
  git -C "$DIR" apply "$p" || { echo "FAIL: applying $(basename "$p")"; exit 1; }
  n=$((n+1))
done
echo "OK: $n patch(es) applied to $BASE"
