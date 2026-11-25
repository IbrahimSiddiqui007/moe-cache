#!/bin/bash
# Runs tools/apply_llama_patches.sh against a tiny temporary git repo (no llama.cpp needed).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
S="$HERE/tools/apply_llama_patches.sh"
R="$T/repo"; mkdir "$R"; cd "$R"
git init -q; git config user.email t@t; git config user.name t
echo one > f.txt; git add f.txt; git commit -qm base
BASE="$(git rev-parse HEAD)"
export LLAMA_BASE_COMMIT="$BASE"
fail() { echo "FAIL: $*"; exit 1; }

mkdir "$T/none"; LLAMA_PATCH_DIR="$T/none" "$S" "$R" | grep -q "0 patch" || fail "zero patches should succeed"

mkdir "$T/good"; echo two > f.txt; git diff > "$T/good/0001-a.patch"; git checkout -q f.txt
echo three > g.txt; git add -N g.txt; git diff > "$T/good/0002-b.patch"; git rm -q --cached g.txt; rm g.txt
LLAMA_PATCH_DIR="$T/good" "$S" "$R" | grep -q "2 patch" || fail "good patches should apply"
[ "$(cat f.txt)" = two ] && [ -f g.txt ] || fail "files not patched"
git checkout -q f.txt; rm -f g.txt

mkdir "$T/bad"; cp "$T/good/0001-a.patch" "$T/bad/"; sed 's/^-one/-zzz/' "$T/good/0001-a.patch" > "$T/bad/0002-bad.patch"
if LLAMA_PATCH_DIR="$T/bad" "$S" "$R" >"$T/out" 2>&1; then fail "bad patch should stop"; fi
grep -q "0002-bad.patch" "$T/out" || fail "failure should name the patch"

echo x > h.txt; git add h.txt; git commit -qm other
if LLAMA_PATCH_DIR="$T/good" "$S" "$R" >"$T/out" 2>&1; then fail "wrong base should stop"; fi
grep -q "need" "$T/out" || fail "wrong-base message"
# AUD-018: the async slot-save writer must use a temp name that is unique per task (two saves of one filename wrote the same NAME.tmp)
P20="$HERE/patches/llama.cpp/0020-server-slot-save-async.patch"
grep -q 'const std::string tmp = filepath + "." + std::to_string(id_task) + ".tmp"' "$P20" || fail "0020: temp file name must be unique per task"
if grep -q 'filepath + ".tmp"' "$P20"; then fail "0020: shared NAME.tmp temp name"; fi
# AUD-064: a patch directory with spaces in its path (the old `for p in $(ls ...)` split it)
git checkout -q -f "$BASE"; git clean -qfd
mkdir "$T/with space"; cp "$T/good/"*.patch "$T/with space/"
LLAMA_PATCH_DIR="$T/with space" "$S" "$R" | grep -q "2 patch" || fail "patch dir with spaces should apply"
# PATCH-834/835: the Vulkan error-path patches keep their guards (model-free; the real apply + build is llama_patched.yml)
P93="$HERE/patches/llama.cpp/0093-vulkan-oom-catch.patch"; P94="$HERE/patches/llama.cpp/0094-vulkan-buffer-leak.patch"
grep -q 'return ggml_backend_vk_graph_compute_impl(backend, cgraph);' "$P93" || fail "0093: compute must run inside the try"
# 0093 behaviour: cut the two functions the patch adds out of the patch text, compile them against stand-ins for the Vulkan types, throw through them
awk '/^\+\/\/ moe-cache patch 0093/{on=1} on&&/^\+/{print substr($0,2)} on&&!/^\+/{exit}' "$P93" > "$T/vk_oom_body.inc"
[ -s "$T/vk_oom_body.inc" ] || fail "0093: could not cut the wrapper out of the patch"
g++ -std=c++17 -Wall -DVK_OOM_BODY="\"$T/vk_oom_body.inc\"" -o "$T/vk_oom_wrapper" "$HERE/tests/vk_oom_wrapper_stub.cpp" 2>"$T/vk_oom.err" || { cat "$T/vk_oom.err"; fail "0093: the wrapper does not compile against the stand-ins"; }
"$T/vk_oom_wrapper" 2>/dev/null | grep -q "vk_oom_wrapper: PASS" || fail "0093: the wrapper test failed (status mapping or state reset)"
grep -q 'device->device.freeMemory(buf->device_memory);' "$P94" || fail "0094: failure path must free the device memory"
grep -q 'device->device.destroyBuffer(buf->buffer);' "$P94" || fail "0094: failure path must destroy the buffer"
grep -q '^+        throw;' "$P94" || fail "0094: failure path must rethrow"
echo "apply_llama_patches_test: PASS"
