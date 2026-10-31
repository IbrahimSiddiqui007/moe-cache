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
echo "apply_llama_patches_test: PASS"
