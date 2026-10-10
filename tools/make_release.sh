#!/bin/bash
# Build the release tarball: tools/make_release.sh VERSION   (e.g. 0.1.2). Output: dist/moe-cache-vVERSION-linux-x86_64.tar.gz + SHA256SUMS
# The tarball is always built from a clean export of the last commit (git archive HEAD), never from the working tree, so untracked or
# uncommitted files cannot end up in a release. Uncommitted changes are ignored (a warning is printed).
set -e
cd "$(dirname "$0")/.."
V="$1"; [ -n "$V" ] || { echo "usage: tools/make_release.sh VERSION"; exit 1; }
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9]+)?$ ]] || { echo "VERSION must look like 0.1.2 (got '$V')"; exit 1; }
if [ -z "$MOE_CACHE_RELEASE_EXPORT" ]; then
  git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git checkout: cannot build a clean export"; exit 1; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || echo "warning: uncommitted changes are NOT part of this release (only the last commit is)"
  X="$(mktemp -d)"; trap 'rm -rf "$X"' EXIT
  git archive HEAD | tar -x -C "$X"
  MOE_CACHE_RELEASE_EXPORT=1 "$X/tools/make_release.sh" "$V"
  mkdir -p dist; cp "$X"/dist/moe-cache-v"$V"-linux-x86_64.tar.gz "$X"/dist/SHA256SUMS dist/
  echo "built from commit $(git rev-parse --short HEAD) -> dist/"; ( cd dist && cat SHA256SUMS )
  echo "note: SHA256SUMS detects corruption only. Releases built by the GitHub workflow also carry a build attestation (see docs/SECURITY.md)."
  exit 0
fi
./build.sh --portable
N=moe-cache-v$V-linux-x86_64; rm -rf dist/$N; mkdir -p dist/$N
cp -r bin planner tests tools profiles docs bench LICENSE README.md build.sh install.sh third_party dist/$N/ ; mkdir -p dist/$N/build dist/$N/src; cp build/libggml-moe-cache.so dist/$N/build/; cp src/*.c* dist/$N/src/
rm -f dist/$N/tools/make_release.sh
tar -C dist -czf dist/$N.tar.gz $N
( cd dist && sha256sum $N.tar.gz > SHA256SUMS )
