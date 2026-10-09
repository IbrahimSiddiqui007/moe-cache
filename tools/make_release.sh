#!/bin/bash
# Build the release tarball: tools/make_release.sh VERSION   (e.g. 0.1.1). Output: dist/moe-cache-vVERSION-linux-x86_64.tar.gz + SHA256SUMS
set -e
cd "$(dirname "$0")/.."
V="$1"; [ -n "$V" ] || { echo "usage: tools/make_release.sh VERSION"; exit 1; }
./build.sh --portable
N=moe-cache-v$V-linux-x86_64; rm -rf dist/$N; mkdir -p dist/$N
cp -r bin planner tests tools profiles docs LICENSE README.md build.sh third_party dist/$N/ ; mkdir -p dist/$N/build dist/$N/src; cp build/libggml-moe-cache.so dist/$N/build/; cp src/*.c* dist/$N/src/
rm -f dist/$N/tools/make_release.sh
tar -C dist -czf dist/$N.tar.gz $N
( cd dist && sha256sum $N.tar.gz > SHA256SUMS && cat SHA256SUMS )
