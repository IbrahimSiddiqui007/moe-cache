#!/usr/bin/env bash
# Build the Linux app (AppImage + deb). One build at a time, low priority: it compiles a few hundred Rust crates.
# Needs: Rust (rustup or distro), Node 18+, and the Tauri Linux libraries
#   Debian/Ubuntu: libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev patchelf
#   Arch:          webkit2gtk-4.1 gtk3 libayatana-appindicator librsvg patchelf
# Nothing is installed system-wide by this script. Usage: scripts/build-linux.sh [--debug]
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/stage.sh
[ -d node_modules ] || npm install --no-audit --no-fund
nice -n 19 npx tauri build ${1:+"$1"} ${TAURI_BUNDLES:+--bundles "$TAURI_BUNDLES"}
find src-tauri/target -path '*bundle*' \( -name '*.AppImage' -o -name '*.deb' \) -print
