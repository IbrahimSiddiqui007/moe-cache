#!/usr/bin/env bash
# Copy the kit files the app needs into src-tauri/resources/moe-cache/ (what `bundle.resources` packs).
# Usage: scripts/stage.sh [--with-plugin PATH_TO_PLUGIN_FILE]...   (the plugin and llama.cpp are added by the CI job; see docs/internal/DESKTOP_APP_DESIGN.md)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"; repo="$(cd "$here/.." && pwd)"
dst="$here/src-tauri/resources/moe-cache"
mkdir -p "$dst"; find "$dst" -mindepth 1 ! -name .gitkeep -delete
cp -r "$repo/bin" "$repo/ui" "$repo/planner" "$dst/"
cp "$repo/models.json" "$repo/desktop/app_entry.py" "$dst/"
find "$dst" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -f "$dst"/bin/moe-cache-ui.bat
while [ "${1:-}" = "--with-plugin" ]; do cp "$2" "$dst/"; shift 2; done
echo "staged $(du -sh "$dst" | cut -f1) in $dst"
