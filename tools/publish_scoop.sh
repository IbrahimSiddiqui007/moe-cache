#!/usr/bin/env bash
# Put a filled scoop manifest into our scoop bucket repo and push it: tools/publish_scoop.sh BUCKET_GIT_URL dist/packaging/scoop/moe-cache.json
# Used by .github/workflows/packages.yml after a release; works on an empty bucket repo too (first release). SCOOP_BUCKET_TOKEN (optional) is sent
# as an HTTP header, never written into a URL or .git/config. Nothing is pushed when the manifest in the bucket is already the same.
set -euo pipefail
url="$1"; manifest="$2"
ver=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$manifest")
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
auth=()
if [ -n "${SCOOP_BUCKET_TOKEN:-}" ]; then
  auth=(-c "http.https://github.com/.extraheader=AUTHORIZATION: basic $(printf 'x-access-token:%s' "$SCOOP_BUCKET_TOKEN" | base64 -w0)")
fi
git "${auth[@]}" clone -q "$url" "$work/bucket" 2>&1 | grep -v "cloned an empty repository" || true
cd "$work/bucket"
if git rev-parse -q --verify origin/main >/dev/null; then git checkout -q -B main origin/main; else git checkout -q --orphan main 2>/dev/null || git checkout -q -B main; fi   # build on the bucket's main, or start it
mkdir -p bucket
cp "$manifest" bucket/moe-cache.json
[ -f README.md ] || printf '# moe-cache scoop bucket\n\n```\nscoop bucket add moe-cache https://github.com/IbrahimSiddiqui007/scoop-moe-cache\nscoop install moe-cache\n```\n\nUpdated by the release workflow of https://github.com/IbrahimSiddiqui007/moe-cache.\n' > README.md
git add -A
if git diff --cached --quiet; then echo "scoop bucket already has moe-cache $ver: nothing to push"; exit 0; fi
git -c user.name="moe-cache release" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" commit -q -m "moe-cache $ver"
git "${auth[@]}" push -q origin HEAD:main
echo "scoop bucket: moe-cache $ver pushed"
