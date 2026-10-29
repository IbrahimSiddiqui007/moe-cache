#!/bin/bash
# Put the release's version into every place that names it: tools/set_version.sh VERSION   (e.g. 0.2.4)
# Until the owner names the release, those places hold the placeholder 0.0.0.TBD. Places: CHANGELOG heading (and today's date), README badge,
# docs/ROADMAP.md, docs/SECURITY.md, install.sh MOE_CACHE_REF (the release workflow checks it matches the tag), the plugin's start-up banner, the launcher (--version and the check header),
# and the release notes (renamed to docs/RELEASE_NOTES_vVERSION.md, which the release workflow uses as the release text).
# It only edits files: no commit, no tag, nothing published.
set -e
cd "$(dirname "$0")/.."
V="$1"; P=0.0.0.TBD
[ -n "$V" ] || { echo "usage: tools/set_version.sh VERSION   (e.g. 0.2.4)"; exit 1; }
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?([.-][A-Za-z0-9]+)?$ ]] || { echo "VERSION must look like 0.2.4 or 0.3.0-beta (got '$V')"; exit 1; }
FILES="CHANGELOG.md README.md docs/ROADMAP.md docs/SECURITY.md install.sh src/mc_state.inl bin/moe_cache.py docs/RELEASE_NOTES_v$P.md"
for f in $FILES; do grep -q -F "$P" "$f" || { echo "no placeholder $P in $f (already set?)"; exit 1; }; done
B="${V//-/--}"                                   # shields.io badge text: a dash is written as two
sed -i "s/status-v$P%20beta/status-v$B%20beta/" README.md
sed -i "s/^## $P ([0-9-]*)/## $V ($(date +%Y-%m-%d))/" CHANGELOG.md
for f in $FILES; do sed -i "s/$P/$V/g" "$f"; done
git mv "docs/RELEASE_NOTES_v$P.md" "docs/RELEASE_NOTES_v$V.md" 2>/dev/null || mv "docs/RELEASE_NOTES_v$P.md" "docs/RELEASE_NOTES_v$V.md"
if grep -rn -F "$P" --exclude-dir=.git --exclude=set_version.sh .; then echo "placeholder still found above"; exit 1; fi
echo "version $V set in: $FILES (renamed to docs/RELEASE_NOTES_v$V.md). Review with git diff; nothing committed or tagged."
