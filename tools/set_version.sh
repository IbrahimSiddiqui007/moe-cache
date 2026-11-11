#!/bin/bash
# Put the release's version into every place that names it: tools/set_version.sh VERSION   (e.g. 0.3.6)
# It only edits files: no commit, no tag, nothing published. Two flows, picked by what the files hold:
#
#  1. After a release (the normal case): the files name the CURRENT release (bin/moe_cache.py VERSION = "0.3.5") and CHANGELOG.md has a
#     "## Unreleased" (or "## X.Y.Z (unreleased)") heading. The script changes, from the current version to VERSION:
#       CHANGELOG.md          "## Unreleased" or "## X.Y.Z (unreleased)" -> "## VERSION (today's date)"
#       README.md             the status badge, and the heading "Known issues in vCURRENT"
#       install.sh            MOE_CACHE_REF default (the release workflow checks it matches the tag)
#       src/mc_state.inl      the plugin's start-up banner
#       bin/moe_cache.py      VERSION (--version, the check header)
#     docs/releases/RELEASE_NOTES_vVERSION.md (the release text the release workflow uses) must exist already: write it first, the script does not invent it.
#     docs/ROADMAP.md and the history lines in README.md stay as they are (they are prose; review them in git diff).
#  2. Older flow, kept: every place holds the placeholder 0.0.0.TBD and docs/releases/RELEASE_NOTES_v0.0.0.TBD.md exists: the placeholder is replaced
#     everywhere and the notes file is renamed.
set -e
cd "$(dirname "$0")/.."
V="$1"; P=0.0.0.TBD
[ -n "$V" ] || { echo "usage: tools/set_version.sh VERSION   (e.g. 0.3.6)"; exit 1; }
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?([.-][A-Za-z0-9]+)?$ ]] || { echo "VERSION must look like 0.3.6 or 0.4.0-beta (got '$V')"; exit 1; }
esc() { printf '%s' "$1" | sed 's/[][\.*^$/&]/\\&/g'; }      # a version as a sed pattern / replacement
B="${V//-/--}"                                                # shields.io badge text: a dash is written as two

FILES="CHANGELOG.md README.md docs/ROADMAP.md docs/SECURITY.md install.sh src/mc_state.inl bin/moe_cache.py docs/releases/RELEASE_NOTES_v$P.md"
have_placeholder=1
for f in $FILES; do [ -f "$f" ] && grep -q -F "$P" "$f" || have_placeholder=0; done

if [ "$have_placeholder" = 1 ]; then
  sed -i "s/status-v$P%20beta/status-v$B%20beta/" README.md
  sed -i "s/^## $P ([0-9-]*)/## $V ($(date +%Y-%m-%d))/" CHANGELOG.md
  for f in $FILES; do sed -i "s/$P/$V/g" "$f"; done
  git mv "docs/releases/RELEASE_NOTES_v$P.md" "docs/releases/RELEASE_NOTES_v$V.md" 2>/dev/null || mv "docs/releases/RELEASE_NOTES_v$P.md" "docs/releases/RELEASE_NOTES_v$V.md"
  if grep -rn -F "$P" --exclude-dir=.git --exclude=set_version.sh .; then echo "placeholder still found above"; exit 1; fi
  echo "version $V set in: $FILES (renamed to docs/releases/RELEASE_NOTES_v$V.md). Review with git diff; nothing committed or tagged."
  exit 0
fi

# ---- flow 1: from the current release to V ----
C="$(sed -n 's/^VERSION = "\([^"]*\)".*/\1/p' bin/moe_cache.py | head -n 1)"
[ -n "$C" ] || { echo "cannot read the current version from bin/moe_cache.py (VERSION = \"...\")"; exit 1; }
[ "$V" != "$C" ] || { echo "VERSION $V is the current version already (bin/moe_cache.py)"; exit 1; }
CB="${C//-/--}"; Ce="$(esc "$C")"; Ve="$(esc "$V")"
fail() { echo "$1 (nothing was changed)"; exit 1; }
UNREL='^## \(Unreleased\|[0-9][0-9A-Za-z.-]* ([Uu]nreleased)\)'      # the heading of the version being written: "## Unreleased" or, since 0.3.8, "## 0.3.8 (unreleased)"
grep -q "$UNREL" CHANGELOG.md                               || fail 'CHANGELOG.md has no "## Unreleased" (or "## X.Y.Z (unreleased)") heading to turn into the release heading'
[ -f "docs/releases/RELEASE_NOTES_v$V.md" ]                           || fail "docs/releases/RELEASE_NOTES_v$V.md does not exist: write the release notes first (the release workflow publishes them)"
grep -q "status-v$CB\(%20beta\)\{0,1\}-" README.md          || fail "README.md: no status badge for v$C"
grep -q "MOE_CACHE_REF:-v$C}" install.sh                    || fail "install.sh: MOE_CACHE_REF default is not v$C"
grep -q "plugin v$C (ggml backend API" src/mc_state.inl     || fail "src/mc_state.inl: the start-up banner does not say v$C"
sed -i "s/$UNREL.*/## $V ($(date +%Y-%m-%d))/" CHANGELOG.md
sed -i "s/status-v$(esc "$CB")\(%20beta\)\{0,1\}-/status-v$(esc "$B")\1-/; s/^## Known issues in v$Ce\b/## Known issues in v$Ve/" README.md
sed -i "s/MOE_CACHE_REF:-v$Ce}/MOE_CACHE_REF:-v$Ve}/" install.sh
sed -i "s/plugin v$Ce (ggml backend API/plugin v$Ve (ggml backend API/" src/mc_state.inl
sed -i "s/^VERSION = \"$Ce\"/VERSION = \"$Ve\"/" bin/moe_cache.py
# every place must now say V
for pair in "CHANGELOG.md:^## $Ve (" "README.md:status-v$(esc "$B")" "install.sh:MOE_CACHE_REF:-v$Ve}" "src/mc_state.inl:plugin v$Ve (" "bin/moe_cache.py:^VERSION = \"$Ve\""; do
  grep -q -- "${pair#*:}" "${pair%%:*}" || { echo "FAIL: ${pair%%:*} does not name $V after the edit"; exit 1; }
done
echo "version $C -> $V in: CHANGELOG.md README.md install.sh src/mc_state.inl bin/moe_cache.py. Review README.md / docs/ROADMAP.md prose with git diff; nothing committed or tagged."
