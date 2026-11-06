#!/bin/bash
# tools/set_version.sh (AUD-080) on a temporary copy of the files it edits: the flow after a release (current version -> next, "## Unreleased" heading)
# and the older placeholder flow. Nothing in the real tree is touched.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail() { echo "FAIL: $*"; exit 1; }

mk() {   # mk DIR VERSION_TEXT   a tree holding the given version in every place
  local d="$1" v="$2" b="${2//-/--}"
  mkdir -p "$d/tools" "$d/docs/releases" "$d/bin" "$d/src"
  cp "$HERE/tools/set_version.sh" "$d/tools/"
  printf '# Changelog\n\n## Unreleased\n\n| Area | Change |\n|---|---|\n| X | new thing |\n\n## 0.3.4 (2026-10-30)\n\nold\n' > "$d/CHANGELOG.md"
  printf '<img src="https://img.shields.io/badge/status-v%s%%20beta-orange">\n## Known issues in v%s (honest list)\n- **v%s vs v0.3.4:** history stays\n' "$b" "$v" "$v" > "$d/README.md"
  printf 'MOE_CACHE_REF="${MOE_CACHE_REF:-v%s}"   # tag\n' "$v" > "$d/install.sh"
  printf 'fprintf(stderr, "moe-cache: plugin v%s (ggml backend API %%d)\\n", 3);\n' "$v" > "$d/src/mc_state.inl"
  printf 'VERSION = "%s"                    # tools/set_version.sh fills this in\n' "$v" > "$d/bin/moe_cache.py"
  printf 'status as of 0.3.3\n' > "$d/docs/ROADMAP.md"; printf 'security\n' > "$d/docs/SECURITY.md"
}

# ---- flow 1: after the 0.3.5 release ----
R="$T/after"; mk "$R" 0.3.5
echo "notes" > "$R/docs/releases/RELEASE_NOTES_v0.3.6.md"
bash "$R/tools/set_version.sh" 0.3.6 >"$T/out" 2>&1 || { cat "$T/out"; fail "next release: set_version.sh stopped"; }
grep -q '^## 0.3.6 (20' "$R/CHANGELOG.md" && ! grep -q 'Unreleased' "$R/CHANGELOG.md" || fail "CHANGELOG heading not set"
grep -q 'status-v0.3.6%20beta' "$R/README.md" || fail "badge"
grep -q '^## Known issues in v0.3.6 ' "$R/README.md" || fail "known issues heading"
grep -q 'v0.3.5 vs v0.3.4' "$R/README.md" || fail "history line in the README must stay"
grep -q 'MOE_CACHE_REF:-v0.3.6}' "$R/install.sh" || fail "install.sh ref"
grep -q 'plugin v0.3.6 (ggml' "$R/src/mc_state.inl" || fail "banner"
grep -q '^VERSION = "0.3.6"' "$R/bin/moe_cache.py" || fail "launcher VERSION"
grep -q '^## 0.3.4 (2026-10-30)' "$R/CHANGELOG.md" || fail "older CHANGELOG heading must stay"

# a pre-release name with a dash (badge text doubles the dash)
R="$T/beta"; mk "$R" 0.3.5; echo n > "$R/docs/releases/RELEASE_NOTES_v0.4.0-beta.md"
bash "$R/tools/set_version.sh" 0.4.0-beta >"$T/out" 2>&1 || { cat "$T/out"; fail "0.4.0-beta: set_version.sh stopped"; }
grep -q 'status-v0.4.0--beta%20beta' "$R/README.md" && grep -q 'VERSION = "0.4.0-beta"' "$R/bin/moe_cache.py" || fail "dash version"
# ... and the next one after that (the current version has a dash now)
sed -i 's/^## 0.4.0-beta.*/&\n/' "$R/CHANGELOG.md"; printf '## Unreleased\n\nx\n\n' | cat - "$R/CHANGELOG.md" > "$R/c2" && mv "$R/c2" "$R/CHANGELOG.md"
echo n > "$R/docs/releases/RELEASE_NOTES_v0.4.0.md"
bash "$R/tools/set_version.sh" 0.4.0 >"$T/out" 2>&1 || { cat "$T/out"; fail "from a dash version: set_version.sh stopped"; }
grep -q 'status-v0.4.0%20beta' "$R/README.md" || fail "badge after a dash version"

# refusals leave every file as it was
R="$T/refuse"; mk "$R" 0.3.5
sum() { (cd "$R" && cat CHANGELOG.md README.md install.sh src/mc_state.inl bin/moe_cache.py | md5sum); }
s0="$(sum)"
if bash "$R/tools/set_version.sh" 0.3.6 >"$T/out" 2>&1; then fail "no release notes file: must refuse"; fi
grep -q "RELEASE_NOTES_v0.3.6.md does not exist" "$T/out" || fail "message for missing notes"
echo n > "$R/docs/releases/RELEASE_NOTES_v0.3.5.md"
if bash "$R/tools/set_version.sh" 0.3.5 >"$T/out" 2>&1; then fail "same version: must refuse"; fi
echo n > "$R/docs/releases/RELEASE_NOTES_v0.3.6.md"; sed -i 's/^## Unreleased/## Later/' "$R/CHANGELOG.md"; s1="$(sum)"
if bash "$R/tools/set_version.sh" 0.3.6 >"$T/out" 2>&1; then fail "no Unreleased heading: must refuse"; fi
grep -q 'no "## Unreleased"' "$T/out" && [ "$s1" = "$(sum)" ] || fail "refusal must say why and change nothing"
if bash "$R/tools/set_version.sh" "x.y" >"$T/out" 2>&1; then fail "bad version name: must refuse"; fi
[ "$s0" != "" ] || fail "checksum"

# ---- flow 2: the placeholder ----
R="$T/placeholder"; mk "$R" 0.0.0.TBD
sed -i "s/^## Unreleased/## 0.0.0.TBD (2026-01-01)/" "$R/CHANGELOG.md"
echo "notes 0.0.0.TBD" > "$R/docs/releases/RELEASE_NOTES_v0.0.0.TBD.md"
for f in docs/ROADMAP.md docs/SECURITY.md; do echo "v0.0.0.TBD" >> "$R/$f"; done
bash "$R/tools/set_version.sh" 0.3.6 >"$T/out" 2>&1 || { cat "$T/out"; fail "placeholder flow stopped"; }
[ -f "$R/docs/releases/RELEASE_NOTES_v0.3.6.md" ] && [ ! -f "$R/docs/releases/RELEASE_NOTES_v0.0.0.TBD.md" ] && grep -q '^VERSION = "0.3.6"' "$R/bin/moe_cache.py" && grep -q '^## 0.3.6 ' "$R/CHANGELOG.md" || fail "placeholder flow result"
echo "set_version_test: PASS"
