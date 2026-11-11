#!/bin/bash
# The release tarball's src/ step (tools/release_copy_src.sh, called by tools/make_release.sh): both v0.3.8 kit builds on 3ec1728 stopped at "cp: -r not specified; omitting
# directory 'src/top'" because src/top (the experimental native moe-cache-top, not shipped) is a folder. Runs the real helper on this repository's src/ and on a fake one.
#   tests/release_src_test.sh
cd "$(dirname "$0")/.." || exit 1
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fails=0
ok() { echo "  ok   $1"; }
bad() { echo "  FAIL $1"; fails=$((fails+1)); }
chk() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
HELPER="$PWD/tools/release_copy_src.sh"
# 1 this repository's src/: succeeds, ships every plain file, no sub-folder
bash "$HELPER" "$T/real" >"$T/real.out" 2>&1; rc=$?
chk "the real src/: the helper exits 0 ($(head -c 120 "$T/real.out"))" '[ $rc = 0 ]'
n_src=$(find src -maxdepth 1 -type f | wc -l); n_out=$(find "$T/real" -maxdepth 1 -type f | wc -l)
chk "every plain file of src/ is in the tarball's src/ ($n_out of $n_src)" '[ "$n_src" = "$n_out" ] && [ "$n_out" -gt 10 ]'
chk "no sub-folder of src/ is shipped (src/top is experimental)" '[ -z "$(find "$T/real" -mindepth 1 -type d)" ]'
chk "the plugin's own sources are there (mc_common.inl)" '[ -f "$T/real/mc_common.inl" ]'
# 2 a fake src/ with a folder in it, run from its root like make_release.sh does
mkdir -p "$T/fake/src/top" "$T/fake/src/other"; : > "$T/fake/src/a.h"; : > "$T/fake/src/b.cpp"; : > "$T/fake/src/top/main.cpp"; : > "$T/fake/src/other/x.h"
( cd "$T/fake" && bash "$HELPER" "$T/fakeout" >"$T/fake.out" 2>&1 ); rc=$?
chk "a src/ with sub-folders: exit 0 (cp src/* failed here)" '[ $rc = 0 ]'
chk "...only the plain files are copied" '[ "$(ls "$T/fakeout" | tr "\n" " ")" = "a.h b.cpp " ]'
# 3 the old command really fails on that tree (so this test would have caught the break)
( cd "$T/fake" && set -e && mkdir -p "$T/old" && cp src/* "$T/old/" ) >/dev/null 2>&1; rc=$?
chk "the old 'cp src/*' exits non-zero on such a tree" '[ $rc != 0 ]'
# 4 make_release.sh uses the helper and no longer has the bare cp
chk "make_release.sh calls tools/release_copy_src.sh" 'grep -q "tools/release_copy_src.sh dist/\$N/src" tools/make_release.sh'
chk "make_release.sh has no bare 'cp src/*'" '! grep -q "cp src/\*" tools/make_release.sh'
chk "the helper is not shipped inside the tarball's tools/" 'grep -q "dist/\$N/tools/release_copy_src.sh" tools/make_release.sh'
[ $fails = 0 ] && echo "release_src_test: all ok" || { echo "release_src_test: $fails failed"; exit 1; }
