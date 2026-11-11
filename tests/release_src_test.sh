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
# 5 the whole tarball step on a copy of this tree with the compile stubbed (build.sh makes a dummy plugin): both tarballs are built, and what must not be in them is not
#   (src/top: experimental native top; docs/internal: design notes; the release scripts themselves)
if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
  mkdir -p "$T/tree"; git ls-files -z | while IFS= read -r -d '' f; do [ -e "$f" ] && mkdir -p "$T/tree/$(dirname "$f")" && cp -p "$f" "$T/tree/$f"; done
  printf '#!/bin/bash\nmkdir -p build && echo stub > build/libggml-moe-cache.so\n' > "$T/tree/build.sh"; chmod +x "$T/tree/build.sh"
  ( cd "$T/tree" && MOE_CACHE_RELEASE_EXPORT=1 MOE_CACHE_RELEASE_NO_LLAMA=1 bash tools/make_release.sh 0.0.1 >"$T/rel.out" 2>&1 ); rc=$?
  chk "make_release.sh runs to the end on this tree (stubbed compile) ($(tail -c 150 "$T/rel.out" | tr '\n' ' '))" '[ $rc = 0 ]'
  n_tar=$(ls "$T"/tree/dist/*.tar.gz 2>/dev/null | wc -l)
  chk "both tarballs (API 2 and API 3) are built" '[ "$n_tar" = 2 ]'
  for tb in "$T"/tree/dist/*.tar.gz; do
    b=$(basename "$tb" .tar.gz); tar -tzf "$tb" > "$T/$b.list"
    chk "$b: no src/top (the experimental native top is not shipped)" '! grep -q "/src/top" "$T/$b.list"'
    chk "$b: no docs/internal (design notes are not shipped)" '! grep -q "/docs/internal" "$T/$b.list"'
    chk "$b: the control page, the plugin sources and the public docs are in" 'grep -q "/ui/index.html$" "$T/$b.list" && grep -q "/src/mc_common.inl$" "$T/$b.list" && grep -q "/docs/QUICKSTART.md$" "$T/$b.list"'
    chk "$b: the release scripts are not shipped" '! grep -q "/tools/make_release.sh$\|/tools/release_copy_src.sh$" "$T/$b.list"'
  done
else
  echo "  skip the end-to-end tarball check: not a git checkout"
fi
[ $fails = 0 ] && echo "release_src_test: all ok" || { echo "release_src_test: $fails failed"; exit 1; }
