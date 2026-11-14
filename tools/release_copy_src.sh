#!/bin/bash
# The plugin source that goes into a release tarball: tools/release_copy_src.sh DEST_DIR   (run from the repository root; used by tools/make_release.sh)
# Only the regular files of src/ (the C++ plugin: .cpp, .h, .inl). Sub-folders are NOT shipped: a folder such as src/top (the removed native terminal monitor) must not break the copy;
# `cp src/*` stopped the tarball step with "cp: -r not specified; omitting directory 'src/top'" (both v0.3.8 kit builds on 3ec1728). A new sub-folder under src/ is left out
# the same way until it is meant to ship (then list it here and in tests/release_src_test.sh).
set -e
D="${1:?usage: tools/release_copy_src.sh DEST_DIR}"
mkdir -p "$D"
find src -maxdepth 1 -type f -exec cp {} "$D"/ \;
