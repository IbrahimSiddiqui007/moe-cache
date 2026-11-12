#!/bin/bash
# PACKAGING-828: the same files give the same archive bytes (zip and tar.gz recipes of the release scripts), whatever the file times are.
set -e
cd "$(dirname "$0")/.."
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/a/pkg/sub" "$T/b/pkg/sub"
for d in a b; do printf 'one\n' > "$T/$d/pkg/x.txt"; printf 'two\n' > "$T/$d/pkg/sub/y.py"; printf '#!/bin/sh\n' > "$T/$d/pkg/run.sh"; chmod 755 "$T/$d/pkg/run.sh"; done
touch -d '2020-01-02 03:04:05' "$T/a/pkg/x.txt" "$T/a/pkg/sub/y.py"; touch -d '2024-05-06 07:08:09' "$T/b/pkg/x.txt" "$T/b/pkg/sub/y.py" "$T/b/pkg/run.sh"
export SOURCE_DATE_EPOCH=1700000000
python3 tools/reproducible_zip.py "$T/a" pkg "$T/a.zip"; python3 tools/reproducible_zip.py "$T/b" pkg "$T/b.zip"
[ "$(sha256sum < "$T/a.zip")" = "$(sha256sum < "$T/b.zip")" ] && echo "  ok   zip: same bytes for different file times" || { echo "  FAIL zip differs"; exit 1; }
python3 - "$T/a.zip" <<'P'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); n = [i.filename for i in z.infolist()]
assert n == sorted(n, key=lambda s: (s.rsplit("/", 1)[0], s)) or n, n
assert all(i.date_time[:3] == (2023, 11, 14) for i in z.infolist()), [i.date_time for i in z.infolist()]
print("  ok   zip: entries dated by SOURCE_DATE_EPOCH, names", n)
P
for d in a b; do tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$SOURCE_DATE_EPOCH" -C "$T/$d" -cf - pkg | gzip -n -9 > "$T/$d.tgz"; done
[ "$(sha256sum < "$T/a.tgz")" = "$(sha256sum < "$T/b.tgz")" ] && echo "  ok   tar.gz: same bytes for different file times" || { echo "  FAIL tar.gz differs"; exit 1; }
tar -tvzf "$T/a.tgz" | awk '{print $2}' | grep -qv '^0/0$' && { echo "  FAIL tar keeps a user name"; exit 1; } || echo "  ok   tar.gz: owner 0/0 only"
grep -q 'tools/reproducible_zip.py' tools/make_windows_kit.sh && grep -q -- '--numeric-owner' tools/make_release.sh && grep -q -- 'gzip -n' tools/make_release.sh && grep -q -- '--no-insert-timestamp' build.sh tools/make_windows_kit.sh && echo "  ok   the release scripts use these recipes" || { echo "  FAIL scripts do not use the recipes"; exit 1; }
echo "reproducible_test: all passed"
