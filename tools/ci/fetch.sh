#!/bin/bash
# Download one file for CI: skip when OUT already has the right sha256 (an actions/cache hit), else retry with backoff, then check the sha256.
#   tools/ci/fetch.sh URL OUT [SHA256]
# CI_FETCH_TRIES (default 6) tries, waits 15 30 60 120 240 s between them (about 8 min of waiting in all); CI_FETCH_WAIT sets the first wait (tests use 0).
# Without SHA256 (a file pinned only by its URL, e.g. a Hugging Face commit) an existing OUT is kept as is.
# When every try fails the message names the host, so a red run reads "host down", not "code broken". A WRONG sha256 on a finished download is a
# different failure (the file changed or is cut off); it is retried too, but the final message says so.
set -u
URL="${1:-}"; OUT="${2:-}"; SHA="${3:-}"
[ -n "$URL" ] && [ -n "$OUT" ] || { sed -n 2,7p "$0"; exit 2; }
TRIES="${CI_FETCH_TRIES:-6}"; WAIT="${CI_FETCH_WAIT:-15}"
HOST="$(printf '%s' "$URL" | sed -E 's#^[a-z]+://([^/]+).*#\1#')"
ok() { [ -s "$OUT" ] && { [ -z "$SHA" ] || [ "$(sha256sum "$OUT" | cut -d' ' -f1)" = "$SHA" ]; }; }
if ok; then echo "ci_fetch: $OUT already here${SHA:+ (sha256 ok)}, not downloading"; exit 0; fi
why=""
for i in $(seq 1 "$TRIES"); do
  rm -f "$OUT.part"
  code=$(curl -fsSL --connect-timeout 20 --max-time 900 -o "$OUT.part" -w '%{http_code}' "$URL"); rc=$?
  if [ $rc = 0 ]; then
    if [ -n "$SHA" ] && [ "$(sha256sum "$OUT.part" | cut -d' ' -f1)" != "$SHA" ]; then
      why="sha256 mismatch: got $(sha256sum "$OUT.part" | cut -d' ' -f1), pinned $SHA"
    else mv "$OUT.part" "$OUT"; echo "ci_fetch: $URL ok (try $i)"; exit 0; fi
  else why="curl exit $rc, HTTP $code (host $HOST)"; fi
  echo "ci_fetch: try $i of $TRIES failed: $why"
  # 4xx other than 408 / 429 is the file being gone or the URL wrong, not an outage: retrying for minutes would only hide it
  case "$code" in 408|429) ;; 4??) echo "::error::$URL answered HTTP $code: the file is not there or the URL is wrong. Not a host outage."; rm -f "$OUT.part"; exit 1 ;; esac
  [ "$i" -lt "$TRIES" ] && { sleep "$WAIT"; WAIT=$((WAIT * 2)); [ "$WAIT" -gt 240 ] && WAIT=240; }
done
rm -f "$OUT.part"
case "$why" in
  sha256*) echo "::error::$URL downloaded but the file is not the pinned one ($why). This is NOT a host outage: the asset changed or the pin is wrong." ;;
  *)       echo "::error::download host down: $URL ($why) after $TRIES tries. Not a code failure: re-run when $HOST is back; a cache hit skips this step." ;;
esac
exit 1
