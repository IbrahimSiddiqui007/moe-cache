#!/bin/bash
# tools/friend_test.sh scrub_report (AUD-064): the report a friend sends back must not hold their user name or home path, also outside /home
# (/mnt/ssd/NAME/..., /Users/NAME/...). Runs the function alone, with fake HOME / USER.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
eval "$(sed -n '/^# scrub-begin/,/^# scrub-end/p' "$HERE/tools/friend_test.sh")"
fail() { echo "FAIL: $*"; exit 1; }
export HOME=/home/zelda USER=zelda
out=$(printf '%s\n' 'model /home/zelda/models/m.bin' 'on /mnt/ssd/zelda/llm/m.bin' 'mac /Users/linkx/m.bin' 'other /home/someone/x' 'zelda@box$ ls' | scrub_report)
echo "$out" | grep -q "zelda" && fail "the user name is still in the report: $out"
echo "$out" | grep -q "/mnt/ssd/USER/llm/m.bin" || fail "/mnt/ssd/NAME/ not scrubbed: $out"
echo "$out" | grep -q "/Users/USER/m.bin" || fail "/Users/NAME not scrubbed: $out"
echo "$out" | grep -q "/home/USER/x" || fail "another user's /home/NAME not scrubbed: $out"
echo "$out" | grep -q "^model ~/models/m.bin" || fail "HOME not shortened to ~: $out"
# a short user name (2 characters) must not mangle ordinary words
export HOME=/home/al USER=al
out=$(printf '%s\n' 'the model alpha is a normal word list' | scrub_report)
[ "$out" = 'the model alpha is a normal word list' ] || fail "short user name changed the text: $out"
echo "friend_scrub_test: PASS"
