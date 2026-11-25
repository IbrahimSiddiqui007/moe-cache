#!/bin/bash
# tools/release_gate.sh with a fake gh (RELEASE-687): a run counts only when its artifact `tested-sha` equals the commit being released.
#   tests/release_gate_test.sh
cd "$(dirname "$0")/.." || exit 1
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fails=0
ok() { echo "  ok   $1"; }
bad() { echo "  FAIL $1"; fails=$((fails+1)); }
chk() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
A=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; B=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
mkdir -p "$T/bin" "$T/data"
# data/<workflow>.runs = run ids, newest first; data/<id>.sha = the tested-sha artifact of that run (no file = the run has no artifact)
cat > "$T/bin/gh" <<'GH'
#!/bin/bash
D="${FAKE_DATA:?}"
case "$1 $2" in
  "run list") w=""; while [ $# -gt 0 ]; do [ "$1" = -w ] && w="$2"; shift; done; [ -f "$D/$w.runs" ] && tr ' ' '\n' < "$D/$w.runs" | grep . ; exit 0;;
  "run download") id="$3"; dest=""; while [ $# -gt 0 ]; do [ "$1" = -D ] && dest="$2"; shift; done
                  [ -f "$D/$id.sha" ] || { echo "no artifact" >&2; exit 1; }; mkdir -p "$dest"; cp "$D/$id.sha" "$dest/tested-sha.txt"; exit 0;;
esac
exit 1
GH
chmod +x "$T/bin/gh"
run() { GATE_SKIP_LOST_WORK=1 FAKE_DATA="$T/data" PATH="$T/bin:$PATH" tools/release_gate.sh "$@" 2>&1; }
setup() { rm -rf "$T/data"; mkdir -p "$T/data"; }

setup; for w in identity windows quick; do echo "1 2" > "$T/data/$w.yml.runs"; done; echo $A > "$T/data/1.sha"; echo $A > "$T/data/2.sha"
out="$(run $A owner/repo)"; rc=$?
chk "all three workflows tested the commit: gate passes" "[ $rc = 0 ] && echo '$out' | grep -c 'tested $A' | grep -qx 3"

setup; for w in identity windows quick; do echo "1" > "$T/data/$w.yml.runs"; done; echo $A > "$T/data/1.sha"
out="$(run $B owner/repo)"; rc=$?
chk "runs that tested another commit do not count (the run's own head sha is not looked at)" "[ $rc != 0 ] && echo '$out' | grep -q 'no successful run tested $B' && echo '$out' | grep -q 'aaaaaaaa'"

setup; echo "1" > "$T/data/identity.yml.runs"; echo "2" > "$T/data/windows.yml.runs"; echo "3" > "$T/data/quick.yml.runs"; echo $A > "$T/data/1.sha"; echo $A > "$T/data/3.sha"
out="$(run $A owner/repo)"; rc=$?
chk "one workflow without a matching run (windows has none) stops the gate and names it" "[ $rc != 0 ] && echo '$out' | grep -q '^windows.yml: no successful run tested' && echo '$out' | grep -q '^identity.yml: run 1 tested'"

setup; for w in identity windows quick; do echo "5 4" > "$T/data/$w.yml.runs"; done; for i in 4; do echo $A > "$T/data/$i.sha"; done
out="$(run $A owner/repo)"; rc=$?
chk "a newer run without the artifact (older than this change) is skipped, an older one that has it counts" "[ $rc = 0 ]"

setup; for w in identity windows quick; do echo "1" > "$T/data/$w.yml.runs"; done
out="$(run $A owner/repo)"; rc=$?
chk "runs without the artifact never count" "[ $rc != 0 ] && echo '$out' | grep -q 'found: none'"

out="$(run abc123 owner/repo)"; rc=$?
chk "a short or odd commit is refused" "[ $rc != 0 ] && echo '$out' | grep -q 'not a full commit sha'"

# TOOLS-1300: the gate also runs tools/lost_work_check.py (as release.yml does), so an unaccepted in-flight branch fails BEFORE the tag.
# Real script and real check in a throw-away git repo: origin/experimental plus one claude/* branch with a commit that is not in it.
setup; for w in identity windows quick; do echo "1" > "$T/data/$w.yml.runs"; done; echo $A > "$T/data/1.sha"
L="$T/lw"; mkdir -p "$L/tools" "$L/archive"; cp tools/release_gate.sh tools/lost_work_check.py "$L/tools/"; : > "$L/tools/lost_work_accepted.tsv"; : > "$L/archive/README.md"
( cd "$L" && git init -q -b experimental . && git config user.email t@t && git config user.name t && git add -A && git commit -qm base \
  && git update-ref refs/remotes/origin/experimental HEAD && git checkout -q -b claude/inflight && echo "work that is not in the release line yet" > newwork.txt && git add newwork.txt && git commit -qm "in-flight work" ) >/dev/null 2>&1
lw() { ( cd "$L" && FAKE_DATA="$T/data" PATH="$T/bin:$PATH" tools/release_gate.sh "$@" 2>&1 ); }
out="$(lw $A owner/repo)"; rc=$?
chk "an unaccepted in-flight claude/* branch fails the gate (runs are fine) and the output names it" "[ $rc != 0 ] && echo '$out' | grep -q 'lost-work check: FAILED' && echo '$out' | grep -q 'claude/inflight' && echo '$out' | grep -q 'identity.yml: run 1 tested'"
printf 'claude/inflight\tparked\tin-flight until the next release\n' > "$L/tools/lost_work_accepted.tsv"
out="$(lw $A owner/repo)"; rc=$?
chk "the same branch listed in the accept file passes the gate" "[ $rc = 0 ] && echo '$out' | grep -q 'lost-work check: ok'"
# a shallow checkout (no origin/experimental, as in the release.yml build job) must not fail the gate
S="$T/shallow"; mkdir -p "$S/tools"; cp tools/release_gate.sh tools/lost_work_check.py "$S/tools/"; ( cd "$S" && git init -q -b main . ) >/dev/null 2>&1
out="$( cd "$S" && FAKE_DATA="$T/data" PATH="$T/bin:$PATH" tools/release_gate.sh $A owner/repo 2>&1 )"; rc=$?
chk "no origin/experimental (shallow CI checkout): the lost-work step is skipped with a reason, the gate still passes" "[ $rc = 0 ] && echo '$out' | grep -q 'lost-work check: skipped here' && ! echo '$out' | grep -q FAILED"
out="$(run $A owner/repo)"
chk "GATE_SKIP_LOST_WORK=1 (tests only) says it skipped, never 'ok'" "echo '$out' | grep -q 'lost-work check: SKIPPED' && ! echo '$out' | grep -q 'lost-work check: ok'"
chk "release.yml and the gate run the same lost_work_check command line" "[ \"\$(grep -o 'lost_work_check.py --repo .*accepted.tsv' .github/workflows/release.yml | head -1)\" = \"\$(grep -o 'lost_work_check.py --repo .*accepted.tsv' tools/release_gate.sh | head -1)\" ]"

# the workflows really write what the gate reads
chk "quick, windows and identity each end with a record job that uploads tested-sha" "[ \$(grep -c 'name: tested-sha' .github/workflows/quick.yml .github/workflows/windows.yml .github/workflows/identity.yml | grep -c ':1\$') = 3 ]"
chk "release.yml calls the gate script and no longer matches runs by head sha" "grep -q 'tools/release_gate.sh' .github/workflows/release.yml && ! grep -q -- '-c \"\$GITHUB_SHA\"' .github/workflows/release.yml"
bash -n tools/release_gate.sh || bad "release_gate.sh parses"
[ $fails = 0 ] && echo "release_gate_test: all passed" || { echo "release_gate_test: $fails failed"; exit 1; }
