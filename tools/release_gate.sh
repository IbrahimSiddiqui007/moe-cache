#!/bin/bash
# Release gate (RELEASE-687): identity, windows and quick must each have a successful run that TESTED the commit being released, and the lost-work check must pass.
#   tools/release_gate.sh COMMIT [REPO]        (needs gh and a token that can read Actions runs and artifacts)
# A run's own head SHA is not enough: a run started on one branch can test another ref (the `ref` input of windows.yml / identity.yml). Every one
# of those workflows therefore ends with a `record` job (only after all its jobs passed, on the same commit) that uploads the artifact `tested-sha`
# holding `git rev-parse HEAD` of what it tested. This script looks at the newest successful runs of each workflow and accepts the first whose
# tested-sha equals COMMIT.
# TOOLS-1305 (text-only reuse): for identity.yml and windows.yml (the ~1 h workflows) a run that tested an ANCESTOR counts too, as long as every commit between
# that ancestor and COMMIT changed only text files (docs/**/*.md and *.txt, any *.md, tools/lost_work_accepted.tsv; images, html, json and code never; a merge commit or any other path ends the walk). quick.yml
# always has to have run on COMMIT itself. Needs the history: release.yml checks out 20 commits (GATE_TEXT_DEPTH, default 20, is how far the walk goes). Runs older than the change (no artifact) never count. GATE_RUNS = how many runs per workflow to look at (default 100: quick also runs on every claude/** push, so the newest 30 could miss the release commit).
set -u
SHA="${1:?commit sha}"; REPO="${2:-${GITHUB_REPOSITORY:-}}"; N="${GATE_RUNS:-100}"
echo "$SHA" | grep -Eq '^[0-9a-f]{40}$' || { echo "release gate: '$SHA' is not a full commit sha"; exit 1; }
[ -n "$REPO" ] || { echo "release gate: no repository (pass it as the 2nd argument or set GITHUB_REPOSITORY)"; exit 1; }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
bad=0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# CHAIN = COMMIT, then its parent while the commit's diff to that parent touches only text files (single-parent commits only)
text_only() { # $1 = commit: true when it has exactly one parent and its diff to it holds only text paths
  local c="$1" files f
  [ "$(git -C "$ROOT" rev-list --parents -n 1 "$c" 2>/dev/null | wc -w)" = 2 ] || return 1
  files="$(git -C "$ROOT" diff --name-only "$c^" "$c" 2>/dev/null)" || return 1
  [ -n "$files" ] || return 0
  while IFS= read -r f; do
    case "$f" in docs/*.md|docs/*.txt|*.md|tools/lost_work_accepted.tsv) ;; *) return 1;; esac
  done <<< "$files"
}
CHAIN="$SHA"; cur="$SHA"; depth="${GATE_TEXT_DEPTH:-20}"
while [ "$depth" -gt 0 ] && git -C "$ROOT" cat-file -e "$cur^{commit}" 2>/dev/null && text_only "$cur"; do
  cur="$(git -C "$ROOT" rev-parse "$cur^")"; CHAIN="$CHAIN $cur"; depth=$((depth-1))
done
for w in identity.yml windows.yml quick.yml; do
  ids="$(gh run list -R "$REPO" -w "$w" -s success -L "$N" --json databaseId --jq '.[].databaseId' 2>"$TMP/err")" || { echo "release gate: could not list the runs of $w: $(cat "$TMP/err")"; exit 1; }
  found=""; seen=""; via=""
  for id in $ids; do
    rm -rf "$TMP/r$id"
    if gh run download "$id" -R "$REPO" -n tested-sha -D "$TMP/r$id" >/dev/null 2>&1 && [ -f "$TMP/r$id/tested-sha.txt" ]; then
      t="$(tr -d '[:space:]' < "$TMP/r$id/tested-sha.txt")"
      seen="$seen ${t:0:8}"
      if [ "$t" = "$SHA" ]; then found="$id"; break; fi
      if [ "$w" != quick.yml ] && case " $CHAIN " in *" $t "*) true;; *) false;; esac; then found="$id"; via="$t"; break; fi
    fi
  done
  if [ -n "$found" ] && [ -n "$via" ]; then echo "$w: run $found tested $via, an ancestor of $SHA with only text files (docs, *.md, lost_work_accepted.tsv) changed since"
  elif [ -n "$found" ]; then echo "$w: run $found tested $SHA"
  else echo "$w: no successful run tested $SHA (looked at ${N} newest successful runs; tested commits found:${seen:- none}). Run it on this tag: gh workflow run ${w} --ref <tag>, then re-run this workflow."; bad=1; fi
done
# Lost-work check (TOOLS-1300; since TOOLS-1305 it is no longer a job of the tag run in .github/workflows/release.yml, this is its only place): no unrecorded work left on a claude/* branch.
# Run it BEFORE pushing the tag (docs/internal/RELEASE_CHECKLIST.md), from a clone that has just done `git fetch --prune origin`: it also looks at your local
# branches, which CI does not see. In release.yml the build job has a shallow checkout without origin/experimental: there the check is skipped (the
# pre-tag run from a full clone did it).
# GATE_SKIP_LOST_WORK=1 is for tests/release_gate_test.sh only (it works only together with the test's fake gh, FAKE_DATA); release.yml never sets it.
if [ "${GATE_SKIP_LOST_WORK:-0}" = "1" ] && [ -n "${FAKE_DATA:-}" ]; then
  echo "lost-work check: SKIPPED (test use only)"
elif ! git -C "$ROOT" rev-parse -q --verify refs/remotes/origin/experimental >/dev/null 2>&1; then
  echo "lost-work check: skipped here, no origin/experimental in this checkout (shallow CI checkout: the check runs before the tag, TOOLS-1305; before tagging run this script from a full clone after 'git fetch --prune origin')"
elif out="$(cd "$ROOT" && python3 tools/lost_work_check.py --repo . --base origin/experimental --public --records archive/README.md --accept tools/lost_work_accepted.tsv 2>&1)"; then
  echo "lost-work check: ok"
else
  lwrc=$?
  [ "$lwrc" = 2 ] && { echo "lost-work check: ERROR (the check could not run, e.g. a git call failed; this is not a verdict on the branches):"; echo "$out" | tail -40; exit 1; }
  echo "lost-work check: FAILED (a claude/* branch has work that is not in origin/experimental and not accepted in tools/lost_work_accepted.tsv):"; echo "$out" | tail -40; bad=1
fi
exit $bad
