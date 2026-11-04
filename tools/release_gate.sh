#!/bin/bash
# Release gate (RELEASE-687): identity, windows and quick must each have a successful run that TESTED the commit being released.
#   tools/release_gate.sh COMMIT [REPO]        (needs gh and a token that can read Actions runs and artifacts)
# A run's own head SHA is not enough: a run started on one branch can test another ref (the `ref` input of windows.yml / identity.yml). Every one
# of those workflows therefore ends with a `record` job (only after all its jobs passed, on the same commit) that uploads the artifact `tested-sha`
# holding `git rev-parse HEAD` of what it tested. This script looks at the newest successful runs of each workflow and accepts the first whose
# tested-sha equals COMMIT. Runs older than the change (no artifact) never count. GATE_RUNS = how many runs per workflow to look at (default 30).
set -u
SHA="${1:?commit sha}"; REPO="${2:-${GITHUB_REPOSITORY:-}}"; N="${GATE_RUNS:-30}"
echo "$SHA" | grep -Eq '^[0-9a-f]{40}$' || { echo "release gate: '$SHA' is not a full commit sha"; exit 1; }
[ -n "$REPO" ] || { echo "release gate: no repository (pass it as the 2nd argument or set GITHUB_REPOSITORY)"; exit 1; }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
bad=0
for w in identity.yml windows.yml quick.yml; do
  ids="$(gh run list -R "$REPO" -w "$w" -s success -L "$N" --json databaseId --jq '.[].databaseId' 2>"$TMP/err")" || { echo "release gate: could not list the runs of $w: $(cat "$TMP/err")"; exit 1; }
  found=""; seen=""
  for id in $ids; do
    rm -rf "$TMP/r$id"
    if gh run download "$id" -R "$REPO" -n tested-sha -D "$TMP/r$id" >/dev/null 2>&1 && [ -f "$TMP/r$id/tested-sha.txt" ]; then
      t="$(tr -d '[:space:]' < "$TMP/r$id/tested-sha.txt")"
      seen="$seen ${t:0:8}"
      if [ "$t" = "$SHA" ]; then found="$id"; break; fi
    fi
  done
  if [ -n "$found" ]; then echo "$w: run $found tested $SHA"
  else echo "$w: no successful run tested $SHA (looked at ${N} newest successful runs; tested commits found:${seen:- none}). Run it on this tag: gh workflow run ${w} --ref <tag>, then re-run this workflow."; bad=1; fi
done
exit $bad
