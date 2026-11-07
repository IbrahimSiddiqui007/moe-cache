#!/bin/bash
# GitHub Actions caches that belong to a closed PR or to a branch that no longer exists, and (with --delete) deletion of exactly those.
#   tools/cache_cleanup.sh            list every cache with its verdict; nothing is deleted
#   tools/cache_cleanup.sh --delete   delete the ones marked DEAD
# DEAD  = ref refs/pull/N/merge of a PR that is not open, or refs/heads/B where B is gone from the remote.
# UNKNOWN = gh / git could not answer (network, rate limit): never deleted, run again.
# KEEP  = open PR, existing branch (also a merged one that still exists: delete the branch first, then run this), main, experimental.
# Needs gh (logged in) and the repo's remote as origin. Run it after merging and deleting a work branch (docs/internal/PC_CI.md).
set -uo pipefail
cd "$(dirname "$0")/.."
DEL=0; [ "${1:-}" = "--delete" ] && DEL=1
REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner) || exit 1
git fetch -q --prune origin 2>/dev/null
total=0; dead=0; deadbytes=0
while IFS=$'\t' read -r id ref key bytes; do
  total=$((total + bytes)); verdict=KEEP
  case "$ref" in
    refs/pull/*/merge) n=${ref#refs/pull/}; n=${n%/merge}; st=$(gh pr view "$n" --repo "$REPO" --json state --jq .state 2>/dev/null)
      case "$st" in MERGED|CLOSED) verdict=DEAD ;; OPEN) ;; *) verdict=UNKNOWN ;; esac ;;       # a failed gh call must never mean DEAD
    refs/heads/*) b=${ref#refs/heads/}; git ls-remote --exit-code --heads origin "$b" >/dev/null 2>&1; rc=$?
      [ $rc = 0 ] || { [ $rc = 2 ] && verdict=DEAD || verdict=UNKNOWN; } ;;                      # 2 = the branch is not there; any other code (network) = unknown
  esac
  printf '%s\t%s\t%s\t%s MiB\t%s\n' "$verdict" "$id" "$ref" $((bytes / 1048576)) "$key"
  if [ $verdict = DEAD ]; then dead=$((dead + 1)); deadbytes=$((deadbytes + bytes)); [ $DEL = 1 ] && gh cache delete "$id" --repo "$REPO" >/dev/null; fi
done < <(gh cache list --repo "$REPO" --limit 200 --json id,ref,key,sizeInBytes --jq '.[] | [.id, .ref, .key, .sizeInBytes] | @tsv')
echo "total $((total / 1048576)) MiB of 10240 MiB; dead: $dead cache(s), $((deadbytes / 1048576)) MiB$([ $DEL = 1 ] && echo ' (deleted)' || echo ' (run with --delete to remove)')"
