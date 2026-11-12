#!/bin/bash
# Publish a release draft after its own files were smoke-tested: tools/publish_release.sh vX.Y.Z
# The release workflow leaves a DRAFT (since v0.3.7); this checks that all seven files are attached, then makes it public: "latest" for a stable tag,
# a pre-release (never latest) for a tag with a suffix (-beta1, -rc1). Needs `gh` logged in with write access. Does not touch tags or branches.
set -euo pipefail
TAG="${1:-}"; REPO="${GITHUB_REPOSITORY:-IbrahimSiddiqui007/moe-cache}"
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]] || { echo "usage: tools/publish_release.sh vX.Y.Z"; exit 1; }
V="${TAG#v}"
[ "$(gh release view "$TAG" -R "$REPO" --json isDraft --jq .isDraft)" = true ] || { echo "$TAG is not a draft (already published?)"; exit 1; }
assets=$(gh release view "$TAG" -R "$REPO" --json assets --jq '.assets[].name')
for pat in "moe-cache-v$V-linux-x86_64.tar.gz" "moe-cache-v$V-ggml0.26-linux-x86_64.tar.gz" "SHA256SUMS" "install.sh" "install.ps1" "moe-cache-v$V-windows-x64.zip" "SHA256SUMS-windows"; do
  echo "$assets" | grep -Fxq "$pat" || { echo "FAIL: $pat is not attached to $TAG; not publishing"; exit 1; }
done
case "$V" in
  *-*) gh release edit "$TAG" -R "$REPO" --draft=false --prerelease --latest=false ;;
  *)   gh release edit "$TAG" -R "$REPO" --draft=false --prerelease=false --latest ;;
esac
gh release view "$TAG" -R "$REPO" --json isDraft,isPrerelease,name --jq '"published: draft=\(.isDraft) prerelease=\(.isPrerelease) title=\(.name)"'
echo "next: run the packages workflow by hand for $TAG (it does not run on a draft): gh workflow run packages.yml -f tag=$TAG"
