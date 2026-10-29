# Release checklist

Only with the owner's go and version name (D1). Nothing here is done by CI on its own.

| # | Step | How |
|---|---|---|
| 1 | Version name | `tools/set_version.sh NAME` (CHANGELOG heading and date, README badge, ROADMAP, SECURITY.md, install.sh MOE_CACHE_REF, plugin banner, release notes file); review `git diff`, commit |
| 2 | Release notes | `docs/RELEASE_NOTES_vNAME.md` is the release text (the release workflow uses it) |
| 3 | CI on the release commit | identity (`gh workflow run identity.yml --ref BRANCH`) and windows (`gh workflow run windows.yml --ref BRANCH`) green on that exact commit |
| 4 | main | merge the release branch into main (owner's OK) |
| 5 | Tag | `git tag vNAME && git push origin vNAME`: the release workflow checks install.sh against the tag, builds the Linux tarballs and the Windows kit, attests them, and publishes only after identity and windows passed on the tagged commit |
| 6 | Check the release | download the Windows zip, compare `Get-FileHash` with `SHA256SUMS-windows`; `gh attestation verify FILE -R IbrahimSiddiqui007/moe-cache` |
| 7 | **Submit scoop / winget manifests (owner's OK)** | filled manifests are in the release run's `moe-cache-windows-kit` artifact under `packaging/` (or run `python3 packaging/fill_manifests.py dist/moe-cache-vNAME-windows-x64.zip`). winget: check with `winget validate --manifest DIR` on Windows, then a pull request to microsoft/winget-pkgs (`manifests/i/IbrahimSiddiqui007/MoeCache/NAME/`). scoop: the JSON goes into a bucket (our own `scoop-bucket` repository or a PR to an existing one) |
| 8 | Announce | the owner writes posts and issue replies |

The kit is not code-signed (owner, 2026-10-28): the README, docs/WINDOWS.md, README_WINDOWS.txt and the release notes say how to unblock it and check its hash.
