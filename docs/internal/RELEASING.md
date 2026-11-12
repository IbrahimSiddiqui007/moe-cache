# Release checklist

Only with the owner's go and version name (D1). Nothing here is done by CI on its own.

| # | Step | How |
|---|---|---|
| 1 | Version name | first write the release notes (step 2) and put the changes under `## Unreleased` in CHANGELOG.md. Then `tools/set_version.sh NAME`: it turns that heading into `## NAME (date)` and moves the version in the README badge and "Known issues" heading, install.sh MOE_CACHE_REF, the plugin banner and bin/moe_cache.py. It stops and changes nothing when a piece is missing (no `## Unreleased`, no release notes file). Nothing has to be re-seeded after a release (the old `0.0.0.TBD` placeholder flow still works if the files hold it). Review the diff, also the prose in README.md and docs/ROADMAP.md that names a version, commit. `tests/set_version_test.sh` covers the script |
| 2 | Release notes | `docs/releases/RELEASE_NOTES_vNAME.md` is the release text (the release workflow uses it); it must exist before step 1 |
| 3 | CI on the release commit | quick (runs on every push and pull request), identity (`gh workflow run identity.yml --ref BRANCH`) and windows (`gh workflow run windows.yml --ref BRANCH`) green on that exact commit. The nightly identity / windows runs also cover `experimental` (a second cron; the workflow file must be on main for the schedule to run) |
| 4 | Tag (the owner pushes it) | `git tag vNAME && git push origin vNAME`: the release workflow checks install.sh against the tag, builds the Linux tarballs and the Windows kit, attests them, and publishes only after identity and windows passed on the tagged commit |
| 5 | main follows the tag | only after the release is published and for a stable tag: fast-forward `main` to the **tagged commit** (details and check: RELEASE_CHECKLIST.md section 4, "`main` follows the tag"); `main` is never merged before the tag |
| 6 | Check the release | download the Windows zip, compare `Get-FileHash` with `SHA256SUMS-windows`; `gh attestation verify FILE -R IbrahimSiddiqui007/moe-cache` |
| 7 | **Submit scoop / winget manifests (owner's OK)** | filled manifests are in the release run's `moe-cache-windows-kit` artifact under `packaging/` (or run `python3 packaging/fill_manifests.py dist/moe-cache-vNAME-windows-x64.zip`). winget: check with `winget validate --manifest DIR` on Windows, then a pull request to microsoft/winget-pkgs (`manifests/i/IbrahimSiddiqui007/MoeCache/NAME/`). scoop: the JSON goes into a bucket (our own `scoop-bucket` repository or a PR to an existing one) |
| 8 | Announce | the owner writes posts and issue replies |

The kit is not code-signed (owner, 2026-10-28): the README, docs/WINDOWS.md, README_WINDOWS.txt and the release notes say how to unblock it and check its hash.
