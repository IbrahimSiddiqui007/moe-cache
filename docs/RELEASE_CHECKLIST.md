# Release checklist and gates

For whoever cuts a release. Checked against `.github/workflows/*.yml`, `tools/set_version.sh` and `install.sh` on 2026-11-03.

## 1. What runs automatically

| Workflow | When it runs | What it proves |
|---|---|---|
| `quick` | every push to `experimental` or `main`, every pull request, by hand | the model-free tests: `tools/run_quick_tests.sh` (compile check and all quick tests) |
| `identity` | daily 05:00 UTC (`main`) and 06:30 UTC (`experimental`), by hand | stock and plugin write the same tokens on a small test model (Granite 3.1 1B-A400M), at each supported llama.cpp pin, plus the mutation checks that prove the test can fail |
| `windows` | daily 03:30 UTC (`main`) and 04:30 UTC (`experimental`), by hand, push to branch `windows-ci` | the same identity test on a Windows runner, the DLL, the launcher and the kit |
| `sanitize` | Mondays 06:00 UTC, by hand | ThreadSanitizer and AddressSanitizer builds on the hardest identity setups (forced eviction, delayed reads); a report that names plugin code fails it |
| `release` | a pushed tag `v*`, or by hand (build only, nothing published) | builds the Linux tarballs and the Windows kit, attests them, creates the release |
| `packages` | after `release` finishes, by hand | fills the scoop and winget manifests from the published Windows zip after its sha256 and attestation are checked (pre-releases are skipped for winget unless a repository variable says otherwise) |

The release workflow already has its own gates:

| Gate | Where | What happens when it fails |
|---|---|---|
| `identity`, `windows` and `quick` each have a successful run that **tested** the tagged commit (RELEASE-687: each ends with a `record` job that uploads the artifact `tested-sha` = `git rev-parse HEAD` of what it tested; `tools/release_gate.sh` compares it with the tag's commit, the run's own head sha is not used) | first step of the `build` job (tags only) | the build stops; run the missing workflow on the tag (`gh workflow run NAME.yml --ref vX.Y.Z`), then re-run `release`. A run from before this change has no artifact and does not count |
| install.sh names the tag; install.ps1 gets the version; the package carries the official llama.cpp; no newer glibc than 2.34 | `build` job | the build stops |
| both the Linux and the Windows files are attached | `publish` job | the release stays a **draft** that nobody sees |
| a tag with a suffix (`-beta1`, `-rc1`) | `publish` job | published as a pre-release and never as "latest" |

The release text is `docs/RELEASE_NOTES_<tag>.md` when that file exists, else a short default text plus GitHub's generated notes. `tools/set_version.sh` needs the notes file to exist already.

## 2. Before you tag

Do these in order. Every command runs from the repository root.

| # | Step | Command or check |
|---|---|---|
| 1 | Merge the latest `origin/experimental`; quick tests green locally | `nice -n 19 tools/run_quick_tests.sh` |
| 2 | Write `docs/RELEASE_NOTES_vX.Y.Z.md` and the `## Unreleased` part of `CHANGELOG.md` | numbers only from `docs/RESULTS.md` and `docs/RESULTS_OLD_LAPTOP.md`; plain words. After promoting to `main`, open the card footer: its links to `RESULTS.md` and `RESULTS_OLD_LAPTOP.md` (blob/main) resolve only once `main` has the files |
| 3 | Put the version into every place that names it | `tools/set_version.sh X.Y.Z`, then read `git diff` (it changes the changelog heading, the README badge and "Known issues" heading, `install.sh` `MOE_CACHE_REF`, the plugin banner and `VERSION` in `bin/moe_cache.py`) |
| 4 | The docs agree with the code | `python3 tools/env_names_check.py` (every `MOE_CACHE_` name the code reads is in the docs); the pinned llama.cpp build is the same in `tools/make_windows_kit.sh`, `tools/make_release.sh`, `README.md`, `docs/INSTALL.md` |
| 5 | No competitor name in any public file | `git grep -i -n -E 'NAME1\|NAME2' -- . ':!archive'` with the names from the owner's list (the list itself is not kept in the repository); the result must be empty |
| 6 | Push `experimental`, wait for `quick`; run `identity` and `windows` on the commit that will be tagged | `gh workflow run identity.yml --ref experimental` and `gh workflow run windows.yml --ref experimental`; they must be green on **that** commit |
| 7 | Tag a pre-release first when the release workflow or an installer changed | `git tag vX.Y.Z-beta1 && git push origin vX.Y.Z-beta1`: it publishes as a pre-release, never "latest". Install from it (step 9), then tag the real version |
| 8 | Tag the release | `git tag vX.Y.Z && git push origin vX.Y.Z`; watch `release` until the `publish` job is green |

## 3. Smoke test of the packaged files (on the main user path)

Do this on the files from the release page, not on the checkout. A release once shipped without `bin/moe_cache_pick.py` because only the checkout was tested.

| # | Where | Check |
|---|---|---|
| 9 | Linux | take the tarball (or run the one-line `install.sh`): `moe-cache --version` names the new version; `moe-cache launch` reaches the model suggestion; `bin/moe-cache-check MODEL.gguf --run` says "cache runs: yes" |
| 10 | Linux | `gh attestation verify FILE -R IbrahimSiddiqui007/moe-cache` for the tarball, `install.sh` and `install.ps1`; the `sha256sum -c SHA256SUMS` line matches |
| 11 | Windows | unzip the kit (Properties > Unblock first), double-click `moe-cache.bat`: the control page opens; import a model; press Start; send one chat message |
| 12 | Windows | `python bin\moe_cache.py pick` runs; `python bin\moe_cache.py check D:\models\model.gguf --run` passes |
| 13 | both | every `bin/*.py` of the unpacked kit or tarball imports from there alone (`tools/make_windows_kit.sh` checks the kit's launcher modules itself) |
| 14 | Linux, Windows | on a copy installed by `install.sh` / `install.ps1` from the **previous release that has an installer** (`install.sh` / `install.ps1`; v0.3.5 had none, so v0.3.6 was tested from its beta2): `moe-cache update --apply --dry-run` names the new tag; after `update --apply --yes` the new version runs, the old one is still there (Linux: `versions/`, roll back with `ln -sfn versions/<old folder> <prefix>/current`; Windows: the folder `kit.prev`) |

README "Known issues" must name anything from steps 9-14 that was **not** done (for example "a full model run from the final zip was not tested").

## 4. After publishing

| Step | Check |
|---|---|
| The release page | pre-release or latest as intended; the six file names the `publish` job checks are all there |
| `packages` | ran for a stable tag, did not run for a suffix tag |
| The one-line installs | `releases/latest/download/install.sh` and `install.ps1` are the new versions |
| Ledger and README | the "Known issues in vX.Y.Z" heading and the benchmark lines name the released version |

## 5. Cadence

Small releases are safer than big ones, because each one has fewer changes to blame. Rule of thumb for the owner to confirm: a stable release when the daily `identity` and `windows` runs on `experimental` have been green for a few days and the changelog has something users notice; a `-beta` tag first whenever `release.yml`, `install.sh`, `install.ps1` or the kit script changed.

### Ideas for the release workflow (not applied; no change to `release.yml` before v0.3.6 ships)

| Idea | Why it is not in yet |
|---|---|
| Build the release text from `CHANGELOG.md` instead of a hand-written notes file | the notes file carries measured tables that the changelog does not; a generated text would need a check that every number comes from `docs/RESULTS.md` |
| Run the `update --apply` test (step 14) in the workflow against the draft release | needs the draft's files to be reachable without a login; try it on a `-beta` tag first |
| A scheduled run of step 9 on the newest published release (install, `--version`, `check --run` on the small test model) | nightly cost is small, but it needs a runner that can fetch the test model; today `identity` only tests the checkout |
