# moe-cache v0.3.7.1 (installer fix)

**What this release is.** One fix to the installers and nothing else. The program, the plugin and every number are the same as in v0.3.7 (the plugin banner and `--version` say 0.3.7.1).

## What was wrong

On a PC where `gh` (the GitHub command line tool) has more than one saved account and one of them has an expired token, `install.ps1` stopped after the sha256 check with only `moe-cache install: github.com ... Stopped`, although the active login worked. `install.sh` and `moe_cache.py update --apply` had the same gate and refused the same way (with a clearer line). The cause: they asked `gh auth status` first, and that command exits 1 when any saved account is bad; on Windows PowerShell 5.1 its first line of output also ended the script.

## What changed

| Where | Change |
|---|---|
| `install.ps1`, `install.sh`, `moe_cache.py update --apply` | The build attestation is checked by `gh attestation verify` alone; its exit code decides. No working login, GitHub unreachable and gh missing are still "could not be checked" (refused unless `-YesUnverified` / `--yes-unverified`); a check that ran and failed is still never overridden. The message names the reason and the way out. |
| Tests | The fake `gh` in the installer tests writes progress text to stderr like the real one, and a test has `gh auth status` failing while verify passes: the install must go on. |

Checked on the Windows PC (two saved gh accounts, one expired): the fixed `install.ps1` installed v0.3.7 with "build attestation verified".

## Measured

No new measurements: nothing that changes speed or output was touched. The v0.3.7 numbers in `docs/RELEASE_NOTES_v0.3.7.md` stand for this release.

## Update

Installed copies: `moe_cache.py update --apply` (it checks the attestation of the new installer the same way), or run the new `install.ps1` / `install.sh` from this release. If an older installer stops with "github.com ... Stopped", run the new one, or install with `-YesUnverified` after checking the file by hand: `gh attestation verify FILE -R IbrahimSiddiqui007/moe-cache`.
