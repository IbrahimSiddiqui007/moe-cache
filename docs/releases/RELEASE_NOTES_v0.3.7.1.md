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

No new measurements: nothing that changes speed or output was touched. The v0.3.7 numbers in `docs/RELEASE_NOTES_v0.3.7.md` stand for this release. The only difference in `src/` from v0.3.7 is the plugin's start-up banner (`git diff v0.3.7 v0.3.7.1 -- src`), so the three v0.3.7 claims (`v037-conv-q2_0`, `v037-conv-iq3xxs`, `v037-fits-qwen36`) now say "valid up to 0.3.7.1" in `benchmarks/claims/`; the claims check (tags only) would otherwise call them stale because the version number moved.

## Update

Installed copies: `moe_cache.py update --apply` (it checks the attestation of the new installer the same way), or run the new `install.ps1` / `install.sh` from this release. If the v0.3.7 `install.ps1` stops with "github.com ... Stopped", run the `install.ps1` of this release instead (`-YesUnverified` does not help with the old one: its `gh auth status` call ends the script before that flag is read). The new one needs no flag. An already installed v0.3.7 on such a PC: `update --apply` in v0.3.7 has the old check (it refuses with a clear line; `--yes-unverified` goes on after you checked the file by hand with `gh attestation verify FILE -R IbrahimSiddiqui007/moe-cache`), or simply run the new `install.ps1`. Checked on the Windows PC (two saved gh accounts, one expired): the published v0.3.7.1 `install.ps1` verified the attestation and installed; the v0.3.7 installer with `-YesUnverified` still stopped.

Checked before publishing, from the draft's own files: sha256 and attestation of all files; Linux unpack, `check`, `pick`, control page, module imports, one small-model `check --run`; Windows zip in a new folder: sha256, attestation, `--version`, imports, `pick`, control page, small-model `check --run` with the kit's own llama.cpp. After publishing: `install.sh` fresh and `update --apply` from v0.3.7 on Linux, `install.ps1` fresh on Windows. Not done: `update --apply` from an installed v0.3.7 on Windows (the old installer cannot be installed on that PC; the update test runs in CI on a runner), and `moe-cache.exe --version` (it still opens the control page, as in v0.3.7).
