# release-prep 2026-11-14 (worker, branch claude/pretag-038, worktree /home/ibrum/Dev/StrataTest/wt-pretag)

## Part 1 (owner)
Branch head 5142625 (pushed; version 0.3.8 set). quick 43620465682 + windows 43620468957 GREEN on it. Critic (notes/critic_pretag038_3b7d2d8.md) fixes applied: Windows kit icon moved to packaging/moe-cache.ico, manifests reworded, +77 % percentages dropped. Still open: findings 3 and 5 (owner). Older status below is superseded.

OLD: Branch head 3d782bf. Done: archive tag, GUI removals, release-note blanks, GUI-1155 picker fix. CI (quick 43617793235, windows 43617796523, dispatched by hand on 3d782bf) running; critic review requested (local_082f5d5a-...). Not done: critic fixes, tools/set_version.sh 0.3.8, final CI. No tag, no release.yml, no merge (owner).

## Part 2 (detail)
| Step | State |
|---|---|
| 1 archive tag `archive/pretag-038-with-guis` (laptop only, = b0edc24) | done. `git push` prints a harmless "pack-refs failed" from the lost-work hook; the tag is intact |
| 2 removals (3b7d2d8): desktop/, bin/moe_cache_top.py, build_top.sh, src/top, third_party/ftxui, desktop.yml, top/--top subcommands, tests/top_test.py (kept parts -> tests/mini_hw_test.py), ui-top.json handshake. KEPT bin/moe_cache_monitor.py (control page needs it, main confirmed) | done |
| 3 RN blanks (3b7d2d8): 0 left; rn_blanks_check + rn_blanks_test pass | done; sources below |
| GUI-1155 (3d782bf): picker says "Nothing fits" under 11 GB server memory (bin/moe_cache_pick.py NOTHING_FITS_BELOW_CAP_GB, launch.py, ui/index.html), speeds labelled "our test PC", tests in tests/picker_test.py | done, picker/launch/ui tests pass locally |
| 4 CI on branch: runs are workflow_dispatch only (push does not trigger) | quick + windows running; kit (release.yml dry run) NOT dispatched: brief says do not dispatch release.yml, asked main |
| 5 critic | asked, no answer yet |
| 6 set_version 0.3.8 + push + CI | not started |

## Sources of the filled numbers
- long-prompt 187.4 -> 331.8 tok/s (+77.0 %), 19.01 -> 19.35 (+1.8 %), 42 -> 45 layers: handover/plans/overnight/2026-11-07/PLAN.md item 05 (layers: old README text)
- KL 0.0017 / same top 97.78 % (27B, PC, 3x2048): release notes row "KV cache ladder", also CURRENT_STATE.md line 37
- mmap spike 27.1 GB, none +2-6 GB; commit estimate 13.5-20.0 GB planned vs 11.9-20.7 real: ledgers/LEDGER_TASKS.md TEST-1030 / LAUNCHER-1065 (sessions/chain_2026-11-09/files/slot_2026_11_10_results.md)
- RAM 6000 MT/s EXPO since 2026-11-07: docs/RESULTS_CURRENT.md
- Cut (no claim): Nemotron 3.5 Super, quality gate, Q2_0 vs IQ3_S top-1, every stock comparison

## Open for main/owner
1. `python3 bin/moe_cache.py bench-record lint` fails: all v038-* claims stale vs b12207 (README, docs/RESULTS_CURRENT.md, docs/benchmark_card.html, which still carry the 11-08 stock comparisons). Tag gate needs an owner waiver or those files cut.
2. Dispatch the kit CI (release.yml dry run)?
3. Laptop stock-vs-plugin lines (README 22/131/143, notes) left: not blanks, a negative result.
