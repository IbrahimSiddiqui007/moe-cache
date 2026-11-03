## What and why

<!-- One or two plain sentences. -->

## Tests

<!-- For each: Passed / Failed / Not run (and why). The list is in docs/CODING_STANDARDS.md ("Tests"). -->

| Check | Result |
|---|---|
| `python3 -m py_compile bin/*.py tools/*.py tests/*.py planner/*.py` | |
| The model-free `tests/*_test.py` that `quick.yml` runs | |
| C++ tests (`args_test`, `tested_ggml_test`, `rebalance_test`), both header sets with `-Wall -Werror` | |
| `bash -n` on changed shell scripts | |
| `tests/identity.sh` on one MoE model (needed when `src/` changed) | |

## Checklist

- [ ] Plain words; only measured claims (anything else says "not measured" or "untested").
- [ ] A new switch is off by default until a paired run (A B B A) wins.
- [ ] Numbers changed: the claim is added in `benchmarks/claims`, the README table, `docs/benchmark_card.html` and the images are updated in the same step.
- [ ] Windows touched (launcher, paths, installers, `platform.h`): the Windows CI is green (it runs nightly and on demand: push the commit to the `windows-ci` branch, `git push origin <commit>:refs/heads/windows-ci`), or I say here that it was not run.
- [ ] Launch output changed on purpose: the golden file (`tests/launch_plan_golden*.json`) is re-recorded with `--record` and the commit says so.
- [ ] No private data in the change (prompts, paths, user or host names).
