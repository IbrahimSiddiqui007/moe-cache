# Contributing

Thank you for looking. This project is small; the rules below keep it trustworthy. The full standards (code, tests, speed work, commits, text) are in [docs/internal/CODING_STANDARDS.md](docs/internal/CODING_STANDARDS.md).

## The one rule
**moe-cache must not change the generated tokens.** Any change that touches the plugin has to pass the token-identity test on at least one MoE model before it is merged:

```bash
./build.sh
tests/identity.sh MODEL.gguf --llama-server /path/to/llama-server       # PASS = same tokens as stock
tests/identity.sh MODEL.gguf --llama-server /path/to/llama-server --long
MOE_CACHE_EXEC_CHECK=1 MOE_CACHE_STATS=1 tests/identity.sh MODEL.gguf ...   # executor: "mismatches 0"
```

Models bigger than your RAM: add `--limit GB` (hard cgroup, swap off). Never run such a model without a limit.

## Other checks
- `tools/run_quick_tests.sh`: the whole model-free set (py_compile on the python files and every `tests/*_test.py` / `apply_llama_patches_test.sh`, with fake servers, no model, a few minutes). The list of tests lives only in that script; quick.yml calls it on every push and pull request. A new `tests/*_test.py` goes into its `PY_TESTS` (the script fails on a test file it does not know). With `REQUIRE_NODE=1` (CI sets it) a missing `node` fails instead of silently skipping the JavaScript checks in `ui_test.py`.
- `bash -n` on every script (CI does this).
- Kernel changes: `g++ -O2 -std=c++17 -Isrc tests/kernel_check.cpp -o /tmp/kernel_check -ldl && /tmp/kernel_check [libggml-cpu.so]` must report 0 mismatches.
- Memory safety: build the plugin with `-fsanitize=address,undefined` and run an identity test with a tiny cache (`--ram 1`) to force evictions.
- Memory limits: `tests/limit.sh MODEL.gguf --limit GB` must PASS.

## Measuring
Use `python3 bin/moe_cache.py bench-report MODEL.gguf` (or **Share my numbers** on the control page; `--chat` runs a multi-turn conversation, see docs/SHARE_YOUR_NUMBERS.md); runs that were not clean (hot CPU, background load) are flagged and must be repeated, not quoted. Differences below 10 % need paired runs in both orders. See docs/TEST_PROTOCOL.md.

## Workflow
Work happens on the `experimental` branch and is promoted to `main` when identity and the checks above pass. Releases are tags (`vX.Y.Z`) on the release commit of `experimental`; GitHub Actions builds and attests them, and once the release is published `main` is fast-forwarded to the tagged commit (docs/internal/RELEASE_CHECKLIST.md, section 4).

## Style
Match the surrounding code. No new dependencies in the plugin beyond ggml and the C++ standard library. Comments say what is not obvious.
