# Contributing

Thank you for looking. This project is small; the rules below keep it trustworthy.

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
- `bash -n` on every script, `python3 -m py_compile` on the python files (CI does this).
- Kernel changes: `g++ -O2 -std=c++17 -Isrc tests/kernel_check.cpp -o /tmp/kernel_check -ldl && /tmp/kernel_check [libggml-cpu.so]` must report 0 mismatches.
- Memory safety: build the plugin with `-fsanitize=address,undefined` and run an identity test with a tiny cache (`--ram 1`) to force evictions.
- Memory limits: `tests/limit.sh MODEL.gguf --limit GB` must PASS.

## Measuring
Use the benchmark tool ([moe-cache-benchmark](https://github.com/IbrahimSiddiqui007/moe-cache-benchmark): `bin/moe-cache-bench` or `bench/standard.py`); runs that were not clean (hot CPU, background load) are flagged and must be repeated, not quoted. Differences below 10 % need paired runs in both orders. See docs/TEST_PROTOCOL.md.

## Workflow
Work happens on the `experimental` branch and is promoted to `main` when identity and the checks above pass. Releases are tags (`vX.Y.Z`) on `main`; GitHub Actions builds and attests them.

## Style
Match the surrounding code. No new dependencies in the plugin beyond ggml and the C++ standard library. Comments say what is not obvious.
