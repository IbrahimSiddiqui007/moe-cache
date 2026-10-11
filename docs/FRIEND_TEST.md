# Testing moe-cache on your machine (about 10 minutes)

You need: Linux, a `llama-server` built with shared ggml libraries (see `docs/BACKENDS.md`), `g++`, and optionally one small MoE `.gguf` (1-4 GB).

```bash
git clone https://github.com/IbrahimSiddiqui007/moe-cache && cd moe-cache
GGML_API=3 ./build.sh          # llama.cpp 23b0202a1 or newer; plain ./build.sh for 7fe450e19
tools/friend_test.sh --llama-server $(command -v llama-server) --model ~/models/some-small-moe.gguf
```

It prints what it is doing and writes `moe-cache-friend-report-DATE.txt` in the current folder. Please send that file.

What it checks: your CPU instruction sets, the GPU devices your llama-server can use, whether the optional AVX-512 kernels give bit-identical results and
how fast they are, and (with `--model`) whether outputs match stock llama.cpp, with and without llama.cpp's GPU expert cache.

It uploads nothing and replaces your home directory with `~` in the report.
