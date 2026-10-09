#!/bin/bash
# moe-cache setup for a fresh Ubuntu / Debian machine: installs the build tools, builds the matching llama.cpp (CPU, optionally Vulkan)
# and the moe-cache plugin, creates a Python environment for the planner and prints the commands to run a model.
#   ./install.sh [--vulkan] [--prefix DIR] [--jobs N] [--no-apt]
#     --vulkan   also build the Vulkan backend (use an AMD / Intel / NVIDIA GPU or integrated GPU)
#     --prefix   where everything goes (default: ~/moe-cache-setup)
#     --no-apt   do not install packages (you already have cmake, g++, git, python3-venv)
# Takes about 10-20 minutes, almost all of it compiling llama.cpp. Needs about 3 GB of disk. Run it as a normal user (it asks for sudo only for apt).
set -e
LLAMA_SHA=7fe450e19305b828c199d602c23a8337aaa1f03b   # the llama.cpp commit moe-cache was built and tested against (ggml 0.25.1)
MOE_CACHE_REF=v0.1.2                                  # tag of moe-cache used when this script is run without a moe-cache checkout
GGUF_PY_VERSION=0.19.0                                # python 'gguf' package version the planner was tested with
VULKAN=0; PREFIX="$HOME/moe-cache-setup"; JOBS=$(nproc); APT=1
while [ $# -gt 0 ]; do case "$1" in --vulkan) VULKAN=1; shift;; --prefix) PREFIX=$2; shift 2;; --jobs) JOBS=$2; shift 2;; --no-apt) APT=0; shift;;
  -h|--help) sed -n 2,9p "$0"; exit 0;; *) echo "unknown option: $1"; exit 1;; esac; done
say() { printf '\n== %s\n' "$*"; }
[ "$(uname -s)" = Linux ] || { echo "Linux only"; exit 1; }
[ "$(id -u)" != 0 ] || { echo "run as a normal user, not root"; exit 1; }
mkdir -p "$PREFIX"

if [ $APT = 1 ]; then
  say "1/5 packages (sudo needed)"
  command -v apt-get >/dev/null || { echo "apt-get not found: install cmake, g++, git, python3-venv yourself and use --no-apt"; exit 1; }
  PK="build-essential cmake git curl ca-certificates python3 python3-venv python3-pip ninja-build"
  if [ $VULKAN = 1 ]; then PK="$PK libvulkan-dev mesa-vulkan-drivers vulkan-tools spirv-headers"; fi
  sudo apt-get update -y
  sudo apt-get install -y $PK
  if [ $VULKAN = 1 ]; then sudo apt-get install -y glslc || sudo apt-get install -y glslang-tools shaderc || true; fi
else
  say "1/5 packages: skipped (--no-apt)"
fi
for c in cmake g++ git python3; do command -v $c >/dev/null || { echo "missing: $c"; exit 1; }; done

say "2/5 llama.cpp at the tested commit"
L="$PREFIX/llama.cpp"
if [ ! -d "$L/.git" ]; then mkdir -p "$L"; git -C "$L" init -q; git -C "$L" remote add origin https://github.com/ggml-org/llama.cpp.git; fi
git -C "$L" fetch -q --depth 1 origin "$LLAMA_SHA"
git -C "$L" checkout -q FETCH_HEAD

say "3/5 building llama.cpp ($JOBS jobs; this is the long step)"
FLAGS=(-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)
[ $VULKAN = 1 ] && FLAGS+=(-DGGML_VULKAN=ON)
GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
cmake -S "$L" -B "$L/build" "${GEN[@]}" "${FLAGS[@]}" >/dev/null
cmake --build "$L/build" --target llama-server llama-bench llama-perplexity -j "$JOBS"
BIN="$L/build/bin"; [ -x "$BIN/llama-server" ] || { echo "llama-server was not built"; exit 1; }

say "4/5 moe-cache plugin"
if [ -f "$(dirname "$0")/build.sh" ] && [ -d "$(dirname "$0")/src" ]; then M="$(cd "$(dirname "$0")" && pwd)"; else
  M="$PREFIX/moe-cache"
  if [ -d "$M/.git" ]; then git -C "$M" fetch -q --tags origin && git -C "$M" checkout -q "$MOE_CACHE_REF"
  else git clone -q --branch "$MOE_CACHE_REF" https://github.com/IbrahimSiddiqui007/moe-cache "$M"; fi
fi
GGML_LIB_DIR="$BIN" "$M/build.sh"

say "5/5 python environment for the planner"
python3 -m venv "$PREFIX/venv" && "$PREFIX/venv/bin/pip" install -q "gguf==$GGUF_PY_VERSION"
PY="$PREFIX/venv/bin/python3"

{
  echo "# source this file:  . $(printf '%q' "$PREFIX/env.sh")"
  printf 'export LLAMA_SERVER=%q\n' "$BIN/llama-server"
  printf 'export MOE_CACHE_PYTHON=%q\n' "$PY"
  printf 'export PATH=%q:"$PATH"\n' "$M/bin"
} > "$PREFIX/env.sh"
{
  echo "# fish:  source $PREFIX/env.fish"
  printf 'set -gx LLAMA_SERVER %q\n' "$BIN/llama-server"
  printf 'set -gx MOE_CACHE_PYTHON %q\n' "$PY"
  printf 'set -gx PATH %q $PATH\n' "$M/bin"
} > "$PREFIX/env.fish"
say "setup check"
LLAMA_SERVER="$BIN/llama-server" MOE_CACHE_PYTHON="$PY" "$M/bin/moe-cache-check" --llama-server "$BIN/llama-server" || true
cat <<EOT

Done. Next steps (bash or zsh; for fish use: source $PREFIX/env.fish):
  . $PREFIX/env.sh
  # get any MoE model in GGUF format, for example (12 GB):
  #   curl -L -o gpt-oss-20b.gguf https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf
  moe-cache-check gpt-oss-20b.gguf
  $M/tests/identity.sh gpt-oss-20b.gguf --llama-server $BIN/llama-server     # PASS = same tokens as stock
  moe-cache-server gpt-oss-20b.gguf --ram 20 --open                          # chat UI at http://localhost:8080
  $M/bin/moe-cache-bench                                                      # the benchmark GUI
EOT
