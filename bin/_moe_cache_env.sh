# sourced by the moe-cache scripts. moe_cache_load_env LLAMA_SERVER PLUGIN_LIB [auto|preload|path]
# Sets MOE_CACHE_LOAD (environment assignments that make that llama-server load the plugin) and MOE_CACHE_METHOD.
#  preload: llama-server links its backends directly (typical for distro packages): LD_PRELOAD
#  path   : llama-server loads its backends as separate library files (official releases): GGML_BACKEND_PATH
moe_cache_load_env() {
  local lsp method="${3:-auto}"
  lsp="$(command -v "$1" 2>/dev/null || echo "$1")"
  if ! ldd "$lsp" 2>/dev/null | grep -q "libggml-base"; then
    echo "warning: $lsp does not use shared ggml libraries, the plugin cannot be loaded into it (build llama.cpp with BUILD_SHARED_LIBS=ON)" >&2
  fi
  if [ "$method" = auto ]; then
    if ldd "$lsp" 2>/dev/null | grep -q "libggml-cpu"; then method=preload; else method=path; fi
  fi
  case "$method" in
    preload) MOE_CACHE_LOAD=(LD_PRELOAD="$2" MOE_CACHE_PRELOAD=1);;
    path)    MOE_CACHE_LOAD=(GGML_BACKEND_PATH="$2");;
    *) echo "unknown load method: $method (use auto, preload or path)" >&2; return 1;;
  esac
  MOE_CACHE_METHOD="$method"
}

# API of the ggml a llama-server uses, from the version of the libggml-base it loads (0.24-0.25 -> backend API 2, 0.26-0.27 -> API 3). Empty when unknown.
moe_cache_ls_api() {
  local lsp base v
  lsp="$(command -v "$1" 2>/dev/null || echo "$1")"
  base="$(ldd "$lsp" 2>/dev/null | awk '/libggml-base/ {print $3; exit}')"
  [ -n "$base" ] || return 0
  v="$(basename "$(readlink -f "$base")" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  case "$v" in 0.24.*|0.25.*) echo 2;; 0.26.*|0.27.*) echo 3;; esac
}
# API a plugin build was made for (the tag is a plain string inside the library); empty for builds from before 0.2.3
moe_cache_lib_api() { grep -a -o 'moe-cache-api=[0-9]*' "$1" 2>/dev/null | head -1 | cut -d= -f2; }
# moe_cache_pick_lib LLAMA_SERVER HERE: the plugin build in HERE/build*/ that matches the server's ggml, else the default build
moe_cache_pick_lib() {
  local want c
  want="$(moe_cache_ls_api "$1")"
  if [ -n "$want" ]; then
    for c in "$2/build/libggml-moe-cache.so" "$2"/build-*/libggml-moe-cache.so; do
      [ -f "$c" ] && [ "$(moe_cache_lib_api "$c")" = "$want" ] && { echo "$c"; return 0; }
    done
  fi
  echo "$2/build/libggml-moe-cache.so"
}
