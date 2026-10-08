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
