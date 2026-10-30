// moe-cache: out-of-tree ggml backend (ACCEL device) that manages which MoE experts are resident in RAM. No llama.cpp patch.
// Expert tensors that llama.cpp keeps on the CPU (for example with --n-cpu-moe N) land in this backend's buffer, which is
// virtual memory only, at stock addresses. The cache decides which experts have physical pages: a miss is an O_DIRECT read
// from the GGUF, an eviction is madvise(DONTNEED). The unmodified CPU kernels then run, so output is token-identical.
// The backend claims exactly MUL_MAT_ID and GLU (plus trivial view ops). Batches of MOE_CACHE_GPU_MIN_TOKENS or more run on the GPU.
// Load: LD_PRELOAD + MOE_CACHE_PRELOAD=1 (builds that link backends directly), or GGML_BACKEND_PATH (loadable backends).
// Main env:
//   MOE_CACHE_SIZE_GIB=N|auto   MOE_CACHE_GGUF=model file   MOE_CACHE_PROFILE / MOE_CACHE_WARM=profile file
//   MOE_CACHE_RAM_GIB, MOE_CACHE_MARGIN_GIB   MOE_CACHE_REPACK=auto|on|off   MOE_CACHE_LAYERS=lo-hi
//   MOE_CACHE_STATS=1   MOE_CACHE_DISABLE=1 (load nothing)
// All variables are listed in the env parsing block of state_t (mc_state.inl). The code is split into parts, included at the end of this file.
//

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cmath>
#include <map>
#include <unordered_set>
#include "platform.h"
#include "mc_args.h"
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#include <filesystem>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <fstream>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#ifdef __linux__
#include <dlfcn.h>
#include <link.h>
#endif
// the plugin's own x86 kernels (Q2_0, IQ3_XXS gather) run in its executor on Linux and Windows; only the LD_PRELOAD symbol override is Linux-only
#if defined(__x86_64__) && (defined(__linux__) || defined(_WIN32))
#define MC_X86_KERNELS 1
#include "fast_q2_0.h"
#include "fast_iq3.h"
#endif
#ifdef __linux__
#include <sched.h>
#endif

// trace and profile files show what the user ran: create them readable by the owner only
static FILE * fopen_private(const char * path) {
#ifdef _WIN32
    return plat::fopen_utf8(path, "wb");
#else
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);   // never follows a symlink someone put there
    return fd < 0 ? nullptr : fdopen(fd, "wb");
#endif
}

// ggml versions CI runs this build against (comma-separated): API 2 = identity CI at 7fe450e19; API 3 = identity CI at 23b0202a1 (0.26.0)
// and Windows CI + the Windows test PC at release b11669 (0.27.0)
#ifndef MOE_CACHE_TESTED_GGML
#if GGML_BACKEND_API_VERSION >= 3
#define MOE_CACHE_TESTED_GGML "0.26.0, 0.27.0"
#else
#define MOE_CACHE_TESTED_GGML "0.25.1"
#endif
#endif

// true when v is one of the comma-separated versions in MOE_CACHE_TESTED_GGML
static bool moe_cache_tested_ggml(const char * v) {
    const size_t n = strlen(v);
    for (const char * p = MOE_CACHE_TESTED_GGML; *p; ) {
        while (*p == ' ' || *p == ',') {
            p++;
        }
        const char * e = p;
        while (*e && *e != ',') {
            e++;
        }
        if (n > 0 && (size_t)(e - p) == n && strncmp(p, v, n) == 0) {
            return true;
        }
        p = e;
    }
    return false;
}

// readable without loading the library (grep -a): lets the launcher pick the build that matches a llama-server's ggml
#define MC_STR_(x) #x
#define MC_STR(x) MC_STR_(x)
extern "C" __attribute__((used)) const char moe_cache_api_tag[] = "moe-cache-api=" MC_STR(GGML_BACKEND_API_VERSION) ";";

// The backend API of the ggml this server runs, by version number (-1: unknown version). A plugin built for another API has a different
// struct layout and would crash, so setup refuses it (the model then loads the normal way).
static int moe_cache_api_of_ggml(const char * v) {
    int maj = 0, min = 0;
    if (sscanf(v, "%d.%d", &maj, &min) != 2) return -1;
    if (maj == 0 && min <= 25) return 2;
    // 0.26 and newer 0.x: API 3, as the launcher says (bin/moe_cache.py ggml_api_of; ggml 0.28 did not bump it). A future API 4 is not guessed here: ggml's
    // loader compares reg.api_version for GGML_BACKEND_PATH, and ggml_struct_layout_mismatch() (start-up, also LD_PRELOAD) refuses a changed struct layout.
    if (maj == 0 && min >= 26) return 3;
    return -1;
}

#ifdef __linux__
#include <signal.h>
#endif
// The plugin is one translation unit split into parts (T9, 2026-10-15); each part is included here, in this order:
#include "mc_common.inl"
#include "mc_pool.inl"
#include "mc_state.inl"
#include "mc_buffer.inl"
#include "mc_backend.inl"
#include "mc_exec.inl"
#include "mc_hybrid.inl"
#include "mc_compute.inl"
#include "mc_device.inl"
#include "mc_interpose.inl"
