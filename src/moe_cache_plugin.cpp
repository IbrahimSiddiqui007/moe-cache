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
// All variables are listed in the env parsing block of state_t below.
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
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#include <filesystem>
#include <condition_variable>
#include <deque>
#include <fstream>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#ifdef __linux__
#include <dlfcn.h>
#include <link.h>
#endif
#if defined(__x86_64__) && defined(__linux__)
#include "fast_q2_0.h"
#include "fast_iq3.h"
#endif
#ifdef __linux__
#include <sched.h>
#endif

// trace and profile files show what the user ran: create them readable by the owner only
static FILE * fopen_private(const char * path) {
#ifdef _WIN32
    return fopen(path, "wb");
#else
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);   // never follows a symlink someone put there
    return fd < 0 ? nullptr : fdopen(fd, "wb");
#endif
}

#ifndef MOE_CACHE_TESTED_GGML
#if GGML_BACKEND_API_VERSION >= 3
#define MOE_CACHE_TESTED_GGML "0.26.0"
#else
#define MOE_CACHE_TESTED_GGML "0.25.1"
#endif
#endif

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
    if (maj == 0 && min == 26) return 3;
    return -1;
}

#ifdef __linux__
#include <signal.h>
#endif
namespace {

#ifdef __linux__
int g_stats_pipe[2] = { -1, -1 };
std::atomic<bool> g_stats_stop{false};   // set when the state is destroyed: the snapshot thread then stops without touching it
#endif
constexpr size_t PAGE = 4096;

// GiB value from an environment variable: finite, not negative, at most 1 PiB; anything else is ignored with a message
bool parse_gib(const char * name, const char * txt, size_t & out) {
    char * end = nullptr;
    const double v = strtod(txt, &end);
    if (end == txt || !std::isfinite(v) || v < 0.0 || v > 1048576.0) {
        fprintf(stderr, "moe-cache: ignoring %s=%s (expected a number of GiB, 0 to 1048576)\n", name, txt);
        return false;
    }
    out = (size_t) (v * 1073741824.0);
    return true;
}

size_t round_up(size_t x, size_t a = PAGE) {
    return (x + a - 1) / a * a;
}

struct node_t {
    int      prev = -1;
    int      next = -1;
    uint64_t epoch = 0;
    bool     resident = false;
    bool     loading = false;      // a lookahead read is still filling this expert
    bool     prefetched = false;   // loaded ahead of time and not used yet
};

struct tinfo_t {
    std::string name;
    size_t  file_off = 0;
    size_t  nb2 = 0;
    int64_t n_expert = 0;
    char *  data = nullptr;   // the tensor's own address: expert e lives at data + e * nb2
    int     node0 = 0;        // index of expert 0 in the global node table
    int     layer = -1;       // from the tensor name blk.N.
    int     fdi = 0;          // model file index (split GGUF)
    bool    direct = false;   // can be read straight into place
    bool    repack = false;   // stored in llama.cpp's repacked CPU layout (faster CPU kernels)
    ggml_tensor * tensor = nullptr;
};

struct usage_t {
    int n_layers = 0;
    int n_exp = 0;
    std::vector<double> c;
    double at(int l, int e) const {
        return (l >= 0 && l < n_layers && e >= 0 && e < n_exp) ? c[(size_t) l * n_exp + e] : 0.0;
    }
};

// file "XUSE", int32 layers, int32 experts, float counts[layers*experts], optional uint64 model fingerprint; or old int16 pairs (layer, expert), most used first
bool load_usage(const char * path, usage_t & u, uint64_t * fp) {
    constexpr size_t MAX_CELLS = (size_t) 32 << 20;   // far above any real model (48 layers x 512 experts = 24k)
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    if (fp) {
        *fp = 0;
    }
    fseek(f, 0, SEEK_END);
    const long fsz = ftell(f);
    rewind(f);
    char magic[4];
    if (fsz >= 4 && fread(magic, 1, 4, f) == 4 && memcmp(magic, "XUSE", 4) == 0) {
        int32_t nl = 0, ne = 0;
        if (fread(&nl, 4, 1, f) != 1 || fread(&ne, 4, 1, f) != 1 || nl <= 0 || ne <= 0 || nl > 4096 || ne > 65536 ||
            (size_t) nl * (size_t) ne > MAX_CELLS || (size_t) fsz < 12 + (size_t) nl * (size_t) ne * sizeof(float)) {   // the file must really hold the counts it announces
            fclose(f);
            return false;
        }
        std::vector<float> tmp((size_t) nl * ne);
        const bool ok = fread(tmp.data(), sizeof(float), tmp.size(), f) == tmp.size();
        uint64_t trailer = 0;
        if (ok && fp && fread(&trailer, 8, 1, f) == 1) {
            *fp = trailer;
        }
        fclose(f);
        if (!ok) {
            return false;
        }
        u.n_layers = nl;
        u.n_exp = ne;
        u.c.assign(tmp.begin(), tmp.end());
        return true;
    }
    // old format: int16 pairs (layer, expert), most used first
    if (fsz <= 0 || fsz % 4 != 0 || (size_t) fsz > ((size_t) 64 << 20)) {
        fclose(f);
        return false;
    }
    rewind(f);
    std::vector<int16_t> pairs;
    int16_t v;
    while (fread(&v, sizeof(v), 1, f) == 1) {
        if (v < 0) {                  // a negative index would write outside the table
            fclose(f);
            return false;
        }
        pairs.push_back(v);
    }
    fclose(f);
    const size_t n = pairs.size() / 2;
    if (n == 0) {
        return false;
    }
    int ml = 0, me = 0;
    for (size_t i = 0; i < n; i++) {
        ml = std::max<int>(ml, pairs[2 * i] + 1);
        me = std::max<int>(me, pairs[2 * i + 1] + 1);
    }
    if ((size_t) ml * (size_t) me > MAX_CELLS) {
        return false;
    }
    u.n_layers = ml;
    u.n_exp = me;
    u.c.assign((size_t) ml * me, 0.0);
    for (size_t i = 0; i < n; i++) {
        u.c[(size_t) pairs[2 * i] * me + pairs[2 * i + 1]] = (double) (n - i);
    }
    return true;
}

// memory limit of this process and its private memory (see platform.h)
// The limit is the smaller of the configured one (cgroup / job) and what the machine can really give: a 28 GB cgroup limit on a 31 GB laptop
// with the desktop running (25 GB available) was OOM-killed by the kernel at 26.3 GB before the cgroup limit was reached.
size_t read_mem_limit() {
    const size_t lim = plat::mem_limit();
    const size_t avail = plat::mem_available();
    return avail ? std::min(lim, avail + plat::rss_private()) : lim;
}

size_t read_rss_anon() {
    return plat::rss_private();
}

// router dot product: eight independent sums so the compiler can vectorise without -ffast-math; AVX2 version picked at run time
#if defined(__linux__) && defined(__x86_64__)
__attribute__((target_clones("avx2,fma", "default")))
#endif
float dot_f32(const float * a, const float * b, size_t n) {
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        for (int k = 0; k < 8; k++) {
            acc[k] += a[i + k] * b[i + k];
        }
    }
    float sum = 0.0f;
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    for (int k = 0; k < 8; k++) {
        sum += acc[k];
    }
    return sum;
}

struct io_job_t {
    char * dst;
    size_t off;
    size_t nb2;
    int    fdi;      // which model file (split GGUF)
    bool   direct;   // memory address matches the file offset (mod 4096): read straight into place
    const ggml_tensor * rtensor = nullptr;   // set: repack this expert while loading it
    int    node = -1;        // index in the node table (asynchronous loads clear its "loading" flag when done)
};

// ---- compute pool for decode-time expert matrix products (MOE_CACHE_EXEC=1) ----
// llama.cpp splits the rows of an expert statically over its threads and waits for all SSD reads before it starts. This pool takes row chunks
// from a shared counter, starts on the experts that are already in RAM and waits only for the others, so it can also use slower cores.
// It calls the same dot-product functions with the same arguments, so every output value is bit-identical to llama.cpp's.
static inline void cpu_relax() {
#if defined(__x86_64__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

constexpr int EXEC_MAXCOL = 8;   // tokens per step the executor takes (single-token decode and small batches such as speculative verification)

struct exec_task_t {
    const char * w;        // first row of this expert's matrix
    int          ncol = 0; // activation columns that use this expert (one per token that picked it)
    const char * cols[EXEC_MAXCOL];   // quantised activation columns
    float *      dsts[EXEC_MAXCOL];   // outputs (rows floats each)
    const uint8_t * flag;  // non-null: the expert is still being read while *flag is non-zero
    int64_t      rows;
};

struct exec_job_t {
    const exec_task_t * tasks = nullptr;
    int      total = 0;        // chunks
    int      nch = 1;          // chunks per task
    int      chunk = 32;       // rows per chunk
    int64_t  ne00 = 0;
    size_t   nb01 = 0;
    ggml_vec_dot_t dot = nullptr;
    std::atomic<int> next{0};
    std::atomic<int> done{0};
};

struct exec_pool_t {
    std::vector<std::thread> th;
    std::atomic<uint32_t> gen{0};
    std::atomic<bool> stop{false};
    std::atomic<exec_job_t *> cur{nullptr};
    std::atomic<int> inside{0};
    std::atomic<int> parked{0};
    std::mutex m;
    std::condition_variable cv;
    exec_job_t jobs[2];
    int which = 0;

    static void work(exec_job_t * j) {
        for (;;) {
            const int c = j->next.fetch_add(1, std::memory_order_relaxed);
            if (c >= j->total) {
                return;
            }
            const exec_task_t & t = j->tasks[c / j->nch];
            const int64_t r0 = (int64_t) (c % j->nch) * j->chunk;
            const int64_t r1 = std::min<int64_t>(r0 + j->chunk, t.rows);
            if (t.flag) {
                while (__atomic_load_n(t.flag, __ATOMIC_ACQUIRE)) {
                    cpu_relax();
                }
            }
            for (int64_t r = r0; r < r1; r++) {
                const char * wr = t.w + (size_t) r * j->nb01;       // the weight row stays in L1 while every token column uses it
                for (int k = 0; k < t.ncol; k++) {
                    j->dot((int) j->ne00, &t.dsts[k][r], 0, wr, 0, t.cols[k], 0, 1);
                }
            }
            j->done.fetch_add(1, std::memory_order_release);
        }
    }

    void start(const std::vector<int> & cpus, int n_workers) {
        for (int i = 0; i < n_workers; i++) {
            const int cpu = i < (int) cpus.size() ? cpus[(size_t) i] : -1;
            th.emplace_back([this, cpu] {
#ifdef __linux__
                if (cpu >= 0 && cpu < CPU_SETSIZE) {
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    CPU_SET(cpu, &set);
                    sched_setaffinity(0, sizeof(set), &set);
                }
#endif
                uint32_t seen = gen.load(std::memory_order_acquire);
                for (;;) {
                    int spins = 0;
                    while (gen.load(std::memory_order_acquire) == seen) {
                        if (stop.load(std::memory_order_relaxed)) {
                            return;
                        }
                        if (++spins < 40000) {
                            cpu_relax();
                        } else {
                            std::unique_lock<std::mutex> lk(m);
                            parked++;
                            cv.wait_for(lk, std::chrono::microseconds(200));
                            parked--;
                            spins = 0;
                        }
                    }
                    seen = gen.load(std::memory_order_acquire);
                    inside.fetch_add(1);
                    exec_job_t * j = cur.load(std::memory_order_acquire);
                    if (j) {
                        work(j);
                    }
                    inside.fetch_sub(1);
                }
            });
        }
    }

    // runs the job on the pool and the calling thread; returns when every chunk is done and no worker is inside the job any more
    void run(const exec_task_t * tasks, int n_tasks, int64_t rows, int64_t ne00, size_t nb01, ggml_vec_dot_t dot, int chunk) {
        exec_job_t * j = &jobs[which];
        which ^= 1;
        j->tasks = tasks;
        j->nch = (int) ((rows + chunk - 1) / chunk);
        j->total = n_tasks * j->nch;
        j->chunk = chunk;
        j->ne00 = ne00;
        j->nb01 = nb01;
        j->dot = dot;
        j->next.store(0, std::memory_order_relaxed);
        j->done.store(0, std::memory_order_relaxed);
        cur.store(j, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
        if (parked.load() > 0) {
            cv.notify_all();
        }
        work(j);
        while (j->done.load(std::memory_order_acquire) < j->total || inside.load() > 0) {
            cpu_relax();
        }
    }

    ~exec_pool_t() {
        stop = true;
        cv.notify_all();
        for (auto & t : th) {
            t.join();
        }
    }
};

// true when the server was started with --no-repack / -nr (or LLAMA_ARG_REPACK=0): stock then runs the plain CPU kernels, so the plugin must not repack either
// (repacked kernels add up in another order and the tokens would no longer match stock)
static bool server_repack_disabled() {
    if (const char * v = getenv("LLAMA_ARG_REPACK")) {
        if (!strcmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "off") || !strcasecmp(v, "no") || !strcasecmp(v, "disabled")) {
            return true;
        }
    }
#ifdef __linux__
    if (FILE * cf = fopen("/proc/self/cmdline", "rb")) {
        std::string cmd;
        char buf[4096];
        size_t nr;
        while ((nr = fread(buf, 1, sizeof(buf), cf)) > 0) {
            cmd.append(buf, nr);
        }
        fclose(cf);
        for (size_t b = 0; b < cmd.size();) {
            const size_t e = cmd.find('\0', b);
            const std::string a = cmd.substr(b, e == std::string::npos ? std::string::npos : e - b);
            if (a == "--no-repack" || a == "-nr") {
                return true;
            }
            if (e == std::string::npos) {
                break;
            }
            b = e + 1;
        }
    }
#endif
    return false;
}

struct state_t {
    bool   cache_on = false;
    bool   stats = false;
    int    lo = 0, hi = 1 << 30;
    std::vector<plat::file_t> fds;              // one per GGUF file (split models have several)
    std::vector<std::string> shard_paths;
    std::vector<plat::file_t> fds_buf;          // buffered (no O_DIRECT) handles, opened only if a direct read keeps failing
    std::mutex fds_buf_mu;
    int    fault_every = 0;                     // MOE_CACHE_FAULT_EVERY=N: test switch, every Nth expert read fails all its direct attempts
    std::atomic<uint64_t> fault_ctr{0}, read_fallbacks{0};
    std::unordered_map<std::string, int> file_shard;
    ggml_backend_buffer_type_t repack_buft = nullptr;   // llama.cpp CPU_REPACK buffer type (mode on)
    ggml_backend_buffer_t repack_buf = nullptr;         // a real repack buffer, used for init_tensor / set_tensor only
    ggml_backend_buffer_t repack_tag = nullptr;         // empty buffer of the repack type, swapped in while an operation runs
    std::atomic<uint64_t> repack_jobs{0};
    ggml_backend_dev_t gpu_dev = nullptr;               // first GPU backend device, used for long prompts
    int    gpu_min_tokens = 32;                         // batches of at least this many tokens run their expert block on the GPU (0 = never)
    std::atomic<uint64_t> hyb_groups{0}, hyb_picks{0}, hyb_hits{0};   // MOE_CACHE_HYBRID: layers split between GPU bank and CPU, picks, picks served by the bank
    std::atomic<uint64_t> hyb_drops{0}, hyb_reloads{0}, hyb_reload_ns{0};
    std::atomic<uint64_t> hyb_swaps{0}, hyb_swap_ns{0}, hyb_reverts{0};   // MOE_CACHE_HYBRID_ADAPT: experts swapped into the bank, time spent uploading them   // bank freed for a long prompt's GPU block, loaded again at the next decode step
    std::atomic<uint64_t> gpu_segments{0}, gpu_fallbacks{0}, gpu_nomem{0};   // gpu_nomem: blocks kept on the CPU because the GPU had too little free memory
    std::atomic<uint64_t> gpu_up_bytes{0};
    // live statistics: with MOE_CACHE_STATS_SIGNAL=1, SIGUSR1 makes the plugin print one "moe-cache-stats {json}" line (cumulative counters)
    std::atomic<uint64_t> hits_dec{0}, misses_dec{0}, evict_age_sum{0};
    bool cur_decode = false;                            // the group being prepared is a single-token (decode) step
    bool cur_scan = false;                              // the group being prepared is a prompt batch: it scans (nearly) all experts of the layer
    int  scan_tokens = 0;                               // MOE_CACHE_SCAN_TOKENS=N: batches of at least N tokens are loaded scan-resistant (default off: measured -25 % prefill)
    // approximate routing (MOE_CACHE_ROUTE=skip|subst, decode and small batches, executor only; off by default; quality-gated, see docs/EXPERIMENTS.md):
    // the lowest-ranked picks of a token that are NOT in RAM are dropped (skip) or replaced by the most recently used resident expert of that layer (subst).
    // Ranked picks are in order of the router's weights, so the experts given up are the ones that contribute least.
    int  route_mode = 0;                                // 0 off, 1 skip, 2 subst
    int  route_keep = -1;                               // MOE_CACHE_ROUTE_KEEP: picks always honoured (default: all but the last 2)
    uint64_t route_skipped = 0, route_substituted = 0, route_considered = 0;
    std::unordered_map<const ggml_tensor *, std::vector<int32_t>> route_ids;   // ids tensor -> effective expert per (token, pick); -1 = skipped
    FILE * trace_f = nullptr;                           // MOE_CACHE_TRACE_IDS=file: record the experts picked in every decode step (int16 layer, int16 n, n x int16 expert)
    std::vector<uint64_t> layer_hits, layer_misses;     // per layer, updated under mu
    // ---- lookahead prefetch (MOE_CACHE_LOOKAHEAD=1): while layer N computes, predict the experts of layer N+1 from the router weights and read them
    bool   l3_prefetch = false;                         // MOE_CACHE_L3PREFETCH=1: a helper thread pulls the later tensors of a group into the L3 cache while the first ones compute
    bool   siblings = true;                             // MOE_CACHE_SIBLINGS=0 turns the sibling batch load off
    std::atomic<uint64_t> sibling_loads{0};
    bool   lookahead = false;
    bool   prompt_prefetch = false;
    size_t coalesce_bytes = 0;                          // MOE_CACHE_COALESCE_MIB=N: neighbouring expert reads merged up to N MiB (synchronous reads)
    std::atomic<uint64_t> coalesced{0};                     // MOE_CACHE_PROMPT_PREFETCH=1: during long prompts read layer N+1's experts while layer N runs
    int    la_guess = 0;                                // experts looked at per layer (0 = the number used per token)
    int    la_cap = 2;                                  // at most this many experts not yet in RAM are prefetched per layer
    bool   la_ecores = true;                            // run the helper threads on the efficiency cores of a hybrid CPU
    size_t max_nb2_all = 0;                             // largest expert slice, for the helper threads' bounce buffers
    std::vector<std::thread> pf_threads;
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    std::deque<std::function<void()>> pf_q;
    bool   pf_stop = false;
    std::condition_variable load_cv;                    // used with mu: a lookahead read finished
    std::unordered_map<int, std::vector<int>> layer_tensors;   // layer -> indices into tensors
    struct tmeta_t { int type; size_t off; size_t size; int shard; };
    std::unordered_map<std::string, tmeta_t> tmeta;     // router and norm tensors of the GGUF
    struct router_t { std::vector<float> w, bias, normw; bool ok = false; };
    std::unordered_map<int, router_t> routers;
    std::mutex router_mu;
    std::unordered_map<int, std::vector<char>> pred_flags;     // layer -> experts predicted for it (under mu)
    std::atomic<uint64_t> pf_issued{0}, pf_used{0}, pf_wasted{0}, pf_late{0}, pf_pred_total{0}, pf_pred_hit{0}, pf_skipped{0};
    bool   disabled = false;                    // setup failed or MOE_CACHE_DISABLE: the plugin claims nothing, the model loads normally
    bool   direct_ok = false;                   // read experts straight into place when the alignment allows
    size_t budget = 0;
    bool   auto_mode = false;          // MOE_CACHE_SIZE_GIB=auto: size the cache from the real memory limit
    size_t limit_bytes = 0;
    size_t margin = (size_t) 1 << 30;  // keep this much below the limit (MOE_CACHE_MARGIN_GIB, default 1 GiB)
    size_t pending_bytes = 0;          // expert bytes marked resident but not read yet
    size_t max_budget = 0;             // all registered expert bytes
    size_t min_budget = 0;
    uint64_t tune_shrinks = 0, tune_grows = 0;
    size_t resident_bytes = 0;
    uint64_t epoch = 0;
    int    thp = -1;                 // -1 default, 0 never, 1 madvise(HUGEPAGE)
    std::unordered_map<std::string, std::pair<size_t, size_t>> file_tensors;
    std::vector<std::unique_ptr<tinfo_t>> tensors;
    std::unordered_map<const ggml_tensor *, int> by_tensor;
    std::vector<node_t> nodes;
    int    mru = -1;
    int    lru = -1;
    int    unmatched = 0;              // expert tensors that could not be cached (see register_tensor)
    // executor (MOE_CACHE_EXEC=1): asynchronous reads + the plugin's own expert matrix products for decode steps
    bool   exec_on = true;                              // on by default; MOE_CACHE_EXEC=0 switches the executor off
    bool   exec_force = false;                          // MOE_CACHE_EXEC=2: the executor runs every eligible step (tests)
    bool   exec_check = false;                          // MOE_CACHE_EXEC_CHECK=1: recompute every op the stock way and compare (slow, for tests)
    int    exec_threads = 0;                            // 0 = the thread count llama.cpp asks for
    int    exec_maxcol = EXEC_MAXCOL;                   // MOE_CACHE_EXEC_MAXCOL=N (1-8): largest batch (tokens) the executor takes
    int    exec_ecores = 0;                             // extra workers on efficiency cores
    int    exec_readers = 4;
    std::unique_ptr<exec_pool_t> pool;
    std::vector<std::thread> readers;
    std::mutex rq_mu;
    std::condition_variable rq_cv;
    std::deque<io_job_t> rq;
    bool   rq_stop = false;
    std::atomic<uint64_t> exec_ops{0}, exec_mismatch{0}, exec_wait_ns{0}, exec_checked{0};
    // per weight type: wall time of the expert products (includes waiting for experts still loading), ops, expert bytes touched
    std::atomic<uint64_t> exec_type_ns[64] = {}, exec_type_ops[64] = {}, exec_type_bytes[64] = {};

    // usage profile and warm start
    std::string profile_path;          // learn here and warm start from it
    std::string warm_path;             // warm start only
    bool   preload_on = false;
    bool   preload_done = false;
    double preload_frac = -1.0;        // < 0: everything if it fits, else 0.7
    double profile_decay = 0.5;
    uint64_t fingerprint = 0;
    usage_t learned;
    std::vector<std::vector<double>> session;   // [layer][expert] picks of this run
    uint64_t groups_since_save = 0;
    std::mutex mu;

    std::atomic<uint64_t> hits{0}, misses{0}, bytes{0}, groups{0}, ops{0}, graphs{0};
    std::atomic<uint64_t> ns_prep{0}, ns_cpu{0}, ns_total{0}, ns_read{0}, evictions{0}, direct_jobs{0}, bounce_jobs{0};

    // io workers
    std::vector<std::thread> workers;
    std::mutex wmu;
    std::condition_variable wcv, dcv;
    std::function<void(int)> task;
    int    gen = 0;
    int    busy = 0;
    bool   stop = false;

    state_t() {
        stats = getenv("MOE_CACHE_STATS") != nullptr;
        fprintf(stderr, "moe-cache: plugin v0.2.3.2 (ggml backend API %d), running with ggml %s (%s)\n", GGML_BACKEND_API_VERSION, ggml_version(), ggml_commit());
        if (strcmp(ggml_version(), MOE_CACHE_TESTED_GGML) != 0) {
            fprintf(stderr, "moe-cache: warning: tested with ggml %s only; if the server crashes or the plugin misbehaves, rebuild it against your llama.cpp headers (or set MOE_CACHE_DISABLE=1)\n", MOE_CACHE_TESTED_GGML);
        }
        if (const char * v = getenv("MOE_CACHE_THP")) {
            thp = atoi(v);
        }
        if (const char * v = getenv("MOE_CACHE_PROFILE")) {
            profile_path = v;
        }
        if (const char * v = getenv("MOE_CACHE_WARM")) {
            warm_path = v;
        }
        if (const char * v = getenv("MOE_CACHE_PRELOAD_FRAC")) {
            preload_frac = atof(v);
        }
        if (const char * v = getenv("MOE_CACHE_PROFILE_DECAY")) {
            profile_decay = atof(v);
        }
        if (const char * v = getenv("MOE_CACHE_LAYERS")) {
            int a = 0, b = 0;
            if (sscanf(v, "%d-%d", &a, &b) == 2) {
                lo = a;
                hi = b;
            }
        }
        const char * gib = getenv("MOE_CACHE_SIZE_GIB");
        const char * file = getenv("MOE_CACHE_GGUF");
        if (!gib) {
            fprintf(stderr, "moe-cache: pass-through mode (no cache), layers %d-%d\n", lo, hi);
            return;
        }
        auto fail = [&](const std::string & m) {
            fprintf(stderr, "moe-cache: %s - plugin disabled, the model loads the normal way\n", m.c_str());
            disabled = true;
        };
        if (getenv("MOE_CACHE_DISABLE")) {
            fail("MOE_CACHE_DISABLE is set");
            return;
        }
        {
            const int api = moe_cache_api_of_ggml(ggml_version());
            if (api > 0 && api != GGML_BACKEND_API_VERSION) {
                fail("this plugin was built for ggml backend API " + std::to_string(GGML_BACKEND_API_VERSION) + " but the server uses ggml " + ggml_version() +
                     " (API " + std::to_string(api) + "): use the plugin build that matches your llama.cpp (see docs/QUICKSTART.md)");
                return;
            }
        }
        if (!file) {
            fail("MOE_CACHE_GGUF is not set");
            return;
        }
        // split models: name-00001-of-0000N.gguf -> open all parts
        std::vector<std::string> shards = { file };
        {
            std::string f = file;
            int idx = 0, cnt = 0;
            const size_t of = f.rfind("-of-");
            if (of != std::string::npos && of >= 6) {
                const size_t start = of - 6;   // "-00001-of-00003.gguf"
                if (sscanf(f.c_str() + start, "-%d-of-%d.gguf", &idx, &cnt) == 2 && cnt > 1 && cnt < 1000) {
                    shards.clear();
                    for (int i = 1; i <= cnt; i++) {
                        char suffix[64];
                        snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, cnt);
                        shards.push_back(f.substr(0, start) + suffix);
                    }
                }
            }
        }
        for (size_t si = 0; si < shards.size(); si++) {
            gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
            gguf_context * g = gguf_init_from_file(shards[si].c_str(), ip);
            if (!g) {
                fail("cannot read " + shards[si] + " (MOE_CACHE_GGUF must be the model file llama-server loads)");
                return;
            }
            const size_t data_off = gguf_get_data_offset(g);
            const int64_t n = gguf_get_n_tensors(g);
            for (int64_t i = 0; i < n; i++) {
                const char * nm = gguf_get_tensor_name(g, i);
                file_tensors[nm] = { data_off + gguf_get_tensor_offset(g, i), gguf_get_tensor_size(g, i) };
                if (strstr(nm, "ffn_gate_inp") || strstr(nm, "ffn_norm") || strstr(nm, "post_attention_norm")) {
                    tmeta[nm] = { (int) gguf_get_tensor_type(g, i), data_off + gguf_get_tensor_offset(g, i), gguf_get_tensor_size(g, i), (int) si };
                }
                file_shard[nm] = (int) si;
            }
            gguf_free(g);
            const plat::file_t f = plat::file_open(shards[si].c_str(), true);
            if (!f.ok()) {
                fail("cannot open " + shards[si] + " with O_DIRECT");
                return;
            }
            fds.push_back(f);
            shard_paths.push_back(shards[si]);
            fds_buf.push_back(plat::file_t());
        }
        if (const char * v = getenv("MOE_CACHE_FAULT_EVERY")) {
            fault_every = atoi(v);
        }
        const char * direct_env = getenv("MOE_CACHE_DIRECT");   // 1 / 0 forces it; unset = on when the model does not fit (decided below)
        direct_ok = direct_env && strcmp(direct_env, "0") != 0;
        if (const char * v = getenv("MOE_CACHE_GPU_MIN_TOKENS")) {
            gpu_min_tokens = atoi(v);
            if (gpu_min_tokens == 1) {
                // 1-token batches on the GPU produce one repeated token (Granite 3.1, T25; 2+ tokens are fine) and would upload whole layers for every
                // decoded token anyway: the executor handles single tokens
                fprintf(stderr, "moe-cache: MOE_CACHE_GPU_MIN_TOKENS=1 is not supported (wrong output for single-token batches): using 2\n");
                gpu_min_tokens = 2;
            }
        }
        for (size_t i = 0; i < ggml_backend_dev_count() && gpu_min_tokens > 0; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                gpu_dev = d;
                break;
            }
        }
        {
            const char * rp = getenv("MOE_CACHE_REPACK");
            const std::string mode = rp ? rp : "auto";
            bool want = mode == "on";
            if (mode == "auto") {
                // stock llama.cpp repacks CPU weights only when no GPU backend is present: do the same, so results stay identical
                bool gpu = false;
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    const auto t = ggml_backend_dev_type(ggml_backend_dev_get(i));
                    gpu = gpu || t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
                }
                want = !gpu && !server_repack_disabled();
            }
            if (want) {
                ggml_backend_dev_t cd = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                auto fn = cd ? (ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cd), "ggml_backend_dev_get_extra_bufts") : nullptr;
                for (ggml_backend_buffer_type_t * b = fn ? fn(cd) : nullptr; b && *b; ++b) {
                    if (strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) {
                        repack_buft = *b;
                    }
                }
                if (repack_buft) {
                    repack_buf = ggml_backend_buft_alloc_buffer(repack_buft, 64);
                    ggml_backend_buffer_i empty;
                    memset(&empty, 0, sizeof(empty));
                    repack_tag = ggml_backend_buffer_init(repack_buft, empty, nullptr, 0);
                }
                fprintf(stderr, "moe-cache: CPU weight repacking %s\n", repack_buf && repack_tag ? "on (faster CPU kernels)" : "requested but not available in this llama.cpp");
                if (!repack_buf || !repack_tag) {
                    repack_buf = nullptr;
                    repack_tag = nullptr;
                }
            }
        }   // off by default: reading straight into place measured no faster than the bounce buffer
        if (strcmp(gib, "auto") == 0) {
            auto_mode = true;
            limit_bytes = read_mem_limit();
            if (const char * v = getenv("MOE_CACHE_RAM_GIB")) {
                size_t x = 0;
                if (parse_gib("MOE_CACHE_RAM_GIB", v, x) && x > 0) {
                    limit_bytes = x;
                }
            }
            if (const char * v = getenv("MOE_CACHE_MARGIN_GIB")) {
                parse_gib("MOE_CACHE_MARGIN_GIB", v, margin);
            }
            // llama-server keeps saved prompt states in host RAM (--cache-ram, MiB): that memory is outside this cache's control and arrives in bursts
            // when a new request starts, so it is added to the margin (a 24 GB limit was OOM-killed at 128k context with the 8 GiB default)
            if (!getenv("MOE_CACHE_MARGIN_GIB")) {
                long cram = -1;
                const char * reserve = getenv("MOE_CACHE_HOST_RESERVE_MIB");   // set by moe-cache-server; the command line below is only a fallback
                if (reserve) {
                    cram = atol(reserve);
                } else if (const char * v = getenv("LLAMA_ARG_CACHE_RAM")) {
                    cram = atol(v);
                }
#ifdef __linux__
                if (FILE * cf = reserve ? nullptr : fopen("/proc/self/cmdline", "rb")) {
                    std::string cmd;
                    char buf[4096];
                    size_t nr;
                    while ((nr = fread(buf, 1, sizeof(buf), cf)) > 0) {
                        cmd.append(buf, nr);
                    }
                    fclose(cf);
                    std::vector<std::string> argv_;
                    for (size_t b = 0; b < cmd.size();) {
                        const size_t e = cmd.find('\0', b);
                        argv_.push_back(cmd.substr(b, e == std::string::npos ? std::string::npos : e - b));
                        if (e == std::string::npos) break;
                        b = e + 1;
                    }
                    for (size_t k = 0; k + 1 < argv_.size(); k++) {
                        if (argv_[k] == "--cache-ram" || argv_[k] == "-cram") {
                            cram = atol(argv_[k + 1].c_str());
                        }
                    }
                }
#endif
                if (cram > 0) {
                    const size_t extra = std::min<size_t>((size_t) cram << 20, limit_bytes / 3);
                    margin += extra;
                    fprintf(stderr, "moe-cache: llama-server's prompt cache (--cache-ram %ld MiB) is kept free: margin now %.2f GiB\n", cram, margin / 1073741824.0);
                }
            }
            if (!direct_env) {
                // the model does not fit: SSD reads dominate, and reading straight into place (no bounce buffer, no page re-zeroing) was +22 % on gpt-oss-120b
                size_t total = 0;
                for (const auto & sp : shard_paths) {
                    std::error_code ec;
                    total += (size_t) std::filesystem::file_size(sp, ec);
                }
                direct_ok = total > limit_bytes * 0.85;
            }
            const size_t guess = (size_t) 7 << 29;   // 3.5 GiB for everything that is not expert cache, until measured
            budget = limit_bytes > margin + guess + ((size_t) 1 << 30) ? limit_bytes - margin - guess : (size_t) 1 << 30;
            fprintf(stderr, "moe-cache: auto cache size, memory limit %.2f GiB, margin %.2f GiB, start budget %.2f GiB\n", limit_bytes / 1073741824.0, margin / 1073741824.0, budget / 1073741824.0);
        } else {
            if (!parse_gib("MOE_CACHE_SIZE_GIB", gib, budget) || budget == 0) {
                fail(std::string("MOE_CACHE_SIZE_GIB='") + gib + "' is not a valid size");
                return;
            }
        }
        cache_on = true;
        {
            std::vector<std::string> names;
            for (const auto & kv : file_tensors) {
                names.push_back(kv.first);
            }
            std::sort(names.begin(), names.end());
            uint64_t h = 1469598103934665603ull;
            for (const auto & nm : names) {
                const auto & v = file_tensors[nm];
                for (unsigned char ch : nm) { h = (h ^ ch) * 1099511628211ull; }
                h = (h ^ v.second) * 1099511628211ull;
            }
            fingerprint = h;
        }
        const std::string & src = !profile_path.empty() ? profile_path : warm_path;
        if (!src.empty()) {
            uint64_t fp = 0;
            if (load_usage(src.c_str(), learned, &fp)) {
                if (fp != 0 && fp != fingerprint) {
                    fprintf(stderr, "moe-cache: profile %s belongs to another model, ignored\n", src.c_str());
                    learned = usage_t();
                } else {
                    preload_on = true;
                    fprintf(stderr, "moe-cache: profile %s loaded (%d layers x %d experts)%s\n", src.c_str(), learned.n_layers, learned.n_exp, fp ? "" : " (no model fingerprint)");
                }
            } else {
                fprintf(stderr, "moe-cache: no profile at %s yet%s\n", src.c_str(), profile_path.empty() ? "" : " (it will be created at exit)");
            }
        }
        int n_io = 5;   // helper threads; the calling thread reads too, so MOE_CACHE_IO_THREADS=N means N parallel readers in total
        if (const char * v = getenv("MOE_CACHE_IO_THREADS")) {
            n_io = std::max(0, std::min(63, atoi(v) - 1));
        }
        for (int i = 0; i < n_io; i++) {
            workers.emplace_back([this, i] { worker_main(i + 1); });
        }
        fprintf(stderr, "moe-cache: cache mode, %.2f GiB, %zu tensors in %s, layers %d-%d\n", budget / 1073741824.0, file_tensors.size(), file, lo, hi);
        if (const char * l3 = getenv("MOE_CACHE_L3PREFETCH")) {
            l3_prefetch = strcmp(l3, "0") != 0;
            if (l3_prefetch && pf_threads.empty()) {
                pf_threads.emplace_back([this] { pf_main(); });
                fprintf(stderr, "moe-cache: L3 prefetch helper thread on\n");
            }
        }
        if (const char * ex = getenv("MOE_CACHE_EXEC")) {
            exec_on = strcmp(ex, "0") != 0;
            exec_force = strcmp(ex, "2") == 0;        // tests: never step aside, even when everything is resident and bandwidth-bound
        }
        if (const char * v = getenv("MOE_CACHE_ROUTE")) {
            route_mode = strcmp(v, "skip") == 0 ? 1 : (strcmp(v, "subst") == 0 ? 2 : 0);
        }
        if (const char * v = getenv("MOE_CACHE_ROUTE_KEEP")) {
            route_keep = atoi(v);
        }
        if (const char * v = getenv("MOE_CACHE_SCAN_TOKENS")) {
            scan_tokens = atoi(v);
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_CHECK")) {
            exec_check = strcmp(v, "0") != 0;
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_MAXCOL")) {
            exec_maxcol = std::max(1, std::min(EXEC_MAXCOL, atoi(v)));
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_THREADS")) {
            exec_threads = std::max(0, std::min(64, atoi(v)));
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_ECORES")) {
            exec_ecores = std::max(0, std::min(32, atoi(v)));
        }
        if (const char * v = getenv("MOE_CACHE_EXEC_READERS")) {
            exec_readers = std::max(1, std::min(16, atoi(v)));
        }
        if (exec_on) {
            for (int i = 0; i < exec_readers; i++) {
                readers.emplace_back([this] { reader_main(); });
            }
            fprintf(stderr, "moe-cache: expert executor on (%d reader threads, %d extra efficiency-core workers)\n", exec_readers, exec_ecores);
        }
        if (const char * sb = getenv("MOE_CACHE_SIBLINGS")) {
            siblings = strcmp(sb, "0") != 0;
        }
        if (const char * tp = getenv("MOE_CACHE_TRACE_IDS")) {
            trace_f = fopen_private(tp);
        }
        if (const char * v = getenv("MOE_CACHE_COALESCE_MIB")) {
            coalesce_bytes = (size_t) std::max(0, atoi(v)) << 20;
        }
        if (const char * v = getenv("MOE_CACHE_PROMPT_PREFETCH")) {
            prompt_prefetch = strcmp(v, "0") != 0;
        }
        if (const char * v = getenv("MOE_CACHE_LOOKAHEAD")) {
            lookahead = strcmp(v, "0") != 0;
            if (const char * g = getenv("MOE_CACHE_LOOKAHEAD_GUESS")) {
                la_guess = atoi(g);
            }
            if (const char * g = getenv("MOE_CACHE_LOOKAHEAD_CAP")) {
                la_cap = std::max(1, atoi(g));
            }
            if (const char * g = getenv("MOE_CACHE_LOOKAHEAD_ECORES")) {
                la_ecores = strcmp(g, "0") != 0;
            }
            if (lookahead) {
                int nt = 4;
                if (const char * t = getenv("MOE_CACHE_LOOKAHEAD_THREADS")) {
                    nt = std::max(1, std::min(32, atoi(t)));
                }
                for (int i = 0; i < nt; i++) {
                    pf_threads.emplace_back([this] { pf_main(); });
                }
                fprintf(stderr, "moe-cache: lookahead prefetch on (%d helper threads)\n", nt);
            }
        }
        if (prompt_prefetch && pf_threads.empty()) {
            for (int i = 0; i < 4; i++) {
                pf_threads.emplace_back([this] { pf_main(); });
            }
            fprintf(stderr, "moe-cache: prompt prefetch on (long prompts read the next layer's experts in the background)\n");
        }
#ifdef __linux__
        if (getenv("MOE_CACHE_STATS_SIGNAL")) {
            if (pipe(g_stats_pipe) == 0) {
                struct sigaction sa;
                memset(&sa, 0, sizeof(sa));
                sa.sa_handler = [](int) { const char c = 1; const ssize_t r = write(g_stats_pipe[1], &c, 1); (void) r; };   // async-signal-safe: only write()
                sa.sa_flags = SA_RESTART;
                sigaction(SIGUSR1, &sa, nullptr);
                std::thread([this] { char c; while (read(g_stats_pipe[0], &c, 1) == 1 && !g_stats_stop) { print_snapshot(); } }).detach();
            }
        }
#endif
    }

    // ---- executor: background expert reads ----
    void reader_main() {
        static thread_local char * bounce = nullptr;
        static thread_local size_t bounce_size = 0;
        for (;;) {
            io_job_t j;
            {
                std::unique_lock<std::mutex> lk(rq_mu);
                rq_cv.wait(lk, [this] { return rq_stop || !rq.empty(); });
                if (rq_stop && rq.empty()) {
                    return;
                }
                j = rq.front();
                rq.pop_front();
            }
            const size_t need = round_up(max_nb2_all) + 2 * PAGE;
            if (bounce_size < need) {
                plat::aligned_free_page(bounce);
                bounce = (char *) plat::aligned_alloc_page(need);
                if (!bounce) {
                    fprintf(stderr, "moe-cache: out of memory for a read buffer\n");
                    abort();
                }
                bounce_size = need;
            }
            const auto r0 = std::chrono::steady_clock::now();
            read_job(j, bounce);
            ns_read += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - r0).count();
            {
                std::lock_guard<std::mutex> lk(mu);
                pending_bytes = pending_bytes > j.nb2 ? pending_bytes - j.nb2 : 0;
                __atomic_store_n(&nodes[(size_t) j.node].loading, false, __ATOMIC_RELEASE);
            }
            load_cv.notify_all();
        }
    }

    // ---- lookahead prefetch ----
    void pf_submit(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lk(pf_mu);
            pf_q.push_back(std::move(fn));
        }
        pf_cv.notify_one();
    }

    void pf_main() {
#ifdef __linux__
        if (la_ecores) {
            std::ifstream f("/sys/devices/cpu_atom/cpus");
            std::string txt;
            if (f && std::getline(f, txt)) {
                cpu_set_t set;
                CPU_ZERO(&set);
                size_t i = 0;
                bool any = false;
                while (i < txt.size()) {
                    int a = atoi(txt.c_str() + i), b = a;
                    while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
                    if (i < txt.size() && txt[i] == '-') {
                        i++;
                        b = atoi(txt.c_str() + i);
                        while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
                    }
                    for (int c = a; c <= b && c < CPU_SETSIZE; c++) { CPU_SET(c, &set); any = true; }
                    while (i < txt.size() && !isdigit((unsigned char) txt[i])) { i++; }
                }
                if (any) {
                    sched_setaffinity(0, sizeof(set), &set);
                }
            }
        }
#endif
        for (;;) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> lk(pf_mu);
                pf_cv.wait(lk, [this] { return pf_stop || !pf_q.empty(); });
                if (pf_stop && pf_q.empty()) {
                    return;
                }
                fn = std::move(pf_q.front());
                pf_q.pop_front();
            }
            fn();
        }
    }

    plat::file_t buffered_file(int shard) {
        std::lock_guard<std::mutex> lk(fds_buf_mu);
        if (!fds_buf[shard].ok()) {
            fds_buf[shard] = plat::file_open(shard_paths[shard].c_str(), false);
        }
        return fds_buf[shard];
    }

    bool read_floats(const std::string & name, size_t n, std::vector<float> & out) {
        auto it = tmeta.find(name);
        if (it == tmeta.end()) {
            return false;
        }
        const tmeta_t & m = it->second;
        const ggml_type ty = (ggml_type) m.type;
        if (ty != GGML_TYPE_F32 && ty != GGML_TYPE_F16 && ty != GGML_TYPE_BF16) {
            return false;
        }
        const size_t esz = ggml_type_size(ty);
        if (m.size != n * esz) {
            return false;
        }
        std::vector<char> raw(m.size);
        const plat::file_t f = buffered_file(m.shard);
        size_t done = 0;
        while (done < m.size) {
            const ptrdiff_t got = plat::file_pread(f, raw.data() + done, m.size - done, (uint64_t) (m.off + done));
            if (got <= 0) {
                return false;
            }
            done += (size_t) got;
        }
        out.resize(n);
        for (size_t i = 0; i < n; i++) {
            if (ty == GGML_TYPE_F32) {
                memcpy(&out[i], raw.data() + i * 4, 4);
            } else if (ty == GGML_TYPE_F16) {
                ggml_fp16_t h;
                memcpy(&h, raw.data() + i * 2, 2);
                out[i] = ggml_fp16_to_fp32(h);
            } else {
                ggml_bf16_t h;
                memcpy(&h, raw.data() + i * 2, 2);
                out[i] = ggml_bf16_to_fp32(h);
            }
        }
        return true;
    }

    router_t * get_router(int layer, size_t n_embd, size_t n_exp) {
        std::lock_guard<std::mutex> lk(router_mu);
        auto it = routers.find(layer);
        if (it != routers.end()) {
            return it->second.ok ? &it->second : nullptr;
        }
        router_t & r = routers[layer];
        char nm[160];
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer);
        r.ok = read_floats(nm, n_embd * n_exp, r.w);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.bias", layer);
        read_floats(nm, n_exp, r.bias);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", layer);
        if (!read_floats(nm, n_embd, r.normw)) {
            snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", layer);
            read_floats(nm, n_embd, r.normw);
        }
        return r.ok ? &r : nullptr;
    }

    // reserve cache room for the given experts of a layer (without evicting anything the current group needs) and read them in the background
    void reserve_and_read(int layer, const std::vector<int> & guess, int cap = -1) {
        if (cap < 0) {
            cap = la_cap;
        }
        std::vector<std::pair<io_job_t, int>> todo;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto lt = layer_tensors.find(layer);
            if (lt == layer_tensors.end()) {
                return;
            }
            int fresh = 0;
            for (int e : guess) {
                {
                    const tinfo_t & t0 = *tensors[lt->second[0]];
                    if (e < 0 || e >= t0.n_expert || nodes[t0.node0 + e].resident) {
                        continue;                 // out of range, or already in RAM: nothing to read
                    }
                }
                if (++fresh > cap) {
                    break;
                }
                for (int ti_i : lt->second) {
                    tinfo_t & ti = *tensors[ti_i];
                    const int s = ti.node0 + e;
                    if (nodes[s].resident) {
                        continue;
                    }
                    while (resident_bytes + ti.nb2 > budget) {
                        int v = lru;
                        while (v >= 0 && (nodes[v].epoch == epoch || nodes[v].loading)) {
                            v = nodes[v].prev;
                        }
                        if (v < 0) {
                            break;
                        }
                        evict(v);
                    }
                    if (resident_bytes + ti.nb2 > budget) {
                        pf_skipped++;
                        continue;
                    }
                    node_t & n = nodes[s];
                    n.resident = true;
                    n.loading = true;
                    n.prefetched = true;
                    n.epoch = epoch;
                    resident_bytes += ti.nb2;
                    push_mru(s);
                    todo.push_back({ io_job_t{ ti.data + (size_t) e * ti.nb2, ti.file_off + (size_t) e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr }, s });
                    pf_issued++;
                }
            }
        }
        for (auto & td : todo) {
            pf_submit([this, j = td.first, s = td.second] {
                static thread_local char * bounce = nullptr;
                static thread_local size_t bounce_size = 0;
                const size_t need = round_up(max_nb2_all) + 2 * PAGE;
                if (bounce_size < need) {
                    plat::aligned_free_page(bounce);
                    bounce = (char *) plat::aligned_alloc_page(need);
                    if (!bounce) {
                        fprintf(stderr, "moe-cache: out of memory for a read buffer\n");
                        abort();
                    }
                    bounce_size = need;
                }
                read_job(j, bounce);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    nodes[s].loading = false;
                }
                load_cv.notify_all();
            });
        }
    }

    // x = the input of the experts of this layer; predict which experts the next layer will pick and start reading them
    void predict_and_prefetch(int layer, const std::vector<float> & x, int n_used) {
        const int nl = layer + 1;
        size_t n_exp = 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto lt = layer_tensors.find(nl);
            if (lt == layer_tensors.end() || lt->second.empty()) {
                return;
            }
            n_exp = (size_t) tensors[lt->second[0]]->n_expert;
        }
        const size_t n_embd = x.size();
        router_t * rn = get_router(nl, n_embd, n_exp);
        if (!rn) {
            return;
        }
        router_t * rc = get_router(layer, n_embd, n_exp);
        std::vector<float> xs(n_embd);
        const bool scale = rc && rn->normw.size() == n_embd && rc->normw.size() == n_embd;
        for (size_t i = 0; i < n_embd; i++) {
            xs[i] = x[i] * (scale && fabsf(rc->normw[i]) > 1e-6f ? rn->normw[i] / rc->normw[i] : 1.0f);
        }
        std::vector<float> logit(n_exp);
        for (size_t e = 0; e < n_exp; e++) {
            const float * w = rn->w.data() + e * n_embd;
            logit[e] = (rn->bias.size() == n_exp ? rn->bias[e] : 0.0f) + dot_f32(w, xs.data(), n_embd);
        }
        const size_t G = std::min<size_t>(n_exp, la_guess > 0 ? (size_t) la_guess : (size_t) n_used);
        std::vector<int> idx(n_exp);
        for (size_t e = 0; e < n_exp; e++) { idx[e] = (int) e; }
        std::partial_sort(idx.begin(), idx.begin() + G, idx.end(), [&](int a, int b) { return logit[a] > logit[b]; });
        idx.resize(G);
        {
            std::lock_guard<std::mutex> lk(mu);
            auto & fl = pred_flags[nl];
            fl.assign(n_exp, 0);
            for (int e : idx) { fl[(size_t) e] = 1; }
        }
        reserve_and_read(nl, idx);
    }

    // one line of cumulative counters on stderr (see MOE_CACHE_STATS_SIGNAL)
    void print_snapshot() {
        std::string j;
        {
            std::lock_guard<std::mutex> lk(mu);
            char b[1600];
            snprintf(b, sizeof(b), "{\"hits\":%llu,\"misses\":%llu,\"hits_dec\":%llu,\"misses_dec\":%llu,\"read_bytes\":%llu,\"evictions\":%llu,\"evict_age_sum\":%llu,"
                     "\"resident_bytes\":%llu,\"budget\":%llu,\"ns_prep\":%llu,\"ns_read\":%llu,\"ns_cpu\":%llu,\"ns_total\":%llu,\"graphs\":%llu,"
                     "\"direct\":%llu,\"bounce\":%llu,\"repack\":%llu,\"gpu_segments\":%llu,\"gpu_up_bytes\":%llu,\"fallbacks\":%llu,\"shrinks\":%llu,\"grows\":%llu,\"pf_issued\":%llu,\"pf_used\":%llu,\"pf_wasted\":%llu,\"pf_late\":%llu,\"pf_pred_total\":%llu,\"pf_pred_hit\":%llu,\"pf_skipped\":%llu,\"sibling_loads\":%llu",
                     (unsigned long long) hits, (unsigned long long) misses, (unsigned long long) hits_dec, (unsigned long long) misses_dec, (unsigned long long) bytes,
                     (unsigned long long) evictions, (unsigned long long) evict_age_sum, (unsigned long long) resident_bytes, (unsigned long long) budget,
                     (unsigned long long) ns_prep, (unsigned long long) ns_read, (unsigned long long) ns_cpu, (unsigned long long) ns_total, (unsigned long long) graphs,
                     (unsigned long long) direct_jobs, (unsigned long long) bounce_jobs, (unsigned long long) repack_jobs, (unsigned long long) gpu_segments,
                     (unsigned long long) gpu_up_bytes, (unsigned long long) read_fallbacks, (unsigned long long) tune_shrinks, (unsigned long long) tune_grows,
                     (unsigned long long) pf_issued, (unsigned long long) pf_used, (unsigned long long) pf_wasted, (unsigned long long) pf_late,
                     (unsigned long long) pf_pred_total, (unsigned long long) pf_pred_hit, (unsigned long long) pf_skipped, (unsigned long long) sibling_loads);
            j = b;
            j += ",\"layer_hits\":[";
            for (size_t i = 0; i < layer_hits.size(); i++) { j += (i ? "," : "") + std::to_string(layer_hits[i]); }
            j += "],\"layer_misses\":[";
            for (size_t i = 0; i < layer_misses.size(); i++) { j += (i ? "," : "") + std::to_string(layer_misses[i]); }
            j += "]}";
        }
        fprintf(stderr, "moe-cache-stats %s\n", j.c_str());
        fflush(stderr);
    }

    ~state_t() {
        {
            std::lock_guard<std::mutex> lk(rq_mu);
            rq_stop = true;
        }
        rq_cv.notify_all();
        for (auto & t : readers) {
            t.join();
        }
        pool.reset();
#ifdef __linux__
        g_stats_stop = true;
        if (g_stats_pipe[1] >= 0) {
            close(g_stats_pipe[1]);          // wakes the snapshot thread, which then exits
        }
#endif
        if (trace_f) {
            fclose(trace_f);
        }
        {
            std::lock_guard<std::mutex> lk(pf_mu);
            pf_stop = true;
        }
        pf_cv.notify_all();
        for (auto & t : pf_threads) {
            t.join();
        }
        {
            std::lock_guard<std::mutex> lk(wmu);
            stop = true;
        }
        wcv.notify_all();
        for (auto & t : workers) {
            t.join();
        }
        if (cache_on) {
            save_profile();
        }
        if (!cache_on && stats) {
            fprintf(stderr, "moe-cache: pass-through graphs=%llu MUL_MAT_ID ops=%llu\n", (unsigned long long) graphs, (unsigned long long) ops);
        }
        if (cache_on && stats && hyb_groups) {
            fprintf(stderr, "moe-cache: hybrid: %llu layer steps, %llu of %llu picks computed on the GPU from its bank (%.1f%%), the rest on the CPU\n",
                    (unsigned long long) hyb_groups, (unsigned long long) hyb_hits, (unsigned long long) hyb_picks, hyb_picks ? 100.0 * hyb_hits / hyb_picks : 0.0);
            if (hyb_swaps) {
                fprintf(stderr, "moe-cache: hybrid: %llu experts swapped into the bank while running (%llu of them profile experts brought back; %.2f s uploading)\n",
                        (unsigned long long) hyb_swaps, (unsigned long long) hyb_reverts, hyb_swap_ns / 1e9);
            }
            if (hyb_drops) {
                fprintf(stderr, "moe-cache: hybrid: bank freed %llu times for long prompts, loaded again %llu times (%.2f s in total)\n",
                        (unsigned long long) hyb_drops, (unsigned long long) hyb_reloads, hyb_reload_ns / 1e9);
            }
        }
        if (cache_on && stats && gpu_dev) {
            if (coalesced) {
                fprintf(stderr, "moe-cache: %llu expert reads merged into their neighbours (MOE_CACHE_COALESCE_MIB)\n", (unsigned long long) coalesced.load());
            }
            fprintf(stderr, "moe-cache: long-prompt blocks computed on the GPU %llu (uploaded %.1f GB), fell back to the CPU %llu (of which too little free GPU memory: %llu)\n", (unsigned long long) gpu_segments, gpu_up_bytes / 1e9, (unsigned long long) gpu_fallbacks, (unsigned long long) gpu_nomem);
        }
        if (cache_on && stats) {
            const uint64_t h = hits, m = misses;
            if (read_fallbacks) {
                fprintf(stderr, "moe-cache: reads that needed the buffered fallback: %llu\n", (unsigned long long) read_fallbacks);
            }
            if (lookahead) {
                fprintf(stderr, "moe-cache: lookahead: prefetched %llu experts, used %llu, wasted %llu, late %llu; prediction recall %.1f%% (%llu of %llu), skipped %llu\n",
                        (unsigned long long) pf_issued, (unsigned long long) pf_used, (unsigned long long) pf_wasted, (unsigned long long) pf_late,
                        pf_pred_total ? 100.0 * pf_pred_hit / pf_pred_total : 0.0, (unsigned long long) pf_pred_hit, (unsigned long long) pf_pred_total, (unsigned long long) pf_skipped);
            }
            if (exec_on) {
                fprintf(stderr, "moe-cache: executor ran %llu expert ops itself%s\n", (unsigned long long) exec_ops,
                        exec_check ? (std::string(", checked ") + std::to_string(exec_checked.load()) + ", mismatches " + std::to_string(exec_mismatch.load())).c_str() : "");
            }
            if (exec_on) {
                for (int ty = 0; ty < 64; ty++) {
                    if (exec_type_ops[ty]) {
                        fprintf(stderr, "moe-cache: executor by type %-8s ops=%llu time=%.2fs expert data=%.1f GB (%.2f GB/s incl. waits)\n", ggml_type_name((ggml_type) ty),
                                (unsigned long long) exec_type_ops[ty], exec_type_ns[ty] / 1e9, exec_type_bytes[ty] / 1e9, exec_type_ns[ty] ? (double) exec_type_bytes[ty] / exec_type_ns[ty] : 0.0);
                    }
                }
            }
            if (route_mode) {
                fprintf(stderr, "moe-cache: approximate routing (%s): %llu low-ranked picks looked at, %llu given up, %llu replaced by a resident expert\n",
                        route_mode == 1 ? "skip" : "subst", (unsigned long long) route_considered, (unsigned long long) route_skipped, (unsigned long long) route_substituted);
            }
            fprintf(stderr, "moe-cache: reads straight into place %llu, via bounce buffer %llu, repacked %llu\n", (unsigned long long) direct_jobs, (unsigned long long) bounce_jobs, (unsigned long long) repack_jobs);
            fprintf(stderr, "moe-cache: time prep=%.2fs (of which reads %.2fs) cpu_compute=%.2fs total_in_backend=%.2fs\n", ns_prep / 1e9, ns_read / 1e9, ns_cpu / 1e9, ns_total / 1e9);
            fprintf(stderr, "moe-cache: graphs=%llu groups=%llu ops=%llu hits=%llu misses=%llu (hit %.1f%%) read=%.1f MB evictions=%llu resident=%.2f GiB%s\n",
                    (unsigned long long) graphs, (unsigned long long) groups, (unsigned long long) ops, (unsigned long long) h, (unsigned long long) m,
                    h + m ? 100.0 * h / (h + m) : 0.0, bytes / 1e6, (unsigned long long) evictions, resident_bytes / (double) (1u << 30),
                    auto_mode ? (std::string(" | auto: final budget ") + std::to_string(budget / 1073741824.0).substr(0, 5) + " GiB, shrinks " + std::to_string(tune_shrinks) + ", grows " + std::to_string(tune_grows)).c_str() : "");
        }
    }

    // run fn(0..n_io) on the workers and the caller, wait until all are done
    void run_parallel(const std::function<void(int)> & fn) {
        {
            std::lock_guard<std::mutex> lk(wmu);
            task = fn;
            busy = (int) workers.size();
            gen++;
        }
        wcv.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lk(wmu);
        dcv.wait(lk, [this] { return busy == 0; });
    }

    void worker_main(int id) {
        int seen = 0;
        for (;;) {
            std::function<void(int)> fn;
            {
                std::unique_lock<std::mutex> lk(wmu);
                wcv.wait(lk, [&] { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen = gen;
                fn = task;
            }
            fn(id);
            {
                std::lock_guard<std::mutex> lk(wmu);
                busy--;
            }
            dcv.notify_all();
        }
    }

    // read one expert slice into place (or through the bounce buffer); retries, then falls back to a buffered read, then stops
    void read_job(const io_job_t & j, char * bounce) {
        const size_t aoff = j.off / PAGE * PAGE;
        const size_t shift = j.off - aoff;
        const size_t len = round_up(shift + j.nb2);
        char * buf = j.direct ? j.dst - shift : bounce;   // direct: the aligned read lands exactly around the expert
        // the pages must exist before they are written (Windows); one extra page covers kernels that read a little past the end
        plat::mem_commit(j.direct ? j.dst - shift : j.dst, (j.direct ? len : j.nb2) + PAGE);
        static const bool prefault = [] { const char * e = getenv("MOE_CACHE_PREFAULT"); return e && atoi(e) != 0; }();
        if (prefault && j.direct) plat::mem_prefault(j.dst - shift, len);   // the pages of an evicted slot are gone: fault them here, not inside the read
        // direct read with retries; if it keeps failing, one more try through a normal buffered read; then stop (never serve wrong data)
        const bool inject_job = fault_every > 0 && (++fault_ctr % (uint64_t) fault_every) == 0;   // test switch: this read fails every direct attempt
        auto read_all = [&](const plat::file_t & rfd, bool inject) {
            size_t done = 0;
            int retries = 0;
            while (done < shift + j.nb2) {
                ptrdiff_t got;
                if (inject && inject_job) {
                    got = -1;
                    errno = EIO;
                } else {
                    got = plat::file_pread(rfd, buf + done, len - done, (uint64_t) (aoff + done));
                }
                if (got <= 0) {
                    if (++retries <= 3) {
                        plat::sleep_ms(20);
                        continue;
                    }
                    return false;
                }
                done += (size_t) got;
            }
            return true;
        };
        if (!read_all(fds[j.fdi], true)) {
            plat::file_t bfd;
            {
                std::lock_guard<std::mutex> lk(fds_buf_mu);
                if (!fds_buf[j.fdi].ok()) {
                    fds_buf[j.fdi] = plat::file_open(shard_paths[j.fdi].c_str(), false);
                }
                bfd = fds_buf[j.fdi];
            }
            if (!bfd.ok() || !read_all(bfd, false)) {
                fprintf(stderr, "moe-cache: read failed at offset %zu of %s: %s (disk error, or the model file changed after loading)\n", aoff, shard_paths[j.fdi].c_str(), plat::last_error().c_str());
                abort();
            }
            if (read_fallbacks++ == 0) {
                fprintf(stderr, "moe-cache: direct read failed, used a buffered read instead (output stays correct; this may be slower)\n");
            }
        }
        if (j.rtensor) {
            // repack this one expert into its place: use a copy of the tensor that describes a single expert
            ggml_tensor v = *j.rtensor;
            v.ne[2] = 1;
            v.ne[3] = 1;
            v.nb[2] = v.nb[1] * v.ne[1];
            v.nb[3] = v.nb[2];
            v.data = j.dst;
            v.view_src = nullptr;
            repack_buf->iface.set_tensor(repack_buf, &v, bounce + shift, 0, j.nb2);
            repack_jobs++;
        } else if (!j.direct) {
            memcpy(j.dst, bounce + shift, j.nb2);
            bounce_jobs++;
        } else {
            direct_jobs++;
        }
        bytes += len;
    }

    void do_reads(const std::vector<io_job_t> & jobs, size_t max_nb2) {
        std::atomic<size_t> next{0};
        auto fn = [&](int) {
            static thread_local char * bounce = nullptr;
            static thread_local size_t bounce_size = 0;
            const size_t need = round_up(max_nb2) + 2 * PAGE;
            if (bounce_size < need) {
                plat::aligned_free_page(bounce);
                bounce = (char *) plat::aligned_alloc_page(need);
                if (!bounce) {
                    abort();
                }
                bounce_size = need;
            }
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= jobs.size()) {
                    break;
                }
                read_job(jobs[i], bounce);
            }
        };
        if (jobs.size() <= 1) {
            fn(0);
        } else {
            run_parallel(fn);
        }
    }

    // ---- registration (at load time) ----
    int register_tensor(const ggml_tensor * t) {
        auto it = by_tensor.find(t);
        if (it != by_tensor.end()) {
            return it->second;
        }
        auto ft = file_tensors.find(t->name);
        if (ft == file_tensors.end() || t->ne[2] <= 1 || ft->second.second != t->nb[2] * (size_t) t->ne[2]) {
            // not the model MOE_CACHE_GGUF describes: keep this tensor as plain memory (the stock way) instead of stopping the server
            if (unmatched++ == 0) {
                fprintf(stderr, "moe-cache: tensor %s is not in the GGUF or has another size: it is loaded normally and not cached (is MOE_CACHE_GGUF the model you loaded?)\n", t->name);
            }
            return -1;
        }
        auto ti = std::make_unique<tinfo_t>();
        ti->name = t->name;
        ti->file_off = ft->second.first;
        ti->nb2 = t->nb[2];
        ti->n_expert = t->ne[2];
        ti->data = (char *) t->data;
        {
            auto sh = file_shard.find(t->name);
            ti->fdi = sh != file_shard.end() ? sh->second : 0;
            ti->tensor = const_cast<ggml_tensor *>(t);
            if (repack_buf) {
                ti->tensor->extra = nullptr;
                repack_buf->iface.init_tensor(repack_buf, ti->tensor);     // llama.cpp picks the layout, or none for types it cannot repack
                ti->repack = ti->tensor->extra != nullptr;
            }
            ti->direct = direct_ok && !ti->repack && ((uintptr_t) ti->data % PAGE) == (ti->file_off % PAGE);
        }
        sscanf(t->name, "blk.%d.", &ti->layer);
        ti->node0 = (int) nodes.size();
        nodes.resize(nodes.size() + (size_t) t->ne[2]);
        if (thp >= 0) {
            const uintptr_t a = (uintptr_t) ti->data / PAGE * PAGE;
            const uintptr_t b = ((uintptr_t) ti->data + ggml_nbytes(t) + PAGE - 1) / PAGE * PAGE;
            plat::mem_hugepage((void *) a, b - a, thp != 0);
        }
        const int idx = (int) tensors.size();
        if (ti->layer >= 0) {
            layer_tensors[ti->layer].push_back(idx);
        }
        max_nb2_all = std::max(max_nb2_all, ti->nb2);
        tensors.push_back(std::move(ti));
        by_tensor[t] = idx;
        return idx;
    }

    void save_profile() {
        if (profile_path.empty() || session.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lk(mu);
        int nl = std::max<int>(learned.n_layers, (int) session.size());
        int ne = learned.n_exp;
        for (const auto & v : session) {
            ne = std::max<int>(ne, (int) v.size());
        }
        std::vector<float> out((size_t) nl * ne, 0.0f);
        for (int l = 0; l < nl; l++) {
            for (int e = 0; e < ne; e++) {
                double v = learned.at(l, e) * profile_decay;
                if (l < (int) session.size() && e < (int) session[l].size()) {
                    v += session[l][e];
                }
                out[(size_t) l * ne + e] = (float) v;
            }
        }
        const size_t slash = profile_path.find_last_of("/\\");
        if (slash != std::string::npos && slash > 0) {
            const std::string dir = profile_path.substr(0, slash);
            size_t pos = 0;
            while ((pos = dir.find_first_of("/\\", pos + 1)) != std::string::npos) {
                plat::make_dir(dir.substr(0, pos).c_str());
            }
            plat::make_dir(dir.c_str());
        }
        const std::string tmp = profile_path + ".tmp";
        FILE * f = fopen_private(tmp.c_str());
        if (!f) {
            fprintf(stderr, "moe-cache: cannot write profile %s\n", tmp.c_str());
            return;
        }
        const int32_t h[2] = { nl, ne };
        fwrite("XUSE", 1, 4, f);
        fwrite(h, sizeof(int32_t), 2, f);
        fwrite(out.data(), sizeof(float), out.size(), f);
        fwrite(&fingerprint, 8, 1, f);
        fclose(f);
        plat::replace_file(tmp.c_str(), profile_path.c_str());
    }

    // warm start: read the most used experts of the profile into place, most used last (so it is the most recently used)
    void preload() {
        std::lock_guard<std::mutex> lk(mu);
        if (preload_done) {
            return;
        }
        preload_done = true;
        std::map<int, std::vector<int>> by_layer;
        size_t total = 0;
        for (size_t i = 0; i < tensors.size(); i++) {
            if (tensors[i]->layer >= 0) {
                by_layer[tensors[i]->layer].push_back((int) i);
                total += tensors[i]->nb2 * (size_t) tensors[i]->n_expert;
            }
        }
        if (by_layer.empty()) {
            return;
        }
        // no profile: if every CPU expert fits in the cache, load them all in layer order (a cache filled expert by expert on demand
        // decodes about 15 % slower than one filled in order; MOE_CACHE_PRELOAD_ALL=0 turns this off)
        if (!preload_on) {
            const char * pa = getenv("MOE_CACHE_PRELOAD_ALL");
            if ((pa && strcmp(pa, "0") == 0) || total > budget * 0.97) {
                return;
            }
        }
        double frac = preload_frac;
        if (frac < 0) {
            frac = total <= budget * 0.97 ? 1.0 : 0.7;
        }
        const double limit = frac * (double) budget * 0.97;
        struct cand_t { int l, e; double c; };
        std::vector<cand_t> cand;
        for (auto & kv : by_layer) {
            const int ne = (int) tensors[kv.second[0]]->n_expert;
            for (int e = 0; e < ne; e++) {
                cand.push_back({ kv.first, e, learned.at(kv.first, e) });
            }
        }
        std::stable_sort(cand.begin(), cand.end(), [](const cand_t & a, const cand_t & b) { return a.c > b.c; });
        std::vector<io_job_t> jobs;
        size_t acc = 0, max_nb2 = 0, n_taken = 0;
        std::vector<cand_t> take;
        for (const auto & c : cand) {
            size_t bytes = 0;
            for (int ti : by_layer[c.l]) {
                bytes += tensors[ti]->nb2;
            }
            if ((double) (acc + bytes) > limit) {
                break;
            }
            acc += bytes;
            take.push_back(c);
        }
        for (size_t i = take.size(); i-- > 0;) {
            for (int ti_i : by_layer[take[i].l]) {
                tinfo_t & ti = *tensors[ti_i];
                const int s = ti.node0 + take[i].e;
                if (nodes[s].resident) {
                    continue;
                }
                nodes[s].resident = true;
                nodes[s].epoch = 0;
                resident_bytes += ti.nb2;
                push_mru(s);
                max_nb2 = std::max(max_nb2, ti.nb2);
                jobs.push_back({ ti.data + (size_t) take[i].e * ti.nb2, ti.file_off + (size_t) take[i].e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr });
                n_taken++;
            }
        }
        const auto r0 = std::chrono::steady_clock::now();
        do_reads(jobs, max_nb2);
        fprintf(stderr, "moe-cache: warm start loaded %zu expert tensors (%.2f GiB, %.0f%% of the cache) in %.1f s\n", n_taken, acc / (double) (1u << 30),
                100.0 * acc / (double) budget, std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count());
    }

    // auto mode: keep the real anonymous memory of the process under (limit - margin). Called every 32 groups and after every 256 MiB of reads.
    void tune() {
        if (max_budget == 0) {
            size_t tot = 0, big = 0;
            for (const auto & t : tensors) {
                tot += t->nb2 * (size_t) t->n_expert;
                big = std::max(big, t->nb2 * (size_t) t->n_expert);
            }
            max_budget = tot;
            min_budget = std::min(tot, 3 * big);
            budget = std::min(budget, max_budget);
            const size_t tgt = limit_bytes > margin ? limit_bytes - margin : limit_bytes / 2;
            if (min_budget + ((size_t) 512 << 20) > tgt) {
                fprintf(stderr, "moe-cache: warning: the memory limit (%.1f GiB) is very small for this model (the cache needs at least %.1f GiB); expect out-of-memory kills\n",
                        limit_bytes / 1073741824.0, min_budget / 1073741824.0);
            }
        }
        const size_t rss = read_rss_anon();
        if (!rss) {
            return;
        }
        const size_t target = limit_bytes > margin ? limit_bytes - margin : limit_bytes / 2;
        // bytes of expert data that are really in memory (pending ones are marked but not read yet)
        const size_t have = resident_bytes > pending_bytes ? resident_bytes - pending_bytes : 0;
        if (rss > target) {
            // over the target: the new budget is what is loaded now minus the excess, so eviction starts at once
            const size_t over = rss - target + ((size_t) 64 << 20);
            size_t nb = have > over ? have - over : 0;
            nb = std::min(std::max(nb, min_budget), budget);
            if (nb < budget) {
                budget = nb;
                tune_shrinks++;
            }
            while (resident_bytes > budget) {
                int v = lru;
                while (v >= 0 && (nodes[v].epoch == epoch || nodes[v].loading)) {
                    v = nodes[v].prev;
                }
                if (v < 0) {
                    break;
                }
                evict(v);
            }
        } else if (resident_bytes + ((size_t) 512 << 20) >= budget && rss + ((size_t) 512 << 20) < target && budget < max_budget) {
            // the cache is full and there is clearly room: grow, at most 1 GiB per step
            const size_t g = std::min<size_t>(target - rss - ((size_t) 256 << 20), (size_t) 1 << 30);
            budget = std::min(max_budget, budget + g);
            tune_grows++;
        }
    }

    // ---- LRU over resident experts ----
    void unlink(int s) {
        node_t & n = nodes[s];
        if (n.prev >= 0) { nodes[n.prev].next = n.next; } else { mru = n.next; }
        if (n.next >= 0) { nodes[n.next].prev = n.prev; } else { lru = n.prev; }
        n.prev = n.next = -1;
    }

    // scan-resistant insertion: the expert becomes the first candidate for eviction, so a pass over the whole layer recycles its own slots
    // instead of pushing out the experts that decode steps keep using
    void push_lru(int s) {
        node_t & n = nodes[s];
        n.next = -1;
        n.prev = lru;
        if (lru >= 0) { nodes[lru].next = s; }
        lru = s;
        if (mru < 0) { mru = s; }
    }

    void push_mru(int s) {
        node_t & n = nodes[s];
        n.prev = -1;
        n.next = mru;
        if (mru >= 0) { nodes[mru].prev = s; }
        mru = s;
        if (lru < 0) { lru = s; }
    }

    // give the physical pages of one expert back to the system (pages shared with a neighbour expert stay)
    void evict(int s) {
        // find the tensor that owns node s
        size_t lo_t = 0, hi_t = tensors.size();
        while (lo_t + 1 < hi_t) {
            const size_t mid = (lo_t + hi_t) / 2;
            if (tensors[mid]->node0 <= s) { lo_t = mid; } else { hi_t = mid; }
        }
        tinfo_t & ti = *tensors[lo_t];
        const size_t e = (size_t) (s - ti.node0);
        const uintptr_t start = (uintptr_t) ti.data + e * ti.nb2;
        const uintptr_t a = (start + PAGE - 1) / PAGE * PAGE;
        const uintptr_t b = (start + ti.nb2) / PAGE * PAGE;
        if (b > a) {
            plat::mem_decommit((void *) a, b - a);
        }
        evict_age_sum += epoch - nodes[s].epoch;   // how many groups ago this expert was last used
        if (nodes[s].prefetched) {
            nodes[s].prefetched = false;
            pf_wasted++;
        }
        unlink(s);
        nodes[s].resident = false;
        resident_bytes -= ti.nb2;
        evictions++;
    }

    // make sure all experts used by the ops of one group (same ids tensor) are resident
    // returns the number of reads handed to the reader threads (async mode), otherwise 0
    size_t prepare_group(const std::vector<int> & tidx, const std::vector<int> & cnt, bool async = false) {
        std::unique_lock<std::mutex> lk(mu);
        epoch++;
        std::vector<int> wait_list;       // experts a lookahead read is still filling
        if (lookahead && cur_decode) {
            auto pf = pred_flags.find(tensors[tidx[0]]->layer);
            if (pf != pred_flags.end()) {
                for (size_t e = 0; e < cnt.size() && e < pf->second.size(); e++) {
                    if (cnt[e]) {
                        pf_pred_total++;
                        pf_pred_hit += pf->second[e] ? 1 : 0;
                    }
                }
                pred_flags.erase(pf);
            }
        }
        groups++;
        if (auto_mode && (groups % (async ? 4 : 32)) == 1) {
            tune();
        }
        if (!profile_path.empty()) {
            const int layer = tensors[tidx[0]]->layer;
            if (layer >= 0) {
                if ((int) session.size() <= layer) {
                    session.resize((size_t) layer + 1);
                }
                auto & row = session[layer];
                if (row.size() < cnt.size()) {
                    row.resize(cnt.size(), 0.0);
                }
                for (size_t e = 0; e < cnt.size(); e++) {
                    row[e] += cnt[e];
                }
            }
        }
        std::vector<io_job_t> jobs;
        size_t max_nb2 = 0;
        // The gate, up and down tensors of a layer use the same experts. Some models (gpt-oss: bias operations in between) hand them to us in
        // separate calls; then load the siblings' experts in the same parallel batch now, instead of three small batches one after the other.
        std::vector<int> tl = tidx;
        const size_t n_counted = tidx.size();
        if (siblings) {
            auto lt = layer_tensors.find(tensors[tidx[0]]->layer);
            if (lt != layer_tensors.end()) {
                for (int t : lt->second) {
                    if (std::find(tl.begin(), tl.end(), t) == tl.end() && tensors[t]->n_expert == tensors[tidx[0]]->n_expert) {
                        tl.push_back(t);
                    }
                }
            }
        }
        for (size_t q = 0; q < tl.size(); q++) {
            const int ti_i = tl[q];
            const bool counted = q < n_counted;     // sibling tensors are loaded now, but their hits and misses are counted when their own call comes
            tinfo_t & ti = *tensors[ti_i];
            max_nb2 = std::max(max_nb2, ti.nb2);
            for (int64_t e = 0; e < ti.n_expert; e++) {
                if (!cnt[(size_t) e]) {
                    continue;
                }
                const int s = ti.node0 + (int) e;
                node_t & n = nodes[s];
                if (!counted) {
                    if (n.resident) {
                        if (n.loading) { wait_list.push_back(s); }
                        unlink(s);
                        push_mru(s);
                        n.epoch = epoch;
                        continue;
                    }
                    sibling_loads++;
                    while (resident_bytes + ti.nb2 > budget) {
                        int v = lru;
                        for (int k = 0; k < 100000 && v >= 0 && (nodes[v].epoch == epoch || nodes[v].loading); k++) {
                            v = nodes[v].prev;
                        }
                        if (v < 0 || nodes[v].epoch == epoch || nodes[v].loading) {
                            break;     // no room without evicting what this group needs: leave it to the sibling's own call
                        }
                        evict(v);
                    }
                    if (resident_bytes + ti.nb2 > budget) {
                        continue;
                    }
                    n.resident = true;
                    n.epoch = epoch;
                    resident_bytes += ti.nb2;
                    if (cur_scan) { push_lru(s); } else { push_mru(s); }
                    { io_job_t jb{ ti.data + (size_t) e * ti.nb2, ti.file_off + (size_t) e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr }; jb.node = s; if (async) { __atomic_store_n(&n.loading, true, __ATOMIC_RELEASE); } jobs.push_back(jb); }
                    pending_bytes += ti.nb2;
                    if (!async && auto_mode && pending_bytes >= ((size_t) 256 << 20)) {
                        flush_reads(jobs, max_nb2);
                    }
                    continue;
                }
                if (ti.layer >= 0) {
                    if ((size_t) ti.layer >= layer_hits.size()) {
                        layer_hits.resize((size_t) ti.layer + 1, 0);
                        layer_misses.resize((size_t) ti.layer + 1, 0);
                    }
                    (n.resident ? layer_hits : layer_misses)[(size_t) ti.layer]++;
                }
                if (n.resident) {
                    hits++;
                    if (cur_decode) { hits_dec++; }
                    if (n.prefetched) {
                        n.prefetched = false;
                        pf_used++;
                        if (n.loading) { pf_late++; }
                    }
                    if (n.loading) { wait_list.push_back(s); }
                    if (!cur_scan) {          // a prompt batch uses everything: it does not make an expert "recently used"
                        unlink(s);
                        push_mru(s);
                    }
                    n.epoch = epoch;
                    continue;
                }
                misses++;
                if (cur_decode) { misses_dec++; }
                while (resident_bytes + ti.nb2 > budget) {
                    int v = lru;
                    for (int k = 0; k < 100000 && v >= 0 && (nodes[v].epoch == epoch || nodes[v].loading); k++) {
                        v = nodes[v].prev;
                    }
                    if (v < 0 || nodes[v].epoch == epoch || nodes[v].loading) {
                        fprintf(stderr, "moe-cache: cache too small for one op, raise MOE_CACHE_SIZE_GIB\n");
                        abort();
                    }
                    evict(v);
                }
                n.resident = true;
                n.epoch = epoch;
                resident_bytes += ti.nb2;
                if (cur_scan) { push_lru(s); } else { push_mru(s); }
                { io_job_t jb{ ti.data + (size_t) e * ti.nb2, ti.file_off + (size_t) e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr }; jb.node = s; if (async) { __atomic_store_n(&n.loading, true, __ATOMIC_RELEASE); } jobs.push_back(jb); }
                pending_bytes += ti.nb2;
                if (!async && auto_mode && pending_bytes >= ((size_t) 256 << 20)) {
                    flush_reads(jobs, max_nb2);
                }
            }
        }
        if (async) {
            // hand the reads to the reader threads and return: the executor waits for each expert's flag when it needs it
            const size_t njobs = jobs.size();
            if (!jobs.empty()) {
                {
                    std::lock_guard<std::mutex> ql(rq_mu);
                    for (const auto & j : jobs) {
                        rq.push_back(j);
                    }
                }
                rq_cv.notify_all();
            }
            return njobs;
        }
        flush_reads(jobs, max_nb2);
        for (int s : wait_list) {
            while (nodes[s].loading) {
                load_cv.wait(lk);
            }
        }
        return 0;
    }

    // read the queued experts, then (auto mode) look at the real memory use again
    void flush_reads(std::vector<io_job_t> & jobs, size_t max_nb2) {
        const bool did_reads = !jobs.empty();
        if (coalesce_bytes && jobs.size() > 1) {
            // long prompts need nearly every expert of a layer, and neighbouring experts of a tensor are neighbours in the file and in memory:
            // one large read instead of many half-megabyte ones (the drive reaches its speed only with large or many parallel reads)
            std::sort(jobs.begin(), jobs.end(), [](const io_job_t & a, const io_job_t & b) { return a.fdi != b.fdi ? a.fdi < b.fdi : a.off < b.off; });
            std::vector<io_job_t> merged;
            merged.reserve(jobs.size());
            for (const auto & j : jobs) {
                if (!merged.empty()) {
                    io_job_t & m = merged.back();
                    if (m.direct && j.direct && !m.rtensor && !j.rtensor && m.fdi == j.fdi && m.off + m.nb2 == j.off && m.dst + m.nb2 == j.dst &&
                        m.nb2 + j.nb2 <= coalesce_bytes) {
                        m.nb2 += j.nb2;
                        coalesced++;
                        continue;
                    }
                }
                merged.push_back(j);
            }
            jobs.swap(merged);
            for (const auto & j : jobs) {
                max_nb2 = std::max(max_nb2, j.nb2);
            }
        }
        if (!jobs.empty()) {
            const auto r0 = std::chrono::steady_clock::now();
            do_reads(jobs, max_nb2);
            ns_read += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - r0).count();
            jobs.clear();
        }
        pending_bytes = 0;
        if (auto_mode && did_reads) {   // nothing was read: memory did not change (prepare_group re-checks every 32 groups anyway); reading /proc each layer cost ~2 ms per token
            tune();
        }
    }
};

state_t & S() {
    static state_t s;
    return s;
}

// ---- buffer type ----

struct moe_cache_buffer_ctx {
    void * base;
    size_t size;
};

const char * buft_name(ggml_backend_buffer_type_t) {
    return "MOE_CACHE";
}

void buf_free(ggml_backend_buffer_t buffer) {
    auto * c = (moe_cache_buffer_ctx *) buffer->context;
    plat::mem_release(c->base, c->size);
    delete c;
}

void * buf_base(ggml_backend_buffer_t buffer) {
    return ((moe_cache_buffer_ctx *) buffer->context)->base;
}

void buf_memset(ggml_backend_buffer_t, ggml_tensor * t, uint8_t v, size_t off, size_t size) {
    plat::mem_commit((char *) t->data + off, size);
    memset((char *) t->data + off, v, size);
}

void buf_set(ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t off, size_t size) {
    state_t & s = S();
    if (s.cache_on && t->ne[2] > 1 && strncmp(t->name, "blk.", 4) == 0 && off == 0 && size == ggml_nbytes(t)) {
        std::lock_guard<std::mutex> lk(s.mu);
        if (s.register_tensor(t) >= 0) {
            return;
        }
    }
    plat::mem_commit((char *) t->data + off, size);
    memcpy((char *) t->data + off, data, size);
}

void buf_get(ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t off, size_t size) {
    memcpy(data, (const char *) t->data + off, size);
}

void buf_clear(ggml_backend_buffer_t buffer, uint8_t v) {
    auto * c = (moe_cache_buffer_ctx *) buffer->context;
    if (S().cache_on) {
        return;   // the buffer is address space only; experts get their data from the file
    }
    memset(c->base, v, c->size);
}

// expert tensors get 12 KiB of padding so that init_tensor can move the data to an address that matches its file offset (mod 4096)
bool is_expert_tensor(const ggml_tensor * t) {
    return S().cache_on && S().direct_ok && t->ne[2] > 1 && strncmp(t->name, "blk.", 4) == 0;
}

ggml_status buf_init_tensor(ggml_backend_buffer_t, ggml_tensor * t) {
    if (is_expert_tensor(t)) {
        auto ft = S().file_tensors.find(t->name);
        if (ft != S().file_tensors.end()) {
            uintptr_t d = (uintptr_t) t->data + PAGE;
            d += ((ft->second.first % PAGE) + PAGE - (d % PAGE)) % PAGE;
            t->data = (void *) d;
        }
    }
    return GGML_STATUS_SUCCESS;
}

size_t buft_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) {
    return ggml_nbytes(t) + (is_expert_tensor(t) ? 3 * PAGE : 0);
}

const ggml_backend_buffer_i g_buf_i = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ buf_init_tensor,
    /* .memset_tensor = */ buf_memset,
    /* .set_tensor    = */ buf_set,
    /* .get_tensor    = */ buf_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ buf_clear,
    /* .reset         = */ nullptr,
};

ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    size = round_up(std::max<size_t>(size, 1));
    void * p = plat::mem_reserve(size);
    if (!p) {
        return nullptr;
    }
    if (!S().cache_on) {
        plat::mem_commit(p, size);   // pass-through mode: the weights are copied in, all pages are needed
    }
    return ggml_backend_buffer_init(buft, g_buf_i, new moe_cache_buffer_ctx{ p, size }, size);
}

size_t buft_align(ggml_backend_buffer_type_t) {
    return 64;
}

// MOE_CACHE_GPU_CACHE=1: llama.cpp's own GPU expert cache (--moe-cache-mib) only caches experts of layers whose weights sit in a host buffer and
// uploads them by reading the weight memory. Our buffer then says "host", and the upload is intercepted (ggml_backend_tensor_set_async below) to
// make the experts being read resident first.
bool gpu_cache_mode() {
    static const bool on = [] { const char * e = getenv("MOE_CACHE_GPU_CACHE"); return e && atoi(e) != 0; }();
    return on;
}

bool buft_is_host(ggml_backend_buffer_type_t) {
    return gpu_cache_mode();
}

// make the experts that cover [p, p+size) resident, if p points into one of our expert tensors
void gpu_cache_ensure(const void * p, size_t size) {
    if (size == 0) {
        return;
    }
    state_t & s = S();
    if (!s.cache_on) {
        return;
    }
    struct range_t { uintptr_t a, b; int idx; };
    static std::mutex rmu;
    static std::vector<range_t> ranges;
    static size_t built_for = (size_t) -1;
    const uintptr_t a = (uintptr_t) p;
    int idx = -1;
    {
        std::lock_guard<std::mutex> lk(rmu);
        if (built_for != s.tensors.size()) {
            ranges.clear();
            for (size_t i = 0; i < s.tensors.size(); i++) {
                const tinfo_t & ti = *s.tensors[i];
                if (ti.data && ti.nb2 && ti.n_expert > 0) {
                    ranges.push_back({ (uintptr_t) ti.data, (uintptr_t) ti.data + (size_t) ti.n_expert * ti.nb2, (int) i });
                }
            }
            std::sort(ranges.begin(), ranges.end(), [](const range_t & x, const range_t & y) { return x.a < y.a; });
            built_for = s.tensors.size();
        }
        auto it = std::upper_bound(ranges.begin(), ranges.end(), a, [](uintptr_t v, const range_t & r) { return v < r.a; });
        if (it != ranges.begin()) {
            --it;
            if (a >= it->a && a < it->b) {
                idx = it->idx;
            }
        }
    }
    if (idx < 0) {
        return;
    }
    tinfo_t & ti = *s.tensors[(size_t) idx];
    const size_t e0 = (a - (uintptr_t) ti.data) / ti.nb2;
    size_t e1 = (a + size - 1 - (uintptr_t) ti.data) / ti.nb2;
    e1 = std::min(e1, (size_t) ti.n_expert - 1);
    std::vector<int> cnt((size_t) ti.n_expert, 0);
    for (size_t e = e0; e <= e1; e++) {
        cnt[e] = 1;
    }
    s.prepare_group(std::vector<int>{ idx }, cnt, false);
}

ggml_backend_dev_t g_dev = nullptr;

ggml_backend_buffer_type_t get_buft() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ buft_name,
            /* .alloc_buffer   = */ buft_alloc,
#if GGML_BACKEND_API_VERSION >= 3
            /* .alloc_buffer_n = */ nullptr,      // default: buft_alloc + linear allocator
#endif
            /* .get_alignment  = */ buft_align,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ buft_alloc_size,
#if GGML_BACKEND_API_VERSION >= 3
            /* .get_alloc_size_n = */ nullptr,    // default: per-tensor buft_alloc_size
#endif
            /* .is_host        = */ buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    buft.device = g_dev;
    return &buft;
}

// ---- backend ----

// The CPU backend is reached through the registry only, so this works with builds that load their backends as separate libraries.
struct cpu_api_t {
    ggml_backend_dev_t dev = nullptr;
    void              (*set_n_threads)(ggml_backend_t, int) = nullptr;
    ggml_threadpool_t (*tp_new)(const ggml_threadpool_params *) = nullptr;
    void              (*tp_free)(ggml_threadpool_t) = nullptr;
    void              (*set_tp)(ggml_backend_t, ggml_threadpool_t) = nullptr;
};

cpu_api_t & CPU() {
    static cpu_api_t a = [] {
        cpu_api_t r;
        r.dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!r.dev) {
            fprintf(stderr, "moe-cache: no CPU backend is loaded\n");
            abort();
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(r.dev);
        r.set_n_threads = (decltype(r.set_n_threads)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        r.tp_new  = (decltype(r.tp_new))  ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        r.tp_free = (decltype(r.tp_free)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        r.set_tp  = (decltype(r.set_tp))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        return r;
    }();
    return a;
}

// CPUs to pin the plugin's compute threads to, best first: one thread per performance core, then their hyper-threading siblings.
// Only on hybrid Intel CPUs (performance + efficiency cores), where a worker on an efficiency core or on a busy sibling slows every layer.
// MOE_CACHE_CPU_PIN=off disables it; MOE_CACHE_CPU_MASK=hex picks the CPUs yourself (bit i = CPU i).
std::vector<int> pin_cpus() {
    std::vector<int> out;
#ifdef __linux__
    const char * off = getenv("MOE_CACHE_CPU_PIN");
    if (off && strcmp(off, "off") == 0) {
        return out;
    }
    if (const char * m = getenv("MOE_CACHE_CPU_MASK")) {
        const size_t n = strlen(m);
        for (size_t i = 0; i < n; i++) {
            const char ch = m[n - 1 - i];
            const int v = ch >= '0' && ch <= '9' ? ch - '0' : (ch | 32) >= 'a' && (ch | 32) <= 'f' ? (ch | 32) - 'a' + 10 : 0;
            for (int b = 0; b < 4; b++) {
                if ((v & (1 << b)) && i * 4 + b < GGML_MAX_N_THREADS) {   // cpumask has GGML_MAX_N_THREADS entries
                    out.push_back((int) (i * 4 + b));
                }
            }
        }
        return out;
    }
    auto parse = [](const std::string & txt) {
        std::vector<int> r;
        size_t i = 0;
        while (i < txt.size()) {
            int a = atoi(txt.c_str() + i), b = a;
            while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
            if (i < txt.size() && txt[i] == '-') {
                i++;
                b = atoi(txt.c_str() + i);
                while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
            }
            for (int c = a; c <= b && c < GGML_MAX_N_THREADS; c++) { r.push_back(c); }
            while (i < txt.size() && !isdigit((unsigned char) txt[i])) { i++; }
        }
        return r;
    };
    auto slurp = [](const std::string & fn) {
        std::string t;
        if (FILE * f = fopen(fn.c_str(), "r")) {
            char b[256];
            while (fgets(b, sizeof(b), f)) { t += b; }
            fclose(f);
        }
        return t;
    };
    const std::vector<int> pcpus = parse(slurp("/sys/devices/cpu_core/cpus"));   // exists only on hybrid Intel CPUs
    std::vector<int> primary, secondary;
    for (int c : pcpus) {
        const std::vector<int> sib = parse(slurp("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
        (!sib.empty() && sib[0] != c ? secondary : primary).push_back(c);
    }
    out = primary;
    out.insert(out.end(), secondary.begin(), secondary.end());
#endif
    return out;
}

struct moe_cache_backend_ctx {
    ggml_backend_t     cpu = nullptr;
    ggml_threadpool_t  tp = nullptr;
    int                tp_n = 0;
    int                n_threads = 6;
    ggml_backend_t     gpu = nullptr;
    ggml_gallocr_t     galloc = nullptr;
    bool               gpu_failed = false;
    bool               pin_reported = false;
    // MOE_CACHE_HYBRID=MiB: a bank of the most used experts in GPU memory (from the usage profile); per weight tensor: bank tensor and slot of each expert
    bool               hb_tried = false;
    ggml_context *     hb_ctx = nullptr;
    ggml_backend_buffer_t hb_buf = nullptr;
    bool               hb_dropped = false;   // the bank was freed so a long prompt's GPU block fits; it is loaded again at the next decode step
    bool               seg_nomem = false;    // the last gpu_segment() call failed for lack of free GPU memory
    // MOE_CACHE_HYBRID_ADAPT=N,K (default 8,16; 0 = off): every N tokens up to K of the experts the bank missed most replace the coldest expert of their layer's bank
    struct hb_layer_t {
        std::vector<int>   tidx;       // weight tensors of this layer that have a bank (gate, up, down)
        std::vector<int>   slot_exp;   // expert held by each slot
        std::vector<float> score;      // decayed hits per slot
        std::vector<float> miss;       // misses per expert since the last swap
        std::vector<char>  base;       // the experts the profile chose for this layer (the shadow bank)
    };
    // MOE_CACHE_HYBRID_DUEL=W (default 32, 0 = off): count what the profile's own bank would have served (the shadow) next to the real bank over the last W
    // tokens; while the shadow is ahead (the session matches the profile) no new swaps, and profile experts come back instead
    int                duel_w = 32;
    int                ha_cur = 0, hb_cur = 0;
    std::deque<std::pair<int, int>> duel;
    std::map<int, hb_layer_t> hbl;
    int                hb_first = -1;          // first layer with a bank: one token has passed when it comes round again
    uint64_t           hb_tok = 0;
    int                ad_every = 8, ad_swaps = 16;
    std::unordered_map<int, std::pair<ggml_tensor *, std::vector<int>>> hb;
};

const char * be_name(ggml_backend_t) {
    return "MOE_CACHE";
}

void be_free(ggml_backend_t backend) {
    auto * c = (moe_cache_backend_ctx *) backend->context;
    if (c->cpu) {
        ggml_backend_free(c->cpu);
    }
    if (c->tp && CPU().tp_free) {
        CPU().tp_free(c->tp);
    }
    if (c->galloc) {
        ggml_gallocr_free(c->galloc);
    }
    if (c->gpu) {
        ggml_backend_free(c->gpu);
    }
    delete c;
    delete backend;
}

// Run the expert block (MUL_MAT_ID / GLU nodes first..last) on the GPU: clone the nodes into a small graph, upload the weights and inputs
// to a reusable staging area, compute there and copy the last node's result back. Returns false (nothing changed) if anything does not fit.
bool gpu_segment(moe_cache_backend_ctx * c, state_t & s, ggml_cgraph * cg, int first, int last, std::vector<ggml_tensor *> & seg) {
    const int n_all = ggml_graph_n_nodes(cg);
    std::unordered_set<const ggml_tensor *> inside;
    for (int i = first; i <= last; i++) {
        ggml_tensor * n = ggml_graph_node(cg, i);
        if (n->op != GGML_OP_MUL_MAT_ID && n->op != GGML_OP_GLU) {
            return false;
        }
        inside.insert(n);
        seg.push_back(n);
    }
    ggml_tensor * out = seg.back();
    if (!ggml_is_contiguous(out)) {
        return false;
    }
    // inputs must exist already (not produced by another node of this graph) and be plain F32 / I32 / weights
    std::unordered_set<const ggml_tensor *> produced;
    for (int i = 0; i < n_all; i++) {
        produced.insert(ggml_graph_node(cg, i));
    }
    for (ggml_tensor * n : seg) {
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            const ggml_tensor * src = n->src[k];
            if (!src || inside.count(src)) {
                continue;
            }
            if (produced.count(src) || src->view_src) {
                return false;
            }
            const auto it = s.by_tensor.find(src);
            if (it != s.by_tensor.end()) {
                if (s.tensors[it->second]->repack) {
                    return false;
                }
            } else if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32) {
                return false;
            }
        }
    }
    if (!c->gpu) {
        c->gpu = ggml_backend_dev_init(s.gpu_dev, nullptr);
        if (!c->gpu) {
            c->gpu_failed = true;
            return false;
        }
        c->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->gpu));
    }
    ggml_init_params ip = { ggml_tensor_overhead() * 4 * (size_t) (seg.size() + 8) + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    std::unordered_map<const ggml_tensor *, ggml_tensor *> clone;
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> uploads;
    auto ext = [&](const ggml_tensor * t) {
        auto it = clone.find(t);
        if (it != clone.end()) {
            return it->second;
        }
        ggml_tensor * nt = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne);
        ggml_set_input(nt);
        clone[t] = nt;
        uploads.emplace_back(t, nt);
        return nt;
    };
    for (ggml_tensor * n : seg) {
        ggml_tensor * nn = ggml_new_tensor(ctx, n->type, GGML_MAX_DIMS, n->ne);
        nn->op = n->op;
        memcpy(nn->op_params, n->op_params, sizeof(nn->op_params));
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (n->src[k]) {
                nn->src[k] = inside.count(n->src[k]) ? clone[n->src[k]] : ext(n->src[k]);
            }
        }
        clone[n] = nn;
    }
    ggml_tensor * gout = clone[out];
    ggml_set_output(gout);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, gout);
    {
        // a whole layer of expert weights is uploaded for this block: with a GPU expert cache next to it the card can be full.
        // Check first and keep the block on the CPU instead of letting the allocation (or the CUDA pool) fail hard.
        size_t need = 0;
        for (auto & u : uploads) {
            need += ggml_nbytes(u.second);
        }
        for (ggml_tensor * n : seg) {
            need += ggml_nbytes(clone[n]);
        }
        const size_t held = ggml_gallocr_get_buffer_size(c->galloc, 0);       // already allocated for an earlier block: reused
        const size_t extra = need > held ? need - held : 0;
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(s.gpu_dev, &fr, &tot);
        const size_t headroom = (size_t) 256 << 20;                              // pool buffers, graph, fragmentation
        if (tot > 0 && fr < extra + headroom) {
            c->seg_nomem = true;
            if (!c->hb_buf && s.gpu_nomem++ == 0) {
                fprintf(stderr, "moe-cache: a long-prompt block needs %.0f MiB of GPU memory but only %.0f MiB are free: it stays on the CPU "
                                "(a smaller --moe-cache-mib or --gpu-margin leaves room; about %.0f MiB per layer for this model)\n",
                        (extra + headroom) / 1048576.0, fr / 1048576.0, need / 1048576.0);
            }
            ggml_free(ctx);
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(c->galloc, g)) {
        ggml_free(ctx);
        c->gpu_failed = true;
        return false;
    }
    std::vector<uint8_t> pack;
    for (auto & u : uploads) {
        const ggml_tensor * t = u.first;
        ggml_tensor * d = u.second;
        if (ggml_is_contiguous(t)) {
            ggml_backend_tensor_set(d, t->data, 0, ggml_nbytes(d));
            s.gpu_up_bytes += ggml_nbytes(d);
        } else {
            // activations / ids that are views: pack row by row
            const size_t es = ggml_type_size(t->type);
            if (t->nb[0] != es) {
                ggml_free(ctx);
                return false;
            }
            pack.assign(ggml_nbytes(d), 0);
            size_t o = 0;
            for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
                for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                    for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                        memcpy(pack.data() + o, (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], (size_t) t->ne[0] * es);
                        o += (size_t) t->ne[0] * es;
                    }
                }
            }
            ggml_backend_tensor_set(d, pack.data(), 0, ggml_nbytes(d));
        }
    }
    const enum ggml_status st = ggml_backend_graph_compute(c->gpu, g);
    ggml_backend_synchronize(c->gpu);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(gout, out->data, 0, ggml_nbytes(out));
    ggml_free(ctx);
    s.gpu_segments++;
    return true;
}


// ---- executor: decode-time expert matrix products run by the plugin itself ----
// the CPU backend's type table, looked up at run time: builds that load their CPU backend as a separate library do not link it into the plugin
typedef const struct ggml_type_traits_cpu * (*cpu_traits_fn_t)(enum ggml_type);
// Lookup order: linked into the process (LD_PRELOAD builds), the CPU backend's registry, then the CPU backend library that was loaded
// separately (GGML_BACKEND_DL builds: official releases, Vulkan, Windows), whose variant (libggml-cpu-haswell.so, ...) exports the function.
cpu_traits_fn_t find_cpu_traits_fn() {
    cpu_traits_fn_t fn = nullptr;
#ifdef __linux__
    fn = (cpu_traits_fn_t) dlsym(RTLD_DEFAULT, "ggml_get_type_traits_cpu");
#endif
    if (!fn) {
        if (ggml_backend_dev_t cd = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            fn = (cpu_traits_fn_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cd), "ggml_get_type_traits_cpu");
        }
    }
#ifdef __linux__
    if (!fn) {
        std::vector<std::string> libs;
        dl_iterate_phdr([](struct dl_phdr_info * info, size_t, void * data) {
            const char * name = info->dlpi_name;
            if (name && strstr(name, "libggml-cpu")) {
                ((std::vector<std::string> *) data)->push_back(name);
            }
            return 0;
        }, &libs);
        for (const std::string & l : libs) {
            if (void * h = dlopen(l.c_str(), RTLD_LAZY | RTLD_NOLOAD)) {
                fn = (cpu_traits_fn_t) dlsym(h, "ggml_get_type_traits_cpu");
                dlclose(h);   // RTLD_NOLOAD took a reference to an already loaded library; it stays loaded
                if (fn) {
                    break;
                }
            }
        }
    }
#elif defined(_WIN32)
    if (!fn) {
        HMODULE mods[512];
        DWORD need = 0;
        if (EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &need)) {
            for (DWORD i = 0; i < need / sizeof(HMODULE) && i < 512 && !fn; i++) {
                char name[MAX_PATH];
                if (GetModuleFileNameA(mods[i], name, MAX_PATH) && strstr(name, "ggml-cpu")) {
                    fn = (cpu_traits_fn_t) (void *) GetProcAddress(mods[i], "ggml_get_type_traits_cpu");
                }
            }
        }
    }
#endif
    return fn;
}
const ggml_type_traits_cpu * cpu_traits(enum ggml_type t) {
    static cpu_traits_fn_t fn = find_cpu_traits_fn();
    return fn ? fn(t) : nullptr;
}

#if defined(__x86_64__) && defined(__linux__)
// Q2_0 weights: the executor rearranges the quantised activation column once and uses the permuted-layout kernel (bit-identical, about 2x the other one)
void q2_0_perm_dot(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    fastk::vec_dot_q2_0_perm_avx2(n, s, bs, vx, bx, vy, by, nrc);
}
// IQ3_XXS weights (GLM-5.3-Flash down projections): gather-based dot product, bit-identical, about 1.2x; Intel CPUs only (slow gathers on AMD Zen 2/3)
ggml_vec_dot_t fast_iq3_dot() {
    static const ggml_vec_dot_t fn = [] {
        bool vnni = false;
        const char * e = getenv("MOE_CACHE_FAST_KERNELS");
        if ((e && strcmp(e, "0") == 0) || !fastk3::usable(vnni)) {
            return (ggml_vec_dot_t) nullptr;
        }
        return vnni ? (ggml_vec_dot_t) fastk3::vec_dot_iq3_xxs_vnni : (ggml_vec_dot_t) fastk3::vec_dot_iq3_xxs_avx2;
    }();
    return fn;
}
bool fast_kernels_enabled() {
    static const bool on = [] {
        const char * e = getenv("MOE_CACHE_FAST_KERNELS");
        return !(e && strcmp(e, "0") == 0) && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    }();
    return on;
}
#endif

bool exec_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// true when every expert op of the graph is a single-token decode step that the executor can reproduce exactly
bool exec_eligible(state_t & s, ggml_cgraph * cg) {
    if (!s.exec_on || s.lookahead || s.l3_prefetch || s.repack_tag || s.gpu_min_tokens == 1) {
        return false;
    }
    const int n = ggml_graph_n_nodes(cg);
    int seen = 0;
    for (int i = 0; i < n; i++) {
        const ggml_tensor * nd = ggml_graph_node(cg, i);
        if (nd->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        const ggml_tensor * w = nd->src[0], * a = nd->src[1], * ids = nd->src[2];
        auto it = s.by_tensor.find(w);
        if (it == s.by_tensor.end() || s.tensors[it->second]->repack) {
            return false;
        }
        const ggml_type_traits_cpu * tt = cpu_traits(w->type);
        if (!tt || !tt->vec_dot || !cpu_traits(tt->vec_dot_type) || ids->type != GGML_TYPE_I32 || ids->ne[1] < 1 || ids->ne[1] > s.exec_maxcol || ids->ne[2] != 1 || ids->ne[3] != 1) {
            return false;
        }
        if (a->type != GGML_TYPE_F32 || a->nb[0] != sizeof(float) || a->ne[2] != ids->ne[1] || a->ne[3] != 1 || nd->type != GGML_TYPE_F32 || nd->nb[0] != sizeof(float)) {
            return false;
        }
        if (w->nb[0] != ggml_type_size(w->type) || w->ne[2] <= 1 || tt->nrows != 1 || cpu_traits(tt->vec_dot_type)->from_float == nullptr) {
            if (!(a->type == tt->vec_dot_type)) {
                return false;
            }
        }
        seen++;
    }
    return seen > 0;
}

// computes exactly the listed nodes with the CPU backend (all other nodes are switched to NONE for the call)
enum ggml_status exec_run_nodes(moe_cache_backend_ctx * c, ggml_cgraph * cg, const std::vector<int> & which) {
    bool real = false;
    for (int i : which) {
        real = real || !exec_view_op(ggml_graph_node(cg, i)->op);
    }
    if (!real) {
        return GGML_STATUS_SUCCESS;
    }
    const int n = ggml_graph_n_nodes(cg);
    std::vector<char> keep((size_t) n, 0);
    for (int i : which) {
        keep[(size_t) i] = 1;
    }
    std::vector<std::pair<ggml_tensor *, ggml_op>> saved;
    for (int i = 0; i < n; i++) {
        ggml_tensor * nd = ggml_graph_node(cg, i);
        if (!keep[(size_t) i] && nd->op != GGML_OP_NONE) {
            saved.emplace_back(nd, nd->op);
            nd->op = GGML_OP_NONE;
        }
    }
    const enum ggml_status st = ggml_backend_graph_compute(c->cpu, cg);
    for (auto & sv : saved) {
        sv.first->op = sv.second;
    }
    return st;
}

// llama.cpp (ggml 0.26) computes an expert with a separate tiled routine once it has 8 or more rows (tokens) in the batch (ggml-cpu/tiled, rows >= 8),
// which adds up in another order than the one-row kernel. Such ops are left to llama.cpp so the result stays identical (found on GLM-5.3-Flash:
// one expert picked by all 8 tokens of a prompt batch). Older ggml has no tiled routine; handing the op back is exact there too.
bool stock_uses_tiled(const ggml_tensor * nd) {
    const ggml_tensor * ids = nd->src[2];
    if (ids->ne[1] < 8) {
        return false;
    }
    std::unordered_map<int32_t, int> cnt;
    for (int64_t t = 0; t < ids->ne[1]; t++) {
        for (int64_t k = 0; k < ids->ne[0]; k++) {
            if (++cnt[*(const int32_t *) ((const char *) ids->data + k * ids->nb[0] + t * ids->nb[1])] >= 8) {
                return true;
            }
        }
    }
    return false;
}

// In executor graphs the reads are asynchronous: the executor waits for each expert's flag itself. An op handed back to llama.cpp has to wait here first.
void exec_wait_loaded(state_t & s, const ggml_tensor * nd) {
    const ggml_tensor * ids = nd->src[2];
    const tinfo_t & ti = *s.tensors[s.by_tensor.at(nd->src[0])];
    for (int64_t t = 0; t < ids->ne[1]; t++) {
        for (int64_t k = 0; k < ids->ne[0]; k++) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + k * ids->nb[0] + t * ids->nb[1]);
            if (e < 0 || e >= ti.n_expert) {
                continue;
            }
            const bool * f = &s.nodes[(size_t) (ti.node0 + e)].loading;
            while (__atomic_load_n(f, __ATOMIC_ACQUIRE)) {
                std::this_thread::yield();
            }
        }
    }
}

void exec_one_op(state_t & s, ggml_tensor * nd, const std::vector<char> * skip = nullptr) {
    const ggml_tensor * w = nd->src[0], * a = nd->src[1], * ids = nd->src[2];
    const tinfo_t & ti = *s.tensors[s.by_tensor.at(w)];
    const ggml_type_traits_cpu * tt = cpu_traits(w->type);
    const ggml_type vdt = tt->vec_dot_type;
    const int n_ids = (int) ids->ne[0];
    const int n_tok = (int) ids->ne[1];
    const int64_t ne10 = a->ne[0], ne11 = a->ne[1];
    const size_t row_size = ggml_row_size(vdt, ne10);
    static thread_local std::vector<uint8_t> wbuf;
    static thread_local std::vector<uint8_t> pbuf;
    const char * base;
    size_t stride_i11, stride_tok;       // column (token t, activation row i11) = base + i11 * stride_i11 + t * stride_tok
    ggml_vec_dot_t dot = tt->vec_dot;
    bool permuted = false;
#if defined(__x86_64__) && defined(__linux__)
    permuted = w->type == GGML_TYPE_Q2_0 && vdt == GGML_TYPE_Q8_0 && a->type == GGML_TYPE_F32 && ne10 % 64 == 0 && fast_kernels_enabled();
    if (w->type == GGML_TYPE_IQ3_XXS && vdt == GGML_TYPE_Q8_K && fast_iq3_dot()) {
        dot = fast_iq3_dot();
    }
#endif
    if (a->type == vdt) {
        base = (const char *) a->data;
        stride_i11 = a->nb[1];
        stride_tok = a->nb[2];
    } else {
        wbuf.resize((size_t) ne11 * n_tok * row_size + 64);
        const ggml_from_float_t from_float = cpu_traits(vdt)->from_float;
        for (int t = 0; t < n_tok; t++) {
            for (int64_t i11 = 0; i11 < ne11; i11++) {
                from_float((const float *) ((const char *) a->data + (size_t) i11 * a->nb[1] + (size_t) t * a->nb[2]),
                           wbuf.data() + ((size_t) t * ne11 + i11) * row_size, ne10);
            }
        }
        base = (const char *) wbuf.data();
        stride_i11 = row_size;
        stride_tok = (size_t) ne11 * row_size;
#if defined(__x86_64__) && defined(__linux__)
        if (permuted) {
            const size_t nbk = (size_t) (ne10 / 32);
            const size_t ncols = (size_t) ne11 * n_tok;
            pbuf.resize(ncols * nbk * sizeof(fastk::block_q8_0_perm) + 64);
            for (size_t c = 0; c < ncols; c++) {
                fastk::permute_q8_0((const fastk::block_q8_0 *) (wbuf.data() + c * row_size),
                                    (fastk::block_q8_0_perm *) (pbuf.data() + c * nbk * sizeof(fastk::block_q8_0_perm)), (int) nbk);
            }
            base = (const char *) pbuf.data();
            stride_i11 = nbk * sizeof(fastk::block_q8_0_perm);
            stride_tok = (size_t) ne11 * stride_i11;
            dot = q2_0_perm_dot;
        }
#endif
    }
    // one task per expert: the tokens that picked it share its weight rows
    std::vector<exec_task_t> all;
    all.reserve((size_t) n_ids * (size_t) n_tok);
    std::vector<int> task_of((size_t) w->ne[2], -1);
    const std::vector<int32_t> * routed = nullptr;
    if (s.route_mode) {
        auto rit = s.route_ids.find(ids);
        if (rit != s.route_ids.end()) routed = &rit->second;
    }
    for (int t = 0; t < n_tok; t++) {
        for (int slot = 0; slot < n_ids; slot++) {
            if (skip && (*skip)[(size_t) t * (size_t) n_ids + (size_t) slot]) {   // computed elsewhere (hybrid: the GPU bank)
                continue;
            }
            int32_t e = *(const int32_t *) ((const char *) ids->data + (size_t) slot * ids->nb[0] + (size_t) t * ids->nb[1]);
            if (routed) {
                e = (*routed)[(size_t) t * (size_t) n_ids + (size_t) slot];
                if (e < 0) {            // given up: this pick contributes nothing
                    memset((char *) nd->data + (size_t) slot * nd->nb[1] + (size_t) t * nd->nb[2], 0, (size_t) w->ne[1] * sizeof(float));
                    continue;
                }
            }
            if (e < 0 || e >= w->ne[2]) {
                fprintf(stderr, "moe-cache: expert id %d out of range\n", e);
                abort();
            }
            int & ti_ = task_of[(size_t) e];
            if (ti_ < 0) {
                exec_task_t nt;
                nt.w = (const char *) w->data + (size_t) e * w->nb[2];
                nt.ncol = 0;
                nt.flag = (const uint8_t *) &s.nodes[(size_t) (ti.node0 + e)].loading;
                nt.rows = w->ne[1];
                ti_ = (int) all.size();
                all.push_back(nt);
            }
            exec_task_t & tk = all[(size_t) ti_];
            tk.cols[tk.ncol] = base + (size_t) (slot % ne11) * stride_i11 + (size_t) t * stride_tok;
            tk.dsts[tk.ncol] = (float *) ((char *) nd->data + (size_t) slot * nd->nb[1] + (size_t) t * nd->nb[2]);
            tk.ncol++;
        }
    }
    std::vector<exec_task_t> tasks;
    tasks.reserve(all.size());
    std::vector<exec_task_t> late;
    for (const auto & t : all) {
        (__atomic_load_n(t.flag, __ATOMIC_ACQUIRE) ? late : tasks).push_back(t);   // experts already in RAM first
    }
    for (const auto & t : late) {
        tasks.push_back(t);
    }
    const auto tp0 = std::chrono::steady_clock::now();
    s.pool->run(tasks.data(), (int) tasks.size(), w->ne[1], w->ne[0], w->nb[1], dot, 32);
    if ((unsigned) w->type < 64) {
        s.exec_type_ns[w->type] += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tp0).count();
        s.exec_type_ops[w->type]++;
        s.exec_type_bytes[w->type] += (uint64_t) tasks.size() * (uint64_t) w->nb[1] * (uint64_t) w->ne[1];
    }
    s.exec_ops++;
}

// set by the AVX-512 kernel block further down (C linkage: independent of the namespace the kernels live in)
extern "C" { int (*moe_cache_iq512_probe)(void) = nullptr; }

// One line that says what is really active, and why not when something is off (several speed-ups depend on how the plugin was loaded).
void print_capabilities(state_t & s) {
    std::string o = "moe-cache: active:";
    const bool preload = getenv("MOE_CACHE_PRELOAD") != nullptr;
    o += std::string(" load=") + (preload ? "LD_PRELOAD" : "GGML_BACKEND_PATH");
    o += std::string(" cache=") + (s.cache_on ? "on" : "off(pass-through or disabled)");
    if (!s.exec_on) {
        o += " executor=off(MOE_CACHE_EXEC=0)";
    } else if (s.lookahead || s.l3_prefetch || s.repack_tag || s.gpu_min_tokens == 1) {
        o += std::string(" executor=off(") + (s.lookahead ? "lookahead" : s.l3_prefetch ? "l3prefetch" : s.repack_tag ? "repacked weights" : "gpu_min_tokens=1") + ")";
    } else {
        o += " executor=on(<=" + std::to_string(EXEC_MAXCOL) + " tokens" + (s.exec_force ? ", forced" : ", steps aside when resident and bandwidth-bound") + ")";
    }
#if defined(__x86_64__) && defined(__linux__)
    o += std::string(" q2_0_kernel=") + (!fast_kernels_enabled() ? "off(MOE_CACHE_FAST_KERNELS=0 or no AVX2)" : preload ? "on(interposed)" : "executor only(not preloaded)");
    o += std::string(" iq3_xxs_kernel=") + (fast_iq3_dot() ? "on(executor)" : "off(MOE_CACHE_FAST_KERNELS=0, not Intel, or no AVX2)");
    o += std::string(" iq_avx512=") + (getenv("MOE_CACHE_AVX512") ? (moe_cache_iq512_probe && moe_cache_iq512_probe() ? "on" : "off(self-check or CPU)") : "off(MOE_CACHE_AVX512 unset)");
#endif
    o += std::string(" gpu_cache_host_buffer=") + (gpu_cache_mode() ? "on" : "off");
    o += s.gpu_dev ? " long_prompt_gpu=on(>=" + std::to_string(s.gpu_min_tokens) + " tokens)" : " long_prompt_gpu=off(no GPU device)";
    if (s.route_mode) o += std::string(" approx_route=") + (s.route_mode == 1 ? "skip" : "subst");
    if (s.scan_tokens > 0) o += " scan_resistant=" + std::to_string(s.scan_tokens);
    fprintf(stderr, "%s\n", o.c_str());
}

// ---- hybrid (MOE_CACHE_HYBRID=MiB, experimental, not token-identical): GPU computes the picks whose expert is in its bank, the CPU the rest, at the same time ----
// The bank holds the most used experts of the usage profile (static for now). Nothing is uploaded per token: only the activation and the remapped ids go up,
// the bank's partial outputs come back and replace those picks' columns of the down projection's output.
void hybrid_upload(moe_cache_backend_ctx * c, state_t & s) {
    for (auto & h : c->hb) {
        const tinfo_t & ti = *s.tensors[(size_t) h.first];
        ggml_tensor * bank = h.second.first;
        for (size_t e = 0; e < h.second.second.size(); e++) {
            const int sl = h.second.second[e];
            if (sl < 0) {
                continue;
            }
            const char * src = ti.data + e * ti.nb2;
            gpu_cache_ensure(src, bank->nb[2]);
            ggml_backend_tensor_set(bank, src, (size_t) sl * bank->nb[2], bank->nb[2]);
        }
    }
}

void hybrid_drop(moe_cache_backend_ctx * c, state_t & s) {
    ggml_backend_buffer_free(c->hb_buf);
    c->hb_buf = nullptr;
    c->hb_dropped = true;
    for (auto & h : c->hb) {
        h.second.first->data = nullptr;
        h.second.first->buffer = nullptr;
    }
    if (s.hyb_drops++ == 0) {
        fprintf(stderr, "moe-cache: hybrid: bank freed so a long prompt can use the GPU; it is loaded again when decoding starts\n");
    }
}

bool hybrid_init(moe_cache_backend_ctx * c, state_t & s);

void hybrid_adapt_setup(moe_cache_backend_ctx * c, state_t & s) {
    c->hbl.clear();
    c->hb_first = -1;
    c->hb_tok = 0;
    if (const char * v = getenv("MOE_CACHE_HYBRID_ADAPT")) {
        c->ad_every = atoi(v);
        const char * k = strchr(v, ',');
        if (k) {
            c->ad_swaps = atoi(k + 1);
        }
    }
    if (c->ad_every <= 0 || c->ad_swaps <= 0) {
        c->ad_every = 0;
        return;
    }
    if (const char * v = getenv("MOE_CACHE_HYBRID_DUEL")) {
        c->duel_w = std::max(0, atoi(v));
    }
    c->duel.clear();
    c->ha_cur = c->hb_cur = 0;
    for (auto & h : c->hb) {
        const tinfo_t & ti = *s.tensors[(size_t) h.first];
        auto & L = c->hbl[ti.layer];
        L.tidx.push_back(h.first);
        if (L.slot_exp.empty()) {
            L.slot_exp.assign((size_t) h.second.first->ne[2], -1);
            for (size_t e = 0; e < h.second.second.size(); e++) {
                if (h.second.second[e] >= 0) {
                    L.slot_exp[(size_t) h.second.second[e]] = (int) e;
                }
            }
            L.score.assign(L.slot_exp.size(), 0.0f);
            L.miss.assign(h.second.second.size(), 0.0f);
            L.base.assign(h.second.second.size(), 0);
            for (size_t e = 0; e < h.second.second.size(); e++) {
                L.base[e] = h.second.second[e] >= 0;
            }
        }
    }
    c->hb_first = c->hbl.empty() ? -1 : c->hbl.begin()->first;
}

// upload expert e of this layer into bank slot sl (all its weight tensors) and update the slot maps
void hybrid_put(moe_cache_backend_ctx * c, state_t & s, moe_cache_backend_ctx::hb_layer_t & L, int sl, int e) {
    const int old = L.slot_exp[(size_t) sl];
    for (int t : L.tidx) {
        const tinfo_t & ti = *s.tensors[(size_t) t];
        auto & h = c->hb[t];
        const char * src = ti.data + (size_t) e * ti.nb2;
        gpu_cache_ensure(src, h.first->nb[2]);
        ggml_backend_tensor_set(h.first, src, (size_t) sl * h.first->nb[2], h.first->nb[2]);
        if (old >= 0) {
            h.second[(size_t) old] = -1;
        }
        h.second[(size_t) e] = sl;
    }
    L.slot_exp[(size_t) sl] = e;
}

// between two tokens (the GPU is idle): the most missed experts replace the coldest bank expert of their own layer when they were used more often
void hybrid_adapt(moe_cache_backend_ctx * c, state_t & s) {
    const auto t0 = std::chrono::steady_clock::now();
    const float dec = powf(0.97f, (float) c->ad_every);
    if (c->duel_w > 0) {
        int sa = 0, sb = 0;
        for (const auto & d : c->duel) {
            sa += d.first;
            sb += d.second;
        }
        if (sa < sb) {
            // the profile's bank would have served more: bring profile experts back in place of adapted ones (lowest score first), no new swaps
            int done = 0;
            for (auto & lv : c->hbl) {
                auto & L = lv.second;
                for (float & sc : L.score) {
                    sc *= dec;
                }
                std::fill(L.miss.begin(), L.miss.end(), 0.0f);
                if (done >= c->ad_swaps) {
                    continue;
                }
                std::vector<char> in(L.base.size(), 0);
                for (int e : L.slot_exp) {
                    if (e >= 0) {
                        in[(size_t) e] = 1;
                    }
                }
                std::vector<std::pair<double, int>> want;     // profile experts missing from the bank, most used first
                for (size_t e = 0; e < L.base.size(); e++) {
                    if (L.base[e] && !in[e]) {
                        want.emplace_back(-s.learned.at(lv.first, (int) e), (int) e);
                    }
                }
                std::sort(want.begin(), want.end());
                for (const auto & w : want) {
                    if (done >= c->ad_swaps) {
                        break;
                    }
                    int cold = -1;
                    for (size_t sl = 0; sl < L.slot_exp.size(); sl++) {
                        const int e = L.slot_exp[sl];
                        if (e >= 0 && !L.base[(size_t) e] && (cold < 0 || L.score[sl] < L.score[(size_t) cold])) {
                            cold = (int) sl;
                        }
                    }
                    if (cold < 0) {
                        break;
                    }
                    hybrid_put(c, s, L, cold, w.second);
                    L.score[(size_t) cold] = 0.0f;
                    done++;
                }
            }
            s.hyb_swaps += (uint64_t) done;
            s.hyb_reverts += (uint64_t) done;
            s.hyb_swap_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            return;
        }
    }
    std::vector<std::tuple<float, int, int>> cand;          // misses, layer, expert
    for (auto & lv : c->hbl) {
        auto & L = lv.second;
        for (float & sc : L.score) {
            sc *= dec;
        }
        for (size_t e = 0; e < L.miss.size(); e++) {
            if (L.miss[e] >= 2.0f) {
                cand.emplace_back(L.miss[e], lv.first, (int) e);
            }
        }
    }
    std::sort(cand.begin(), cand.end(), [](const auto & x, const auto & y) { return std::get<0>(x) > std::get<0>(y); });
    int done = 0;
    for (const auto & cd : cand) {
        if (done >= c->ad_swaps) {
            break;
        }
        auto & L = c->hbl[std::get<1>(cd)];
        const int e = std::get<2>(cd);
        int cold = -1;
        float best = 0.0f;
        for (size_t sl = 0; sl < L.slot_exp.size(); sl++) {
            const float v = L.score[sl] + 0.001f * (float) s.learned.at(std::get<1>(cd), L.slot_exp[sl]);
            if (cold < 0 || v < best) {
                cold = (int) sl;
                best = v;
            }
        }
        if (cold < 0 || L.score[(size_t) cold] >= std::get<0>(cd)) {
            continue;
        }
        hybrid_put(c, s, L, cold, e);
        L.score[(size_t) cold] = std::get<0>(cd);
        done++;
    }
    for (auto & lv : c->hbl) {
        std::fill(lv.second.miss.begin(), lv.second.miss.end(), 0.0f);
    }
    s.hyb_swaps += (uint64_t) done;
    s.hyb_swap_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

// Same bank as before the long prompt: release the long-prompt block's GPU buffer first (it holds about one layer of experts), then upload the bank again.
bool hybrid_reload(moe_cache_backend_ctx * c, state_t & s) {
    const auto t0 = std::chrono::steady_clock::now();
    c->hb_dropped = false;
    ggml_gallocr_free(c->galloc);
    c->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->gpu));
    c->hb_buf = ggml_backend_alloc_ctx_tensors(c->hb_ctx, c->gpu);
    if (!c->hb_buf) {
        fprintf(stderr, "moe-cache: hybrid: the bank no longer fits after a long prompt: choosing a smaller one\n");
        c->hb_tried = false;
        return hybrid_init(c, s);
    }
    hybrid_upload(c, s);
    s.hyb_reloads++;
    s.hyb_reload_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool hybrid_init(moe_cache_backend_ctx * c, state_t & s) {
    if (c->hb_tried) {
        if (!c->hb_buf && c->hb_dropped) {
            return hybrid_reload(c, s);
        }
        return c->hb_buf != nullptr;
    }
    c->hb_tried = true;
    const char * env = getenv("MOE_CACHE_HYBRID");
    size_t mib = env ? (strcmp(env, "auto") == 0 ? (size_t) -1 : (size_t) atoll(env)) : 0;
    if (!mib || !s.gpu_dev || c->gpu_failed) {
        return false;
    }
    {
        // never more than the GPU has free, minus room for llama.cpp's own buffers that come later (auto = all of that)
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(s.gpu_dev, &fr, &tot);
        const size_t keep = (size_t) 600 << 20;
        const size_t avail = fr > keep ? (fr - keep) >> 20 : 0;
        if (mib > avail) {
            mib = avail;
        }
        if (mib < 256) {
            fprintf(stderr, "moe-cache: hybrid: only %zu MiB of GPU memory free, no bank\n", fr >> 20);
            return false;
        }
    }
    if (s.learned.n_exp == 0) {
        fprintf(stderr, "moe-cache: hybrid: no usage profile (run moe-cache-learn or use the model once), so no bank\n");
        return false;
    }
    if (!c->gpu) {
        c->gpu = ggml_backend_dev_init(s.gpu_dev, nullptr);
        if (!c->gpu) {
            c->gpu_failed = true;
            return false;
        }
        c->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->gpu));
    }
    std::map<int, size_t> layer_bytes;                        // all expert tensors of one layer, for one expert
    for (const auto & tp : s.tensors) {
        if (tp->layer >= 0 && !tp->repack && tp->tensor) {
            layer_bytes[tp->layer] += tp->nb2;
        }
    }
    std::vector<std::tuple<double, int, int>> cand;
    for (const auto & lb : layer_bytes) {
        for (int e = 0; e < s.learned.n_exp; e++) {
            const double n = s.learned.at(lb.first, e);
            if (n > 0) {
                cand.emplace_back(n, lb.first, e);
            }
        }
    }
    std::sort(cand.begin(), cand.end(), [](const auto & x, const auto & y) { return std::get<0>(x) > std::get<0>(y); });
    for (;; mib = mib * 85 / 100) {   // allocation failed: try again with 85 % until it fits or gets too small
    if (mib < 256) {
        fprintf(stderr, "moe-cache: hybrid: could not allocate a bank on the GPU\n");
        return false;
    }
    std::map<int, std::vector<int>> chosen;
    size_t used = 0, n_exp = 0;
    const size_t budget = mib << 20;
    for (const auto & cd : cand) {
        const size_t b = layer_bytes[std::get<1>(cd)];
        if (used + b > budget) {
            continue;
        }
        chosen[std::get<1>(cd)].push_back(std::get<2>(cd));
        used += b;
        n_exp++;
    }
    if (c->hb_ctx) {
        ggml_free(c->hb_ctx);
        c->hb.clear();
    }
    ggml_init_params ip = { ggml_tensor_overhead() * (s.tensors.size() + 1), nullptr, true };
    c->hb_ctx = ggml_init(ip);
    for (size_t i = 0; i < s.tensors.size(); i++) {
        const tinfo_t & ti = *s.tensors[i];
        auto it = chosen.find(ti.layer);
        if (it == chosen.end() || ti.repack || !ti.tensor) {
            continue;
        }
        const ggml_tensor * t = ti.tensor;
        ggml_tensor * bank = ggml_new_tensor_3d(c->hb_ctx, t->type, t->ne[0], t->ne[1], (int64_t) it->second.size());
        std::vector<int> slot((size_t) ti.n_expert, -1);
        for (size_t k = 0; k < it->second.size(); k++) {
            if (it->second[k] < ti.n_expert) {
                slot[(size_t) it->second[k]] = (int) k;
            }
        }
        c->hb[(int) i] = { bank, slot };
    }
    c->hb_buf = ggml_backend_alloc_ctx_tensors(c->hb_ctx, c->gpu);
    if (!c->hb_buf) {
        continue;
    }
    hybrid_upload(c, s);
    hybrid_adapt_setup(c, s);
    fprintf(stderr, "moe-cache: hybrid: GPU bank of %zu experts in %zu layers (%.0f MiB), chosen from the usage profile\n", n_exp, chosen.size(), used / 1048576.0);
    return true;
    }
}

// One expert block (gate, up, activation, down: MUL_MAT_ID / GLU nodes sharing the same ids) starting at node `first`. Returns false to leave it to the executor.
bool hybrid_group(moe_cache_backend_ctx * c, state_t & s, ggml_cgraph * cg, int first, int & last_out) {
    const int n_all = ggml_graph_n_nodes(cg);
    const ggml_tensor * ids = ggml_graph_node(cg, first)->src[2];
    std::vector<ggml_tensor *> seg;
    for (int j = first; j < n_all; j++) {
        ggml_tensor * n = ggml_graph_node(cg, j);
        if ((n->op == GGML_OP_MUL_MAT_ID && n->src[2] == ids) || n->op == GGML_OP_GLU) {
            seg.push_back(n);
        } else {
            break;
        }
    }
    while (!seg.empty() && seg.back()->op != GGML_OP_MUL_MAT_ID) {
        seg.pop_back();
    }
    if (seg.size() < 2 || !ggml_is_contiguous(seg.back()) || seg.back()->type != GGML_TYPE_F32) {
        return false;
    }
    const std::vector<int> * slot = nullptr;
    for (ggml_tensor * n : seg) {
        if (n->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        auto it = s.by_tensor.find(n->src[0]);
        if (it == s.by_tensor.end() || !c->hb.count(it->second)) {
            return false;
        }
        slot = &c->hb[it->second].second;
    }
    const int64_t n_used = ids->ne[0], n_tok = ids->ne[1];
    const int layer = s.tensors[(size_t) s.by_tensor.at(seg[0]->src[0])]->layer;
    moe_cache_backend_ctx::hb_layer_t * L = nullptr;
    if (c->ad_every > 0) {
        if (layer == c->hb_first) {
            if (c->duel_w > 0 && c->hb_tok > 0) {      // one token has passed: keep its real and shadow hits for the comparison
                c->duel.emplace_back(c->ha_cur, c->hb_cur);
                if ((int) c->duel.size() > c->duel_w) {
                    c->duel.pop_front();
                }
            }
            c->ha_cur = c->hb_cur = 0;
            if (++c->hb_tok % (uint64_t) c->ad_every == 0) {
                hybrid_adapt(c, s);
            }
        }
        auto lt = c->hbl.find(layer);
        L = lt != c->hbl.end() ? &lt->second : nullptr;
    }
    std::vector<char> hit((size_t) (n_used * n_tok), 0);
    std::vector<int32_t> rid((size_t) (n_used * n_tok), 0);
    int64_t n_hit = 0;
    for (int64_t t = 0; t < n_tok; t++) {
        for (int64_t u = 0; u < n_used; u++) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + u * ids->nb[0] + t * ids->nb[1]);
            const int sl = (e >= 0 && (size_t) e < slot->size()) ? (*slot)[(size_t) e] : -1;
            if (sl >= 0) {
                hit[(size_t) (t * n_used + u)] = 1;
                rid[(size_t) (t * n_used + u)] = sl;
                n_hit++;
                if (L) {
                    L->score[(size_t) sl] += 1.0f;
                    c->ha_cur++;
                }
            } else if (L && e >= 0 && (size_t) e < L->miss.size()) {
                L->miss[(size_t) e] += 1.0f;
            }
            if (L && e >= 0 && (size_t) e < L->base.size() && L->base[(size_t) e]) {
                c->hb_cur++;
            }
        }
    }
    if (n_hit == 0) {
        return false;
    }
    // the GPU graph: same nodes, weights replaced by the bank, ids by the slots (misses point at slot 0; their columns are not used)
    ggml_init_params ip = { ggml_tensor_overhead() * 4 * (seg.size() + 8) + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    std::unordered_map<const ggml_tensor *, ggml_tensor *> clone;
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> uploads;
    ggml_tensor * gids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_set_input(gids);
    for (ggml_tensor * n : seg) {
        ggml_tensor * nn = ggml_new_tensor(ctx, n->type, GGML_MAX_DIMS, n->ne);
        nn->op = n->op;
        memcpy(nn->op_params, n->op_params, sizeof(nn->op_params));
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            const ggml_tensor * src = n->src[k];
            if (!src) {
                continue;
            }
            if (clone.count(src)) {
                nn->src[k] = clone[src];
            } else if (n->op == GGML_OP_MUL_MAT_ID && k == 0) {
                nn->src[k] = c->hb[s.by_tensor.at(src)].first;
            } else if (n->op == GGML_OP_MUL_MAT_ID && k == 2) {
                nn->src[k] = gids;
            } else {
                ggml_tensor * nt = ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
                ggml_set_input(nt);
                clone[src] = nt;
                uploads.emplace_back(src, nt);
                nn->src[k] = nt;
            }
        }
        clone[n] = nn;
    }
    ggml_tensor * gout = clone[seg.back()];
    ggml_set_output(gout);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, gout);
    if (!ggml_gallocr_alloc_graph(c->galloc, g)) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(gids, rid.data(), 0, rid.size() * sizeof(int32_t));
    std::vector<uint8_t> pack;
    for (auto & up : uploads) {
        const ggml_tensor * t = up.first;
        if (ggml_is_contiguous(t)) {
            ggml_backend_tensor_set(up.second, t->data, 0, ggml_nbytes(up.second));
            continue;
        }
        const size_t es = ggml_type_size(t->type);
        pack.assign(ggml_nbytes(up.second), 0);
        size_t o = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) for (int64_t i2 = 0; i2 < t->ne[2]; i2++) for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
            memcpy(pack.data() + o, (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], (size_t) t->ne[0] * es);
            o += (size_t) t->ne[0] * es;
        }
        ggml_backend_tensor_set(up.second, pack.data(), 0, pack.size());
    }
    if (ggml_backend_graph_compute_async(c->gpu, g) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return false;
    }
    // meanwhile the CPU computes the misses (the activation node runs for all picks; the hit picks' values are replaced below)
    if (n_hit < n_used * n_tok) {
        for (int j = 0; j < (int) seg.size(); j++) {
            if (seg[(size_t) j]->op == GGML_OP_MUL_MAT_ID) {
                exec_one_op(s, seg[(size_t) j], &hit);
            } else {
                exec_run_nodes(c, cg, std::vector<int>{ first + j });
            }
        }
    }
    ggml_backend_synchronize(c->gpu);
    ggml_tensor * out = seg.back();
    std::vector<float> res((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(gout, res.data(), 0, ggml_nbytes(gout));
    for (int64_t t = 0; t < n_tok; t++) {
        for (int64_t u = 0; u < n_used; u++) {
            if (hit[(size_t) (t * n_used + u)]) {
                memcpy((char *) out->data + u * out->nb[1] + t * out->nb[2], res.data() + (size_t) ((t * n_used + u) * out->ne[0]), (size_t) out->ne[0] * sizeof(float));
            }
        }
    }
    ggml_free(ctx);
    s.hyb_groups++;
    s.hyb_picks += (uint64_t) (n_used * n_tok);
    s.hyb_hits += (uint64_t) n_hit;
    last_out = first + (int) seg.size() - 1;
    return true;
}

enum ggml_status exec_run_graph(moe_cache_backend_ctx * c, state_t & s, ggml_cgraph * cg) {
    if (!s.pool) {
        const int n = s.exec_threads > 0 ? s.exec_threads : std::max(1, c->n_threads);
        static const std::vector<int> pins = pin_cpus();
        std::vector<int> cpus;
        for (int i = 1; i < n; i++) {
            cpus.push_back(pins.size() >= (size_t) n ? pins[(size_t) i] : -1);
        }
        int extra = 0;
#ifdef __linux__
        if (s.exec_ecores > 0) {
            std::ifstream f("/sys/devices/cpu_atom/cpus");
            std::string txt;
            if (f && std::getline(f, txt)) {
                size_t i = 0;
                while (i < txt.size() && extra < s.exec_ecores) {
                    int a = atoi(txt.c_str() + i), b = a;
                    while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
                    if (i < txt.size() && txt[i] == '-') {
                        i++;
                        b = atoi(txt.c_str() + i);
                        while (i < txt.size() && isdigit((unsigned char) txt[i])) { i++; }
                    }
                    for (int cpu = a; cpu <= b && extra < s.exec_ecores; cpu++) { cpus.push_back(cpu); extra++; }
                    while (i < txt.size() && !isdigit((unsigned char) txt[i])) { i++; }
                }
            }
        }
#endif
        s.pool.reset(new exec_pool_t());
        s.pool->start(cpus, (int) cpus.size());
        fprintf(stderr, "moe-cache: executor pool: %d compute threads (%d on efficiency cores)\n", (int) cpus.size() + 1, extra);
    }
    if (CPU().set_n_threads) {
        CPU().set_n_threads(c->cpu, 1);       // the small nodes between the matrix products need no pool
    }
    const int n = ggml_graph_n_nodes(cg);
    std::vector<int> batch;
    enum ggml_status st = GGML_STATUS_SUCCESS;
    for (int i = 0; i < n && st == GGML_STATUS_SUCCESS; i++) {
        ggml_tensor * nd = ggml_graph_node(cg, i);
        if (nd->op == GGML_OP_MUL_MAT_ID && s.by_tensor.find(nd->src[0]) != s.by_tensor.end() && hybrid_init(c, s)) {
            st = exec_run_nodes(c, cg, batch);
            batch.clear();
            int last = i;
            if (st == GGML_STATUS_SUCCESS && hybrid_group(c, s, cg, i, last)) {
                i = last;
                continue;
            }
        }
        if (nd->op == GGML_OP_MUL_MAT_ID && s.by_tensor.find(nd->src[0]) != s.by_tensor.end() && !stock_uses_tiled(nd)) {
            st = exec_run_nodes(c, cg, batch);
            batch.clear();
            exec_one_op(s, nd);
            if (s.exec_check) {
                const size_t nb = ggml_nbytes(nd);
                std::vector<char> mine((const char *) nd->data, (const char *) nd->data + nb);
                memset(nd->data, 0, nb);
                exec_run_nodes(c, cg, std::vector<int>{ i });
                s.exec_checked++;
                if (memcmp(mine.data(), nd->data, nb) != 0) {
                    if (++s.exec_mismatch <= 8) {   // say where: weight type, tokens in the batch, first differing value
                        const float * x = (const float *) mine.data(), * y = (const float *) nd->data;
                        size_t k = 0;
                        while (k + 1 < nb / sizeof(float) && x[k] == y[k]) { k++; }
                        fprintf(stderr, "moe-cache: executor mismatch %s: type %s, %lld tokens, %lld picks, first at float %zu: %.9g vs %.9g\n", nd->name,
                                ggml_type_name(nd->src[0]->type), (long long) nd->ne[2], (long long) nd->src[2]->ne[0], k, x[k], y[k]);
                        // recompute that one value by hand with llama.cpp's kernel: which side is the odd one out, and for which expert
                        const ggml_tensor * w = nd->src[0], * a = nd->src[1], * idt = nd->src[2];
                        const int64_t r = (int64_t) k % nd->ne[0], slot = ((int64_t) k / nd->ne[0]) % nd->ne[1], t = (int64_t) k / (nd->ne[0] * nd->ne[1]);
                        const int32_t e = *(const int32_t *) ((const char *) idt->data + slot * idt->nb[0] + t * idt->nb[1]);
                        const ggml_type_traits_cpu * tt = cpu_traits(w->type);
                        std::vector<uint8_t> col(ggml_row_size(tt->vec_dot_type, a->ne[0]) + 64);
                        const float * af = (const float *) ((const char *) a->data + (slot % a->ne[1]) * a->nb[1] + t * a->nb[2]);
                        cpu_traits(tt->vec_dot_type)->from_float(af, col.data(), a->ne[0]);
                        float v = 0;
                        tt->vec_dot((int) w->ne[0], &v, 0, (const char *) w->data + (size_t) e * w->nb[2] + (size_t) r * w->nb[1], 0, col.data(), 0, 1);
                        int same_e = 0;
                        for (int64_t tt2 = 0; tt2 < idt->ne[1]; tt2++) for (int64_t s2 = 0; s2 < idt->ne[0]; s2++)
                            same_e += *(const int32_t *) ((const char *) idt->data + s2 * idt->nb[0] + tt2 * idt->nb[1]) == e;
                        fprintf(stderr, "moe-cache:   row %lld, pick %lld, token %lld, expert %d (picked %d times in this batch); by hand %.9g -> %s; activation type %s, ne1 %lld\n",
                                (long long) r, (long long) slot, (long long) t, e, same_e, v, v == x[k] ? "matches the executor" : v == y[k] ? "matches stock" : "matches neither",
                                ggml_type_name(a->type), (long long) a->ne[1]);
                    }
                    memcpy(nd->data, mine.data(), nb);
                }
            }
        } else {
            if (nd->op == GGML_OP_MUL_MAT_ID && s.by_tensor.find(nd->src[0]) != s.by_tensor.end()) {
                exec_wait_loaded(s, nd);   // left to llama.cpp (tiled): its experts may still be filling (reads are asynchronous in executor graphs)
            }
            batch.push_back(i);
        }
    }
    if (st == GGML_STATUS_SUCCESS) {
        st = exec_run_nodes(c, cg, batch);
    }
    s.route_ids.clear();
    return st;
}

enum ggml_status be_compute(ggml_backend_t backend, ggml_cgraph * cg) {
    auto * c = (moe_cache_backend_ctx *) backend->context;
    state_t & s = S();
    const auto t_in = std::chrono::steady_clock::now();
    struct total_t { state_t & s; std::chrono::steady_clock::time_point t; ~total_t() { s.ns_total += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count(); } } total_guard{ s, t_in };
    if (!c->cpu) {
        c->cpu = ggml_backend_dev_init(CPU().dev, nullptr);
    }
    if (c->tp_n != c->n_threads && CPU().tp_new && CPU().set_tp) {
        if (c->tp && CPU().tp_free) {
            CPU().tp_free(c->tp);
        }
        ggml_threadpool_params tpp;
        memset(&tpp, 0, sizeof(tpp));
        tpp.n_threads = c->n_threads;
        tpp.prio = GGML_SCHED_PRIO_NORMAL;
        tpp.poll = 50;
        static const std::vector<int> pins = pin_cpus();
        if (!pins.empty() && (size_t) c->n_threads <= pins.size()) {
            for (int i = 0; i < c->n_threads; i++) {
                if (pins[(size_t) i] >= 0 && pins[(size_t) i] < GGML_MAX_N_THREADS) {
                    tpp.cpumask[pins[(size_t) i]] = true;
                }
            }
            tpp.strict_cpu = true;
            if (!c->pin_reported) {
                c->pin_reported = true;
                fprintf(stderr, "moe-cache: compute threads pinned to %d performance cores (MOE_CACHE_CPU_PIN=off to disable)\n", c->n_threads);
            }
        }
        if (const char * v = getenv("MOE_CACHE_POLL")) {
            tpp.poll = (uint32_t) atoi(v);
        }
        c->tp = CPU().tp_new(&tpp);
        c->tp_n = c->n_threads;
        CPU().set_tp(c->cpu, c->tp);
    }
    if (CPU().set_n_threads) {
        CPU().set_n_threads(c->cpu, c->n_threads);
    }

    s.graphs++;
    {
        static bool reported = false;
        if (!reported) {
            reported = true;
            print_capabilities(s);
        }
    }
    if (!s.cache_on) {
        for (int i = 0; i < ggml_graph_n_nodes(cg); i++) {
            if (ggml_graph_node(cg, i)->op == GGML_OP_MUL_MAT_ID) {
                s.ops++;
            }
        }
    }
    struct group_t { int first; int last; int n_tok; };
    std::vector<group_t> groups_found;
    bool exec_graph = s.cache_on && exec_eligible(s, cg);
    size_t exec_jobs = 0;          // reads queued for this graph
    bool exec_heavy = false;       // weights whose dot product is compute-bound (IQ types, Q2_0)
    if (s.cache_on) {
        if (!s.preload_done) {
            s.preload();
        }
        const int n_nodes = ggml_graph_n_nodes(cg);
        std::vector<char> done((size_t) n_nodes, 0);
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * node = ggml_graph_node(cg, i);
            if (done[i] || node->op != GGML_OP_MUL_MAT_ID || s.by_tensor.find(node->src[0]) == s.by_tensor.end()) {
                continue;
            }
            ggml_tensor * ids = node->src[2];
            for (int k = 0; k < i; k++) {
                if (ggml_graph_node(cg, k) == ids) {
                    fprintf(stderr, "moe-cache: ids tensor is computed inside the same split, not supported\n");
                    abort();
                }
            }
            std::vector<int> tidx;
            std::vector<int> used((size_t) node->src[0]->ne[2], 0);
            for (int j = i; j < n_nodes; j++) {
                ggml_tensor * n2 = ggml_graph_node(cg, j);
                if (n2->op != GGML_OP_MUL_MAT_ID || n2->src[2] != ids) {
                    continue;
                }
                auto it2 = s.by_tensor.find(n2->src[0]);
                if (it2 == s.by_tensor.end()) {
                    continue;
                }
                if (std::find(tidx.begin(), tidx.end(), it2->second) == tidx.end()) {
                    tidx.push_back(it2->second);
                }
                done[j] = 1;
                s.ops++;
            }
            const int64_t n_used = ids->ne[0];
            const int64_t n_tok = ids->ne[1];
            for (int64_t t = 0; t < n_tok; t++) {
                for (int64_t u = 0; u < n_used; u++) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + u * ids->nb[0] + t * ids->nb[1]);
                    if (e >= 0 && e < (int32_t) used.size()) {
                        used[(size_t) e]++;
                    }
                }
            }
            groups_found.push_back({ i, 0, (int) n_tok });
            for (int j = 0; j < n_nodes; j++) {
                if (done[j] && ggml_graph_node(cg, j)->src[2] == ids) {
                    groups_found.back().last = j;
                }
            }
            const auto p0 = std::chrono::steady_clock::now();
            s.cur_decode = n_tok == 1;
            s.cur_scan = s.scan_tokens > 0 && n_tok >= s.scan_tokens;
            if (s.l3_prefetch && n_tok == 1 && tidx.size() >= 2) {
                // ranges of the experts this step will read in its second and later tensors (not yet touched by the kernel)
                std::vector<std::pair<const char *, size_t>> ranges;
                for (size_t q = 1; q < tidx.size(); q++) {
                    const tinfo_t & ti = *s.tensors[tidx[q]];
                    for (size_t e = 0; e < used.size() && e < (size_t) ti.n_expert; e++) {
                        if (used[e]) {
                            ranges.push_back({ ti.data + e * ti.nb2, ti.nb2 });
                        }
                    }
                }
                s.pf_submit([ranges = std::move(ranges)] {
                    [[maybe_unused]] static volatile uint64_t sink;
                    uint64_t acc = 0;
                    for (const auto & r : ranges) {
                        for (size_t off = 0; off + 8 <= r.second; off += 64) {
                            acc += *(const volatile uint64_t *) (r.first + off);   // a real load: a prefetch hint can be dropped on a TLB miss
                        }
                    }
                    sink = acc;
                });
            }
            if (s.trace_f && n_tok == 1) {
                int16_t hdr[2] = { (int16_t) s.tensors[tidx[0]]->layer, (int16_t) n_used };
                fwrite(hdr, sizeof(int16_t), 2, s.trace_f);
                for (int64_t u = 0; u < n_used; u++) {
                    const int16_t e = (int16_t) *(const int32_t *) ((const char *) ids->data + u * ids->nb[0]);
                    fwrite(&e, sizeof(int16_t), 1, s.trace_f);
                }
            }
            {
                const ggml_type wt = s.tensors[(size_t) tidx[0]]->tensor->type;
                exec_heavy = exec_heavy || wt == GGML_TYPE_IQ1_S || wt == GGML_TYPE_IQ1_M || wt == GGML_TYPE_IQ2_XXS || wt == GGML_TYPE_IQ2_XS ||
                             wt == GGML_TYPE_IQ2_S || wt == GGML_TYPE_IQ3_XXS || wt == GGML_TYPE_IQ3_S || wt == GGML_TYPE_Q2_0;
            }
            if (s.route_mode && exec_graph && n_tok <= s.exec_maxcol && !tidx.empty() && n_used >= 3) {
                const tinfo_t & t0 = *s.tensors[(size_t) tidx[0]];
                const int keep = s.route_keep >= 0 ? (int) std::min<int64_t>(s.route_keep, n_used) : (int) n_used - 2;
                std::vector<int32_t> eff((size_t) (n_used * n_tok));
                for (int64_t t = 0; t < n_tok; t++) {
                    std::vector<char> taken((size_t) t0.n_expert, 0);
                    for (int64_t u = 0; u < n_used; u++) {
                        const int32_t e = *(const int32_t *) ((const char *) ids->data + u * ids->nb[0] + t * ids->nb[1]);
                        eff[(size_t) (t * n_used + u)] = e;
                        if (e >= 0 && e < (int32_t) taken.size()) taken[(size_t) e] = 1;
                    }
                    for (int64_t u = keep; u < n_used; u++) {
                        const int32_t e = eff[(size_t) (t * n_used + u)];
                        if (e < 0 || e >= (int32_t) t0.n_expert) continue;
                        s.route_considered++;
                        if (s.nodes[(size_t) (t0.node0 + e)].resident) continue;     // in RAM (or already being read): use it
                        int32_t best = -1;
                        if (s.route_mode == 2) {
                            uint64_t best_epoch = 0;
                            for (int32_t c = 0; c < (int32_t) t0.n_expert; c++) {
                                const node_t & nd2 = s.nodes[(size_t) (t0.node0 + c)];
                                if (!taken[(size_t) c] && nd2.resident && !nd2.loading && (best < 0 || (uint64_t) nd2.epoch > best_epoch)) {
                                    best = c;
                                    best_epoch = (uint64_t) nd2.epoch;
                                }
                            }
                        }
                        if (e < (int32_t) used.size() && used[(size_t) e] > 0) used[(size_t) e]--;
                        eff[(size_t) (t * n_used + u)] = best;
                        if (best >= 0) {
                            taken[(size_t) best] = 1;
                            if (best < (int32_t) used.size()) used[(size_t) best]++;
                            s.route_substituted++;
                        } else {
                            s.route_skipped++;
                        }
                    }
                }
                s.route_ids[ids] = std::move(eff);
            }
            exec_jobs += s.prepare_group(tidx, used, exec_graph);
            // long prompts use nearly every expert of every layer: start reading the next layer's experts now, while this layer is uploaded and computed
            if (s.prompt_prefetch && n_tok >= s.gpu_min_tokens && (size_t) (n_tok * n_used) >= 2 * used.size()) {
                const int layer = s.tensors[(size_t) tidx[0]]->layer;
                if (layer >= 0) {
                    std::vector<int> all(used.size());
                    for (size_t e = 0; e < all.size(); e++) {
                        all[e] = (int) e;
                    }
                    s.reserve_and_read(layer + 1, all, (int) all.size());
                }
            }
            if (s.lookahead && n_tok == 1) {
                const ggml_tensor * xin = node->src[1];
                if (xin && xin->type == GGML_TYPE_F32 && xin->nb[0] == sizeof(float) && xin->data) {
                    std::vector<float> xv((size_t) xin->ne[0]);
                    memcpy(xv.data(), xin->data, xv.size() * sizeof(float));
                    const int layer = s.tensors[tidx[0]]->layer;
                    if (layer >= 0) {
                        s.pf_submit([&s, layer, xv = std::move(xv), n_used = (int) n_used] { s.predict_and_prefetch(layer, xv, n_used); });
                    }
                }
            }
            if (!s.profile_path.empty() && ++s.groups_since_save >= 100000) {
                s.groups_since_save = 0;
                s.save_profile();
            }
            s.ns_prep += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - p0).count();
        }
    }

    // everything is resident and the weights are bandwidth-bound (K-quants, legacy types): llama.cpp's own path is as fast and has less overhead
    if (exec_graph && exec_jobs == 0 && !exec_heavy && !s.exec_force) {
        exec_graph = false;
        if (CPU().set_n_threads) {
            CPU().set_n_threads(c->cpu, c->n_threads);
        }
    }
    if (exec_graph) {
        const auto e0 = std::chrono::steady_clock::now();
        const enum ggml_status est = exec_run_graph(c, s, cg);
        s.ns_cpu += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - e0).count();
        return est;
    }

    // long prompts: run the expert block on the GPU (what llama.cpp's own offload does for batches of 32+ tokens)
    std::vector<std::pair<ggml_tensor *, ggml_op>> silenced;
    if (s.cache_on && s.gpu_dev && !c->gpu_failed) {
        for (const auto & gr : groups_found) {
            if (gr.n_tok < s.gpu_min_tokens) {
                continue;
            }
            std::vector<ggml_tensor *> seg;
            c->seg_nomem = false;
            bool ok = gpu_segment(c, s, cg, gr.first, gr.last, seg);
            if (!ok && c->seg_nomem && c->hb_buf) {
                hybrid_drop(c, s);         // the hybrid bank holds the memory the long-prompt block needs: free it now, load it again at the next decode step
                seg.clear();
                ok = gpu_segment(c, s, cg, gr.first, gr.last, seg);
            }
            if (ok) {
                for (ggml_tensor * n : seg) {
                    silenced.emplace_back(n, n->op);
                    n->op = GGML_OP_NONE;
                }
            } else {
                s.gpu_fallbacks++;
            }
        }
    }

    // repacked experts: tag the weight as living in llama.cpp's repack buffer type while the kernel runs, so it picks the repacked code path
    std::vector<std::pair<ggml_tensor *, ggml_backend_buffer_t>> tagged;
    if (s.cache_on && s.repack_tag) {
        const int n_all = ggml_graph_n_nodes(cg);
        for (int i = 0; i < n_all; i++) {
            ggml_tensor * node = ggml_graph_node(cg, i);
            if (node->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            auto it = s.by_tensor.find(node->src[0]);
            if (it != s.by_tensor.end() && s.tensors[it->second]->repack && node->src[0]->buffer != s.repack_tag) {
                tagged.emplace_back(node->src[0], node->src[0]->buffer);
                node->src[0]->buffer = s.repack_tag;
            }
        }
    }
    const auto c0 = std::chrono::steady_clock::now();
    const enum ggml_status st = ggml_backend_graph_compute(c->cpu, cg);
    for (auto & tg : tagged) {
        tg.first->buffer = tg.second;
    }
    for (auto & sl : silenced) {
        sl.first->op = sl.second;
    }
    s.ns_cpu += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - c0).count();

    return st;
}

const ggml_backend_i g_backend_i = {
    /* .get_name           = */ be_name,
    /* .free               = */ be_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async= */ nullptr,
    /* .get_tensor_2d_async= */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ be_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

void set_n_threads(ggml_backend_t backend, int n) {
    ((moe_cache_backend_ctx *) backend->context)->n_threads = n;
}

// ---- device ----

const char * dev_name(ggml_backend_dev_t) {
    return "MOE_CACHE";
}

const char * dev_desc(ggml_backend_dev_t) {
    return "moe-cache expert residency backend";
}

void dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
}

enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    memset(props, 0, sizeof(*props));
    props->name = dev_name(dev);
    props->description = dev_desc(dev);
    props->type = dev_type(dev);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

ggml_backend_t dev_init(ggml_backend_dev_t dev, const char *) {
    static ggml_guid guid = { 0x6d, 0x6f, 0x65, 0x2d, 0x63, 0x61, 0x63, 0x68, 0x65, 0x2d, 0x70, 0x6c, 0x75, 0x67, 0x69, 0x6e };
    return new ggml_backend{ &guid, g_backend_i, dev, new moe_cache_backend_ctx() };
}

ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t) {
    if (getenv("MOE_CACHE_TRACE")) {
        fprintf(stderr, "moe-cache: get_buffer_type\n");
    }
    return get_buft();
}

bool layer_ok(const ggml_tensor * w) {
    state_t & s = S();
    int l = -1;
    if (sscanf(w->name, "blk.%d.", &l) != 1) {
        return s.lo == 0 && s.hi >= (1 << 30);
    }
    return l >= s.lo && l <= s.hi;
}

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    if (S().disabled) {
        return false;
    }
    if (getenv("MOE_CACHE_TRACE")) {
        fprintf(stderr, "moe-cache: supports_op %s src0=%s buf=%p\n", ggml_op_name(op->op), op->src[0] ? op->src[0]->name : "-", op->src[0] ? (void *) op->src[0]->buffer : nullptr);
    }
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GLU:
            return true;
        case GGML_OP_MUL_MAT_ID:
            break;
        default:
            return false;
    }
    const ggml_tensor * w = op->src[0];
    if (!w || !w->buffer || w->buffer->buft->iface.get_name != buft_name) {
        return false;
    }
    return w->ne[2] > 1 && layer_ok(w);
}

bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == buft_name;
}

const ggml_backend_device_i g_dev_i = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_desc,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init,
    /* .get_buffer_type      = */ dev_buft,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---- registry ----

const char * reg_name(ggml_backend_reg_t) {
    return "MOE_CACHE";
}

size_t reg_count(ggml_backend_reg_t) {
    return 1;
}

ggml_backend_dev_t reg_get(ggml_backend_reg_t reg, size_t) {
    static ggml_backend_device dev = { g_dev_i, nullptr, nullptr };
    dev.reg = reg;
    g_dev = &dev;
    return &dev;
}

void * reg_proc(ggml_backend_reg_t, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) set_n_threads;
    }
    return nullptr;
}

const ggml_backend_reg_i g_reg_i = {
    /* .get_name         = */ reg_name,
    /* .get_device_count = */ reg_count,
    /* .get_device       = */ reg_get,
    /* .get_proc_address = */ reg_proc,
};

}

extern "C" MOE_CACHE_EXPORT ggml_backend_reg_t ggml_backend_init(void) {
    static ggml_backend_reg reg = { GGML_BACKEND_API_VERSION, g_reg_i, nullptr };
    return &reg;
}

// LD_PRELOAD=libggml-moe-cache.so registers the backend at start-up. Needed for builds that link their backends directly,
// because llama.cpp only reads GGML_BACKEND_PATH when no backend is registered yet.
static bool g_registered = false;

extern "C" MOE_CACHE_EXPORT void moe_cache_register(void) {
    if (!g_registered) {
        g_registered = true;
        ggml_backend_register(ggml_backend_init());
    }
}

#ifndef _WIN32
// LD_PRELOAD loading (Linux builds that link their backends directly); on Windows the library is loaded through GGML_BACKEND_PATH
__attribute__((constructor)) static void moe_cache_autoload() {
    if (getenv("MOE_CACHE_PRELOAD") && !getenv("MOE_CACHE_DISABLE")) {
        const int api = moe_cache_api_of_ggml(ggml_version());
        if (api > 0 && api != GGML_BACKEND_API_VERSION) {   // registering would hand the server a struct of the wrong layout
            fprintf(stderr, "moe-cache: this plugin was built for ggml backend API %d but the server uses ggml %s (API %d): not loaded, the model loads the normal way. "
                            "Use the plugin build that matches your llama.cpp (docs/QUICKSTART.md)\n", GGML_BACKEND_API_VERSION, ggml_version(), api);
            return;
        }
        moe_cache_register();
    }
}
#endif

#if defined(__x86_64__) && defined(__linux__)
// ---- faster Q2_0 kernel (bit-identical to llama.cpp's) ----
// llama.cpp has no SIMD version of the Q2_0 x Q8_0 dot product on x86: it runs a plain C loop. This definition replaces it when the
// plugin is preloaded (LD_PRELOAD), because the function table of libggml-cpu is filled by symbol name. Results are bit-identical
// (tests/kernel_check.cpp compares them with the real library function). MOE_CACHE_FAST_KERNELS=0 switches it off.
#include <dlfcn.h>
#include "fast_q2_0.h"
extern "C" MOE_CACHE_EXPORT void ggml_vec_dot_q2_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    typedef void (*fn_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    static fn_t orig = (fn_t) dlsym(RTLD_NEXT, "ggml_vec_dot_q2_0_q8_0");
    static const bool fast = [] {
        const char * e = getenv("MOE_CACHE_FAST_KERNELS");
        return !(e && strcmp(e, "0") == 0) && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    if (fast && nrc == 1 && n > 0 && n % 64 == 0) {
        fastk::vec_dot_q2_0_q8_0_avx2<true>(n, s, vx, vy);
        return;
    }
    if (orig) {
        orig(n, s, bs, vx, bx, vy, by, nrc);
        return;
    }
    fprintf(stderr, "moe-cache: no Q2_0 dot product available\n");
    abort();
}

// ---- AVX-512 IQ2_XXS / IQ3_XXS dot products (MOE_CACHE_AVX512=1; off by default) ----
// See src/fast_iq_avx512.h. They are only used after a self-check that compares them bit-for-bit with llama.cpp's own function on random data;
// a single mismatch switches them off for the whole run. MOE_CACHE_AVX512=1 on a CPU without AVX-512 does nothing.
#include <random>
#include "fast_iq_avx512.h"
namespace {
typedef void (*iq_dot_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

int iq512_decide(iq_dot_t o2, iq_dot_t o3) {      // 0 off, 1 AVX-512, 2 AVX-512 + VNNI
    const char * e = getenv("MOE_CACHE_AVX512");
    if (!(e && atoi(e) != 0) || !o2 || !o3) {
        return 0;
    }
    if (!fastk512::cpu_has_avx512()) {
        fprintf(stderr, "moe-cache: MOE_CACHE_AVX512=1 but this CPU has no AVX-512 (BW, VL, DQ): not used\n");
        return 0;
    }
    const bool vnni = fastk512::cpu_has_avx512_vnni();
    std::mt19937 rng(12345);
    alignas(64) static fastk512::block_iq2_xxs w2[8];
    alignas(64) static fastk512::block_iq3_xxs w3[8];
    alignas(64) static fastk512::block_q8_K a[8];
    bool ok = true;
    for (int trial = 0; trial < 200 && ok; trial++) {
        for (auto & b : w2) { for (auto & q : b.qs) q = (uint16_t) rng(); b.d = (uint16_t) (0x2800 + rng() % 0x1000); }
        for (auto & b : w3) { for (auto & q : b.qs) q = (uint8_t) rng(); b.d = (uint16_t) (0x2800 + rng() % 0x1000); }
        for (auto & b : a) { b.d = 0.001f * (float) (1 + rng() % 100); for (auto & q : b.qs) q = (int8_t) ((int) (rng() % 255) - 127); }
        const int n = 8 * 256;
        float r2, r3, f2, f3;
        o2(n, &r2, 0, w2, 0, a, 0, 1);
        o3(n, &r3, 0, w3, 0, a, 0, 1);
        fastk512::dot_iq2_xxs(n, &f2, w2, a);
        fastk512::dot_iq3_xxs(n, &f3, w3, a);
        ok = memcmp(&r2, &f2, 4) == 0 && memcmp(&r3, &f3, 4) == 0;
        if (ok && vnni) {
            fastk512::dot_iq2_xxs_vnni(n, &f2, w2, a);
            fastk512::dot_iq3_xxs_vnni(n, &f3, w3, a);
            ok = memcmp(&r2, &f2, 4) == 0 && memcmp(&r3, &f3, 4) == 0;
        }
    }
    if (!ok) {
        fprintf(stderr, "moe-cache: AVX-512 IQ kernels FAILED the bit-exact self-check: using llama.cpp's kernels (please report this CPU)\n");
        return 0;
    }
    fprintf(stderr, "moe-cache: AVX-512 IQ2_XXS / IQ3_XXS kernels on (%s, self-check passed)\n", vnni ? "AVX-512 + VNNI" : "AVX-512");
    return vnni ? 2 : 1;
}

int iq512_mode() {
    static const int mode = iq512_decide((iq_dot_t) dlsym(RTLD_NEXT, "ggml_vec_dot_iq2_xxs_q8_K"), (iq_dot_t) dlsym(RTLD_NEXT, "ggml_vec_dot_iq3_xxs_q8_K"));
    return mode;
}
}  // namespace

static int iq512_probe_trampoline() { return iq512_mode(); }
static const int iq512_registered = (moe_cache_iq512_probe = &iq512_probe_trampoline, 0);

extern "C" MOE_CACHE_EXPORT void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    static iq_dot_t orig = (iq_dot_t) dlsym(RTLD_NEXT, "ggml_vec_dot_iq2_xxs_q8_K");
    const int m = iq512_mode();
    if (m && nrc == 1 && n > 0 && n % 256 == 0) {
        if (m == 2) fastk512::dot_iq2_xxs_vnni(n, s, vx, vy); else fastk512::dot_iq2_xxs(n, s, vx, vy);
        return;
    }
    orig(n, s, bs, vx, bx, vy, by, nrc);
}

extern "C" MOE_CACHE_EXPORT void ggml_vec_dot_iq3_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    static iq_dot_t orig = (iq_dot_t) dlsym(RTLD_NEXT, "ggml_vec_dot_iq3_xxs_q8_K");
    const int m = iq512_mode();
    if (m && nrc == 1 && n > 0 && n % 256 == 0) {
        if (m == 2) fastk512::dot_iq3_xxs_vnni(n, s, vx, vy); else fastk512::dot_iq3_xxs(n, s, vx, vy);
        return;
    }
    orig(n, s, bs, vx, bx, vy, by, nrc);
}

// ---- llama.cpp's GPU expert cache (--moe-cache-mib) on top of this cache: see gpu_cache_mode() ----
// Its uploads read the expert bytes straight from the weight memory, where only the experts we hold in RAM exist. Make them resident first.
extern "C" MOE_CACHE_EXPORT void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    typedef void (*fn_t)(ggml_backend_t, struct ggml_tensor *, const void *, size_t, size_t);
    static fn_t orig = (fn_t) dlsym(RTLD_NEXT, "ggml_backend_tensor_set_async");
    if (gpu_cache_mode()) {
        gpu_cache_ensure(data, size);
    }
    if (orig) {
        orig(backend, tensor, data, offset, size);
    } else {
        fprintf(stderr, "moe-cache: ggml_backend_tensor_set_async not found\n");
        abort();
    }
}
#endif
