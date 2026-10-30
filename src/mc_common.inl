// moe-cache plugin part: helpers, expert/tensor records, usage profiles, memory limits, read jobs.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

#ifdef __linux__
int g_stats_pipe[2] = { -1, -1 };
int g_idle_pipe[2] = { -1, -1 };            // SIGUSR2: the server is idle, give the cache's RAM back (N3, idle_release)
#endif
#ifdef _WIN32
HANDLE g_stats_event = nullptr;            // MOE_CACHE_STATS_SIGNAL on Windows: named event that asks for one snapshot
HANDLE g_save_event = nullptr;             // Windows: named event Local\moe-cache-save-<pid> that asks for a profile save (the server is killed, not stopped)
HANDLE g_idle_event = nullptr;             // Windows: named event Local\moe-cache-idle-<pid>: the server is idle, give the cache's RAM back (N3, idle_release)
#endif
#ifdef _WIN32
std::atomic<ggml_cgraph *> g_cpu_graph{nullptr};   // the graph the plugin is handing to the CPU backend right now (crash report)
#endif
#if defined(__linux__) || defined(_WIN32)
std::atomic<bool> g_stats_stop{false};   // set when the state is destroyed: the snapshot thread then stops without touching it
#endif
constexpr size_t PAGE = 4096;
// GGML_OP_GLU of the running ggml, read from a GLU node it builds at start-up (ggml_abi_mismatch): ggml master inserted TOP_K_MASK before it,
// so GLU is 101 there and 100 in the headers this plugin is built with. The ops below GLU that the plugin compares have not moved.
int g_op_glu = GGML_OP_GLU;

// The running ggml against the headers this plugin was built with: the op and type numbers the plugin compares, the tensor fields it reads and
// the threadpool parameters it passes. The backend API number only covers the backend structs, and ggml 0.27 (llama.cpp b11669) is not
// in its table at all, so a changed enum or field would give wrong output or a crash without a word. Returns "" when everything matches.
std::string ggml_abi_mismatch() {
    struct { int v; const char * name; } ops[] = {
        { GGML_OP_NONE, "NONE" }, { GGML_OP_MUL_MAT_ID, "MUL_MAT_ID" }, { GGML_OP_VIEW, "VIEW" },
        { GGML_OP_RESHAPE, "RESHAPE" }, { GGML_OP_PERMUTE, "PERMUTE" }, { GGML_OP_TRANSPOSE, "TRANSPOSE" },
    };
#ifdef MOE_CACHE_TEST_MUTANT_ABI      // CI only: a build that expects another op number must refuse to run
    ops[1].v = GGML_OP_MUL_MAT;
#endif
    for (const auto & o : ops) {
        if (strcmp(ggml_op_name((enum ggml_op) o.v), o.name) != 0) {
            return std::string("op ") + std::to_string(o.v) + " is " + ggml_op_name((enum ggml_op) o.v) + ", expected " + o.name;
        }
    }
    struct { ggml_type t; const char * name; } types[] = {
        { GGML_TYPE_F32, "f32" }, { GGML_TYPE_F16, "f16" }, { GGML_TYPE_BF16, "bf16" }, { GGML_TYPE_I32, "i32" }, { GGML_TYPE_Q8_0, "q8_0" },
        { GGML_TYPE_Q8_K, "q8_K" }, { GGML_TYPE_Q2_0, "q2_0" }, { GGML_TYPE_IQ1_S, "iq1_s" }, { GGML_TYPE_IQ1_M, "iq1_m" },
        { GGML_TYPE_IQ2_XXS, "iq2_xxs" }, { GGML_TYPE_IQ2_XS, "iq2_xs" }, { GGML_TYPE_IQ2_S, "iq2_s" }, { GGML_TYPE_IQ3_XXS, "iq3_xxs" },
        { GGML_TYPE_IQ3_S, "iq3_s" },
    };
    for (const auto & t : types) {
        if (strcmp(ggml_type_name(t.t), t.name) != 0) {
            return std::string("type ") + std::to_string((int) t.t) + " is " + ggml_type_name(t.t) + ", expected " + t.name;
        }
    }
    // tensor fields: build a small MUL_MAT_ID graph node with the running ggml and read it back through this plugin's struct definition
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return "ggml_init failed";
    }
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 6);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 3);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 3);
    ggml_set_name(b, "mc_abi");
    ggml_tensor * m = ggml_mul_mat_id(ctx, w, b, ids);
    ggml_tensor * v = ggml_view_2d(ctx, b, 8, 2, b->nb[1], b->nb[2]);
    ggml_tensor * g = ggml_swiglu(ctx, b);
    const bool tensor_ok = b->type == GGML_TYPE_F32 && b->ne[0] == 8 && b->ne[1] == 2 && b->ne[2] == 3 && b->ne[3] == 1 && b->nb[0] == 4 &&
                           b->nb[1] == 32 && b->nb[2] == 64 && b->op == GGML_OP_NONE && strcmp(b->name, "mc_abi") == 0 && ggml_nbytes(b) == 192 &&
                           m->op == GGML_OP_MUL_MAT_ID && m->src[0] == w && m->src[1] == b && m->src[2] == ids && m->ne[0] == 4 && m->ne[1] == 2 &&
                           m->ne[2] == 3 && v->op == GGML_OP_VIEW && v->view_src == b && v->view_offs == b->nb[2];
    const int glu = g->op;
    ggml_free(ctx);
    if (!tensor_ok) {
        return "the fields of ggml_tensor moved";
    }
    if (strcmp(ggml_op_name((enum ggml_op) glu), "GLU") != 0) {
        return std::string("a GLU node has op ") + std::to_string(glu) + " (" + ggml_op_name((enum ggml_op) glu) + ")";
    }
    if (glu != GGML_OP_GLU) {
        fprintf(stderr, "moe-cache: the running ggml numbers GLU %d (these headers: %d): using %d\n", glu, (int) GGML_OP_GLU, glu);
    }
    g_op_glu = glu;
    // threadpool parameters: filled by the running ggml into a buffer larger than this plugin's struct, which must end where the plugin thinks
    alignas(64) unsigned char buf[sizeof(ggml_threadpool_params) + 4096];
    memset(buf, 0xA5, sizeof(buf));
    ggml_threadpool_params & tp = *(ggml_threadpool_params *) buf;
    ggml_threadpool_params_init(&tp, 7);
    bool tail_ok = true;
    for (size_t i = sizeof(ggml_threadpool_params); i < sizeof(buf); i++) {
        tail_ok = tail_ok && buf[i] == 0xA5;
    }
    if (!tail_ok || tp.n_threads != 7 || tp.poll != 50 || tp.strict_cpu || tp.paused || tp.cpumask[0] || tp.cpumask[GGML_MAX_N_THREADS - 1]) {
        return "ggml_threadpool_params changed";
    }
    return "";
}

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
    float    crf = 0.0f;           // classlru: decayed use count (kept after eviction)
    uint32_t crf_tok = 0;          // classlru: decode token of the last scored use
    uint32_t last_tok = 0;         // classlru: decode token of the last use of any kind (age for the victim choice)
    uint8_t  cls = 0;              // classlru: which class list the node sits in (always 0 with the plain LRU policy)
    bool     resident = false;
    bool     loading = false;      // a lookahead read is still filling this expert
    bool     prefetched = false;   // loaded ahead of time and not used yet
    bool     mapped = false;       // not resident, but its range maps the model file (cache smaller than one op, see map_from_file)
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
    int     pack_rec = -1;    // BO-1: layer record of the .moepack file this tensor is stored in (-1: not in the pack) and its slot in the expert block
    int     pack_k = -1;
    bool    view = false;     // Windows file-view cache: data points into the plugin's read-only view of the model file (no reads, no commit)
    char *  buf_addr = nullptr;   // view: the address the buffer gave this tensor (forget_range matches the buffer by it)
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
    FILE * f = plat::fopen_utf8(path, "rb");
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
        for (float & v : tmp) {
            if (!std::isfinite(v) || v < 0.0f) {
                v = 0.0f;          // a damaged or hostile profile: NaN or negative counts would only mislead the warm start and the GPU bank
            }
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
// extra: memory this process holds that is neither available nor private (the Windows file-view cache's resident experts)
// Hard cap on what the server may use in all (owner, 2026-10-29: "nothing more than 28 GB of system RAM", Linux and Windows alike): physical RAM
// minus max(4 GiB, 12.5 %), = 28 GiB on a 32 GiB machine, so the OS, the desktop and the GPU driver keep room (Windows commit errors, Linux OOM).
// MOE_CACHE_MAX_RAM_GB=N (decimal GB, as the launcher's --max-ram) sets it; 0 turns it off. The cache budget is the limit minus the rest of the
// process (llama.cpp's host buffers, KV, compute), so the whole server stays under it. (size_t) -1 when off or unknown.
size_t ram_cap() {
    static const size_t cap = [] {
        if (const char * v = getenv("MOE_CACHE_MAX_RAM_GB")) {
            char * end = nullptr;
            const double gb = strtod(v, &end);
            if (end != v && gb >= 0) {        // also false for nan; a typo keeps the default cap below, as the launcher does
                return gb > 0 ? (size_t) (gb * 1e9) : (size_t) -1;
            }
            fprintf(stderr, "moe-cache: MOE_CACHE_MAX_RAM_GB=%s is not a number of GB: the default cap is used\n", v);
        }
        const size_t total = plat::mem_total();
        if (!total) {
            return (size_t) -1;
        }
        const size_t keep = std::max<size_t>((size_t) 4 << 30, total / 8);
        return total > keep ? total - keep : total / 2;
    }();
    return cap;
}

size_t read_mem_limit(size_t extra = 0) {
    const size_t lim = std::min(plat::mem_limit(), ram_cap());
    const size_t avail = plat::mem_available();
    return avail ? std::min(lim, avail + plat::rss_private() + extra) : lim;
}

size_t read_rss_anon() {
    return plat::rss_private();
}

// CPUs set in a hex mask such as "0x555" or "f0" (llama.cpp's -C format), lowest first
std::vector<int> mask_cpus(const char * m) {
    std::vector<int> out;
    if (m[0] == '0' && (m[1] | 32) == 'x') {
        m += 2;
    }
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

// router dot product: eight independent sums so the compiler can vectorise without -ffast-math; AVX2 version picked at run time.
// Used only to guess the next layer's experts (lookahead prefetch), never for the model's output.
static inline __attribute__((always_inline)) float dot_f32_body(const float * a, const float * b, size_t n) {
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
// ThreadSanitizer builds use it too (I9): target_clones makes an ifunc resolver, which the loader runs while it relocates a preloaded plugin,
// before any constructor, so before libtsan is set up; the resolver is instrumented and its __tsan_func_entry crashed (segfault, empty log)
#if defined(__SANITIZE_THREAD__)
#define MC_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define MC_TSAN 1
#endif
#endif
#if (defined(_WIN32) || defined(MC_TSAN)) && defined(__x86_64__)
// mingw has no ifunc, so no target_clones: the same two versions, picked once with the CPU's feature bits
__attribute__((target("avx2,fma"))) float dot_f32_avx2(const float * a, const float * b, size_t n) {
    return dot_f32_body(a, b, n);
}
float dot_f32_default(const float * a, const float * b, size_t n) {
    return dot_f32_body(a, b, n);
}
float dot_f32(const float * a, const float * b, size_t n) {
    static const bool avx2 = [] { __builtin_cpu_init(); return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }();
    return avx2 ? dot_f32_avx2(a, b, n) : dot_f32_default(a, b, n);
}
#else
#if defined(__linux__) && defined(__x86_64__)
__attribute__((target_clones("avx2,fma", "default")))
#endif
float dot_f32(const float * a, const float * b, size_t n) {
    return dot_f32_body(a, b, n);
}
#endif

struct io_job_t {
    char * dst;
    size_t off;
    size_t nb2;
    int    fdi;      // which model file (split GGUF)
    bool   direct;   // memory address matches the file offset (mod 4096): read straight into place
    const ggml_tensor * rtensor = nullptr;   // set: repack this expert while loading it
    int    node = -1;        // index in the node table (asynchronous loads clear its "loading" flag when done)
    struct pack_job_t * pk = nullptr;   // set: a read from the .moepack file that fills several slices of one expert (owned by whoever dequeues the job)
};

// BO-1: one expert's slices (gate, up, down...) read from the pack file with ONE unbuffered read, then copied to their destinations.
struct pack_job_t {
    uint64_t off = 0;        // pack file offset of the first byte to read (4096-aligned)
    size_t   len = 0;        // bytes to read (multiple of 4096)
    size_t   lo = 0;         // offset inside the expert block where `off` starts
    int      n = 0;          // slices wanted (2..4)
    io_job_t j[4];           // the per-slice jobs (destination, size, node, and the fallback if the pack read fails)
    size_t   in_off[4] = {}; // offset of each slice inside the block
};

inline uint32_t crc32_update(uint32_t crc, const void * data, size_t n) {
    static const std::vector<uint32_t> tab = [] {
        std::vector<uint32_t> t(256);
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) { c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; }
            t[i] = c;
        }
        return t;
    }();
    uint32_t c = crc ^ 0xffffffffu;
    const unsigned char * p = (const unsigned char *) data;
    for (size_t i = 0; i < n; i++) { c = tab[(c ^ p[i]) & 0xff] ^ (c >> 8); }
    return c ^ 0xffffffffu;
}

// Windows refused to commit memory the plugin must write (RAM + page file full, or another commit error): stop with a message that says so,
// instead of writing into pages that do not exist (a fault, or a read failing with error 998 that looks like a disk error)
[[noreturn]] inline void commit_fatal(const char * what, size_t bytes) {
    const unsigned long e = plat::commit_error();
    fprintf(stderr, "moe-cache: Windows refused %.1f MiB of memory for %s (error %lu%s). A larger page file, a smaller context, fewer GPU layers or a "
            "lower --ram leaves room; stopping here instead of writing into memory that does not exist\n", bytes / 1048576.0, what, e,
            e == 1455 ? ": RAM + page file are full" : "");
    abort();
}

}  // namespace
