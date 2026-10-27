// moe-cache plugin part: helpers, expert/tensor records, usage profiles, memory limits, read jobs.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

#ifdef __linux__
int g_stats_pipe[2] = { -1, -1 };
#endif
#ifdef _WIN32
HANDLE g_stats_event = nullptr;            // MOE_CACHE_STATS_SIGNAL on Windows: named event that asks for one snapshot
HANDLE g_save_event = nullptr;             // Windows: named event Local\moe-cache-save-<pid> that asks for a profile save (the server is killed, not stopped)
#endif
#ifdef _WIN32
std::atomic<ggml_cgraph *> g_cpu_graph{nullptr};   // the graph the plugin is handing to the CPU backend right now (crash report)
#endif
#if defined(__linux__) || defined(_WIN32)
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
size_t read_mem_limit(size_t extra = 0) {
    const size_t lim = plat::mem_limit();
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
#if defined(_WIN32) && defined(__x86_64__)
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
};

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
