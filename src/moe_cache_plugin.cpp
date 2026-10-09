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

#ifndef MOE_CACHE_TESTED_GGML
#define MOE_CACHE_TESTED_GGML "0.25.1"
#endif

#ifdef __linux__
#include <signal.h>
#endif
namespace {

#ifdef __linux__
int g_stats_pipe[2] = { -1, -1 };
#endif
constexpr size_t PAGE = 4096;

size_t round_up(size_t x, size_t a = PAGE) {
    return (x + a - 1) / a * a;
}

struct node_t {
    int      prev = -1;
    int      next = -1;
    uint64_t epoch = 0;
    bool     resident = false;
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
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    if (fp) {
        *fp = 0;
    }
    char magic[4];
    if (fread(magic, 1, 4, f) == 4 && memcmp(magic, "XUSE", 4) == 0) {
        int32_t nl = 0, ne = 0;
        if (fread(&nl, 4, 1, f) != 1 || fread(&ne, 4, 1, f) != 1 || nl <= 0 || ne <= 0 || nl > 4096 || ne > 65536) {
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
    rewind(f);
    std::vector<int16_t> pairs;
    int16_t v;
    while (fread(&v, sizeof(v), 1, f) == 1) {
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
    u.n_layers = ml;
    u.n_exp = me;
    u.c.assign((size_t) ml * me, 0.0);
    for (size_t i = 0; i < n; i++) {
        u.c[(size_t) pairs[2 * i] * me + pairs[2 * i + 1]] = (double) (n - i);
    }
    return true;
}

// memory limit of this process and its private memory (see platform.h)
size_t read_mem_limit() {
    return plat::mem_limit();
}

size_t read_rss_anon() {
    return plat::rss_private();
}

struct io_job_t {
    char * dst;
    size_t off;
    size_t nb2;
    int    fdi;      // which model file (split GGUF)
    bool   direct;   // memory address matches the file offset (mod 4096): read straight into place
    const ggml_tensor * rtensor = nullptr;   // set: repack this expert while loading it
};

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
    std::atomic<uint64_t> gpu_segments{0}, gpu_fallbacks{0};
    std::atomic<uint64_t> gpu_up_bytes{0};
    // live statistics: with MOE_CACHE_STATS_SIGNAL=1, SIGUSR1 makes the plugin print one "moe-cache-stats {json}" line (cumulative counters)
    std::atomic<uint64_t> hits_dec{0}, misses_dec{0}, evict_age_sum{0};
    bool cur_decode = false;                            // the group being prepared is a single-token (decode) step
    std::vector<uint64_t> layer_hits, layer_misses;     // per layer, updated under mu
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
        fprintf(stderr, "moe-cache: plugin v0.1.1 (ggml backend API %d), running with ggml %s (%s)\n", GGML_BACKEND_API_VERSION, ggml_version(), ggml_commit());
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
        direct_ok = getenv("MOE_CACHE_DIRECT") != nullptr;
        if (const char * v = getenv("MOE_CACHE_GPU_MIN_TOKENS")) {
            gpu_min_tokens = atoi(v);
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
                want = !gpu;
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
                limit_bytes = (size_t) (atof(v) * (1u << 30));
            }
            if (const char * v = getenv("MOE_CACHE_MARGIN_GIB")) {
                margin = (size_t) (atof(v) * (1u << 30));
            }
            const size_t guess = (size_t) 7 << 29;   // 3.5 GiB for everything that is not expert cache, until measured
            budget = limit_bytes > margin + guess + ((size_t) 1 << 30) ? limit_bytes - margin - guess : (size_t) 1 << 30;
            fprintf(stderr, "moe-cache: auto cache size, memory limit %.2f GiB, margin %.2f GiB, start budget %.2f GiB\n", limit_bytes / 1073741824.0, margin / 1073741824.0, budget / 1073741824.0);
        } else {
            budget = (size_t) (atof(gib) * (1u << 30));
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
        const int n_io = 5;
        for (int i = 0; i < n_io; i++) {
            workers.emplace_back([this, i] { worker_main(i + 1); });
        }
        fprintf(stderr, "moe-cache: cache mode, %.2f GiB, %zu tensors in %s, layers %d-%d\n", budget / 1073741824.0, file_tensors.size(), file, lo, hi);
#ifdef __linux__
        if (getenv("MOE_CACHE_STATS_SIGNAL")) {
            if (pipe(g_stats_pipe) == 0) {
                struct sigaction sa;
                memset(&sa, 0, sizeof(sa));
                sa.sa_handler = [](int) { const char c = 1; const ssize_t r = write(g_stats_pipe[1], &c, 1); (void) r; };   // async-signal-safe: only write()
                sa.sa_flags = SA_RESTART;
                sigaction(SIGUSR1, &sa, nullptr);
                std::thread([this] { char c; while (read(g_stats_pipe[0], &c, 1) == 1) { print_snapshot(); } }).detach();
            }
        }
#endif
    }

    // one line of cumulative counters on stderr (see MOE_CACHE_STATS_SIGNAL)
    void print_snapshot() {
        std::string j;
        {
            std::lock_guard<std::mutex> lk(mu);
            char b[1024];
            snprintf(b, sizeof(b), "{\"hits\":%llu,\"misses\":%llu,\"hits_dec\":%llu,\"misses_dec\":%llu,\"read_bytes\":%llu,\"evictions\":%llu,\"evict_age_sum\":%llu,"
                     "\"resident_bytes\":%llu,\"budget\":%llu,\"ns_prep\":%llu,\"ns_read\":%llu,\"ns_cpu\":%llu,\"ns_total\":%llu,\"graphs\":%llu,"
                     "\"direct\":%llu,\"bounce\":%llu,\"repack\":%llu,\"gpu_segments\":%llu,\"gpu_up_bytes\":%llu,\"fallbacks\":%llu,\"shrinks\":%llu,\"grows\":%llu",
                     (unsigned long long) hits, (unsigned long long) misses, (unsigned long long) hits_dec, (unsigned long long) misses_dec, (unsigned long long) bytes,
                     (unsigned long long) evictions, (unsigned long long) evict_age_sum, (unsigned long long) resident_bytes, (unsigned long long) budget,
                     (unsigned long long) ns_prep, (unsigned long long) ns_read, (unsigned long long) ns_cpu, (unsigned long long) ns_total, (unsigned long long) graphs,
                     (unsigned long long) direct_jobs, (unsigned long long) bounce_jobs, (unsigned long long) repack_jobs, (unsigned long long) gpu_segments,
                     (unsigned long long) gpu_up_bytes, (unsigned long long) read_fallbacks, (unsigned long long) tune_shrinks, (unsigned long long) tune_grows);
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
        if (cache_on && stats && gpu_dev) {
            fprintf(stderr, "moe-cache: long-prompt blocks computed on the GPU %llu (uploaded %.1f GB), fell back to the CPU %llu\n", (unsigned long long) gpu_segments, gpu_up_bytes / 1e9, (unsigned long long) gpu_fallbacks);
        }
        if (cache_on && stats) {
            const uint64_t h = hits, m = misses;
            if (read_fallbacks) {
                fprintf(stderr, "moe-cache: reads that needed the buffered fallback: %llu\n", (unsigned long long) read_fallbacks);
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
                const io_job_t & j = jobs[i];
                const size_t aoff = j.off / PAGE * PAGE;
                const size_t shift = j.off - aoff;
                const size_t len = round_up(shift + j.nb2);
                char * buf = j.direct ? j.dst - shift : bounce;   // direct: the aligned read lands exactly around the expert
                // the pages must exist before they are written (Windows); one extra page covers kernels that read a little past the end
                plat::mem_commit(j.direct ? j.dst - shift : j.dst, (j.direct ? len : j.nb2) + PAGE);
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
            fprintf(stderr, "moe-cache: cannot cache tensor %s (not in the GGUF or size mismatch)\n", t->name);
            abort();
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
        FILE * f = fopen(tmp.c_str(), "wb");
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
                while (v >= 0 && nodes[v].epoch == epoch) {
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
        unlink(s);
        nodes[s].resident = false;
        resident_bytes -= ti.nb2;
        evictions++;
    }

    // make sure all experts used by the ops of one group (same ids tensor) are resident
    void prepare_group(const std::vector<int> & tidx, const std::vector<int> & cnt) {
        std::lock_guard<std::mutex> lk(mu);
        epoch++;
        groups++;
        if (auto_mode && (groups % 32) == 1) {
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
        for (int ti_i : tidx) {
            tinfo_t & ti = *tensors[ti_i];
            max_nb2 = std::max(max_nb2, ti.nb2);
            for (int64_t e = 0; e < ti.n_expert; e++) {
                if (!cnt[(size_t) e]) {
                    continue;
                }
                const int s = ti.node0 + (int) e;
                node_t & n = nodes[s];
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
                    unlink(s);
                    push_mru(s);
                    n.epoch = epoch;
                    continue;
                }
                misses++;
                if (cur_decode) { misses_dec++; }
                while (resident_bytes + ti.nb2 > budget) {
                    int v = lru;
                    for (int k = 0; k < 100000 && v >= 0 && nodes[v].epoch == epoch; k++) {
                        v = nodes[v].prev;
                    }
                    if (v < 0 || nodes[v].epoch == epoch) {
                        fprintf(stderr, "moe-cache: cache too small for one op, raise MOE_CACHE_SIZE_GIB\n");
                        abort();
                    }
                    evict(v);
                }
                n.resident = true;
                n.epoch = epoch;
                resident_bytes += ti.nb2;
                push_mru(s);
                jobs.push_back({ ti.data + (size_t) e * ti.nb2, ti.file_off + (size_t) e * ti.nb2, ti.nb2, ti.fdi, ti.direct, ti.repack ? ti.tensor : nullptr });
                pending_bytes += ti.nb2;
                if (auto_mode && pending_bytes >= ((size_t) 256 << 20)) {
                    flush_reads(jobs, max_nb2);
                }
            }
        }
        flush_reads(jobs, max_nb2);
    }

    // read the queued experts, then (auto mode) look at the real memory use again
    void flush_reads(std::vector<io_job_t> & jobs, size_t max_nb2) {
        const bool did_reads = !jobs.empty();
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
        s.register_tensor(t);
        return;
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

bool buft_is_host(ggml_backend_buffer_type_t) {
    return false;
}

ggml_backend_dev_t g_dev = nullptr;

ggml_backend_buffer_type_t get_buft() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ buft_name,
            /* .alloc_buffer   = */ buft_alloc,
            /* .get_alignment  = */ buft_align,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ buft_alloc_size,
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
                if (v & (1 << b)) {
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
            for (int c = a; c <= b; c++) { r.push_back(c); }
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
                tpp.cpumask[pins[(size_t) i]] = true;
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
    if (!s.cache_on) {
        for (int i = 0; i < ggml_graph_n_nodes(cg); i++) {
            if (ggml_graph_node(cg, i)->op == GGML_OP_MUL_MAT_ID) {
                s.ops++;
            }
        }
    }
    struct group_t { int first; int last; int n_tok; };
    std::vector<group_t> groups_found;
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
            s.prepare_group(tidx, used);
            if (!s.profile_path.empty() && ++s.groups_since_save >= 100000) {
                s.groups_since_save = 0;
                s.save_profile();
            }
            s.ns_prep += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - p0).count();
        }
    }

    // long prompts: run the expert block on the GPU (what llama.cpp's own offload does for batches of 32+ tokens)
    std::vector<std::pair<ggml_tensor *, ggml_op>> silenced;
    if (s.cache_on && s.gpu_dev && !c->gpu_failed) {
        for (const auto & gr : groups_found) {
            if (gr.n_tok < s.gpu_min_tokens) {
                continue;
            }
            std::vector<ggml_tensor *> seg;
            if (gpu_segment(c, s, cg, gr.first, gr.last, seg)) {
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
    static ggml_guid guid = { 0x53, 0x54, 0x52, 0x41, 0x54, 0x41, 0x2d, 0x54, 0x37, 0x2d, 0x70, 0x6c, 0x75, 0x67, 0x69, 0x6e };
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
        moe_cache_register();
    }
}
#endif
