// moe-cache plugin part: buffer type (address space only) and the GPU expert cache hook.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

// ---- buffer type ----

struct moe_cache_buffer_ctx {
    void * base;
    size_t size;
    std::vector<uint8_t> committed;   // Windows, cache mode: one flag per 64 KiB chunk already committed (see buf_init_tensor)
    std::mutex cmu;
};

// In cache mode the buffer is reserved address space only. On Linux untouched pages appear zero-filled on first touch; on Windows a reserved page
// faults on any access until it is committed. Expert tensors get their pages from the reads; every other tensor in the buffer (the scheduler's
// compute buffer: node outputs that ggml's kernels write through tensor->data, with no buffer call) is committed here, once per 64 KiB chunk.
// Found on the MSI Claw (RESULTS 50): cache mode crashed in the first graph with an access violation inside the compute buffer.
void buf_commit_once(moe_cache_buffer_ctx * c, const void * p, size_t size) {
    if (!plat::is_windows || size == 0) {
        return;
    }
    constexpr size_t CH = 64 * 1024;
    const uintptr_t base = (uintptr_t) c->base;
    const uintptr_t a = (uintptr_t) p;
    if (a < base || a + size > base + c->size) {
        return;
    }
    const size_t c0 = (a - base) / CH, c1 = (a + size - 1 - base) / CH;
    std::lock_guard<std::mutex> lk(c->cmu);
    if (c->committed.empty()) {
        c->committed.assign(c->size / CH + 1, 0);
    }
    for (size_t i = c0; i <= c1; i++) {
        if (!c->committed[i]) {
            const size_t len = std::min(CH, c->size - i * CH);
            if (!plat::mem_commit((char *) c->base + i * CH, len)) {
                commit_fatal("llama.cpp's compute buffer or a non-expert weight", len);
            }
            c->committed[i] = 1;
        }
    }
}

// ggml_backend_buffer's layout is private to ggml and changes without a backend API bump: llama.cpp master 418f905a4 (2026-10-29) appended three
// slots to its interface, which moved `buft` and `context` 24 bytes; a plugin built with older headers read the wrong fields and crashed in
// llama-server's memory-fit pass (CI master row, 2026-10-30). So the plugin reads no field of a ggml buffer: the type through
// ggml_backend_buffer_get_type(), our own context through this registry, and buffer_init_padded() hands ggml an interface followed by zeros.
std::mutex g_bufs_mu;
std::vector<std::pair<ggml_backend_buffer_t, moe_cache_buffer_ctx *>> g_bufs;   // a handful of buffers, created at load

std::atomic<uint64_t> g_bufs_gen{1};     // bumped whenever the registry changes: invalidates the per-thread last-hit cache (idea B05)

moe_cache_buffer_ctx * buf_ctx(ggml_backend_buffer_t buffer) {
    static thread_local ggml_backend_buffer_t last_buf = nullptr;
    static thread_local moe_cache_buffer_ctx * last_ctx = nullptr;
    static thread_local uint64_t last_gen = 0;
    if (last_buf == buffer && last_gen == g_bufs_gen.load(std::memory_order_acquire)) {
        return last_ctx;
    }
    std::lock_guard<std::mutex> lk(g_bufs_mu);
    for (const auto & b : g_bufs) {
        if (b.first == buffer) {
            last_buf = buffer;
            last_ctx = b.second;
            last_gen = g_bufs_gen.load(std::memory_order_acquire);
            return b.second;
        }
    }
    fprintf(stderr, "moe-cache: internal error: unknown buffer %p\n", (void *) buffer);
    abort();
}

// ggml_backend_buffer_init takes the interface by value; a newer ggml with a longer interface reads past the end of ours. Our interface followed by
// zeros makes every slot it added NULL ("optional, not supported"). Same calling convention for both struct sizes: SysV x86-64 copies a large struct
// onto the stack (the callee reads its own size from the start), Win64 passes a pointer to the caller's copy; the other arguments keep their registers.
struct buffer_i_padded {
    ggml_backend_buffer_i i;
    void * spare[24];
};

ggml_backend_buffer_t buffer_init_padded(ggml_backend_buffer_type_t buft, const ggml_backend_buffer_i & i, void * context, size_t size) {
    buffer_i_padded p;
    memset(&p, 0, sizeof(p));
    p.i = i;
    using init_fn = ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t, buffer_i_padded, void *, size_t);
    return ((init_fn) (void *) &ggml_backend_buffer_init)(buft, p, context, size);
}

const char * buft_name(ggml_backend_buffer_type_t) {
    return "MOE_CACHE";
}

void buf_free(ggml_backend_buffer_t buffer) {
    auto * c = buf_ctx(buffer);
    {
        std::lock_guard<std::mutex> lk(g_bufs_mu);
        for (size_t k = 0; k < g_bufs.size(); k++) {
            if (g_bufs[k].first == buffer) {
                g_bufs.erase(g_bufs.begin() + (long) k);
                g_bufs_gen.fetch_add(1, std::memory_order_release);
                break;
            }
        }
    }
    if (S().cache_on) {
        S().forget_range((const char *) c->base, c->size);
    }
    plat::mem_release(c->base, c->size);
    delete c;
}

void * buf_base(ggml_backend_buffer_t buffer) {
    return buf_ctx(buffer)->base;
}

void buf_memset(ggml_backend_buffer_t, ggml_tensor * t, uint8_t v, size_t off, size_t size) {
    if (!plat::mem_commit((char *) t->data + off, size)) {
        commit_fatal(t->name, size);
    }
    memset((char *) t->data + off, v, size);
}

void buf_set(ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t off, size_t size) {
    state_t & s = S();
    if (s.cache_on && t->ne[2] > 1 && strncmp(t->name, "blk.", 4) == 0 && off == 0 && size == ggml_nbytes(t)) {
        std::lock_guard<std::mutex> lk(s.mu);
        if (s.register_tensor(t, data) >= 0) {
            return;
        }
    }
    if (!plat::mem_commit((char *) t->data + off, size)) {
        commit_fatal(t->name, size);
    }
    memcpy((char *) t->data + off, data, size);
}

void gpu_cache_ensure(const void * p, size_t size);

void buf_get(ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t off, size_t size) {
    gpu_cache_ensure((const char *) t->data + off, size);     // expert bytes: load them first (an evicted expert has no data in place)
    memcpy(data, (const char *) t->data + off, size);
}

void buf_clear(ggml_backend_buffer_t buffer, uint8_t v) {
    auto * c = buf_ctx(buffer);
    if (S().cache_on) {
        return;   // the buffer is address space only; experts get their data from the file
    }
    memset(c->base, v, c->size);
}

// expert tensors get 12 KiB of padding so that init_tensor can move the data to an address that matches its file offset (mod 4096)
bool is_expert_tensor(const ggml_tensor * t) {
    return S().cache_on && S().direct_ok && t->ne[2] > 1 && strncmp(t->name, "blk.", 4) == 0;
}

ggml_status buf_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t) {
    if (S().cache_on && !(t->ne[2] > 1 && strncmp(t->name, "blk.", 4) == 0)) {
        buf_commit_once(buf_ctx(buffer), t->data, ggml_nbytes(t));
    }
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
        if (!plat::mem_commit(p, size)) {   // pass-through mode: the weights are copied in, all pages are needed
            plat::mem_release(p, size);
            return nullptr;              // llama.cpp reports the failed allocation
        }
    }
    auto * c = new moe_cache_buffer_ctx{ p, size, {}, {} };
    ggml_backend_buffer_t b = buffer_init_padded(buft, g_buf_i, c, size);
    std::lock_guard<std::mutex> lk(g_bufs_mu);
    g_bufs.emplace_back(b, c);
    g_bufs_gen.fetch_add(1, std::memory_order_release);
    return b;
}

size_t buft_align(ggml_backend_buffer_type_t) {
    return 64;
}

// MOE_CACHE_GPU_CACHE=1: llama.cpp's own GPU expert cache (--moe-cache-mib) only caches experts of layers whose weights sit in a host buffer and
// uploads them by reading the weight memory. Our buffer then says "host", and the upload is intercepted (ggml_backend_tensor_set_async below) to
// make the experts being read resident first. That interception is symbol interposition, which only works when the plugin is preloaded
// (LD_PRELOAD + MOE_CACHE_PRELOAD=1); loaded through GGML_BACKEND_PATH (install.sh, Vulkan, official and all Windows builds) the hook never runs
// and the GPU would upload experts that are not in RAM: wrong tokens (found on the mini PC, T7: 0/4 identical). So the mode is refused there.
bool gpu_cache_mode() {
    static const bool on = [] {
        const char * e = getenv("MOE_CACHE_GPU_CACHE");
        if (!(e && atoi(e) != 0)) {
            return false;
        }
        if (!getenv("MOE_CACHE_PRELOAD")) {
            fprintf(stderr, "moe-cache: MOE_CACHE_GPU_CACHE=1 ignored: it needs LD_PRELOAD loading (MOE_CACHE_PRELOAD=1); "
                            "this plugin was loaded through GGML_BACKEND_PATH, where the GPU would read experts that are not loaded\n");
            return false;
        }
        return true;
    }();
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
    if (!ti.data || ti.n_expert <= 0) {
        return;          // forgotten (its buffer was freed): the range table above is rebuilt only when tensors are added
    }
    const size_t e0 = (a - (uintptr_t) ti.data) / ti.nb2;
    size_t e1 = (a + size - 1 - (uintptr_t) ti.data) / ti.nb2;
    e1 = std::min(e1, (size_t) ti.n_expert - 1);
    std::vector<int> cnt((size_t) ti.n_expert, 0);
    for (size_t e = e0; e <= e1; e++) {
        cnt[e] = 1;
    }
    // idea I28 kill check (MOE_CACHE_GPU_ENSURE_STATS=1): how many of the experts llama.cpp's GPU cache uploads are not in RAM at that moment.
    // Near zero would let a build that loads the plugin by path (Windows, official releases) keep --gpu-cache with a cheap fallback.
    static const bool ens_stats = getenv("MOE_CACHE_GPU_ENSURE_STATS") != nullptr;
    if (ens_stats) {
        static std::atomic<uint64_t> calls{0}, wanted{0}, absent{0};
        uint64_t miss = 0;
        {
            std::lock_guard<std::mutex> lk(s.mu);
            for (size_t e = e0; e <= e1; e++) {
                miss += s.nodes[ti.node0 + (int) e].resident ? 0 : 1;
            }
        }
        const uint64_t c = ++calls, w = (wanted += e1 - e0 + 1), m = (absent += miss);
        if ((c & 1023) == 0) {
            fprintf(stderr, "moe-cache: gpu-cache uploads: %llu calls, %llu experts, %llu not in RAM (%.2f %%)\n", (unsigned long long) c,
                    (unsigned long long) w, (unsigned long long) m, w ? 100.0 * m / w : 0.0);
        }
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

}  // namespace
