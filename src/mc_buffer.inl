// moe-cache plugin part: buffer type (address space only) and the GPU expert cache hook.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

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
    if (S().cache_on) {
        S().forget_range((const char *) c->base, c->size);
    }
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
        if (s.register_tensor(t, data) >= 0) {
            return;
        }
    }
    plat::mem_commit((char *) t->data + off, size);
    memcpy((char *) t->data + off, data, size);
}

void gpu_cache_ensure(const void * p, size_t size);

void buf_get(ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t off, size_t size) {
    gpu_cache_ensure((const char *) t->data + off, size);     // expert bytes: load them first (an evicted expert has no data in place)
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
