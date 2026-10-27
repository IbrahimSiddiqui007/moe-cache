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

ggml_status buf_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t) {
    if (S().cache_on && !(t->ne[2] > 1 && strncmp(t->name, "blk.", 4) == 0)) {
        buf_commit_once((moe_cache_buffer_ctx *) buffer->context, t->data, ggml_nbytes(t));
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
    return ggml_backend_buffer_init(buft, g_buf_i, new moe_cache_buffer_ctx{ p, size, {}, {} }, size);
}

size_t buft_align(ggml_backend_buffer_type_t) {
    return 64;
}

// MOE_CACHE_GPU_CACHE=1: llama.cpp's own GPU expert cache (--moe-cache-mib) only caches experts of layers whose weights sit in a host buffer and
// uploads them by reading the weight memory. Our buffer then says "host", and the upload (ggml_backend_tensor_set_async, called from libllama)
// is intercepted to make the experts being read resident first. Preloaded (LD_PRELOAD + MOE_CACHE_PRELOAD=1) that is symbol interposition
// (mc_interpose.inl). Loaded as a file (GGML_BACKEND_PATH: install.sh, Vulkan, official and all Windows builds; idea I28) the plugin rewrites the
// import entry of that function in every other loaded library instead (Windows: import address table, Linux: GOT), see install_upload_hook().
// Without either, the GPU would upload experts that are not in RAM: wrong tokens (mini PC, T7: 0/4 identical), so the mode is refused then.
int install_upload_hook();
bool gpu_cache_mode() {
    static const bool on = [] {
        const char * e = getenv("MOE_CACHE_GPU_CACHE");
        if (!(e && atoi(e) != 0)) {
            return false;
        }
        if (getenv("MOE_CACHE_PRELOAD")) {
            return true;
        }
        const int n = install_upload_hook();
        if (n <= 0) {
            fprintf(stderr, "moe-cache: MOE_CACHE_GPU_CACHE=1 ignored: the plugin was loaded through GGML_BACKEND_PATH and found no call of "
                            "ggml_backend_tensor_set_async to route through it, so the GPU would read experts that are not loaded\n");
            return false;
        }
        fprintf(stderr, "moe-cache: GPU expert cache with the plugin loaded as a file: llama.cpp's expert uploads go through the plugin (%d import "
                        "entr%s rewritten)\n", n, n == 1 ? "y" : "ies");
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

// ---- the upload hook for path loading (idea I28) ----
typedef void (*set_async_fn_t)(ggml_backend_t, struct ggml_tensor *, const void *, size_t, size_t);
set_async_fn_t g_set_async_orig = nullptr;

void hooked_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    S().hooked_uploads++;
#ifndef MOE_CACHE_TEST_MUTANT_NO_UPLOAD_ENSURE   // test only: without this the GPU cache must get wrong experts (proves the GPU test covers it)
    gpu_cache_ensure(data, size);
#endif
    g_set_async_orig(backend, tensor, data, offset, size);
}

#if defined(_WIN32)
// every import address table entry of every other module that holds ggml-base's ggml_backend_tensor_set_async now points to the hook.
// Matched by address (not by DLL or function name), so a renamed ggml-base or an import by ordinal is found too.
int install_upload_hook() {
    HMODULE base = GetModuleHandleA("ggml-base.dll");
    if (!base || !(g_set_async_orig = (set_async_fn_t) (void *) GetProcAddress(base, "ggml_backend_tensor_set_async"))) {
        return 0;
    }
    HMODULE self = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) (void *) &hooked_tensor_set_async, &self);
    HMODULE mods[1024];
    DWORD need = 0;
    if (!EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &need)) {
        return 0;
    }
    int n = 0;
    for (DWORD m = 0; m < need / sizeof(HMODULE) && m < 1024; m++) {
        char * img = (char *) mods[m];
        if (mods[m] == self || mods[m] == base) {
            continue;
        }
        const IMAGE_DOS_HEADER * dos = (const IMAGE_DOS_HEADER *) img;
        if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
            continue;
        }
        const IMAGE_NT_HEADERS * nt = (const IMAGE_NT_HEADERS *) (img + dos->e_lfanew);
        const IMAGE_DATA_DIRECTORY & dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        if (!dir.VirtualAddress || !dir.Size) {
            continue;
        }
        for (const IMAGE_IMPORT_DESCRIPTOR * imp = (const IMAGE_IMPORT_DESCRIPTOR *) (img + dir.VirtualAddress); imp->Name; imp++) {
            for (IMAGE_THUNK_DATA * ft = (IMAGE_THUNK_DATA *) (img + imp->FirstThunk); ft->u1.Function; ft++) {
                if (ft->u1.Function != (ULONG_PTR) (void *) g_set_async_orig) {
                    continue;
                }
                DWORD old = 0;
                if (VirtualProtect(&ft->u1.Function, sizeof(ft->u1.Function), PAGE_READWRITE, &old)) {
                    ft->u1.Function = (ULONG_PTR) (void *) &hooked_tensor_set_async;
                    VirtualProtect(&ft->u1.Function, sizeof(ft->u1.Function), old, &old);
                    n++;
                }
            }
        }
    }
    return n;
}
#elif defined(__linux__) && defined(__x86_64__)
// the GOT entries (PLT jump slots and GLOB_DAT) of every other loaded object that name ggml_backend_tensor_set_async now point to the hook.
// The original is the one the global scope resolves to (libggml-base; a plugin loaded by path is not in that scope).
int install_upload_hook() {
    g_set_async_orig = (set_async_fn_t) dlsym(RTLD_DEFAULT, "ggml_backend_tensor_set_async");
    Dl_info self_info;
    if (!g_set_async_orig || !dladdr((void *) &hooked_tensor_set_async, &self_info)) {
        return 0;
    }
    Dl_info base_info;
    const void * base_addr = dladdr((void *) g_set_async_orig, &base_info) ? base_info.dli_fbase : nullptr;
    struct ctx_t { const void * self; const void * base; int n; } ctx = { self_info.dli_fbase, base_addr, 0 };
    dl_iterate_phdr([](struct dl_phdr_info * info, size_t, void * data) {
        ctx_t & c = *(ctx_t *) data;
        const uintptr_t lb = (uintptr_t) info->dlpi_addr;
        const ElfW(Dyn) * dyn = nullptr;
        uintptr_t first = UINTPTR_MAX, relro0 = 0, relro1 = 0;
        for (int i = 0; i < info->dlpi_phnum; i++) {
            if (info->dlpi_phdr[i].p_type == PT_GNU_RELRO) {
                relro0 = lb + info->dlpi_phdr[i].p_vaddr;
                relro1 = relro0 + info->dlpi_phdr[i].p_memsz;
            }
            if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
                dyn = (const ElfW(Dyn) *) (lb + info->dlpi_phdr[i].p_vaddr);
            }
            if (info->dlpi_phdr[i].p_type == PT_LOAD) {
                first = std::min<uintptr_t>(first, lb + info->dlpi_phdr[i].p_vaddr);
            }
        }
        const uintptr_t start = first / 4096 * 4096;
        if (!dyn || (const void *) start == c.self || (const void *) start == c.base || (c.self && lb == (uintptr_t) c.self) || (c.base && lb == (uintptr_t) c.base)) {
            return 0;
        }
        // ld.so has already turned these into addresses for most objects; the main program's or a vDSO's may still be offsets
        auto addr = [lb](ElfW(Addr) v) { return v < lb ? lb + v : v; };
        const ElfW(Sym) * symtab = nullptr;
        const char * strtab = nullptr;
        const ElfW(Rela) * jmprel = nullptr, * rela = nullptr;
        size_t jmprelsz = 0, relasz = 0;
        for (const ElfW(Dyn) * d = dyn; d->d_tag != DT_NULL; d++) {
            switch (d->d_tag) {
                case DT_SYMTAB:   symtab = (const ElfW(Sym) *) addr(d->d_un.d_ptr); break;
                case DT_STRTAB:   strtab = (const char *) addr(d->d_un.d_ptr); break;
                case DT_JMPREL:   jmprel = (const ElfW(Rela) *) addr(d->d_un.d_ptr); break;
                case DT_PLTRELSZ: jmprelsz = d->d_un.d_val; break;
                case DT_RELA:     rela = (const ElfW(Rela) *) addr(d->d_un.d_ptr); break;
                case DT_RELASZ:   relasz = d->d_un.d_val; break;
                default: break;
            }
        }
        if (!symtab || !strtab) {
            return 0;
        }
        auto scan = [&](const ElfW(Rela) * r, size_t sz) {
            for (size_t i = 0; r && i < sz / sizeof(ElfW(Rela)); i++) {
                const unsigned type = ELF64_R_TYPE(r[i].r_info);
                if (type != R_X86_64_JUMP_SLOT && type != R_X86_64_GLOB_DAT) {
                    continue;
                }
                const ElfW(Sym) & sym = symtab[ELF64_R_SYM(r[i].r_info)];
                if (strcmp(strtab + sym.st_name, "ggml_backend_tensor_set_async") != 0) {
                    continue;
                }
                void ** slot = (void **) (lb + r[i].r_offset);
                char * page = (char *) ((uintptr_t) slot / 4096 * 4096);
                const bool in_relro = (uintptr_t) slot >= relro0 && (uintptr_t) slot < relro1;   // read-only after relocation (-z relro / now)
                if (in_relro ? mprotect(page, 4096, PROT_READ | PROT_WRITE) == 0 : true) {
                    *slot = (void *) &hooked_tensor_set_async;
                    if (in_relro) {
                        mprotect(page, 4096, PROT_READ);     // a writable .got.plt stays as it was: lazy binding still writes its other slots
                    }
                    c.n++;
                }
            }
        };
        scan(jmprel, jmprelsz);
        scan(rela, relasz);
        return 0;
    }, &ctx);
    return ctx.n;
}
#else
int install_upload_hook() {
    return 0;
}
#endif

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
