// moe-cache plugin part: executor: decode-time expert matrix products run by the plugin.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

// ---- executor: decode-time expert matrix products run by the plugin itself ----
// the CPU backend's type table, looked up at run time: builds that load their CPU backend as a separate library do not link it into the plugin
typedef const struct ggml_type_traits_cpu * (*cpu_traits_fn_t)(enum ggml_type);
#ifdef _WIN32
std::string g_cpu_traits_from = "none";   // which module the executor's dot products come from (shown in the "active:" line)
#endif
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
#ifdef _WIN32
            if (fn) {
                g_cpu_traits_from = "CPU backend registry";
            }
#endif
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
                    if (fn) {
                        const char * b = std::max(strrchr(name, '\\'), strrchr(name, '/'));
                        g_cpu_traits_from = b ? b + 1 : name;   // e.g. ggml-cpu-zen4.dll: the same kernels stock llama.cpp runs
                    }
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

#ifdef MC_X86_KERNELS
// Q2_0 weights: the executor rearranges the quantised activation column once and uses the permuted-layout kernel (bit-identical, about 2x the other one)
void q2_0_perm_dot(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    fastk::vec_dot_q2_0_perm_avx2(n, s, bs, vx, bx, vy, by, nrc);
}
// Q2_0 weights on a llama.cpp with its own AVX2 kernel (#30447): the activation's lane sums are prepared once per column (exact, N20)
void q2_0_ys_dot(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    fastk::vec_dot_q2_0_ys_avx2(n, s, bs, vx, bx, vy, by, nrc);
}
// IQ3_XXS weights (GLM-5.3-Flash down projections): gather-based dot product, bit-identical, about 1.2x; Intel CPUs only by default (slow gathers
// on AMD Zen 2/3, and 0.58-0.64x on a Zen 4 7950X); MOE_CACHE_IQ3_KERNEL=1 uses it on any AVX2 CPU, =0 turns it off
ggml_vec_dot_t fast_iq3_dot() {
    static const ggml_vec_dot_t fn = [] {
        bool vnni = false;
        const char * e = getenv("MOE_CACHE_FAST_KERNELS");
        const char * k = getenv("MOE_CACHE_IQ3_KERNEL");
        if ((e && strcmp(e, "0") == 0) || (k && strcmp(k, "0") == 0) || !fastk3::usable(vnni, k && strcmp(k, "1") == 0)) {
            return (ggml_vec_dot_t) nullptr;
        }
        const ggml_vec_dot_t mine = vnni ? (ggml_vec_dot_t) fastk3::vec_dot_iq3_xxs_vnni : (ggml_vec_dot_t) fastk3::vec_dot_iq3_xxs_avx2;
        // bit-identical only to upstream's AVX2 function: compare with the running library's own on random blocks first, as for Q2_0 (a llama.cpp
        // built without AVX2, or a later change to that kernel, would otherwise give other tokens than stock without a word; audit 2026-10-29 N3)
        const ggml_type_traits_cpu * tt = cpu_traits(GGML_TYPE_IQ3_XXS);
        if (!tt || !tt->vec_dot) {
            return (ggml_vec_dot_t) nullptr;
        }
        std::vector<fastk512::block_iq3_xxs> w(8);
        std::vector<fastk512::block_q8_K> a(8);
        uint32_t r = 54321;
        auto rnd = [&]() { r = r * 1664525u + 1013904223u; return r >> 8; };
        for (int trial = 0; trial < 64; trial++) {
            for (auto & b : w) { for (auto & q : b.qs) q = (uint8_t) rnd(); b.d = (uint16_t) (0x2800 + rnd() % 0x1000); }
            for (auto & b : a) { b.d = 0.001f * (float) (1 + rnd() % 100); for (auto & q : b.qs) q = (int8_t) ((int) (rnd() % 255) - 127); }
            float x = 0.0f, y = 0.0f;
            tt->vec_dot(8 * 256, &x, 0, w.data(), 0, a.data(), 0, 1);
            mine(8 * 256, &y, 0, w.data(), 0, a.data(), 0, 1);
            if (memcmp(&x, &y, sizeof(float)) != 0) {
                fprintf(stderr, "moe-cache: this llama.cpp's IQ3_XXS kernel gives other bits than the plugin's: llama.cpp's own is used\n");
                return (ggml_vec_dot_t) nullptr;
            }
        }
        return mine;
    }();
    return fn;
}
}   // namespace
#ifdef __linux__
extern "C" MOE_CACHE_EXPORT void ggml_vec_dot_q2_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
#endif
// Which Q2_0 x Q8_0 dot product the CPU library has decides which of the plugin's kernels may replace it, because each must give the same bits:
//   the plain C loop (llama.cpp up to 23b0202a1)      -> the permuted-layout kernel (and, preloaded on Linux, the interposed one)
//   its own AVX2 kernel (#30447: b11669, master)      -> the column-sum kernel, same lanes and order as upstream's (N20)
//   anything else                                     -> llama.cpp's own, unchanged
// Found once by comparing the library's function with each reference on random blocks, bit for bit. Same on Linux and Windows; only where the
// library's function comes from differs.
typedef void (*q2_dot_fn_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
static q2_dot_fn_t q2_0_library_fn() {
    q2_dot_fn_t lib = nullptr;
#ifdef __linux__
    lib = (q2_dot_fn_t) dlsym(RTLD_NEXT, "ggml_vec_dot_q2_0_q8_0");             // preloaded: llama.cpp's own symbol behind ours
    if (lib == (q2_dot_fn_t) &ggml_vec_dot_q2_0_q8_0) {
        lib = nullptr;
    }
#endif
    if (!lib) {
        const ggml_type_traits_cpu * tt = cpu_traits(GGML_TYPE_Q2_0);           // loaded as a backend (and Windows): the CPU backend's table
#ifdef __linux__
        lib = tt && (q2_dot_fn_t) tt->vec_dot != (q2_dot_fn_t) &ggml_vec_dot_q2_0_q8_0 ? (q2_dot_fn_t) tt->vec_dot : nullptr;
#else
        lib = tt ? (q2_dot_fn_t) tt->vec_dot : nullptr;
#endif
    }
    return lib;
}
template <typename REF>
static bool q2_0_library_matches(q2_dot_fn_t lib, REF ref) {
    const int n = 64 * 41;                // odd block count: the tail of upstream's two-accumulator loop too
    std::vector<fastk::block_q2_0> x((size_t) n / 64);
    std::vector<fastk::block_q8_0> y((size_t) n / 32);
    uint32_t r = 12345;
    auto rnd = [&]() { r = r * 1664525u + 1013904223u; return r >> 8; };
    for (int trial = 0; trial < 64; trial++) {
        for (auto & b : x) {
            b.d = (uint16_t) (0x2000 + rnd() % 0x2000);
            for (auto & q : b.qs) q = (uint8_t) rnd();
        }
        for (auto & b : y) {
            b.d = (uint16_t) (0x2000 + rnd() % 0x2000);
            for (auto & q : b.qs) q = (int8_t) (rnd() % 255 - 127);
        }
        float a = 0.0f, b = 0.0f;
        lib(n, &a, 0, x.data(), 0, y.data(), 0, 1);
        ref(n, &b, x.data(), y.data());
        if (memcmp(&a, &b, sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}
bool q2_0_library_is_plain() {
    static const bool plain = [] {
        const q2_dot_fn_t lib = q2_0_library_fn();
        if (!lib) {
            return true;                  // nothing to compare with: keep the measured behaviour of the supported versions
        }
        if (!q2_0_library_matches(lib, fastk::vec_dot_q2_0_q8_0_avx2<true>)) {
            fprintf(stderr, "moe-cache: this llama.cpp has its own Q2_0 kernel (results differ from the plain C loop)\n");
            return false;
        }
        return true;
    }();
    return plain;
}
// true when the library's Q2_0 dot is upstream's AVX2 kernel (#30447), bit for bit: the executor may then use the column-sum kernel
bool q2_0_library_is_upstream() {
    static const bool up = [] {
        const q2_dot_fn_t lib = q2_0_library_fn();
        if (!lib || q2_0_library_is_plain() || !__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") || !__builtin_cpu_supports("f16c")) {
            return false;
        }
        const bool same = q2_0_library_matches(lib, fastk::vec_dot_q2_0_up_avx2);
        fprintf(stderr, "moe-cache: Q2_0: %s\n", same ? "llama.cpp's AVX2 kernel (#30447): the executor uses an exact faster version of it"
                                                     : "llama.cpp's kernel is neither the plain loop nor #30447's AVX2 one: using it unchanged");
        return same;
    }();
    return up;
}
namespace {
bool fast_kernels_enabled() {
    static const bool on = [] {
        const char * e = getenv("MOE_CACHE_FAST_KERNELS");
        return !(e && strcmp(e, "0") == 0) && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") &&
               q2_0_library_is_plain();
    }();
    return on;
}
bool upstream_q2_0_enabled() {
    static const bool on = [] {
        const char * e = getenv("MOE_CACHE_FAST_KERNELS");
        return !(e && strcmp(e, "0") == 0) && q2_0_library_is_upstream();
    }();
    return on;
}
#endif

bool exec_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// true when every expert op of the graph is a single-token decode step that the executor can reproduce exactly
bool exec_eligible(state_t & s, ggml_cgraph * cg) {
    if (!s.exec_on || (s.lookahead && !s.la_exec) || s.repack_tag || s.gpu_min_tokens == 1) {
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
#ifdef _WIN32
    g_cpu_graph = cg;
#endif
    const enum ggml_status st = ggml_backend_graph_compute(c->cpu, cg);
#ifdef _WIN32
    g_cpu_graph = nullptr;
#endif
    for (auto & sv : saved) {
        sv.first->op = sv.second;
    }
    return st;
}

// llama.cpp (ggml 0.26) computes an expert with a separate tiled routine once it has 8 or more rows (tokens) in the batch (ggml-cpu/tiled, rows >= 8),
// which adds up in another order than the one-row kernel. Such ops are left to llama.cpp so the result stays identical (found on GLM-5.3-Flash:
// one expert picked by all 8 tokens of a prompt batch). Older ggml has no tiled routine; handing the op back is exact there too.
// Counted for every batch size: a token can list the same expert more than once (likely a GPU top-k fed NaN router scores, which can repeat an index), so
// 2 tokens can already give one expert 8+ rows. Such an op must not reach the executor either: its tasks hold at most EXEC_MAXCOL columns
// (Windows crash at request 2, 2026-10-27: Qwen3.5-122B on the GPU router, a 4-token prompt tail, a pool worker writing through a garbage pointer).
bool stock_uses_tiled(const ggml_tensor * nd) {
    const ggml_tensor * ids = nd->src[2];
#ifdef MOE_CACHE_TEST_MUTANT_DUP_IDS    // CI only: the old count (batches of 8+ tokens only); with the no-split below it must fail tests/dup_ids.sh
    if (ids->ne[1] < 8) {
        return false;
    }
#endif
    // GGML_CPU_TILED_MM_FORCE=1 (a llama.cpp test switch, honoured unless GGML_CPU_TILED_MM=0) makes stock take the tiled routine for any row
    // count: then every op goes back to llama.cpp
    static const bool tiled_forced = [] {
        const char * f = getenv("GGML_CPU_TILED_MM_FORCE"), * m = getenv("GGML_CPU_TILED_MM");
        const bool on = f && atoi(f) == 1 && !(m && atoi(m) == 0);
        if (on) {
            fprintf(stderr, "moe-cache: GGML_CPU_TILED_MM_FORCE=1: every expert op is left to llama.cpp's tiled routine (the executor would add up in another order)\n");
        }
        return on;
    }();
    if (tiled_forced) {
        return true;
    }
    if (ids->ne[0] * ids->ne[1] < 8) {
        return false;
    }
    static thread_local std::vector<int32_t> v;   // runs for every expert op of a decode step: no allocation after the first
    v.clear();
    for (int64_t t = 0; t < ids->ne[1]; t++) {
        for (int64_t k = 0; k < ids->ne[0]; k++) {
            v.push_back(*(const int32_t *) ((const char *) ids->data + k * ids->nb[0] + t * ids->nb[1]));
        }
    }
    std::sort(v.begin(), v.end());
    for (size_t i = 0; i + 7 < v.size(); i++) {
        if (v[i] == v[i + 7]) {
            return true;
        }
    }
    return false;
}

// In executor graphs the reads are asynchronous: the executor waits for each expert's flag itself. An op handed back to llama.cpp has to wait here first.
// wait until every expert this op uses is complete; returns true when at least one was still loading
bool exec_wait_loaded(state_t & s, const ggml_tensor * nd) {
    bool waited = false;
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
                waited = true;
                std::this_thread::yield();
            }
        }
    }
    return waited;
}

// MOE_CACHE_EXEC_CHECK only: what the last executor op handed to the kernel, so a mismatch report can tell a wrong activation column from a
// wrong timing (expert bytes not complete when the op ran) or a wrong result on a pool worker
struct exec_last_t {
    const ggml_tensor *  nd = nullptr;
    const char *         base = nullptr;
    size_t               stride_i11 = 0, stride_tok = 0;
    ggml_vec_dot_t       dot = nullptr;
    std::vector<int32_t> late;             // experts still loading when the op started
};
exec_last_t g_exec_last;

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
    static const bool etimes = getenv("MOE_CACHE_EXEC_TIMES") != nullptr;
    const uint64_t eq0 = etimes ? exec_now_ns() : 0;
    const char * base;
    size_t stride_i11, stride_tok;       // column (token t, activation row i11) = base + i11 * stride_i11 + t * stride_tok
    ggml_vec_dot_t dot = tt->vec_dot;
#ifdef MC_X86_KERNELS
    const bool q2_f32 = w->type == GGML_TYPE_Q2_0 && vdt == GGML_TYPE_Q8_0 && a->type == GGML_TYPE_F32 && ne10 % 64 == 0;
    const bool permuted = q2_f32 && fast_kernels_enabled();
    const bool colsum = q2_f32 && !permuted && upstream_q2_0_enabled();
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
#ifdef MC_X86_KERNELS
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
        } else if (colsum) {
            const size_t nbk = (size_t) (ne10 / 32);
            const size_t ncols = (size_t) ne11 * n_tok;
            pbuf.resize(ncols * nbk * sizeof(fastk::block_q8_0_ys) + 64);
            for (size_t c = 0; c < ncols; c++) {
                fastk::prepare_q8_0_ys((const fastk::block_q8_0 *) (wbuf.data() + c * row_size),
                                       (fastk::block_q8_0_ys *) (pbuf.data() + c * nbk * sizeof(fastk::block_q8_0_ys)), (int) nbk);
            }
            base = (const char *) pbuf.data();
            stride_i11 = nbk * sizeof(fastk::block_q8_0_ys);
            stride_tok = (size_t) ne11 * stride_i11;
            dot = q2_0_ys_dot;
        }
#endif
    }
    const uint64_t eb0 = etimes ? exec_now_ns() : 0;
    if (etimes) {
        s.ex_quant_ns += eb0 - eq0;
    }
    // one task per expert: the tokens that picked it share its weight rows
    std::vector<exec_task_t> all;
    all.reserve((size_t) n_ids * (size_t) n_tok);
    std::vector<int> task_of((size_t) w->ne[2], -1);
    const std::vector<int32_t> * routed = nullptr;
    if (s.route_mode || s.fewer > 0) {
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
            // one column per token that picked this expert; a full task gets a second one for the same expert (an expert listed twice by one
            // token: each column is still its own dot product, so the result does not change). The old code wrote past the task here.
#ifndef MOE_CACHE_TEST_MUTANT_DUP_IDS
            if (ti_ < 0 || all[(size_t) ti_].ncol == EXEC_MAXCOL) {
#else
            if (ti_ < 0) {
#endif
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
    if (s.exec_check) {
        g_exec_last.nd = nd;
        g_exec_last.base = base;
        g_exec_last.stride_i11 = stride_i11;
        g_exec_last.stride_tok = stride_tok;
        g_exec_last.dot = dot;
        g_exec_last.late.clear();
        for (const auto & t : late) {
            g_exec_last.late.push_back((int32_t) ((t.w - (const char *) w->data) / (ptrdiff_t) w->nb[2]));
        }
    }
    // every task has 1..EXEC_MAXCOL columns and every output row lies inside this op's output tensor: checked before the pool runs, because a
    // wrong pointer here is a write anywhere in the process, found later as a crash somewhere else (cheap: at most n_tok x picks values)
    {
        const char * lo = (const char *) nd->data, * hi = lo + ggml_nbytes(nd);
        const size_t row_bytes = (size_t) w->ne[1] * sizeof(float);
        for (const auto & t : tasks) {
            if (t.ncol < 1 || t.ncol > EXEC_MAXCOL) {
                fprintf(stderr, "moe-cache: internal: an expert task of %s has %d columns (1-%d allowed; %d tokens, %d picks); stopping before memory is "
                        "overwritten (please report this line)\n", nd->name, t.ncol, EXEC_MAXCOL, n_tok, n_ids);
                abort();
            }
            for (int k = 0; k < t.ncol; k++) {
                const char * d = (const char *) t.dsts[k];
                if (!lo || d < lo || d + row_bytes > hi) {
                    fprintf(stderr, "moe-cache: internal: output of %s (%p, %zu bytes) does not hold column %d of an expert at %p (%d tokens, %d picks); "
                            "stopping before memory is overwritten (please report this line)\n", nd->name, (const void *) lo, ggml_nbytes(nd), k,
                            (const void *) d, n_tok, n_ids);
                    abort();
                }
            }
        }
    }
    const auto tp0 = std::chrono::steady_clock::now();
    if (etimes) {
        s.ex_build_ns += exec_now_ns() - eb0;
    }
    s.pool->run(tasks.data(), (int) tasks.size(), w->ne[1], w->ne[0], w->nb[1], dot, s.exec_chunk, nd->name);
    if (etimes) {
        const uint64_t rn = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tp0).count();
        const int part = strstr(nd->name, "gate") ? 0 : strstr(nd->name, "up") ? 1 : 2;
        s.ex_run_ns += rn;
        s.ex_part_ns[part] += rn; s.ex_part_n[part]++;
        s.ex_part_bytes[part] += (uint64_t) tasks.size() * (uint64_t) w->nb[1] * (uint64_t) w->ne[1];
    }
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
    } else if ((s.lookahead && !s.la_exec) || s.repack_tag || s.gpu_min_tokens == 1) {
        o += std::string(" executor=off(") + (s.lookahead ? "lookahead" : s.repack_tag ? "repacked weights" : "gpu_min_tokens=1") + ")";
    } else {
        o += " executor=on(<=" + std::to_string(EXEC_MAXCOL) + " tokens" + (s.exec_force ? ", forced" : ", steps aside when resident and bandwidth-bound") +
             (s.exec_chunk != 128 ? ", " + std::to_string(s.exec_chunk) + " rows per chunk" : "") + ")";
    }
#ifdef MC_X86_KERNELS
    o += std::string(" q2_0_kernel=") + (!fast_kernels_enabled() ? (q2_0_library_is_plain() ? "off(MOE_CACHE_FAST_KERNELS=0 or no AVX2)"
                                                                : upstream_q2_0_enabled() ? "executor(column sums, exact to llama.cpp's #30447)" : "off(llama.cpp's own)")
                                                               : preload ? "on(interposed)" : "executor only(not preloaded)");
    o += std::string(" iq3_xxs_kernel=") + (fast_iq3_dot() ? "on(executor)" : "off(MOE_CACHE_FAST_KERNELS=0 or MOE_CACHE_IQ3_KERNEL=0, no AVX2, or not Intel: MOE_CACHE_IQ3_KERNEL=1 forces it on)");
    o += std::string(" iq_avx512=") + (getenv("MOE_CACHE_AVX512") ? (moe_cache_iq512_probe && moe_cache_iq512_probe() ? "on" : "off(self-check or CPU)") : "off(MOE_CACHE_AVX512 unset)");
#endif
#ifdef _WIN32
    cpu_traits(GGML_TYPE_F32);   // resolves the lookup once
    o += " cpu_kernels=" + g_cpu_traits_from;
#endif
    o += std::string(" gpu_cache_host_buffer=") + (gpu_cache_mode() ? "on" : "off");
    o += s.gpu_dev ? " long_prompt_gpu=on(>=" + std::to_string(s.gpu_min_tokens) + " tokens)" : " long_prompt_gpu=off(no GPU device)";
    if (s.fewer > 0) o += std::string(" approx_fewer=") + std::to_string(s.fewer);
    if (s.route_mode) o += std::string(" approx_route=") + (s.route_mode == 1 ? "skip" : "subst");
    fprintf(stderr, "%s\n", o.c_str());
}

}  // namespace
