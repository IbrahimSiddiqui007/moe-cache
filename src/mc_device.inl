// moe-cache plugin part: device, registry, ggml_backend_init, LD_PRELOAD registration.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

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
    if (op->op == g_op_glu) {
        return true;
    }
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT_ID:
            break;
        default:
            return false;
    }
    const ggml_tensor * w = op->src[0];
    if (!w || !w->buffer || ggml_backend_buffer_get_type(w->buffer)->iface.get_name != buft_name) {
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

bool g_layout_bad = false;   // set by ggml_backend_init when the running ggml lays out one of our structs differently: the registry then lists no device

size_t reg_count(ggml_backend_reg_t) {
    return g_layout_bad ? 0 : 1;
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

// Layout self-test (audit row 10). ggml reads three of the structs the plugin builds through the plugin's own pointer: ggml_backend_buffer_type,
// ggml_backend_device and ggml_backend. Each is `interface (a row of function pointers), then data fields`. If ggml appends a slot to such an interface
// without a backend API bump (it did for ggml_backend_buffer, master 418f905a4), the data fields move: ggml reads `device` / `reg` from a slot where
// the plugin has something else, and a new function slot reads the plugin's next field as a function. Padding cannot help (the plugin owns the struct),
// so the plugin asks ggml where it reads the field: it hands the accessor an array of distinct numbers and sees which one comes back. The reg struct
// is only read for its interface (ggml_backend_init pads it with zeros instead). Returns "" when ggml agrees with the headers.
std::string ggml_struct_layout_mismatch() {
    constexpr int N = 96;
    constexpr uintptr_t B = 0x10000;
    void * a[N];
    for (int i = 0; i < N; i++) {
        a[i] = (void *) (B + (uintptr_t) i * sizeof(void *));
    }
    auto slot = [&](const void * r) -> long {
        const uintptr_t v = (uintptr_t) r;
        return (v >= B && v < B + N * sizeof(void *) && (v - B) % sizeof(void *) == 0) ? (long) ((v - B) / sizeof(void *)) : -1;
    };
#ifdef MOE_CACHE_TEST_MUTANT_LAYOUT      // CI only: a build that expects another layout must refuse to run
    const long skew = 1;
#else
    const long skew = 0;
#endif
    struct { const char * what; long got, want; } t[] = {
        { "ggml_backend_buffer_type.device", slot(ggml_backend_buft_get_device((ggml_backend_buffer_type_t) a)),
          (long) (offsetof(ggml_backend_buffer_type, device) / sizeof(void *)) + skew },
        { "ggml_backend.device", slot(ggml_backend_get_device((ggml_backend_t) a)), (long) (offsetof(ggml_backend, device) / sizeof(void *)) },
        { "ggml_backend_device.reg", slot(ggml_backend_dev_backend_reg((ggml_backend_dev_t) a)), (long) (offsetof(ggml_backend_device, reg) / sizeof(void *)) },
    };
    for (const auto & x : t) {
        if (x.got != x.want) {
            return std::string("the running ggml reads ") + x.what + " at slot " + std::to_string(x.got) + ", the plugin's headers put it at slot " + std::to_string(x.want) +
                   " (ggml changed a struct without a backend API bump)";
        }
    }
    return "";
}

}

extern "C" MOE_CACHE_EXPORT ggml_backend_reg_t ggml_backend_init(void) {
#ifdef _WIN32
    // the DLL has its own C runtime state; unbuffered, so our lines reach the log even when the process dies right after (a crash in CI left none)
    setvbuf(stderr, nullptr, _IONBF, 0);
#endif
    // reg is read by ggml for its interface only; zeros after it make every slot a newer ggml appends read as "not supported"
    struct reg_padded { ggml_backend_reg r; void * spare[16]; };
    static reg_padded reg = { { GGML_BACKEND_API_VERSION, g_reg_i, nullptr }, {} };
    static const bool once = [] {
        const std::string m = ggml_struct_layout_mismatch();
        if (!m.empty()) {
            g_layout_bad = true;
            fprintf(stderr, "moe-cache: %s: plugin not loaded, the model loads the normal way. Use the plugin build that matches your llama.cpp (docs/QUICKSTART.md)\n", m.c_str());
        }
        return true;
    }();
    (void) once;
    return &reg.r;
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
