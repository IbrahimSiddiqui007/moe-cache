// CLEANUP-002: CPU repack buffer members
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_repack_t {
    ggml_backend_buffer_type_t repack_buft = nullptr;   // llama.cpp CPU_REPACK buffer type (mode on)
    ggml_backend_buffer_t repack_buf = nullptr;         // a real repack buffer, used for init_tensor / set_tensor only
    ggml_backend_buffer_t repack_tag = nullptr;         // empty buffer of the repack type, swapped in while an operation runs
    std::atomic<uint64_t> repack_jobs{0};
};
