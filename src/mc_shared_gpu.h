// Sizing rules for the long-prompt GPU block on a shared-memory GPU (CACHE-149, owner decision 2026-11-03: option (a)). Header-only, no ggml:
// tests/shared_gpu_test.cpp includes it.
//
// On an integrated GPU (APU, Intel Arc iGPU, Apple-style unified memory) there is no VRAM of its own: every byte the plugin "uploads" for a long-prompt block
// is a second copy of the expert weights in system RAM, next to the cache's copy and the OS page cache. The cache budget did not know about it, so on a
// 32 GB machine the block's copy (up to a whole layer of experts, plus the pinned read-back and staging buffers) came on top of a budget that already
// used all the RAM it was allowed. Two rules, both sizes only (what the block computes does not change, a block that does not fit is split into parts
// or stays on the CPU, which the plugin already did when VRAM ran short):
//   1. cap:   the GPU-side copy may not grow beyond stage_cap() (a block bigger than that is split into more parts: same arithmetic, same tokens);
//   2. count: the bytes the copy holds are taken from the expert cache's budget (and from its growth ceiling) while they are held.
// Discrete GPUs are untouched: their buffers are VRAM.
#pragma once
#include <algorithm>
#include <cstddef>

namespace mcshared {

constexpr size_t MiB = (size_t) 1 << 20;
constexpr size_t GiB = (size_t) 1 << 30;

// Largest GPU-side copy of expert weights (+ graph buffers) one block may hold on a shared-memory GPU.
//   env_mib > 0: the user's MOE_CACHE_GPU_SHARED_CAP_MIB, as is.
//   else 1/8 of the RAM the plugin may use (the memory limit in auto mode, else the cache budget), at least 1 GiB (but never more than 1/4 of that RAM:
//   a tiny machine still gets a usable part size) and at most 4 GiB. The Claw (Arc 140V, 32 GB, ~28 GB limit): 3.5 GiB. To be measured there.
inline size_t stage_cap(size_t ram_ref, long env_mib) {
    if (env_mib > 0) {
        return (size_t) env_mib * MiB;
    }
    const size_t lo = std::min(GiB, ram_ref / 4);
    return std::min(4 * GiB, std::max(lo, ram_ref / 8));
}

// The "free GPU memory" the block's fit and split decisions see on a shared-memory GPU: what the driver reports, but not more than the cap leaves after the
// buffer the block already holds (`held`). fits(need) in gpu_segment is gpu_free >= (need - held) + headroom, so a block fits only while need + headroom <= cap;
// a block that needs no more than `held` allocates nothing and always fits.
inline size_t free_for_block(size_t driver_free, size_t held, size_t cap) {
    return std::min(driver_free, cap > held ? cap - held : (size_t) 0);
}

// Takes d bytes off *v, but never below `floor_b`; returns what was taken.
inline size_t take_down(size_t & v, size_t d, size_t floor_b) {
    const size_t can = v > floor_b ? v - floor_b : 0;
    const size_t t = std::min(d, can);
    v -= t;
    return t;
}

// What the copy holds counts against the cache. `reserved` is what is counted now; `now` what the copy holds after this block. When it grew, the budget
// (and the ceiling max_budget the autotuner grows towards, if it has one yet) are lowered by the growth, never below floor_b; when it shrank, only the
// ceiling is given back (the autotuner grows the budget to it on its own, at most 1 GiB per step). `max_taken` is how much the ceiling has been lowered
// in total, so that exactly that comes back. Returns true when the budget was lowered (the caller then evicts down to it).
struct account_t {
    size_t reserved  = 0;
    size_t max_taken = 0;
};
inline bool account(account_t & a, size_t now, size_t & budget, size_t & max_budget, size_t floor_b) {
    const size_t old = a.reserved;
    a.reserved = now;
    if (now > old) {
        const size_t d = now - old;
        const size_t tb = take_down(budget, d, floor_b);
        if (max_budget) {
            a.max_taken += take_down(max_budget, d, floor_b);
        }
        return tb > 0;
    }
    if (now < old && max_budget) {
        const size_t back = std::min(a.max_taken, old - now);
        max_budget += back;
        a.max_taken -= back;
    }
    return false;
}

}  // namespace mcshared
