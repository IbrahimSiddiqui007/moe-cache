// CLEANUP-002: main hit/miss/byte and time counters
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_counters_t {
    std::atomic<uint64_t> hits{0}, misses{0}, bytes{0}, groups{0}, ops{0}, graphs{0};
    std::atomic<uint64_t> ns_prep{0}, ns_cpu{0}, ns_total{0}, ns_read{0}, ns_ssd_wait{0}, n_ssd_wait{0}, evictions{0}, direct_jobs{0}, bounce_jobs{0}, async_in{0};
};
