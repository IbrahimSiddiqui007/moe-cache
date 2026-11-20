// CLEANUP-002: read fault-injection test switches and counters
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_testfault_t {
    int    test_read_delay_us = 0;              // MOE_CACHE_TEST_READ_DELAY_US=N: test switch, background expert reads start N us late
    int    fault_every = 0;                     // MOE_CACHE_FAULT_EVERY=N: test switch, every Nth expert read fails all its direct attempts
    bool   test_poison = false;                 // MOE_CACHE_TEST_POISON_READS=1: test switch, a read first fills what it will write with 0xFF and waits
                                                //   MOE_CACHE_TEST_READ_DELAY_US (what a page under an unbuffered Windows read looks like to other threads)
    int    test_skip_pct = -1;                  // MOE_CACHE_TEST_SKIP_PCT=0..100 (TEST-1170, timing only, output not valid): share of executor expert picks not computed; -1 off (src/mc_test_skip.h)
    std::atomic<uint64_t> test_skipped{0}, test_computed{0};   // picks (one per token, slot and gate/up/down op) skipped / computed by the executor while the switch is on
    std::atomic<uint64_t> fault_ctr{0}, read_fallbacks{0};
};
