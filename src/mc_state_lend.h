// CLEANUP-002: prefill lend members
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_lend_t {
    // idea P20 (MOE_CACHE_PREFILL_LEND_MIB=N, default 0 = off): while long prompt batches run, the expert cache gives N MiB back to the system (the cache is cut by N and
    // the surplus evicted), and takes them back at the first small (decode) graph
    size_t lent_bytes = 0;
    size_t lent_budget = 0;                             // the budget right after the lend (a cut by tune() meanwhile is not given back twice, AUD-021)
};
