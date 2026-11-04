// CLEANUP-002: usage profile and warm start members
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_profile_t {
    // usage profile and warm start
    std::string profile_path;          // learn here and warm start from it
    std::string warm_path;             // warm start only
    bool   preload_on = false;
    bool   preload_done = false;
    double preload_frac = -1.0;        // < 0: everything if it fits, else 0.7
    double profile_decay = 0.5;
    uint64_t fingerprint = 0;
    usage_t learned;
    std::vector<std::vector<double>> session;   // [layer][expert] picks of this run
    uint64_t groups_since_save = 0;
};
