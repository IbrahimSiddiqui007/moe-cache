// CLEANUP-002: MOE_CACHE_PACK members (moepack file, layer records, counters)
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_pack_t {
    // BO-1 (MOE_CACHE_PACK, see bin/moe_cache_pack.py): MODEL.moepack holds each expert's gate/up/down slices in one 4 KiB-aligned block; a missing expert is
    // then one large unbuffered read instead of three. Replaces the file-view cache while it is in use (reads fill private memory).
    struct pack_layer_t { uint64_t base = 0, stride = 0; int n_expert = 0, n = 0; std::string name[4]; size_t nb2[4] = {}, in_off[4] = {}; };
    bool   pack_on = false;
    plat::file_t pack_fd;
    std::string pack_path;
    std::vector<pack_layer_t> pack_layers;
    std::unordered_map<std::string, std::pair<int, int>> pack_by_name;   // tensor name -> (layer record, slot)
    std::atomic<uint64_t> pack_reads{0}, pack_bytes{0}, pack_slices{0}, pack_fallbacks{0};
};
