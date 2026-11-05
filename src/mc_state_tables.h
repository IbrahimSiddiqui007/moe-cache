// CLEANUP-002: tensor and node tables (file tensors, registered tensors, nodes, owners)
// A data-only base of state_t (src/mc_state.inl): the members moved here unchanged, so every s.name and every method of state_t still reads them by the same name.
#pragma once

struct state_tables_t {
    std::unordered_map<std::string, std::pair<size_t, size_t>> file_tensors;
    std::vector<std::unique_ptr<tinfo_t>> tensors;
    std::unordered_map<const ggml_tensor *, int> by_tensor;
    std::vector<node_t> nodes;
    std::vector<uint32_t> node_owner;           // index in `tensors` of the tensor that owns each node (same size as nodes)
};
