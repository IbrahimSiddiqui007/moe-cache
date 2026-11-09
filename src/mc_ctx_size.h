// CACHE-988: how big the ggml context of a GPU block is. The old size was 4 * (nodes + 8) tensors plus ggml_graph_overhead() (a graph of the default
// 2048 nodes) plus 4 KiB; on Nemotron 3.5 Super under ggml 0.29 ggml_new_graph asked for 688 bytes more than was left, ggml_new_object returned
// null and the next line dereferenced it (access violation). Now: the slots are the worst case from the real node count (every node and every
// source of every node is a tensor of its own), the graph is built with exactly that many slots (ggml_new_graph_custom) and the slack is 16 KiB.
#pragma once
#include <cstddef>

namespace mcctx {

constexpr size_t SLACK = 16384;

inline size_t slots(size_t n_nodes, size_t max_src) {
    return n_nodes * (max_src + 1) + 8;
}
// tensor_oh = ggml_tensor_overhead(), graph_oh = ggml_graph_overhead_custom(slots, false)
inline size_t bytes(size_t n_slots, size_t tensor_oh, size_t graph_oh) {
    return tensor_oh * n_slots + graph_oh + SLACK;
}

}  // namespace mcctx
