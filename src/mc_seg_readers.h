// CACHE-1230: the long-prompt GPU block silences every node of its range (op = NONE) and copies back only the LAST node's output. That is only
// right when nothing outside the range reads one of the other nodes (or marks it a graph output): their buffers are never written, so such a reader
// would get stale memory. Header-only, no ggml: tests/seg_readers_test.cpp includes it with a plain node list.
#pragma once
#include <cstddef>

namespace mcseg {

// Nodes first..last (graph order) form the block; `last` is the one node whose output is copied back.
// src_node(i, k): index of the graph node that source k of node i resolves to (through views), or -1 (no source, a weight, an input, not a node).
// is_output(i): node i is flagged as a graph output. Returns true when the block is SAFE to take over (no other reader of its inner nodes).
template <class SrcFn, class OutFn>
bool inner_nodes_private(int n_all, int first, int last, int n_src, SrcFn src_node, OutFn is_output) {
    for (int i = first; i < last; i++) {
        if (is_output(i)) {
            return false;
        }
    }
    for (int i = 0; i < n_all; i++) {
        if (i >= first && i <= last) {
            continue;
        }
        for (int k = 0; k < n_src; k++) {
            const int s = src_node(i, k);
            if (s >= first && s < last) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace mcseg
