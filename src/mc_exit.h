// moe-cache plugin part: what the exit guard may do with the plugin state (AUD-013). Plain C++, so tests/exit_test.cpp can check the choice alone.
#pragma once

namespace mcexit {

enum class action { delete_state, leave_state };

// process_exiting: Windows is already tearing the process down. lock_free: the cache lock could be taken. graph_active: graphs in flight.
// The lock is free for most of a graph (it is held only for short steps), so "lock free" alone does not mean "no graph is running": a forced exit
// (llama-server's second Ctrl+C calls exit(1)) mid-prompt would delete the state and the pool under the running compute threads.
inline action choose(bool process_exiting, bool lock_free, int graph_active) {
    return process_exiting || !lock_free || graph_active > 0 ? action::leave_state : action::delete_state;
}

}   // namespace mcexit
