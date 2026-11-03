// moe-cache plugin part: the retry rule for one expert read piece. Plain C++ without ggml, so tests/read_retry_test.cpp can drive it with a fake
// reader. A read that fails (or returns 0 bytes before the piece is complete) is tried again after a pause; when the tries are used up the caller
// falls back to a buffered read and then stops. It never continues with missing bytes (zeros or stale data would change the output).
#pragma once
#include <cerrno>
#include <cstddef>

namespace mcretry {

constexpr int kRetries = 3;         // tries after the first failure (CACHE-271 option b)
constexpr int kDelayMs = 200;       // pause between tries (owner decision CACHE-271 b: 3 tries, 200 ms apart; a fault-injected read now costs 3 x 200 ms)

// a direct read that fails like this will fail every time on this file system (fscrypt, FUSE, 9p, virtiofs accept open(O_DIRECT) but not the read):
// no retry, the caller switches that file to buffered reads for good (AUD-011)
inline bool direct_unsupported(int err) {
#if defined(EOPNOTSUPP)
    if (err == EOPNOTSUPP) {
        return true;
    }
#endif
    return err == EINVAL;
}

// rd(done) reads the rest of the piece starting `done` bytes in: returns bytes read, 0 at end of file, -1 on error (errno set).
// sl(ms) sleeps. Returns true when `need` bytes have arrived. On false, *unsupported tells that the failure was direct_unsupported (when
// check_unsupported is set); otherwise the tries ran out.
// A FUTURE option (c) "fail only the current request instead of abort()" would be decided by the caller of this function, not here.
template <class Read, class Sleep>
bool read_fully(Read rd, Sleep sl, size_t need, int retries, int delay_ms, bool check_unsupported, bool * unsupported) {
    size_t done = 0;
    int tries = 0;
    *unsupported = false;
    while (done < need) {
        errno = 0;
        const ptrdiff_t got = rd(done);
        if (got <= 0) {
            if (got < 0 && check_unsupported && direct_unsupported(errno)) {
                *unsupported = true;
                return false;
            }
            if (++tries <= retries) {
                sl(delay_ms);
                continue;
            }
            return false;
        }
        done += (size_t) got;
    }
    return true;
}

}   // namespace mcretry
