// Model-free test of the expert-read retry rule (src/mc_retry.h): AUD-011 (an unsupported direct read is not retried) and CACHE-271 option (b)
// (a persistent read error is retried kRetries times, kDelayMs apart, then reported as failed so the caller stops; never succeeds with missing bytes).
#include "mc_retry.h"
#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    // persistent error: 1 try + kRetries retries, a pause before each retry, result false
    {
        int calls = 0; std::vector<int> sleeps; bool uns = true;
        const bool ok = mcretry::read_fully([&](size_t) -> ptrdiff_t { calls++; errno = EIO; return -1; },
                                            [&](int ms) { sleeps.push_back(ms); }, 100, mcretry::kRetries, mcretry::kDelayMs, true, &uns);
        CHECK(!ok); CHECK(!uns); CHECK(calls == 1 + mcretry::kRetries); CHECK((int) sleeps.size() == mcretry::kRetries);
        for (int ms : sleeps) CHECK(ms == mcretry::kDelayMs);
    }
    // a read that recovers on the second retry succeeds
    {
        int calls = 0; size_t seen_done = 99; bool uns;
        const bool ok = mcretry::read_fully([&](size_t done) -> ptrdiff_t { if (++calls < 3) { errno = EIO; return -1; } seen_done = done; return 100; },
                                            [](int) {}, 100, 3, 1, true, &uns);
        CHECK(ok); CHECK(calls == 3); CHECK(seen_done == 0);
    }
    // partial reads: progress continues, then EOF (0) before the piece is complete is a failure after the retries
    {
        int calls = 0; bool uns;
        const bool ok = mcretry::read_fully([&](size_t done) -> ptrdiff_t { calls++; return done == 0 ? 40 : 0; }, [](int) {}, 100, 3, 1, true, &uns);
        CHECK(!ok); CHECK(calls == 1 + 1 + 3);
    }
    // AUD-011: EINVAL / EOPNOTSUPP on a direct read: no retry, no sleep, reported as unsupported
    for (int err : { EINVAL, EOPNOTSUPP }) {
        int calls = 0, slept = 0; bool uns = false;
        const bool ok = mcretry::read_fully([&](size_t) -> ptrdiff_t { calls++; errno = err; return -1; }, [&](int) { slept++; }, 100, 3, 20, true, &uns);
        CHECK(!ok); CHECK(uns); CHECK(calls == 1); CHECK(slept == 0);
    }
    // the same errno on a buffered read (check_unsupported false) is an ordinary error: retried
    {
        int calls = 0; bool uns = true;
        const bool ok = mcretry::read_fully([&](size_t) -> ptrdiff_t { calls++; errno = EINVAL; return -1; }, [](int) {}, 100, 3, 1, false, &uns);
        CHECK(!ok); CHECK(!uns); CHECK(calls == 4);
    }
    // EIO / EAGAIN on a direct read are still retried
    for (int err : { EIO, EAGAIN }) {
        int calls = 0; bool uns = true;
        mcretry::read_fully([&](size_t) -> ptrdiff_t { calls++; errno = err; return -1; }, [](int) {}, 100, 3, 1, true, &uns);
        CHECK(calls == 4); CHECK(!uns);
    }
    if (fails) return 1;
    printf("read_retry_test: ok\n");
    return 0;
}
