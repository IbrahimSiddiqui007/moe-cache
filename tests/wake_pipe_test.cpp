// AUD-012 (Linux): the signal-to-thread pipes are close-on-exec, and ending one wakes its reader even while another process still holds the write end.
//   g++ -std=c++17 -Wall -Werror -Isrc tests/wake_pipe_test.cpp -o /tmp/wake_pipe_test -lpthread && /tmp/wake_pipe_test
#include "platform.h"
#include <atomic>
#include <chrono>
#include <thread>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
#ifdef __linux__
    int p[2];
    CHECK(plat::wake_pipe_create(p));
    CHECK((fcntl(p[0], F_GETFD) & FD_CLOEXEC) != 0);
    CHECK((fcntl(p[1], F_GETFD) & FD_CLOEXEC) != 0);
    const int held = dup(p[1]);         // what a child that inherited the write end would hold (plain dup is not close-on-exec)
    std::atomic<bool> stop{false}, done{false};
    std::thread th([&] { char c; while (read(p[0], &c, 1) == 1 && !stop) {} done = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!done);
    stop = true;
    plat::wake_pipe_finish(p[1]);
    for (int i = 0; i < 100 && !done; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    CHECK(done);                        // with a plain close() this stays blocked: `held` keeps the pipe open
    if (!done) {
        const char c = 1;
        const ssize_t r = write(held, &c, 1);   // free the thread so the test can exit
        (void) r;
    }
    th.join();
    close(held);
    close(p[0]);
#endif
    if (fails) return 1;
    printf("wake_pipe_test: ok\n");
    return 0;
}
