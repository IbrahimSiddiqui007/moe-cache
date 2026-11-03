// AUD-014: a failed address-space reservation (here forced with a small RLIMIT_AS) has a message that names the cause and MOE_CACHE_DISABLE=1.
//   g++ -std=c++17 -Wall -Werror -Wno-unused-function -Isrc tests/reserve_msg_test.cpp -o /tmp/reserve_msg_test && /tmp/reserve_msg_test
#include "platform.h"
#include "mc_msg.h"
#ifndef _WIN32
#include <sys/resource.h>
#endif
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    const size_t big = (size_t) 8 << 30;
#ifndef _WIN32
    struct rlimit rl;
    getrlimit(RLIMIT_AS, &rl);
    rl.rlim_cur = (rlim_t) 2 << 30;
    CHECK(setrlimit(RLIMIT_AS, &rl) == 0);
    CHECK(plat::mem_reserve(big) == nullptr);       // the failure the message is for
#endif
    const std::string m = mcmsg::reserve_failed(big);
    CHECK(m.find("8.00 GiB") != std::string::npos);
    CHECK(m.find("ulimit -v") != std::string::npos);
    CHECK(m.find("vm.overcommit_memory") != std::string::npos);
    CHECK(m.find("MOE_CACHE_DISABLE=1") != std::string::npos);
    if (fails) return 1;
    printf("reserve_msg_test: ok\n");
    return 0;
}
