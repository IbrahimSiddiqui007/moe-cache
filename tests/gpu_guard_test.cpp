// Checks src/mc_guard.h (AUD-002): an exception out of the GPU block is caught by default and becomes "stay on the CPU"; =0 opts out; =2 injects once.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/gpu_guard_test.cpp -o /tmp/gpu_guard_test && /tmp/gpu_guard_test
#include "mc_guard.h"
#include <cstdio>
#include <string>
#include <system_error>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    // the default: unset or empty catches; 0/off/false/no opts out; 2 injects; other on-values catch
    CHECK(mcguard::catch_mode(nullptr) == 1 && mcguard::catch_mode("") == 1 && mcguard::catch_mode("1") == 1 && mcguard::catch_mode("on") == 1);
    CHECK(mcguard::catch_mode("0") == 0 && mcguard::catch_mode("off") == 0 && mcguard::catch_mode("false") == 0 && mcguard::catch_mode("no") == 0);
    CHECK(mcguard::catch_mode("2") == 2);

    int calls = 0, failures = 0;
    std::string why;
    bool injected = false;
    auto failed = [&](const char * w) { failures++; why = w; };
    auto ok_block = [&] { calls++; return true; };
    auto throwing_block = [&]() -> bool { calls++; throw std::runtime_error("vk::OutOfDeviceMemoryError"); };

    CHECK(mcguard::guarded(1, injected, ok_block, failed) && calls == 1 && failures == 0);                        // no fault: result passes through
    CHECK(!mcguard::guarded(1, injected, [&] { calls++; return false; }, failed) && failures == 0);              // a plain "no" is not a failure
    CHECK(!mcguard::guarded(1, injected, throwing_block, failed) && failures == 1 && why == "vk::OutOfDeviceMemoryError");   // the throw is caught
    // ggml-vulkan throws vk::SystemError, a std::system_error: also a std::exception
    CHECK(!mcguard::guarded(1, injected, [&]() -> bool { throw std::system_error(std::make_error_code(std::errc::not_enough_memory)); }, failed) && failures == 2);

    bool threw = false;
    try { mcguard::guarded(0, injected, throwing_block, failed); } catch (const std::exception &) { threw = true; }
    CHECK(threw && failures == 2);                                                                                 // =0: unchanged, the throw escapes

    calls = 0; failures = 0; injected = false;
    CHECK(!mcguard::guarded(2, injected, ok_block, failed) && calls == 0 && failures == 1 && why == "injected by MOE_CACHE_GPU_CATCH=2");   // first block: injected
    CHECK(mcguard::guarded(2, injected, ok_block, failed) && calls == 1 && failures == 1);                          // only once
    if (fails) {
        fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    printf("gpu-guard checks passed\n");
    return 0;
}
