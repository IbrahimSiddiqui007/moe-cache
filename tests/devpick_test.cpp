// MULTIGPU-026: the long-prompt block never targets an OpenVINO device or an NPU (src/mc_devpick.h). No ggml, no GPU.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/devpick_test.cpp -o /tmp/devpick_test && /tmp/devpick_test
#include "mc_devpick.h"
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    CHECK(mcdev::usable_for_gpu_block("Vulkan", "AMD Radeon RX 9070"));
    CHECK(mcdev::usable_for_gpu_block("CUDA", "NVIDIA GeForce RTX 4090"));
    CHECK(mcdev::usable_for_gpu_block("SYCL", "Intel(R) Arc(TM) A770"));
    CHECK(mcdev::usable_for_gpu_block("Metal", nullptr) && mcdev::usable_for_gpu_block(nullptr, nullptr));
    CHECK(!mcdev::usable_for_gpu_block("OpenVINO", "Intel(R) AI Boost"));       // the registry name decides
    CHECK(!mcdev::usable_for_gpu_block("openvino", "x") && !mcdev::usable_for_gpu_block("GGML-OPENVINO", nullptr));
    CHECK(!mcdev::usable_for_gpu_block("Vulkan", "Intel(R) NPU"));              // a device that calls itself an NPU
    CHECK(mcdev::usable_for_gpu_block("Vulkan", "Radeon Pro VII"));             // "NPU" must be a substring match of the description only
    CHECK(!mcdev::contains_nocase("abc", "") && !mcdev::contains_nocase(nullptr, "a") && mcdev::contains_nocase("xOpenVINOx", "openvino"));
    if (fails) { fprintf(stderr, "%d checks failed\n", fails); return 1; }
    printf("devpick checks passed\n");
    return 0;
}
