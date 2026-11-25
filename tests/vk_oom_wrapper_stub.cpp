// Model-free test of patch 0093: the two functions the patch adds are cut out of the patch text (VK_OOM_BODY, written by
// apply_llama_patches_test.sh) and compiled against minimal stand-ins for the Vulkan types, so a throw really goes through the
// real wrapper code. It checks the status mapping and the state reset after the catch. It does not run Vulkan.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>

enum ggml_status { GGML_STATUS_ALLOC_FAILED = -2, GGML_STATUS_FAILED = -1, GGML_STATUS_SUCCESS = 0 };
#define GGML_LOG_ERROR(...) std::fprintf(stderr, __VA_ARGS__)
namespace vk {
struct SystemError : std::runtime_error { using std::runtime_error::runtime_error; };
struct OutOfDeviceMemoryError : SystemError { using SystemError::SystemError; };
struct OutOfHostMemoryError : SystemError { using SystemError::SystemError; };
struct Semaphore {};
}
struct fake_vk_device { void waitIdle() { n_wait_idle++; if (idle_throws) throw vk::SystemError("device lost"); } uint64_t getSemaphoreCounterValue(vk::Semaphore) { return reached; }
                        int n_wait_idle = 0; bool idle_throws = false; uint64_t reached = 5; };
struct vk_device_struct { fake_vk_device device; bool async_use_transfer_queue = true; };
struct vk_ctx_struct {};
struct fake_sem { vk::Semaphore s; uint64_t value; };
struct ggml_backend_vk_context {
    std::shared_ptr<vk_device_struct> device = std::make_shared<vk_device_struct>();
    std::weak_ptr<vk_ctx_struct> compute_ctx, transfer_ctx;
    bool submit_pending = true;
    fake_sem transfer_semaphore{{}, 9};
    uint64_t transfer_semaphore_last_submitted = 9;
};
struct ggml_backend { void * context; };
typedef ggml_backend * ggml_backend_t;
struct ggml_cgraph {};
static int n_cleanup = 0;
static void ggml_vk_graph_cleanup(ggml_backend_vk_context *) { n_cleanup++; }
static int mode = 0;
static ggml_status ggml_backend_vk_graph_compute_impl(ggml_backend_t, ggml_cgraph *) {
    switch (mode) {
        case 0: return GGML_STATUS_SUCCESS;
        case 1: throw vk::OutOfDeviceMemoryError("oom device");
        case 2: throw vk::OutOfHostMemoryError("oom host");
        case 3: throw std::bad_alloc();
        case 4: throw vk::SystemError("device lost");
        default: throw std::logic_error("other");
    }
}
#include VK_OOM_BODY
static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s\n", m); fails++; } } while (0)
int main() {
    const ggml_status want[] = { GGML_STATUS_SUCCESS, GGML_STATUS_ALLOC_FAILED, GGML_STATUS_ALLOC_FAILED, GGML_STATUS_ALLOC_FAILED, GGML_STATUS_FAILED, GGML_STATUS_FAILED };
    for (mode = 0; mode <= 5; mode++) {
        ggml_backend_vk_context c; ggml_backend b{ &c }; ggml_cgraph g; n_cleanup = 0;
        auto cc = std::make_shared<vk_ctx_struct>(), tc = std::make_shared<vk_ctx_struct>(); c.compute_ctx = cc; c.transfer_ctx = tc;
        ggml_status s = ggml_backend_vk_graph_compute(&b, &g);
        CHECK(s == want[mode], "status mapping");
        if (mode == 0) { CHECK(n_cleanup == 0 && c.device->device.n_wait_idle == 0, "success path untouched"); continue; }
        CHECK(c.device->device.n_wait_idle == 1, "device drained once");
        CHECK(!c.compute_ctx.lock() && !c.transfer_ctx.lock(), "compute_ctx and transfer_ctx dropped");
        CHECK(!c.submit_pending, "submit_pending cleared");
        CHECK(c.transfer_semaphore.value == 5 && c.transfer_semaphore_last_submitted == 5, "transfer semaphore value back to the reached counter");
        CHECK(n_cleanup == 1, "ggml_vk_graph_cleanup run");
    }
    { // waitIdle itself throws (device lost): still returns, still resets
        mode = 4; ggml_backend_vk_context c; c.device->device.idle_throws = true; ggml_backend b{ &c }; ggml_cgraph g; n_cleanup = 0;
        CHECK(ggml_backend_vk_graph_compute(&b, &g) == GGML_STATUS_FAILED, "device lost: FAILED");
        CHECK(n_cleanup == 1 && !c.submit_pending, "reset runs even when waitIdle throws");
    }
    std::puts(fails ? "vk_oom_wrapper: FAIL" : "vk_oom_wrapper: PASS");
    return fails ? 1 : 0;
}
