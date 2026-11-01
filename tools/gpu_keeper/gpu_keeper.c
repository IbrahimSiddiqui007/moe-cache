// Runs a small fixed compute job on one GPU in a loop and logs how long it takes. Two uses:
//   as a probe: the job time follows the core clock, so it shows the GPU's clock state (with --sleep-ms large, it adds little load)
//   as a keeper: a steady light load may hold the clocks up while another program sends short bursts
// gpu_keeper [--device TEXT] [--sleep-ms N] [--iters N] [--groups N] [--seconds N] [--report-s N]
// Build: tools/gpu_keeper/build.sh  (needs glslc; Windows through mingw-w64). Uses only the Vulkan loader, found at run time.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
static void * load_loader(void) { return (void *) LoadLibraryA("vulkan-1.dll"); }
static void * load_sym(void * h, const char * n) { return (void *) GetProcAddress((HMODULE) h, n); }
static void sleep_ms(int ms) { Sleep(ms); }
static double now_ms(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return 1000.0 * (double) c.QuadPart / (double) f.QuadPart; }
#else
#include <dlfcn.h>
#include <unistd.h>
static void * load_loader(void) { return dlopen("libvulkan.so.1", RTLD_NOW); }
static void * load_sym(void * h, const char * n) { return dlsym(h, n); }
static void sleep_ms(int ms) { usleep(ms * 1000); }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000.0 + t.tv_nsec / 1e6; }
#endif
#include "work_spv.h"

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "gpu_keeper: %s failed: %d\n", #x, (int) r_); return 1; } } while (0)
#define INST(n) PFN_vk##n n = (PFN_vk##n) GetInstanceProcAddr(inst, "vk" #n)
#define DEV(n) PFN_vk##n n = (PFN_vk##n) GetDeviceProcAddr(dev, "vk" #n)

static int cmp_d(const void * a, const void * b) { double x = *(const double *) a, y = *(const double *) b; return x < y ? -1 : x > y; }

int main(int argc, char ** argv) {
    const char * want = NULL;
    int sleep_for = 1, secs = 0, report_s = 5;
    uint32_t iters = 2000, groups = 1024;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--device")) want = argv[i + 1];
        else if (!strcmp(argv[i], "--sleep-ms")) sleep_for = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--iters")) iters = (uint32_t) atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--groups")) groups = (uint32_t) atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seconds")) secs = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--report-s")) report_s = atoi(argv[i + 1]);
        else { fprintf(stderr, "gpu_keeper: unknown option %s\n", argv[i]); return 2; }
    }
    void * lib = load_loader();
    if (!lib) { fprintf(stderr, "gpu_keeper: no Vulkan loader\n"); return 1; }
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) load_sym(lib, "vkGetInstanceProcAddr");
    PFN_vkCreateInstance CreateInstance = (PFN_vkCreateInstance) GetInstanceProcAddr(NULL, "vkCreateInstance");
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "gpu_keeper", 1, NULL, 0, VK_API_VERSION_1_0 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL };
    VkInstance inst;
    CHECK(CreateInstance(&ici, NULL, &inst));
    INST(EnumeratePhysicalDevices); INST(GetPhysicalDeviceProperties); INST(GetPhysicalDeviceQueueFamilyProperties);
    INST(GetPhysicalDeviceMemoryProperties); INST(CreateDevice); INST(GetDeviceProcAddr);
    uint32_t n = 16;
    VkPhysicalDevice pds[16], pd = VK_NULL_HANDLE;
    CHECK(EnumeratePhysicalDevices(inst, &n, pds));
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < n && !pd; i++) {
        GetPhysicalDeviceProperties(pds[i], &props);
        if (want ? strstr(props.deviceName, want) != NULL : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pd = pds[i];
    }
    if (!pd) { fprintf(stderr, "gpu_keeper: no matching GPU\n"); return 1; }
    GetPhysicalDeviceProperties(pd, &props);
    uint32_t qn = 16, qf = UINT32_MAX;
    VkQueueFamilyProperties qp[16];
    GetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp);
    for (uint32_t i = 0; i < qn && qf == UINT32_MAX; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) qf = i;
    float prio = 0.0f;   // lowest: it must yield to the real work
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 0, NULL, NULL };
    VkDevice dev;
    CHECK(CreateDevice(pd, &dci, NULL, &dev));
    DEV(GetDeviceQueue); DEV(CreateBuffer); DEV(GetBufferMemoryRequirements); DEV(AllocateMemory); DEV(BindBufferMemory);
    DEV(CreateShaderModule); DEV(CreateDescriptorSetLayout); DEV(CreatePipelineLayout); DEV(CreateComputePipelines);
    DEV(CreateDescriptorPool); DEV(AllocateDescriptorSets); DEV(UpdateDescriptorSets); DEV(CreateCommandPool);
    DEV(AllocateCommandBuffers); DEV(BeginCommandBuffer); DEV(CmdBindPipeline); DEV(CmdBindDescriptorSets); DEV(CmdPushConstants);
    DEV(CmdDispatch); DEV(EndCommandBuffer); DEV(CreateFence); DEV(ResetFences); DEV(QueueSubmit); DEV(WaitForFences);
    VkQueue q;
    GetDeviceQueue(dev, qf, 0, &q);
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    VkBuffer buf;
    CHECK(CreateBuffer(dev, &bci, NULL, &buf));
    VkMemoryRequirements mr;
    GetBufferMemoryRequirements(dev, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    GetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && mt == UINT32_MAX; i++)
        if ((mr.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) mt = i;
    if (mt == UINT32_MAX) for (uint32_t i = 0; i < mp.memoryTypeCount && mt == UINT32_MAX; i++) if (mr.memoryTypeBits >> i & 1) mt = i;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt };
    VkDeviceMemory mem;
    CHECK(AllocateMemory(dev, &mai, NULL, &mem));
    CHECK(BindBufferMemory(dev, buf, mem, 0));
    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof(work_spv), work_spv };
    VkShaderModule sm;
    CHECK(CreateShaderModule(dev, &smi, NULL, &sm));
    VkDescriptorSetLayoutBinding lb = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &lb };
    VkDescriptorSetLayout dsl;
    CHECK(CreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr };
    VkPipelineLayout pl;
    CHECK(CreatePipelineLayout(dev, &pli, NULL, &pl));
    VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL }, pl, VK_NULL_HANDLE, 0 };
    VkPipeline pipe;
    CHECK(CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &dps };
    VkDescriptorPool dp;
    CHECK(CreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl };
    VkDescriptorSet ds;
    CHECK(AllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo dbi = { buf, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet wds = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi, NULL };
    UpdateDescriptorSets(dev, 1, &wds, 0, NULL);
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, qf };
    VkCommandPool cp;
    CHECK(CreateCommandPool(dev, &cpci, NULL, &cp));
    VkCommandBufferAllocateInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    VkCommandBuffer cb;
    CHECK(AllocateCommandBuffers(dev, &cbi, &cb));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, 0, NULL };
    CHECK(BeginCommandBuffer(cb, &bi));
    CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    CmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &iters);
    CmdDispatch(cb, groups, 1, 1);
    CHECK(EndCommandBuffer(cb));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    VkFence fence;
    CHECK(CreateFence(dev, &fci, NULL, &fence));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL };
    printf("gpu_keeper: %s, iters %u, groups %u, sleep %d ms\n", props.deviceName, iters, groups, sleep_for);
    fflush(stdout);
    enum { CAP = 4096 };
    static double t[CAP];
    int m = 0;
    const double t0 = now_ms();
    double last = t0, busy = 0;
    for (;;) {
        const double a = now_ms();
        CHECK(QueueSubmit(q, 1, &si, fence));
        CHECK(WaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        CHECK(ResetFences(dev, 1, &fence));
        const double b = now_ms();
        if (m < CAP) t[m++] = b - a;
        busy += b - a;
        if (b - last >= report_s * 1000.0) {
            qsort(t, (size_t) m, sizeof(double), cmp_d);
            printf("%8.1f s  jobs %5d  job ms: min %7.3f  median %7.3f  max %8.3f  busy %3.0f %%\n", (b - t0) / 1000.0, m, t[0], t[m / 2], t[m - 1], 100.0 * busy / (b - last));
            fflush(stdout);
            m = 0; busy = 0; last = b;
        }
        if (secs && b - t0 >= secs * 1000.0) break;
        if (sleep_for) sleep_ms(sleep_for);
    }
    return 0;
}
