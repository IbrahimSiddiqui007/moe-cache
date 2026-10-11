// Host-to-device copy speed for expert-sized chunks: pageable memory vs pinned memory vs "copy into a pinned staging buffer first".
//   nvcc -O2 tests/h2d_bench.cu -o /tmp/h2d_bench && /tmp/h2d_bench [chunk_KiB=1440] [chunks_per_batch=10] [batches=400]
// llama.cpp's GPU expert cache uploads each missed expert with cudaMemcpyAsync from the model's (pageable) memory; the CUDA driver then stages the data
// through its own bounce buffer. This measures what that costs against a pinned source. The batch mimics one layer of one decoded token.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const size_t chunk = (size_t) (argc > 1 ? atoi(argv[1]) : 1440) * 1024;
    const int per = argc > 2 ? atoi(argv[2]) : 10;
    const int batches = argc > 3 ? atoi(argv[3]) : 400;
    const size_t span = (size_t) 2 << 30;                       // 2 GiB of source, chunks taken at pseudo-random places (not cache resident)
    char * pageable = (char *) malloc(span);
    memset(pageable, 1, span);
    char * pinned_src = nullptr;
    CK(cudaMallocHost((void **) &pinned_src, span));
    memset(pinned_src, 2, span);
    char * stage = nullptr;
    CK(cudaMallocHost((void **) &stage, chunk * per * 2));
    char * dev = nullptr;
    CK(cudaMalloc((void **) &dev, chunk * per));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    unsigned long long r = 88172645463325252ULL;
    auto next = [&]() { r ^= r << 13; r ^= r >> 7; r ^= r << 17; return (size_t) (r % ((span - chunk) / 4096)) * 4096; };
    std::vector<size_t> offs((size_t) batches * per);
    for (auto & o : offs) o = next();
    auto report = [&](const char * name, double secs) {
        const double gb = (double) chunk * per * batches / 1e9;
        printf("%-44s %7.2f GB/s   %6.2f ms per layer-batch\n", name, gb / secs, 1e3 * secs / batches);
    };
    for (int pass = 0; pass < 2; pass++) {                      // second pass is the one reported (first warms the driver)
        double t0 = now();
        for (int b = 0; b < batches; b++) {
            for (int i = 0; i < per; i++) CK(cudaMemcpyAsync(dev + i * chunk, pageable + offs[(size_t) b * per + i], chunk, cudaMemcpyHostToDevice, st));
            CK(cudaStreamSynchronize(st));
        }
        if (pass) report("pageable source, cudaMemcpyAsync", now() - t0);
        t0 = now();
        for (int b = 0; b < batches; b++) {
            for (int i = 0; i < per; i++) CK(cudaMemcpyAsync(dev + i * chunk, pinned_src + offs[(size_t) b * per + i], chunk, cudaMemcpyHostToDevice, st));
            CK(cudaStreamSynchronize(st));
        }
        if (pass) report("pinned source (upper bound)", now() - t0);
        t0 = now();
        for (int b = 0; b < batches; b++) {
            char * s0 = stage + ((b & 1) ? chunk * per : 0);
            for (int i = 0; i < per; i++) memcpy(s0 + i * chunk, pageable + offs[(size_t) b * per + i], chunk);
            CK(cudaMemcpyAsync(dev, s0, chunk * per, cudaMemcpyHostToDevice, st));      // one transfer for the whole layer batch
            CK(cudaStreamSynchronize(st));
        }
        if (pass) report("memcpy into pinned staging + one async copy", now() - t0);
        {
            cudaEvent_t ev[2];
            CK(cudaEventCreate(&ev[0])); CK(cudaEventCreate(&ev[1]));
            bool used[2] = { false, false };
            t0 = now();
            for (int b = 0; b < batches; b++) {
                const int k = b & 1;
                if (used[k]) CK(cudaEventSynchronize(ev[k]));                           // the copy that read this staging buffer must be done
                char * s0 = stage + (k ? chunk * per : 0);
                for (int i = 0; i < per; i++) memcpy(s0 + i * chunk, pageable + offs[(size_t) b * per + i], chunk);
                CK(cudaMemcpyAsync(dev, s0, chunk * per, cudaMemcpyHostToDevice, st));  // runs while the next batch is being staged
                CK(cudaEventRecord(ev[k], st));
                used[k] = true;
            }
            CK(cudaStreamSynchronize(st));
        }
        if (pass) report("staging, copy overlapped with next staging", now() - t0);
    }
    printf("chunk %zu KiB x %d per batch, %d batches\n", chunk / 1024, per, batches);
    return 0;
}
