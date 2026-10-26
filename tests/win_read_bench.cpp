// Where does the time of an expert read go: the drive, or the memory the read lands in? Portable (src/platform.h), made for Windows, where the
// plugin's ~1 MB unbuffered decode reads reached only ~1.2 GB/s on the desktop PC (Qwen3.5-122B; sequential reads 2.4 GB/s) and 8 or 16 reader
// threads were no faster than 4. Random 4 KiB-aligned reads of SLICE_KIB from FILE, THREADS in parallel, SECONDS per row:
//   unbuffered fresh : decommit + commit the destination before each read, then read into it (what eviction + reload does in the plugin)
//   unbuffered reused: the destination stays committed and touched
//   bounce           : unbuffered read into a reused buffer, memcpy into fresh committed pages (the plugin's bounce-buffer path)
//   commit only      : decommit + commit + write the pages, no read (the cost of fresh pages alone)
//   buffered fresh   : like "unbuffered fresh" through the file cache (fast only for data the cache already holds: say which run it is)
//   x86_64-w64-mingw32-g++ -std=c++17 -O2 -Isrc tests/win_read_bench.cpp -o win_read_bench.exe -static
//   win_read_bench.exe MODEL.gguf 1024 4 10        (Linux: g++ -std=c++17 -O2 -Isrc tests/win_read_bench.cpp -o /tmp/wrb -lpthread)
#include "platform.h"
#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: win_read_bench FILE SLICE_KIB THREADS SECONDS\n");
        return 1;
    }
    const char * path = argv[1];
    const size_t SL = (size_t) atol(argv[2]) * 1024, SPAN = (SL + 2 * 4096 + 4095) / 4096 * 4096;
    const int T = std::max(1, atoi(argv[3]));
    const double SEC = atof(argv[4]);
    plat::file_t probe = plat::file_open(path, false);
    if (!probe.ok()) {
        fprintf(stderr, "cannot open %s: %s\n", path, plat::last_error().c_str());
        return 1;
    }
    uint64_t fsz = 0;
    {
        FILE * f = plat::fopen_utf8(path, "rb");
        if (f) {
#ifdef _WIN32
            _fseeki64(f, 0, SEEK_END); fsz = (uint64_t) _ftelli64(f);
#else
            fseeko(f, 0, SEEK_END); fsz = (uint64_t) ftello(f);
#endif
            fclose(f);
        }
    }
    if (fsz < SL * 64) {
        fprintf(stderr, "file too small for this slice size\n");
        return 1;
    }
    printf("file %.1f GiB, slice %zu KiB, %d threads, %.0f s per row\n", fsz / 1073741824.0, SL / 1024, T, SEC);
    enum { UNBUF_FRESH, UNBUF_REUSED, BOUNCE, COMMIT_ONLY, BUF_FRESH };
    const char * names[] = { "unbuffered fresh ", "unbuffered reused", "bounce           ", "commit only      ", "buffered fresh   " };
    for (int mode = 0; mode < 5; mode++) {
        std::atomic<uint64_t> bytes{0}, ops{0};
        std::atomic<bool> stop{false};
        std::vector<std::thread> th;
        for (int t = 0; t < T; t++) {
            th.emplace_back([&, t] {
                const bool buffered = mode == BUF_FRESH;
                plat::file_t f = plat::file_open(path, !buffered);
                char * dst = (char *) plat::mem_reserve(SPAN);
                char * bounce = (char *) plat::aligned_alloc_page(SPAN);
                plat::mem_commit(dst, SPAN);
                memset(dst, 1, SPAN);
                memset(bounce, 1, SPAN);
                std::mt19937_64 rng(1234 + t);
                const uint64_t nslots = (fsz - SPAN) / 4096;
                while (!stop.load(std::memory_order_relaxed)) {
                    const uint64_t off = (rng() % nslots) * 4096;
                    if (mode == UNBUF_FRESH || mode == BUF_FRESH || mode == COMMIT_ONLY || mode == BOUNCE) {
                        plat::mem_decommit(dst, SPAN);
                        plat::mem_commit(dst, SPAN);
                    }
                    size_t done = 0;
                    if (mode == COMMIT_ONLY) {
                        memset(dst, 2, SL);
                        done = SL;
                    } else {
                        char * into = mode == BOUNCE ? bounce : dst;
                        while (done < SL) {
                            const ptrdiff_t got = plat::file_pread(f, into + done, SL - done, off + done);
                            if (got <= 0) {
                                fprintf(stderr, "read failed: %s\n", plat::last_error().c_str());
                                exit(1);
                            }
                            done += (size_t) got;
                        }
                        if (mode == BOUNCE) {
                            memcpy(dst, bounce, SL);
                        }
                    }
                    bytes += done;
                    ops++;
                }
                plat::aligned_free_page(bounce);
                plat::mem_release(dst, SPAN);
#ifdef _WIN32
                CloseHandle(f.h);
#else
                close(f.fd);
#endif
            });
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::duration<double>(SEC));
        stop = true;
        for (auto & x : th) {
            x.join();
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("%s  %6.2f GB/s  %8.0f ops/s  %7.3f ms per op per thread\n", names[mode], bytes / s / 1e9, ops / s, ops ? 1000.0 * s * T / ops : 0.0);
        fflush(stdout);
    }
#ifdef _WIN32
    CloseHandle(probe.h);
#else
    close(probe.fd);
#endif
    return 0;
}
