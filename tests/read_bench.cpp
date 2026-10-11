// How fast can expert-sized O_DIRECT reads land in anonymous memory, and what do page faults and syscalls cost? (ideas 7 and 9 of docs/EXPERIMENTS.md)
//   g++ -O2 -std=c++17 tests/read_bench.cpp -o /tmp/read_bench -luring -lpthread && /tmp/read_bench FILE SLICE_KIB THREADS SECONDS [BATCH]
// Random 4 KiB-aligned reads of SLICE_KIB each from FILE (use a large model file, it is read with O_DIRECT, the page cache is bypassed).
//   pread   fresh   : madvise(DONTNEED) on the destination before each read (what the plugin's eviction does), then pread
//   pread   reused  : the destination stays mapped
//   pread   populate: DONTNEED, then MADV_POPULATE_WRITE (pre-fault), then pread
//   uring   fresh / reused / populate : BATCH reads submitted with one io_uring_enter, same three destination states
#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: read_bench FILE SLICE_KIB THREADS SECONDS [BATCH]\n"); return 1; }
    const char * path = argv[1];
    const size_t SL = (size_t) atol(argv[2]) * 1024;
    const int T = atoi(argv[3]);
    const double secs = atof(argv[4]);
    const int B = argc > 5 ? atoi(argv[5]) : 4;
    const int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); return 1; }
    const off_t fsz = lseek(fd, 0, SEEK_END);
    const size_t n_slots = (size_t) (fsz / 4096) - SL / 4096 - 1;
    printf("file %.1f GB, slice %zu KiB, %d thread(s), batch %d\n%-26s %8s\n", fsz / 1e9, SL / 1024, T, B, "case", "GB/s");
    for (int mode = 0; mode < 6; mode++) {
        const bool ring = mode >= 3;
        const int dest = mode % 3;   // 0 fresh, 1 reused, 2 fresh + populate
        std::atomic<bool> stop{false};
        std::atomic<unsigned long long> bytes{0};
        std::vector<std::thread> th;
        for (int t = 0; t < T; t++) th.emplace_back([&, t] {
            std::mt19937_64 rnd(99 + t);
            const int nb = ring ? B : 1;
            std::vector<char *> buf(nb);
            for (auto & b : buf) b = (char *) mmap(nullptr, SL, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            io_uring r;
            if (ring) io_uring_queue_init(B * 2, &r, 0);
            auto prep = [&](char * b) {
                if (dest != 1) madvise(b, SL, MADV_DONTNEED);
                if (dest == 2) madvise(b, SL, MADV_POPULATE_WRITE);
            };
            while (!stop) {
                for (int i = 0; i < nb; i++) prep(buf[i]);
                if (!ring) {
                    const ssize_t n = pread(fd, buf[0], SL, (off_t) ((rnd() % n_slots) * 4096));
                    bytes += n > 0 ? n : 0;
                } else {
                    for (int i = 0; i < nb; i++) io_uring_prep_read(io_uring_get_sqe(&r), fd, buf[i], SL, (off_t) ((rnd() % n_slots) * 4096));
                    io_uring_submit_and_wait(&r, nb);
                    for (int i = 0; i < nb; i++) {
                        io_uring_cqe * c;
                        if (io_uring_wait_cqe(&r, &c) == 0) { bytes += c->res > 0 ? c->res : 0; io_uring_cqe_seen(&r, c); }
                    }
                }
            }
            if (ring) io_uring_queue_exit(&r);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const unsigned long long b0 = bytes;
        const auto t0 = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::duration<double>(secs));
        const unsigned long long b1 = bytes;
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        stop = true;
        for (auto & x : th) x.join();
        const char * dn[] = { "fresh", "reused", "fresh+populate" };
        printf("%-26s %8.2f\n", ((ring ? std::string("uring x") + std::to_string(B) + " " : std::string("pread ")) + dn[dest]).c_str(), (b1 - b0) / dt / 1e9);
    }
    return 0;
}
