// moe-cache plugin part: compute pool of the executor.
// Included by moe_cache_plugin.cpp (one translation unit); not compiled on its own.

namespace {

// ---- compute pool for decode-time expert matrix products (MOE_CACHE_EXEC=1) ----
// llama.cpp splits the rows of an expert statically over its threads and waits for all SSD reads before it starts. This pool takes row chunks
// from a shared counter, starts on the experts that are already in RAM and waits only for the others, so it can also use slower cores.
// It calls the same dot-product functions with the same arguments, so every output value is bit-identical to llama.cpp's.
static inline void cpu_relax() {
#if defined(__x86_64__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

constexpr int EXEC_MAXCOL = 8;   // tokens per step the executor takes (single-token decode and small batches such as speculative verification)

struct exec_task_t {
    const char * w;        // first row of this expert's matrix
    int          ncol = 0; // activation columns that use this expert (one per token that picked it)
    const char * cols[EXEC_MAXCOL];   // quantised activation columns
    float *      dsts[EXEC_MAXCOL];   // outputs (rows floats each)
    const uint8_t * flag;  // non-null: the expert is still being read while *flag is non-zero
    int64_t      rows;
};

struct exec_job_t {
    const exec_task_t * tasks = nullptr;
    int      total = 0;        // chunks
    int      nch = 1;          // chunks per task
    int      chunk = 32;       // rows per chunk
    int64_t  ne00 = 0;
    size_t   nb01 = 0;
    ggml_vec_dot_t dot = nullptr;
    std::atomic<int> next{0};
    std::atomic<int> done{0};
};

struct exec_pool_t {
    std::vector<std::thread> th;
    std::atomic<uint32_t> gen{0};
    std::atomic<bool> stop{false};
    std::atomic<exec_job_t *> cur{nullptr};
    std::atomic<int> inside{0};
    std::atomic<int> parked{0};
    std::mutex m;
    std::condition_variable cv;
    exec_job_t jobs[2];
    int which = 0;

    static void work(exec_job_t * j) {
        for (;;) {
            const int c = j->next.fetch_add(1, std::memory_order_relaxed);
            if (c >= j->total) {
                return;
            }
            const exec_task_t & t = j->tasks[c / j->nch];
            const int64_t r0 = (int64_t) (c % j->nch) * j->chunk;
            const int64_t r1 = std::min<int64_t>(r0 + j->chunk, t.rows);
#ifndef MOE_CACHE_TEST_MUTANT_NO_EXEC_WAIT     // CI only: a build without this wait must fail the delayed-read Q2_0 gpt-oss step
            if (t.flag) {
                while (__atomic_load_n(t.flag, __ATOMIC_ACQUIRE)) {
                    cpu_relax();
                }
            }
#endif
            for (int64_t r = r0; r < r1; r++) {
                const char * wr = t.w + (size_t) r * j->nb01;       // the weight row stays in L1 while every token column uses it
                for (int k = 0; k < t.ncol; k++) {
                    j->dot((int) j->ne00, &t.dsts[k][r], 0, wr, 0, t.cols[k], 0, 1);
                }
            }
            j->done.fetch_add(1, std::memory_order_release);
        }
    }

    void start(const std::vector<int> & cpus, int n_workers) {
        for (int i = 0; i < n_workers; i++) {
            const int cpu = i < (int) cpus.size() ? cpus[(size_t) i] : -1;
            th.emplace_back([this, cpu] {
#ifdef __linux__
                if (cpu >= 0 && cpu < CPU_SETSIZE) {
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    CPU_SET(cpu, &set);
                    sched_setaffinity(0, sizeof(set), &set);
                }
#endif
                uint32_t seen = gen.load(std::memory_order_acquire);
                for (;;) {
                    int spins = 0;
                    while (gen.load(std::memory_order_acquire) == seen) {
                        if (stop.load(std::memory_order_relaxed)) {
                            return;
                        }
                        if (++spins < 40000) {
                            cpu_relax();
                        } else {
                            std::unique_lock<std::mutex> lk(m);
                            parked++;
                            cv.wait_for(lk, std::chrono::microseconds(200));
                            parked--;
                            spins = 0;
                        }
                    }
                    seen = gen.load(std::memory_order_acquire);
                    inside.fetch_add(1);
                    exec_job_t * j = cur.load(std::memory_order_acquire);
                    if (j) {
                        work(j);
                    }
                    inside.fetch_sub(1);
                }
            });
        }
    }

    // runs the job on the pool and the calling thread; returns when every chunk is done and no worker is inside the job any more.
    // One caller only (the backend's compute thread).
    void run(const exec_task_t * tasks, int n_tasks, int64_t rows, int64_t ne00, size_t nb01, ggml_vec_dot_t dot, int chunk) {
        // a worker that saw the previous generation late may only now be entering (it counts itself in `inside` before it reads `cur`):
        // wait for it to leave before this job slot is rewritten, so no worker ever reads a job while it is being filled
        while (inside.load() > 0) {
            cpu_relax();
        }
        exec_job_t * j = &jobs[which];
        which ^= 1;
        j->tasks = tasks;
        j->nch = (int) ((rows + chunk - 1) / chunk);
        j->total = n_tasks * j->nch;
        j->chunk = chunk;
        j->ne00 = ne00;
        j->nb01 = nb01;
        j->dot = dot;
        j->next.store(0, std::memory_order_relaxed);
        j->done.store(0, std::memory_order_relaxed);
        cur.store(j, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
        if (parked.load() > 0) {
            cv.notify_all();
        }
        work(j);
        while (j->done.load(std::memory_order_acquire) < j->total || inside.load() > 0) {
            cpu_relax();
        }
    }

    ~exec_pool_t() {
        stop = true;
        cv.notify_all();
        for (auto & t : th) {
            t.join();
        }
    }
};

// true when the server was started with --no-repack / -nr (or LLAMA_ARG_REPACK=0): stock then runs the plain CPU kernels, so the plugin must not repack either
// (repacked kernels add up in another order and the tokens would no longer match stock)

}  // namespace
