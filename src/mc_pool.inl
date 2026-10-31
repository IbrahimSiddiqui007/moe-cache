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

// what this thread's executor work is computing right now: the Windows crash report prints it (a fault inside a pool worker otherwise says only
// "not inside a graph"). Plain stores, one thread-local lookup per job.
struct exec_where_t {
    const char *        op = nullptr;     // name of the op's output tensor; nullptr: not inside executor work
    const exec_task_t * task = nullptr;
    int64_t             r = -1;
    int                 k = -1;
    const float *       dst = nullptr;
};
thread_local exec_where_t g_exec_where;

struct exec_job_t {
    const char * name = nullptr;          // the op (for the crash report)
    const exec_task_t * tasks = nullptr;
    int      total = 0;        // chunks
    int      nch = 1;          // chunks per task
    int      chunk = 32;       // rows per chunk
    int64_t  ne00 = 0;
    size_t   nb01 = 0;
    ggml_vec_dot_t dot = nullptr;
    std::atomic<int> next{0};
    std::atomic<int> done{0};
    uint64_t t_pub_ns = 0;     // MOE_CACHE_EXEC_TIMES: when run() published the job
};

inline uint64_t exec_now_ns() { return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct exec_pool_t {
    std::vector<std::thread> th;
    std::atomic<uint32_t> gen{0};
    std::atomic<bool> stop{false};
    std::atomic<exec_job_t *> cur{nullptr};
    std::atomic<int> inside{0};
    std::atomic<int> parked{0};
    std::atomic<uint64_t> parks{0};    // times a worker fell asleep (stats "pool_parks"; idea N30: each one costs an OS wake-up at the next layer)
    // MOE_CACHE_POOL_SPIN_US=N: a worker keeps spinning N microseconds after its last job before it sleeps (0 = the fixed 40,000 pauses).
    // The GPU part between two CPU expert layers can outlast the fixed spin, so every layer may wake the workers through the OS (N30)
    const int spin_us = [] { const char * v = getenv("MOE_CACHE_POOL_SPIN_US"); return v ? std::max(0, std::min(1000000, atoi(v))) : 0; }();
    // MOE_CACHE_EXEC_TIMES=1 (idea N25): where the executor's wall time goes. busy = threads inside the dot products, flagwait = spinning for an expert
    // still being read, wake = from publish to the moment a worker starts on the job (sum over workers), n_wake = workers that did.
    const bool times = getenv("MOE_CACHE_EXEC_TIMES") != nullptr;
    std::atomic<uint64_t> busy_ns{0}, flagwait_ns{0}, wake_ns{0}, n_wake{0}, chunks{0}, t_disp_ns{0}, t_main_ns{0}, t_tail_ns{0}, n_runs{0};
    std::mutex m;
    std::condition_variable cv;
    exec_job_t jobs[2];
    int which = 0;

    static void work(exec_job_t * j, exec_pool_t * pool = nullptr, bool worker = false) {
        exec_where_t & wh = g_exec_where;
        wh.op = j->name;
        const bool tm = pool && pool->times;
        uint64_t busy = 0, fw = 0, nch_done = 0;
        if (tm && worker) {
            pool->wake_ns += exec_now_ns() - j->t_pub_ns;
            pool->n_wake++;
        }
        for (;;) {
            const int c = j->next.fetch_add(1, std::memory_order_relaxed);
            if (c >= j->total) {
                wh.op = nullptr;
                if (tm) {
                    pool->busy_ns += busy; pool->flagwait_ns += fw; pool->chunks += nch_done;
                }
                return;
            }
            const exec_task_t & t = j->tasks[c / j->nch];
            const int64_t r0 = (int64_t) (c % j->nch) * j->chunk;
            const int64_t r1 = std::min<int64_t>(r0 + j->chunk, t.rows);
#ifndef MOE_CACHE_TEST_MUTANT_NO_EXEC_WAIT     // CI only: a build without this wait must fail the delayed-read Q2_0 gpt-oss step
            if (t.flag) {
                const uint64_t f0 = tm && __atomic_load_n(t.flag, __ATOMIC_ACQUIRE) ? exec_now_ns() : 0;
                while (__atomic_load_n(t.flag, __ATOMIC_ACQUIRE)) {
                    cpu_relax();
                }
                if (f0) {
                    fw += exec_now_ns() - f0;
                }
            }
#endif
            const uint64_t b0 = tm ? exec_now_ns() : 0;
            for (int64_t r = r0; r < r1; r++) {
                const char * wr = t.w + (size_t) r * j->nb01;       // the weight row stays in L1 while every token column uses it
                for (int k = 0; k < t.ncol; k++) {
                    wh.task = &t;
                    wh.r = r;
                    wh.k = k;
                    wh.dst = &t.dsts[k][r];
                    j->dot((int) j->ne00, &t.dsts[k][r], 0, wr, 0, t.cols[k], 0, 1);
                }
            }
            if (tm) {
                busy += exec_now_ns() - b0;
                nch_done++;
            }
            j->done.fetch_add(1, std::memory_order_release);
        }
    }

    void start(const std::vector<int> & cpus, int n_workers) {
#if defined(_WIN32)
        if (std::any_of(cpus.begin(), cpus.end(), [](int c) { return c >= 64; })) {
            fprintf(stderr, "moe-cache: note: CPUs 64 and above in the mask are not pinned (Windows processor group 0 only); their threads run unpinned\n");
        }
#endif
        for (int i = 0; i < n_workers; i++) {
            const int cpu = i < (int) cpus.size() ? cpus[(size_t) i] : -1;
            th.emplace_back([this, cpu] {
#if defined(__linux__)
                if (cpu >= 0 && cpu < CPU_SETSIZE) {
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    CPU_SET(cpu, &set);
                    sched_setaffinity(0, sizeof(set), &set);
                }
#elif defined(_WIN32)
                if (cpu >= 0 && cpu < 64) {
                    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);   // processor group 0 only
                }
#endif
                uint32_t seen = gen.load(std::memory_order_acquire);
                for (;;) {
                    int spins = 0;
                    auto idle_from = std::chrono::steady_clock::now();
                    while (gen.load(std::memory_order_acquire) == seen) {
                        if (stop.load(std::memory_order_relaxed)) {
                            return;
                        }
                        ++spins;
                        const bool keep = spin_us > 0 ? ((spins & 1023) != 0 ||
                                                         std::chrono::steady_clock::now() - idle_from < std::chrono::microseconds(spin_us))
                                                      : spins < 40000;
                        if (keep) {
                            cpu_relax();
                        } else {
                            // idle: sleep until run() publishes the next job. This used to wake every 200 us and spin again, so an idle server kept
                            // every worker at ~85 % of a core (I30, Flash-Next: ~700 % CPU for 30+ min with no request). No wake-up is lost:
                            // `parked` is raised under the mutex before `gen` is checked again, run() raises `gen` before it reads `parked` (all
                            // sequentially consistent), and run() notifies under the mutex, so the worker sees the new job or is woken
                            std::unique_lock<std::mutex> lk(m);
                            parked.fetch_add(1);
                            parks.fetch_add(1, std::memory_order_relaxed);
                            cv.wait(lk, [&] { return stop.load() || gen.load() != seen; });
                            parked.fetch_sub(1);
                            spins = 0;
                            idle_from = std::chrono::steady_clock::now();
                        }
                    }
                    seen = gen.load(std::memory_order_acquire);
                    inside.fetch_add(1);
                    exec_job_t * j = cur.load(std::memory_order_acquire);
                    if (j) {
                        work(j, this, true);
                    }
                    inside.fetch_sub(1);
                }
            });
        }
    }

    // runs the job on the pool and the calling thread; returns when every chunk is done and no worker is inside the job any more.
    // One caller only (the backend's compute thread).
    void run(const exec_task_t * tasks, int n_tasks, int64_t rows, int64_t ne00, size_t nb01, ggml_vec_dot_t dot, int chunk, const char * name) {
        if (n_tasks <= 0 || rows <= 0) {     // idea D17: no chunks, so no publish, no wake-up, no barrier
            return;
        }
        // a worker that saw the previous generation late may only now be entering (it counts itself in `inside` before it reads `cur`):
        // wait for it to leave before this job slot is rewritten, so no worker ever reads a job while it is being filled
        while (inside.load() > 0) {
            cpu_relax();
        }
        exec_job_t * j = &jobs[which];
        which ^= 1;
        j->name = name;
        j->tasks = tasks;
        j->nch = (int) ((rows + chunk - 1) / chunk);
        j->total = n_tasks * j->nch;
        j->chunk = chunk;
        j->ne00 = ne00;
        j->nb01 = nb01;
        j->dot = dot;
        j->next.store(0, std::memory_order_relaxed);
        j->done.store(0, std::memory_order_relaxed);
        j->t_pub_ns = times ? exec_now_ns() : 0;
        cur.store(j, std::memory_order_release);
        gen.fetch_add(1);              // sequentially consistent, as the parked count: see the worker's idle wait
        if (parked.load() > 0) {
            std::lock_guard<std::mutex> lk(m);
            cv.notify_all();
        }
        const uint64_t w0 = times ? exec_now_ns() : 0;
        n_runs += times;
        work(j, this);
        const uint64_t w1 = times ? exec_now_ns() : 0;
        while (j->done.load(std::memory_order_acquire) < j->total || inside.load() > 0) {
            cpu_relax();
        }
        if (times) {
            t_disp_ns += w0 - j->t_pub_ns;     // publish + wake-up call
            t_main_ns += w1 - w0;              // the calling thread's share of the chunks
            t_tail_ns += exec_now_ns() - w1;   // the calling thread waiting for the others to finish (barrier)
        }
    }

    ~exec_pool_t() {
        {
            std::lock_guard<std::mutex> lk(m);   // a worker between its check and its wait cannot miss this
            stop = true;
        }
        cv.notify_all();
        for (auto & t : th) {
            t.join();
        }
    }
};

// true when the server was started with --no-repack / -nr (or LLAMA_ARG_REPACK=0): stock then runs the plain CPU kernels, so the plugin must not repack either
// (repacked kernels add up in another order and the tokens would no longer match stock)

}  // namespace
