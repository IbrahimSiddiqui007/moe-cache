// mc_stable_vec.h: header-only growable array with address-stable elements.
//   * elements live in fixed 4096-element chunks; growing allocates new chunks
//     and never moves existing ones, so T* taken from operator[] stays valid
//   * single-writer resize() is safe to run while other threads call size()
//     and operator[] on already-published indices (acquire/release protocol)
//   C++17, no dependencies outside the standard library.
#ifndef MC_STABLE_VEC_H
#define MC_STABLE_VEC_H

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace mcstable {

template<class T>
class stable_vec {
public:
    static constexpr size_t CHUNK  = 4096;               // elements per chunk (i & 4095)
    static constexpr size_t CHUNK_SHIFT = 12;            // log2(CHUNK)
    static constexpr size_t CHUNK_SLOTS = 65536;         // chunk-pointer table size
    static constexpr size_t CAPACITY = CHUNK_SLOTS * CHUNK; // 268435456

    stable_vec() = default;
    stable_vec(const stable_vec&) = delete;
    stable_vec& operator=(const stable_vec&) = delete;

    ~stable_vec() {
        size_t n = size_.load(std::memory_order_acquire);
        for (size_t c = 0; c <= (n ? (n - 1) >> CHUNK_SHIFT : 0); ++c) {
            T* chunk = chunks_[c].load(std::memory_order_relaxed);
            if (chunk) delete[] chunk;
        }
    }

    size_t size() const { return size_.load(std::memory_order_acquire); }

    T& operator[](size_t i) {
        return chunks_[i >> CHUNK_SHIFT].load(std::memory_order_acquire)[i & (CHUNK - 1)];
    }
    const T& operator[](size_t i) const {
        return chunks_[i >> CHUNK_SHIFT].load(std::memory_order_acquire)[i & (CHUNK - 1)];
    }

    // grow-only; callers other than size()/operator[] must not race this
    void resize(size_t n) {
        if (n <= size_.load(std::memory_order_acquire)) return;
        if (n > CAPACITY) {
            fprintf(stderr, "mc_stable_vec: requested size %zu exceeds capacity %zu\n", n, CAPACITY);
            abort();
        }
        size_t have_chunks = published_chunks_.load(std::memory_order_relaxed);
        size_t need_chunks = ((n - 1) >> CHUNK_SHIFT) + 1;
        for (size_t c = have_chunks; c < need_chunks; ++c) {
            // trailing () value-initialises: PODs are zeroed, class types default-construct
            chunks_[c].store(new T[CHUNK](), std::memory_order_relaxed);
            published_chunks_.store(c + 1, std::memory_order_relaxed);
        }
        // publish: chunks fully linked before any reader may observe the new size
        size_.store(n, std::memory_order_release);
    }

private:
    std::atomic<T*> chunks_[CHUNK_SLOTS] = {};
    std::atomic<size_t> published_chunks_{0};
    std::atomic<size_t> size_{0};
};

} // namespace mcstable

#endif // MC_STABLE_VEC_H
