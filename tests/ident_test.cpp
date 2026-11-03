// AUD-017: the tensor identity check looks at the middle and the end of a tensor too, not only its first 4 KiB.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/ident_test.cpp -o /tmp/ident_test && /tmp/ident_test
#include "mc_ident.h"
#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

static bool check(const std::vector<char> & mem, const std::vector<char> & file) {
    return mcident::same_samples(mem.data(), mem.size(), [&](size_t off, char * buf, size_t n) {
        if (off + n > file.size()) return false;
        memcpy(buf, file.data() + off, n);
        return true;
    });
}

int main() {
    const size_t N = 1 << 20;
    std::vector<char> mem(N);
    for (size_t i = 0; i < N; i++) mem[i] = (char) (i * 31 + (i >> 8));
    std::vector<char> file = mem;
    CHECK(check(mem, file));
    // same first 4 KiB, a difference in the middle / at the end / in the last byte: the old check passed all of these
    for (size_t pos : { N / 2, N / 2 + 100, N - 1, N - 4096 }) {
        std::vector<char> f = mem;
        f[pos] ^= 0x55;
        CHECK(!check(mem, f));
    }
    { std::vector<char> f = mem; f[100] ^= 1; CHECK(!check(mem, f)); }         // start still counts
    { std::vector<char> f = mem; f[N / 4] ^= 1; CHECK(check(mem, f)); }         // a difference outside the samples is not seen (three samples, not a full hash)
    // small tensors (below and just above one sample), including a file that is too short
    for (size_t sz : { (size_t) 1, (size_t) 100, (size_t) 4096, (size_t) 5000, (size_t) 8192, (size_t) 12288 }) {
        std::vector<char> m(sz), f;
        for (size_t i = 0; i < sz; i++) m[i] = (char) (i + 7);
        f = m;
        CHECK(check(m, f));
        f[sz - 1] ^= 1;
        CHECK(!check(m, f));
        f = m;
        f.resize(sz / 2);
        CHECK(!check(m, f) || sz == 1);
    }
    if (fails) return 1;
    printf("ident_test: ok\n");
    return 0;
}
