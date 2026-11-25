// Joint check: a pack written by `moe_cache.py pack --hash` passes the plugin's MOE_CACHE_PACK_VERIFY=1 payload check, and a corrupted one is refused.
//   g++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/pack_verify_joint.cpp -o /tmp/pack_verify_joint && /tmp/pack_verify_joint MODEL.moepack
// argv[1] = path to a .moepack. Replicates the verify loop of src/mc_state_pack.inl (pack_try_open, MOE_CACHE_PACK_VERIFY branch):
// header u32 at 28 must hold mcpack::HASH_FLAG, then for every layer record the payload CRC-32 is re-read and compared with the stored one.
// Prints "ok: N layers" (exit 0), "REFUSED: no payload hash" (exit 2) or "REFUSED: layer N: <why>" (exit 3).
#include "mc_pack_check.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.moepack\n", argv[0]);
        return 1;
    }
    FILE * f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "REFUSED: cannot open %s\n", argv[1]);
        return 1;
    }
    unsigned char h[256];
    if (fread(h, 1, sizeof(h), f) != sizeof(h) || memcmp(h, "MOEPACK1", 8) != 0) {
        fclose(f);
        fprintf(stderr, "REFUSED: not a pack file (or unfinished)\n");
        return 1;
    }
    auto rd32 = [&](size_t o) { uint32_t v; memcpy(&v, h + o, 4); return v; };
    if (!(rd32(28) & mcpack::HASH_FLAG)) {
        fclose(f);
        printf("REFUSED: no payload hash\n");
        return 2;
    }
    const uint32_t n_rec = rd32(12);
    if (n_rec == 0 || n_rec > 100000) {
        fclose(f);
        fprintf(stderr, "REFUSED: damaged header\n");
        return 1;
    }
    std::vector<unsigned char> rec((size_t) n_rec * 512);
    if (fread(rec.data(), 1, rec.size(), f) != rec.size()) {
        fclose(f);
        fprintf(stderr, "REFUSED: the layer table is cut short\n");
        return 1;
    }
    const auto vread = [&](uint64_t off, void * b, size_t n) -> size_t {
        if (fseek(f, (long long) off, SEEK_SET) != 0) { return 0; }
        return fread(b, 1, n, f);
    };
    const std::function<size_t(uint64_t, void *, size_t)> vfn = vread;
    std::vector<char> buf((size_t) (1 << 20) + 100);       // ~1 MiB, not a divisor of a layer's payload
    for (uint32_t r = 0; r < n_rec; r++) {
        const unsigned char * b = rec.data() + (size_t) r * 512;
        uint32_t layer, n_exp, want;
        uint64_t base, stride;
        memcpy(&layer, b, 4); memcpy(&n_exp, b + 4, 4); memcpy(&want, b + 12, 4);
        memcpy(&base, b + 16, 8); memcpy(&stride, b + 24, 8);
        std::string why;
        if (!mcpack::payload_crc_ok(vfn, base, stride * (uint64_t) n_exp, want, buf.data(), buf.size(), why)) {
            fclose(f);
            printf("REFUSED: layer %u: %s\n", layer, why.c_str());
            return 3;
        }
    }
    fclose(f);
    printf("ok: %u layers\n", n_rec);
    return 0;
}
