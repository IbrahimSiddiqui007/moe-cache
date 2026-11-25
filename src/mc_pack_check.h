// moe-cache plugin part: overflow-safe size checks for the layer table of a .moepack file (AUD-019). The numbers come from the file, so a damaged or
// hostile one must not wrap a 64-bit sum around into a value that passes. Plain C++, tested by tests/pack_check_test.cpp.
#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <functional>

namespace mcpack {

// a * b <= limit, without overflow
inline bool product_le(uint64_t a, uint64_t b, uint64_t limit) {
    return a == 0 || b <= limit / a;
}

// the layer's block [base, base + stride * n_exp) lies inside a pack of pack_size bytes
inline bool layer_fits(uint64_t base, uint64_t stride, uint64_t n_exp, uint64_t pack_size) {
    return base <= pack_size && product_le(stride, n_exp, pack_size - base);
}

// a tensor's slice [in_off, in_off + nb2) lies inside one expert stride
inline bool slice_fits(uint64_t in_off, uint64_t nb2, uint64_t stride) {
    return in_off <= stride && nb2 <= stride - in_off;
}

// ---- payload hash (CACHE-695, opt-in) ----
// `moe_cache.py pack --hash` stores one CRC-32 per layer over the layer's payload bytes [base, base + stride * n_expert) of the pack: header bytes 28..31 = 1
// (HASH_FLAG) and bytes 12..15 of each layer record = that CRC. MOE_CACHE_PACK_VERIFY=1 makes the plugin re-read the payload once at load and compare.
constexpr uint32_t HASH_FLAG = 1;

// CRC-32 (zlib polynomial), continuing from `crc` (start with 0); the same value as zlib.crc32 and crc32_update in mc_common.inl
// slice-by-8: processes 8 bytes per step through 8 tables (T[0] = the plain byte-at-a-time table); the byte-by-byte loads below make it endian-independent
inline uint32_t crc32_bytes(uint32_t crc, const void * data, size_t n) {
    struct tables_t {
        uint32_t t[8][256];
        tables_t() {
            for (uint32_t i = 0; i < 256; i++) {
                uint32_t c = i;
                for (int k = 0; k < 8; k++) { c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; }
                t[0][i] = c;
            }
            for (uint32_t i = 0; i < 256; i++) {
                for (int k = 1; k < 8; k++) { t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xff]; }
            }
        }
    };
    static const tables_t tab;    // thread-safe: C++11 magic static, built exactly once
    uint32_t c = crc ^ 0xffffffffu;
    const unsigned char * p = (const unsigned char *) data;
    while (n >= 8) {
        uint32_t one  = (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
        uint32_t two  = (uint32_t) p[4] | ((uint32_t) p[5] << 8) | ((uint32_t) p[6] << 16) | ((uint32_t) p[7] << 24);
        one ^= c;
        c = tab.t[7][one & 0xff] ^ tab.t[6][(one >> 8) & 0xff] ^ tab.t[5][(one >> 16) & 0xff] ^ tab.t[4][one >> 24] ^
            tab.t[3][two & 0xff] ^ tab.t[2][(two >> 8) & 0xff] ^ tab.t[1][(two >> 16) & 0xff] ^ tab.t[0][two >> 24];
        p += 8;
        n -= 8;
    }
    for (size_t i = 0; i < n; i++) { c = tab.t[0][(c ^ p[i]) & 0xff] ^ (c >> 8); }
    return c ^ 0xffffffffu;
}

// Check one layer's payload: `read(off, buf, n)` returns the bytes it could read at pack offset `off` (n_read < n = short or failed read).
// Returns true when the CRC-32 of [base, base + len) equals `want`; false with `why` filled otherwise.
inline bool payload_crc_ok(const std::function<size_t(uint64_t, void *, size_t)> & read, uint64_t base, uint64_t len, uint32_t want,
                           void * buf, size_t buf_size, std::string & why) {
    uint32_t crc = 0;
    uint64_t done = 0;
    while (done < len) {
        const size_t n = (size_t) (len - done < buf_size ? len - done : buf_size);
        if (read(base + done, buf, n) != n) {
            why = "cannot read the pack payload to check it";
            return false;
        }
        crc = crc32_bytes(crc, buf, n);
        done += n;
    }
    if (crc != want) {
        why = "payload bytes differ from the hash stored in the pack";
        return false;
    }
    return true;
}

}   // namespace mcpack
