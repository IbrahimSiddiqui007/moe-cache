// moe-cache plugin part: where a usage profile lives when MOE_CACHE_PROFILE points at a file that belongs to another model (AUD-016).
// Plain C++, so tests/profile_path_test.cpp can check it without a model.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>

namespace mcprofile {

// The file name this model uses next to a profile of another model: "<path>.<16 hex digits of the model fingerprint>". The other model's file is
// neither read nor overwritten, and a second start of this model finds its own file again.
inline std::string own_path(const std::string & path, uint64_t fingerprint) {
    char hex[24];
    snprintf(hex, sizeof(hex), ".%016llx", (unsigned long long) fingerprint);
    return path + hex;
}

}   // namespace mcprofile
