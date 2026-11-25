// One way to read a boolean MOE_CACHE_* environment switch (audit AUD-004/005/006). Header-only, no ggml: tests/env_flag_test.cpp includes it.
//
//   unset or empty        -> the default (an empty value, e.g. docker-compose "VAR=", counts as unset)
//   0 / off / false / no  -> false (any case; surrounding blanks ignored)
//   1 / on / true / yes   -> true
//   a number              -> true when > 0, false otherwise
//   anything else         -> true (a default-off switch set to some word was always "on"; a default-on one stays on)
//
// Before this every switch had its own parse: some read atoi() (so "true" switched a default-on feature off), some tested only strcmp(v,"0")
// (so "" or "off" switched a default-off feature on), diagnostics tested only that the variable existed (so "=0" turned them on).
#pragma once
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

namespace mcenv {

inline bool flag_value(const char * v, bool def) {
    if (!v) {
        return def;
    }
    std::string s(v);
    size_t b = 0, e = s.size();
    while (b < e && isspace((unsigned char) s[b])) b++;
    while (e > b && isspace((unsigned char) s[e - 1])) e--;
    s = s.substr(b, e - b);
    if (s.empty()) {
        return def;
    }
    for (char & ch : s) ch = (char) tolower((unsigned char) ch);
    if (s == "off" || s == "false" || s == "no") {
        return false;
    }
    if (s == "on" || s == "true" || s == "yes") {
        return true;
    }
    char * end = nullptr;
    const long long n = strtoll(s.c_str(), &end, 10);
    if (end != s.c_str() && *end == '\0') {
        return n > 0;
    }
    return true;
}

inline bool env_flag(const char * name, bool def) { return flag_value(getenv(name), def); }
inline bool env_on(const char * name) { return env_flag(name, false); }

// MOE_CACHE_HYBRID is a size, not a flag (AUD-004): MiB of GPU memory for the bank, "auto" = all that is free (as size_t -1), anything that is not a
// positive number ("", 0, off, false, no, junk) = 0 = not asked. be_compute and hybrid_init both use this, so "off" cannot count as asked in one and as 0 MiB in the other.
inline size_t hybrid_mib(const char * v) {
    if (!v) {
        return 0;
    }
    std::string s(v);
    size_t b = 0, e = s.size();
    while (b < e && isspace((unsigned char) s[b])) b++;
    while (e > b && isspace((unsigned char) s[e - 1])) e--;
    s = s.substr(b, e - b);
    if (s == "auto") {
        return (size_t) -1;
    }
    char * end = nullptr;
    const long long n = strtoll(s.c_str(), &end, 10);
    return (!s.empty() && end != s.c_str() && n > 0) ? (size_t) n : 0;
}

// DECODE-104: MOE_CACHE_GPU_MIN_TOKENS (the smallest batch the GPU takes). 1 is not allowed: 1-token batches on the GPU produce one repeated token
// (Granite 3.1, T25; 2 and more are fine), so 1 becomes 2 and *clamped says so (the caller prints the warning). Unset keeps `dflt`.
inline int gpu_min_tokens_value(const char * v, int dflt, bool * clamped = nullptr) {
    if (clamped) *clamped = false;
    if (!v) return dflt;
    int n = atoi(v);
    if (n == 1) {
        n = 2;
        if (clamped) *clamped = true;
    }
    return n;
}

}  // namespace mcenv
