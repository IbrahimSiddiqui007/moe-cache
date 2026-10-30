// moe-cache plugin part: which ggml builds this plugin build has been tested with. Plain C++ without ggml, so tests/tested_ggml_test.cpp can check it on its own.
// The list is comma-separated; an entry is a ggml version ("0.27.0": every build of it) or a version and the start of a commit ("0.28.0@ff0d396e3": that build
// only). A ggml version number covers many llama.cpp builds (0.28.0 began at b11966 and a build after b12000 changed a struct layout without a bump), so the
// builds CI really runs are named by commit.
#pragma once
#include <cstring>

namespace mctested {

// true when `version` (ggml_version()) and `commit` (ggml_commit(), may be null or empty) match an entry of `list`
inline bool tested(const char * list, const char * version, const char * commit) {
    const size_t n = strlen(version);
    const size_t nc = commit ? strlen(commit) : 0;
    for (const char * p = list; *p; ) {
        while (*p == ' ' || *p == ',') {
            p++;
        }
        const char * e = p;
        while (*e && *e != ',') {
            e++;
        }
        const char * at = p;
        while (at < e && *at != '@') {
            at++;
        }
        const size_t vlen = (size_t)(at - p);
        if (n > 0 && vlen == n && strncmp(p, version, n) == 0) {
            if (at == e) {
                return true;                                   // a bare version: any build of it
            }
            const size_t clen = (size_t)(e - at - 1);          // "@commit": the running build's commit starts with it (or it starts with the running one's)
            const size_t m = clen < nc ? clen : nc;
            if (clen > 0 && m >= 7 && strncmp(at + 1, commit, m) == 0) {
                return true;
            }
        }
        p = e;
    }
    return false;
}

}  // namespace mctested
