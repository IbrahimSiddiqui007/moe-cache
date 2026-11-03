// moe-cache plugin part: texts of the explanatory messages that have logic of their own. Plain C++, so tests can check them without a model.
#pragma once
#include <cstdio>
#include <string>

namespace mcmsg {

// AUD-014: the address-space reservation behind a model buffer failed. llama.cpp then aborts the load with a generic error that names nothing.
inline std::string reserve_failed(size_t size) {
    char b[512];
    snprintf(b, sizeof(b),
             "moe-cache: could not reserve %.2f GiB of address space for a model buffer (a limit such as ulimit -v or vm.overcommit_memory=2 can cause this); "
             "llama-server will fail to load the model. MOE_CACHE_DISABLE=1 runs without the plugin",
             (double) size / 1073741824.0);
    return b;
}

}   // namespace mcmsg
