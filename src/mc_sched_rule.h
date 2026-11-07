// WINDOWS-013: which Windows scheduling switches are on (plat::sched_tuning in platform.h applies them). Header-only, no Windows calls: tests/sched_rule_test.cpp.
// Default ON (owner decision 2026-11-07; measured on the PC: EcoQoS off + CPU and GPU priority above normal gave +2.0 % prompt, both A B B A pairs won, decode the same):
//   MOE_CACHE_SCHED_TUNING=0|off|false|no|none   opt out of the whole bundle (launcher: --no-sched-tuning)
//   MOE_CACHE_ECOQOS_OFF=1|0                     EcoQoS (power throttling) off / leave it
//   MOE_CACHE_PROC_PRIO=above|high|off           CPU priority class of the process
//   MOE_CACHE_GPU_PRIO=above|high|realtime|off   GPU scheduling class (realtime needs rights)
// A single variable that is set wins over the bundle either way; with the bundle off, only variables set explicitly act (the behaviour before this default).
#pragma once
#include <cstring>
namespace mcsched {

struct choice {
    bool eco_off;   // opt out of EcoQoS power throttling
    int  proc;      // CPU priority: 0 none, 1 above normal, 2 high
    int  gpu;       // GPU priority class: 0 none, 3 above normal, 4 high, 5 realtime (the D3DKMT numbers)
};

inline bool is_set(const char * v) { return v && v[0]; }
inline bool is_off(const char * v) {
    return is_set(v) && (!strcmp(v, "0") || !strcmp(v, "off") || !strcmp(v, "false") || !strcmp(v, "no") || !strcmp(v, "none"));
}

inline choice choose(const char * master, const char * eco, const char * proc, const char * gpu) {
    const bool bundle = !is_off(master);
    choice c = { false, 0, 0 };
    c.eco_off = is_set(eco) ? eco[0] == '1' : bundle;
    if (is_set(proc)) {
        c.proc = is_off(proc) ? 0 : proc[0] == 'h' ? 2 : 1;
    } else if (bundle) {
        c.proc = 1;
    }
    if (is_set(gpu)) {
        c.gpu = is_off(gpu) ? 0 : gpu[0] == 'r' ? 5 : gpu[0] == 'h' ? 4 : 3;
    } else if (bundle) {
        c.gpu = 3;
    }
    return c;
}

}  // namespace mcsched
