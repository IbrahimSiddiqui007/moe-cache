// Windows timer tick: `timer_res query` prints it, `timer_res hold` keeps it at 1 ms (timeBeginPeriod) until killed.
#include <windows.h>
#include <stdio.h>
#include <string.h>
typedef LONG (NTAPI * query_t)(PULONG, PULONG, PULONG);
typedef UINT (WINAPI * tbp_t)(UINT);
int main(int argc, char ** argv) {
    query_t q = (query_t) GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQueryTimerResolution");
    ULONG lo = 0, hi = 0, cur = 0;
    q(&lo, &hi, &cur);
    printf("timer tick now %.3f ms (coarsest %.3f, finest %.3f)\n", cur / 1e4, lo / 1e4, hi / 1e4);
    fflush(stdout);
    if (argc > 1 && !strcmp(argv[1], "hold")) {
        tbp_t b = (tbp_t) GetProcAddress(LoadLibraryA("winmm.dll"), "timeBeginPeriod");
        b(1);
        q(&lo, &hi, &cur);
        printf("holding: tick now %.3f ms\n", cur / 1e4);
        fflush(stdout);
        for (;;) Sleep(60000);
    }
    return 0;
}
