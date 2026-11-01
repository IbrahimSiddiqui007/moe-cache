// Model-free test of the Linux SIGBUS message (audit A1-04): map a file, cut it short, touch the cut part. The child must print the moe-cache message and
// die from SIGBUS (default action restored), not exit quietly. Run: g++ -std=c++17 -Wall -Werror -Wno-unused-function -Isrc tests/sigbus_test.cpp -o /tmp/sigbus_test && /tmp/sigbus_test
#include "platform.h"
#include <sys/wait.h>

int main() {
#ifndef __linux__
    printf("sigbus_test: skipped (Linux only)\n");
    return 0;
#else
    char path[] = "/tmp/sigbus_test_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0 || ftruncate(fd, 3 * 4096) != 0) {
        printf("sigbus_test: FAIL (temp file)\n");
        return 1;
    }
    close(fd);
    int pipefd[2];
    if (pipe(pipefd) != 0) { return 1; }
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(pipefd[1], 2);
        size_t n = 0;
        const char * v = plat::file_view_map(path, &n);
        if (!v || n != 3 * 4096) { _exit(3); }
        if (truncate(path, 4096) != 0) { _exit(4); }
        volatile char c = v[2 * 4096 + 5];       // past the new end: SIGBUS
        (void) c;
        _exit(0);
    }
    close(pipefd[1]);
    char buf[512] = {0};
    const ssize_t got = read(pipefd[0], buf, sizeof(buf) - 1);
    int status = 0;
    waitpid(pid, &status, 0);
    unlink(path);
    const bool said = got > 0 && strstr(buf, "moe-cache: bus error while reading the model file") != nullptr;
    const bool died = WIFSIGNALED(status) && WTERMSIG(status) == SIGBUS;
    printf("sigbus_test: message %s, process %s\n", said ? "printed" : "MISSING", died ? "died from SIGBUS as before" : "did NOT die from SIGBUS");
    return said && died ? 0 : 1;
#endif
}
