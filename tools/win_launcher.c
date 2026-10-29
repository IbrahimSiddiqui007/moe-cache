/* moe-cache.exe: the Windows kit's command for package managers (winget's portable installs need an .exe; they start it through a symlink).
   It finds its own real folder (through any symlink) and runs moe-cache.bat from there with the same arguments, then returns its exit code.
   Built by tools/make_windows_kit.sh:  x86_64-w64-mingw32-g++ -x c -O2 -municode -static -o moe-cache.exe tools/win_launcher.c   (system DLLs only) */
#include <windows.h>
#include <wchar.h>
#include <stdio.h>

static const wchar_t *skip_program_name(const wchar_t *c) {      /* the rest of our own command line, as Windows gave it */
    if (*c == L'"') { c++; while (*c && *c != L'"') c++; if (*c) c++; }
    else while (*c && *c != L' ' && *c != L'\t') c++;
    while (*c == L' ' || *c == L'\t') c++;
    return c;
}

int wmain(void) {
    wchar_t self[32768], real[32768], sysdir[MAX_PATH], cmd[40000];
    DWORD n = GetModuleFileNameW(NULL, self, 32768);
    if (!n || n >= 32768) { fwprintf(stderr, L"moe-cache: cannot find this program's path\n"); return 1; }
    /* resolve a symlink (winget's Links folder) to the kit's own folder */
    HANDLE h = CreateFileW(self, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    const wchar_t *p = self;
    if (h != INVALID_HANDLE_VALUE) {
        DWORD m = GetFinalPathNameByHandleW(h, real, 32768, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        CloseHandle(h);
        if (m && m < 32768) {
            p = real;
            if (!wcsncmp(p, L"\\\\?\\UNC\\", 8)) { real[6] = L'\\'; p = real + 6; }   /* \\?\UNC\server\x -> \\server\x */
            else if (!wcsncmp(p, L"\\\\?\\", 4)) p = real + 4;                      /* \\?\C:\x -> C:\x */
        }
    }
    wchar_t dir[32768];
    wcsncpy(dir, p, 32767); dir[32767] = 0;
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (!slash) { fwprintf(stderr, L"moe-cache: odd program path %ls\n", p); return 1; }
    slash[1] = 0;
    UINT s = GetSystemDirectoryW(sysdir, MAX_PATH);
    if (!s || s >= MAX_PATH) return 1;
    /* cmd /d /c ""DIR\moe-cache.bat" ARGS": /d skips AutoRun; the outer quotes are the ones cmd /c strips */
    int k = _snwprintf(cmd, 40000, L"\"%ls\\cmd.exe\" /d /c \"\"%lsmoe-cache.bat\" %ls\"", sysdir, dir, skip_program_name(GetCommandLineW()));
    if (k < 0 || k >= 40000) { fwprintf(stderr, L"moe-cache: command line too long\n"); return 1; }
    wchar_t app[MAX_PATH + 16];
    _snwprintf(app, MAX_PATH + 16, L"%ls\\cmd.exe", sysdir);
    STARTUPINFOW si = { sizeof si };
    PROCESS_INFORMATION pi;
    if (!CreateProcessW(app, cmd, NULL, NULL, FALSE, 0, NULL, dir, &si, &pi)) {
        fwprintf(stderr, L"moe-cache: cannot start %lsmoe-cache.bat (error %lu)\n", dir, GetLastError());
        return 1;
    }
    SetConsoleCtrlHandler(NULL, TRUE);       /* Ctrl+C goes to the batch file and the server; we wait for them and pass their exit code on */
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    return (int)code;
}
