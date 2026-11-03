#!/usr/bin/env python3
"""LINUX-018: moe_cache.py check says the hipblas / rocblas libraries are probably missing when ggml-hip is next to llama-server but no AMD device is listed.
Model-free (hip_missing_hint takes the file names and a finder). Also WINDOWS-082's pure parts live in tests/job_limit_test.py.
   python3 tests/hip_hint_test.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin"))
import moe_cache as mc
bad = 0
def check(ok, what):
    global bad
    print(("  ok   " if ok else "  FAIL ") + what)
    bad += 0 if ok else 1
H = mc.hip_missing_hint
NO = "Available devices:\n"
GPU = "Available devices:\n  ROCm0: AMD Radeon RX 7900 XT (20464 MiB, 20000 MiB free)\n"
W = ["llama-server.exe", "ggml-hip.dll", "ggml-cpu.dll", "rocblas.dll"]
r = H("/x/llama-server.exe", NO, windows=True, names=["llama-server.exe", "ggml-hip.dll"], find=lambda n: False)
check(r and "hipblas.dll" in r and "rocblas.dll" in r and "PATH" in r, "Windows: ggml-hip.dll, no device, both DLLs missing -> names them and says PATH")
r = H("/x/llama-server.exe", NO, windows=True, names=W, find=lambda n: n == "rocblas.dll")
check(r and "hipblas.dll" in r and "rocblas.dll is not" not in r and "hipblas.dll is not on PATH" in r, "Windows: only hipblas.dll missing -> only that one named")
r = H("/x/llama-server.exe", NO, windows=True, names=["ggml-hip.dll"], find=lambda n: True)
check(r and "does not see a supported card" in r, "Windows: DLLs present but no device -> driver / supported card hint")
check(H("/x/llama-server.exe", GPU, windows=True, names=W, find=lambda n: False) is None, "a HIP / ROCm device is listed -> no hint")
check(H("/x/llama-server.exe", NO, windows=True, names=["llama-server.exe", "ggml-vulkan.dll"], find=lambda n: False) is None, "no ggml-hip next to llama-server -> no hint")
r = H("/x/llama-server", NO, windows=False, names=["llama-server", "libggml-hip.so"], find=lambda n: False)
check(r and "libhipblas.so" in r and "librocblas.so" in r and "LD_LIBRARY_PATH" in r, "Linux: libggml-hip.so, libraries missing -> names them, mentions LD_LIBRARY_PATH")
check(H("/nonexistent/llama-server", NO, windows=False) is None, "unreadable folder -> no hint, no crash")
print("hip_hint_test:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
