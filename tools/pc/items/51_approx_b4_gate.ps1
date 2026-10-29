# approx session: B4 (q8_0 KV, level kv) through the quality gate on Flash-Next Q2_0: KL / top-1 against stock, then the 202 tasks.
# Limits (docs/QUALITY_GATE.md, written before measuring): mean KLD <= 0.015, same top token >= 95 %, tasks >= stock - 2, loops <= stock + 2.
. C:\dl\pcb_ovlib.ps1
$qO = $O
try {
  Script1 "approx_gate_kv_kl" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "kv", "-Part", "kl", "-Dll", "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll")
  Script1 "approx_gate_kv_tasks" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "kv", "-Part", "tasks", "-Dll", "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll")
} finally { $O = $qO }
