# approx session: B3 (level fewer = --fast + the 2 lowest-ranked experts dropped on CPU-held layers) through the gate on Flash-Next Q2_0.
# Limits: mean KLD <= 0.060, same top token >= 88 %, tasks >= stock - 5, loops <= stock + 2. Stock side from C:\dl\approx\gate_cache_fn_q2_0 (item 54).
$gq = $O
$gd = "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll"
try {
  Script1 "approx_gate_fewer_kl" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "fewer", "-Part", "kl", "-Dll", $gd)
  Script1 "approx_gate_fewer_tasks" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "fewer", "-Part", "tasks", "-Dll", $gd)
} finally { $O = $gq }
