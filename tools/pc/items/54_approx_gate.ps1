# approx session 2026-10-29: the quality gate (docs/QUALITY_GATE.md) on Flash-Next Q2_0, ONE item, four locked jobs of at most 30 min each:
# stock base logits (3 x 512 tokens, batch 8, long30k.txt), stock 202 tasks, then level kv (B4, q8_0 KV) KL / top-1 and its 202 tasks.
# Limits for kv: mean KLD <= 0.015, same top token >= 95 %, tasks >= stock - 2, loops <= stock + 2. Results: C:\dl\approx\gate_*.txt and gate_*\gate.json.
# Later items reuse the stock side from C:\dl\approx\gate_cache_fn_q2_0. NOT CLEAN: the owner's desktop uses ~2.8 GB VRAM.
$gq = $O
$gd = "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll"
try {
  Script1 "approx_gate_stockkl" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "kv", "-Part", "stockkl", "-Tag", "stock")
  Script1 "approx_gate_stocktasks" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "kv", "-Part", "stocktasks", "-Tag", "stock")
  Script1 "approx_gate_kv_kl" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "kv", "-Part", "kl", "-Dll", $gd)
  Script1 "approx_gate_kv_tasks" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "kv", "-Part", "tasks", "-Dll", $gd)
} finally { $O = $gq }
