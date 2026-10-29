# approx session 2026-10-29: stock side of the quality gate (docs/QUALITY_GATE.md), Flash-Next Q2_0, made once and kept for every level.
# stockkl: llama-perplexity, stock, f16 KV, 3 x 512 tokens of long30k.txt at batch 8 (base logits). stocktasks: stock server, 202 tasks.
$qO = $O
try {
  Script1 "approx_gate_stockkl" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "kv", "-Part", "stockkl", "-Tag", "stock")
  Script1 "approx_gate_stocktasks" 30 "C:\moe-cache-approx\tools\pc\approx_gate.ps1" @("-Level", "kv", "-Part", "stocktasks", "-Tag", "stock")
} finally { $O = $qO }
