# Quality gate on the PC (docs/QUALITY_GATE.md): one part of one level under the PC timed lock (max 30 min).
#   approx_gate.ps1 -Level kv|fast|fewer -Part stockkl|stocktasks|kl|tasks [-Tag NAME] [-Dll PATH]
# Output: C:\dl\approx\gate_<tag>_<part>\ (logs, gate.json), stdout line "GATE PASS/FAIL" in C:\dl\approx\gate_<tag>_<part>.txt. The stock base logits and the stock
# task run are made once (stockkl, stocktasks) and kept in C:\dl\approx\gate_cache_<model tag>. NOT CLEAN when the owner's desktop uses VRAM.
param([string]$Level, [string]$Part, [string]$Tag = "", [string]$Dll = "C:\moe-cache-kit\approx\ggml-moe-cache.dll", [string]$Model = "D:\models\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
       [string]$Ncpu = "46")
. C:\dl\pcb_lock.ps1
if (-not $Tag) { $Tag = $Level }
$root = "C:\dl\approx"; New-Item -ItemType Directory -Force $root | Out-Null
$cache = "$root\gate_cache_fn_q2_0"
$out = "$root\gate_${Tag}_$Part"
$g = @("C:\moe-cache-approx\tools\quality_gate.py", "--llama-server", "C:\llama-vk\llama-server.exe", "--perplexity", "C:\llama-vk\llama-perplexity.exe", "--model", $Model,
       "--lib", $Dll, "--level", $Level, "--n-cpu-moe", $Ncpu, "--ctx", "4096", "--text", "C:\dl\long30k.txt", "--cache-dir", $cache, "--out", $out, "--no-think",
       "--ram-env", "MOE_CACHE_MAX_RAM_GB=28", "--common-args=-t 16 --load-mode mmap")
switch ($Part) {
  "stockkl"     { $g += @("--stock-only", "--no-tasks") }
  "stocktasks"  { $g += @("--stock-only", "--no-kl") }
  "kl"          { $g += @("--no-tasks") }
  "tasks"       { $g += @("--no-kl") }
  default       { throw "unknown part $Part" }
}
Take-PcLock "approx-gate-$Tag-$Part"
try {
  "$(Get-Date -Format s) gate $Tag $Part start (not clean: owner desktop may use VRAM)" | Add-Content "$root\gate.log"
  & python -I @g 2>&1 | Tee-Object "$out.txt"
  "$(Get-Date -Format s) gate $Tag $Part end, exit $LASTEXITCODE" | Add-Content "$root\gate.log"
} finally { Free-PcLock }
