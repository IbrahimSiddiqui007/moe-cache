# approx session: B4 speed. A = exact default, B = --fast=kv (q8_0 KV only; the tuner uses the freed KV for more GPU layers and -ub 1024).
# ctx 262144 on both arms (the long-context target; the KV saving is the point), --gpu-margin 1.5 on both arms (same as n22r: every long-prompt block fits).
# 4k and 64k together, 128k alone, so each run stays under the 30 min lock. NOT CLEAN: the owner's desktop uses ~2.8 GB VRAM.
$bq = $O
. C:\dl\pcb_t0lib.ps1
try {
  $FN = "D:\models\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"; $D = "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll"
  ABBA "approx_b4_short" 30 $FN @("--launcher", "C:\moe-cache-approx", "--dll", $D, "--ctx", "262144", "--largs=--gpu-margin 1.5", "--prompts", "p4k,p64k") `
                                @("--launcher", "C:\moe-cache-approx", "--dll", $D, "--ctx", "262144", "--largs=--gpu-margin 1.5 --fast=kv", "--prompts", "p4k,p64k")
  ABBA "approx_b4_128k" 30 $FN @("--launcher", "C:\moe-cache-approx", "--dll", $D, "--ctx", "262144", "--largs=--gpu-margin 1.5", "--prompts", "p128k") `
                               @("--launcher", "C:\moe-cache-approx", "--dll", $D, "--ctx", "262144", "--largs=--gpu-margin 1.5 --fast=kv", "--prompts", "p128k")
} finally { $O = $bq }
