# approx session: B3 speed. A = --fast (route skip + q8_0 KV), B = --fast=fewer (A + 2 lowest-ranked experts dropped on CPU-held layers). ctx 262144, --gpu-margin 1.5
# on both arms. 4k + 64k, then 128k, so each run stays under the 30 min lock. NOT CLEAN: the owner's desktop uses ~2.8 GB VRAM.
$bq = $O
. C:\dl\pcb_t0lib.ps1
try {
  $FN = "D:\models\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"; $D = "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll"
  ABBA "approx_b3_short" 30 $FN @("--launcher", "C:\moe-cache-approx", "--dll", $D, "--ctx", "262144", "--largs=--gpu-margin 1.5 --fast", "--prompts", "p4k,p64k") `
                                @("--launcher", "C:\moe-cache-approx", "--dll", $D, "--ctx", "262144", "--largs=--gpu-margin 1.5 --fast=fewer", "--prompts", "p4k,p64k")
  ABBA "approx_b3_128k" 30 $FN @("--launcher", "C:\moe-cache-approx", "--dll", $D, "--ctx", "262144", "--largs=--gpu-margin 1.5 --fast", "--prompts", "p128k") `
                               @("--launcher", "C:\moe-cache-approx", "--dll", $D, "--ctx", "262144", "--largs=--gpu-margin 1.5 --fast=fewer", "--prompts", "p128k")
} finally { $O = $bq }
