# S4 of the diffusion plan: what the GPU block does for canvas-sized steps. RND1-Base-0910 Q8_0, plugin cache 24 GiB, -ub = canvas (llama-diffusion-cli sets canvas = ub).
# Per canvas 256 / 512 / 1024: GPU block default vs MOE_CACHE_GPU_MIN_TOKENS=999999 (experts on the CPU only), 16 steps; plus llama-bench prefill of the same size
# (the "prefill rate at equal tokens" reference: kill check = step time within 10 % of it). Order per canvas: block, no block, no block, block, then bench.
# Needs 80_diffusion_rnd1's kit in C:\moe-cache-diff. Results: C:\dl\afternoon\diff_gpublock.tsv / .log.
& {
  . C:\dl\pcb_difflib.ps1
  $dmodel = "D:\models\rnd1-base-0910-q8_0.gguf"; $dcli = "C:\moe-cache-diff\llama-diffusion-cli.exe"; $ddll = "C:\moe-cache-diff\ggml-moe-cache.dll"; $dbench = "C:\llama-new\llama-bench.exe"
  $dprompt = "The three most important differences between a diffusion language model and an autoregressive one are"
  DNote "gpublock" "##### start"
  $dn = 0
  foreach ($c in 256, 512, 1024) {
    foreach ($arm in @(@("block", @{}), @("noblock", @{ MOE_CACHE_GPU_MIN_TOKENS = "999999" }), @("noblock", @{ MOE_CACHE_GPU_MIN_TOKENS = "999999" }), @("block", @{}))) {
      $dn++; DRun "gpublock" $dn ("{0}_c{1}" -f $arm[0], $c) 30 $dcli $dmodel $ddll 24 $c 16 0.001 1 0.5 $dprompt 0 $arm[1]
    }
    $dn++; DBench "gpublock" $dn ("bench_pp{0}" -f $c) 20 $dbench $dmodel $ddll 24 $c
  }
  DNote "gpublock" "##### end"
}
