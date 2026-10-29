# S2 of the diffusion plan: RND1-Base-0910 30B-A3B Q8_0 (32.5 GB, bigger than the PC's RAM) through llama-diffusion-cli: stock vs plugin, plugin cache 24 GiB and 12 GiB.
# Order (palindrome, so drift shows): stock, p24, p12, p12, p24, stock. Canvas 256, 32 steps, eps 0.001, algorithm 1 (entropy), temp 0.5, seed 42 (llama.cpp's own RND1 example).
# Needs: C:\moe-cache-diff\llama-diffusion-cli.exe (+ its DLLs, mingw build of 23b0202a1) and C:\moe-cache-diff\ggml-moe-cache.dll (API 3). Results: C:\dl\afternoon\diff_rnd1.tsv / .log.
& {
  . C:\dl\pcb_difflib.ps1
  $dmodel = "D:\models\rnd1-base-0910-q8_0.gguf"; $dcli = "C:\moe-cache-diff\llama-diffusion-cli.exe"; $ddll = "C:\moe-cache-diff\ggml-moe-cache.dll"
  $dprompt = "The three most important differences between a diffusion language model and an autoregressive one are"
  $dplan = @(@("stock", 0), @("p24", 24), @("p12", 12), @("p12", 12), @("p24", 24), @("stock", 0))
  DNote "rnd1" "##### start"; DNote "rnd1" ("cli: " + ((& $dcli --version 2>&1 | Select-Object -Last 2) -join " | "))
  $dn = 0
  foreach ($d in $dplan) { $dn++; DRun "rnd1" $dn $d[0] 30 $dcli $dmodel $ddll $d[1] 256 32 0.001 1 0.5 $dprompt 0 @{} }
  DNote "rnd1" "##### end"
}
