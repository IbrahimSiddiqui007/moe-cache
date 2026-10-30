# approx session 2026-10-30 (re-queue of 54: the runner's `$o = Get-CimInstance` overwrites $O, so Script1 failed at once; this item starts the gate itself).
# Gate on Flash-Next Q2_0: stock base logits, stock 202 tasks, then level kv KL and tasks (limits in docs/QUALITY_GATE.md). NOT CLEAN: owner desktop VRAM.
$gjobs = @(
  @("stock", "kv", "stockkl", "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll"),
  @("stock", "kv", "stocktasks", "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll"),
  @("kv", "kv", "kl", "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll"),
  @("kv", "kv", "tasks", "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll")
)
foreach ($gj in $gjobs) {
  $gn = "approx_gate_" + $gj[0] + "_" + $gj[2]
  $gp = Start-Process -FilePath powershell -PassThru -NoNewWindow -RedirectStandardOutput "C:\dl\approx\$gn.stdout.txt" -RedirectStandardError "C:\dl\approx\$gn.stderr.txt" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\moe-cache-approx\tools\pc\approx_gate.ps1", "-Level", $gj[1], "-Part", $gj[2], "-Tag", $gj[0], "-Dll", $gj[3])
  if (-not $gp.WaitForExit(45 * 60000)) {
    KillTree $gp.Id
    $gc = Get-Content C:\dl\pc_job.lock -ErrorAction SilentlyContinue; if ("$gc" -like "timed pc-bench-approx-gate*") { Remove-Item C:\dl\pc_job.lock }
    "$(Get-Date -Format s) TIMEOUT $gn" | Add-Content C:\dl\approx\gate.log
  }
}
