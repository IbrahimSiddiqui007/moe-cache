# approx session 2026-10-30 (re-queue of 56: the runner's `$o = Get-CimInstance` overwrites $O, so Script1 failed at once; this item starts the gate itself).
# Gate level fewer (B3) on Flash-Next Q2_0; the stock side is reused from C:\dl\approx\gate_cache_fn_q2_0 (item 58 must have made it).
$gjobs = @(
  @("fewer", "fewer", "kl", "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll"),
  @("fewer", "fewer", "tasks", "C:\moe-cache-kit\approx_b4\ggml-moe-cache.dll")
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
