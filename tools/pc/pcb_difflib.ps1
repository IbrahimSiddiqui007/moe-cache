# Diffusion runs on the PC (session "diffusion", 2026-10-29): one llama-diffusion-cli run under the timed lock, with a timeout, a RAM log and the
# plugin's end-of-run stats. Dot-source it inside a script block in a queue item (& { . C:\dl\pcb_difflib.ps1 ... }). Needs Take-PcLock / Free-PcLock
# (C:\dl\pcb_lock.ps1, already loaded by the runner). Results: C:\dl\afternoon\diff_<item>_<run>.log (stderr+stdout of the CLI) and diff_<item>.tsv.
# Variable names avoid $O, $o and $I (the runner's). Stock runs on Windows have no memory limit (their model lives in the file cache); the plugin's
# cache is sized by MOE_CACHE_RAM_GIB.
$DOUT = "C:\dl\afternoon"
function DNote($item, $text) { "$(Get-Date -Format s) $text" | Tee-Object -Append "$DOUT\diff_$item.log" }
function DAvail { (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1KB }
function DKill($root) { $all = Get-CimInstance Win32_Process; $ids = @($root); $add = $true
  while ($add) { $add = $false; foreach ($p in $all) { if ($ids -contains $p.ParentProcessId -and -not ($ids -contains $p.ProcessId)) { $ids += $p.ProcessId; $add = $true } } }
  [array]::Reverse($ids); foreach ($i2 in $ids) { Stop-Process -Id $i2 -Force -ErrorAction SilentlyContinue } }
# DProc item run label minutes exe args(array) envs(hashtable) model: one process under the timed lock, with a timeout, RAM peak and the last "time per step" line
function DProc($item, $run, $label, $min, $exe, $a, $envs, $model, $tail) {
  foreach ($x in @($exe, $model)) { if (-not (Test-Path $x)) { DNote $item "SKIPPED run $run ($label): missing $x"; return } }
  Take-PcLock "diff-$item-$run"
  $p = $null; $peak = 0.0; $tot = (Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1MB; $log = "$DOUT\diff_${item}_$run"
  try {
    & C:\moe-cache-ov\tools\win_flush_cache.ps1 -Target $model -Other "$($model.Substring(0,1)):\models" | Out-Null
    DNote $item ("run {0} ({1}) start; not clean: owner desktop uses ~2.8 GB VRAM; available RAM {2:N1} GiB of {3:N1}" -f $run, $label, (DAvail), $tot)
    foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
    $t0 = Get-Date
    $p = Start-Process -FilePath $exe -ArgumentList $a -PassThru -NoNewWindow -RedirectStandardOutput "$log.out.txt" -RedirectStandardError "$log.err.txt"
    $end = $t0.AddMinutes($min)
    while (-not $p.HasExited -and (Get-Date) -lt $end) { Start-Sleep 5; $u = $tot - (DAvail); if ($u -gt $peak) { $peak = $u } }
    if (-not $p.HasExited) { DKill $p.Id; DNote $item "TIMEOUT run $run ($label) after $min min: killed its processes" }
    $sec = ((Get-Date) - $t0).TotalSeconds
    $ts = (Select-String -Path "$log.err.txt", "$log.out.txt" -Pattern $tail -ErrorAction SilentlyContinue | Select-Object -Last 1).Line
    DNote $item ("run {0} ({1}) end after {2:N0} s; peak RAM in use on the PC {3:N1} GiB; {4}" -f $run, $label, $sec, $peak, $ts)
    $f = "$DOUT\diff_$item.tsv"; if (-not (Test-Path $f)) { "item`trun`tlabel`tseconds`tpeak_gib`tresult" | Set-Content $f }
    "{0}`t{1}`t{2}`t{3:N0}`t{4:N1}`t{5}" -f $item, $run, $label, $sec, $peak, $ts | Add-Content $f
  } finally {
    if ($p -and -not $p.HasExited) { DKill $p.Id }
    foreach ($k in $envs.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    Free-PcLock
  }
}
function DPluginEnv($dll, $model, $ram, $more) { $e = @{ GGML_BACKEND_PATH = $dll; MOE_CACHE_SIZE_GIB = "auto"; MOE_CACHE_GGUF = $model; MOE_CACHE_RAM_GIB = "$ram"; MOE_CACHE_STATS = "1" }
  foreach ($k in $more.Keys) { $e[$k] = $more[$k] }; return $e }
# DRun item run label minutes cli model dll ramGiB canvas steps eps algorithm temp prompt ngl moreEnv  (ramGiB 0 = stock, no plugin)
function DRun($item, $run, $label, $min, $cli, $model, $dll, $ram, $canvas, $steps, $eps, $alg, $temp, $prompt, $ngl, $more) {
  if ($ram -gt 0 -and -not (Test-Path $dll)) { DNote $item "SKIPPED run $run ($label): missing $dll"; return }
  $a = @("-m", "`"$model`"", "-ngl", "$ngl", "-ub", "$canvas", "-b", "$canvas", "-c", "$canvas", "-fa", "on", "-lm", "mmap", "-s", "42", "--temp", "$temp",
         "--diffusion-steps", "$steps", "--diffusion-algorithm", "$alg", "--diffusion-eps", "$eps", "-p", "`"$prompt`"")
  $e = if ($ram -gt 0) { DPluginEnv $dll $model $ram $more } else { @{} }
  DProc $item $run $label $min $cli $a $e $model "time per step"
}
# DBench item run label minutes bench model dll ramGiB tokens: llama-bench prefill of `tokens` (ub = b = tokens), 2 reps; ramGiB 0 = stock
function DBench($item, $run, $label, $min, $bench, $model, $dll, $ram, $tokens) {
  $a = @("-m", "`"$model`"", "-ngl", "0", "-fa", "1", "-lm", "mmap", "-n", "0", "-p", "$tokens", "-b", "$tokens", "-ub", "$tokens", "-r", "2")
  $e = if ($ram -gt 0) { DPluginEnv $dll $model $ram @{} } else { @{} }
  DProc $item $run $label $min $bench $a $e $model "pp$tokens"
}
