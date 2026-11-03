#!/usr/bin/env python3
"""bench_record_test (docs/BENCHMARK_TEMPLATE.md): the benchmark record's collector without a model, a server or a GPU. Linux and Windows readers are driven with
fake /proc files and fake command output (nvidia-smi, lsblk, typeperf, powercfg, PowerShell JSON); the Sampler with a fake probe; the record against the schema;
the renderers (card lines, row, RESULTS section, issue body); and a smoke run of the real probe on this machine's own process. Same file on both systems."""
import json, os, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
import moe_cache_record as rr  # noqa: E402

FAILS = []
NR = rr.NR


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


files, cmds = {}, {}                     # fake file system and fake commands: key = path / command name -> text (or a function of the argv)
rr.READ = lambda p: files.get(p, "")
rr.GLOB = lambda pat: sorted({k for f in files for k in (f, os.path.dirname(f)) if __import__("fnmatch").fnmatch(k, pat)})
rr.RUN = lambda cmd, timeout=20: (cmds[cmd[0]](cmd) if callable(cmds.get(cmd[0])) else cmds.get(cmd[0], ""))

# ---- 1. pure parsers
name, cores, logical = rr.parse_cpuinfo("processor\t: 0\nmodel name\t: Test CPU 9\nphysical id\t: 0\ncore id\t: 0\nprocessor\t: 1\nmodel name\t: Test CPU 9\nphysical id\t: 0\ncore id\t: 0\n"
                                         "processor\t: 2\nphysical id\t: 0\ncore id\t: 1\n")
check((name, cores, logical) == ("Test CPU 9", 2, 3), f"cpuinfo: model, 2 physical cores (SMT pair counted once), 3 logical ({name}, {cores}, {logical})")
check(rr.parse_meminfo("MemTotal:       32768000 kB\nMemAvailable:   1024 kB\n") == {"total": 32768000 * 1024, "available": 1024 * 1024}, "meminfo")
dmi = "Memory Device\n\tSize: 16 GB\n\tType: DDR5\n\tSpeed: 5600 MT/s\n\tConfigured Memory Speed: 5200 MT/s\nMemory Device\n\tSize: No Module Installed\n\tType: Unknown\n"
check(rr.parse_dmidecode(dmi) == ("DDR5", 5200, 1, 16.0), f"dmidecode: DDR5, configured speed, empty slot ignored ({rr.parse_dmidecode(dmi)})")
check(rr.parse_nvidia_smi("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 581.15\n") == [{"name": "NVIDIA GeForce RTX 4060 Laptop GPU", "vram_mib": 8188.0, "driver": "581.15"}], "nvidia-smi gpu line")
dev = "ggml_vulkan: Found 1 devices:\nAvailable devices:\n  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15890 MiB free)\n  CPU: Ryzen (32000 MiB, 20000 MiB free)\n"
check(rr.parse_list_devices(dev) == [{"dev": "Vulkan0", "name": "AMD Radeon RX 9070 XT", "total_mib": 16304.0, "free_mib": 15890.0}], "list-devices: the GPU, not the CPU line")
tp = '"(PDH-CSV 4.0)","\\\\PC\\GPU Adapter Memory(luid_0x0_0x1_phys_0)\\Dedicated Usage","\\\\PC\\GPU Adapter Memory(luid_0x0_0x2_phys_0)\\Dedicated Usage"\n' \
     '"11/03/2026 10:00:00.123","12884901888","536870912"\n\nThe command completed successfully.\n'
check(rr.parse_typeperf(tp) == {"luid_0x0_0x1_phys_0": 12884901888.0, "luid_0x0_0x2_phys_0": 536870912.0}, "typeperf: one value per adapter instance")
log = ["print_info: n_layer          = 48", "load_tensors: offloaded 49/49 layers to GPU", "load_tensors:   CPU_Mapped model buffer size = 60000.00 MiB",
       "load_tensors:      Vulkan0 model buffer size = 11234.50 MiB", "llama_kv_cache:    Vulkan0 KV buffer size =   384.00 MiB", "sched_reserve:    Vulkan0 compute buffer size =  1500.00 MiB"]
pl = rr.parse_llama_log(log)
check(pl["n_layer"] == 48 and pl["offloaded"] == 49 and pl["vram_from_log_mib"] == 13118.5 and "CPU_Mapped" in pl["buffers"], f"llama log: layers and GPU buffers ({pl['vram_from_log_mib']} MiB, CPU not counted)")
cl = "env MOE_CACHE_SIZE_GIB=auto GGML_VK_X=1 llama-server -m x.gguf -c 65536 -ngl 99 --n-cpu-moe 39 -ub 2048 -b 4096 -ctk q8_0 -ctv q8_0 -t 8 -fa on"
pc = rr.parse_cmdline(cl)
check((pc["ctx"], pc["ub"], pc["b"], pc["n_cpu_moe"], pc["ngl"], pc["kv_k"], pc["threads"], pc["flash_attn"]) == (65536, 2048, 4096, 39, 99, "q8_0", 8, "on"), f"command line: {pc}")
check(rr.env_in_line(cl) == {"MOE_CACHE_SIZE_GIB": "auto", "GGML_VK_X": "1"}, "env prefix of the command line")
check(rr.parse_cmdline("llama-server -m x --cpu-moe")["n_cpu_moe"] == 0, "--cpu-moe: all experts on the CPU")
check(rr.parse_powercfg("Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\n") == "Balanced", "powercfg plan name")
check(rr.llama_build("version: 11966 (ff0d396e3)\nbuilt with x") == (11966, "ff0d396e3"), "llama --version")

# ---- 2. Linux probe on fake /proc and fake commands
files.update({"/proc/cpuinfo": "processor : 0\nmodel name : Test CPU 9\nphysical id : 0\ncore id : 0\n", "/proc/meminfo": "MemTotal: 33554432 kB\nMemAvailable: 8388608 kB\n",
              "/proc/uptime": "7200.5 100.0\n", "/proc/stat": "cpu  100 0 100 800 0 0 0 0 0 0\n",
              "/proc/100/stat": "100 (llama-server) S 1 1 1 0 -1 0 0 0 0 0 500 100 0 0 20 0 8 0 1 0 0\n",
              "/proc/101/stat": "101 (a b) c) S 100 1 1 0 -1 0 0 0 0 0 50 50 0 0 20 0 1 0 1 0 0\n", "/proc/200/stat": "200 (other) S 1 1 1 0 -1 0 0 0 0 0 5 5 0 0 20 0 1 0 1 0 0\n",
              "/proc/100/status": "VmHWM:\t 4096000 kB\nVmRSS:\t 2048000 kB\nRssAnon:\t 1024000 kB\n", "/proc/101/status": "VmHWM: 2048 kB\nVmRSS: 1024 kB\nRssAnon: 512 kB\n",
              "/sys/class/drm/card1/device/vendor": "0x1002\n", "/sys/class/drm/card1/device/mem_info_vram_used": str(6 * 1024 ** 3) + "\n",
              "/sys/class/hwmon/hwmon3/name": "k10temp\n", "/sys/class/hwmon/hwmon3/temp1_input": "56500\n"})
cmds.update({"findmnt": "/dev/nvme0n1p2\n", "lsblk": lambda c: "nvme0n1\n" if "PKNAME" in c else "WD PC SN560 SDDPNQE-1T00 nvme 0\n",
             "dmidecode": dmi, "nvidia-smi": "", "powerprofilesctl": "balanced\n"})
lp = rr.LinuxProbe()
lp.hz = 100
check(lp.tree(100) == [100, 101] or sorted(lp.tree(100)) == [100, 101], f"linux: process tree = the server and its child, not other processes ({lp.tree(100)})")
p = lp.proc([100, 101])
check(p["rss"] == (2048000 + 1024) * 1024 and p["private"] == (1024000 + 512) * 1024 and abs(p["cpu_s"] - 7.0) < 1e-9, f"linux: RSS, anonymous RSS and CPU seconds summed ({p})")
check(lp.sys_cpu() == (2.0, 10.0) and lp.uptime_s() == 7200.5, "linux: system CPU busy / total, uptime")
check(lp.vram({100}) == (6144.0, None, "amdgpu sysfs mem_info_vram_used"), f"linux AMD: VRAM from sysfs ({lp.vram({100})})")
cmds["nvidia-smi"] = lambda c: ("100, 7000\n55, 100\n" if "compute-apps" in c[1] else "7600\n") if "memory.used" in c[1] or "compute-apps" in c[1] else "NVIDIA RTX 4060 Laptop GPU, 8188, 581.15\n"
check(lp.vram({100, 101}) == (7600.0, 7000.0, "nvidia-smi"), f"linux NVIDIA: whole card and the server's share ({lp.vram({100, 101})})")
hw = lp.hardware("/models/x.gguf")
check(hw["cpu_model"] == "Test CPU 9" and hw["ram_gib"] == 32.0 and hw["ram_type"] == "DDR5" and hw["storage"] == {"model": "WD PC SN560 SDDPNQE-1T00", "interface": "nvme", "drive": "/dev/nvme0n1p2"}
      and hw["power_plan"] == "profile balanced", f"linux hardware: cpu, ram, storage model + interface, power ({hw['storage']}, {hw['power_plan']})")
cmds["nvidia-smi"] = "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 581.15\n"
check(lp.gpu("")["gpu_driver"] == "581.15" and lp.gpu("")["vram_mib"] == 8188.0, "linux gpu: name, VRAM, driver from nvidia-smi")
cmds["nvidia-smi"] = ""
check(lp.gpu(dev)["gpu_model"] == "AMD Radeon RX 9070 XT" and lp.gpu(dev)["gpu_driver"] == NR, "linux gpu without nvidia-smi: from --list-devices, driver not recorded")
check(lp.temperatures() == {"cpu_c": 56.5}, "linux temperature: k10temp")
cmds["dmidecode"] = ""
check(rr.LinuxProbe().hardware("/m")["ram_type"] == NR, "linux: RAM type not recorded without root (dmidecode empty)")

# ---- 3. Windows probe: typeperf path and PowerShell JSON with fake output; ctypes readers are not run here
wp = rr.WindowsProbe()
procs = '"(PDH-CSV 4.0)","\\\\PC\\GPU Process Memory(pid_100_luid_0x0_0x1_phys_0)\\Dedicated Usage","\\\\PC\\GPU Process Memory(pid_555_luid_0x0_0x1_phys_0)\\Dedicated Usage"\n' \
        '"11/03/2026 10:00:00.123","10737418240","1073741824"\n'
cmds["typeperf"] = lambda c: tp if "Adapter" in c[1] else procs
check(wp.vram({100}) == (12288.0, 10240.0, "typeperf GPU Process Memory"), f"windows: card 12 GiB in use, the server's own 10 GiB, other process excluded ({wp.vram({100})})")
cmds["typeperf"] = lambda c: tp if "Adapter" in c[1] else ""
check(wp.vram({100}) == (12288.0, None, "typeperf GPU Adapter Memory"), "windows: no per-process counter -> the card's usage, labelled as such")
cmds["typeperf"] = ""
check(wp.vram({100}) == (None, None, None), "windows: no typeperf -> nothing (the Sampler then falls back to the log)")


def ps(cmd):
    s = cmd[-1]
    if "Win32_Processor" in s:
        return json.dumps({"Name": "AMD Ryzen 9 7950X 16-Core Processor  ", "NumberOfCores": 16, "NumberOfLogicalProcessors": 32})
    if "Win32_PhysicalMemory" in s:
        return json.dumps([{"Capacity": 17179869184, "Speed": 4800, "ConfiguredClockSpeed": 5600, "SMBIOSMemoryType": 34}] * 2)
    if "Get-Partition" in s:
        return json.dumps({"FriendlyName": "Samsung SSD 990 PRO 2TB", "BusType": "NVMe"}) if "-DriveLetter D" in s else ""
    if "Win32_VideoController" in s:
        return json.dumps([{"Name": "AMD Radeon RX 9070 XT", "DriverVersion": "32.0.21025.10016"}])
    return ""


cmds.update({"powershell": ps, "powercfg": "Power Scheme GUID: 8c5e7fda  (High performance)\n"})
wp.mem = lambda: {"total": 32 * 2 ** 30, "available": 10 * 2 ** 30}
hw = wp.hardware("D:\\models\\x.gguf")
check(hw["cpu_model"] == "AMD Ryzen 9 7950X 16-Core Processor" and hw["cpu_cores"] == 16 and hw["cpu_threads"] == 32, f"windows: CPU through PowerShell ({hw['cpu_model']})")
check(hw["ram_gib"] == 32.0 and hw["ram_type"] == "DDR5" and hw["ram_speed_mts"] == 5600, "windows: RAM size, DDR5, configured speed")
check(hw["storage"] == {"model": "Samsung SSD 990 PRO 2TB", "interface": "NVMe", "drive": "D:"} and hw["power_plan"] == "High performance", f"windows: the drive holding the model, power plan ({hw['storage']})")
g = wp.gpu(dev)
check(g == {"gpu_model": "AMD Radeon RX 9070 XT", "vram_mib": 16304.0, "gpu_driver": "32.0.21025.10016"}, f"windows gpu: name and VRAM from --list-devices, driver from the video controller ({g})")
cmds["powershell"] = ""
check(wp.hardware("D:\\m.gguf")["cpu_model"] == NR and wp.gpu("")["gpu_driver"] == NR, "windows without PowerShell: not recorded, no exception")

# ---- 4. Sampler on a fake probe: peaks, minima, CPU %


class FakeProbe:
    ncpu = 4

    def __init__(self):
        self.i = 0

    def tree(self, pid): return [pid, pid + 1]

    def proc(self, pids):
        self.i += 1
        return {"rss": [2, 9, 5][min(self.i - 1, 2)] * rr.MiB, "private": 1 * rr.MiB * self.i, "cpu_s": 2.0 * self.i, "hwm": 0}

    def mem(self): return {"total": 100 * rr.MiB, "available": [50, 20, 40][min(self.i - 1, 2)] * rr.MiB}

    def sys_cpu(self): return (self.i * 1.0, self.i * 4.0)

    def uptime_s(self): return 3600 * 5.0

    def temperatures(self): return {}

    def vram(self, pids): return (9000.0 + self.i, 8000.0 + self.i, "fake counter")


sm = rr.Sampler(1, pr=FakeProbe())
box = [None]
for k in range(3):
    time.sleep(0.05)
    sm.sample_once(box)
res = sm.result()
check(res["process_ram_peak_mib"] == 9 and res["system_ram_free_min_mib"] == 20 and res["samples"] == 3, f"sampler: RAM peak 9 MiB, free min 20 MiB ({res['process_ram_peak_mib']}, {res['system_ram_free_min_mib']})")
check(res["vram_used_peak_mib"] == 8003 and res["vram_system_peak_mib"] == 9003 and res["vram_source"] == "fake counter (this server)", "sampler: VRAM = the process share; whole-card peak kept apart")
check(isinstance(res["cpu_avg_pct"], float) and res["cpu_peak_pct"] >= res["cpu_avg_pct"] > 0 and res["system_cpu_avg_pct"] == 25.0, f"sampler: CPU avg <= peak, system 25 % ({res['cpu_avg_pct']}, {res['cpu_peak_pct']}, {res['system_cpu_avg_pct']})")
empty = rr.Sampler(1, pr=type("P", (FakeProbe,), {"proc": lambda s, p: None, "vram": lambda s, p: (None, None, None), "mem": lambda s: {"total": None, "available": None},
                                                    "sys_cpu": lambda s: None})(), log_lines=log)
box = [None]; empty.sample_once(box)
r0 = empty.result()
check(r0["process_ram_peak_mib"] == NR and r0["system_ram_free_min_mib"] == NR and r0["cpu_avg_pct"] == NR, "sampler: nothing readable -> not recorded")
check(r0["vram_used_peak_mib"] == 13118 and "log buffer sizes" in r0["vram_source"], f"sampler: VRAM falls back to llama-server's buffer lines, labelled ({r0['vram_used_peak_mib']}, {r0['vram_source'][:30]})")

# ---- 5. the record: build, validate, render
env = {"hardware": {**hw, **g, "cpu_model": "AMD Ryzen 9 7950X 16-Core Processor", "cpu_cores": 16, "cpu_threads": 32, "ram_gib": 32.0, "ram_type": "DDR5", "ram_speed_mts": 5600,
                    "storage": {"model": "Samsung SSD 990 PRO 2TB", "interface": "NVMe", "drive": "D:"}, "power_plan": "High performance", "os": "Windows 11", "os_build": "10.0.26100"},
       "software": {"moe_cache_version": "0.3.5", "moe_cache_commit": "abc123def", "plugin_file": "ggml-moe-cache.dll", "plugin_sha256_prefix": "0123456789ab", "llama_build": 11966,
                    "llama_commit": "ff0d396e3", "llama_backend": "Vulkan", "python": "3.12.1"},
       "model": {"file": "Fake-Q2_0.gguf", "quant": "Q2_0", "size_gb": 66.4, "sha256_head16MiB": "aabbccddeeff0011", "sha256_full": NR, "source_url": NR}}
st = rr.settings_from(cl, env={"MOE_CACHE_X": "1", "PATH": "/x", "LLAMA_API_KEY": "secret"}, log_lines=log)
check(st["gpu_expert_layers"] == 9 and st["n_cpu_moe"] == 39 and st["environment"]["MOE_CACHE_X"] == "1" and "PATH" not in st["environment"] and "LLAMA_API_KEY" not in st["environment"]
      and st["environment"]["GGML_VK_X"] == "1" and st["command_line"] == cl, f"settings: GPU expert layers = n_layer 48 - n-cpu-moe 39 = 9; env filtered, key never copied ({st['gpu_expert_layers']})")
runs = []
for i, (arm, txt, dec) in enumerate((("stock", "hello", 10.0), ("moe-cache", "hello", 22.0), ("moe-cache", "hello", 23.0), ("stock", "hello", 10.4)), 1):
    runs.append(rr.run_entry(arm, i, {"decode": dec, "prefill": 300.0, "ttft_s": 1.2, "text": txt, "wall_s": 20.0, "hit_rate": 91.0 if arm == "moe-cache" else None}, res))
cond = rr.conditions_now(pr=FakeProbe(), cold=True, baseline="PASS 49.4 tok/s", other_programs="none", temps=False)
proto = {"arms": ["stock", "moe-cache"], "order": "ABBA", "runs_per_arm": 2, "prompt_set": "bench-report/chat-v1", "prompt_sha256": "1234abcd", "prompt_tokens": 2000,
         "n_predict": 128, "temperature": 0.0, "seed": 42, "sampler": "greedy"}
rec = rr.assemble(env, st, cond, proto, runs, reproduce={"commands": [cl], "kit": "https://github.com/IbrahimSiddiqui007/moe-cache/releases", "model_url": NR})
if os.environ.get("BENCH_RECORD_DUMP"):                  # writes the filled record of this test (a sample for the docs)
    json.dump(rec, open(os.environ["BENCH_RECORD_DUMP"], "w"), indent=1)
schema = rr.load_schema()
errs = rr.validate(rec, schema)
check(not errs, f"the filled record is valid against docs/bench_record.schema.json ({errs[:2]})")
bad = json.loads(json.dumps(rec)); del bad["settings"]["gpu_expert_layers"]; bad["runs"][0]["decode_tps"] = "fast"; bad["outputs"]["identical_to_baseline"] = "maybe"
errs = rr.validate(bad, schema)
check(any("gpu_expert_layers" in e and "required" in e for e in errs) and any("decode_tps" in e for e in errs) and any("identical_to_baseline" in e for e in errs),
      f"validation catches a missing required field, a text where a number belongs, a wrong enum ({len(errs)} errors)")
guess = json.loads(json.dumps(rec)); guess["software"]["llama_build"] = "about 11966"
check(rr.validate(guess, schema), "a guessed text in a number field is refused (only \"not recorded\" is allowed there)")
check(rec["outputs"]["identical_to_baseline"] == "yes" and rec["summary"]["moe-cache"]["decode_tps"] == 22.5 and rec["summary"]["stock"]["decode_tps"] == 10.2, "summary: means per arm, text identical to baseline")
runs2 = json.loads(json.dumps(runs)); runs2[1]["text_sha256"] = "ffff"
check(rr.outputs_from(runs2)["identical_to_baseline"] == "no", "a different text hash -> identical no")
d = rr.details_lines(rec, section=80, date="11-03")
check(len(d) == 4 and d[0].startswith("GPU expert layers: 9 (n-cpu-moe 39) · -ub 2048") and "VRAM peak 7.8 GiB" in d[2] and "launcher abc123def, DLL 0123456" in d[3] and d[3].endswith("11-03 · §80")
      and "ctx 65536" in d[1] and "order ABBA" in d[1] and "cold file cache" in d[1], f"card detail lines: {d}")
check("not recorded" not in " ".join(d), "no gaps in this filled record's card lines")
check(rr.card_row(rec).endswith("| 2.21x | yes |"), f"card row: ratio and identity ({rr.card_row(rec)})")
sec = rr.results_section(rec, "Fake run (Windows PC)", 80)
check(sec.startswith("## 80. Fake run") and "Order ABBA" in sec and "PASS 49.4 tok/s" in sec and "MOE_CACHE_SIZE_GIB" in sec and "records/NAME.json" in sec, "RESULTS section carries the fields")
body = rr.issue_body(rec)
check("<details>" in body and json.loads(body.split("```json\n")[1].split("```")[0])["record_schema"] == 1, "issue body: card lines plus the whole record as JSON")
# gaps are visible
sparse = json.loads(json.dumps(rec)); sparse["hardware"]["ram_type"] = NR; sparse["runs"][0]["vram_used_peak_mib"] = NR
gaps = rr.not_recorded_fields(sparse)
check("$.hardware.ram_type" in gaps and "$.runs[0].vram_used_peak_mib" in gaps and not rr.validate(sparse, schema), "gaps are listed, and a record with gaps is still valid")

# ---- 6. command line round trip with files
rr.READ, rr.GLOB, rr.RUN = rr._read, rr._glob, rr._run
tmp = tempfile.mkdtemp(prefix="bench record ")
rpath = os.path.join(tmp, "rec.json")
json.dump(rec, open(rpath, "w"))
import io, contextlib
for fmt, want in (("details", "GPU expert layers: 9"), ("row", "Fake-Q2_0.gguf"), ("results", "## N."), ("issue", "<details>"), ("gaps", "")):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = rr.main(["card", rpath, "--as", fmt])
    check(rc == 0 and want in buf.getvalue(), f"card --as {fmt}")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = rr.main(["validate", rpath])
check(rc == 0 and buf.getvalue().startswith("valid: 0 problems"), buf.getvalue().strip())
json.dump(bad, open(rpath, "w"))
with contextlib.redirect_stdout(io.StringIO()):
    check(rr.main(["validate", rpath]) == 1, "validate exits 1 on an invalid record")

# ---- 7. the real probe on this machine: own process, no GPU or model needed
real = rr.probe()
sm = rr.Sampler(os.getpid(), pr=real, interval=0.05, vram_interval=0.2)
sm.start()
t_end = time.time() + 2.0
while time.time() < t_end:
    sum(i * i for i in range(20000))
    time.sleep(0.01)                                      # let the sampler thread run
out = sm.stop()
check(out["samples"] >= 2 and isinstance(out["process_ram_peak_mib"], (int, float)) and out["process_ram_peak_mib"] > 5, f"real probe: this process, RSS {out['process_ram_peak_mib']} MiB, {out['samples']} samples")
check(isinstance(out["system_ram_free_min_mib"], (int, float)) and isinstance(out["cpu_avg_pct"], (int, float)), f"real probe: free RAM and CPU % read (cpu avg {out['cpu_avg_pct']} %)")
model = os.path.join(tmp, "m.gguf")
open(model, "wb").write(b"GGUF" + b"\0" * 4096)
envr = rr.collect_environment(model, None, listing="", version_line="version: 1 (abcdef1)", backend="CPU only")
check(envr["model"]["file"] == "m.gguf" and envr["model"]["sha256_head16MiB"] != NR and envr["software"]["llama_build"] == 1 and envr["software"]["python"], "collect: model file, hash prefix, llama build, python")
check(not [e for e in rr.validate({"hardware": envr["hardware"], "software": envr["software"], "model": envr["model"]},
                                  {"type": "object", "properties": {k: schema["properties"][k] for k in ("software", "model")}, "$defs": schema["$defs"]}) if "required" in e and "plugin" not in e],
      "collect: the Software and Model sections have every required field (values may say not recorded)")

print("bench-record test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
sys.exit(1 if FAILS else 0)
