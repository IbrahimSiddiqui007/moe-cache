#!/usr/bin/env python3
"""Stage 0 of the IQ3_XXS speed plan: the benchmark record fills what the server command line, the server log (stdout and stderr), the model file and the kit
already say, and keeps "not recorded" only where the fact is truly absent. No model, no server, no GPU (a tiny GGUF header is written here).
   python3 tests/stage0_records_test.py"""
import json, os, re, signal, struct, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
sys.path.insert(0, os.path.join(HERE, "tools"))
import moe_cache_record as rr  # noqa: E402
import conversation_bench as cb  # noqa: E402

NR = rr.NR
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

tmp = tempfile.mkdtemp(prefix="stage0-")

# ---- a tiny GGUF: block_count, optional file_type, one expert tensor of a given ggml type
def s_(x):
    b = x.encode(); return struct.pack("<Q", len(b)) + b
def kv_u32(k, v): return s_(k) + struct.pack("<II", 4, v)
def kv_str(k, v): return s_(k) + struct.pack("<I", 8) + s_(v)
def write_gguf(name, block_count, ftype, ggml_type):
    kvs = [kv_str("general.architecture", "qwen3moe"), kv_u32("qwen3moe.block_count", block_count), kv_u32("qwen3moe.expert_count", 8)]
    if ftype is not None:
        kvs.append(kv_u32("general.file_type", ftype))
    t = s_("blk.0.ffn_up_exps.weight") + struct.pack("<I", 3) + struct.pack("<3Q", 128, 4, 8) + struct.pack("<IQ", ggml_type, 0)
    h = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs) + t
    p = os.path.join(tmp, name)
    with open(p, "wb") as f:
        f.write(h + b"\0" * (4096 + 4096))
    return p

# ---- 1. the server logs of v0.3.7 night 11-06 (llama.cpp b12064 prints almost nothing at the default verbosity)
stock_err = ["0.00.091.602 I srv  llama_server: initializing ...", "0.12.691.866 I cmn          init: llama threadpool init, n_threads = 16",
             "0.14.119.499 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 65536, kv_unified = 'true'"]
mc_out = ["moe-cache: profile C:\\x.usage loaded (48 layers x 512 experts)", "moe-cache: experts of 42 layers stay on the CPU and are managed by the cache; plugin loaded by path"]
lg = rr.parse_llama_log(stock_err + mc_out)
check(lg["threads"] == 16 and lg["plugin_layers"] == 48 and lg["plugin_cpu_expert_layers"] == 42, f"log: threads 16 from the threadpool line, 48 layers and 42 CPU expert layers from the plugin ({lg['threads']}, {lg['plugin_layers']}, {lg['plugin_cpu_expert_layers']})")
check(rr.parse_llama_log([])["threads"] is None, "an empty log gives no threads")

cmd_stock = r"llama-server -m D:\models\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -ngl 99 --n-cpu-moe 42 -c 65536 -fa on --load-mode mmap --port 8099 -ub 2048 -b 2048 --cache-ram 2048"
st = rr.settings_from(cmd_stock, env={}, log_lines=stock_err)
check(st["threads"] == 16 and st["fact_sources"]["threads"].startswith("server log"), "threads come from the server log when the command line has no -t")
check(st["kv_type_k"].startswith("f16") and st["kv_type_v"].startswith("f16") and "default" in st["fact_sources"]["kv_type_k"], "no -ctk / -ctv on a real command line: the llama.cpp default f16, and the record says it is the default")
check(st["n_layer"] == NR and st["gpu_expert_layers"] == NR, "stock log without layer facts and no model file here: n_layer and GPU expert layers stay 'not recorded'")
st2 = rr.settings_from(cmd_stock, env={}, log_lines=stock_err + mc_out)
check(st2["n_layer"] == 48 and st2["gpu_expert_layers"] == 6 and st2["fact_sources"]["gpu_expert_layers"] == "n_layer - n_cpu_moe", "with the plugin's layer line: n_layer 48, GPU expert layers 48 - 42 = 6")
g_iq3 = write_gguf("m-iq3.gguf", 48, 23, 18)
st3 = rr.settings_from(cmd_stock, env={}, log_lines=stock_err, model=g_iq3)
check(st3["n_layer"] == 48 and st3["gpu_expert_layers"] == 6 and "GGUF header" in st3["fact_sources"]["n_layer"], "stock arm: n_layer from the model's own GGUF block_count (the log has none)")
st4 = rr.settings_from(cmd_stock.replace("-ub 2048", "-ub 2048 -ctk q8_0 -ctv q4_0 -t 8"), env={}, log_lines=stock_err)
check(st4["kv_type_k"] == "q8_0" and st4["kv_type_v"] == "q4_0" and st4["threads"] == 8 and "kv_type_k" not in st4.get("fact_sources", {}), "explicit -ctk / -ctv / -t win over the defaults and the log")
st5 = rr.settings_from("", env={}, log_lines=[])
check(st5["kv_type_k"] == NR and st5["threads"] == NR and st5["n_layer"] == NR, "no command line and no log: everything stays 'not recorded' (nothing guessed)")
check(rr.parse_cmdline(r'llama-server -m "C:\my models\x.gguf" -c 8')["model"] == r"C:\my models\x.gguf", "the -m path is read, also with spaces in quotes")

# ---- 2. model quant for every file, Q2_0 included
check(rr._file_quant(g_iq3) == "IQ3_XXS", "file_type 23 names IQ3_XXS")
g_q2 = write_gguf("m-q2.gguf", 48, None, 42)
check(rr._file_quant(g_q2) == "Q2_0", "a Q2_0 file has no file_type name: the type of its expert tensors names it (Q2_0)")
check(rr._file_quant(os.path.join(tmp, "nope.gguf")) is None, "an unreadable file gives no quant (not recorded)")
env = rr.collect_environment(g_q2, None, listing="", version_line="version: 12064 (51fe42c5d)", backend="Vulkan")
check(env["model"]["quant"] == "Q2_0", f"software/model section: quant {env['model']['quant']}")

# ---- 3. kit commit without git: REVISION, else README_WINDOWS.txt
kit = os.path.join(tmp, "kit"); os.makedirs(kit)
check(rr.git_commit(kit) is None, "a folder with neither git nor a revision file: no commit")
open(os.path.join(kit, "README_WINDOWS.txt"), "w").write("moe-cache v0.3.7 for Windows (x64), built from commit f928d1c. Experimental.\r\n")
check(rr.git_commit(kit) == "f928d1c", "kit README_WINDOWS.txt names the commit it was built from")
open(os.path.join(kit, "REVISION"), "w").write("0123abc\n")
check(rr.git_commit(kit) == "0123abc", "a REVISION file wins over the README line")
env2 = rr.collect_environment(None, None, listing="", version_line="", backend="Vulkan", root=kit)
check(env2["software"]["moe_cache_commit"] == "0123abc", "collect_environment records the kit's commit (root = the kit folder)")
mk = open(os.path.join(HERE, "tools", "make_windows_kit.sh")).read()
check('> "$K/REVISION"' in mk, "make_windows_kit.sh writes the REVISION file into the kit")

# ---- 4. disk reads, stock arm on Windows
check(cb.parse_drive_raw("123456789012\r\n") == 123456789012 and cb.parse_drive_raw("") is None and cb.parse_drive_raw("error") is None, "the drive raw counter text is read (a number or nothing)")
check(cb.model_drive(r"llama-server -m D:\models\x.gguf -c 8") == "D" and cb.model_drive(r'env A=1 llama-server -m "e:/m/x.gguf"') == "E" and cb.model_drive("llama-server -m /m/x.gguf") is None, "the model's drive letter comes from -m")
f = cb.disk_read_facts([], True, 100, 5_000_000_100, 10, 90_000_000_010, "D")
check(f["disk_read_gb"] == 90.0 and "PhysicalDisk" in f["disk_read_source"] and "D:" in f["disk_read_source"] and f["disk_read_process_gb"] == 5.0, "Windows stock arm: the model drive's counter is the disk read, the process counter is kept next to it")
f = cb.disk_read_facts([], True, 100, 5_000_000_100, None, None, "D")
check(f["disk_read_gb"] == 5.0 and "lower bound" in f["disk_read_source"], "Windows without the drive counter: the process counter, labelled as a lower bound with mmap")
f = cb.disk_read_facts([2.0, 2.0], True, 100, 5_000_000_100, 10, 90_000_000_010, "D")
check(f["disk_read_gb"] == 4.0 and "plugin" in f["disk_read_source"], "the plugin's own counter still wins for the plugin arm")
f = cb.disk_read_facts([], False, 0, 3_000_000_000, None, None, None)
check(f["disk_read_gb"] == 3.0 and "/proc" in f["disk_read_source"], "Linux: /proc/<pid>/io")
check(cb.disk_read_facts([], True, None, None, None, None, "D") == {}, "nothing readable: no field (the record says not recorded)")
b = cb.proc_read_bytes(os.getpid())
check(b is None or b >= 0, "proc_read_bytes reads this process (a non-negative integer, or None where the OS refuses)")
if os.name != "nt":
    check(isinstance(b, int), "Linux: /proc/<pid>/io is readable for our own process")

# ---- 5. the plugin's stats line carries the new counters (source check: the C++ side needs ggml headers and a model to run)
snap = open(os.path.join(HERE, "src", "mc_state_io.inl")).read()
check(all(k in snap for k in ('\\"gpu_up_saved\\":', '\\"ns_ssd_wait\\":', '\\"n_ssd_wait\\":')), "moe-cache-stats prints gpu_up_saved, ns_ssd_wait and n_ssd_wait")
cnt = open(os.path.join(HERE, "src", "mc_state_counters.h")).read()
check("std::atomic<uint64_t>" in cnt and re.search(r"ns_ssd_wait\{0\}", cnt) and re.search(r"n_ssd_wait\{0\}", cnt), "the SSD-wait counters are plain atomics")
be = open(os.path.join(HERE, "src", "mc_backend.inl")).read()
body = be[be.index("static void wait_experts("):]
body = body[:body.index("\n}\n")]
check("ns_ssd_wait +=" in body and "lock_guard" not in body and "mutex" not in body and "unique_lock" not in body, "wait_experts counts the wait without a lock")
check(body.index("steady_clock::now()") > body.index("if (__atomic_load_n(f"), "the clock is read only after the flag was seen set (a flag already clear costs one atomic load, as before)")



print("stage 0 records test: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
