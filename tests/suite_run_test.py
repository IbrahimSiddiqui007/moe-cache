#!/usr/bin/env python3
"""TEST-1142: bin/moe_cache_suite_run.py against a fake llama-server and a fake llama-perplexity: the speed probe, the task sections, the hard section, the KL section, a whole `quick` run into a private record,
the conversation converter, the CLI entry (`moe_cache.py bench-suite`), and that records are git-ignored (benchmarks/suites/).
   python3 tests/suite_run_test.py"""
import http.server, json, os, subprocess, sys, tempfile, threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_suite as S, moe_cache_suite_run as R      # noqa: E402
HARD = {t["prompt"]: t for t in json.load(open(os.path.join(HERE, "tools", "hard_tasks.json"), encoding="utf-8"))}
fails = 0


def chk(name, cond):
    global fails
    print(("  ok   " if cond else "  FAIL ") + name)
    fails += 0 if cond else 1


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self): self.send({"model_path": "fake.gguf", "build_info": "fake", "status": "ok"})

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template": return self.send({"prompt": b["messages"][0]["content"]})
        if self.path == "/tokenize": return self.send({"tokens": list(range(len(b["content"]) // 4))})
        p = b["prompt"]
        if p in HARD: return self.send({"content": "x</think>\n\nANSWER: " + HARD[p]["check"]["answer"], "tokens": [1, 2, 3, 4], "stop_type": "eos"})
        self.send({"content": "391", "tokens": [1, 2], "stop_type": "eos", "timings": {"prompt_per_second": 321.0, "predicted_per_second": 12.5, "prompt_ms": 80.0, "predicted_ms": 20.0}})


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}"; tmp = tempfile.mkdtemp()

sp = R.run_speed_probe(url, 4096)
chk("speed probe: one band at the asked depth with the server's tok/s and a TTFT", sp["bands"] == [{"depth": 4096, "prompt_tps": 321.0, "decode_tps": 12.5}] and sp["ttft_ms"] == 100 and sp["decode_b1"] == 12.5)
chk("speed probe protocol hash depends on the depth", R.run_speed_probe(url, 4096)["protocol"] == sp["protocol"] and R.run_speed_probe(url, 8192)["protocol"] != sp["protocol"])
t = R.run_tasks(url, "off", 5, tmp)
chk("tasks section: 5 tasks run, the fake answers 391 so one question about 391 passes at most, think=off", t["n"] == 5 and t["think"] == "off" and t["cap"] == 64 and 0 <= t["pass"] <= 5)
h = R.run_hard(url, 0, 2, tmp)
chk("hard section: 27 tasks, all right with the fake, no loops, no cap hits", h["n"] == 27 and h["pass"] == 27 and h["loops"] == 0 and h["cap_hits"] == 0 and h["cap"] == 0)
# fake perplexity: prints KL lines for the second (--kl-divergence) call
fp = os.path.join(tmp, "fakeppl"); open(fp, "w").write("#!/bin/sh\ncase \"$*\" in *--kl-divergence-base*) echo 'Mean    KLD:   0.0123 +- 0.001'; echo 'Median  KLD:   0.0010'; echo '90.0%   KLD:   0.0300'; echo 'Same top p: 96.500 +- 0.1 %';; esac\n"); os.chmod(fp, 0o755)
q = R.run_quality("model.gguf", ["-ctk", "q8_0"], 2, tmp, fp, windows=None, ref_label="same model f16 KV")
chk("quality section from the fake llama-perplexity", q["kl_mean"] == 0.0123 and q["top1_pct"] == 96.5 and q["windows"] == 2 and q["ref"] == "same model f16 KV")
# conversation converter
conv = {"bands": [{"band": "0k-16k", "lo": 0, "hi": 16384, "prompt_tps": 300.0, "decode_tps": 20.0, "ttft_median_s": 1.5}, {"band": "16k-32k", "lo": 16384, "hi": 32768, "prompt_tps": 280.0, "decode_tps": 18.0, "ttft_median_s": 2.0}],
        "process_ram_peak_mib": 20480, "vram_used_peak_mib": 7168, "settings": {"seed": 1}}
spd, mem = R.sections_from_conversation(conv, 32768)
chk("conversation converter: bands by depth, TTFT of the first band, RSS and VRAM in GB", [b["depth"] for b in spd["bands"]] == [16384, 32768] and spd["ttft_ms"] == 1500 and mem["rss_gb"] == 20.0 and mem["vram_gb"] == 7.0)
# a whole quick run against the fake server (no launch)
rec = R.measure(url, "fake-model", "Q3", "quick", None, (), tmp, 1, fp, progress=lambda m: None)
chk("quick measure: speed + tasks sections, valid, private, a note that quality was skipped is absent for quick", S.validate_record(rec) == [] and set(rec["sections"]) == {"speed", "tasks"} and rec["published"] is False)
out = tempfile.mkdtemp(); r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "bench-suite", "run", "--url", url, "--name", "fake-model", "--quant", "Q3", "--profile", "quick", "--out", out], capture_output=True, text=True)
files = os.listdir(out)
chk("`moe_cache.py bench-suite run --url` writes one valid private record", r.returncode == 0 and len(files) == 1 and S.validate_record(json.load(open(os.path.join(out, files[0])))) == [])
r2 = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "bench-suite", "compare", os.path.join(out, files[0]), os.path.join(out, files[0])], capture_output=True, text=True)
chk("`moe_cache.py bench-suite compare` works", r2.returncode == 0 and "decode at 8k" in r2.stdout)
chk("benchmarks/suites/ is git-ignored (private by default)", "benchmarks/suites/" in open(os.path.join(HERE, ".gitignore")).read())
chk("profiles: quick has no KL and no hard; night has the hard set", R.PROFILES["quick"]["kl_chunks"] == 0 and not R.PROFILES["quick"]["hard"] and R.PROFILES["night"]["hard"])
print("suite_run_test: " + ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)
