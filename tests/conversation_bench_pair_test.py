#!/usr/bin/env python3
"""tools/conversation_bench_pair.sh (the Linux A B B A driver) against fake launchers and a fake llama-server (no model, no GPU):
the arms run in the order A B B A, each with its own XDG_CACHE_HOME, launcher root, arguments and environment; one record validates and holds four runs;
the spread table is printed; a stopped driver (SIGTERM mid-run) leaves no server behind; a busy port is refused.
   python3 tests/conversation_bench_pair_test.py"""
import json, os, shutil, signal, socket, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRV = os.path.join(HERE, "tools", "conversation_bench_pair.sh")
FAKE = os.path.join(HERE, "tests", "fake_llama_server.py")
sys.path.insert(0, os.path.join(HERE, "bin"))
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

if os.name == "nt" or not shutil.which("bash") or not shutil.which("ss") or not shutil.which("setsid"):
    print("skipped: needs bash, ss and setsid (Linux)"); sys.exit(0)

tmp = tempfile.mkdtemp()
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
PORT = free_port()

# a fake launcher checkout: bin/moe_cache.py "server MODEL --llama-server LS --ctx N --port P [--tag X]" starts the fake llama-server as a child and records how it was called
def make_root(name):
    root = os.path.join(tmp, name); os.makedirs(os.path.join(root, "bin"))
    open(os.path.join(root, "bin", "moe_cache.py"), "w").write(f'''#!/usr/bin/env python3
import json, os, signal, subprocess, sys, time
a = sys.argv[1:]
port = a[a.index("--port") + 1]; ctx = a[a.index("--ctx") + 1]
with open(os.path.join(os.path.dirname(__file__), "calls.jsonl"), "a") as f:
    f.write(json.dumps({{"root": {name!r}, "argv": a, "xdg": os.environ.get("XDG_CACHE_HOME"), "fake_d": os.environ.get("FAKE_D_TPS")}}) + "\\n")
open(os.path.join(os.environ["XDG_CACHE_HOME"], "tuned-by-" + {name!r}), "w").write("x")     # what a prep start leaves in the cache (the stored auto-tune plan)
ch = subprocess.Popen([sys.executable, {FAKE!r}, port, ctx])
def stop(*_):
    ch.terminate()      # the outer ch.wait() below then returns (waiting again inside the handler would deadlock on Popen's lock)
signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
ch.wait()
''')
    return root

ROOT_A, ROOT_B = make_root("rootA"), make_root("rootB")
model = os.path.join(tmp, "fake.gguf"); open(model, "wb").write(b"not a gguf")
ls = os.path.join(tmp, "llama-server"); open(ls, "w").write("#!/bin/sh\necho 'version: fake'\n"); os.chmod(ls, 0o755)

def alive(port):
    s = socket.socket(); s.settimeout(1)
    try:
        s.connect(("127.0.0.1", port)); return True
    except OSError:
        return False
    finally:
        s.close()

out = os.path.join(tmp, "out")
cmd = ["bash", DRV, "--out", out, "--model", model, "--llama-server", ls, "--ctx", "4096", "--port", str(PORT), "--max-tokens", "32", "--no-evict", "--no-cool",
       "--root-a", ROOT_A, "--root-b", ROOT_B, "--args-a", "--load-mode none", "--args-b", "--n-cpu-moe 24 -- --alias x", "--env-b", "FAKE_D_TPS=20", "--ready-s", "60", "--settle-s", "0"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print(r.stdout[-1500:], r.stderr[-800:])
check(r.returncode == 0, f"the driver finishes ({r.returncode})")
calls = [json.loads(l) for d in (ROOT_A, ROOT_B) for l in open(os.path.join(d, "bin", "calls.jsonl"))]
log = [l for l in open(os.path.join(out, "progress.log")) if "run " in l and ("ready after" in l)]
check(len(log) == 4, "four arms started and became ready")
ca, cb_ = [json.loads(l) for l in open(os.path.join(ROOT_A, "bin", "calls.jsonl"))], [json.loads(l) for l in open(os.path.join(ROOT_B, "bin", "calls.jsonl"))]
check(len(ca) == 2 and len(cb_) == 2, "each arm's own launcher root is used twice")
check("--load-mode" in ca[0]["argv"] and "none" in ca[0]["argv"] and "--n-cpu-moe" in cb_[0]["argv"] and "--alias" in cb_[0]["argv"] and "--load-mode" not in cb_[0]["argv"], "arm arguments go to that arm only")
check(len({c["xdg"] for c in calls}) == 4 and all(c["xdg"].startswith(out) for c in calls), "every arm has its own XDG_CACHE_HOME under the output folder")
check(all(c["fake_d"] is None for c in ca) and all(c["fake_d"] == "20" for c in cb_), "arm environment goes to that arm only")
check(all(c["argv"][0] == "server" and c["argv"][1] == model and "--ctx" in c["argv"] for c in calls), "the launcher is called as: server MODEL --llama-server --ctx --port")
rec = json.load(open(os.path.join(out, "record.json")))
check([x["arm"] for x in rec["runs"]] == ["A", "B", "B", "A"] and rec["protocol"]["order"] == "ABBA", "the record has the runs in A B B A order")
import moe_cache_record as rr
check(rr.validate(rec, rr.load_schema()) == [], "the record validates against the schema")
sa, sb = rec["summary"]["A"]["decode_tps"], rec["summary"]["B"]["decode_tps"]
check(abs(sa - 10.0) < 0.1 and abs(sb - 20.0) < 0.1, f"arm B's environment reaches its server (decode {sa:.1f} vs {sb:.1f} tok/s)")
tables = open(os.path.join(out, "tables.md")).read()
check("Spread between the runs of one arm" in tables and "| A | 2 |" in tables, "tables.md has the spread table")
check(all(os.path.exists(os.path.join(out, f"peaks-{i}-{l}.json")) for i, l in enumerate("ABBA", 1)), "peak files exist per arm")
check(not alive(PORT), "no server is left on the port")

# ---- --prep: each distinct setup starts once alone, its cache is copied to the arms; identical setups share one prep
def calls_of(root):
    f = os.path.join(root, "bin", "calls.jsonl")
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
for root in (ROOT_A, ROOT_B):
    if os.path.exists(os.path.join(root, "bin", "calls.jsonl")):
        os.remove(os.path.join(root, "bin", "calls.jsonl"))
out4 = os.path.join(tmp, "out4")
c4 = cmd[:3] + [out4] + cmd[4:] + ["--prep", "--order", "A,B"]
r = subprocess.run(c4, capture_output=True, text=True, timeout=200)
check(r.returncode == 0, f"--prep: the driver finishes ({r.returncode})")
check(len(calls_of(ROOT_A)) == 2 and len(calls_of(ROOT_B)) == 2, "--prep: each distinct setup starts twice (once alone to tune, once as the arm)")
check(os.path.exists(os.path.join(out4, "cache-1-A", "tuned-by-rootA")) and os.path.exists(os.path.join(out4, "cache-2-B", "tuned-by-rootB")), "--prep: each arm's cache holds what its prep start stored")
check(os.path.exists(os.path.join(out4, "server-prep-A.log")) and not alive(PORT), "--prep: the prep server is stopped before the arm starts")
for root in (ROOT_A, ROOT_B):
    os.remove(os.path.join(root, "bin", "calls.jsonl")) if os.path.exists(os.path.join(root, "bin", "calls.jsonl")) else None
out5 = os.path.join(tmp, "out5")
same = ["bash", DRV, "--out", out5, "--model", model, "--llama-server", ls, "--ctx", "4096", "--port", str(PORT), "--max-tokens", "32", "--no-evict", "--no-cool",
        "--root-a", ROOT_A, "--root-b", ROOT_A, "--ready-s", "60", "--settle-s", "0", "--prep", "--order", "A,B,B,A"]
r = subprocess.run(same, capture_output=True, text=True, timeout=250)
check(r.returncode == 0 and len(calls_of(ROOT_A)) == 5, f"--prep: two arms with the same setup share one prep start (5 starts: 1 prep + 4 runs, got {len(calls_of(ROOT_A))})")
rec5 = json.load(open(os.path.join(out5, "record.json")))
check(rec5["protocol"]["order"] == "ABBA", "A = B (the noise-floor case) gives an ABBA record too")

# ---- a busy port is refused (nothing started)
busy = socket.socket(); busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); busy.bind(("127.0.0.1", PORT)); busy.listen(1)
r = subprocess.run(cmd[:3] + [os.path.join(tmp, "out2")] + cmd[4:], capture_output=True, text=True, timeout=60)
busy.close()
check(r.returncode != 0 and "in use" in r.stdout, "a busy port stops the driver before anything starts")

# ---- stopped mid-run (SIGTERM): the trap stops the arm's whole process group
out3 = os.path.join(tmp, "out3")
c3 = cmd[:3] + [out3] + cmd[4:]
p = subprocess.Popen(c3, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
t0 = time.time()
while time.time() - t0 < 60 and not alive(PORT):
    time.sleep(0.3)
check(alive(PORT), "(stop test) the first arm's server is up")
os.killpg(p.pid, signal.SIGTERM)
try:
    p.wait(60)
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGKILL); p.wait()
time.sleep(1)
check(not alive(PORT), "SIGTERM mid-run leaves no server on the port")
ps = subprocess.run(["pgrep", "-f", FAKE], capture_output=True, text=True).stdout.split()
check(not ps, f"no fake llama-server process is left ({ps})")

# ---- SIGTERM to the driver's own pid only (not its group), while it waits before the conversation: the arm is stopped at once, not after the conversation
out6 = os.path.join(tmp, "out6")
c6 = cmd[:3] + [out6] + cmd[4:-2] + ["--settle-s", "60"]
p = subprocess.Popen(c6, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
t0 = time.time()
while time.time() - t0 < 60 and not alive(PORT):
    time.sleep(0.3)
time.sleep(1)
os.kill(p.pid, signal.SIGTERM)
try:
    p.wait(30); ended = True
except subprocess.TimeoutExpired:
    ended = False; os.killpg(p.pid, signal.SIGKILL); p.wait()
time.sleep(1)
check(ended and not alive(PORT), "SIGTERM to the driver pid alone stops the arm at once (the trap runs while it waits)")
ps = subprocess.run(["pgrep", "-f", FAKE], capture_output=True, text=True).stdout.split()
check(not ps, f"(pid-only stop) no fake llama-server process is left ({ps})")

# ---- an --env word without = is refused
r = subprocess.run(cmd[:3] + [os.path.join(tmp, "out7")] + cmd[4:-2] + ["--env-a", "FOO"], capture_output=True, text=True, timeout=60)
check(r.returncode == 2 and "K=V" in r.stderr and not alive(PORT), "--env-a FOO (no =) is refused and nothing is left running")
rec_t = open(os.path.join(out, "tables.md")).read()
check("against A" in rec_t and "rough spread" in rec_t and "noise floor" not in rec_t, "tables.md says how to read the B-against-A line")
check(any("one machine, forecast" in n and "(machine label: " in n and "R16" not in n for n in rec["notes"]), "the record carries the machine / forecast label, no rule ids")
h = subprocess.run(["bash", DRV, "--help"], capture_output=True, text=True).stdout
check("--machine" in h and "set -u" not in h and "HERE=" not in h and "cool_run" not in h, "--help prints the header only")
r = subprocess.run(cmd[:3] + [os.path.join(tmp, "out8")] + cmd[4:] + ["--machine", "box9", "--env-b", "BAD"], capture_output=True, text=True, timeout=60)
check(r.returncode == 2 and not os.path.exists(os.path.join(tmp, "out8", "1-A.json")), "a bad --env-b is refused before any arm starts")

shutil.rmtree(tmp, ignore_errors=True)
print("FAIL: %d" % len(fails) if fails else "ok")
sys.exit(1 if fails else 0)
