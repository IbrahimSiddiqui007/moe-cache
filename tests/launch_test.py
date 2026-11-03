#!/usr/bin/env python3
"""moe_cache.py launch (bin/moe_cache_launch.py): the guided start's decisions on a fake machine, with fake pick output and answers piped in.
No model, no llama-server, no network, nothing downloaded or started: Deps is replaced by a fake that records what would have run.
   python3 tests/launch_test.py"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-launch-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["HOME"] = os.environ["USERPROFILE"] = os.path.join(TMP, "home")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_launch as L  # noqa: E402
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

GIB = 2**30
MODEL = os.path.join(TMP, "models", "big.gguf")
os.makedirs(os.path.dirname(MODEL)); open(MODEL, "wb").write(b"x")

def rec(id_, name, size_gb, local=None, ctx=65536, use="agents", measured=None, fits=True, warns=()):
    return {"id": id_, "name": name, "use": use, "size": size_gb * 1e9, "ctx": ctx, "args": ["--ctx", str(ctx)], "engine": "stock llama.cpp (fits in RAM)",
            "measured": measured or "untested on this class of machine", "untested": not measured, "key": [("decode", "42 tok/s")] if measured else [],
            "why": [f"{name} fits"], "warnings": list(warns), "alternatives": [], "memory": {}, "local": local, "fits_disk": fits, "disk_missing": 0 if fits else 30e9,
            "download": None if local or not fits else {"size": size_gb * 1e9, "source": "https://huggingface.co/x/y", "folder": os.path.join(TMP, "dl"),
                                                        "steps": ["hf download x/y --include \"y.gguf\" --local-dir \"" + os.path.join(TMP, "dl") + "\""], "command": "hf download x/y"}}

class FakeProc:
    def __init__(self, cmd, dies=False):
        self.cmd, self.returncode, self.dies, self.waited = cmd, None, dies, False
        if dies: self.returncode = 3
    def poll(self): return self.returncode
    def wait(self, timeout=None): self.waited = True; self.returncode = 0; return 0
    def terminate(self): pass

class Fake(L.Deps):
    def __init__(self, answers=(), llama="/kit/llama/llama-server", local_first=False, agents=("opencode", "claude"), hf=True, dies=False, up_after=2):
        super().__init__()
        self.lines, self.answers, self.asked, self.started, self.connected, self.downloaded = [], list(answers), [], [], [], []
        self.llama_path, self.agents, self.hf, self.dies, self.up_after, self.polls = llama, agents, hf, dies, up_after, 0
        self.recs = [rec("bolt", "Alpha 35B", 22, local=MODEL if local_first else None, measured="decode 42 tok/s"), rec("qwen3.6-35b", "Beta 20B", 12, local=MODEL + ".b"), rec("flash-next-q2_0", "Gamma 8B", 5)]
        self.say = self.lines.append
        self.ask_fn = self._ask
    def _ask(self, q, default=""):
        self.asked.append(q)
        a = self.answers.pop(0) if self.answers else ""
        return a.strip().lower() or default
    def llama(self): return {"path": self.llama_path, "source": "test"}
    def machine(self, dirs, lp):
        return {"os": "test", "ram_total": 32 * GIB, "ram_cap": 28 * GIB, "gpu": "Test GPU", "vram": 8 * GIB, "disk_path": "/models", "disk_free": 300e9,
                "disk_kind": "SSD", "avx2": True, "avx512": False, "llama_build": 12064, "local": {}}
    def recommend(self, spec, use, model=None):
        r = next((x for x in self.recs if model in (None, x["id"])), self.recs[0])
        r = dict(r, use=use); r["alternatives"] = [{"id": "qwen3.6-35b", "name": "Beta 20B", "why": "smaller, already here"}, {"id": "flash-next-q2_0", "name": "Gamma 8B", "why": "smallest"}] if model is None else []
        return r
    def find_local(self, dirs): return {"bolt": MODEL}
    def which(self, n): return ("/bin/" + n) if (n == "hf" and self.hf) or n in self.agents else None
    def download(self, r): self.downloaded.append(r["id"]); return 0
    def start(self, cmd): self.started.append(cmd); self.proc = FakeProc(cmd, self.dies); return self.proc
    def up(self, url, on_demand): self.polls += 1; return self.polls > self.up_after
    def connect(self, argv): self.connected.append(argv)
    def sleep(self, s): pass
    def text(self): return "\n".join(self.lines)

def run(fake, *argv):
    return L.launch(list(argv), fake)

# ---- 1. an interactive walk-through: agents, the 2nd model (already here), start the server, connect opencode but not claude ----
f = Fake(answers=["2", "2", "1", "y", "n"])
rc = run(f)
t = f.text()
check(rc == 0, "interactive walk-through ends 0")
check("RAM:" in t and "Test GPU" in t and "8.0 GiB" in t and "llama-server:  /kit/llama/llama-server" in t, "step 1 shows RAM, GPU/VRAM and the llama-server")
check(t.index("1) Alpha") < t.index("2) Beta") < t.index("3) Gamma"), "step 2 lists up to three models")
check("speed: decode 42 tok/s" in t and t.count("speed: not measured on a PC like this one") == 2, "speed is shown only where it was measured; 'not measured' otherwise")
check("on this PC: no (download 22.0 GB" in t and "on this PC: yes" in t, "each model says whether it is already downloaded")
check(len(f.started) == 1 and f.started[0][2:4] == ["server", MODEL + ".b"] and "--agent" in f.started[0] and "65536" in f.started[0], f"it started the server for the chosen model in agent mode ({f.started[0][2:] if f.started else None})")
check(f.downloaded == [], "nothing was downloaded (the chosen model was here)")
check(f.connected == [["opencode", "--url", "http://127.0.0.1:8080", "--write"]], f"only the agent answered 'y' was connected ({f.connected})")
check(f.proc.waited, "it stays in the foreground until the server ends")
print("\n----- sample transcript (interactive, answers: 2, 2, 1, y, n) -----"); print(t); print("----- end of sample -----\n")

# ---- 2. a model that is not here: no download without an explicit yes ----
f = Fake(answers=["", "1", "", "n"])             # use=chat, model 1 (needs download), answer n
rc = run(f, "--start", "no")
check(rc == 2 and f.downloaded == [] and "Not downloaded" in f.text() and "hf download x/y" in f.text(), "download refused: the exact command is printed, nothing downloaded")
f = Fake(answers=["", "1", "y", "", ""])
rc = run(f)
check(f.downloaded == ["bolt"] and rc == 0 and f.started and f.started[0][3] == MODEL, "download only after 'y', then the downloaded file is started")
f = Fake()
rc = run(f, "--yes", "--use", "agents")
check(rc == 0 and f.downloaded == [] and f.started and f.started[0][3] == MODEL + ".b", "--yes alone never downloads; it takes the model that is already here")
f = Fake()
rc = run(f, "--yes", "--model", "bolt")
check(rc == 2 and f.downloaded == [] and f.started == [] and "--download" in f.text(), "--yes --model <not here>: prints the command, mentions --download, downloads nothing")
f = Fake()
rc = run(f, "--yes", "--model", "bolt", "--download")
check(rc == 0 and f.downloaded == ["bolt"] and f.started, "--download is the explicit yes")
f = Fake(hf=False, answers=["", "1", "y"])
rc = run(f)
check(rc == 2 and f.downloaded == [] and "pip install -U huggingface_hub" in f.text(), "no hf tool: says how to get it, downloads nothing")
f = Fake(answers=["", "1"])
f.recs[0]["fits_disk"] = False; f.recs[0]["download"] = None; f.recs[0]["disk_missing"] = 30e9
rc = run(f)
check(rc == 1 and "too little room" in f.text() and f.started == [], "not enough disk: said plainly, nothing started")

# ---- 3. scripts: flags only ----
f = Fake()
rc = run(f, "--yes", "--model", MODEL)
check(rc == 0 and f.asked == [] and f.started[0][2:] [:2] == ["server", MODEL] and "--agent" not in f.started[0] and "32768" in f.started[0], "--yes --model FILE: no question asked, chat server started")
check(f.connected == [], "...and no agent connected without --agent")
f = Fake()
rc = run(f, "--yes", "--model", MODEL, "--agent", "opencode")
check(rc == 0 and "--agent" in f.started[0] and f.connected == [["opencode", "--url", "http://127.0.0.1:8080", "--write"]], "--agent opencode: agent mode, connects only that one")
f = Fake(agents=("opencode", "claude", "hermes"))
rc = run(f, "--yes", "--model", MODEL, "--agent")
check([c[0] for c in f.connected] == ["hermes", "opencode", "claude"], f"bare --agent: every agent found ({[c[0] for c in f.connected]})")
f = Fake(agents=())
rc = run(f, "--yes", "--model", MODEL, "--agent", "claude")
check(rc == 0 and f.connected == [] and "not installed" in f.text(), "--agent NAME that is not installed: said, skipped")
f = Fake()
rc = run(f, "--yes", "--model", MODEL, "--start", "on-demand", "--port", "9090", "--use", "agents")
check(f.started[0][2:] == ["on-demand", MODEL, "--port", "9090"], f"--start on-demand: the listener on the chosen port ({f.started[0][2:]})")
f = Fake()
rc = run(f, "--yes", "--model", MODEL, "--dry-run")
check(rc == 0 and f.started == [] and "nothing was started" in f.text(), "--dry-run starts nothing")
f = Fake()
rc = run(f, "--yes", "--model", MODEL, "--start", "no")
check(rc == 0 and f.started == [], "--start no: prints the commands only")

# ---- 4. things that go wrong ----
f = Fake(llama=None)
rc = run(f, "--yes", "--model", MODEL)
check(rc == 1 and f.started == [] and "NOT FOUND" in f.text() and "LLAMA_SERVER" in f.text(), "no llama-server: stops at step 1 with the way out")
f = Fake(dies=True)
rc = run(f, "--yes", "--model", MODEL)
check(rc == 1 and "exit code 3" in f.text() and "doctor" in f.text(), "the server dies while loading: exit code and `doctor` are named")
f = Fake()
rc = run(f, "--yes", "--model", os.path.join(TMP, "nope.gguf"))
check(rc == 1 and f.started == [], "--model FILE that does not exist: stops")
f = Fake()
rc = run(f, "--yes", "--model", "no-such-candidate")
check(rc == 1 and f.started == [], "--model NAME that is unknown: stops and lists the names")
f = Fake(answers=[])            # EOF on every question: the defaults
rc = run(f)
check(rc == 2 or rc == 0, "closed input takes the defaults and never downloads")
check(f.downloaded == [], "...no download on EOF")

# ---- 5. the real command line ----
r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "launch", "--help"], capture_output=True, text=True, timeout=60)
check(r.returncode == 0 and "--yes" in r.stdout and "--agent" in r.stdout and "--download" in r.stdout, "`moe_cache.py launch --help` lists the flags")
r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py")], capture_output=True, text=True, timeout=60)
check("launch" in (r.stdout + r.stderr), "the command overview names launch")
env = dict(os.environ, LLAMA_SERVER=os.path.join(TMP, "no-such"))
r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "moe_cache.py"), "launch", "--yes", "--model", MODEL, "--dry-run"], capture_output=True, text=True, timeout=120, env=env, stdin=subprocess.DEVNULL)
check(r.returncode in (0, 1) and "This PC" in r.stdout, f"the real command runs on this machine without crashing (exit {r.returncode})")

print("FAIL" if fails else "PASS", f"({len(fails)} failed)")
sys.exit(1 if fails else 0)
