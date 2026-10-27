#!/usr/bin/env python3
"""moe_cache.py on-demand: an always-on agent endpoint that starts the model only when an agent asks for it (the way Ollama works).

A small listener (standard library, a few MB of RAM) holds the agent port (default 8080). The first request that needs the model starts the
launcher in agent mode (`moe_cache.py server MODEL --agent ...`) on a private port, waits until it is ready and passes requests through, streaming
included. After --idle minutes (default 15) without a request the server is stopped and the RAM is free again; the next request starts it again.
The API key check (the stable per-user key, `moe_cache.py key`) and the localhost-only CORS rule sit in front, so a web page cannot wake the model.
Model listing (/v1/models, /props) is answered without starting the model, so agents can probe the endpoint for free.

  moe_cache.py on-demand MODEL.gguf [launcher options] [-- llama-server args]   remember this model and its options, then listen
  moe_cache.py on-demand                                                         listen with the remembered choice
  moe_cache.py on-demand MODEL2.gguf [launcher options] --add                    a second model (Ollama-style: the request's "model" picks
                                                                                 it; one runs at a time, the other is stopped first)
  moe_cache.py on-demand --remove NAME                                           forget an added model (NAME as /v1/models lists it)
  moe_cache.py on-demand [MODEL ...] --install-autostart                         start it at login (systemd --user unit / Windows Startup folder)
  moe_cache.py on-demand --remove-autostart | --stop | --status
"""
import argparse, hmac, http.client, json, os, re, signal, socket, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_cache as mc  # noqa: E402

WINDOWS = mc.WINDOWS
# server-type probes that llama-server answers with 404: answered the same while the model is stopped, so a probe does not load it
NOT_FOUND_PROBES = ("/api/v1/models", "/api/tags", "/v1/props", "/version")
CHOICE_FILE = os.path.join(mc.CACHE_DIR, "on-demand.json")
LOG_FILE = os.path.join(mc.CACHE_DIR, "on-demand.log")       # the launcher's and llama-server's output
PID_FILE = os.path.join(mc.CACHE_DIR, "on-demand.pid")
LISTENER_LOG = os.path.join(mc.CACHE_DIR, "on-demand-listener.log")   # the listener's own lines when it has no console (Windows)
LAUNCHER = [mc.python_cmd(), os.path.join(mc.HERE, "bin", "moe_cache.py"), "server"]     # tests replace it with a fake server
UNIT = "moe-cache-on-demand.service"
LINK = "moe-cache on-demand.lnk"
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "host",
       "content-length"}
OWN = ("--port", "--agent", "--dry-run", "--open")      # options the listener sets itself


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " moe-cache on-demand: " + msg, flush=True)


def load_choice():
    try:
        return json.load(open(CHOICE_FILE, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_choice(c):
    os.makedirs(mc.CACHE_DIR, mode=0o700, exist_ok=True)
    with open(CHOICE_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)
    os.replace(CHOICE_FILE + ".tmp", CHOICE_FILE)


def model_alias(model, args):
    """The model name the launcher gives llama-server in agent mode (its --alias, else the file name without .gguf and the shard suffix)."""
    extra = args[args.index("--") + 1:] if "--" in args else []
    return mc.arg_after(extra, "-a", "--alias") or re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", os.path.basename(model))


def model_ctx(args):
    v = mc.arg_after(args[:args.index("--")] if "--" in args else args, "--ctx")
    try:
        return int(v)
    except (TypeError, ValueError):
        return 65536            # the launcher's agent-mode default


def local_origin(origin):
    """True for pages served from this machine (http(s)://localhost, 127.x, [::1]): the same rule as llama-server's `--cors-origins localhost`."""
    m = re.match(r"^https?://(\[[^\]]+\]|[^:/]+)(:\d+)?/?$", origin or "")
    return bool(m) and (m.group(1) in ("localhost", "[::1]") or m.group(1).startswith("127."))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Backend:
    """The real server: started on demand, stopped when idle. One at a time; requests that come while it starts wait for it."""

    def __init__(self, choice, launcher=None, idle_min=15.0, start_timeout=900, log=log):
        self.c, self.launcher, self.idle, self.start_timeout, self.log = choice, launcher or LAUNCHER, idle_min * 60, start_timeout, log
        self.key = mc.stored_key(create=True)
        self.lock = threading.Lock()             # held while starting / stopping
        self.count = threading.Lock()
        self.p, self.port, self.state, self.active, self.last, self.t0 = None, None, "idle", 0, time.time(), 0.0
        self.profile = None
        self.alias = model_alias(choice["model"], choice.get("args", []))
        self.ctx = model_ctx(choice.get("args", []))
        self.seen = choice.get("seen") or {}     # the real /v1/models and /props answers, kept from the last start

    def cmd(self, port):
        return self.launcher + [self.c["model"], "--agent", "--port", str(port)] + list(self.c.get("args", []))

    def busy(self, delta):
        with self.count:
            self.active += delta
            self.last = time.time()

    def running(self):
        return self.p is not None and self.p.poll() is None

    def ensure(self, why):
        """The port of a ready server; starts one first when needed. RuntimeError when it cannot start."""
        with self.lock:
            if self.running() and self.state == "ready":
                return self.port
            if self.running():                   # starting (another request is not possible here: the lock) or stopping
                self._stop_locked("restart")
            self._start_locked(why)
            return self.port

    def _start_locked(self, why):
        port = free_port()
        self.state, self.t0, self.port, self.profile = "starting", time.time(), port, None
        self.log(f"starting {os.path.basename(self.c['model'])} (first request: {why}; idle stop after {self.idle / 60:g} min)")
        got, ready = {}, threading.Event()
        # Linux sends the parent-death signal when the THREAD that started the child ends: a thread that lives as long as the server starts it
        threading.Thread(target=self._life, args=(port, got, ready), daemon=True).start()
        ready.wait(60)
        p = got.get("p")
        if p is None:
            self.state = "idle"
            raise RuntimeError(f"could not start the launcher: {got.get('error', 'timeout')}")
        while time.time() - self.t0 < self.start_timeout:
            if p.poll() is not None:
                self.state = "idle"
                self.log(f"the server did not start (exit code {p.returncode}); its output is in {LOG_FILE}")
                raise RuntimeError(f"the model server did not start (exit code {p.returncode}); see {LOG_FILE}")
            if mc.healthy(port, self.key):
                break
            time.sleep(0.5)
        else:
            self._stop_locked("start timeout")
            raise RuntimeError(f"the model server was not ready after {self.start_timeout} s; see {LOG_FILE}")
        self.state = "ready"
        self.busy(0)
        self.log(f"ready on 127.0.0.1:{port} after {time.time() - self.t0:.0f} s")
        self._remember(port)

    def _remember(self, port):
        """Keep the real model list and context, so the endpoint answers them exactly while the model is stopped."""
        seen = {}
        for path in ("/v1/models", "/props"):
            st, body = mc.http(f"http://127.0.0.1:{port}{path}", api_key=self.key, timeout=5)
            if st == 200:
                try:
                    seen[path] = json.loads(body)
                except ValueError:
                    pass
        if seen and seen != self.seen:
            self.seen = self.c["seen"] = seen
            try:
                c = load_choice()
                for e in ([c] + list(c.get("extra") or [])) if c else []:
                    if e.get("model") == self.c["model"] and e.get("args") == self.c.get("args"):
                        e["seen"] = seen
                        save_choice(c)
                        break
            except OSError:
                pass

    def _life(self, port, got, ready):
        env = dict(os.environ)
        env.update(self.c.get("env") or {})
        env["LLAMA_API_KEY"] = self.key          # the same key the listener checks
        kw = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "env": env, "stdin": subprocess.DEVNULL}
        if sys.platform.startswith("linux"):
            kw["preexec_fn"] = mc._die_with_parent_linux
        if WINDOWS:
            kw["creationflags"] = 0x08000000     # CREATE_NO_WINDOW: no console window pops up when started from pythonw at login
        try:
            p = got["p"] = subprocess.Popen(self.cmd(port), **kw)
        except Exception as e:
            got["error"] = str(e)
            ready.set()
            return
        self.p = p
        if WINDOWS:
            try:
                mc._kill_with_parent_windows(p)  # the listener ends -> the launcher and llama-server end
            except Exception:
                pass
        ready.set()
        try:
            if os.path.getsize(LOG_FILE) > 20 * 2**20:
                os.replace(LOG_FILE, LOG_FILE + ".1")
        except OSError:
            pass
        with open(LOG_FILE, "ab") as f:
            f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(self.cmd(port)[2:])}\n".encode())
            for raw in iter(p.stdout.readline, b""):
                line = raw.replace(self.key.encode(), b"KEY")
                f.write(line); f.flush()
                m = re.search(rb"profile (\S+\.usage)", line)
                if m:
                    self.profile = m.group(1).rstrip(b";").decode(errors="replace")
        rc = p.wait()
        if p is self.p and self.state == "ready":       # it ended by itself (crash, out of memory): the next request starts it again
            self.state = "idle"
            self.log(f"the server ended by itself (exit code {rc}); the next request starts it again. Output: {LOG_FILE}")

    def _stop_locked(self, reason):
        p = self.p
        if p is None or p.poll() is not None:
            self.state = "idle"
            return
        self.state = "stopping"
        if WINDOWS:
            # a server on Windows is killed, so the plugin's save at exit never runs: ask for the usage profile first (as the launcher does)
            try:
                from moe_cache_ui import child_pid
                kid = child_pid(p.pid)
                if kid and self.profile:
                    mc.win_request_save(kid, self.profile, timeout=30)
            except Exception:
                pass
            p.terminate()
        else:
            p.send_signal(signal.SIGTERM)        # the launcher passes it on and waits for llama-server
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
        self.state = "idle"
        self.log(f"stopped ({reason}) after {(time.time() - self.t0) / 60:.1f} min; the memory is free until the next request")

    def stop(self, reason):
        with self.lock:
            self._stop_locked(reason)

    def watch(self, every=5):
        """Stop the server after `idle` seconds without a request (a request that streams for an hour counts as busy all along)."""
        while True:
            time.sleep(every)
            if self.state == "ready" and self.active == 0 and time.time() - self.last > self.idle and self.lock.acquire(blocking=False):
                try:
                    if self.state == "ready" and self.active == 0 and time.time() - self.last > self.idle:
                        self._stop_locked(f"{self.idle / 60:g} min without a request")
                finally:
                    self.lock.release()

    def models(self):
        return self.seen.get("/v1/models") or {"object": "list", "data": [{"id": self.alias, "object": "model", "created": 0, "owned_by": "moe-cache"}]}

    def props(self):
        p = dict(self.seen.get("/props") or {"default_generation_settings": {"n_ctx": self.ctx}, "model_alias": self.alias})
        p["on_demand"] = self.state
        return p


class Pool:
    """The models on-demand serves (N9; the first is the default). One runs at a time: a request for another model waits until the running one
    has no request in flight, stops it, then starts the one asked for. The request's "model" field picks; a name it does not know goes to the
    running model, else the default (as with one model: llama-server itself ignores the field)."""

    def __init__(self, backends, drain_timeout=600):
        self.bs, self.drain = list(backends), drain_timeout
        self.key = self.bs[0].key
        self.lock = threading.Lock()

    @property
    def state(self):
        r = self.running()
        return r.state if r else "idle"

    def running(self):
        return next((b for b in self.bs if b.running()), None)

    def pick(self, name):
        if name:
            n = re.sub(r"\.gguf$", "", str(name))
            for b in self.bs:
                ids = [m.get("id") for m in (b.seen.get("/v1/models") or {}).get("data", []) if isinstance(m, dict)]
                if n == b.alias or n in ids:
                    return b
        return self.running() or self.bs[0]

    def ensure(self, b, why):
        """The port of b, ready; another running model is stopped first (after its requests end). Counts the request as busy on b (the caller
        ends it with b.busy(-1)), inside the lock, so a swap waiting for b to drain sees it."""
        with self.lock:
            for o in self.bs:
                if o is not b and o.running():
                    t = time.time()
                    while o.active > 0 and time.time() - t < self.drain:
                        time.sleep(0.2)
                    if o.active > 0:
                        raise RuntimeError(f"{o.alias} is still answering another request; try again when it is done")
                    o.log(f"switching from {o.alias} to {b.alias}")
                    o.stop(f"switching to {b.alias}")
            b.busy(+1)
            try:
                return b.ensure(why)
            except Exception:
                b.busy(-1)
                raise

    def models(self):
        if len(self.bs) == 1:
            return self.bs[0].models()
        data, seen = [], set()
        for b in self.bs:
            for m in b.models().get("data", []):
                if m.get("id") not in seen:
                    seen.add(m.get("id")); data.append(m)
        return {"object": "list", "data": data}

    def stop(self, reason):
        for b in self.bs:
            b.stop(reason)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "moe-cache-on-demand"

    def log_message(self, *a):
        pass

    def cors(self):
        o = self.headers.get("Origin")
        if o and local_origin(o):
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")

    def reply(self, code, obj, ctype="application/json"):
        body = (obj if isinstance(obj, str) else json.dumps(obj)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def authed(self):
        h = self.headers.get("Authorization") or ""
        got = h[7:].strip() if h[:7].lower() == "bearer " else (self.headers.get("X-Api-Key") or "")
        return hmac.compare_digest(got.encode(), self.server.pool.key.encode())

    def body(self):
        n = self.headers.get("Content-Length")
        if n is not None:
            return self.rfile.read(int(n))
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return out
                out += self.rfile.read(size)
                self.rfile.readline()
        return None

    def do_OPTIONS(self):
        self.send_response(204)
        o = self.headers.get("Origin")
        if o and local_origin(o):
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", self.headers.get("Access-Control-Request-Headers") or "Authorization, Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def handle_any(self):
        pool = self.server.pool
        path = self.path.split("?")[0]
        data = self.body()
        if path in ("/health", "/v1/health"):        # public, as in llama-server; never starts the model
            r = pool.running()
            return self.reply(200, dict({"status": "ok", "on_demand": pool.state}, **({"model": r.alias} if r and len(pool.bs) > 1 else {})))
        if not self.authed():
            if path == "/" and self.command == "GET":
                return self.reply(200, f"moe-cache on-demand: {', '.join(b.alias for b in pool.bs)}, {pool.state}. Agents: base URL "
                                       f"http://{self.headers.get('Host')}/v1 with the API key from `moe_cache.py key`.\n", "text/plain; charset=utf-8")
            return self.reply(401, {"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}})
        name = None
        if data and self.command == "POST":
            try:
                name = json.loads(data).get("model")
            except (ValueError, AttributeError, UnicodeDecodeError):
                pass
        b = pool.pick(name)
        if self.command == "GET" and (pool.state != "ready" or len(pool.bs) > 1):
            if path in ("/v1/models", "/models"):        # with several models: always all of them, not only the running one
                return self.reply(200, pool.models())
            if pool.state != "ready":
                if path == "/props":
                    return self.reply(200, b.props())
                if path in NOT_FOUND_PROBES:     # Hermes asks these once to tell LM Studio / Ollama / llama.cpp apart; llama-server's own answer is 404
                    return self.reply(404, {"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}})
        try:
            port = pool.ensure(b, f"{self.command} {path}")
        except RuntimeError as e:
            return self.reply(503, {"error": {"message": str(e), "type": "unavailable_error", "code": 503}})
        try:
            self.proxy(port, data)
        finally:
            b.busy(-1)

    def proxy(self, port, data):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if data is not None:
            hdrs["Content-Length"] = str(len(data))
        try:
            conn.request(self.command, self.path, body=data, headers=hdrs)
            r = conn.getresponse()
        except OSError as e:
            conn.close()
            return self.reply(502, {"error": {"message": f"the model server did not answer: {e}", "type": "unavailable_error", "code": 502}})
        try:
            self.send_response_only(r.status, r.reason)
            for k, v in r.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            n = r.getheader("Content-Length")
            nobody = self.command == "HEAD" or r.status in (204, 304) or 100 <= r.status < 200
            if nobody or n is not None:
                self.send_header("Content-Length", "0" if nobody and n is None else n)
                self.end_headers()
                while not nobody:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            else:                                       # streaming (SSE): pass each piece on as it comes
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:                                 # BrokenPipe / ConnectionReset / (Windows) ConnectionAborted
            self.close_connection = True                # the agent hung up: closing ours makes llama-server stop generating
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = handle_any


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = not WINDOWS          # on Windows SO_REUSEADDR would let two listeners share the port

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):   # an agent that hung up
            super().handle_error(request, client_address)


def serve(backend, port, host="127.0.0.1"):
    """Listen until interrupted; returns the server (tests run it in a thread and call shutdown()). backend: a Backend or a Pool of them."""
    srv = Server((host, port), Handler)
    srv.pool = backend if isinstance(backend, Pool) else Pool([backend])
    srv.b = srv.pool.bs[0]
    for b in srv.pool.bs:
        threading.Thread(target=b.watch, daemon=True).start()
    return srv


# ---- start at login ----

def unit_path():
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config"), "systemd", "user", UNIT)


def startup_link():
    return os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs", "Startup", LINK)


def pythonw():
    w = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    return w if os.path.isfile(w) else sys.executable


def unit_text():
    q = lambda s: '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'      # systemd quoting, for paths with spaces
    env = "".join(f"Environment={q(k + '=' + os.environ[k])}\n" for k in ("XDG_CACHE_HOME",) if os.environ.get(k))
    return ("[Unit]\nDescription=moe-cache on-demand endpoint for coding agents (starts the model on the first request, stops it when idle)\n"
            "StartLimitIntervalSec=120\nStartLimitBurst=5\n\n"
            f"[Service]\nExecStart={q(sys.executable)} {q(os.path.join(mc.HERE, 'bin', 'moe_cache.py'))} on-demand\n{env}"
            "Restart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n")


def install_autostart(dry=False):
    if WINDOWS:
        link = startup_link()
        ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:MC_LINK); $s.TargetPath = $env:MC_PY; "
              "$s.Arguments = '\"' + $env:MC_SCRIPT + '\" on-demand'; $s.WorkingDirectory = $env:MC_HERE; $s.WindowStyle = 7; "
              "$s.Description = 'moe-cache on-demand endpoint for coding agents'; $s.Save()")
        env = dict(os.environ, MC_LINK=link, MC_PY=pythonw(), MC_SCRIPT=os.path.join(mc.HERE, "bin", "moe_cache.py"), MC_HERE=mc.HERE)
        if dry:
            print(f"{link} -> {env['MC_PY']} \"{env['MC_SCRIPT']}\" on-demand"); return 0
        os.makedirs(os.path.dirname(link), exist_ok=True)
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], env=env, capture_output=True, text=True, timeout=60)
        if r.returncode or not os.path.isfile(link):
            raise SystemExit(f"could not create the Startup shortcut {link}: {(r.stderr or r.stdout).strip()[-300:]}")
        say(f"autostart: {link} (runs at every login, no admin rights needed)")
        if not running_pid():
            os.makedirs(mc.CACHE_DIR, exist_ok=True)
            with open(LISTENER_LOG, "a", encoding="utf-8") as lf:      # the same file the login start (no console) writes to
                subprocess.Popen([pythonw(), os.path.join(mc.HERE, "bin", "moe_cache.py"), "on-demand"], cwd=mc.HERE, stdin=subprocess.DEVNULL,
                                 stdout=lf, stderr=lf, creationflags=0x00000008 | 0x00000200 | 0x08000000)
            say("autostart: started it now too")      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        return 0
    path = unit_path()
    if dry:
        print(path); print(unit_text()); return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(unit_text())
    say(f"autostart: {path}")
    sc = mc.which("systemctl")
    if not sc:
        say("autostart: systemctl not found: start `moe_cache.py on-demand` from your desktop's autostart settings instead"); return 0
    for args in (["daemon-reload"], ["enable", "--now", UNIT]):
        r = subprocess.run([sc, "--user"] + args, capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise SystemExit(f"systemctl --user {' '.join(args)} failed: {(r.stderr or r.stdout).strip()[-300:]}")
    say(f"autostart: enabled and started (systemctl --user status {UNIT}; log: journalctl --user -u {UNIT})")
    return 0


def remove_autostart():
    if WINDOWS:
        link = startup_link()
        if os.path.isfile(link):
            os.remove(link)
            say(f"autostart: removed {link}")
        else:
            say("autostart: there was no Startup shortcut")
        stop_running()
        return 0
    path = unit_path()
    sc = mc.which("systemctl")
    if sc:
        subprocess.run([sc, "--user", "disable", "--now", UNIT], capture_output=True, text=True, timeout=60)
    if os.path.isfile(path):
        os.remove(path)
        if sc:
            subprocess.run([sc, "--user", "daemon-reload"], capture_output=True, timeout=60)
        say(f"autostart: removed {path}")
    else:
        say("autostart: there was no systemd unit")
    stop_running()
    return 0


# ---- the running listener ----

def running_pid():
    """PID of a listener that is still running (from its PID file), else None."""
    try:
        pid, port = (int(x) for x in open(PID_FILE).read().split()[:2])
    except (OSError, ValueError):
        return None
    if mc.http(f"http://127.0.0.1:{port}/health", timeout=2)[1].find('"on_demand"') < 0:
        return None
    return pid


def stop_running():
    pid = running_pid()
    if not pid:
        return False
    if WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGTERM)
    say(f"on-demand: stopped the listener (PID {pid})")
    return True


def say(msg):
    mc.say(msg)


def status(c):
    """--status: the remembered choice, whether the listener runs and the model server's state."""
    pid = running_pid()
    say(f"on-demand: model {c['model']}  options {' '.join(c.get('args', [])) or '-'}  port {c.get('port', 8080)}  idle stop {c.get('idle', 15):g} min")
    for e in c.get("extra") or []:
        say(f"on-demand: also {model_alias(e['model'], e.get('args', []))}: {e['model']}  options {' '.join(e.get('args', [])) or '-'}")
    say(f"on-demand: listener {'running, PID ' + str(pid) if pid else 'not running'}; autostart "
        + ("on" if os.path.isfile(startup_link() if WINDOWS else unit_path()) else "off"))
    st, body = mc.http(f"http://127.0.0.1:{c.get('port', 8080)}/health", timeout=2)
    if st == 200 and '"on_demand"' in body:
        say(f"on-demand: model server {json.loads(body)['on_demand']}")
    return 0


def on_demand(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py on-demand", allow_abbrev=False, description=__doc__.split("\n\n")[0],
                                 epilog="Any other option goes to `moe_cache.py server` (for example --ram 20, --mtp, --fast, --llama-server PATH); "
                                        "--agent is always on. The choice is kept in " + CHOICE_FILE,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="model to serve (remembered; leave out to use the remembered one)")
    ap.add_argument("--port", type=int, help="the agents' port (default 8080; remembered)")
    ap.add_argument("--idle", type=float, help="stop the model after this many minutes without a request (default 15; remembered)")
    ap.add_argument("--start-timeout", type=int, default=900, help="seconds to wait for the model to load")
    ap.add_argument("--install-autostart", action="store_true", help="start the listener at login (systemd --user unit / Windows Startup folder)")
    ap.add_argument("--remove-autostart", action="store_true", help="undo --install-autostart and stop the listener")
    ap.add_argument("--stop", action="store_true", help="stop a running listener (and its model)")
    ap.add_argument("--status", action="store_true", help="print the remembered choice and whether the listener runs")
    ap.add_argument("--add", action="store_true", help="add MODEL as another model (the default stays); a request's \"model\" picks it")
    ap.add_argument("--remove", metavar="NAME", help="forget an added model (its name as /v1/models lists it)")
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i:]
    a, rest = ap.parse_known_args(argv)
    if a.remove_autostart:
        return remove_autostart()
    if a.stop:
        return 0 if stop_running() else (say("on-demand: no listener is running") or 1)
    c = load_choice()
    if a.remove:
        if not c:
            raise SystemExit("no model chosen yet: moe_cache.py on-demand MODEL.gguf [launcher options]")
        keep = [e for e in c.get("extra") or [] if model_alias(e["model"], e.get("args", [])) != a.remove]
        if len(keep) == len(c.get("extra") or []):
            raise SystemExit(f"no added model named {a.remove} (the default {model_alias(c['model'], c.get('args', []))} is replaced, not removed: "
                             "on-demand MODEL.gguf)")
        c["extra"] = keep
        save_choice(c)
        say(f"on-demand: forgot {a.remove}; restart the listener to use the change")
        return 0
    if a.add and not a.model:
        raise SystemExit("give the model to add: moe_cache.py on-demand MODEL.gguf [options] --add")
    if a.add and not c:
        raise SystemExit("choose the default model first: moe_cache.py on-demand MODEL.gguf [options]; then add others with --add")
    if a.model or rest or extra:
        if not a.model:
            raise SystemExit("give the model first: moe_cache.py on-demand MODEL.gguf [options]")
        bad = [x for x in rest if x.split("=")[0] in OWN]
        if bad:
            raise SystemExit(f"on-demand sets {', '.join(sorted(set(bad)))} itself (the agents' port is on-demand's --port)")
        model = os.path.abspath(a.model)
        if not os.path.isfile(model):
            raise SystemExit(f"model file not found: {a.model}")
        args = rest + extra
        env = {k: os.environ[k] for k in ("LLAMA_SERVER", "MOE_CACHE_LIB") if os.environ.get(k)}   # autostart has no shell environment
        # check the options now (a dry run plans the model from its header) rather than on the first agent request
        r = subprocess.run(LAUNCHER + [model, "--agent", "--dry-run", "--port", "1"] + args, capture_output=True, text=True,
                           env=dict(os.environ, **env), timeout=600)
        if r.returncode:
            raise SystemExit("the launcher refused these options:\n" + (r.stderr or r.stdout).strip()[-1500:])
        if a.add:
            alias = model_alias(model, args)
            if alias == model_alias(c["model"], c.get("args", [])):
                raise SystemExit(f"{alias} is already the default model (give the second one another file or -- --alias NAME)")
            c["extra"] = [e for e in c.get("extra") or [] if model_alias(e["model"], e.get("args", [])) != alias] + [{"model": model, "args": args, "env": env}]
            c.update({k: v for k, v in (("port", a.port), ("idle", a.idle)) if v})
            save_choice(c)
            say(f"on-demand: added {alias} ({os.path.basename(model)}) with {' '.join(args) or 'the default options'}; the default stays "
                f"{model_alias(c['model'], c.get('args', []))}. One model runs at a time; restart the listener to use the change")
            return 0
        else:
            old = c or {}
            c = {"model": model, "args": args, "env": env, "port": a.port or old.get("port", 8080), "idle": a.idle or old.get("idle", 15)}
            save_choice(c)
            say(f"on-demand: remembered {os.path.basename(model)} with {' '.join(args) or 'the default options'} ({CHOICE_FILE})"
                + (f"; the added models ({', '.join(model_alias(e['model'], e.get('args', [])) for e in old.get('extra') or [])}) are forgotten"
                   if old.get("extra") else ""))
    elif not c:
        raise SystemExit("no model chosen yet: moe_cache.py on-demand MODEL.gguf [launcher options]")
    elif a.port or a.idle:
        c.update({k: v for k, v in (("port", a.port), ("idle", a.idle)) if v})
        save_choice(c)
    if a.status:
        return status(c)
    if a.install_autostart:
        return install_autostart()
    if sys.stdout is None:          # pythonw (Windows autostart) has no console: keep the start/stop lines in a file
        os.makedirs(mc.CACHE_DIR, exist_ok=True)
        sys.stdout = sys.stderr = open(LISTENER_LOG, "a", encoding="utf-8", buffering=1)
    port = c.get("port", 8080)
    if mc.port_busy(port):
        raise SystemExit(f"port {port} is already in use (another server, or on-demand already runs: moe_cache.py on-demand --status)")
    if not WINDOWS:
        os.umask(0o077)
    b = Pool([Backend(e, idle_min=c.get("idle", 15), start_timeout=a.start_timeout) for e in [c] + list(c.get("extra") or [])])
    srv = serve(b, port)
    with open(PID_FILE, "w") as f:
        f.write(f"{os.getpid()} {port}\n")
    d = b.bs[0]
    log(f"listening on http://127.0.0.1:{port}/v1 for {', '.join(x.alias for x in b.bs)} (context {d.ctx}); the model starts on the first request "
        f"and stops after {d.idle / 60:g} idle min" + ("; one model at a time, the request's model name picks it" if len(b.bs) > 1 else "")
        + ". Connect agents: moe_cache.py connect hermes|opencode --write")
    def bye(sig, frame):
        raise KeyboardInterrupt
    if not WINDOWS:
        signal.signal(signal.SIGTERM, bye)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        b.stop("the listener was stopped")
        try:
            if open(PID_FILE).read().split()[0] == str(os.getpid()):
                os.remove(PID_FILE)
        except OSError:
            pass
    return 0
