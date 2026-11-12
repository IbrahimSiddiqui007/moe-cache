#!/usr/bin/env python3
"""moe-cache launcher (Linux and Windows; Python 3.8+, standard library only).

  moe_cache.py launch  [--yes] [--model M] [--agent [NAME]]               guided start in plain words: check this PC, pick a model, start the server, connect an agent
  moe_cache.py server  MODEL.gguf [options] [-- extra llama-server args]   run llama-server with the expert-cache plugin
  moe_cache.py pick    [--use agents|chat|docs] [--apply]                  recommend a model for this machine (RAM, VRAM, disk, CPU) and use; --apply uses it
  moe_cache.py models  [--use agents|chat|docs] [--ctx N] [--all] [--json]  the model catalogue for this PC: fits / tight / no, size, downloaded or not, measured speed
  moe_cache.py doctor  [MODEL.gguf] [--llama-server PATH] [--ram GB]       the same as check (preflight: GPU, driver, RAM, disk, llama-server; what to fix)
  moe_cache.py check   [MODEL.gguf] [--llama-server PATH] [--ram GB]       check that this machine, llama-server and the model are ready
  moe_cache.py update --check                                            is a newer moe-cache or llama.cpp out? (downloads nothing)
  moe_cache.py update --apply [--tag vX.Y.Z] [--allow-downgrade] [--yes] [--yes-unverified] [--dry-run]   install the newer release (sha256 always, attestation unless --yes-unverified; Linux keeps the old version)
  moe_cache.py bench-report MODEL.gguf [--ram GB] [--flush-from DIR]     stock vs moe-cache on this machine; a report to paste into a GitHub issue (docs/SHARE_YOUR_NUMBERS.md)
  moe_cache.py bench-record collect|monitor|assemble|validate|card|ingest|ingest-issue|claim|board|lint|replay     the benchmark record (docs/BENCHMARK_TEMPLATE.md): VRAM, RAM, CPU, settings, versions
  moe_cache.py learn   MODEL.gguf [options]                                build a warm-start profile from a set of prompts
  moe_cache.py project save|restore|list [NAME] [--url URL] [--api-key KEY] save / restore a running server's prompt state
  moe_cache.py connect hermes|opencode|claude|codex|pi|openclaw|aider|qwen [--url URL] [--write]   point Hermes Agent, OpenCode, Claude Code, Codex CLI, Pi, OpenClaw, Aider or Qwen Code at a running server (start it with --agent); --check proves the setup works (models, chat, tool call, window; --json)
  moe_cache.py on-demand [MODEL.gguf] [options] [--install-autostart]      agent endpoint that starts the model on the first request, stops it when idle
  moe_cache.py diffusion MODEL.gguf -p PROMPT [--canvas 256 --steps 64]  run llama-diffusion-cli (LLaDA-MoE, RND1) with the plugin, experts on the CPU side
  moe_cache.py pack    MODEL.gguf [--out PATH] [--verify]                 write MODEL.moepack: one block per expert, so a missing expert is one large read
  moe_cache.py key     [--path]                                            print the API key that agent mode and --project servers use
  moe_cache.py ui      [--ui-port 8070] [--no-browser]                     local control page (also: moe-cache-server --ui); bin/moe_cache_ui.py
  moe_cache.py server  MODEL.gguf --top [--ram GB --ctx N --port N --mode M --agent --mtp on|off]   start the model and watch it in the terminal monitor (q stops it)
  moe_cache.py top     [--url URL | --port N] [--once]                    live one-screen monitor in the terminal (speed, cache, SSD, memory, log); bin/moe_cache_top.py
  moe_cache.py --version                                                   print the moe-cache version

`moe_cache.py server --help` (and so on) lists the options. The bash scripts bin/moe-cache-server, -check, -learn and -project call this file.
"""
import argparse, glob, json, math, os, re, shutil, signal, socket, subprocess, sys, tempfile, threading, time, urllib.error, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) not in sys.path else None   # tests load this file by path
from moe_cache_wintool import win_tool  # noqa: E402  (bin/: Windows helpers by full System32 path, AUD-034)

VERSION = "0.3.7.1"                    # tools/set_version.sh fills this in with the release's version
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = os.name == "nt"
CACHE_DIR = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "moe-cache")
NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


# LAUNCHER-003: what launch_plan() sets while it builds a plan belongs to the building thread (control-page threads plan at the same time):
#   _TL.notes  recorders (lists) that want every say() line: (text, is_error) in order
#   _TL.quiet  >0: record but do not print (a caller that wants the plan as data, e.g. --plan-json)
#   _TL.max_ram  --max-ram of the plan being built (mem_cap() reads it before the MOE_CACHE_MAX_RAM_GB variable)
_TL = threading.local()
_UMASK_LOCK = threading.Lock()      # umask is per process: plans that change and restore it take turns, so one cannot restore the other's value


def say(msg, err=False):
    for rec in getattr(_TL, "notes", ()):
        rec.append((str(msg), bool(err)))
    if not getattr(_TL, "quiet", 0):
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def which(prog):
    return shutil.which(prog) or prog


def api_advice(want):
    """What a user does when the plugin build is for the other ggml backend API than llama-server (want = llama-server's API)."""
    want = int(want)
    if WINDOWS:
        return (f"use ggml-moe-cache-api{want}.dll from the kit: set MOE_CACHE_LIB to its full path (setx MOE_CACHE_LIB \"...\\ggml-moe-cache-api{want}.dll\" "
                "for good)")
    return (f"download moe-cache-vX.Y.Z{'-ggml0.26' if want == 3 else ''}-linux-x86_64.tar.gz (API {want}) from the Releases page, or build from source: "
            f"{'GGML_API=3 ' if want == 3 else ''}./build.sh")


RELEASES_URL = "https://github.com/ggml-org/llama.cpp/releases"
_LS = {}        # the last search: {"path", "source", "build", "api", "looked": [places], "others": [paths]} (see find_llama_server)


def llama_exe():
    return "llama-server.exe" if WINDOWS else "llama-server"


def expand_win_vars(v):
    """%NAME% as the registry stores it (REG_EXPAND_SZ), from the process environment; unknown names stay as they are."""
    return re.sub(r"%([^%]+)%", lambda m: os.environ.get(m.group(1), m.group(0)), v)


def win_registry_env(name):
    """[(where, value)] of an environment variable as Windows stores it: the user's, then the machine's. `setx` writes here, and a program started
    earlier (Explorer, so moe-cache.bat started from it) never sees the new value in its own environment. [] off Windows."""
    out = []
    if not WINDOWS:
        return out
    try:
        import winreg
    except ImportError:
        return out
    for where, root, sub_ in (("the user environment (registry)", winreg.HKEY_CURRENT_USER, "Environment"),
                              ("the machine environment (registry)", winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
        try:
            with winreg.OpenKey(root, sub_) as k:
                v = winreg.QueryValueEx(k, name)[0]
            if isinstance(v, str) and v.strip():
                out.append((where, expand_win_vars(v.strip())))
        except OSError:
            pass
    return out


def saved_llama_server():
    """The path the control page saved (ui.json, key llama_server) when it could not find one itself; "" when none."""
    try:
        v = json.load(open(os.path.join(CACHE_DIR, "ui.json"), encoding="utf-8")).get("llama_server")
        return v if isinstance(v, str) else ""
    except Exception:
        return ""


def resolve_llama_server(v):
    """A path to llama-server: the file itself, or a folder that holds it (also bin / build\\bin below it); None when there is none."""
    if not v:
        return None
    v = os.path.expanduser(v.strip().strip('"'))
    if os.path.isfile(v):
        return os.path.abspath(v)
    if os.path.isdir(v):
        for sub_ in ("", "bin", os.path.join("build", "bin"), os.path.join("build", "bin", "Release")):
            c = os.path.join(v, sub_, llama_exe())
            if os.path.isfile(c):
                return os.path.abspath(c)
    return None


def llama_search_dirs():
    """Folders that may hold llama-server, in the order they are looked at (only existing files count)."""
    home = os.path.expanduser("~")
    if WINDOWS:
        up = os.environ.get("USERPROFILE") or home
        return ([r"C:\llama"] + sorted(glob.glob(r"C:\llama*")) + sorted(glob.glob(os.path.join(up, "llama*")))
                + sorted(glob.glob(os.path.join(up, "Downloads", "llama-b*-bin-win-*"))) + [HERE,
                os.path.join(HERE, "llama")])       # the kit's bundled official llama.cpp (one download for friends)
    return [os.path.join(home, "moe-cache-setup", "llama.cpp", "build", "bin"), os.path.join(home, "llama.cpp", "build", "bin"), HERE,
            os.path.join(HERE, "llama")]            # the tarball's bundled official llama.cpp (the -ggml0.26 file)


# llama.cpp commits (first 9 digits) that CI ran this plugin against: the tested "yes" rows of .github/workflows/identity.yml; the @commit entries of
# MOE_CACHE_TESTED_GGML in src/moe_cache_plugin.cpp are a subset. tests/tested_commits_test.py fails when they disagree.
TESTED_COMMITS = ("7fe450e19", "23b0202a1", "ff0d396e3", "51fe42c5d", "998200491")


def tested_commit(ver):
    """True when `ver` ('commit 51fe42c5d', any length from 7 digits) is a build of TESTED_COMMITS."""
    m = re.search(r"[0-9a-f]{7,}", ver or "")
    return bool(m) and any(t.startswith(m.group(0)) or m.group(0).startswith(t) for t in TESTED_COMMITS)


def ls_build(ls):
    """llama-server's build number (`--version`: 'version: 11966 (ff0d396e3)'), None when unknown."""
    try:
        r = subprocess.run([ls, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        text = r.stdout + r.stderr
        m = re.search(r"\(build (\d+)", text) or re.search(r"version:\s*(\d+)\s*\(", text)     # '0.7.0-dev (build 11669, commit ...)' or the old 'version: 4567 (abc)'
        n = int(m.group(1)) if m else 0
        return n if n >= 100 else None        # distro builds report 'build 1' / 0: no usable number
    except Exception:
        return None


def llama_patch_line(ls):
    """The one-line note the kit's patched llama.cpp carries next to llama-server (PATCHES-moe-cache.txt, written by tools/bundle_llama_patched.sh), e.g.
    'llama.cpp b12207 + moe-cache patches 0004,0020,0030,0040,0090,0091 built in, all off'. None for an official or any other build (no such file)."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(ls)), "PATCHES-moe-cache.txt"), encoding="utf-8", errors="replace") as f:
            line = f.readline().strip()
        return line[:200] if line.startswith("llama.cpp ") else None
    except Exception:
        return None


def shipped_apis():
    """Backend APIs of the plugin builds that ship with (or were built next to) this launcher."""
    pats = [default_lib(), os.path.join(HERE, "build", "win*", "ggml-moe-cache*.dll"), os.path.join(HERE, "build-*", "ggml-moe-cache*.dll"),
            os.path.join(HERE, "ggml-moe-cache*.dll"), os.path.join(HERE, "..", "ggml-moe-cache*.dll"), os.path.join(HERE, "build-*", "libggml-moe-cache.so")]
    return {a for pat in pats for c in glob.glob(pat) for a in [lib_api(c)] if a}


def find_llama_server(saved=None):
    """Looks for llama-server (the same code for the launcher, check, the picker and the control page). Order: LLAMA_SERVER in the process
    environment; on Windows LLAMA_SERVER as stored in the registry (user, then machine); the control page's saved path; the PATH (process, and on
    Windows the registry's); then every default folder (Windows: C:\\llama*, %USERPROFILE%\\llama*, Downloads\\llama-b*-bin-win-*, the kit; Linux:
    install.sh's build, ~/llama.cpp/build/bin, the kit). An explicit setting that points at a file wins; among the folders the newest build whose
    ggml API a shipped plugin supports wins. One exception to "the PATH comes before the folders": a PATH llama-server whose ggml API no shipped plugin
    supports, or whose API cannot be read (the distro's, API 2, next to a kit that ships the API 3 plugin and a bundled llama.cpp) gives way to a default-folder one that has a supported
    API, with a note (v0.3.6-beta2 smoke test: `check` failed on /usr/bin/llama-server and advised another download); with no supported one anywhere the
    PATH one stays, as before. Returns {"path", "source", "build", "api", "others", "looked", "note"}; "path" is None when nothing was found."""
    looked = []
    res = {"path": None, "source": None, "build": None, "api": None, "others": [], "looked": looked, "note": None}
    def hit(path, source):
        res.update(path=path, source=source, api=ls_api(path) if os.path.isabs(path) else None)
        return res
    def folder_candidates():
        found = []
        for d in llama_search_dirs():
            r = next((os.path.join(d, x, llama_exe()) for x in ("", "bin", os.path.join("build", "bin"), os.path.join("build", "bin", "Release"))
                      if os.path.isfile(os.path.join(d, x, llama_exe()))), None)
            looked.append(f"folder {d}: " + ("found" if r else "no llama-server"))
            if r and os.path.abspath(r) not in found:
                found.append(os.path.abspath(r))
        return found
    def supported_folder_instead(found_path, label):
        """A default-folder llama-server with a supported API to use instead of the PATH one at found_path (None: keep the PATH one)."""
        apis, api = shipped_apis(), ls_api(found_path)
        if not apis or api in apis:           # LAUNCHER-689: an API that cannot be read (None) is not a supported one: a known-good kit server beats it
            return None
        found = folder_candidates()
        good = [c for c in found if ls_api(c) in apis]
        if not good:
            return None
        best = max(good, key=lambda c: (ls_build(c) or 0, os.path.getmtime(c)))
        hit(best, "a default folder")
        res["build"] = ls_build(best)
        res["others"] = [c for c in found if c != best]
        kit = os.path.abspath(best).startswith(os.path.abspath(HERE) + os.sep)
        res["note"] = (f"{label} llama-server (ggml API {api or 'unreadable'}) skipped: no shipped plugin for it; using "
                       + ("the kit's bundled llama-server" if kit else best) + f" (API {res['api']})")
        looked.append(res["note"])
        return res
    explicit = [("LLAMA_SERVER", os.environ.get("LLAMA_SERVER"))] + [("LLAMA_SERVER in " + w, v) for w, v in win_registry_env("LLAMA_SERVER")]
    explicit.append(("the control page's saved path", saved if saved is not None else saved_llama_server()))
    for src, v in explicit:
        if v:
            r = resolve_llama_server(v) or shutil.which(v)
            looked.append(f"{src}: {v}" + ("" if r else " (no llama-server there)"))
            if r:
                return hit(r, src)
        else:
            looked.append(f"{src}: not set")
    w = shutil.which("llama-server")
    looked.append("PATH: " + ("llama-server" if w else "not on it"))
    if w:
        w = os.path.abspath(w)                # an absolute path, so the one checked is the one started (not whatever "llama-server" means later)
        return supported_folder_instead(w, "PATH") or hit(w, "PATH")
    if WINDOWS:
        for where, v in win_registry_env("PATH"):
            for d in v.split(";"):
                r = resolve_llama_server(d) if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, llama_exe())) else None
                if r:
                    looked.append(f"PATH in {where}: {d}")
                    return supported_folder_instead(r, "PATH in " + where) or hit(r, "PATH in " + where)
        looked.append("PATH in the registry: not on it")
    found = folder_candidates()
    if found:
        apis = shipped_apis()
        def rank(c):
            api = ls_api(c)
            return (2 if (not apis or api in apis) and api else 1 if api is None else 0, ls_build(c) or 0, os.path.getmtime(c))
        best = max(found, key=rank) if len(found) > 1 else found[0]
        hit(best, "a default folder")
        res["build"] = ls_build(best)
        res["others"] = [c for c in found if c != best]
    return res


def default_llama_server(refresh=False):
    """llama-server when --llama-server is not given (see find_llama_server; kept per process, refresh=True looks again). When nothing is found:
    LLAMA_SERVER as set, else plain "llama-server" (the callers then say it is missing and where this looked: ls_looked())."""
    if refresh or "done" not in _LS:
        _LS.clear()
        _LS.update(find_llama_server(), done=True)
    return _LS["path"] or os.environ.get("LLAMA_SERVER") or "llama-server"


def ls_looked():
    default_llama_server()
    return "; ".join(_LS.get("looked") or [])


def ls_found_line(ls):
    """One log line saying which llama-server was found by the search (None when it came from --llama-server or LLAMA_SERVER)."""
    default_llama_server()
    if _LS.get("path") != ls or _LS.get("source") in (None, "LLAMA_SERVER"):
        return None
    return (f"moe-cache: llama-server: {which(ls) if ls != 'llama-server' else ls} (found by {_LS['source']}"
            + (f", build b{_LS['build']}" if _LS.get("build") else "") + (f", ggml API {_LS['api']}" if _LS.get("api") else "")
            + (f"; also found, not used: {', '.join(_LS['others'])}" if _LS.get("others") else "") + (f"; {_LS['note']}" if _LS.get("note") else "") + ")")


# ---- how a llama-server loads the plugin ----

_LDD_MEMO = {}          # PACKAGING-027: path -> (time, ldd output); one launch asks three times, a long-lived control page asks again later
LDD_MEMO_SECONDS = 30


def ldd(path):
    if WINDOWS:
        return ""
    key, now = which(path), time.time()
    hit = _LDD_MEMO.get(key)
    if hit and now - hit[0] < LDD_MEMO_SECONDS:
        return hit[1]
    try:
        out = subprocess.run(["ldd", key], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20).stdout
    except Exception:
        return ""
    _LDD_MEMO[key] = (now, out)
    return out


PROBE_CACHE_NAME = "probe-cache.json"
PROBE_CACHE_KEEP = 8


def ls_fingerprint(ls):
    """What makes a llama-server's `--help` text change: the binary and the ggml/llama libraries it loads (the files next to it, and the ones ldd names).
    A list of [path, mtime_ns, size]; None when the binary cannot be found (a test's fake path: nothing is cached)."""
    try:
        exe = os.path.realpath(which(ls))
        files = [exe]
        d = os.path.dirname(exe)
        files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if re.match(r"(lib)?(ggml|llama)", f)]
        for line in ldd(ls).splitlines():
            m = re.search(r"=>\s*(\S*(?:ggml|llama)\S*)", line)
            if m:
                files.append(os.path.realpath(m.group(1)))
        out = []
        for f in dict.fromkeys(files):
            st = os.stat(f)
            out.append([f, st.st_mtime_ns, st.st_size])
        return out
    except (OSError, TypeError):
        return None


def ls_help(ls, run=None):
    """The text of `llama-server --help` (starting it costs 0.3 s, a third of a dry-run launch), remembered in memory and in CACHE_DIR/probe-cache.json
    while the binary and its libraries are unchanged (PACKAGING-027). `run` replaces subprocess.run (tests, Machine.run)."""
    fp = ls_fingerprint(ls)
    path = os.path.join(CACHE_DIR, PROBE_CACHE_NAME)
    data = {}
    if fp:
        try:
            data = json.load(open(path, encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        hit = data.get(fp[0][0])
        if isinstance(hit, dict) and hit.get("fp") == fp and isinstance(hit.get("help"), str):
            return hit["help"]
    text = (run or subprocess.run)([ls, "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30).stdout
    if fp and text:
        data[fp[0][0]] = {"fp": fp, "help": text}
        for old in list(data)[:-PROBE_CACHE_KEEP]:
            del data[old]
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except OSError:
            pass
    return text


def load_env(ls, lib, method="auto"):
    """Environment that makes this llama-server load the plugin, and the method used.
    preload: llama-server links its backends directly (distro packages): LD_PRELOAD.  path: backends are separate files (official releases): GGML_BACKEND_PATH."""
    deps = ldd(ls)
    if not WINDOWS and "libggml-base" not in deps:
        say(f"warning: {which(ls)} does not use shared ggml libraries, the plugin cannot be loaded into it (build llama.cpp with BUILD_SHARED_LIBS=ON)", err=True)
    if method == "auto":
        method = "preload" if (not WINDOWS and "libggml-cpu" in deps) else "path"
    if method == "preload":
        if WINDOWS:
            raise SystemExit("--load-method preload does not exist on Windows (use path)")
        return [("LD_PRELOAD", lib), ("MOE_CACHE_PRELOAD", "1")], method
    if method == "path":
        return [("GGML_BACKEND_PATH", lib)], method
    raise SystemExit(f"unknown load method: {method} (use auto, preload or path)")


GGML_API_OF = {(0, 24): "2", (0, 25): "2", (0, 26): "3", (0, 27): "3"}


def ggml_api_of(major, minor):
    """Backend API of a ggml version. Newer 0.x versions than the table keep API 3 until ggml bumps GGML_BACKEND_API_VERSION (0.28 = llama.cpp
    b11966+ did not): without this, b11966 got no plugin match and `check` said the DLL was missing (task Q). The plugin's own start-up layout
    check still refuses a ggml whose ops really changed."""
    if (major, minor) in GGML_API_OF:
        return GGML_API_OF[(major, minor)]
    return "3" if major == 0 and minor > max(n for _, n in GGML_API_OF) else None


def ggml_api_in_dll(data):
    """Backend API from the bytes of ggml-base.dll. ggml's version string ("0.27.0") sits next to its commit string (hex digits or "unknown"), so a
    version followed by one of those wins over any other version-looking string in the file (a stray "0.30.1" from a library must not pick API 3 for a
    ggml 0.25). Without such a pair: the highest known version in the file. None when there is no usable version."""
    found = []
    for m in re.finditer(rb"\x00(\d+)\.(\d+)\.(\d+)(?=\x00((?:[0-9a-fA-F]{7,40}|unknown)\x00)?)", data):
        v = (int(m.group(1)), int(m.group(2)))
        if ggml_api_of(*v):
            found.append((v, m.group(4) is not None))
    for v, anchored in found:
        if anchored:
            return ggml_api_of(*v)
    return ggml_api_of(*max(v for v, _ in found)) if found else None


def ls_api(ls):
    """Backend API of the ggml this llama-server uses, from its libggml-base version (0.24-0.25 -> 2, 0.26 and newer -> 3); None when unknown."""
    if WINDOWS:
        # no ldd there (the warning never fired on the Claw): ggml-base.dll next to llama-server.exe holds its version as a plain string ("0.27.0")
        try:
            return ggml_api_in_dll(open(os.path.join(os.path.dirname(which(ls)), "ggml-base.dll"), "rb").read())
        except (OSError, TypeError):
            pass
        return None
    for line in ldd(ls).splitlines():
        if "libggml-base" in line and "=>" in line:
            path = line.split("=>")[1].split()[0]
            m = re.search(r"(\d+)\.(\d+)\.(\d+)", os.path.basename(os.path.realpath(path)))
            if m:
                v = (int(m.group(1)), int(m.group(2)))
                return ggml_api_of(*v)
    return None


def lib_api(lib):
    """API a plugin build was made for (a plain string inside the library); None for builds from before 0.2.3."""
    try:
        m = re.search(rb"moe-cache-api=(\d+)", open(lib, "rb").read())
        return m.group(1).decode() if m else None
    except OSError:
        return None


def default_lib():
    return os.path.join(HERE, "build", "win", "ggml-moe-cache.dll") if WINDOWS else os.path.join(HERE, "build", "libggml-moe-cache.so")


def pick_lib(ls):
    """The plugin build in build*/ that matches the server's ggml, else the default build."""
    want = ls_api(ls)
    if want:
        if os.path.isfile(default_lib()) and lib_api(default_lib()) == want:
            return default_lib()
        # several builds for this API (old test builds, too): the newest one, which is the most likely to match the code
        if WINDOWS:   # build.sh --windows: build/win (API 2), build/win-api3 or BUILD_DIR; the windows-kit has ggml-moe-cache-api2.dll / -api3.dll next to moe_cache.py
            pats = [os.path.join(HERE, "build", "win*", "ggml-moe-cache*.dll"), os.path.join(HERE, "build-*", "ggml-moe-cache*.dll"),
                    os.path.join(HERE, "ggml-moe-cache*.dll"), os.path.join(HERE, "..", "ggml-moe-cache*.dll")]
        else:
            pats = [os.path.join(HERE, "build-*", "libggml-moe-cache.so")]
        found = [c for pat in pats for c in glob.glob(pat) if lib_api(c) == want]
        if found:
            return max(found, key=os.path.getmtime)
    return default_lib()


# ---- small helpers ----

def model_parts(model):
    m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", model)
    if m:
        # the folder and file name are the user's text: '[', '*' and '?' in them match themselves, only the shard number is a wildcard
        return sorted(glob.glob(glob.escape(model[:m.start()]) + "-[0-9][0-9][0-9][0-9][0-9]-of-" + m.group(1) + ".gguf"))
    return [model]


def loading_line(model):
    """CACHE-181: one line before llama-server starts: how much is read and a rough time. The range is from a slow SATA SSD (0.5 GB/s) to a fast NVMe
    drive (3 GB/s); a file already in the OS page cache loads faster, so it is an upper bound for a cold load, not a promise."""
    try:
        gb = sum(os.path.getsize(f) for f in model_parts(model)) / 1e9
    except OSError:
        return "moe-cache: loading the model (size unknown); a first load of a big model can take minutes, the server prints when it is ready"
    lo, hi = max(1, round(gb / 3.0)), max(2, round(gb / 0.5))
    return f"moe-cache: loading {gb:.1f} GB, about {lo}-{hi} s on a cold start (fast NVMe to SATA SSD; less when the file is already cached); the server prints when it is ready"


def plugin_off_line(line):
    """True for the plugin's lines that say it is not caching (setup failed, MOE_CACHE_DISABLE, wrong backend API); its other "refused" lines
    (a working-set minimum or memory Windows did not grant) are warnings of a plugin that keeps running."""
    return b"plugin disabled" in line or b"not loaded, the model loads the normal way" in line

SLOW_DRIVE_GBS = 1.0      # random 1 MiB reads. PC NVMe drives: 3.1 (C:) and 4.5 GB/s (D:); the plugin's own read rate followed the probe (0.74 vs 0.70
                          # C:/D:) but 122B decode differed by 2 % at 98 % hits (RESULTS 61): only SATA / HDD / USB speeds are worth a warning


def drive_read_speed(path, seconds=4.0, block=1 << 20, threads=4):
    """Random reads of `block` bytes from `path` that bypass the file cache (FILE_FLAG_NO_BUFFERING / O_DIRECT), `threads` at a time for
    `seconds` (idea N8: two same-size runs on the PC differed only by drive). -> GB/s, or None when unbuffered reads are not possible here."""
    import random
    size = os.path.getsize(path) // block * block
    if size < 64 * block:
        return None
    stop = time.time() + seconds
    done = [0] * threads
    def opener():
        if WINDOWS:
            import ctypes
            k = ctypes.windll.kernel32
            k.CreateFileW.restype = ctypes.c_void_p
            k.VirtualAlloc.restype = ctypes.c_void_p
            h = k.CreateFileW(path, 0x80000000, 7, None, 3, 0x20000000, None)        # GENERIC_READ, share all, OPEN_EXISTING, NO_BUFFERING
            if h in (None, ctypes.c_void_p(-1).value):
                return None
            buf = k.VirtualAlloc(None, block, 0x3000, 4)                              # page-aligned, as unbuffered reads need
            got = ctypes.c_uint32()
            def read(off):
                if not k.SetFilePointerEx(ctypes.c_void_p(h), ctypes.c_int64(off), None, 0):
                    return 0
                return got.value if k.ReadFile(ctypes.c_void_p(h), ctypes.c_void_p(buf), block, ctypes.byref(got), None) else 0
            def close():
                k.CloseHandle(ctypes.c_void_p(h)); k.VirtualFree(ctypes.c_void_p(buf), 0, 0x8000)
            return read, close
        import mmap
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        except OSError:
            return None
        buf = mmap.mmap(-1, block)                                                       # page-aligned, as O_DIRECT needs
        def read(off):
            try:
                return os.preadv(fd, [buf], off)
            except OSError:
                return 0
        return read, lambda: (os.close(fd), buf.close())
    def worker(i, rw):
        read, close = rw
        rnd = random.Random(i)
        try:
            while time.time() < stop:
                n = read(rnd.randrange(size // block) * block)
                if n <= 0:
                    break
                done[i] += n
        finally:
            close()
    rws = [opener() for _ in range(threads)]
    if not all(rws):
        for rw in rws:
            if rw:
                rw[1]()
        return None
    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(i, rw)) for i, rw in enumerate(rws)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return sum(done) / (time.time() - t0) / 1e9 if sum(done) else None


def win_memory():
    """(total, available) bytes on Windows (GlobalMemoryStatusEx); (None, None) elsewhere or on failure."""
    if not WINDOWS:
        return None, None
    try:
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return float(m.ullTotalPhys), float(m.ullAvailPhys)
    except Exception:
        pass
    return None, None


def win_commit():
    """(commit limit, commit still free) bytes on Windows: RAM + page file that programs may still be promised (GlobalMemoryStatusEx ullTotalPageFile /
    ullAvailPageFile); (None, None) elsewhere. When it runs out, allocations fail although RAM looks free: llama-server's prompt cache then failed
    ("bad allocation") and llama.cpp crashed (PC, RESULTS 73). Vulkan's GPU allocations count here; the plugin's file-view expert cache does not."""
    if not WINDOWS:
        return None, None
    try:
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return float(m.ullTotalPageFile), float(m.ullAvailPageFile)
    except Exception:
        pass
    return None, None


def vram_for_commit(vram_gib, commit_free, host_gib=4.0):
    """GPU memory (GiB) to plan when Vulkan's allocations count against Windows commit: at most the free commit minus llama-server's host side
    (prompt cache, buffers, the process: host_gib); never below 0. commit_free None (not Windows / unknown): vram_gib unchanged. Pure, tested."""
    if not commit_free:
        return vram_gib
    return max(0.0, min(vram_gib, commit_free / 2**30 - host_gib))


def commit_plan(commit_free, vram_gib, cram_mib, margin_gib=2.0):
    """How much prompt cache (MiB) fits in Windows' free commit next to the GPU allocations (Vulkan charges them to commit) and a margin:
    (cache-ram MiB to use, short: True when even with no prompt cache the commit is too tight). Pure, so it is tested without Windows."""
    need_base = (float(vram_gib or 0) + margin_gib) * 2**30
    room_mib = int(max(0.0, commit_free - need_base) / 2**20)
    return min(int(cram_mib), room_mib), commit_free < need_base


def win_dedicated_vram():
    """{adapter name: dedicated VRAM bytes} from the Windows registry (display adapter class); {} elsewhere or on failure."""
    out = {}
    if not WINDOWS:
        return out
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as k:
            for i in range(winreg.QueryInfoKey(k)[0]):
                sub = winreg.EnumKey(k, i)
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(k, sub) as d:
                        name = str(winreg.QueryValueEx(d, "DriverDesc")[0]).strip()
                        mem = winreg.QueryValueEx(d, "HardwareInformation.qwMemorySize")[0]
                        out[name] = mem if isinstance(mem, int) else int.from_bytes(bytes(mem), "little")
                except OSError:
                    pass
    except Exception:
        pass
    return out


def win_driver_version(name):
    """DriverVersion (e.g. "32.0.23034.4") of the display adapter called `name` in the Windows registry; "" elsewhere, when not found or on failure."""
    if not WINDOWS:
        return ""
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as k:
            for i in range(winreg.QueryInfoKey(k)[0]):
                sub = winreg.EnumKey(k, i)
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(k, sub) as d:
                        if str(winreg.QueryValueEx(d, "DriverDesc")[0]).strip() == name:
                            return str(winreg.QueryValueEx(d, "DriverVersion")[0]).strip()
                except OSError:
                    pass
    except Exception:
        pass
    return ""


def dedicated_for(name, ded):
    """DOCS-058 item 19: dedicated VRAM of the adapter `name` that llama.cpp lists, from win_dedicated_vram()'s {registry name: bytes}. Exact name first;
    otherwise the registry name and the listed name may differ in trademark signs, case or a suffix (the Vulkan name can carry the driver), so a name
    contained in the other (after dropping (R), (TM), (C) and extra blanks, at least 8 letters) counts, and the largest match wins. -1 = not found."""
    def norm(t):
        t = re.sub(r"\((?:r|tm|c)\)|[\u00ae\u2122\u00a9]", "", t, flags=re.I)
        return re.sub(r"\s+", " ", t).strip().lower()
    if name in ded:
        return ded[name]
    n = norm(name)
    hits = [v for k, v in ded.items() if len(norm(k)) >= 8 and len(n) >= 8 and (norm(k) in n or n in norm(k))]
    return max(hits) if hits else -1


def dedicated_vram_of(name, ded):
    """Dedicated VRAM in bytes of the adapter `name` (as llama.cpp lists it) from win_dedicated_vram()'s {registry name: bytes}, matched like pick_gpu_full does (dedicated_for: the
    two names may differ in trademark signs, case or a driver suffix); None when the adapter is not in the registry list (LAUNCHER-991 review: the integrated-GPU checks looked the
    name up exactly, so 'AMD Radeon Graphics (RADV ...)' was not found and an iGPU's shared RAM was planned as VRAM)."""
    d = dedicated_for(name, ded or {})
    return None if d == -1 else d


def pick_gpu(listing):
    g = pick_gpu_full(listing)
    return g[:3] if g else None


def pick_gpu_full(listing):
    """(device, name, free MiB, total MiB) of the GPU llama.cpp will use, from `llama-server --list-devices` output; None without a GPU.
    llama.cpp skips an integrated GPU when a discrete one exists, but --list-devices lists both (on a Ryzen 7000 desktop the built-in Radeon
    comes first and reports shared RAM as its memory). Windows: the adapter with the most dedicated VRAM in the registry wins; else the most free memory."""
    devs = re.findall(r"^\s+((?:CUDA|Vulkan|ROCm|HIP|SYCL|Metal)\d*): (.+?) \((\d+) MiB, (\d+) MiB free\)", listing, re.M)
    if not devs:
        m = re.search(r"^\s+((?:CUDA|Vulkan|ROCm|HIP|SYCL|Metal)\d*): (.+)$", listing, re.M)     # a device line without memory figures
        return (m.group(1), m.group(2).strip(), 0, 0) if m else None
    ded = win_dedicated_vram()
    d = max(devs, key=lambda x: (not IGPU_NAME_RE.search(x[1].strip()), dedicated_for(x[1].strip(), ded), int(x[3])))      # LAUNCHER-983: a card that is not integrated (by name) first, then dedicated memory, then free memory
    return d[0], d[1].strip(), int(d[3]), int(d[2])


def cpu_only_decision(forced, extra, listing, vram_given=False):
    """LINUX-011: -> (cpu_only, why). CPU-only means llama-server is told `--device none` and `-ngl 0`: with -ngl 0 alone llama.cpp can still put the KV
    cache on a GPU backend it loaded (an upstream report: CUDA out-of-memory at 256k on a machine used CPU-only). True when --cpu-only was given, or when
    `llama-server --list-devices` answered (it printed "Available devices") and lists no GPU. The user's own --device / -dev after -- and a given --vram
    win; an empty or failed listing never switches anything (we do not know there is no GPU)."""
    if any(x in ("--device", "-dev") or x.startswith("--device=") for x in extra):
        return False, None
    if forced:
        return True, "--cpu-only"
    if vram_given or not listing or "Available devices" not in listing:
        return False, None
    if pick_gpu(listing) is None:
        return True, "llama-server lists no GPU device"
    return False, None


def nvidia_reports():
    """True when nvidia-smi exists and reports at least one GPU's free memory."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        return out.returncode == 0 and out.stdout.strip()[:1].isdigit()
    except Exception:
        return False

def vk_tensor_limits():
    """[(device name, largest tensor in bytes)] for each Vulkan GPU, read from the Vulkan loader (vulkan-1.dll / libvulkan.so.1) as ggml-vulkan
    reads it: the largest buffer (maxBufferSize with maintenance4, else maxMemoryAllocationSize) and maxStorageBufferRange. ggml-vulkan's
    supports_op rejects a bigger tensor (the range is skipped only with VK_EXT_shader_64bit_indexing, which depends on how llama.cpp was built,
    so both count here). GGML_VK_FORCE_MAX_BUFFER_SIZE overrides the buffer size as in ggml. [] when there is no Vulkan loader or GPU."""
    import ctypes
    try:
        vk = ctypes.WinDLL("vulkan-1.dll") if WINDOWS else ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return []
    VP, U32, U64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64

    class AppInfo(ctypes.Structure):
        _fields_ = [("sType", U32), ("pNext", VP), ("pApplicationName", ctypes.c_char_p), ("applicationVersion", U32),
                    ("pEngineName", ctypes.c_char_p), ("engineVersion", U32), ("apiVersion", U32)]

    class InstInfo(ctypes.Structure):
        _fields_ = [("sType", U32), ("pNext", VP), ("flags", U32), ("pApplicationInfo", ctypes.POINTER(AppInfo)), ("enabledLayerCount", U32),
                    ("ppEnabledLayerNames", VP), ("enabledExtensionCount", U32), ("ppEnabledExtensionNames", VP)]

    class Maint(ctypes.Structure):          # VkPhysicalDeviceMaintenance3Properties / ...Maintenance4Properties share the head
        _fields_ = [("sType", U32), ("pNext", VP), ("a", U32), ("b", U64)]

    class Ext(ctypes.Structure):
        _fields_ = [("extensionName", ctypes.c_char * 256), ("specVersion", U32)]

    vk.vkGetInstanceProcAddr.restype = VP
    vk.vkGetInstanceProcAddr.argtypes = [VP, ctypes.c_char_p]
    app = AppInfo(0, None, b"moe-cache", 1, b"moe-cache", 1, (1 << 22) | (1 << 12))      # VK_API_VERSION_1_1
    info = InstInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    inst = VP()
    create = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(InstInfo), VP, ctypes.POINTER(VP))(vk.vkGetInstanceProcAddr(None, b"vkCreateInstance"))
    if create(ctypes.byref(info), None, ctypes.byref(inst)) != 0:
        return []
    fn = lambda name, *args: ctypes.CFUNCTYPE(ctypes.c_int, *args)(vk.vkGetInstanceProcAddr(inst, name))
    out = []
    try:
        enum = fn(b"vkEnumeratePhysicalDevices", VP, ctypes.POINTER(U32), VP)
        props2 = fn(b"vkGetPhysicalDeviceProperties2", VP, VP)
        exts = fn(b"vkEnumerateDeviceExtensionProperties", VP, ctypes.c_char_p, ctypes.POINTER(U32), VP)
        destroy = ctypes.CFUNCTYPE(None, VP, VP)(vk.vkGetInstanceProcAddr(inst, b"vkDestroyInstance"))
        n = U32(0)
        enum(inst, ctypes.byref(n), None)
        devs = (VP * n.value)()
        enum(inst, ctypes.byref(n), devs)
        for d in devs[:n.value]:
            ne = U32(0)
            exts(d, None, ctypes.byref(ne), None)
            el = (Ext * ne.value)()
            exts(d, None, ctypes.byref(ne), el)
            names = {e.extensionName.decode(errors="replace") for e in el[:ne.value]}
            m4, m3 = Maint(1000413001, None), Maint(1000168000, None)        # MAINTENANCE_4_PROPERTIES, MAINTENANCE_3_PROPERTIES
            # VkPhysicalDeviceProperties2: sType, pNext, then VkPhysicalDeviceProperties (824 bytes) at offset 16
            buf = (ctypes.c_ubyte * 1024)()
            ctypes.memmove(buf, ctypes.byref(U32(1000059001)), 4)                # PHYSICAL_DEVICE_PROPERTIES_2
            ctypes.memmove(ctypes.addressof(buf) + 8, ctypes.byref(VP(ctypes.addressof(m3))), ctypes.sizeof(VP))
            props2(d, buf)                     # first without maintenance4: it may only be chained when the device has it
            api = U32.from_buffer(buf, 16).value
            if api >= (1 << 22) | (3 << 12) or "VK_KHR_maintenance4" in names:
                m3.pNext = ctypes.addressof(m4)
                props2(d, buf)
            dev_type = U32.from_buffer(buf, 16 + 16).value                 # VkPhysicalDeviceType: 1 integrated, 2 discrete, 3 virtual, 4 CPU
            name = bytes(buf[36:36 + 256]).split(b"\0")[0].decode(errors="replace")
            ssbo = U32.from_buffer(buf, 16 + 296 + 28).value                 # limits (offset 296) .maxStorageBufferRange (8th uint32)
            big = int(os.environ["GGML_VK_FORCE_MAX_BUFFER_SIZE"]) if os.environ.get("GGML_VK_FORCE_MAX_BUFFER_SIZE", "").isdigit() else \
                (m4.b if m4.b else m3.b)
            if dev_type in (1, 2, 3) and big and ssbo:
                out.append((name, min(big, ssbo)))
        destroy(inst, None)
    except Exception:
        pass
    return out


def vk_cache_cap(ls, model, listing=None):
    """(largest --moe-cache-mib that loads, device tensor limit in bytes) when llama-server runs the GPU expert cache on a Vulkan device, else
    (None, None). llama.cpp keeps each expert tensor's cache slots in one bank tensor and Vulkan refuses a tensor over the device's limit, which
    aborts the load (T5: 8909 MiB aborted on an RX 9070 XT, 5900 loaded). The limit is read from the device, in a child process (a Vulkan driver
    that crashes cannot take the launcher with it); same rule on Linux and Windows. The smallest limit of the Vulkan GPUs llama-server lists counts
    (llama.cpp may share the cache among them)."""
    vkdevs = [n.strip() for n in re.findall(r"^\s+Vulkan\d+: (.+?) \(\d+ MiB", listing or list_devices(ls), re.M)]
    if not vkdevs:
        return None, None
    try:
        r = subprocess.run([python_cmd(), "-c", "import json, sys; sys.path.insert(0, sys.argv[1]); import moe_cache as m; print(json.dumps(m.vk_tensor_limits()))",
                            os.path.join(HERE, "bin")], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                           creationflags=0x08000000 if WINDOWS else 0)          # CREATE_NO_WINDOW: no console flash under pythonw
        lim = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        lim = []
    match = [b for n, b in lim if n.strip() in vkdevs] or [b for n, b in lim]     # the listed devices by name, else all
    if not match:
        return None, None
    sys.path.insert(0, os.path.join(HERE, "planner"))
    from gguf_plan import gpu_cache_cap_mib
    return gpu_cache_cap_mib(model, min(match)), min(match)


def env_has(env, name):
    return any(k == name for k, _ in env)


def sycl_sysman_env(ls, listing=None):
    """ENGINE-048: [("ZES_ENABLE_SYSMAN", "1")] when llama-server uses a SYCL device (its --list-devices output has a SYCL line, or the ggml-sycl library
    sits next to it): level-zero then answers the free-memory query and stops warning on every call. Empty when the user set the variable themselves."""
    if "ZES_ENABLE_SYSMAN" in os.environ:
        return []
    if listing and re.search(r"^\s+SYCL\d*:", listing, re.M):
        return [("ZES_ENABLE_SYSMAN", "1")]
    try:
        names = os.listdir(os.path.dirname(os.path.abspath(ls)))
    except OSError:
        return []
    return [("ZES_ENABLE_SYSMAN", "1")] if any(n.lower() in ("libggml-sycl.so", "ggml-sycl.dll") for n in names) else []


def hip_missing_hint(ls, listing, windows=None, find=None, names=None):
    """LINUX-018: a sentence when the ggml-hip backend sits next to llama-server (ggml-hip.dll / libggml-hip.so) but --list-devices shows no HIP / ROCm
    device and the HIP runtime libraries it needs are not found: hipblas / rocblas (Windows: *.dll on PATH or next to llama-server; Linux: libhipblas.so /
    librocblas.so in the library path, /opt/rocm/lib or next to llama-server). None when there is no ggml-hip, a HIP device is listed, or nothing is missing.
    find(name) -> bool and names (the files next to llama-server) are for tests."""
    windows = WINDOWS if windows is None else windows
    if re.search(r"^\s+(HIP|ROCm)\d*:", listing or "", re.M):
        return None
    d = os.path.dirname(os.path.abspath(ls))
    if names is None:
        try:
            names = os.listdir(d)
        except OSError:
            return None
    low = {n.lower() for n in names}
    if not low & {"ggml-hip.dll", "libggml-hip.so"}:
        return None
    if find is None:
        def find(n):
            if n.lower() in low or shutil.which(n):
                return True
            if windows:
                return bool(os.environ.get("HIP_PATH")) and os.path.isfile(os.path.join(os.environ["HIP_PATH"], "bin", n))
            import glob
            return any(glob.glob(os.path.join(p, n + "*")) for p in ("/opt/rocm/lib", "/opt/rocm/lib64", "/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu",
                                                                     *[x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":") if x]))
    want = ["hipblas.dll", "rocblas.dll"] if windows else ["libhipblas.so", "librocblas.so"]
    missing = [n for n in want if not find(n)]
    if not missing:
        return ("the HIP backend (" + ("ggml-hip.dll" if windows else "libggml-hip.so") + ") is next to llama-server but lists no AMD device: the GPU driver or ROCm "
                "does not see a supported card (rocminfo / hipInfo shows what ROCm sees)")
    return (f"the HIP backend ({'ggml-hip.dll' if windows else 'libggml-hip.so'}) is next to llama-server but no AMD device is listed, and {', '.join(missing)} "
            + ("is not on PATH (add the ROCm / HIP SDK bin folder, e.g. C:\\Program Files\\AMD\\ROCm\\<version>\\bin, or copy the DLLs next to llama-server.exe)"
               if windows else "was not found (install the ROCm runtime: hipblas and rocblas; or add /opt/rocm/lib to LD_LIBRARY_PATH)")
            + ": llama.cpp then loads the backend without a device")


def list_devices(ls):
    try:
        out = subprocess.run([ls, "--list-devices"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                             env=dict(os.environ, **dict(sycl_sysman_env(ls))))
        return out.stdout + out.stderr
    except Exception:
        return ""

def parse_core_records(buf, groups=None):
    """Cores from a GetLogicalProcessorInformationEx(RelationProcessorCore) buffer: [(efficiency_class, [logical CPUs])], processor group 0 only
    (masks, llama.cpp's -C included, address group 0; the groups seen are added to the set `groups` when one is given).
    Record layout: DWORD Relationship, DWORD Size, then PROCESSOR_RELATIONSHIP: BYTE Flags, BYTE EfficiencyClass, BYTE Reserved[20],
    WORD GroupCount, GROUP_AFFINITY GroupMask[] at offset 32 (KAFFINITY Mask, WORD Group, WORD Reserved[3])."""
    import struct
    cores, off = [], 0
    while off + 48 <= len(buf):
        rel, size = struct.unpack_from("<II", buf, off)
        if size < 48:
            break
        if rel == 0:
            eff = buf[off + 9]
            mask, group = struct.unpack_from("<QH", buf, off + 32)
            if groups is not None:
                groups.add(group)
            if group == 0:
                cores.append((eff, [i for i in range(64) if mask >> i & 1]))
        off += size
    return cores


def hybrid_masks(cores):
    """(performance-core mask with one thread per core, efficiency CPUs) when the cores have more than one efficiency class, else None.
    Windows gives the performance cores the HIGHER efficiency class."""
    classes = {e for e, _ in cores}
    if len(classes) < 2:
        return None
    top = max(classes)
    pmask = sum(1 << cpus[0] for e, cpus in cores if e == top and cpus)
    ecpus = sorted(c for e, cpus in cores if e != top for c in cpus)
    return pmask, ecpus


WIN_GROUPS_NOTE = "more than 64 logical CPUs (several Windows processor groups): no core pinning (masks reach group 0 only)"


def win_cores():
    """[(efficiency_class, [logical CPUs])] on Windows, [] elsewhere or on failure (Claw finding 5: Windows had no core pinning).
    [] too with more than 64 logical CPUs (several processor groups): a mask reaches group 0 only, so pinning would leave the other groups idle."""
    if not WINDOWS:
        return []
    try:
        import ctypes
        k = ctypes.windll.kernel32
        n = ctypes.c_uint32(0)
        k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
            return []
        groups = set()
        cores = parse_core_records(buf.raw[:n.value], groups)
        win_cores.groups = len(groups)
        return [] if len(groups) > 1 else cores
    except Exception:
        return []


def physical_cores():
    """Physical CPU cores: Windows from the core records (win_cores), Linux from the distinct (package, core id) pairs in /proc/cpuinfo;
    half the logical CPUs when neither answers (SMT is the common case)."""
    if WINDOWS:
        c = win_cores()
        if c:
            return len(c)
    else:
        try:
            seen, pkg = set(), "0"
            for line in open("/proc/cpuinfo"):
                k, _, v = line.partition(":")
                k = k.strip()
                if k == "physical id":
                    pkg = v.strip()
                elif k == "core id":
                    seen.add((pkg, v.strip()))
            if seen:
                return len(seen)
        except OSError:
            pass
    return max(1, (os.cpu_count() or 2) // 2)


# --leave-room (owner 2026-10-30: a big-model test made the desktop PC stutter while the owner used it; a friend using the PC while a model runs hits
# the same): more RAM, GPU memory and CPU left to the user's own programs, the server at a lower priority. Same rule on Linux and Windows.
LEAVE_ROOM_KEEP_GIB, LEAVE_ROOM_KEEP_FRAC = 8, 0.25     # RAM kept free: max(8 GiB, 25 %) instead of the normal cap's max(4 GiB, 12.5 %): 24 GiB of 32
LEAVE_ROOM_VRAM_GB = 2.0                                # GPU memory kept free for the desktop and browser, at least (normal: about 0.5 GB)
LEAVE_ROOM_FREE_CORES = 2                               # physical cores the server's threads leave free


def leave_room_ram_gb(machine):
    """--leave-room's --ram (decimal GB, as text) when none was given: total RAM minus max(8 GiB, 25 %), and never above what may be used now. None if unknown."""
    total = machine.mem_total()
    if not total:
        return None
    keep = max(LEAVE_ROOM_KEEP_GIB * 2**30, total * LEAVE_ROOM_KEEP_FRAC)
    budget = min(total - keep if total > keep else total / 2, machine.mem_for_server(None) or float("inf"))
    return f"{math.floor(budget / 1e8) / 10:.1f}"


def mem_total():
    if WINDOWS:
        return win_memory()[0]
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def ram_gib(ram_gb):
    """--ram is GB (1e9 bytes, as the Windows job limit and mem_for_server use it); the plugin's MOE_CACHE_RAM_GIB and the planner's --ram-gib are GiB.
    PC 2026-10-27: passing --ram 20 unchanged planned 21.5 GB under a 20 GB job limit, the commit failed (error 1455) and the server died."""
    return f"{float(ram_gb) * 1e9 / 2**30:.3f}"


def leading_number(v):
    """LAUNCHER-005 (A3-22): the leading decimal number of a text, the way the plugin's strtod reads MOE_CACHE_MAX_RAM_GB ("20GB" is 20 there, so it must be 20
    here too, or the two sides plan with different caps). Raises ValueError when the text does not start with a number (hex, inf and nan count as typos here)."""
    m = re.match(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", str(v))
    if not m or re.match(r"\s*[+-]?0[xX]", str(v)):
        raise ValueError(v)
    return m.group(1)


def write_text_atomic(path, text):
    """LAUNCHER-005 (A3-26): the whole text appears under `path` or nothing does (a crash must not leave a truncated note that note_of() then misreads)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def mem_cap():
    """Hard cap on the whole server's memory, bytes (owner 2026-10-29): physical RAM minus max(4 GiB, 12.5 %) = 28 GiB on a 32 GiB machine, so
    Windows does not run out of memory and Linux does not OOM; MOE_CACHE_MAX_RAM_GB (decimal GB; --max-ram sets it) overrides, 0 = no cap.
    The plugin applies the same rule (src/mc_common.inl ram_cap). None when off or unknown."""
    v = getattr(_TL, "max_ram", None)
    if v is None:
        v = os.environ.get("MOE_CACHE_MAX_RAM_GB")
    if v is not None:
        try:
            f = float(leading_number(v))
            if f == f and f >= 0:            # not nan
                return f * 1e9 if f > 0 else None
        except ValueError:
            pass
        if not getattr(mem_cap, "warned", False):    # a typo kept the default cap, as the plugin does (audit 2026-10-29 L6: it dropped the cap)
            mem_cap.warned = True
            say(f"moe-cache: MOE_CACHE_MAX_RAM_GB={v!r} is not a number of GB: the default cap is used", err=True)
    total = mem_total()
    if not total:
        return None
    keep = max(4 * 2**30, total / 8)
    return total - keep if total > keep else total / 2


def mem_for_server(ram_gb):
    """Bytes the server may use: --ram (GB), else the cgroup limit, else available RAM; never above mem_cap(); None when unknown."""
    cap = mem_cap()
    v = _mem_for_server(ram_gb)
    return min(v, cap) if v and cap else (v or cap)


def cgroup_mem_limit(proc_cgroup="/proc/self/cgroup", root="/sys/fs/cgroup"):
    """Smallest memory limit in bytes of this process's cgroup AND its parents (a limit set on a systemd slice or a docker parent applies to a leaf that
    says "max"), cgroup v2 memory.max and memory.high (systemd's MemoryMax and MemoryHigh; SERVER-005: the cache is sized to stay below the soft limit too, or
    the kernel starts reclaiming it), cgroup v1 memory.limit_in_bytes; None when nothing limits it. The paths are arguments for tests/cgroup_test.py."""
    try:
        lines = open(proc_cgroup).read().strip().splitlines()
    except OSError:
        return None
    best = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        hier, ctrl, rel = parts
        if hier == "0" and ctrl == "":
            base, names = root, ("memory.max", "memory.high")
        elif "memory" in ctrl.split(","):
            base, names = os.path.join(root, "memory"), ("memory.limit_in_bytes",)
        else:
            continue
        d = rel or "/"
        while True:
            for name in names:
                try:
                    v = open(os.path.join(base, d.lstrip("/"), name)).read().strip()
                except OSError:
                    v = ""
                if v.isdigit() and int(v) < 2**60:            # v1 says 9223372036854771712 for "no limit"; "max" is not a number
                    best = int(v) if best is None else min(best, int(v))
            if d in ("/", ""):
                break
            d = os.path.dirname(d.rstrip("/")) or "/"
    return best


def _mem_for_server(ram_gb):
    if ram_gb:
        return float(ram_gb) * 1e9
    if WINDOWS:
        return win_memory()[1]      # found on the MSI Claw: /proc does not exist there, so stock-vs-plugin could not be decided without --ram
    lim = cgroup_mem_limit()
    if lim:
        return float(lim)
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024.0
    except (OSError, ValueError):
        pass
    return None


def parse_cpus(text):
    out = []
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def arg_after(args, *names):
    v = None
    for i, a in enumerate(args):
        if a in names and i + 1 < len(args):
            v = args[i + 1]
    return v


def adopt_user_flags(extra):
    """Flags the launcher also sets, when the user gives them after `--`: llama-server takes the user's (the last one wins), so the plan, the health
    checks and the VRAM maths must use the same value. -> ({"port": int, "ctx": int, "ncmoe": str} for the ones found as plain numbers, extra without
    them: the launcher puts the value back itself). A value that is not a plain positive number (-c 0 = the model's own length) is left alone."""
    found, keep, i = {}, [], 0
    names = {"--port": "port", "-c": "ctx", "--ctx-size": "ctx", "--n-cpu-moe": "ncmoe", "-ncmoe": "ncmoe"}
    while i < len(extra):
        key = names.get(extra[i])
        if key and i + 1 < len(extra) and re.fullmatch(r"[0-9]+", extra[i + 1]) and (int(extra[i + 1]) > 0 or key == "ncmoe"):
            found[key] = int(extra[i + 1]) if key != "ncmoe" else str(int(extra[i + 1]))
            i += 2
            continue
        keep.append(extra[i])
        i += 1
    return found, keep


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib copies the Authorization header (our API key) onto a redirect to another host; a redirect is returned as the answer instead."""
    def redirect_request(self, *a, **k):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)
# A request to this machine never goes through a proxy (http_proxy / the Windows registry proxy would send /health and the API key to it; 127.0.0.1 is
# not bypassed unless no_proxy says so). Remote URLs keep the system proxy settings.
_OPENER_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def http(url, body=None, api_key=None, timeout=10):
    """(status, text) of a request; status 0 when the server cannot be reached. Redirects are not followed (they would carry the API key along)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, headers)
    try:
        with (_OPENER_DIRECT if loopback_url(url) else _OPENER).open(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception:
        return 0, ""


KEY_FILE = os.path.join(CACHE_DIR, "api.key")


def model_identity(model, ctx=None, ctk=None, ctv=None):
    """The key of a saved prompt state (warm sessions, --project slots): bin/moe_cache_model.py slot_id, the one place that says what a model is (XA-1). The string is
    the same as before that module existed, so states saved earlier are still found. "" when the file cannot be read."""
    sys.path.insert(0, os.path.join(HERE, "bin"))
    from moe_cache_model import slot_id
    return slot_id(model, ctx, ctk, ctv)


def write_server_id(slotdir, model, ctx, cmd):
    """who runs on this project folder now: `project save` stamps the saved file with server.id, a restore requires the same. Written from the FINAL command line (after
    the long-prompt stage may have undone the q8_0 KV cache) and again when the f16 KV retry changes it; a KV type that differs from the saved state's gives another id."""
    try:
        write_text_atomic(os.path.join(slotdir, "server.id"),
                          model_identity(model, ctx, arg_after(cmd, "-ctk", "--cache-type-k"), arg_after(cmd, "-ctv", "--cache-type-v")) + "\n")
    except OSError:
        pass


def model_profile(model, name, dry=False):
    """The usage profile file of this model (bin/moe_cache_model.py profile_path): the old <file>.usage unless the profile there was learned on another file."""
    sys.path.insert(0, os.path.join(HERE, "bin"))
    from moe_cache_model import profile_path
    return profile_path(model, name, CACHE_DIR, dry=dry, say=say)


def project_id_ok(slotdir, name):
    """True when the saved project file `name`.bin was written by a server with the model identity now recorded in server.id; files without an identity are not trusted."""
    try:
        want = open(os.path.join(slotdir, "server.id")).read().strip()
        have = open(os.path.join(slotdir, name + ".bin.id")).read().strip()
    except OSError:
        return False
    return bool(want) and want == have


def _read_key(wait=0.0):
    """The stored key, "" when there is none; waits up to `wait` seconds for a writer that has created the file but not finished it."""
    end = time.time() + wait
    while True:
        try:
            k = open(KEY_FILE).read().strip()
            if k:
                return k
        except OSError:
            pass
        if time.time() >= end:
            return ""
        time.sleep(0.02)


def stored_key(create=False):
    """The stable per-user API key kept in the cache folder (mode 0600 on Linux; on Windows the user profile is private); "" when there is none yet.
    create=True makes one on first use (secrets.token_urlsafe(24)). Agent mode and --project servers use it, so a web page in the browser cannot
    use or read the server's slots and conversations (RESULTS/SECURITY: DECISION_LLAMA_SERVER_KEY, option C).
    The key is written completely to a private temp file and then published under its final name with os.link, which fails if the name exists:
    exactly one launcher wins, every other one reads the winner's key, and nobody ever sees a half-written file."""
    k = _read_key()
    if k or not create:
        return k
    import secrets, tempfile
    k = secrets.token_urlsafe(24)
    while k.startswith("-"):              # a key starting with "-" breaks `connect --api-key KEY` (argparse reads it as an option)
        k = secrets.token_urlsafe(24)
    os.makedirs(os.path.dirname(KEY_FILE), mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="api.key.", suffix=".tmp", dir=os.path.dirname(KEY_FILE))      # mkstemp: mode 0600
    try:
        with os.fdopen(fd, "w") as f:
            f.write(k + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, KEY_FILE)
            return k
        except FileExistsError:
            w = _read_key(5.0)
            if w:
                return w
            os.replace(tmp, KEY_FILE)         # an empty leftover (a launcher that died between creating and writing): replace it with the complete file
            return k
        except OSError:                   # a file system without hard links: create the final name exclusively; readers wait for the content
            try:
                fd2 = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                return _read_key(5.0) or k
            with os.fdopen(fd2, "w") as f:
                f.write(k + "\n")
            return k
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def loopback_url(url):
    """True when the URL's host is localhost or a loopback address (127.0.0.0/8, ::1), compared exactly."""
    import ipaddress
    try:
        h = (urllib.parse.urlsplit(url if "://" in url else "http://" + url).hostname or "").lower()
    except ValueError:
        return False
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


SERVER_DIR = os.path.join(CACHE_DIR, "servers")


def pid_alive(pid):
    if not pid or pid < 1:
        return False
    if WINDOWS:
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, pid)
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return bool(ok) and code.value == 259          # STILL_ACTIVE
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True            # EPERM: it exists
    return True


def pid_start(pid):
    """An opaque token for when process `pid` started ('' when it cannot be read): a record that names a pid also names this token, so a pid the system
    handed to another program later (pid reuse) is not taken for the process that wrote the record."""
    try:
        if WINDOWS:
            import ctypes
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, int(pid))
            if not h:
                return ""
            t = [wintypes.FILETIME() for _ in range(4)]
            ok = k.GetProcessTimes(h, *[ctypes.byref(x) for x in t])
            k.CloseHandle(h)
            return f"{t[0].dwHighDateTime}.{t[0].dwLowDateTime}" if ok else ""
        if sys.platform.startswith("linux"):
            return open(f"/proc/{int(pid)}/stat").read().rsplit(")", 1)[1].split()[19]        # field 22: start time in clock ticks since boot
        return subprocess.run(["ps", "-o", "lstart=", "-p", str(int(pid))], capture_output=True, text=True, timeout=3).stdout.strip().replace(" ", "_")
    except Exception:
        return ""


def pid_is(pid, start):
    """Is `pid` alive and, when the record gave a start token, still the process that started then?"""
    if not pid_alive(pid):
        return False
    return not start or pid_start(pid) == start


def _url_port(url):
    try:
        return urllib.parse.urlsplit(url if "://" in url else "http://" + url).port
    except ValueError:
        return None


class OwnsPort:
    """While this launcher runs llama-server on `port`, a record in the cache folder says so (its own pid): launcher_owns() is how another moe-cache command
    (connect, project save / restore, the launch ready check) knows a loopback port is the launcher's own server and may be sent the stored API key (SERVER-692)."""
    def __init__(self, port):
        self.path = os.path.join(SERVER_DIR, f"{int(port)}.pid") if port else None

    def __enter__(self):
        if self.path:
            try:
                os.makedirs(SERVER_DIR, mode=0o700, exist_ok=True)
                with open(self.path, "w") as f:
                    f.write(f"{os.getpid()} {pid_start(os.getpid())}".strip())
            except OSError:
                self.path = None
        return self

    def __exit__(self, *a):
        if self.path:
            try:
                if open(self.path).read().split()[0] == str(os.getpid()):
                    os.remove(self.path)
            except OSError:
                pass


def launcher_owns(url):
    """Is `url` a loopback port that a moe-cache launcher of this user is running llama-server on right now (its record names a live pid), or this
    user's on-demand listener (its pid file)? Only then is the stored API key ours to send there."""
    if not loopback_url(url):
        return False
    port = _url_port(url)
    if not port:
        return False
    try:
        rec = open(os.path.join(SERVER_DIR, f"{port}.pid")).read().split()
        if pid_is(int(rec[0]), rec[1] if len(rec) > 1 else ""):
            return True
    except (OSError, ValueError, IndexError):
        pass
    try:
        rec = open(os.path.join(CACHE_DIR, "on-demand.pid")).read().split()
        return int(rec[1]) == port and pid_is(int(rec[0]), rec[2] if len(rec) > 2 else "")
    except (OSError, ValueError, IndexError):
        return False


def key_for(url, given=None, trusted=False):
    """The key to send to the server at `url`: the given one, else the stored one only when `trusted` (the server is one this launcher started itself:
    launcher_owns(url)) and it answers 401/403 without a key. Any other server, however it answers, never gets the stored key: any local program can
    listen on a free port, answer 200 and then 401, and collect it (SERVER-692). A remote URL never does either."""
    if given:
        return given
    if not trusted or not loopback_url(url):
        return None
    st = http(url.rstrip("/") + "/v1/models", timeout=3)[0]
    return stored_key() if st in (401, 403) else None


def gb_arg(s):
    """argparse check for --ram / --vram: a positive finite number (kept as the text given, which the rest of the launcher passes on)."""
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {s!r}")
    if not (0 < v < 1e6):
        raise argparse.ArgumentTypeError(f"out of range: {s!r}")
    return s


def take_api_key_from_args(extra):
    """Remove a user-given `--api-key KEY` from llama-server's arguments (it would show in the process list) and return KEY, else None."""
    for i, x in enumerate(extra):
        if x == "--api-key" and i + 1 < len(extra):
            k = extra[i + 1]
            del extra[i:i + 2]
            return k
        if x.startswith("--api-key="):
            del extra[i]
            return x.split("=", 1)[1]
    return None


def healthy(port, api_key=None):
    return http(f"http://127.0.0.1:{port}/health", api_key=api_key, timeout=2)[0] == 200


def port_busy(port):
    # a listening local port answers in well under 1 ms; a free one is refused at once on Linux, but Windows retries the refused connect until the
    # timeout (1 s with timeout 1, which was most of the on-demand listener's start on the PC)
    with socket.socket() as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0


def python_cmd():
    return os.environ.get("MOE_CACHE_PYTHON") or sys.executable or "python3"


_JOB = None   # Windows job object: kept open for the launcher's lifetime; closing it (the launcher exits or is killed) kills llama-server
_JOBS = []    # [(job handle, child)] of every job made, so the handle of a job whose child has ended can be closed (A3-20: one leaked per model start)


def prune_jobs(jobs, close):
    """Closes the handles of jobs whose child has ended (the job is empty, closing it kills nothing) and drops them from `jobs`; returns how many."""
    n = 0
    for item in list(jobs):
        job, child = item
        if child.poll() is not None:
            try:
                close(job)
            except Exception:
                pass
            jobs.remove(item)
            n += 1
    return n


def _kill_with_parent_windows(p, mem_limit=None):
    """Put the child in a job object that kills it when the launcher's handle closes (the launcher exits, is closed or killed).
    Found on the MSI Claw (2026-10-15): without it a killed launcher left llama-server running on its port.
    mem_limit (bytes, task WP3): also cap the job's committed memory, the Windows counterpart of the cgroup limit used on Linux. It counts the plugin's
    cache and llama-server's own allocations; it does not count the file cache behind a memory-mapped model, so stock llama-server is never limited
    by it (compare stock and the plugin on Windows with a model bigger than the real RAM instead). Returns the limit read back, or None."""
    global _JOB
    import ctypes
    from ctypes import wintypes
    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]
    class IOC(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]
    class EXT(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IOC), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    prune_jobs(_JOBS, k32.CloseHandle)
    job = k32.CreateJobObjectW(None, None)
    info = EXT()
    info.Basic.LimitFlags = 0x2000           # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if mem_limit:
        info.Basic.LimitFlags |= 0x200       # JOB_OBJECT_LIMIT_JOB_MEMORY
        info.JobMemoryLimit = int(mem_limit)
    ok = job and k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)) and \
        k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(p._handle)))
    if ok:
        _JOB = job
        _JOBS.append((job, p))
    else:
        say("moe-cache: warning: could not tie llama-server to the launcher (closing this window may leave it running)", err=True)
        return None
    if mem_limit:
        back = EXT()
        if k32.QueryInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(back), ctypes.sizeof(back), None) and back.Basic.LimitFlags & 0x200:
            say(f"moe-cache: Windows job memory limit {back.JobMemoryLimit / 1e9:.1f} GB on llama-server (--ram; the model's file cache is not counted)", err=True)
            return back.JobMemoryLimit
        say("moe-cache: warning: the Windows job memory limit could not be set; the cache still sizes itself to --ram", err=True)
    return None


_PRCTL = None
if sys.platform.startswith("linux"):
    try:                       # looked up here, not in the forked child: imports and dlopen after fork() can deadlock in a threaded parent (the UI)
        import ctypes
        _PRCTL = ctypes.CDLL("libc.so.6", use_errno=True).prctl
    except Exception:
        _PRCTL = None


def _child_low_linux():
    """preexec_fn for --leave-room: as _die_with_parent_linux, at nice 10."""
    _die_with_parent_linux()
    try:
        os.nice(10)
    except OSError:
        pass


def _die_with_parent_linux():
    """preexec_fn: the child gets SIGTERM when the launcher dies, even if the launcher is killed with SIGKILL."""
    if _PRCTL is not None:
        try:
            _PRCTL(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
        except Exception:
            pass


AMD_CHECKPOINT_FIXED = "32.0.32015.7029"       # first AMD Windows display driver known to survive a Flash-Next context checkpoint (I44, 2026-10-27)


def driver_older(ver, fixed=AMD_CHECKPOINT_FIXED):
    """True when the dotted driver version `ver` is older than `fixed` (compared as integers, missing parts count as 0). An unknown or unparseable
    version counts as older: the workaround is the safe side."""
    try:
        a = [int(x) for x in ver.strip().split(".")]
        f = [int(x) for x in fixed.split(".")]
    except (ValueError, AttributeError):
        return True
    if not a:
        return True
    n = max(len(a), len(f))
    return a + [0] * (n - len(a)) < f + [0] * (n - len(f))


def amd_windows_checkpoint_args(gpu, model, extra, agent=False):
    """["--ctx-checkpoints", "0"] for Flash-Next (qwen4exp) on an AMD GPU through Vulkan on Windows, else []. Desktop PC 2026-10-26: llama-server
    (stock too) crashes in AMD's Windows Vulkan driver (amdvlk64.dll, access violation) when it saves a context checkpoint at the end of a Flash-Next
    prompt; --ctx-checkpoints 0 avoids it (Qwen3.6 / 3.5-122B are not affected, nor is Linux/RADV). Driver 32.0.32015.7029 (2026-10-27) does not crash: the
    workaround applies only to older drivers, and to an unknown version (safe side)."""
    if not (WINDOWS and gpu and gpu[0].startswith("Vulkan") and re.search(r"AMD|Radeon", gpu[1])):
        return []
    if arg_after(extra, "-ctxcp", "--ctx-checkpoints", "--swa-checkpoints") is not None:
        return []
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import model_key
        if not (model_key(model) or "").startswith("qwen4exp"):
            return []
    except Exception:
        return []
    ver = win_driver_version(gpu[1])
    if ver and not driver_older(ver):
        say(f"moe-cache: AMD driver {ver} is {AMD_CHECKPOINT_FIXED} or newer: context checkpoints stay on (older drivers crash when saving one for this model)")
        return []
    say("moe-cache: AMD GPU on Windows + this model: context checkpoints off (--ctx-checkpoints 0; they crash AMD's Vulkan driver"
        + (f" {ver}" if ver else " (version unknown)") + f" here; AMD driver {AMD_CHECKPOINT_FIXED} or newer fixes it)"
        + ("; agents: the model cannot go back to an earlier point of a conversation, so edited history is processed again (Qwen3.6 keeps checkpoints)"
           if agent else ""))
    return ["--ctx-checkpoints", "0"]


MTP_ARGS = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
MTP_GAIN = "Qwen3.6 on the PC: decode +54 %, prompt processing -5..-8 %"


MTP_DENSE_WHY = ("a dense model runs every layer for every token, and MTP measured slower on the Qwen3.8-27B (64k: 26.8 vs 27.8 tok/s; 128k, part of the KV cache "
                 "on the CPU: 9.55 vs 15.4 tok/s)")


def is_dense_model(path):
    """True for a dense model (no expert layers), from the GGUF header; False when the header cannot be read (the MoE rule then applies)."""
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import model_layout
        lay = model_layout(path)
        return not lay["n_layers"] or not lay["expert_bytes_per_layer"]
    except Exception:
        return False


def mtp_decide(want, agent, head, declared, server_has, user_spec, dense=False):
    """Multi-token prediction (I43): the model's own draft head (blk.N.nextn.*) guesses the next 2 tokens, the model checks them in one step; the
    output can differ slightly for some models (GLM-5.3 code text did). want: True (--mtp), False (--no-mtp), None (default: on in --agent mode, EXCEPT for a dense
    model: LAUNCHER-1133, it measured slower there; --mtp still turns it on). -> (on, one-line reason or None)."""
    if user_spec:
        return False, "moe-cache: MTP: speculative decoding set by your own arguments (--spec-type / --model-draft): left as given"
    if want is False:
        return False, ("moe-cache: MTP off (--no-mtp)" if head else None)
    if not head:
        why = (f"the header declares {declared} MTP layer(s) but this file has no blk.N.nextn.* tensors (the quant dropped them)" if declared
               else "the model has no MTP head (no blk.N.nextn.* tensors)")
        if want:
            raise SystemExit(f"--mtp: {why}")
        return False, (f"moe-cache: MTP off: {why}" if agent else None)
    if not server_has:
        if want:
            raise SystemExit("--mtp: this llama-server has no MTP speculative decoding (no draft-mtp in its --help for --spec-type); a newer llama.cpp has it")
        return False, (f"moe-cache: MTP off: the model has an MTP head but this llama-server lacks --spec-type draft-mtp (a newer llama.cpp has it; {MTP_GAIN})"
                       if agent else None)
    if dense and not want:
        return False, f"moe-cache: MTP off: {MTP_DENSE_WHY}; --mtp turns it on"
    if want or agent:
        return True, (f"moe-cache: MTP on: the model has an MTP head ({head} layer) and llama-server has draft-mtp: {' '.join(MTP_ARGS)} "
                      f"(the output can differ slightly for some models; {MTP_GAIN}){'' if want else '; --no-mtp turns it off'}")
    return False, f"moe-cache: this model has an MTP head: --mtp for faster decode ({MTP_GAIN})"


# Model-card sampling (I45): (temp, top_p, top_k, min_p, presence_penalty) per GGUF architecture and mode, as the official Hugging Face cards state
# them (read 2026-10-27, section "Best Practices" / "1. Sampling Parameters"). None = the card gives no value (llama.cpp's default stays).
#   qwen35moe  huggingface.co/Qwen/Qwen3.6-35B-A3B and Qwen/Qwen3.5-122B-A10B (same numbers): thinking general 1.0/0.95/20/0/1.5;
#              thinking precise coding 0.6/0.95/20/0/0.0 (used for --agent); non-thinking 0.7/0.8/20/0/1.5
#   qwen4exp   huggingface.co/Qwen/Qwen3.8-Flash-Next: thinking 1.0/0.95/20/0/0.0; non-thinking 0.7/0.8/20/0/1.5
#   qwen3moe, qwen3next  huggingface.co/Qwen/Qwen3-30B-A3B, Qwen3-Next-80B-A3B-*: thinking 0.6/0.95/20/0, non-thinking 0.7/0.8/20/0, presence
#              "between 0 and 2"; huggingface.co/Qwen/Qwen3-30B-A3B-GGUF: "presence_penalty to 1.5 for quantized models"
#   glm*       huggingface.co/zai-org/GLM-5.3-Flash: temperature 1.0, top_p 0.95 in its evaluation notes; nothing else stated
SAMPLING_CARDS = {
    "qwen35moe": ("Qwen3.5/3.6", {"think": (1.0, 0.95, 20, 0.0, 1.5), "code": (0.6, 0.95, 20, 0.0, 0.0), "nothink": (0.7, 0.8, 20, 0.0, 1.5)}),
    "qwen4exp": ("Qwen3.8", {"think": (1.0, 0.95, 20, 0.0, 0.0), "nothink": (0.7, 0.8, 20, 0.0, 1.5)}),
    "qwen3moe": ("Qwen3", {"think": (0.6, 0.95, 20, 0.0, 1.5), "nothink": (0.7, 0.8, 20, 0.0, 1.5)}),
    "qwen3next": ("Qwen3-Next", {"think": (0.6, 0.95, 20, 0.0, 1.5), "nothink": (0.7, 0.8, 20, 0.0, 1.5)}),
    "glm": ("GLM", {"think": (1.0, 0.95, None, None, None)}),
}
SAMPLING_FLAGS = (("temp", ("--temp", "--temperature")), ("top_p", ("--top-p",)), ("top_k", ("--top-k",)), ("min_p", ("--min-p",)),
                  ("presence_penalty", ("--presence-penalty",)))


def sampling_decide(meta, filename, extra, agent):
    """-> (llama-server args, one-line reason or None): the model card's sampling settings for this model and mode, except those you set (flags
    after --) or that the GGUF carries (general.sampling.*, which llama.cpp applies itself). Request values (an agent's own temperature) still win:
    llama-server uses these only as defaults for fields a request leaves out."""
    arch = meta.get("arch") or ""
    fam = SAMPLING_CARDS.get("glm" if arch.startswith("glm") else arch)
    if not fam:
        return [], None
    label, modes = fam
    names = " ".join([meta.get("name") or "", meta.get("basename") or "", os.path.basename(filename)])
    kw = (arg_after(extra, "--chat-template-kwargs") or "").replace(" ", "")
    off = (arg_after(extra, "-rea", "--reasoning") == "off" or arg_after(extra, "--reasoning-budget") == "0" or '"enable_thinking":false' in kw
           or re.search(r"Instruct", names) is not None)
    mode = "nothink" if off and "nothink" in modes else ("code" if agent and "code" in modes else "think")
    gguf = {"temp", "top_p", "top_k", "min_p"} & set(meta.get("sampling") or {})
    args, kept = [], []
    for (key, flags), v in zip(SAMPLING_FLAGS, modes[mode]):
        if v is None or any(a == f or a.startswith(f + "=") for a in extra for f in flags):      # `--temp 0.2` and `--temp=0.2`
            continue
        if key in gguf:
            kept.append(key.replace("_", "-"))
            continue
        args += [flags[0], f"{v:g}"]
    if not args:
        return [], None
    why = {"think": "thinking", "nothink": "non-thinking", "code": "thinking, its precise-coding values for --agent"}[mode]
    return args, (f"moe-cache: sampling from the {label} model card ({why}): {' '.join(args)}"
                  + (f"; {', '.join(kept)} from the GGUF" if kept else "") + "; requests that send their own values win; --no-sampling-defaults turns it off")


def sampling_args(off, agent, model, extra):
    """sampling_decide() on the model's header; says what it set in one line."""
    if off:
        return []
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import sampling_meta
        args, why = sampling_decide(sampling_meta(model), model, extra, agent)
    except Exception:
        return []
    if why:
        say(why)
    return args


def mtp_args(want, agent, ls, model, extra, exact=False, dense=None):
    """The --spec-* arguments for the model's MTP head (see mtp_decide), and says why in one line. Reads the GGUF header and, only when the model
    has a head, llama-server --help."""
    user_spec = arg_after(extra, "--spec-type", "--model-draft", "-md") is not None
    if exact:                       # LAUNCHER-984: --exact turns every token-changing mode off (an explicit --mtp is refused in approx_stage)
        return []
    import moe_cache_approx as ax
    agent = agent and ax.default_on("mtp")          # the default-on in agent mode follows the table; --mtp still asks for it
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import mtp_layers
        head, declared = mtp_layers(model)
    except Exception as e:
        if want:
            raise SystemExit(f"--mtp: could not read the model header ({e})")
        return []
    server_has = False
    if head and want is not False and not user_spec:
        try:
            server_has = "draft-mtp" in ls_help(ls)
        except Exception:
            pass
    on, why = mtp_decide(want, agent, head, declared, server_has, user_spec, dense=is_dense_model(model) if dense is None else dense)
    if why:
        say(why)
    if not on:
        return []
    return MTP_ARGS[:2] + ([] if arg_after(extra, "--spec-draft-n-max") is not None else MTP_ARGS[2:])

def win_request_save(pid, path, timeout=60.0, proc=None):
    """Windows: ask the plugin inside llama-server `pid` to save its usage profile now (named event Local\\moe-cache-save-<pid>; a server on
    Windows is killed, so the save at exit never runs) and wait up to `timeout` s for `path` to change (less if `proc` exits). True when it changed."""
    if not WINDOWS:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)     # own instance: the types set here do not touch ctypes.windll.kernel32
        k.OpenEventW.restype, k.OpenEventW.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        k.SetEvent.restype, k.SetEvent.argtypes = wintypes.BOOL, [wintypes.HANDLE]
        k.CloseHandle.restype, k.CloseHandle.argtypes = wintypes.BOOL, [wintypes.HANDLE]
        h = k.OpenEventW(0x0002, False, f"Local\\moe-cache-save-{pid}")     # EVENT_MODIFY_STATE
        if not h:
            return False
        before = os.path.getmtime(path) if path and os.path.isfile(path) else None
        k.SetEvent(h)
        k.CloseHandle(h)
        t = time.time()
        while time.time() - t < timeout:
            if path and os.path.isfile(path) and (before is None or os.path.getmtime(path) > before):
                return True
            if proc is not None and proc.poll() is not None:
                break
            time.sleep(0.25)
    except Exception:
        pass
    return False

def keep_awake(cmd, allow_sleep=False):
    """Keep the machine from sleeping while the server runs, without changing any power setting (idea I47: the PC slept for 7 minutes in the middle of a
    benchmark). Windows: a power request on this process (SetThreadExecutionState, ES_CONTINUOUS | ES_SYSTEM_REQUIRED; the display may still turn off);
    Linux: a helper holds `systemd-inhibit --what=sleep:idle` while the server runs (ends with the launcher, however it ends); skipped quietly when
    systemd-inhibit is missing or logind refuses. --allow-sleep / MOE_CACHE_ALLOW_SLEEP=1 turn it off. Returns (cmd, release, note); note is the
    start-up line, or "" when nothing was done."""
    if allow_sleep or os.environ.get("MOE_CACHE_ALLOW_SLEEP") in ("1", "true", "yes"):
        return cmd, (lambda: None), "moe-cache: the machine may sleep while the server runs (--allow-sleep)"
    if WINDOWS:
        try:
            import ctypes
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.SetThreadExecutionState.restype = ctypes.c_uint32
            k.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
            if not k.SetThreadExecutionState(0x80000001):            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                return cmd, (lambda: None), ""
            return cmd, (lambda: k.SetThreadExecutionState(0x80000000)), \
                "moe-cache: keeping the PC awake while the server runs (no power setting changed; --allow-sleep to allow sleep)"
        except Exception:
            return cmd, (lambda: None), ""
    inh = shutil.which("systemd-inhibit")
    if sys.platform.startswith("linux") and inh:
        # the lock is held by a helper (`systemd-inhibit ... cat`) whose stdin is a pipe from this process: it ends when we release it, and also
        # when the launcher dies by any means (the pipe closes). llama-server itself is not wrapped (systemd-inhibit forks its command: pid, signals
        # and exit code would no longer be llama-server's)
        try:
            h = subprocess.Popen([inh, "--what=sleep:idle", "--who=moe-cache", "--why=llama-server is running", "--mode=block", "cat"],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.3)
            if h.poll() is None:
                def release():
                    try:
                        h.stdin.close()
                        h.wait(timeout=5)
                    except Exception:
                        h.kill()
                return cmd, release, \
                    "moe-cache: keeping the machine awake while the server runs (systemd-inhibit; no power setting changed; --allow-sleep to allow sleep)"
        except Exception:
            pass
    return cmd, (lambda: None), ""


# LAUNCHER-001: plugin lines that mean a long-prompt block ran the slow way. One plain sentence per kind, the first time it shows up in the server's output.
# The plugin's own lines stay as they are (src/mc_backend.inl); a CPU block took about 22 s instead of 9 s on the PC (handover n23, Flash-Next, 128k prompt).
SLOW_PATH_LINES = [
    ("no-mem", re.compile(r"a long-prompt block needs (\d+) MiB of GPU memory but only (\d+) MiB are free")),
    ("no-op", re.compile(r"the GPU backend does not support .* such expert blocks stay on the CPU")),
    ("no-ram", re.compile(r"a long-prompt block would upload whole expert tensors with experts not in RAM: it stays on the CPU")),
    ("alloc", re.compile(r"could not allocate GPU memory for a long-prompt block: blocks stay on the CPU for \d+ graphs")),
    ("no-pin", re.compile(r"no pinned host buffer for the long-prompt block")),
]


def slow_path_note(line):
    """(kind, one plain sentence) when a server output line says a long-prompt block took the slow way, else None."""
    for kind, rx in SLOW_PATH_LINES:
        m = rx.search(line)
        if not m:
            continue
        if kind == "no-mem":
            need, free = int(m.group(1)), int(m.group(2))
            short = need - free
            return kind, (f"moe-cache: slow path: a long-prompt block ran on the CPU because the GPU had {free} MiB free and it needs {need} MiB"
                          + (f" (short by {short} MiB: close programs that use the GPU, or --vram / --moe-cache-mib a little lower)" if short <= 64 else
                             " (close programs that use the GPU, or lower --vram / --moe-cache-mib to leave room)")
                          + "; a block on the CPU took about 22 s instead of 9 s in our 128k-prompt measurement")
        return kind, {
            "no-op": "moe-cache: slow path: this GPU backend cannot compute some expert weights, so those blocks of a long prompt run on the CPU (slower prompts, answers are the same)",
            "no-ram": "moe-cache: slow path: a long-prompt block stays on the CPU because its experts are not in RAM (a bigger --ram or a smaller context lets the GPU do it)",
            "alloc": "moe-cache: slow path: the GPU would not give memory for a long-prompt block, so the next blocks run on the CPU for a while (free GPU memory: close programs that use it)",
            "no-pin": "moe-cache: slow path: no pinned host memory for the long-prompt block's result, so it is read back the plain way (a little slower, answers are the same)",
        }[kind]
    return None


GPU_OOM_RE = re.compile(r"ErrorOutOfDeviceMemory|Device memory allocation .* failed|cudaMalloc failed|CUDA error: out of memory|hipMalloc failed|"
                        r"(failed to allocate|unable to allocate) (Vulkan|CUDA|ROCm|HIP|SYCL)\w* (compute )?buffer", re.I)


# llama.cpp's own abort when its scheduler's hash set is full ("ggml-impl.h:330: fatal error", ggml_backend_sched_split_graph, found in llama.cpp's --fit probe):
# seen on the stock server too, with a q8_0 K and V cache on Granite 1B, at any context (2026-11-07). The launcher's q8_0 default from 128k on made it look like a 131072 crash.
GGML_ABORT_RE = re.compile(r"ggml-impl\.h:\d+: fatal error")


def kv_retry_cmd(cmd):
    """The command for the one f16 retry after llama.cpp aborted at load with a quantized KV cache this launcher chose: every -ctk / -ctv (long form too) whose type is not f16
    is dropped, so llama.cpp's f16 default applies to K and V alike. -> (new command, "K q8_0 / V q4_0"-style words for what was dropped), or (cmd, "") when there is nothing to
    drop. The KV ladder (LAUNCHER-1063) also picks K q8_0 / V q4_0 and q4_0 / q4_0: dropping only q8_0 values left V at q4_0 in the first case (a retry with f16 / q4_0, announced as
    f16) and changed nothing in the second (no retry at all). (Written by the outside review of 7c3d024; the caller only asks when the launcher chose the types.)"""
    names = {"-ctk": "K", "--cache-type-k": "K", "-ctv": "V", "--cache-type-v": "V"}
    new, was, i = [], {}, 0
    while i < len(cmd):
        if cmd[i] in names and i + 1 < len(cmd) and cmd[i + 1] != "f16":
            was[names[cmd[i]]] = cmd[i + 1]
            i += 2
            continue
        new.append(cmd[i])
        i += 1
    if not was:
        return list(cmd), ""
    return new, " / ".join(f"{k} {was[k]}" for k in ("K", "V") if k in was)


def run_child(cmd, env, port=None, expect_plugin=False, mem_limit=None, allow_sleep=False, oom_retry=None, kv_retry=None, tune_retry=None):
    """run_child_once, and when llama-server stops before it is ready because the GPU ran out of memory at load (another program took the memory
    after the plan was made: desktop PC 2026-10-30, the control page planned with 13.4 GiB free), ask `oom_retry()` for a new command (it measures the
    free GPU memory again and plans again) and start once more. One retry only."""
    with OwnsPort(port):
        return _run_child(cmd, env, port, expect_plugin, mem_limit, allow_sleep, oom_retry, kv_retry, tune_retry)


def _run_child(cmd, env, port, expect_plugin, mem_limit, allow_sleep, oom_retry, kv_retry=None, tune_retry=None):
    rc, seen = run_child_once(cmd, env, port, expect_plugin, mem_limit, allow_sleep)
    quiet = (rc & 0xFFFFFFFF) == 0xC000013A if WINDOWS else rc in (-signal.SIGINT, -signal.SIGTERM, 130, 143)
    if getattr(seen.get("guard"), "stopped", None):
        # LAUNCHER-1029: the commit guard ended llama-server because free commit ran out. TerminateProcess gives exit code 1 (not "quiet"), and a log with an out-of-memory
        # line would then start a second load from the same low commit: the thing the guard exists to prevent. No retry; the guard's reason is already said.
        say_exit(rc, seen, expect_plugin and port)
        return rc
    if rc not in (0, None) and not quiet and tune_retry and not seen["ready"]:
        # LAUNCHER-1045: the server did not come up with a plan the auto-tune picked: the pick is marked bad and the plan as it is starts once (the other retries still apply to it)
        new = tune_retry(rc, seen)
        if new:
            return _run_child(new[0], new[1], port, expect_plugin, mem_limit, allow_sleep, oom_retry, kv_retry, None)
    if rc not in (0, None) and not quiet and kv_retry and seen.get("kvabort") and not seen["ready"]:
        # The known abort (Granite 1B, 63a07fa) prints only "ggml-impl.h:NNN: fatal error", nothing about the KV type, so the text cannot be narrowed further; the retry is
        # limited instead to: an abort before the server was ready, a q8_0 KV cache this launcher chose (kv_retry is None otherwise), and ONE retry. The matched line is logged.
        new = kv_retry()
        if new:
            say(f"moe-cache: abort line that triggered the f16 KV retry: {seen['kvabort']}", err=True)
            return run_child_once(new, env, port, expect_plugin, mem_limit, allow_sleep, final=True)[0]
    if rc not in (0, None) and not quiet and oom_retry and seen["oom"] and not seen["ready"]:
        new = oom_retry()
        if new:
            return run_child_once(new, env, port, expect_plugin, mem_limit, allow_sleep, final=True)[0]
    say_exit(rc, seen, expect_plugin and port)
    return rc


# Game launchers that keep an embedded browser running. Only the EA app is measured (2026-11-01, paired, Flash-Next 34k: prompts 249 -> 192 tok/s, decode 28 -> 24);
# the others are the same kind of program and are named as untested.
HEAVY_BACKGROUND = {"eadesktop.exe": "EA app", "epicgameslauncher.exe": "Epic Games Launcher", "battle.net.exe": "Battle.net", "upc.exe": "Ubisoft Connect"}


def heavy_background_apps(csv_text):
    """Names of the programs in HEAVY_BACKGROUND found in `tasklist /FO CSV /NH` output."""
    found = {row.split('","')[0].strip('"').lower() for row in csv_text.splitlines()}
    return [name for exe, name in HEAVY_BACKGROUND.items() if exe in found]


# What is using the GPU and the CPU right now (Windows). The numbers are what Windows' own counters say over a few seconds; nothing here claims a
# program slows moe-cache down except the EA app, which was measured (paired, 2026-11-01; with it open the GPU's graphics engine was 27-29 % busy
# with it and 10 % with the desktop compositor, prompts ran 23 % slower, decode 14 % slower; quiet PC 2026-11-02: prompts the same, decode 8-9 % slower. It depends on how much it draws).
BUSY_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$n = @{}; Get-Process | ForEach-Object { $n[[string]$_.Id] = $_.ProcessName }
$c1 = @{}; Get-Process | ForEach-Object { $c1[[string]$_.Id] = $_.TotalProcessorTime.TotalSeconds }
$g = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 3).CounterSamples
$e = @{}
foreach ($s in $g) { if ($s.InstanceName -match 'pid_(\d+)_.*engtype_(\w+?)(?:_\d+)?$' -or $s.InstanceName -match 'pid_(\d+)_.*engtype_(\w+)') { $k = $Matches[1] + '|' + $Matches[2]; $e[$k] += $s.CookedValue / 3 } }
foreach ($k in $e.Keys) { $p = $k.Split('|'); if ($e[$k] -ge 0.5) { "gpu`t$($n[$p[0]])`t$($p[1])`t$([math]::Round($e[$k],1))" } }
(Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples | ForEach-Object { if ($_.CookedValue -ge 100MB -and $_.InstanceName -match 'pid_(\d+)_') { "mem`t$($n[$Matches[1]])`t$([int]($_.CookedValue/1MB))" } }
Get-Process | ForEach-Object { $d = $_.TotalProcessorTime.TotalSeconds - $c1[[string]$_.Id]; if ($c1.ContainsKey([string]$_.Id) -and $d -ge 0.3) { "cpu`t$($_.ProcessName)`t$([math]::Round($d / 3 * 100))" } }
"""
BUSY_SKIP = {"idle", "system", "dwm", "csrss", "llama-server", "python", "pythonw", "powershell", "pwsh", "conhost", "moe-cache", "memory compression", "registry"}
BUSY_LIMITS = {"gpu": 5.0, "mem": 1000.0, "cpu": 50.0}     # % of the engine, MB of GPU memory, % of one core: below these nothing is said


def parse_busy(text):
    """Programs above the limits, from the tab-separated lines of BUSY_SCRIPT, busiest first: [{"name", "kind", "value", "what"}]."""
    sums = {}
    for line in text.splitlines():
        f = line.rstrip("\r").split("\t")
        if len(f) < 3 or f[0] not in BUSY_LIMITS or f[1].lower() in BUSY_SKIP or not f[1]:
            continue
        try:
            val = float(f[-1])
        except ValueError:
            continue
        key = (f[0], f[1])
        if f[0] == "gpu" and f[2].lower() not in ("3d", "compute", "computeengine"):
            continue                   # copy and video engines do not compete with the expert uploads and compute
        sums[key] = sums.get(key, 0.0) + val
    out = []
    for (kind, name), val in sums.items():
        if val >= BUSY_LIMITS[kind]:
            what = {"gpu": f"{val:.0f} % of the GPU's graphics and compute engines", "mem": f"{val / 1024:.1f} GB of GPU memory", "cpu": f"{val / 100:.1f} CPU cores"}[kind]
            out.append({"name": name, "kind": kind, "value": val, "what": what})
    return sorted(out, key=lambda r: (r["kind"] != "gpu", -r["value"]))


def busy_programs(timeout=30):
    """Windows only: what is using the GPU and CPU for about 3 seconds. [] elsewhere or when PowerShell fails."""
    if not WINDOWS:
        return []
    try:
        r = subprocess.run([win_tool("powershell"), "-NoProfile", "-NonInteractive", "-Command", BUSY_SCRIPT], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except Exception:
        return []
    return parse_busy(r.stdout)


def busy_note(item):
    """One sentence for a program from busy_programs(): what Windows measured, and what closing it is worth (measured only for the EA app)."""
    base = f"{item['name']} is using {item['what']} right now."
    if item["name"].lower().startswith(("eacef", "eadesktop", "eabackground", "ealocal")):
        return base + " Closing it: on our test PC (Flash-Next at 34k) decode was 8-14 % faster and long prompts the same to 23 % faster, depending on how much it draws; your machine can differ."
    return base + " Closing it: not measured for this program."


def warn_heavy_background():
    """Windows only; a failed tasklist or PowerShell says nothing."""
    try:
        r = subprocess.run([win_tool("tasklist"), "/FO", "CSV", "/NH"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    except Exception:
        return
    named = heavy_background_apps(r.stdout)
    for name in named:
        if name == "EA app":
            say("moe-cache: the EA app is open: close it for full speed (on our test PC with it open: decode 8-14 % slower, long prompts up to 23 % slower, depending on how much it draws; your machine can differ)", err=True)
        else:
            say(f"moe-cache: {name} is open: if speed is low, close it (not measured for this program; on our test PC the EA app cost up to 23 %)", err=True)
    if "EA app" in named:
        return
    for item in busy_programs()[:3]:       # not the EA app (already named): programs busy on the GPU or CPU in the last seconds
        say("moe-cache: " + busy_note(item), err=True)


def say_exit(rc, seen, with_ready):
    g = seen.get("guard")
    if g is not None and g.min_free is not None and g.min_free < g.soft:
        say(f"moe-cache: memory guard: the lowest {g.label} seen while llama-server ran was {g.min_free / 1e9:.1f} GB (warning line {g.soft / 1e9:.1f} GB)", err=True)
    if g is not None and g.stopped:
        say("moe-cache: llama-server was stopped by the memory guard (see the WARNING above); it did not crash", err=True)
        return
    quiet = (rc & 0xFFFFFFFF) == 0xC000013A if WINDOWS else rc in (-signal.SIGINT, -signal.SIGTERM, 130, 143)   # Ctrl-C / terminate
    if rc not in (0, None) and not quiet:
        say("moe-cache: " + exit_reason(rc, seen["ready"] if with_ready else None) + (f"; last plugin line: {seen['last']}" if seen["last"] else ""), err=True)


def meminfo_available():
    """MemAvailable in bytes (Linux), None elsewhere or when unreadable."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024.0
    except (OSError, ValueError):
        pass
    return None


GUARD_SAVE_WAIT_S = 1.0      # seconds the guard's stop waits for a profile save (LAUNCHER-991 review: it used to wait up to 10 s while free commit was already under the hard floor)


def guard_stop(p, env, free, hard, request_save=None, windows=None):
    """The commit guard's stop of llama-server `p`: free commit is already under the hard floor, so every second counts. A profile save is tried first only on Windows with a
    profile, only while `free` is not under half the hard floor (then it terminates at once), and waits at most GUARD_SAVE_WAIT_S s; then the process is terminated whatever the save did."""
    save = request_save or win_request_save
    if (WINDOWS if windows is None else windows) and env.get("MOE_CACHE_PROFILE") and (free is None or free >= hard / 2):
        try:
            save(p.pid, env["MOE_CACHE_PROFILE"], timeout=GUARD_SAVE_WAIT_S, proc=p)      # TerminateProcess would lose what the cache learned
        except Exception:
            pass
    try:
        p.terminate()
    except OSError:
        pass


def start_commit_guard(p, env, say_fn=None, probe=None, loading=None):
    """LAUNCHER-991, run-time guard: while llama-server (process p) runs, poll free commit (Windows) or MemAvailable (Linux) every 2 s (every 0.25 s while `loading()` is true: a Vulkan load can take several GB in one call). Under the soft line (max(6 GB, 12.5 % of
    the commit limit) on Windows) a loud 'moe-cache: WARNING' line goes to the terminal, which the Monitor and the control page lift from the log; under the hard line (2 GB;
    Linux 1 GB) llama-server is stopped cleanly and the reason said, before the system starts killing programs. Windows stops by default; Linux only warns unless
    MOE_CACHE_COMMIT_GUARD=stop (the Linux line is not measured). =warn only warns, =off is off.
    Returns the guard (its .stopped holds the reason after a stop) or None."""
    sys.path.insert(0, os.path.join(HERE, "planner"))
    import commit_guard as cgd
    asked = env.get("MOE_CACHE_COMMIT_GUARD", os.environ.get("MOE_CACHE_COMMIT_GUARD"))
    mode = cgd.parse_mode(asked if asked else ("stop" if WINDOWS else "warn"))      # Linux warns only until the 1 GiB MemAvailable line is measured on real loads
    if mode == "off":
        return None
    say_fn = say_fn or (lambda m: say(m, err=True))
    if WINDOWS:
        limit = win_commit()[0]
        if not limit:
            say_fn("moe-cache: warning: Windows did not report the commit limit: the memory guard is off for this run")
            return None
        probe = probe or (lambda: win_commit()[1])
        soft, hard, label = cgd.soft_floor(limit), cgd.HARD_FLOOR, "free commit"
    else:
        tot = mem_total()
        if not tot:
            say_fn("moe-cache: warning: the machine's memory size is unknown: the memory guard is off for this run")
            return None
        probe = probe or meminfo_available
        (soft, hard), label = cgd.linux_floors(tot), "available memory"
    def stop(reason):
        guard_stop(p, env, g.last_free if g else None, hard)
    g = cgd.CommitGuard(probe, say_fn, stop, soft, hard, mode, label=label, loading=loading)
    g.start(lambda: p.poll() is None)
    return g


def proc_resident_bytes(pid):
    """Resident memory of one process in bytes (mapped model pages + private): Windows GetProcessMemoryInfo WorkingSetSize, Linux /proc/PID/status VmRSS; None when unreadable."""
    try:
        if WINDOWS:
            import ctypes
            from ctypes import wintypes
            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k = ctypes.WinDLL("kernel32", use_last_error=True); ps = ctypes.WinDLL("psapi", use_last_error=True)      # own instances: the types set here do not touch ctypes.windll
            k.OpenProcess.restype, k.OpenProcess.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k.CloseHandle.restype, k.CloseHandle.argtypes = wintypes.BOOL, [wintypes.HANDLE]
            ps.GetProcessMemoryInfo.restype, ps.GetProcessMemoryInfo.argtypes = wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            h = k.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return None
            try:
                c = PMC(); c.cb = ctypes.sizeof(PMC)
                return float(c.WorkingSetSize) if ps.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb) else None
            finally:
                k.CloseHandle(h)
        for line in open(f"/proc/{int(pid)}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024.0
    except Exception:
        pass
    return None


def ram_note_probe(pid):
    """(server resident, available physical RAM, total RAM) in bytes for the RAM note; Windows: GetProcessMemoryInfo + GlobalMemoryStatusEx, Linux: /proc/PID/status + MemAvailable."""
    srv = proc_resident_bytes(pid)
    if WINDOWS:
        tot, av = win_memory()
    else:
        tot, av = mem_total(), meminfo_available()
    return None if srv is None or not tot or av is None else (srv, float(av), float(tot))


def start_ram_note(p, env, say_fn=None, probe=None):
    """LAUNCHER-1145: while llama-server (process p) runs, ONE plain line (planner/commit_guard.py RamNote) when it holds a big part of the RAM and what is left is low, so the owner knows
    why the desktop got slow (other programs paged out). Read-only, rate limited (10 min), MOE_CACHE_RAM_NOTE=0 turns it off. Returns the note or None."""
    if env.get("MOE_CACHE_RAM_NOTE", os.environ.get("MOE_CACHE_RAM_NOTE", "")) == "0":
        return None
    sys.path.insert(0, os.path.join(HERE, "planner"))
    import commit_guard as cgd
    n = cgd.RamNote(probe or (lambda: ram_note_probe(p.pid)), say_fn or (lambda m: say(m, err=True)))
    n.start(lambda: p.poll() is None)
    return n


def run_child_once(cmd, env, port=None, expect_plugin=False, mem_limit=None, allow_sleep=False, final=False):
    """Run llama-server as a child (so helper threads can keep working), pass Ctrl-C / SIGTERM on and return its exit code.
    The child dies with the launcher. Its log is passed through; when the plugin was meant to load and the server becomes ready without a single
    'moe-cache:' line, a warning says it is running stock (found on the MSI Claw, 2026-10-15: the plugin failed to load and nothing said so)."""
    cmd, release_awake, awake_note = keep_awake(cmd, allow_sleep)
    if awake_note:
        say(awake_note)
    kw = {"env": env, "stderr": subprocess.PIPE}
    low = env.get("MOE_CACHE_LEAVE_ROOM", "0") not in ("", "0")       # --leave-room: the user's own programs come first
    if sys.platform.startswith("linux"):
        kw["preexec_fn"] = _child_low_linux if low else _die_with_parent_linux
    elif low and WINDOWS:
        kw["creationflags"] = 0x00004000      # BELOW_NORMAL_PRIORITY_CLASS
    p = subprocess.Popen(cmd, **kw)
    if env.get("MOE_CACHE_GPU_MEM_LOG", "") not in ("", "0"):         # R2-06, default off: dedicated / shared GPU memory of llama-server, once a second
        try:
            sys.path.insert(0, os.path.join(HERE, "planner"))
            from gguf_plan import start_gpu_mem_log
            start_gpu_mem_log(p.pid, lambda m: say(m, err=True), WINDOWS, env, lambda: p.poll() is None)
        except Exception:
            pass
    if WINDOWS:
        try:
            _kill_with_parent_windows(p, mem_limit)
        except Exception as e:
            say(f"moe-cache: warning: could not tie llama-server to the launcher ({e})", err=True)
    if WINDOWS and env.get("MOE_CACHE_PROFILE"):
        def save_now_and_then():      # closing the console window kills llama-server: keep at most 10 minutes of learning unsaved
            while p.poll() is None:
                time.sleep(600)
                if p.poll() is None:
                    win_request_save(p.pid, env["MOE_CACHE_PROFILE"], timeout=0)
        threading.Thread(target=save_now_and_then, daemon=True).start()
    seen = {"plugin": False, "disabled": "", "last": "", "ready": False, "oom": False}
    try:
        seen["guard"] = start_commit_guard(p, env, loading=lambda: not seen["ready"])       # LAUNCHER-991
    except Exception as e:
        say(f"moe-cache: warning: the memory guard did not start ({e})", err=True)
    try:
        start_ram_note(p, env)       # LAUNCHER-1145: read-only note when this server squeezes the RAM
    except Exception:
        pass
    def pump():
        out = getattr(sys.stderr, "buffer", None)
        for line in iter(p.stderr.readline, b""):
            if out is not None:
                try:
                    out.write(line); out.flush()
                except (OSError, ValueError):
                    out = None          # our own stderr is gone (closed pipe, no console): keep draining, or llama-server blocks in its next log write
            if GPU_OOM_RE.search(line.decode(errors="replace")):
                seen["oom"] = True
            cpn = None if EXPECT_CPU_WORK[0] else cpu_placement_note(line, cpu_run=("--device none" in " ".join(cmd) or " -ngl 0 " in " ".join(cmd) + " "))
            if cpn and cpn[0] not in seen.setdefault("cpuplace", set()):
                seen["cpuplace"].add(cpn[0])
                say(cpn[1], err=True)
            if GGML_ABORT_RE.search(line.decode(errors="replace")):
                seen["kvabort"] = line.decode(errors="replace").strip()[-200:] or True       # the matched line, so the retry can say which abort it answered
            if b"moe-cache:" in line:
                seen["plugin"] = True
                seen["last"] = line.decode(errors="replace").strip()
                note = slow_path_note(seen["last"])
                if note and note[0] not in seen.setdefault("slow", set()):
                    seen["slow"].add(note[0])
                    say(note[1], err=True)
                if plugin_off_line(line):
                    seen["disabled"] = line.decode(errors="replace").strip()
    pump_t = threading.Thread(target=pump, daemon=True)
    pump_t.start()
    if port:
        def check():       # seen["ready"] for every engine (LAUNCHER-1045 needs it for stock runs too); the plugin warnings only when the plugin was meant to load
            while True:    # no cap: a model that loads in 20 minutes is ready then, and a crash hours later must not look like "never came up"
                if p.poll() is not None:
                    return
                if healthy(port):
                    seen["ready"] = True
                    if not expect_plugin:
                        return
                    time.sleep(1)
                    if seen["disabled"]:
                        say(f"moe-cache: WARNING: the plugin loaded but disabled itself ({seen['disabled']}): llama-server runs stock", err=True)
                    elif not seen["plugin"]:
                        say("moe-cache: WARNING: the plugin did not load: llama-server is running stock llama.cpp. Likely causes: the plugin library "
                            "does not match this llama-server's ggml version (API 2 vs 3), a wrong MOE_CACHE_LIB path, or (preload method) a llama-server "
                            "without shared ggml libraries. Run: moe_cache.py check MODEL --llama-server ...", err=True)
                    return
                time.sleep(1)
        threading.Thread(target=check, daemon=True).start()
    def stop(sig, frame):
        p.terminate()
    if not WINDOWS:
        signal.signal(signal.SIGTERM, stop)
    saved = False
    while True:
        try:
            rc = p.wait()
            break
        except KeyboardInterrupt:      # the child got the same Ctrl-C; wait for it to finish
            if WINDOWS and env.get("MOE_CACHE_PROFILE") and not saved and p.poll() is None:
                saved = True           # Windows may end the server without its save at exit: ask for the save first (at most 30 s)
                try:
                    win_request_save(p.pid, env["MOE_CACHE_PROFILE"], timeout=30, proc=p)
                except KeyboardInterrupt:   # a second Ctrl-C: stop waiting for the save
                    pass
            continue
    release_awake()
    pump_t.join(timeout=5)             # the last lines llama-server printed (why it stopped) come before the launcher's own verdict
    if final:
        say_exit(rc, seen, expect_plugin and port)
    return rc, seen


def exit_reason(rc, ready):
    """One line on why llama-server stopped (the Claw's cache-mode crash ended the launcher without a word: RESULTS 50)."""
    when = "" if ready is None else (" after it was ready" if ready else " before it was ready")
    if WINDOWS:
        code = rc & 0xFFFFFFFF
        names = {0xC0000005: "access violation", 0xC0000409: "stack buffer overrun / fast fail", 0xC000001D: "illegal instruction (CPU lacks an instruction set)",
                 0xC0000135: "a DLL was not found", 0xC0000139: "a DLL entry point was not found", 0xC00000FD: "stack overflow", 0xC0000017: "out of memory",
                 0xFFFFFFFF: "stopped from outside (Stop-Process, Task Manager), not a crash"}
        return f"llama-server exited{when} with code 0x{code:08X}" + (f" ({names[code]})" if code in names else "")
    if rc < 0:
        try:
            nm = signal.Signals(-rc).name
        except ValueError:
            nm = f"signal {-rc}"
        hint = " (killed: out of memory? check 'journalctl -k' or the cgroup limit)" if -rc == signal.SIGKILL else ""
        return f"llama-server was stopped{when} by {nm}{hint}"
    return f"llama-server exited{when} with code {rc}"


# ---- server ----

SERVER_HELP = """\
--fast     DEPRECATED alias (still works; forces the mode on): approximate routing (MOE_CACHE_ROUTE=skip): a low-ranked expert that is not in RAM is skipped instead of read from the SSD;
           also a q8_0 KV cache (half the KV memory; KL on its own: 0.0120 on the PC's Vulkan GPU, 0.0045 on the laptop's CUDA GPU; docs/results-archive/RESULTS.md R11, RESULTS 53).
           About +10-12 % when the model is much bigger than RAM; about 1 top token in 40-50 differs from stock (KL 0.005-0.012, docs/results-archive/RESULTS_OLD_LAPTOP.md section 31).
--fast=kv  only the q8_0 KV cache of --fast (no skipped experts): at 128k+ it frees VRAM for more GPU layers and a bigger micro-batch (docs/QUALITY_GATE.md, level kv); the default at --ctx 131072 and above
--exact    turn EVERY mode off that changes tokens (q8_0 KV cache, the long-prompt mode, MTP drafting; the table is TOKEN_MODES in bin/moe_cache_approx.py): the answers are the same as plain
           llama.cpp's (EXACTNESS.md). Without --exact the default is faster and NOT token-identical: it switches on exactly the modes measured as a win (long-prompt from 32k, q8_0 KV from 128k, MTP in agent mode)
--retune   LAUNCHER-994 auto-tune: on the first real start for a model + machine + context the launcher tries 2-4 plans (n-cpu-moe, a bigger -ub from a small plan, RAM_LOCK, -np; only what you did not set; never the KV type; not for a model over 40 % of RAM), for at most 180 s in
           all, measures prompt and decode speed on each (a 1.5k-token prompt and 512 generated tokens; a shorter decode says "short probe"), keeps the fastest (it must beat the plan by 5 %) and stores
           it under <cache>/autotune/. Later starts use the stored pick. --retune measures again. --exact and every flag of yours win (a dimension you set is left alone); --dry-run and --plan-json only show the stored pick
--long-prompt  the long-prompt mode: a prompt chunk of 4096 tokens (-ub 4096) with the q8_0 KV cache, only when the memory plan of this card leaves room for it (otherwise nothing
           changes). It is ON BY DEFAULT from 32k context on (--long-prompt asks for it at any context). Flash-Next IQ3_XXS in a 65k conversation on the PC, against -ub 2048 with
           the f16 KV cache: prompts 331.8 against 187.4 tok/s (+77 %), answers 19.35 against 19.01 tok/s (+1.8 %). The q8_0 KV cache means answers are NOT byte-identical to plain
           llama.cpp's (against f16: KL 0.014-0.015, same top token in 94-95 % of positions, meaning kept in 10 of 10 checks at 55k). Not with --exact, --stock or your own --ub
--no-long-prompt  turn the long-prompt mode off: -ub 2048 and the f16 KV cache, as before v0.3.8
--fast=fewer  --fast, and the 2 lowest-ranked experts of every token are dropped on the CPU-held layers even when they are in RAM (MOE_CACHE_FEWER=2; opt-in, new)
--fast=2   stronger level for memory much smaller than the model: up to 4 low-ranked picks may be skipped (at least half are kept).
           Flash-Next at 12 GB (laptop; not yet measured on the PC): 13.7 vs 9.6 tok/s for --fast; KL 0.022 (two 2-bit quants of one model differ by 0.243). No gain at 24 GB.
--hybrid   opt-in, experimental: a bank of your most used experts (from the usage profile) lives in free GPU memory; the GPU computes the picks it holds
           while the CPU computes the rest at the same time (Flash-Next on 8 GB: 13.1 -> 17.4 tok/s). Not identical to stock (GPU float order, like
           --gpu-cache); replaces --gpu-cache and keeps all experts on the CPU side. MOE_CACHE_HYBRID=MIB sets the bank size (default: all free VRAM)
--approx safe|careful|aggressive  DEPRECATED alias (still works; forces the modes of the tier on; the default launch enables the modes measured as a win by itself). Speed tiers that CHANGE THE ANSWERS (none is identical to stock; `exact` = off). safe: 8-bit KV cache + the model's own MTP drafting if it has the
           head (every expert is computed). careful: --fast (rarely picked experts not in RAM are skipped, about 1 token in 40-50 differs) + the same. aggressive: --fast=2 (up to 4 low-ranked experts skipped, about 1 token
           in 28 differs). Each mode has a quality gate (docs/QUALITY_GATE.md, docs/CONFIGURATION.md) and runs only for models with a gate record, else --approx-unverified. Your own --fast wins and is reported.
           At the start a short canary (about 256 scored tokens, against the exact setting of the same model) re-measures the tier and steps it down when the KL divergence is past its gate
           (--no-approx-canary skips it; it guards, it never proves). Not with --exact or --stock. --approx-off MODE removes one mode (kv8, spec-exact, fast1, fast2)
--profile NAME  use (and keep learning) a separate usage profile for one kind of work, e.g. coding, thinking, research; default: one shared profile
--project NAME  its own profile plus a saved prompt state (KV cache) restored at start; save it with: moe_cache.py project save NAME
--gpu-cache off|auto|MIB  (Linux, preload method only) llama.cpp's GPU expert cache (--moe-cache-mib) on top of this cache. auto = all free VRAM the tuner sees minus --gpu-margin GiB.
           Not bit-identical to CPU-computed experts (GPU float order), identical to llama.cpp's own cache. Needs llama.cpp 23b0202a1 or newer (API 3 plugin build).
--leave-room  I use this PC while the model runs: RAM for the server = total minus max(8 GiB, 25 %) instead of max(4 GiB, 12.5 %), at least 2 GB of
           GPU memory kept free for the desktop (not with --gpu-margin), 2 CPU cores left free (not with pinned hybrid cores or your own -t), lower priority; the start-up
           lines say which of these were applied. Smoother PC, slower model. MOE_CACHE_LEAVE_ROOM=1 turns it on for good
--gpu-margin GiB  VRAM the tuner leaves free next to everything it places (default about 0.47 GiB = 0.5 GB; raise it if the GPU runs out of memory with a large -ub)
--ub auto|N  micro-batch size (prompt speed: 512 -> 2048 was +58 % at a 6k prompt). auto = the largest that fits next to the KV cache of --ctx
--ram GB   memory the cache and the layer plan are sized to (default: the cgroup limit or free RAM). A sizing hint, NOT a RAM cap: the launcher sets no
           limit on Linux (only a cgroup you set yourself is a hard limit). On Windows a job memory limit is set only for the plugin, only with --ram and
           only with MOE_CACHE_WIN_VIEW=0; the default file-view cache, the on-demand listener and stock llama-server get none (a job limit counts GPU
           memory too and broke Vulkan), so the server can use more RAM than --ram says
--max-ram GB  hard cap on the whole server's memory (default: physical RAM minus max(4 GiB, 12.5 %), = 28 GiB on a 32 GiB machine; 0 = none).
           Same rule on Linux and Windows; --ram and the free-memory logic stay below it (MOE_CACHE_MAX_RAM_GB does the same)
           The cap is kept by planning (cache budget, layer plan), never by stopping a running server (owner's rule)
--vram GiB GPU memory to plan with (default: what the GPU reports free)
--n-cpu-moe N  layers whose experts stay on the CPU (default: chosen by the planner from free VRAM)
--no-ecores  do not let the expert executor use efficiency cores (hybrid Intel CPUs: up to 4 are used by default, about +10-15 %)
--always-plugin  use the plugin even when the model fits in the memory the server may use (by default stock llama-server runs then: faster, RESULTS 44)
--stock    run stock llama-server without the plugin whatever the model's size (to compare stock and moe-cache on the same model)
--allow-sleep  let the machine sleep while the server runs (by default it is kept awake without changing a power setting: Windows power request,
           Linux systemd-inhibit; MOE_CACHE_ALLOW_SLEEP=1 does the same)
--no-sched-tuning  accepted and ignored (the Windows scheduling tuning was removed in v0.3.8: it cost 7 % decode; archive/README.md); kept for one release so old commands still run
--no-pin     do not pin llama-server's threads (hybrid Intel CPUs: one thread per performance core by default, about +15 % decode)
--agent    for coding agents (Hermes Agent, OpenCode, ...): context 65536 unless --ctx is given (Hermes refuses less than 64000), tool calls on (--jinja), a stable model name
           (--alias, the file name) and, once the server is ready, the base URL and config snippets for Hermes and OpenCode. Prompt reuse across
           turns is llama-server's own: each slot keeps its conversation, and models with recurrent layers (Qwen3-Next, Qwen3.5/3.6) roll back to a
           checkpoint made at the last user message and just before the end of each prompt (at most 8 per slot here; 1 GiB kept free for them and for restored conversations).
           moe_cache.py connect hermes|opencode [--write] sets the agent up for a running server
           Warm sessions (ideas I29, I34): every idle conversation of 2k+ tokens (with its checkpoints) is saved to disk 20 s after it changes,
           one file per slot, and the two longest are restored at the next start (each agent finds its own), so the agent's 5-20k-token preamble is not processed again after a restart (OpenCode on Qwen3.6: first turn
           7,438 tokens in 45.6 s cold; a session that finds the preamble cached starts with 29 new tokens in 0.7 s). --no-warm to turn it off
--cpu-only    run on the CPU only: llama-server gets --device none and -ngl 0 (also chosen by itself when --list-devices shows no GPU)
--no-ffn-ram  dense model bigger than the GPU: by default attention, the KV cache and most weights stay on the GPU and the FFN weights of the last layers stay in RAM (-ot), which keeps long chats usable (Qwen3-14B on an 8 GB card at 32k tokens: 3.8 tok/s against 1.0 with a plain layer split); this brings llama.cpp's --fit layer split back
--cpu-gpu-split [auto|FRACTION]  dense model with FFN weights in RAM (see --no-ffn-ram), CUDA, patched llama.cpp with patch 0091: the GPU computes part of the rows of those weights while the CPU computes the rest (the weights go over PCIe by DMA from page-locked RAM); auto follows the measured wait, a number (0.05-0.6) fixes the GPU's share. Opt-in until the PC run decides. Loads the model without mmap and without repacking (needed to page-lock it); laptop RTX 4060 + Qwen3-14B: quiet +6.7 / -2.6 % (below the +10 % bar), one busy core +19 to +39 %
--no-commit-check  Windows: do not refuse a start whose estimated commit (GPU memory counts as commit on some drivers) leaves under max(6 GB, 12.5 % of the commit limit) free, and do not shrink the layout for it. The run-time guard stays (MOE_CACHE_COMMIT_GUARD=warn|off changes that)
--no-auto-fit  dense model: pass -ngl 99 again (a file bigger than the GPU then aborts at load)
--no-starter  do not start from the shipped usage profile for this model family when you have no profile yet (profiles/<arch>-<layers>x<experts>.usage)
--mtp / --no-mtp  multi-token prediction with the model's own draft head (blk.N.nextn.*, e.g. Qwen3.6, GLM-5.3): --spec-type draft-mtp --spec-draft-n-max 2.
           Output can differ slightly for some models (GLM-5.3 code text did); Qwen3.6 on the PC: decode 45.6 -> 70.3 tok/s, prompt processing 5-8 % slower. On by default with --agent when the model has the head
           and llama-server has draft-mtp; one line says why it is on or off
--no-sampling-defaults  do not pass the model card's sampling settings. By default, for Qwen3 / 3.5 / 3.6 / 3.8 / Qwen3-Next and GLM models,
           temperature, top-p, top-k, min-p and presence penalty follow the official card for the mode (thinking unless --reasoning off; --agent:
           the card's coding values where it has them), skipping any you pass after -- and any the GGUF carries; low-bit quants loop less with them.
           Requests that send their own values (agents send a temperature) still win
--api-key KEY  require this key on the API (passed to llama-server; better: LLAMA_API_KEY, not visible in the process list); needed when other machines can reach the server
MOE_CACHE_PYTHON  python with the gguf package for the planner (default: this python)"""


DENSE_FIT_TARGET_MIB = 256     # dense models: GPU memory --fit leaves free (llama.cpp default 1024: 6.3 tok/s vs 8.2 at 256 on a 14B that spills, laptop 8 GB)
LONG_PROMPT_MIN_CTX = 32768     # the long-prompt mode is the default from this context up (--long-prompt: any context)
STOCK_HEADROOM = 1.5e9     # stock when model + this fits in the memory the server may use
STOCK_OFFLOAD_MIN_BATCH = 512     # stock runs with CPU expert layers: batches below this stay on the CPU (llama.cpp: 32; T5, RESULTS 68)


def decide(model, ram_gb=None, plugin_wanted=False, stock_wanted=False, machine=None):
    """Stock llama-server or the plugin for this model: (stock, size bytes, memory the server may use). The one rule server() and the control page
    (moe_cache_ui.plan_model) both use. When the model fits, stock is faster (its page cache holds the whole file and stays warm across restarts;
    Qwen3.6-35B at 24 GB: stock 42.9 vs plugin 36.6 warm / 33.4 cold, RESULTS 44). --fast, --hybrid and --always-plugin need the plugin;
    --stock runs stock whatever the size (to compare the two)."""
    try:
        size = sum(os.path.getsize(p) for p in model_parts(model))
    except OSError:
        size = None
    avail = machine.mem_for_server(ram_gb) if machine else mem_for_server(ram_gb)
    return bool(stock_wanted or (size and avail and size + STOCK_HEADROOM <= avail and not plugin_wanted)), size, avail


def prompt_cache_note(cram, extra, agent):
    """One plain line for the log (and so the control page's "What the launcher decided"): how much host RAM llama-server keeps for finished
    conversations (--cache-ram), how many conversations (slots) it serves at once and how many checkpoints each keeps. Two agents on one server
    reuse each other's shared prompt only while the prompt cache holds it (another project sizes its cache from free memory; ours: 2048 MiB, or off below
    16 GB of server memory where the RAM is worth more to the expert cache, RESULTS 44). Change it with: -- --cache-ram MiB."""
    if cram is None:
        mem = "this llama-server has no --cache-ram: its own prompt-cache default"
    elif re.fullmatch(r"-?\d+", str(cram)) and int(cram) == 0:
        mem = "off (--cache-ram 0): a conversation that is replaced is processed again when it comes back"
    elif re.fullmatch(r"-?\d+", str(cram)) and int(cram) < 0:
        mem = "no size limit (--cache-ram -1)"
    else:
        mem = f"up to {cram} MiB of host RAM keeps finished conversations for reuse (--cache-ram)"
    np_ = arg_after(extra, "-np", "--parallel")
    slots = f"{np_} slot{'s' if str(np_) != '1' else ''} (-np)" if np_ else "llama-server's default number of slots"
    cp = arg_after(extra, "-ctxcp", "--ctx-checkpoints", "--swa-checkpoints")
    cps = f", {cp} checkpoint{'s' if str(cp) != '1' else ''} per slot" if cp is not None and agent else ""
    return f"moe-cache: prompt cache: {mem}; {slots}{cps}. To change it: moe-cache-server ... -- --cache-ram MiB"


def default_alias(model):
    """The model name agent mode gives llama-server: the file name without .gguf and the shard suffix; for an Ollama blob (sha256-<hex>, no
    extension) the Ollama tag with ':' and '/' turned into '-' ('qwen3.6-35b-a3b'), so agents and saved conversations get a readable name."""
    base = os.path.basename(model)
    if re.fullmatch(r"sha256-[0-9a-f]{64}", base):
        try:
            from moe_cache_pick import ollama_name
            n = ollama_name(model)
        except Exception:
            n = None
        if n:
            return re.sub(r"[^A-Za-z0-9._-]", "-", n)
    return re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", base)


class Machine:
    """The facts about this computer that launch decisions use (XA-2): memory and commit, the GPU list, drivers, CPU cores, which llama-server and plugin
    build are here, and the child processes the planning runs (the planner, `llama-server --help`). Every method asks the real system through the module's
    small probes (so a test can replace a probe, or pass a Machine subclass with fixed answers to launch_plan). Cheap probes are not cached: a launch
    asks each once or twice."""
    @property
    def windows(self):
        return WINDOWS

    def mem_total(self): return mem_total()
    def physical_cores(self): return physical_cores()
    def mem_cap(self): return mem_cap()
    def mem_for_server(self, ram_gb): return mem_for_server(ram_gb)
    def win_memory(self): return win_memory()
    def win_commit(self): return win_commit()
    def win_dedicated_vram(self): return win_dedicated_vram()
    def win_driver_version(self, name): return win_driver_version(name)
    def win_cores(self): return win_cores()
    def win_core_groups(self): return getattr(win_cores, "groups", 1)      # set by win_cores() on its last call
    def list_devices(self, ls): return list_devices(ls)
    def pick_gpu(self, listing): return pick_gpu(listing)
    def pick_gpu_full(self, listing): return pick_gpu_full(listing)
    def nvidia_reports(self): return nvidia_reports()
    def ls_api(self, ls): return ls_api(ls)
    def lib_api(self, lib): return lib_api(lib)
    def pick_lib(self, ls): return pick_lib(ls)
    def ls_found_line(self, ls): return ls_found_line(ls)
    def ls_looked(self): return ls_looked()
    def load_env(self, ls, lib, method="auto"): return load_env(ls, lib, method)
    def vk_cache_cap(self, ls, model, listing=None): return vk_cache_cap(ls, model, listing)
    def run(self, cmd, **kw): return subprocess.run(cmd, **kw)
    def planner_answer(self, model, ctx, ram_gib, vram_gib): return planner_answer(model, ctx, ram_gib, vram_gib)
    def lib_exists(self, lib): return os.path.isfile(lib)

    def other_llama_servers(self): return other_llama_servers()

    def gpu_process_text(self): return win_gpu_process_text()       # LAUNCHER-1133

    def free_vram_bytes(self):
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import free_vram_bytes
        return free_vram_bytes()

    def gpu_total_free_bytes(self):
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import gpu_total_free_bytes
        return gpu_total_free_bytes()

    def has_cpu_atom(self): return os.access("/sys/devices/cpu_atom/cpus", os.R_OK)
    def has_cpu_core(self): return os.access("/sys/devices/cpu_core/cpus", os.R_OK)

    def cpu_atom_count(self):
        """Efficiency cores of a hybrid Intel CPU on Linux (0 when unreadable)."""
        try:
            return len(parse_cpus(open("/sys/devices/cpu_atom/cpus").read()))
        except Exception:
            return 0

    def perf_core_mask(self):
        """Hex mask of one hardware thread per performance core (Linux hybrid CPUs); None when unreadable."""
        try:
            cores = parse_cpus(open("/sys/devices/cpu_core/cpus").read())
            prim = [c for c in cores if min(parse_cpus(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read())) == c]
            return hex(sum(1 << c for c in prim))
        except Exception:
            return None


    def dense_core_mask(self):
        """(hex mask, thread count) for a dense model that runs partly on the CPU next to a GPU: BOTH hardware threads of all performance cores but two (Linux hybrid CPUs),
        so the GPU driver's threads have cores of their own. Laptop i7-13620H (6 P-cores), Qwen3-14B, FFN in RAM: 8 threads on 4 cores 8.7 tok/s, 12 threads 7.3, the old
        6 single threads (one per core) 5.1; layer split 8.3 vs 5.9-6.8; RAM only +8 %; a MoE does not gain (17.3 with 6 singles, 12-14 with 8 threads). None when unreadable."""
        try:
            cores = parse_cpus(open("/sys/devices/cpu_core/cpus").read())
            prim = sorted(c for c in cores if min(parse_cpus(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read())) == c)
            use = prim[:max(1, len(prim) - 2)]
            cpus = sorted({t for c in use for t in parse_cpus(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read())})
            return hex(sum(1 << c for c in cpus)), len(cpus)
        except Exception:
            return None


class LaunchPlan:
    """What server() will do: the decisions (engine stock/plugin, GPU layers, micro-batch, KV type, prompt cache), the environment and the llama-server
    command line, and every line the launcher said while deciding (notes: (text, is_error)). Built by launch_plan(); run_launch_plan() runs it."""
    notes = ()
    autotune = None              # LAUNCHER-994: the auto-tune result (bin/moe_cache_autotune.py) or None
    tune_fallback = None         # LAUNCHER-1045: callable -> (cmd, env) of the plan without the stored pick, or None
    commit_refusal = None        # LAUNCHER-991: why this plan must not be started (Windows commit too small even for the smallest layout), or None

    @property
    def reasons(self):
        return [t for t, e in self.notes if not e]

    @property
    def warnings(self):
        return [t for t, e in self.notes if e]

    @property
    def kv(self):
        return "q8_0" if self.kv_q8 else "f16"

    def numbers(self):
        """ENGINE-029: the figures the control page shows, from this plan alone (the model's layers and sizes from the planner's answer, the placement the launcher
        chose: n_cpu_moe, micro-batch, KV type). None when the planner gave no answer. The page used to combine planner_info, tuned and these fields itself."""
        pl, tuned = self.planner_info, self.tuned
        if not pl:
            return None
        ncmoe = int(self.n_cpu_moe) if str(self.n_cpu_moe).isdigit() else pl["n_cpu_moe"]
        layers = pl["layers"]
        return {"layers": layers, "gpu_expert_layers": max(0, layers - min(ncmoe, layers)), "n_cpu_moe": ncmoe, "cache_GiB": pl["cache_GiB"],
                "cpu_experts_GiB": pl["cpu_experts_GiB"], "free_vram_GiB": pl["free_vram_GiB"],
                "ub": int(self.ub) if str(self.ub).isdigit() else (tuned["ub"] if tuned else None), "kv_cache": self.kv,
                "experts": pl["experts"], "used": pl["used"], "arch": pl["arch"]}

    def command_line(self, windows=None):
        """The line --dry-run prints: environment settings, then the command (PowerShell form on Windows); the API key is never shown."""
        shown = [(k, "(hidden)" if k == "LLAMA_API_KEY" else v) for k, v in self.env + self.key_env]
        if self.machine.windows if windows is None else windows:
            return "".join(f'$env:{k}="{v}"; ' for k, v in shown) + "& " + " ".join(f'"{c}"' if " " in c else c for c in self.cmd)
        return ("env " + " ".join(f"{k}={v}" for k, v in shown) + " " if shown else "") + " ".join(self.cmd)

    def job_limit_note(self):
        return f"moe-cache: (would set a Windows job memory limit of {self.win_limit / 1e9:.1f} GB on llama-server)" if self.win_limit else None

    def dry_run_lines(self):
        """What --dry-run shows, as one list: the reasons (stdout), the job-limit note, the command, then the warnings (stderr)."""
        out = [t for t, e in self.notes if not e]
        if self.job_limit_note():
            out.append(self.job_limit_note())
        out.append(self.command_line())
        out += [t for t, e in self.notes if e]
        return [l for l in "\n".join(out).splitlines() if l.strip()]

    def autotune_dict(self):
        t = self.autotune
        return None if not t else {k: t.get(k) for k in ("state", "picked", "changed", "why", "rows", "total_s", "budget_s", "tuned_at", "locked", "lines", "prompt_tokens", "decode_tokens")}

    def to_dict(self):
        return {"engine": self.engine, "method": self.method, "model": self.model, "llama_server": self.ls, "ctx": self.ctx, "port": self.port,
                "n_cpu_moe": self.n_cpu_moe, "micro_batch": self.ub, "kv_cache": self.kv, "prompt_cache_mib": self.cache_ram,
                "model_bytes": self.size, "memory_bytes": self.avail, "profile": self.profile, "alias": self.alias,
                "windows_job_limit_bytes": self.win_limit, "argv": list(self.cmd),
                "env": [[k, "(hidden)" if k == "LLAMA_API_KEY" else v] for k, v in self.env + self.key_env],
                "command_line": self.command_line(), "reasons": self.reasons, "warnings": self.warnings,
                "notes": [[t, e] for t, e in self.notes], "dry_run_lines": self.dry_run_lines(), "autotune": self.autotune_dict(),
                **({"approx": getattr(self.a, "approx_info")} if getattr(self.a, "approx_info", None) else {})}


def parse_server(argv):
    """-> (options, arguments for llama-server after --, the launcher's own argument list): the command line of `moe_cache.py server`."""
    ap = argparse.ArgumentParser(prog="moe_cache.py server", description="run llama-server with the expert-cache plugin",
                                 epilog=SERVER_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="the .gguf file (leave out to use the one `moe_cache.py pick --apply` chose)")
    ap.add_argument("--project"); ap.add_argument("--profile"); ap.add_argument("--ram", type=gb_arg); ap.add_argument("--vram", type=gb_arg)
    ap.add_argument("--max-ram", type=lambda v: "0" if v.strip() in ("0", "0.0") else gb_arg(v))    # 0 = no cap (the help says so; gb_arg refused it)
    ap.add_argument("--ctx", type=int); ap.add_argument("--n-cpu-moe"); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--llama-server", default=default_llama_server())
    ap.add_argument("--load-method", default="auto", choices=["auto", "preload", "path"])
    ap.add_argument("--gpu-cache", default="off"); ap.add_argument("--gpu-margin", type=float); ap.add_argument("--ub", default="auto")
    ap.add_argument("--fast", nargs="?", const="1", choices=["1", "2", "kv", "fewer"]); ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--leave-room", action="store_true", default=os.environ.get("MOE_CACHE_LEAVE_ROOM", "0") not in ("", "0"))   # I use this PC too
    ap.add_argument("--long-prompt", action="store_true")      # long-prompt mode (v0.3.8): -ub 4096 + --fast=kv when the memory plan allows; since 2026-11-08 also the DEFAULT (see kv_stage); the flag asks for it at any context
    ap.add_argument("--no-long-prompt", action="store_true")   # opt out of the default long-prompt mode: -ub 2048 and the f16 KV cache, as before v0.3.8
    ap.add_argument("--retune", action="store_true")    # LAUNCHER-994: measure the candidate plans again and replace the stored auto-tune pick (bin/moe_cache_autotune.py)
    ap.add_argument("--exact", action="store_true")     # keep the f16 KV cache at 128k+ (default there is --fast=kv, owner 2026-10-30)
    ap.add_argument("--no-ecores", action="store_true"); ap.add_argument("--no-pin", action="store_true"); ap.add_argument("--allow-sleep", action="store_true"); ap.add_argument("--no-sched-tuning", action="store_true", help=argparse.SUPPRESS)      # no-op since v0.3.8 (WINDOWS-013 removed); accepted for one release
    ap.add_argument("--open", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--always-plugin", action="store_true")
    ap.add_argument("--mmproj", metavar="auto|FILE")        # PROTOTYPE (MM6): pass a vision projector and count it in the plan; default off
    ap.add_argument("--cpu-only", action="store_true", help="run on the CPU only: --device none and -ngl 0 (also chosen by itself when llama-server lists no GPU)")
    ap.add_argument("--auto-fit", action="store_true")      # MM6: a dense model gets no -ngl 99, so llama.cpp's --fit places the layers; since the dense route (LAUNCHER-732) this is the default for dense models
    ap.add_argument("--cpu-gpu-split", nargs="?", const="auto", default=None, metavar="auto|FRACTION")   # DECODE-917 (patch 0091): the GPU also computes part of the rows of the RAM-held FFN weights (opt-in, CUDA)
    ap.add_argument("--no-ffn-ram", action="store_true")    # dense models bigger than the GPU: no FFN-weights-in-RAM plan (-ot), llama.cpp's --fit layer split instead
    ap.add_argument("--no-auto-fit", action="store_true")   # dense models: go back to the blind -ngl 99
    ap.add_argument("--no-commit-check", action="store_true")    # LAUNCHER-991: start although the estimate says Windows' free commit is too small (see commit_stage)
    ap.add_argument("--share-ram", action="store_true")     # PROTOTYPE (MM7): count the RAM other llama-servers hold before sizing this one; default off
    ap.add_argument("--peers", action="store_true")         # PROTOTYPE (N2/MM2): servers started with this share one memory cap at run time through a folder of small files; default off
    ap.add_argument("--multi-gpu", action="store_true")     # PROTOTYPE: plan for 2+ GPUs (planner/multigpu.py); default off
    ap.add_argument("--lan", action="store_true")           # SERVER-016: listen on this PC's home-network address, API key forced (bin/moe_cache_lan.py, docs/LAN_ACCESS.md)
    ap.add_argument("--lan-address", default=None)          # with --lan: which private address of this PC (default: the first one found)
    ap.add_argument("--allow-public", action="store_true")  # lets --host / --lan listen on a public (internet) address; off by default
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--agent", action="store_true"); ap.add_argument("--no-warm", action="store_true"); ap.add_argument("--no-starter", action="store_true"); ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"))
    ap.add_argument("--mtp", action="store_true", default=None); ap.add_argument("--no-mtp", dest="mtp", action="store_false")
    ap.add_argument("--no-sampling-defaults", action="store_true")
    ap.add_argument("--approx", choices=["exact", "safe", "careful", "aggressive"], default=None)      # QUALITY-970: opt-in tiers that change answers (bin/moe_cache_approx.py); off by default
    ap.add_argument("--approx-off", action="append", default=[], metavar="MODE"); ap.add_argument("--approx-unverified", action="store_true"); ap.add_argument("--no-approx-canary", action="store_true")
    ap.add_argument("--print-systemd", action="store_true")  # SERVER-005: print docs/moe-cache-server.service filled with this command line and the memory limits, and exit
    ap.add_argument("--plan-json", action="store_true")      # print the launch plan as JSON (argv, environment, reasons, warnings) and exit; nothing is started
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    a = ap.parse_args(fast_levels_only(argv))
    return a, extra, argv


FAST_LEVELS = ("1", "2", "kv", "fewer")


def fast_levels_only(argv):
    """LAUNCHER-004: a bare `--fast` followed by anything that is not a level (1, 2, kv, fewer) means level 1: `--fast model.gguf` must not read the path as the level."""
    return ["--fast=1" if t == "--fast" and (i + 1 >= len(argv) or argv[i + 1] not in FAST_LEVELS) else t for i, t in enumerate(argv)]


SYSTEMD_TEMPLATE = os.path.join(HERE, "docs", "moe-cache-server.service")


def unit_word(w):
    """One word of a systemd ExecStart= line: double-quoted when it has a space or a quote, \\ and \" escaped, % and $ doubled (systemd expands both)."""
    w = str(w).replace("%", "%%").replace("$", "$$")
    return '"' + w.replace("\\", "\\\\").replace('"', '\\"') + '"' if re.search(r"[\s\"'\\;]", w) else w


def systemd_unit(a, extra, argv, template=None):
    """SERVER-005: the text of docs/moe-cache-server.service with this launcher's paths and this command line in ExecStart (absolute model, llama-server and
    launcher paths, because a user unit has no shell PATH), and MemoryMax / MemoryHigh from --ram, else the launcher's own memory cap (MemoryHigh 8 % below
    it). The API key never goes into the unit (a unit file is readable); returns (text, notes)."""
    notes = []
    text = template if template is not None else open(SYSTEMD_TEMPLATE, encoding="utf-8").read()
    args, i = [], 0
    while i < len(argv):
        t = argv[i]
        if t == "--print-systemd":
            pass
        elif t == "--api-key":
            i += 1
            notes.append("--api-key left out of the unit (it would be readable there): put LLAMA_API_KEY=... in a file and add EnvironmentFile=%h/.config/moe-cache/key.env, "
                         "or use the stored key (python3 bin/moe_cache.py key)")
        elif t.startswith("--api-key="):
            notes.append("--api-key left out of the unit (it would be readable there): use EnvironmentFile= or the stored key (python3 bin/moe_cache.py key)")
        else:
            args.append(os.path.abspath(t) if (a.model and t == a.model) else t)
        i += 1
    if extra and "--" not in args:
        args += ["--"] + extra
    if not any(x == "--llama-server" or x.startswith("--llama-server=") for x in args):
        ls = shutil.which(a.llama_server) or a.llama_server
        if os.path.isabs(ls):
            args = args[:1] + ["--llama-server", ls] + args[1:] if args and not args[0].startswith("-") else ["--llama-server", ls] + args
        else:
            notes.append("llama-server was not found: add --llama-server /full/path to the ExecStart line")
    exe = os.path.join(HERE, "bin", "moe-cache-server")
    lines = ["ExecStart=" + " ".join(unit_word(x) for x in [exe] + args)]
    cap = float(a.ram) * 1e9 if a.ram else mem_cap()
    if cap:
        gib = cap / 2**30
        mx = max(1, int(gib + 0.999))
        hi = max(1, int(gib * 0.92))
        lines += [f"MemoryHigh={min(hi, mx)}G", f"MemoryMax={mx}G"]
        notes.append(f"MemoryMax={mx}G is " + (f"--ram {a.ram} GB" if a.ram else "the launcher's memory cap for this machine") + f"; MemoryHigh={min(hi, mx)}G makes the kernel reclaim first; "
                     "moe-cache sizes its cache below the smaller of the two")
    out = []
    for l in text.splitlines():
        if l.startswith("ExecStart="):
            out.append(lines[0])
        elif l.startswith("MemoryHigh=") and len(lines) == 3:
            out.append(lines[1])
        elif l.startswith("MemoryMax=") and len(lines) == 3:
            out.append(lines[2])
        elif l.startswith("# EDIT 1:"):
            out.append("# Generated by moe-cache-server --print-systemd from your command line: the paths are absolute. Check the model and options.")
        elif l.startswith("# EDIT 2:") and len(lines) == 3:
            out.append("# The memory the server may use in total (see the note printed with this unit). MemoryMax is the hard stop, MemoryHigh makes the kernel reclaim first")
        else:
            out.append(l)
    return "\n".join(out) + "\n", notes


def server(argv):
    full_argv = list(argv)
    a, extra, argv = parse_server(argv)
    if a.print_systemd:
        if not a.model:
            raise SystemExit("--print-systemd needs the model: moe-cache-server /path/MODEL.gguf --agent --print-systemd")
        try:
            unit, notes = systemd_unit(a, extra, argv)
        except OSError as e:
            raise SystemExit(f"--print-systemd: the template {SYSTEMD_TEMPLATE} cannot be read ({e})")
        sys.stdout.write(unit)
        for n in notes:
            say("moe-cache: " + n, err=True)
        return 0
    a.autotune_probe = not (a.dry_run or a.plan_json)       # LAUNCHER-994: only a real start measures; everything that just describes the launch reads the stored pick
    plan = launch_plan(a, extra, argv, quiet=a.plan_json)
    if plan.commit_refusal and not a.dry_run and not a.plan_json:
        raise SystemExit(plan.commit_refusal)       # LAUNCHER-991: before anything loads a model, the --approx canary included
    if a.max_ram and not a.plan_json:
        os.environ["MOE_CACHE_MAX_RAM_GB"] = a.max_ram     # this process is the launch: its later memory decisions (OOM retry, warm-keeper threads) keep the cap
    if a.plan_json:
        print(json.dumps(plan.to_dict(), indent=1))
        return 0
    if getattr(a, "approx_info", None) and not a.dry_run:
        final = approx_canary(plan)
        if final and final != a.approx:
            argv_full = list(full_argv)
            i = next((k for k, t in enumerate(argv_full) if t == "--approx" or t.startswith("--approx=")), None)
            if i is not None:
                argv_full[i:i + (1 if "=" in argv_full[i] else 2)] = [f"--approx={final}"]
                a2, extra2, argv2 = parse_server(argv_full)
                plan = launch_plan(a2, extra2, argv2)
                if getattr(a2, "approx_info", None):
                    a2.approx_info["canary"] = f"stepped down from {a.approx} to {final}"
    return run_launch_plan(plan)


def approx_canary(plan):
    """QUALITY-970 startup canary for --approx: about 256 scored tokens (tools/approx_canary.txt: prose, code, French and German, numbers, SQL) through llama-perplexity with this
    launch's own placement flags, against the exact setting of the same model (no route skip, f16 KV cache). The tier is stepped down when the KL divergence or the top-token share
    is past its gate (bin/moe_cache_approx.py GATE_LEVELS). A result is stamped per model file, plugin library and tier, so the next start does not measure again. It guards, it never proves,
    and a canary that cannot run (no llama-perplexity, a failed run) leaves the tier as asked and says so. Returns the tier to run, or None when nothing was checked."""
    import tempfile
    import moe_cache_approx as ax
    a = plan.a
    info = a.approx_info
    if a.no_approx_canary or a.approx == "exact":
        info["canary"] = "off"
        return None
    if a.fast not in {"safe": (None, "kv"), "careful": ("1",), "aggressive": ("2",)}[a.approx]:
        info["canary"] = "skipped: your own --fast replaced the tier's expert mode, so the tier's gate does not describe this run"
        say("moe-cache: --approx: canary skipped (your own --fast replaced the tier's expert mode): nothing was checked")
        return None
    perp = shutil.which("llama-perplexity", path=os.path.dirname(os.path.abspath(plan.ls)) + os.pathsep + os.environ.get("PATH", "")) or shutil.which("llama-perplexity.exe", path=os.path.dirname(os.path.abspath(plan.ls)))
    if not perp or not os.path.isfile(ax.CANARY_TEXT):
        info["canary"] = "skipped: no llama-perplexity next to llama-server"
        say("moe-cache: --approx: canary skipped (llama-perplexity not found next to llama-server): the tier runs as asked, nothing was checked")
        return None
    stampdir = os.path.join(CACHE_DIR, "approx")
    lib = getattr(plan, "lib", None) or ""
    base_env = {k: v for k, v in plan.env if k not in ax.ROUTE_ENV}
    place = ax.model_args_from_cmd(plan.cmd)
    with tempfile.TemporaryDirectory(prefix="approx_canary_") as work:
        raw = ax.make_measure(perp, plan.model, ax.CANARY_TEXT, place, ax.tier_kv, ax.tier_env, base_env, work)

        def measure(tier):
            key = ax.canary_key(plan.model, lib, tier)
            rec = ax.stamp_read(stampdir, key)
            if rec and "kld" in rec:
                return rec["kld"], rec["top1"]
            say(f"moe-cache: --approx {tier}: startup canary (about 256 tokens against the exact setting; the first start of this model and kit only)")
            got = raw(tier)
            if got:
                ax.stamp_write(stampdir, key, {"kld": got[0], "top1": got[1], "tier": tier, "model": os.path.basename(plan.model), "time": int(time.time())})
            return got
        final, hist, note = ax.canary_decide(a.approx, measure)
    for t, kld, top1, ok in hist:
        say(f"moe-cache: --approx {t}: canary mean KL {kld:.4f}, same top token {top1:.1f} % -> " + ("within its gate" if ok else f"PAST its gate (limit {ax.GATE_LEVELS[ax.TIERS[t]['gate']][0]}): stepping down"))
    if note:
        say("moe-cache: --approx: " + note, err=final != a.approx)
    info["canary"] = ("passed" if final == a.approx and not note else note or f"stepped down to {final}")
    info["canary_history"] = [{"tier": t, "kld": k, "top1": p, "ok": ok} for t, k, p, ok in hist]
    return final if hist or note else None


MMPROJ_VRAM_HEADROOM = 0.5e9      # compute buffer of the vision encoder, on top of the projector's file size (estimate; llama.cpp's own memory margin does not count it either)


def find_mmproj(model):
    """The vision projector next to a model: mmproj*.gguf in the same folder, the one that shares the model's name when several; None when not exactly clear."""
    d = os.path.dirname(os.path.abspath(model))
    try:
        found = sorted(f for f in os.listdir(d) if f.lower().startswith("mmproj") and f.lower().endswith(".gguf"))
    except OSError:
        return None
    if len(found) == 1:
        return os.path.join(d, found[0])
    stem = re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", os.path.basename(model)).lower()
    close = [f for f in found if stem and stem.split("-")[0] in f.lower()]
    return os.path.join(d, close[0]) if len(close) == 1 else None


def vision_dense_start(a, extra, model, stock, size, avail):
    """Two jobs. (1) --mmproj (PROTOTYPE, MM6): find the projector (`--mmproj auto` or a file), pass it to llama-server, count its size in the RAM decision
    (added to the model's size) and report it for the VRAM plan. (2) Every launch (LAUNCHER-732): read the GGUF header and detect a dense model (no experts), which gets
    the dense route. Returns (mmproj file, its bytes, dense, stock, size)."""
    sys.path.insert(0, os.path.join(HERE, "planner"))
    mm, mmb = None, 0
    if a.mmproj:
        have = arg_after(extra, "--mmproj", "-mm")
        mm = have or (find_mmproj(model) if a.mmproj == "auto" else a.mmproj)
        if not mm or not os.path.isfile(mm):
            raise SystemExit(f"--mmproj {a.mmproj}: " + ("no single mmproj*.gguf next to the model; pass the file: --mmproj FILE" if a.mmproj == "auto" else f"file not found: {a.mmproj}"))
        mmb = os.path.getsize(mm)
        if not have:
            extra += ["--mmproj", mm]
        size = (size or 0) + mmb
        if not a.stock and not (a.fast or a.hybrid or a.always_plugin) and size and avail:
            stock = size + STOCK_HEADROOM <= avail
        say(f"moe-cache: --mmproj (PROTOTYPE): vision projector {os.path.basename(mm)} ({mmb / 1e9:.2f} GB) counted in the RAM floor and, with {MMPROJ_VRAM_HEADROOM / 1e9:.1f} GB "
            "for its compute buffer, taken off the GPU memory the planner may use; never run with a real vision model by the author")
    dense = False
    try:           # LAUNCHER-732: every launch asks the header, a dense model gets the dense route
        from gguf_plan import model_layout
        lay = model_layout(model)
        dense = not lay["n_layers"] or not lay["expert_bytes_per_layer"]
    except Exception:
        dense = False
    return mm, mmb, dense, stock, size


def other_llama_servers():
    """[(pid, bytes of RAM in use)] of the llama-server processes that are not this launcher's own descendants; the same answer on Linux (/proc) and
    Windows (tasklist)."""
    out, mine = [], {os.getpid(), os.getppid()}
    if WINDOWS:
        try:
            r = subprocess.run([win_tool("tasklist"), "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
            for row in r.stdout.splitlines():
                f = [x.strip('"') for x in row.split('","')]
                if len(f) >= 5 and f[0].lower().startswith("llama-server"):
                    out.append((int(f[1]), int(re.sub(r"[^0-9]", "", f[4]) or 0) * 1024))
        except Exception:
            pass
        return out
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit() or int(d) in mine:
                continue
            try:
                if not open(f"/proc/{d}/comm").read().strip().startswith("llama-server"):
                    continue
                for line in open(f"/proc/{d}/status"):
                    if line.startswith("VmRSS"):
                        out.append((int(d), int(line.split()[1]) * 1024))
            except OSError:
                pass
    except OSError:
        pass
    return out


def share_ram_start(a, machine):
    """MM7 (PROTOTYPE, --share-ram): other llama-servers hold RAM the cap (28 GiB on a 32 GiB PC) is meant to cover once; their use is taken off this server's
    share unless --ram was given. One line says what was found."""
    others = machine.other_llama_servers()
    if not others:
        say("moe-cache: --share-ram (PROTOTYPE): no other llama-server is running")
        return
    held = sum(b for _, b in others)
    cap = machine.mem_cap() or machine.mem_for_server(None) or 0
    mine = machine.mem_for_server(None) or cap
    share = max(min(mine, cap - held) if cap else mine - held, 0)
    msg = f"moe-cache: --share-ram (PROTOTYPE): {len(others)} other llama-server(s) hold {held / 1e9:.1f} GB"
    if a.ram:
        say(msg + f"; --ram {a.ram} GB as given (the cap leaves {share / 1e9:.1f} GB for this one)")
    elif share >= 2e9:
        a.ram = f"{share / 1e9:.1f}"
        say(msg + f": this server may use {a.ram} GB of the {cap / 1e9:.1f} GB cap")
    else:
        say(msg + f": only {share / 1e9:.1f} GB of the cap is left; start with --ram GB to force a size", err=True)


def multi_gpu_start(a, extra, machine, ls, listing, vram_user, win):
    """--multi-gpu, first half (PROTOTYPE): read the cards and, when the planner will be asked for one card's number anyway, make that the card the launcher
    names (MG4: the planner took line 0 of nvidia-smi, the launcher's pick_gpu the card with most free memory). Returns the facts for multi_gpu_apply."""
    say("moe-cache: --multi-gpu is a PROTOTYPE: planned from the device list, not speed-tested on a machine with two discrete GPUs (docs/PLATFORMS.md)")
    sys.path.insert(0, os.path.join(HERE, "planner"))
    import multigpu
    listing = listing if listing is not None else machine.list_devices(ls)
    cards = multigpu.list_cards(listing, machine.win_dedicated_vram() if win else None)
    mg = {"cards": cards, "listing": listing, "vram_user": vram_user, "ls": ls}
    left = multigpu.skipped_igpus(listing, cards)
    if cards:
        say("moe-cache: GPUs: " + "; ".join(f"{c['dev']} {c['name']} ({c['free_mib'] / 1024:.1f} GiB free)" for c in cards) + (f"; left out: {', '.join(left)}" if left else ""))
    if len(cards) >= 2 and not vram_user and not multigpu.user_split_flags(extra):
        c = cards[0]
        a.vram = f"{c['free_mib'] / 1024:.2f}"            # until the split is known, plan with the card the plugin's GPU work will use (the largest, listed first)
    elif len(cards) >= 2 and not vram_user:
        dev = arg_after(extra, "--device", "-dev")
        first = next((c for c in cards if dev and c["dev"] == dev.split(",")[0]), None)
        if first:
            a.vram = f"{first['free_mib'] / 1024:.2f}"     # the user named the card: plan with that one
    return mg


def multi_gpu_apply(mg, a, extra, model, ctx, kv_q8, ub, gcache, ncmoe, ncmoe_user, tuned, machine, win):
    """--multi-gpu, second half: replace the one-card answer with the split plan when 2+ cards are usable and nothing the user chose is in the way.
    Returns (ncmoe, extra)."""
    import multigpu
    cards = mg["cards"]
    why = None
    if len(cards) < 2:
        why = f"{len(cards)} usable GPU(s)"
    elif mg["vram_user"]:
        why = "--vram given (one pool of memory)"
    elif ncmoe_user is not None:
        why = "--n-cpu-moe given"
    elif multigpu.user_split_flags(extra):
        why = "--device / -ts / -sm / --main-gpu given after --"
    elif gcache != "off" or a.hybrid:
        why = "--gpu-cache / --hybrid manage one GPU"
    elif ncmoe == "0" or ncmoe == "999" and not tuned:
        why = "nothing to place"
    if why:
        if len(cards) >= 2 or a.multi_gpu:
            say(f"moe-cache: --multi-gpu: not applied ({why}): the one-card plan stands")
        return ncmoe, extra
    free = [c["free_mib"] * 2**20 for c in cards]
    # (Windows commit: commit_stage counts the whole server after the plan; the old fixed 4 GiB cap of the pool is gone, LAUNCHER-991)
    kw = {"reserve_gb": a.gpu_margin * 2**30 / 1e9} if a.gpu_margin is not None else {}
    res = multigpu.launch_args([dict(c, free_mib=f / 2**20) for c, f in zip(cards, free)], model, ctx, kv_q8=kv_q8, ub=int(ub) if (ub or "").isdigit() else None, **kw)
    if "skip" in res:
        say(f"moe-cache: --multi-gpu: not applied ({res['skip']}): the one-card plan stands")
        return ncmoe, extra
    say("moe-cache: --multi-gpu (PROTOTYPE, estimate not measured): " + res["line"])
    import gguf_plan
    fp = gguf_plan.find_fit_params(mg.get("ls"))
    if fp:       # MG2 (PROTOTYPE): llama.cpp's own projection for the same flags, per card, next to our plan; it only reports, the plan stands
        ubn = int(res["plan"]["ub"]) if res.get("plan") else 1024
        chk = multigpu.fit_check(fp, model, ctx, ubn, res["args"], res["n_cpu_moe"], cards, kv_q8=kv_q8)
        if chk:
            say("moe-cache: --multi-gpu: llama-fit-params projects " + "; ".join(f"{d} {need / 1024:.1f} of {free / 1024:.1f} GiB free" for d, need, free in chk["rows"]) +
                (f"; {', '.join(chk['over'])} would not leave 256 MiB for the plugin's block: use a smaller --ub or context" if chk["over"] else ""), err=bool(chk["over"]))
        else:
            say("moe-cache: --multi-gpu: llama-fit-params gave no figures for this plan; the plan stands")
    return str(res["n_cpu_moe"]), extra + res["args"]


def launch_plan(a, extra, argv, machine=None, quiet=False):
    """Every launch decision in one place (XA-2): stock or plugin, GPU layers, micro-batch, KV type, cache and prompt-cache sizes, environment and the final
    llama-server command line, from the parsed options `a`, the arguments for llama-server after `--` (`extra`, changed in place), the launcher's own
    arguments `argv` and the facts of the machine (`machine`, a Machine; tests pass a fake). Returns a LaunchPlan; the lines the launcher said while
    deciding are in plan.notes, errors (warnings) flagged. quiet=True: nothing is printed (the control page, bench-report, --plan-json read the plan).
    server() runs the plan; --dry-run prints it."""
    machine = machine or Machine()
    notes = []
    recorders = _TL.__dict__.setdefault("notes", [])
    recorders.append(notes)
    old_quiet, old_max_ram = getattr(_TL, "quiet", 0), getattr(_TL, "max_ram", None)
    _TL.quiet = old_quiet + (1 if quiet else 0)
    old_umask = None
    lock = _UMASK_LOCK if quiet and not WINDOWS else None
    if lock:
        lock.acquire()
        old_umask = os.umask(0o077)       # the plan builder sets 0077 for the files a launch writes; a caller that only describes a launch keeps its own
    try:
        plan = _launch_plan(a, extra, argv, machine)
    finally:
        recorders.remove(notes)
        _TL.quiet, _TL.max_ram = old_quiet, old_max_ram
        if lock:
            os.umask(old_umask)
            lock.release()
    plan.notes = notes
    return plan


_PLAN_MODULE = []


def planner_answer(model, ctx, ram_gib=None, vram_gib=None):
    """ENGINE-029: planner/plan.py's answer (the dict `plan.py --json` prints), computed in this process. The launcher used to start a second Python for it (about
    0.06 s of every launch, and a second decision path nobody could step into); plan.py stays runnable as a script for people who call it directly.
    Raises planner.plan.PlanError (a message for the user) or the GGUF reader's ValueError / OSError for a model it cannot read."""
    if not _PLAN_MODULE:
        import importlib.util
        spec = importlib.util.spec_from_file_location("moe_cache_plan", os.path.join(HERE, "planner", "plan.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PLAN_MODULE.append(mod)
    return _PLAN_MODULE[0].plan_info(model, ctx=ctx, ram_gib=ram_gib, vram_gib=vram_gib)


def plan_placement(a, model, ctx, ncmoe, machine):
    """ENGINE-029, stage "placement": how many layers keep their experts on the GPU, from planner/plan.py (the first answer; the tuner below may refine it).
    ncmoe: --n-cpu-moe as the user gave it (None: ask the planner). Returns (n_cpu_moe as text, the planner's answer as a dict or None, why it failed or None);
    the answer carries the model's layers, experts, cache size and free VRAM that the control page shows."""
    planner_info, planner_error = None, None       # planner/plan.py's answer (the control page shows its numbers) or why it failed
    if ncmoe is None:
        ram_gib = float(f"{machine.mem_for_server(a.ram) / 2**30:.3f}") if (a.ram or machine.mem_cap()) else None      # the planner plans with the capped memory too
        vram_gib = float(a.vram) if a.vram else None
        try:
            planner_info = machine.planner_answer(model, ctx, ram_gib, vram_gib)
            ncmoe = str(planner_info["n_cpu_moe"])
        except Exception as e:
            # say why (the Claw's planner failed on /proc/meminfo and this line blamed the gguf package, which was not the cause)
            why = str(e) if type(e).__name__ == "PlanError" else f"{type(e).__name__}: {e}"
            planner_info, planner_error = None, why
            if "does not look like a MoE" in why:       # LAUNCHER-732: a dense model has no experts to place; the old line blamed the planner and talked about experts
                say("moe-cache: this model has no expert tensors (dense): the cache has nothing to manage (llama.cpp places the layers, see below)"); ncmoe = "999"
            else:
                say(f"moe-cache: planner failed ({why}); keeping all experts on the CPU"); ncmoe = "999"
    return ncmoe, planner_info, planner_error


def tune_launch(a, extra, ls, model, ctx, machine, gpu, gcache, kv_q8, mm_bytes):
    """ENGINE-029, stage "tuner": micro-batch, GPU expert cache and the number of static expert layers, from the one shared tuner (planner/gguf_plan.py: tune),
    with the VRAM other programs hold (desk reserve), --gpu-margin and --leave-room counted. Returns (tuned dict or None, the reserve in GB when the desktop
    forced one or None, the micro-batch the user asked for or None)."""
    tuned = None
    desk_kw = None
    ub_asked = a.ub if a.ub != "auto" else arg_after(extra, "-ub", "--ubatch-size")
    if a.ub == "auto" or gcache == "auto" or (ub_asked or "").isdigit():
        try:
            sys.path.insert(0, os.path.join(HERE, "planner"))
            from gguf_plan import tune
            free = float(a.vram) * 2**30 if a.vram else machine.free_vram_bytes()          # GiB, the same unit as the planner's --vram-gib; None: unknown, as the tuner's own probe
            if mm_bytes and free:
                free = max(free - (mm_bytes + MMPROJ_VRAM_HEADROOM), 0.0)       # MM6: the projector's weights and its compute buffer sit in VRAM too
            kw = {"reserve_gb": a.gpu_margin * 2**30 / 1e9} if a.gpu_margin is not None else {}     # default: the tuner's own 0.5 GB
            if a.gpu_margin is None:
                # the margin grows with the VRAM other programs (desktop, browser) already hold: their use grows while the server runs
                from gguf_plan import desk_reserve_gb
                tf = (gpu[3] * 2**20, gpu[2] * 2**20) if (a.vram and gpu and gpu[3] > 0 and abs(float(a.vram) * 1024 - gpu[2]) < 2) else (None if a.vram else machine.gpu_total_free_bytes())      # --vram given by the user: no way to know what others hold
                if tf and getattr(a, "room_used_gib", None) is not None and a.vram and gpu and tf[0]:
                    tf = (tf[0], max(tf[0] - a.room_used_gib * 2**30, 0.0))       # LAUNCHER-1133: "used" = the other programs' dedicated use from the counters, not total minus the driver's budget figure
                if tf:
                    res, used = desk_reserve_gb(*tf)
                    if res > 0.5:
                        desk_kw = kw["reserve_gb"] = res
                        if not getattr(a, "lp_retune", False):
                            say(f"moe-cache: other programs hold {used:.1f} GiB of the GPU memory: keeping {res:.1f} GB free for them and the long-prompt block "
                                f"(--gpu-margin GiB to change)")
                if a.leave_room and kw.get("reserve_gb", 0) < LEAVE_ROOM_VRAM_GB:
                    desk_kw = kw["reserve_gb"] = LEAVE_ROOM_VRAM_GB
                    say(f"moe-cache: --leave-room: {LEAVE_ROOM_VRAM_GB:.0f} GB of GPU memory kept free for the desktop")
            if (ub_asked or "").isdigit():
                kw["ub"] = int(ub_asked)
            from gguf_plan import plan_rule_on
            if plan_rule_on():          # MOE_CACHE_PLAN_RULE=1 (default off): reserve measured by llama-fit-params or from the formula, monotone in -ub
                from gguf_plan import find_fit_params, plan_table, plan_table_lines
                kw["fit_params"] = find_fit_params(ls)
                say("moe-cache: plan rule: compute-buffer reserve " + ("measured with " + kw["fit_params"] if kw["fit_params"] else "from the formula (no llama-fit-params found)"))
                if a.dry_run and free:
                    for line in plan_table_lines(plan_table(model, ctx, free=free, kv_q8=kv_q8, reserve_gb=kw.get("reserve_gb", 0.5), fit_params=kw["fit_params"]), ctx):
                        say(line)
            if getattr(a, "long_prompt", False) and not (ub_asked or "").isdigit() and getattr(a, "lp_kv", False):
                kw["long_prompt"] = True
            tuned = tune(model, ctx, free=free, want_cache=gcache != "off", kv_q8=kv_q8, **kw)
        except Exception:
            tuned = None
    return tuned, desk_kw, ub_asked


def lan_stage(a, extra):
    """SERVER-016, stage "network address": --lan serves the home network (the server listens on all interfaces because the launcher and the page reach it through
    127.0.0.1; the address told to the user is the PC's private one), and any explicit --host is checked: a public address is refused without --allow-public. The
    address rules live in bin/moe_cache_lan.py. extra gets "--host 0.0.0.0" for --lan. Returns the address to tell the user (None without --lan)."""
    lan_addr = None
    if a.lan or arg_after(extra, "--host"):
        import moe_cache_lan as lan
        addrs = lan.local_addresses()
        if a.lan:
            if arg_after(extra, "--host"):
                raise SystemExit("--lan chooses the address itself: remove --host (or name the address with --lan-address ADDR)")
            lan_addr, err = lan.pick_address(addrs, a.lan_address)
            if err:
                raise SystemExit("--lan: " + err)
            extra += ["--host", "0.0.0.0"]
        err = lan.check_bind(arg_after(extra, "--host"), addrs, a.allow_public) if arg_after(extra, "--host") else None
        if err:
            raise SystemExit("--host: " + err if not a.lan else "--lan: " + err)
    return lan_addr


def api_key_stage(a, extra, lan_addr=None):
    """ENGINE-029, stage "API key": a key given on our command line or after `--` goes to llama-server through the environment (LLAMA_API_KEY), never its argument list; agent mode, --project, --lan and any other non-loopback --host always get the stable per-user key; plain chat servers get none, but a CORS allow-list. Returns (api_key or None, the environment settings for it)."""
    api_key = take_api_key_from_args(extra) or a.api_key
    import moe_cache_lan as lan
    host_open = bool(arg_after(extra, "--host")) and not lan.is_loopback_host(arg_after(extra, "--host"))        # ANY non-loopback bind gets a key, --lan or not
    if not api_key and (a.agent or a.project or a.lan or host_open):
        api_key = stored_key(create=True)
        say(f"moe-cache: this server needs an API key ({'--lan: other computers can reach it' if a.lan else '--host ' + arg_after(extra, '--host') + ': other computers can reach it' if host_open else 'agent mode / projects keep conversations on disk'}); it is stored in {KEY_FILE} "
            "(`python3 bin/moe_cache.py key` prints it; `connect hermes|opencode --write` puts it into the agent's config)")
    if a.lan:
        for line in lan.risk_lines(f"http://{lan_addr}:{a.port}", KEY_FILE if api_key == stored_key() else None):
            say(line)
    key_env = [("LLAMA_API_KEY", api_key)] if api_key else []
    if "--cors-origins" not in extra and "LLAMA_ARG_CORS_ORIGINS" not in os.environ:
        # llama-server answers any website's cross-origin request by default (it echoes the Origin back); `localhost` allows only local pages.
        # Passed as the environment variable so a llama-server without the flag just ignores it
        key_env.append(("LLAMA_ARG_CORS_ORIGINS", "localhost"))
    return api_key, key_env


def agent_stage(a, extra, model):
    """ENGINE-029, stage "agent flags": --jinja, recurrent checkpoints, slot similarity, the alias and (without --no-warm / --project) the folder for saved conversations. extra grows in place. Returns (alias or None, slot folder or None)."""
    alias = None
    slotdir = None
    if a.agent:
        if not any(x in ("--jinja", "--no-jinja") for x in extra):
            extra.append("--jinja")         # the default in current llama.cpp; explicit for older builds (tool calls need the model's own template)
        if arg_after(extra, "-ctxcp", "--ctx-checkpoints", "--swa-checkpoints") is None:
            # models with recurrent layers keep a copy of that state per checkpoint in host RAM (about 60 MiB on Qwen3.6-35B), outside the cache's budget
            # idea S03 (MOE_CACHE_CKPT=N, default 8): how many recurrent checkpoints each conversation keeps; more cost host RAM, fewer cost reprocessing
            ck = os.environ.get("MOE_CACHE_CKPT", "8")
            extra += ["--ctx-checkpoints", ck if ck.isdigit() else "8"]
        # idea S02 (MOE_CACHE_SLOT_SIMILARITY=0..1, default unset = the server's own): a request takes a busy conversation's slot only if its prompt is at least
        # this similar to what the slot holds, so a short request from another agent goes to a free slot instead of displacing a long conversation
        sps = os.environ.get("MOE_CACHE_SLOT_SIMILARITY")
        if sps and arg_after(extra, "-sps", "--slot-prompt-similarity") is None:
            try:
                if 0.0 <= float(sps) <= 1.0:
                    extra += ["--slot-prompt-similarity", sps]
            except ValueError:
                pass
        alias = arg_after(extra, "-a", "--alias")
        if alias is None:
            alias = default_alias(model)
            extra += ["--alias", alias]
        if not a.no_warm and not a.project:
            if arg_after(extra, "--slot-save-path") is None:
                slotdir = os.path.join(CACHE_DIR, "slots", "agent"); os.makedirs(slotdir, exist_ok=True)
                extra += ["--slot-save-path", slotdir]
            else:
                slotdir = arg_after(extra, "--slot-save-path")
    return alias, slotdir


def commit_check_on(a, machine):
    """LAUNCHER-991: the whole-server commit estimate (commit_stage) is active: Windows, and neither --no-commit-check nor MOE_CACHE_COMMIT_CHECK=0. It then owns the
    prompt-cache shrink too (the older commit_plan rule stays for --no-commit-check)."""
    return bool(machine.windows and not a.no_commit_check and os.environ.get("MOE_CACHE_COMMIT_CHECK") != "0")


_GPU_PROC = [0.0, ""]
GPU_PROC_TTL_S = 15.0       # the control page plans every model of its list: one PowerShell read serves them all (a plan only describes; a start a few seconds later sees the same desktop)


def win_gpu_process_text():
    """LAUNCHER-1133: every process's dedicated GPU memory now (Windows counters, one PowerShell read, kept GPU_PROC_TTL_S seconds); '' when it fails (not kept). Its own function so the
    recorded launch-plan tests can replace it."""
    now = time.monotonic()
    if _GPU_PROC[1] and now - _GPU_PROC[0] < GPU_PROC_TTL_S:
        return _GPU_PROC[1]
    sys.path.insert(0, os.path.join(HERE, "planner"))
    from gguf_plan import counters_text
    text = counters_text()
    if text:
        _GPU_PROC[0], _GPU_PROC[1] = now, text
    return text


def real_room_stage(a, g, machine):
    """LAUNCHER-1133 (Windows AMD / Intel, --vram not given; build's LAUNCHER-1131 WIP, pre-load half only): llama-server's free figure is the Vulkan budget; with the desktop on the card the
    process keeps only its dedicated share and the rest is paged silently (PC 2026-11-11, RX 9070 XT: 15.4 GiB "free", 13.1-13.5 GiB kept; q8_0 KV at 64k ran 18.5 tok/s against 27.8 with q4_0).
    Sets a.room_gib = the card's size minus the other programs' dedicated use minus a reserve when that is smaller than the driver's figure (the dense fit plans against it) and a.room_used_gib
    (what the desk reserve counts as used). The counters unreadable: the driver's figure as before, said in one line. Never raises."""
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import others_dedicated_mib, real_room_mib, ROOM_RESERVE_MIB
        total, driver = float(g[3]), float(g[2])
        others = others_dedicated_mib(machine.gpu_process_text())
        if others is None:
            say("moe-cache: GPU room: the other programs' GPU memory use could not be read (Windows counters): planning with llama-server's free figure as before")
            return
        a.room_used_gib = others / 1024
        room = real_room_mib(total, others)
        if room is None or room >= driver - 256:
            return
        a.room_gib = room / 1024
        say(f"moe-cache: GPU room: planning with {a.room_gib:.2f} GiB, not llama-server's {driver / 1024:.2f} GiB (the desktop and other programs hold {others / 1024:.1f} GiB of the card's "
            f"{total / 1024:.1f} GiB, and a reserve of {ROOM_RESERVE_MIB} MiB stays free); --vram GiB sets it by hand")
    except Exception:
        pass


def gpu_stage(a, ls, win, machine):
    """ENGINE-029, stage "GPU facts": which GPU llama-server will use and how much of its memory is free. On Windows with an AMD or Intel GPU nothing else answers, so --list-devices does (an integrated GPU plans without dedicated VRAM; Vulkan allocations are charged to the free commit). Sets a.vram when it found the figure. Returns (the --vram the user gave, the device listing or None, the picked GPU or None)."""
    vram_user = a.vram
    listing = machine.list_devices(ls) if win else None
    gpu = machine.pick_gpu_full(listing) if win else None
    if win and not a.vram and not machine.nvidia_reports():
        # the planner reads free VRAM from nvidia-smi or Linux sysfs; on Windows with an AMD or Intel GPU neither answers and it would plan without a GPU
        # (nvidia-smi.exe can stay installed after an NVIDIA card is gone: seen on the desktop PC)
        g = gpu
        ded = dedicated_vram_of(g[1], machine.win_dedicated_vram()) if g else None
        if g and ded is not None and ded < 2**30:
            # an integrated GPU (the Claw's Arc 140V, Ryzen APUs): Vulkan's "free" is shared system RAM, which the cache also counts;
            # plan with the dedicated carve-out only, as Linux does (planner/plan.py amd_free)
            say(f"moe-cache: GPU {g[1]} ({g[0]}) is integrated ({ded / 2**20:.0f} MiB of its own memory): planning without dedicated VRAM")
            g = None
        if g and g[2] > 0:
            a.vram = f"{g[2] / 1024:.2f}"
            say(f"moe-cache: GPU {g[1]} ({g[0]}): {a.vram} GiB free (llama-server --list-devices; --vram GiB to override)")
            real_room_stage(a, g, machine)
            # (Vulkan's GPU allocations are charged to Windows commit on AMD / Intel drivers: commit_stage estimates the whole server after the plan; LAUNCHER-991
            # replaced the old cap of the GPU plan, free commit minus a fixed 4 GiB, which ignored the KV cache and the compute buffers)
    return vram_user, listing, gpu


def dense_fit_free(a, machine):
    """(free GPU bytes, gap in GiB) for the dense fit estimate: --vram or the figure llama-server's own --list-devices gave (Windows AMD / Intel; no gap) else nvidia-smi / sysfs
    (SMI_GAP_GIB too high, see planner/gguf_plan.py); (None, 0) without a GPU figure."""
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import SMI_GAP_GIB
        if a.vram:
            room = getattr(a, "room_gib", None)           # LAUNCHER-1133: the room the card really gives (gpu_stage; never set for the user's own --vram)
            return (min(float(a.vram), room) if room else float(a.vram)) * 2**30, 0.0
        f = machine.free_vram_bytes()
        return (f, SMI_GAP_GIB) if f else (None, 0.0)
    except Exception:
        return None, 0.0


def dense_gpu_fit_pair(a, model, ctx, machine, with_q4=False):
    """LAUNCHER-982: (f16 fit, q8_0 fit) of a model without experts at `ctx` from planner/gguf_plan.py dense_gpu_fit(); None when the layout or the GPU memory is unknown.
    with_q4 (LAUNCHER-1063): the whole ladder (f16, q8_0, K q8_0 / V q4_0, q4_0 fit).
    Constants: with llama-server's own free-memory figure (--vram, or --list-devices on Windows AMD / Intel) compute buffer + context 0.20 GiB and no margin, calibrated on the PC
    (Qwen3.8-27B Q3_K_XL on the 16 GB RX 9070 XT, 14.65 GiB free, 2026-11-08: f16 KV at 32k and q8_0 KV at 64k run all on the GPU at 26.5 / 26.6 tok/s, f16 at 64k spills to 18.9);
    with nvidia-smi / sysfs the CUDA-measured 0.30 GiB + 0.25 GiB margin and the 0.4 GiB reporting gap."""
    free, gap = dense_fit_free(a, machine)
    if not free:
        return None
    try:
        from gguf_plan import dense_gpu_fit
        if a.vram:
            fixed, m = 0.20, (a.gpu_margin if a.gpu_margin is not None else 0.0)
        else:
            fixed, m = None, (a.gpu_margin if a.gpu_margin is not None else 0.25)
        f16 = dense_gpu_fit(model, ctx, free, False, m, gap, fixed)
        q8 = dense_gpu_fit(model, ctx, free, True, m, gap, fixed)
        kq8vq4 = dense_gpu_fit(model, ctx, free, KV_K8V4, m, gap, fixed) if with_q4 else True
        q4 = dense_gpu_fit(model, ctx, free, "q4_0", m, gap, fixed) if with_q4 else True
    except Exception:
        return None
    if not (f16 and q8 and kq8vq4 and q4):
        return None
    return (f16, q8, kq8vq4, q4) if with_q4 else (f16, q8)


def kv_quantized(cmd):
    """True when the command stores the KV cache in a quantized type (-ctk / -ctv: q8_0, q4_0 or a mix; LAUNCHER-1063 review: callers compared -ctk with "q8_0" only)."""
    return bool(kv_plan_type(cmd, False))


KV_K8V4 = ("q8_0", "q4_0")          # LAUNCHER-1063: K q8_0 / V q4_0 (laptop, 27B, against f16: top-1 94.6 % between q8/q8 96.5 % and q4/q4 92.9 %: K matters more than V)


def kv_plan_type(extra, kv_q8=False):
    """The KV type the memory planner sizes, from -ctk / -ctv in the command (a type missing there is f16, llama.cpp's default) or the launcher's own q8_0 flag:
    False (f16 both), True (q8_0 both), "q4_0" (both), or a (K, V) pair for any other mix."""
    k, v = arg_after(extra, "-ctk", "--cache-type-k"), arg_after(extra, "-ctv", "--cache-type-v")
    if k is None and v is None:
        return bool(kv_q8)
    k, v = k or "f16", v or "f16"
    if k == v:
        return {"f16": False, "q8_0": True, "q4_0": "q4_0"}.get(k, False)
    return (k, v)


def kv_fit_index(kv_k, kv_v=None):
    """Which entry of the dense_gpu_fit_pair(..., with_q4=True) ladder (f16, q8_0/q8_0, K q8_0 / V q4_0, q4_0/q4_0) sizes the KV types in the command; anything else is f16 (0).
    A V type that is not given counts as the K type for the quantized ones (the launcher always gives both)."""
    return {("q8_0", "q8_0"): 1, ("q8_0", "q4_0"): 2, ("q4_0", "q4_0"): 3}.get((kv_k or "f16", kv_v or kv_k or "f16"), 0)


def load_mode_fit_none(a, cmd, extra, machine, ctx, pair, kv_name, kv_v=None):
    """LAUNCHER-1068: Windows, a dense model that fits fully on the GPU (the LAUNCHER-982 estimate, not --fit spilling, no -ot) loads with `--load-mode none`: the default mmap load
    spiked used RAM by ~12 GB for Qwen3.8-27B on the PC (2026-11-10), `--load-mode none` added 2-6 GB and was ready in under a minute. Returns the plain note, or None (mmap stays):
    Linux, --exact / --stock, a layout that spills to RAM (mmap keeps those weights file-backed), an unknown fit, or a load flag of your own (--load-mode / -lm / --mmap / --no-mmap / --mlock)."""
    if not machine.windows or a.exact or a.stock or not pair or "-ot" in cmd or arg_after(cmd, "-ngl", "--n-gpu-layers") == "0":
        return None
    if any(x in ("--load-mode", "-lm", "--mmap", "--no-mmap", "--mlock") or x.startswith(("--load-mode=", "-lm=")) for x in extra):
        return None
    i = kv_fit_index(kv_name, kv_v)          # LAUNCHER-1063: the mix and q4_0/q4_0 are sized as themselves (a 2-entry pair only knows f16 and q8_0: q4_0 then counts as q8_0, errs high)
    r = pair[min(i, len(pair) - 1)] if pair else None
    if not r or not r["fits"]:
        return None
    return "the model fits fully on the GPU: --load-mode none (no memory-mapped copy of the file in RAM while loading; --exact or --load-mode after -- keeps llama.cpp's default)"


def gpu_fit_words(r, kv_name):
    """The numbers behind a fit estimate, plain: model + KV + compute (+ margin) = need of free."""
    return (f"model {r['model_gib']:.1f} + {kv_name} KV {r['kv_gib']:.1f} + compute {r['fixed_gib']:.1f}" + (f" + margin {r['margin_gib']:.2f}" if r['margin_gib'] else "")
            + f" = {r['need_gib']:.1f} GiB of {r['free_gib']:.1f} GiB free")


EXPECT_CPU_WORK = [False]     # set by command_stage: the plan itself puts work on the CPU (the log scan then stays quiet)
CPU_ATTN_RE = re.compile(r"layer\s+(\d+) is assigned to device (\S+) but Flash Attention over K/V rows is assigned to device CPU")
OFFLOADED_RE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")


def cpu_placement_note(line, cpu_run=False):
    """LAUNCHER-983: (key, warning) for a llama-server log line that says work was placed on the CPU although a GPU is used, else None. cpu_run: the launcher itself asked for the
    CPU (--cpu-only / --device none / -ngl 0), so 'offloaded 0/N' is expected."""
    text = line.decode(errors="replace") if isinstance(line, bytes) else line
    m = CPU_ATTN_RE.search(text)
    if m:
        return ("attn", f"moe-cache: WARNING: llama-server put the attention over the KV cache of layer {m.group(1)} (weights on {m.group(2)}) on the CPU: the KV cache did not fit in GPU "
                        "memory next to the model, and decode will be much slower (Qwen3.8-27B on the 16 GB RX 9070 XT at 64k: 35 -> 19 tok/s). A smaller --ctx or an 8-bit KV cache "
                        "keeps it on the GPU; the 'all on GPU' line above says by how much it was short")
    m = OFFLOADED_RE.search(text)
    if m and not cpu_run and int(m.group(1)) < int(m.group(2)):
        return ("offload", f"moe-cache: WARNING: only {m.group(1)} of {m.group(2)} layers are on the GPU: the rest run on the CPU and decode will be slower. "
                           "Free GPU memory, use a smaller --ctx or an 8-bit KV cache")
    return None


IGPU_NAME_RE = re.compile(r"(Radeon\(TM\) Graphics|Radeon\s+\d{3}M|UHD Graphics|Iris|Intel\(R\) Graphics|Graphics$)", re.I)


def auto_gpu_device(extra, listing, cpu_only=False, multi_gpu=False):
    """LAUNCHER-983 (owner's choice 2026-11-09): the --device value that leaves integrated GPUs out when the listing has at least one integrated GPU (by name: 'Radeon(TM) Graphics',
    'UHD Graphics', 'Iris') AND at least one card that is not: (ids joined by commas, names, all listed ids). The plan, the control page and now the server use the same cards; two real
    cards still split as llama.cpp does by default. None when there is nothing to leave out, an integrated GPU is the only kind (used as before), the user chose (--device / -dev / -ts /
    -sm / --main-gpu), or the CPU / the multi-GPU prototype is wanted."""
    if cpu_only or multi_gpu or not listing:
        return None
    if any(x in ("--device", "-dev", "-ts", "--tensor-split", "-sm", "--split-mode", "--main-gpu", "-mg") or x.startswith(("--device=", "--tensor-split=", "--split-mode=", "--main-gpu=")) for x in extra):
        return None
    devs = re.findall(r"^\s+((?:CUDA|Vulkan|ROCm|HIP|SYCL)\d*): (.+?) \((\d+) MiB, (\d+) MiB free\)", listing, re.M)
    igpu = [d for d in devs if IGPU_NAME_RE.search(d[1].strip())]
    real = [d for d in devs if not IGPU_NAME_RE.search(d[1].strip())]
    if not igpu or not real:
        return None
    return ",".join(d[0] for d in real), ", ".join(d[1].strip() for d in real), [d[0] for d in devs]


def plugin_for_partial_offload(a, ls, model, ctx, machine):
    """v0.3.8, laptop evidence (RTX 4060 8 GB, Qwen3.6-35B-A3B IQ3_XXS, 30 expert layers on the CPU, 32k chat, A B B A): a model that fits in RAM but not in VRAM decodes
    faster through the plugin's executor than through stock llama-server: +10 to +29 % in all six paired band comparisons (mean +15 %; 8k probe +26 %), prompt speed about 10 %
    lower (-22 % in the short-answer chat); NOT reproduced on the release code (audit 2026-11-12: decode -4 %, prompt -19 %): the user-facing text no longer claims the gain (LAUNCHER-1129). Returns the line to say when the plugin should run although the model fits in RAM, else None. --stock turns it off.
    Stock is kept when: --stock, --cpu-only, MOE_CACHE_STOCK_WHEN_FITS=1, no plugin build that loads here (a missing MOE_CACHE_LIB counts), a dense or unreadable model,
    or every expert fits the GPU (n_cpu_moe 0).
    NO-GPU MACHINES (owner decision 2026-11-08): deliberately NOT excluded. With no GPU (or --vram 0) the planner puts every expert layer on the CPU, so n_cpu_moe > 0 and
    the plugin runs; only an explicit --cpu-only / --stock / MOE_CACHE_STOCK_WHEN_FITS=1 gives stock. The laptop evidence above has a GPU; the no-GPU case rides on the same
    default until a measurement says otherwise. Do not add a GPU requirement here without a new owner decision (tests/planner_test.py pins this)."""
    if a.stock or a.cpu_only or os.environ.get("MOE_CACHE_STOCK_WHEN_FITS") == "1":
        return None
    try:
        # the same library plugin_lib_stage will pick: MOE_CACHE_LIB first (a missing file there means stock, not "ask for the plugin"), else the build that matches the server
        want = machine.ls_api(ls)
        lib = os.environ.get("MOE_CACHE_LIB") or machine.pick_lib(ls)
        have = machine.lib_api(lib)
        if want and have and want != have:
            lib = machine.pick_lib(ls)                    # plugin_lib_stage lets the matching build replace an env value built for another API
            have = machine.lib_api(lib)
        if not machine.lib_exists(lib) or (want and have and want != have):
            return None                                   # no matching plugin build here: stock stays (this only ever adds the plugin when it can load)
        ram_gib = float(f"{machine.mem_for_server(a.ram) / 2**30:.3f}") if (a.ram or machine.mem_cap()) else None
        info = machine.planner_answer(model, ctx, ram_gib, float(a.vram) if a.vram else None)
        ncmoe = int(info["n_cpu_moe"])
    except Exception:
        return None                                       # dense, unreadable or no planner answer: stock stays
    if ncmoe <= 0:
        return None                                       # every expert fits the GPU: nothing for the executor to do
    return ("moe-cache: the model fits in memory but some layers keep their experts on the CPU: the plugin's expert executor runs those layers (on our test laptop with one model it was faster "
            "than stock llama-server, but a later re-run on the release code did not reproduce that, so on your machine it can be slower). --stock runs plain llama-server instead (MOE_CACHE_STOCK_WHEN_FITS=1 does the same)")


def plugin_lib_stage(ls, dry, stock, win, machine):
    """ENGINE-029, stage "plugin build": which plugin library goes with this llama-server (MOE_CACHE_LIB gives way to the matching build when its backend API differs), and the refusals: llama-server missing, plugin not built, API mismatch. Returns the library path."""
    lib = os.environ.get("MOE_CACHE_LIB")
    if lib:
        # moe-cache.bat sets MOE_CACHE_LIB to the first DLL it finds (the api3 one); for an older llama-server (ggml 0.24 / 0.25) that build refuses to load.
        # An env value built for another API than the server uses gives way to the matching build the launcher finds itself (audit A4-01)
        want, have = machine.ls_api(ls), machine.lib_api(lib)
        if want and have and want != have:
            alt = machine.pick_lib(ls)
            if alt != lib and machine.lib_exists(alt) and machine.lib_api(alt) == want:
                say(f"moe-cache: MOE_CACHE_LIB {lib} was built for ggml backend API {have}, {ls} uses API {want}: using {alt} instead", err=True)
                lib = alt
    else:
        lib = machine.pick_lib(ls)
    if not dry and not (os.path.isfile(ls) or shutil.which(ls)):
        # before the plugin check: without llama-server the build that fits is unknown, and "plugin not built: run ./build.sh" sent kit and
        # tarball users the wrong way (audit 2026-10-29 K2)
        raise SystemExit(f"moe-cache: llama-server not found: {ls}. Pass --llama-server PATH or set LLAMA_SERVER"
                         + (r' (once for good: setx LLAMA_SERVER "D:\path\to\llama-server.exe", then a new window; or unpack llama.cpp to C:\llama; the control page asks for the path and remembers it)' if win else " (Linux: the -ggml0.26 download has llama.cpp inside, or run ./install.sh; with your own llama.cpp the file is build/bin/llama-server)")
                         + f". Looked: {machine.ls_looked()}. The official release: {RELEASES_URL}. (moe_cache.py check tells what is missing)")
    if (line := machine.ls_found_line(ls)):
        say(line)
    if not stock:     # a model that fits runs stock llama-server: no plugin build needed (Bolt on the PC failed here overnight 2026-10-29, T5)
        want, have = machine.ls_api(ls), machine.lib_api(lib)
        if want and have and want != have:
            say(f"moe-cache: WARNING: {lib} was built for ggml backend API {have} but {ls} uses API {want}: the plugin will refuse to load; "
                + api_advice(want), err=True)
        if not machine.lib_exists(lib):
            if os.environ.get("MOE_CACHE_LIB"):
                raise SystemExit(f"MOE_CACHE_LIB points to a file that does not exist: {lib}")
            raise SystemExit(f"plugin not built: run ./build.sh first (looked for {lib})")
    return lib


def profile_stage(a, model, pname, dry):
    """ENGINE-029, stage "usage profile": the path of the profile this model learns into (idea I18: with none yet, the shipped one for the model family is copied there, a learned profile made the first request 2.1-2.4x faster, RESULTS 16). Returns the path."""
    profile = model_profile(model, pname, dry=dry)
    if not os.path.isfile(profile) and not a.no_starter and not dry:
        # idea I18: no profile yet: start from the shipped one for this model family (a learned profile made the first request 2.1-2.4x faster, RESULTS 16)
        try:
            sys.path.insert(0, os.path.join(HERE, "planner"))
            from gguf_plan import model_key
            key = model_key(model)
            starter = os.path.join(HERE, "profiles", key + ".usage") if key else None
            if starter and os.path.isfile(starter):
                os.makedirs(CACHE_DIR, exist_ok=True)
                shutil.copyfile(starter, profile)
                say(f"moe-cache: no usage profile yet: starting from the shipped one for {key} (it learns from your prompts from now on)")
        except Exception as e:
            say(f"moe-cache: could not use a starter profile ({e})", err=True)
    return profile


def memory_env_stage(a, ls, lib, model, profile, stock, gcache, machine):
    """ENGINE-029, stage "plugin environment": how the plugin is loaded (preload or backend path), the cache and memory settings of the plugin in environment form (RAM, cap, peers) and the memory cap the plan prints. --gpu-cache falls back to static layers when the plugin can only be loaded as a file. Returns (load settings, load method, gpu cache setting, environment, memory cap or None)."""
    load, method = ([], "none (stock)") if stock else machine.load_env(ls, lib, a.load_method)
    if gcache != "off" and not stock and method == "path":
        # the plugin can feed llama.cpp's GPU cache only when preloaded; loaded as a file it would let the GPU read experts that are not in RAM (T7)
        say("moe-cache: --gpu-cache needs a llama-server that links its backends directly (LD_PRELOAD loading); this one loads them as files "
            "(GGML_BACKEND_PATH): using static GPU expert layers instead", err=True)
        gcache = "off"
    env = load + [("MOE_CACHE_SIZE_GIB", "auto"), ("MOE_CACHE_GGUF", model), ("MOE_CACHE_PROFILE", profile)]
    if a.ram:
        env.append(("MOE_CACHE_RAM_GIB", ram_gib(a.ram)))
    cap = machine.mem_cap()
    if a.max_ram:
        env.append(("MOE_CACHE_MAX_RAM_GB", a.max_ram))
    if a.peers and not stock:
        # N2/MM2 (PROTOTYPE): the plugin writes its memory use into this folder once a second and reads the other servers' files; a quiet server gives memory to a busy
        # one. --ram stays this server's own share. A stock llama-server (no plugin) is not in the folder: --share-ram counts it at start.
        peers_dir = os.path.join(CACHE_DIR, "peers")
        env.append(("MOE_CACHE_PEERS", peers_dir))
        if cap:
            env.append(("MOE_CACHE_PEER_CAP_GIB", f"{cap / 2**30:.2f}"))
        say(f"moe-cache: --peers is a PROTOTYPE: this server shares {(cap or 0) / 2**30:.1f} GiB with every moe-cache server that names {peers_dir}; not run with two real servers yet (docs/PLATFORMS.md)")
    if cap:
        say(f"moe-cache: memory cap {cap / 2**30:.1f} GiB ({cap / 1e9:.1f} GB) for the whole server (" + ("--max-ram" if a.max_ram else "MOE_CACHE_MAX_RAM_GB" if os.environ.get("MOE_CACHE_MAX_RAM_GB")
            else "physical RAM minus max(4 GiB, 12.5 %)") + "; --max-ram GB changes it)")
        if a.ram and float(a.ram) * 1e9 > cap:
            say(f"moe-cache: --ram {a.ram} GB is above the memory cap: the cap ({cap / 1e9:.1f} GB) wins", err=True)
    return load, method, gcache, env, cap


def approx_stage(a, model):
    """QUALITY-970, stage "approximate tier": --approx safe|careful|aggressive turns on a group of the switches below (bin/moe_cache_approx.py): a.fast, the 8-bit KV cache
    (a.approx_kv) and the model's own MTP drafting (a.approx_spec). Off unless asked. Your own --fast wins; --exact and --stock refuse. Sets a.approx_info (the control page and --plan-json show it)."""
    a.approx_kv = a.approx_spec = False
    a.approx_info = None
    if a.exact and (a.fast or a.mtp) and not a.long_prompt:          # (--long-prompt --exact is refused with its own message in kv_stage)
        raise SystemExit(f"--exact does not go with {'--fast' if a.fast else '--mtp'}: --exact turns every mode off that changes tokens and gives the answers of plain llama.cpp")
    if not a.fast and not a.approx and not a.exact and not a.stock:         # LAUNCHER-984: an expert-skipping row marked "won" in TOKEN_MODES is on by default (none is today)
        import moe_cache_approx as ax0
        lvl = ax0.default_expert_level()
        if lvl:
            a.fast = lvl
            say(f"moe-cache: --fast={lvl} is on by default (measured as a win, TOKEN_MODES in bin/moe_cache_approx.py); --exact turns it off")
    if not a.approx:
        if a.approx_off or a.approx_unverified:
            raise SystemExit("--approx-off / --approx-unverified need --approx safe|careful|aggressive")
        return
    import moe_cache_approx as ax
    if a.approx != "exact" and (a.exact or a.stock):
        raise SystemExit(f"--approx {a.approx} does not go with --{'exact' if a.exact else 'stock'}: the tier changes the answers, that option asks for the same answers as stock")
    try:
        r = ax.resolve(a.approx, os.path.basename(model), off=a.approx_off, unverified=a.approx_unverified, user_fast=a.fast)
    except ax.ApproxError as e:
        raise SystemExit(str(e))
    if r["fast"] and not a.fast:
        a.fast = r["fast"]
    a.approx_kv, a.approx_spec = r["kv"], r["spec"]
    if r["kv"] and not a.fast and not getattr(a, "long_prompt", False) and not a.no_long_prompt:
        a.long_prompt = True              # safe keeps the long-prompt mode where it applies (-ub 4096 with the same 8-bit KV cache); kv_stage turns the KV cache on anyway where it does not
    for l in r["lines"]:
        say(l)
    a.approx_info = {"tier": r["tier"], "gate": r["gate"], "modes": r["modes"], "canary": None}


def kv_stage(a, extra, ctx, ls, listing, machine, stock=False, dense=False):
    """ENGINE-029, stage "KV cache": q8_0 is the default from 128k context on a GPU machine (owner 2026-10-30; not with --stock, --exact or your own -ctk/-ctv); --fast also stores it as q8_0 (half the KV memory, which at agent-sized contexts is the VRAM a bigger micro-batch needs). Sets a.fast = "kv" in the default case and adds -ctk/-ctv to extra. Returns whether the KV cache is q8_0."""
    # --no-long-prompt gives "-ub 2048 and the f16 KV cache" (SERVER_HELP), so it also switches the 128k q8_0 default off (fixed 2026-11-08: it used to leave q8_0 on at >= 128k)
    import moe_cache_approx as ax          # LAUNCHER-984: the defaults below follow the status rows of TOKEN_MODES ("won" = on by default)
    kv128_default = False
    if (not a.fast and not a.exact and not a.stock and not getattr(a, "no_long_prompt", False) and ctx >= 131072 and ax.default_on("kv8")
            and not any(x in ("-ctk", "--cache-type-k", "-ctv", "--cache-type-v") for x in extra)
            and (a.vram or machine.pick_gpu(listing if listing is not None else machine.list_devices(ls)))):     # a GPU: listed, or --vram given
        a.fast = "kv"
        kv128_default = True
        say(f"moe-cache: context {ctx // 1024}k: q8_0 KV cache by default (--fast=kv: about 2x prompt speed at 256k on a 16 GB card; it passed the "
            "quality gate at 4k context, docs/QUALITY_GATE.md); --exact keeps the f16 KV")
    if (dense and ax.default_on("kv8") and not a.fast and not a.exact and not a.stock and not getattr(a, "approx_kv", False)
            and not any(x in ("-ctk", "--cache-type-k", "-ctv", "--cache-type-v") for x in extra)):
        # LAUNCHER-982 (PC 2026-11-08: Qwen3.8-27B UD-Q3_K_XL at --agent --ctx 65536 ran 18.9 tok/s against 35 at 24k: the f16 KV cache did not fit beside the model, llama.cpp's --fit
        # moved the attention over the KV cache to the CPU): estimate model + KV + compute against the free GPU memory; f16 if it fits, else the 8-bit cache when that fits (or at least
        # spills less); --exact keeps f16 and the 'all on GPU' line says what it costs
        triple = dense_gpu_fit_pair(a, a.model, ctx, machine, with_q4=True)
        pair = triple[:2] if triple else None
        if pair and not pair[0]["fits"]:
            a.fast, a.dense_kv = "kv", True
            if not pair[1]["fits"]:
                # LAUNCHER-1063 (owner 2026-11-10): f16 -> q8_0/q8_0 -> K q8_0 / V q4_0 -> q4_0/q4_0 -> only then layers to RAM; each step only when the one before would spill; never a
                # shorter context; --exact and your own -ctk/-ctv are gated above. Laptop, 27B, top-1 against f16: q8/q8 96.5 %, K q8/V q4 94.6 %, q4/q4 92.9 % (KL 0.173); PC, 27B 64k with the
                # desktop on the card: decode 18.5 tok/s with q8_0 (spilling) against 27.8 with q4_0
                mixed, q4 = triple[2], triple[3]
                over8 = pair[1]["over_gib"]
                if mixed["fits"]:
                    a.dense_kv_types, step, got = KV_K8V4, "K q8_0 / V q4_0", mixed
                    quality = "approximate: same top token in 94.6 % of positions against f16 (measured on Qwen3.8-27B; q8_0/q8_0: 96.5 %)"
                else:
                    a.dense_kv_types, step, got = ("q4_0", "q4_0"), "q4_0 / q4_0", q4
                    quality = f"approximate: same top token in 92.9 % of positions against f16, KL 0.173 (measured on Qwen3.8-27B; K q8_0 / V q4_0 would still spill by {mixed['over_gib']:.1f} GiB)"
                saves = pair[1]["kv_gib"] - got["kv_gib"]
                say(f"moe-cache: context {ctx // 1024}k: KV {step}: q8_0 / q8_0 would spill by {over8:.1f} GiB ({gpu_fit_words(pair[1], 'q8_0')}); this step saves {saves:.1f} GiB of KV cache and " + (
                    f"does not spill ({gpu_fit_words(got, step)}). " if got["fits"] else
                    f"is still {got['over_gib']:.1f} GiB over ({gpu_fit_words(got, step)}), so part of the model runs from system RAM. ") +
                    quality + "; --exact keeps f16, your own -ctk/-ctv after -- win")
            elif pair[1]["fits"]:
                say(f"moe-cache: context {ctx // 1024}k: the f16 KV cache does not fit the GPU next to the model ({gpu_fit_words(pair[0], 'f16')}): q8_0 KV cache, which does "
                    "(half the GPU memory for the same context; --exact keeps f16 and lets llama.cpp move work to the CPU)")
    a.lp_kv = False
    # the long-prompt mode is the default (owner 2026-11-08, measured: prompts +77 %, answers +1.8 %, claim v038-longprompt-default) for a GPU plan of a big model at agent-sized
    # contexts; --long-prompt asks for it at any context. Not with --no-long-prompt, --exact, --stock, a model that runs on plain llama.cpp or a dense model, and below LONG_PROMPT_MIN_CTX
    # (the prompts of a short chat are shorter than one chunk, so q8_0 would cost exact answers for nothing)
    a.lp_explicit = bool(getattr(a, "long_prompt", False)) and not getattr(a, "no_long_prompt", False)
    if a.lp_explicit and (a.exact or a.stock):           # checked before anything else, so --ub / --fast / -ctk cannot hide the conflict
        raise SystemExit("--long-prompt does not go with --exact or --stock (it stores the KV cache as q8_0, which is not identical to stock)")
    a.long_prompt = a.lp_explicit or (ax.default_on("long-prompt") and not getattr(a, "no_long_prompt", False) and not a.exact and not a.stock and not stock and not dense and ctx >= LONG_PROMPT_MIN_CTX)
    # LAUNCHER-985: a token mode given by the user (--fast, --fast=2, --fast=fewer, --fast=kv, --approx) only adds its own settings (plugin environment, the q8_0 KV cache it implies) on top of the
    # normal plan: the long-prompt mode (-ub 4096) still applies. lp_with_fast marks that --fast was there first, so the KV cache is never undone and the mode is not switched off.
    a.lp_with_fast = bool(a.fast) and not kv128_default
    if getattr(a, "long_prompt", False) and (not a.fast or a.lp_with_fast) and a.ub == "auto" and not any(x in ("-ub", "--ubatch-size") for x in extra):          # the long-prompt mode stores the KV cache as q8_0 (the measured setting is --fast=kv); undone below when -ub 4096 does not fit
        if not any(x in ("-ctk", "--cache-type-k", "-ctv", "--cache-type-v") for x in extra) and (a.vram or machine.pick_gpu(listing if listing is not None else machine.list_devices(ls))):
            a.fast = a.fast or "kv"
            a.lp_kv = True
    if (getattr(a, "approx_kv", False) and not a.fast and not any(x in ("-ctk", "--cache-type-k", "-ctv", "--cache-type-v") for x in extra)
            and (a.vram or machine.pick_gpu(listing if listing is not None else machine.list_devices(ls)))):
        a.fast = "kv"                       # --approx safe: the 8-bit KV cache at any context (where the long-prompt mode did not already switch it on)
    kv_q8 = bool(a.fast) and not any(x in ("-ctk", "--cache-type-k", "-ctv", "--cache-type-v") for x in extra)
    if kv_q8:
        tk, tv = getattr(a, "dense_kv_types", None) or ("q8_0", "q8_0")
        extra += ["-ctk", tk, "-ctv", tv]
    return kv_q8


def long_prompt_stage(a, extra, ls, model, ctx, machine, gpu, gcache, kv_q8, mm_bytes, tuned, desk_kw, ub_asked):
    """The long-prompt mode (--long-prompt; the default since v0.3.8 for a GPU plan of a big model at 32k context and up, owner 2026-11-08): -ub 4096 with the q8_0 KV cache, measured on Flash-Next IQ3_XXS in a 65k conversation on the PC
    against -ub 2048 with the f16 KV cache (prompts +77.0 %, answers +1.8 %, claim v038-longprompt-default; --no-long-prompt gives the old setting). It applies only when the memory plan of this card leaves room for -ub 4096 (the tuner tried it first);
    when it does not, nothing changes: the q8_0 KV cache is undone (alone it lost prompt speed in the same test) and the normal plan is made again. Returns kv_q8, tuned, desk_kw, ub_asked."""
    if not getattr(a, "long_prompt", False):
        return kv_q8, tuned, desk_kw, ub_asked
    loud = getattr(a, "lp_explicit", False)                  # the default mode stays quiet when it does not apply; the flag says why
    user_ub = a.ub != "auto" or any(x in ("-ub", "--ubatch-size") for x in extra)
    if user_ub:
        loud and say("moe-cache: --long-prompt: your own --ub wins, so the long-prompt mode changes nothing")
        return kv_q8, tuned, desk_kw, ub_asked
    if not getattr(a, "lp_kv", False):
        loud and say("moe-cache: --long-prompt: no GPU found, or your own -ctk/-ctv is set: the long-prompt mode changes nothing")
        return kv_q8, tuned, desk_kw, ub_asked
    if tuned and tuned.get("ub") == 4096:
        say("moe-cache: long-prompt mode" + ("" if loud else " (the default; --no-long-prompt or --exact turns it off)") + ": a prompt chunk of 4096 tokens with the q8_0 KV cache (long prompts are read faster, "
            "answers come as fast; the q8_0 KV cache means answers are not byte-identical to stock: same top token in 94-95 % of positions, the meaning stayed the same in our checks; measured on Flash-Next, other models can differ)")
        return kv_q8, tuned, desk_kw, ub_asked
    if getattr(a, "approx_kv", False) or getattr(a, "lp_with_fast", False):         # --approx safe / --fast asked for the 8-bit KV cache itself: it stays, the micro-batch is the usual one
        a.lp_kv = False
        loud and say("moe-cache: --long-prompt: -ub 4096 does not fit the memory plan of this card; your --fast / --approx keeps its 8-bit KV cache with the usual micro-batch")
        return kv_q8, tuned, desk_kw, ub_asked
    # -ub 4096 does not fit this card's memory plan: undo the KV cache and plan again as without the flag
    for i in range(len(extra) - 3):
        if extra[i:i + 4] == ["-ctk", "q8_0", "-ctv", "q8_0"]:
            del extra[i:i + 4]
            break
    a.fast, a.lp_kv = None, False
    loud and say("moe-cache: --long-prompt: -ub 4096 does not fit the memory plan of this card (or the plan is not a GPU plan), so nothing changes: the normal micro-batch and the f16 KV cache stay")
    a.lp_retune = True                                       # the "other programs hold" line was said by the first plan
    tuned, desk_kw, ub_asked = tune_launch(a, extra, ls, model, ctx, machine, gpu, gcache, False, mm_bytes)
    return False, tuned, desk_kw, ub_asked


def fast_env_stage(a, env, stock, model, machine):
    """ENGINE-029, stage "approximate modes": the plugin settings and the one plain line for --fast, --fast=2, --fast=fewer and --fast=kv; without --fast, the hint where it would pay (the model is much bigger than the memory it may use). env grows in place."""
    if a.fast == "kv" and (getattr(a, "lp_kv", False) or getattr(a, "dense_kv", False)):
        pass                                                    # --long-prompt says what it did after the memory plan (long_prompt_stage); the dense KV choice said it in kv_stage
    elif a.fast == "kv":
        say("moe-cache: --fast=kv: q8_0 KV cache only (every expert is computed; KL about 0.005-0.012 against stock, output no longer identical)")
    elif a.fast:
        env.append(("MOE_CACHE_ROUTE", "skip"))
        if a.fast == "2":
            env.append(("MOE_CACHE_ROUTE_DROP", "4"))
            say("moe-cache: --fast=2: up to 4 low-ranked experts per token that are not in RAM are skipped (for memory much smaller than the model: "
                "Flash-Next at 12 GB +43 %; about 1 top token in 28 differs from stock, KL 0.022; RESULTS 53)")
        elif a.fast == "fewer":
            env.append(("MOE_CACHE_FEWER", "2"))
            say("moe-cache: --fast=fewer: --fast, and on the CPU-held layers the 2 lowest-ranked experts of every token are never computed (top-10 -> 8 on Flash-Next); "
                "output no longer identical to stock, quality limits in docs/QUALITY_GATE.md (level fewer)")
        else:
            say("moe-cache: --fast: low-ranked experts that are not in RAM are skipped (about +10 %, output no longer identical to stock)")
    elif not stock:
        # suggest --fast only where it pays: the model is much bigger than the memory it may use
        try:
            size = sum(os.path.getsize(p) for p in model_parts(model))
            lim = float(a.ram) * 1e9 if a.ram else machine.mem_total()
            if lim and size > 2 * lim:
                say(f"moe-cache: the model is {size / lim:.1f}x the memory it may use: --fast skips rarely needed experts for about +10 % (changes some tokens; off by default)")
        except Exception:
            pass


def dense_runs_on_cpu(a, model, machine):
    """True when a dense model has layers or FFN weights on the CPU next to a GPU: a GPU is there and the model file is bigger than the free GPU memory (--vram wins) minus the
    memory --fit leaves free. model None / unreadable size: unknown, treated as "runs partly on the CPU" (the rule's behaviour before 2026-11-08)."""
    free = float(a.vram) * 2**30 if getattr(a, "vram", None) else machine.free_vram_bytes()
    if not free:
        return False
    try:
        size = sum(os.path.getsize(p) for p in model_parts(model))
    except (OSError, TypeError):
        return True
    return size + DENSE_FIT_TARGET_MIB * 2**20 > free


def cpu_pin_stage(a, env, extra, stock, win, machine, dense=False, model=None):
    """ENGINE-029, stage "CPU cores": efficiency cores for the expert executor and pinning llama-server to the performance cores on hybrid CPUs (Windows core groups, Linux sysfs). env grows in place. Returns the pinning arguments for llama-server."""
    if not stock and not a.no_ecores and not os.environ.get("MOE_CACHE_EXEC_ECORES") and machine.has_cpu_atom():
        ne = min(machine.cpu_atom_count(), 4)
        if ne > 0:
            env.append(("MOE_CACHE_EXEC_ECORES", str(ne)))
            say(f"moe-cache: expert executor will also use {ne} efficiency cores (--no-ecores to disable)")
    pinargs = []
    hm = hybrid_masks(machine.win_cores()) if win else None
    if machine.win_core_groups() > 1 and not a.no_pin:
        say("moe-cache: " + WIN_GROUPS_NOTE)
    if hm and not any(x in ("-C", "--cpu-mask", "-Cr", "--cpu-range") for x in extra):
        pmask, ecpus = hm
        if not a.no_pin:
            pinargs = ["-C", hex(pmask), "--cpu-strict", "1"]
            env.append(("MOE_CACHE_CPU_MASK", hex(pmask)))
            say(f"moe-cache: hybrid CPU: pinning llama-server threads to the performance cores (-C {hex(pmask)}); --no-pin to disable")
        ne = min(len(ecpus), 4)
        if ne and not stock and not a.no_ecores and not os.environ.get("MOE_CACHE_EXEC_ECORES"):
            env += [("MOE_CACHE_EXEC_ECORES", str(ne)), ("MOE_CACHE_ECORE_MASK", hex(sum(1 << c for c in ecpus)))]
            say(f"moe-cache: expert executor will also use {ne} efficiency cores (--no-ecores to disable)")
    if not a.no_pin and machine.has_cpu_core() and not any(x in ("-C", "--cpu-mask", "-Cr", "--cpu-range") for x in extra):
        mask = machine.perf_core_mask()
        # Dense thread rule (laptop, Qwen3-14B with the FFN in RAM: +30-70 % decode): 8 decode threads on 4 P-cores with two cores left for the GPU driver. It only pays when part of the model
        # really runs on the CPU, i.e. the file does not fit the free GPU memory (then the dense route keeps FFN weights in RAM or --fit splits layers). A dense model that fits the GPU
        # runs entirely there: the decode thread is only a driver loop, so it keeps one thread per performance core. No GPU: the two cores are not needed for a driver either.
        # This branch is LINUX ONLY (/sys/devices/cpu_core, Intel hybrid CPUs): on Windows the hybrid branch above pins the performance cores / core groups and the dense decode
        # threads are llama.cpp's default; the +30-70 % was measured on Linux only and has no Windows measurement yet.
        dm = machine.dense_core_mask() if (dense and not a.cpu_only and dense_runs_on_cpu(a, model, machine)) else None
        if dm and not any(x in ("-t", "--threads") or x.startswith("--threads=") for x in extra):
            pinargs = ["-C", dm[0], "--cpu-strict", "1", "-t", str(dm[1])]
            if mask and not any(x in ("-tb", "--threads-batch", "-Cb", "--cpu-mask-batch") for x in extra):
                pinargs += ["-Cb", mask, "--cpu-strict-batch", "1", "-tb", str(bin(int(mask, 16)).count("1"))]       # prompt batches: one thread per performance core (8k chat, 14B: prompt 107 -> 143 tok/s, decode 9.6 -> 10.1; 12 batch threads no better)
            say(f"moe-cache: hybrid CPU, dense model: {dm[1]} decode threads on both hardware threads of all performance cores but two (-C {dm[0]}), the GPU driver keeps the other two; "
                "+30-70 % decode against one thread per core when part of the model runs on the CPU (laptop, Qwen3-14B); prompt batches keep one thread per performance core; --no-pin to disable")
        elif mask:
            pinargs = ["-C", mask, "--cpu-strict", "1"]
            say(f"moe-cache: hybrid CPU: pinning llama-server threads to the performance cores (-C {mask}); --no-pin to disable")
    return pinargs


def layers_stage(a, extra, model, ctx, ncmoe, ncmoe_user, gcache, kv_q8, tuned, desk_kw, machine):
    """ENGINE-029, stage "micro-batch and static layers": -ub from the tuner (or the user), and without a GPU cache the number of static expert layers from the same decision (the planner alone assumes a small compute buffer, RESULTS 61); the hint for --fast=kv; the MTP warning for an --n-cpu-moe that puts more layers on the GPU than planned. Returns (ub as text, n_cpu_moe as text, whether the tuner had no GPU memory figure)."""
    ub = a.ub
    if ub == "auto":
        ub = "" if any(x in ("-ub", "--ubatch-size") for x in extra) else (str(tuned["ub"]) if tuned else "")
    if ub and ub != "auto":
        if not ub.isdigit():
            say("moe-cache: --ub must be auto or a number")
        else:
            extra += ["-ub", ub, "-b", str(max(int(ub), 2048))]
            say(f"moe-cache: micro-batch -ub {ub} (prompt processing; --ub N to change)")
    # without a GPU cache, take the static expert layers from the same decision as the micro-batch (the planner alone assumes a small compute
    # buffer and fills the card, so the long-prompt GPU block found no room next to -ub 2048: Flash-Next prefill 18 instead of 50+ tok/s)
    no_vram = bool(tuned) and any("no GPU memory information" in w for w in tuned.get("why", []))   # e.g. an integrated GPU: keep the planner's answer
    if gcache == "off" and ncmoe_user is None and ncmoe != "0" and tuned and not no_vram:
        ncmoe = str(tuned["n_cpu_moe"])
        if not kv_q8 and not a.fast:
            try:
                from gguf_plan import fast_kv_hint
                h = fast_kv_hint(model, ctx, free=float(a.vram) * 2**30 if a.vram else machine.free_vram_bytes(), ub=int(ub) if (ub or "").isdigit() else None,
                                 **({"reserve_gb": a.gpu_margin * 2**30 / 1e9} if a.gpu_margin is not None else ({"reserve_gb": desk_kw} if desk_kw else {})))
                if h:
                    say("moe-cache: " + h)
            except Exception:
                pass
    # MTP's draft context needs GPU memory next to the layers; with more expert layers on the GPU than planned, llama.cpp over-commits the card
    # (it logs "failed to fit params to free device memory") and on Windows the spill then starves host memory: GGML_ASSERT(ctx->mem_buffer != NULL)
    # while creating the draft context (GLM on the PC, 41 and 43 CPU layers with MTP; the planned 45 loads, RESULTS 62). Same warning on Linux.
    if (ncmoe_user or "").isdigit() and tuned and not no_vram and gcache == "off" and "draft-mtp" in extra and int(ncmoe_user) < int(tuned["n_cpu_moe"]) < 999:
        say(f"moe-cache: WARNING: --n-cpu-moe {ncmoe_user} puts more expert layers on the GPU than planned ({tuned['n_cpu_moe']}) while MTP is on: "
            "MTP's draft context needs GPU memory too, and llama-server may abort at load (seen on GLM: GGML_ASSERT(ctx->mem_buffer != NULL)). "
            f"Use --n-cpu-moe {tuned['n_cpu_moe']} or more, or --no-mtp", err=True)
    return ub, ncmoe, no_vram


def gpu_cache_stage(a, extra, env, ls, model, listing, gcache, ncmoe, tuned, stock, profile, machine):
    """ENGINE-029, stage "GPU expert cache and hybrid bank": the size of llama.cpp's GPU expert cache (--gpu-cache auto/MiB, capped by one Vulkan device buffer), or the hybrid bank (--hybrid); both put every expert on the CPU side (n_cpu_moe 999). env and extra grow in place. Returns n_cpu_moe as text."""
    if gcache != "off" and ncmoe != "0":
        gcmib = gcache
        if gcache == "auto":
            gcmib = str(tuned["gpu_cache_mib"]) if tuned and tuned["gpu_cache_mib"] else ""
            for w in (tuned or {}).get("why", []):
                say("moe-cache: tuner: " + w)
        if not gcmib.isdigit():
            say("moe-cache: --gpu-cache: could not work out a size (planner unavailable?), using the planner's static GPU layers"); gcmib = ""
        else:
            cap, lim = machine.vk_cache_cap(ls, model, listing)
            if cap is not None and int(gcmib) > cap:
                say(f"moe-cache: --gpu-cache: {gcmib} MiB capped to {cap} MiB: on Vulkan each of llama.cpp's cache banks must fit in one device buffer "
                    f"({lim / 2**20:.0f} MiB on this GPU), a bigger cache aborts the load")
                gcmib = str(cap)
        if gcmib and int(gcmib) < 512:
            say(f"moe-cache: --gpu-cache: only {gcmib} MiB of VRAM would be left, using the planner's static GPU layers"); gcmib = ""
        if gcmib:
            ncmoe = "999"; extra += ["--moe-cache-mib", gcmib]; env.append(("MOE_CACHE_GPU_CACHE", "1"))
            say(f"moe-cache: GPU expert cache of {gcmib} MiB; " + ("llama.cpp keeps the hottest experts on the GPU" if stock else "all experts are managed by the plugin (RAM) and the hottest ones are computed on the GPU"))
    if a.hybrid and ncmoe != "0":
        ncmoe = "999"; env.append(("MOE_CACHE_HYBRID", os.environ.get("MOE_CACHE_HYBRID", "auto")))
        say("moe-cache: --hybrid (experimental): the most used experts are computed on the GPU from a bank, the rest on the CPU in parallel (output no longer identical to stock)")
        if not os.path.isfile(profile):
            say(f"moe-cache: --hybrid: no usage profile yet ({profile}): the bank is chosen from it, so this first session runs without one")
    return ncmoe


def host_reserve_mib(cram, agent):
    """The host memory (MiB) the plugin keeps free for llama-server's prompt cache: the cache size (0 for none / unlimited) plus 1 GiB for the agent's slots."""
    return max(int(cram), 0) + (1024 if agent else 0)


def prompt_cache_stage(a, extra, env, ls, machine, win, dry):
    """ENGINE-029, stage "prompt cache": llama-server's --cache-ram (8 GiB by default, which competes with the expert cache): 2 GiB, 0 below 16 GB of memory, MOE_CACHE_PROMPT_CACHE_MIB, cut to what Windows' free commit can carry; and the host reserve told to the plugin. Yours after -- always wins. extra and env grow in place. Returns the cache size as text or None."""
    cram = arg_after(extra, "--cache-ram", "-cram")
    cram_auto = False             # True when the launcher chose it (yours after -- always wins)
    if cram is None:
        try:
            if "--cache-ram" in ls_help(ls, machine.run):
                # below 16 GB the reserve is a large share of the expert cache (Qwen3.6-35B at 12 GB: 16.9 tok/s with 2048 MiB, 19.2 with 512, 22.6 with 0;
                # RESULTS 44). The prompt cache only helps when one server switches between conversations; a single chat reuses its slot either way
                lim = machine.mem_for_server(a.ram)
                cram = "0" if lim and lim < 16e9 else "2048"
                # idea L05 (MOE_CACHE_PROMPT_CACHE_MIB=N, unset = the rule above): set the prompt cache from evidence instead of the fixed split (0 = off); the number
                # to use is the one where the saved reprocessing stops paying for the expert-cache RAM it takes (tools/agent_bench.py at two settings)
                pcm = os.environ.get("MOE_CACHE_PROMPT_CACHE_MIB", "")
                if re.fullmatch(r"[0-9]+", pcm):
                    cram = pcm
                extra += ["--cache-ram", cram]
                cram_auto = True
        except Exception:
            pass
    if win:     # XA-3: llama-server's own buffers and Vulkan's GPU memory need Windows commit (RAM + page file), not only free RAM
        if not dry:
            warn_heavy_background()
        ctot, cfree = machine.win_commit()
        if ctot and cfree is not None:
            say(f"moe-cache: Windows memory: RAM {(machine.win_memory()[1] or 0) / 1e9:.1f} GB free; commit {cfree / 1e9:.1f} GB free of {ctot / 1e9:.1f} GB (RAM + page file)")
            if cram_auto and re.fullmatch(r"[0-9]+", str(cram)) and not commit_check_on(a, machine):
                fit, short = commit_plan(cfree, a.vram, int(cram))
                if fit < int(cram):
                    extra[len(extra) - 1 - extra[::-1].index("--cache-ram") + 1] = str(fit)
                    say(f"moe-cache: prompt cache {cram} -> {fit} MiB so llama-server stays inside Windows' free commit (other programs hold the rest)", err=True)
                    cram = str(fit)
                if short:
                    say(f"moe-cache: WARNING: Windows commit is nearly full ({cfree / 1e9:.1f} GB free of {ctot / 1e9:.1f} GB): llama-server may fail to allocate "
                        "memory and stop. Close big programs, or set the page file to system-managed (or larger) in Windows' Advanced system settings", err=True)
    say(prompt_cache_note(cram, extra, a.agent))
    if cram is not None and re.fullmatch(r"[0-9-]+", cram):
        env.append(("MOE_CACHE_HOST_RESERVE_MIB", str(host_reserve_mib(cram, a.agent))))
    elif a.agent:
        env.append(("MOE_CACHE_HOST_RESERVE_MIB", "1024"))     # told to the plugin directly instead of it parsing our command line
    return cram


def stock_env_stage(a, env, stock, ncmoe, size, avail, method):
    """ENGINE-029, stage "stock llama-server": a model that fits (or --stock) runs without the plugin: one line says so, every plugin setting leaves the environment, and a short-batch GPU offload threshold (T5, RESULTS 68) is added. Returns (environment, method)."""
    if stock:
        if a.stock:
            say("moe-cache: --stock: running stock llama-server without the plugin, as asked (to compare; a model bigger than RAM is read through the file cache)")
        else:
            say(f"moe-cache: the model ({size / 1e9:.1f} GB) fits in the memory it may use ({avail / 1e9:.1f} GB): running stock llama-server without the plugin, "
                "which is faster when everything fits (RESULTS 44). --always-plugin to use the plugin anyway.")
        env = [(k, v) for k, v in env if not (k.startswith("MOE_CACHE") or k in ("LD_PRELOAD", "GGML_BACKEND_PATH"))]
        method = "none (stock)"
        if ncmoe != "0" and size and avail and size + STOCK_HEADROOM <= avail and not os.environ.get("GGML_OP_OFFLOAD_MIN_BATCH"):
            # llama.cpp copies every CPU-held expert to the GPU for a batch of 32+ tokens (~1 s for Nemotron Bolt on the PC); the server cuts a prompt
            # 4 tokens before its end for a checkpoint, so every prompt of 36+ tokens paid it. Short batches are faster on the CPU (T5, RESULTS 68)
            env.append(("GGML_OP_OFFLOAD_MIN_BATCH", str(STOCK_OFFLOAD_MIN_BATCH)))
    return env, method


def leave_room_stage(a, env, extra, pinargs, machine):
    """ENGINE-029, stage "--leave-room": lower priority for llama-server and, unless threads are pinned or given, two physical cores left free. Returns the arguments for llama-server (a new list when -t was added)."""
    if a.leave_room:
        env.append(("MOE_CACHE_LEAVE_ROOM", "1"))      # run_child_once starts llama-server at a lower priority
        if not pinargs and not any(x in ("-t", "--threads") or x.startswith("--threads=") for x in extra):
            extra = extra + ["-t", str(max(1, machine.physical_cores() - LEAVE_ROOM_FREE_CORES))]    # hybrid CPUs keep their performance-core pinning
            say(f"moe-cache: --leave-room: {LEAVE_ROOM_FREE_CORES} CPU cores left free (-t {extra[-1]})")
    return extra


def multi_gpu_stage(a, env, extra, mg, model, ctx, kv_q8, ub, gcache, ncmoe, ncmoe_user, tuned, stock, machine, win):
    """ENGINE-029, stage "--multi-gpu" (PROTOTYPE): the split across the cards and the second card for the long-prompt block. Returns (n_cpu_moe, arguments for llama-server)."""
    if mg:
        extra_before = extra
        ncmoe, extra = multi_gpu_apply(mg, a, extra, model, ctx, kv_q8, ub, gcache, ncmoe, ncmoe_user, tuned, machine, win)
        if extra is not extra_before and not stock and not os.environ.get("MOE_CACHE_GPU_DEV2"):
            # N3/MG5 (PROTOTYPE): the plugin's long-prompt block may also use the second card of the --device list when it does not fit on the first
            env.append(("MOE_CACHE_GPU_DEV2", "auto"))
            say("moe-cache: --multi-gpu: the plugin's long-prompt GPU block may use the second card too (MOE_CACHE_GPU_DEV2=auto, PROTOTYPE)")
    return ncmoe, extra


def dense_ffn_override(a, model, ctx, kv_q8, vram_user, machine, extra, mm_bytes=0):
    """Owner lever (a), LAUNCHER-732: a pure-attention dense model bigger than the GPU keeps attention, KV and most weights on the GPU and sends the FFN weights of the last
    layers to the RAM (-ot), instead of llama.cpp's layer split, whose CPU layers also compute attention over their KV (14B at 32k: 0.9 tok/s split vs 3.4-4.0 this way).
    Returns the -ot text or None (then --fit places the layers). --no-ffn-ram turns it off; your own -ot / -ngl after -- stand."""
    if getattr(a, "no_ffn_ram", False) or any(x in ("-ot", "--override-tensor") or x.startswith("--override-tensor=") for x in extra):
        return None
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import dense_ffn_split
        free = float(a.vram) * 2**30 if a.vram else machine.free_vram_bytes()
        if free and mm_bytes:
            free = max(free - (mm_bytes + MMPROJ_VRAM_HEADROOM), 0.0)           # a vision projector's weights and compute buffer sit in VRAM too (same as the tuner)
        kv_q8 = kv_plan_type(extra, kv_q8)       # your own cache type after -- counts; q4_0 is sized as q4_0 (LAUNCHER-1063)
        margin = a.gpu_margin if a.gpu_margin is not None else 0.25
        if getattr(a, "cpu_gpu_split", None):
            margin += CPU_GPU_SPLIT_VRAM_GIB            # the split's staging tensors and its own CUDA context (measured +56 MiB on the laptop)
        r = dense_ffn_split(model, ctx, free, kv_q8=kv_q8, margin_gib=margin)
    except Exception:
        return None
    if not r:
        return None
    if r["n_off"] is None:
        say(f"moe-cache: dense model: the KV cache for {ctx} tokens ({r['kv_gib']:.1f} GiB) leaves too little GPU memory even with every FFN weight in RAM; llama.cpp's --fit places the layers"
            + ("" if kv_q8 else " (a q8_0 KV cache halves it, approximate: moe-cache-server ... -- -ctk q8_0 -ctv q8_0)"))
        return None
    if not r["n_off"]:
        return None
    layers = "|".join(str(i) for i in range(r["first_layer"], r["n_layers"]))
    say(f"moe-cache: dense model: attention, the KV cache and the other weights stay on the GPU; the FFN weights of layers {r['first_layer']}-{r['n_layers'] - 1} ({r['n_off']} of {r['n_layers']}, "
        f"{r['n_off'] * r['ffn_gib']:.1f} GiB) stay in RAM (-ot; the model plus a {ctx}-token KV cache needs {r['need_gib']:.1f} GiB, the GPU has {r['free_gib']:.1f}). "
        "A plain layer split would run the attention of the RAM layers on the CPU: far slower in long chats. --no-ffn-ram turns this off")
    return f"blk\\.({layers})\\.ffn_(up|down|gate|gate_up)\\.weight=CPU"


def dense_oom_replan(cmd, model, ctx, kv_q8, free_bytes, margin_gib, user_ot=False):
    """LAUNCHER-850: llama-server of a dense model ran out of GPU memory at load (a game or browser took memory after the plan was made). -> (new command, plain text) or
    (None, reason). One step smaller, from the free memory measured again: the FFN weights of at least one more layer go to RAM (-ot; never fewer than before + 1; without a
    free-memory figure an eighth of the layers more), or, when every layer's FFN is already there or the model is a hybrid recurrent one the FFN plan does not cover, half the
    micro-batch (-ub, its compute buffer is what scales), or a larger llama.cpp --fit margin (-fitt, +512 MiB) when --fit places the layers. Your own -ot / -ngl stand (user_ot)."""
    if user_ot:
        return None, "your own -ot stands"
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import dense_ffn_split
        r = dense_ffn_split(model, ctx, free_bytes, kv_q8=kv_q8, margin_gib=margin_gib) if free_bytes else None
    except Exception:
        r = None
    new = list(cmd)
    old_n = 0
    if "-ot" in new:
        m = re.search(r"blk\\\.\(([0-9|]+)\)\\\.ffn_", new[new.index("-ot") + 1])
        old_n = len(m.group(1).split("|")) if m else 0
    if r is None and free_bytes is None:
        try:
            from gguf_plan import dense_ffn_split as _s
            r = _s(model, ctx, 1e15, kv_q8=kv_q8, margin_gib=margin_gib)          # only for the layer count; the free memory is unknown
        except Exception:
            r = None
        if r:
            r = dict(r, n_off=0)
    if r and r["n_layers"]:
        n_layers = r["n_layers"]
        if r.get("n_off") is None:                  # even every FFN in RAM leaves too little: all of them, and a smaller micro-batch below
            want = n_layers
        else:
            want = old_n + max(1, -(-n_layers // 8)) if not free_bytes else max(r["n_off"], old_n + 1)
        if want <= n_layers:
            layers = "|".join(str(i) for i in range(n_layers - want, n_layers))
            ot = f"blk\\.({layers})\\.ffn_(up|down|gate|gate_up)\\.weight=CPU"
            if "-ot" in new:
                new[new.index("-ot") + 1] = ot
            else:
                new += ["-ot", ot]
            if r.get("n_off") is None and "-ub" in new and int(new[new.index("-ub") + 1]) >= 256:
                ub = int(new[new.index("-ub") + 1]) // 2
                new[new.index("-ub") + 1] = str(ub)
                if "-b" in new:
                    new[new.index("-b") + 1] = str(max(ub, 2048))
                return new, f"the FFN weights of all {n_layers} layers in RAM (was {old_n}) and -ub {ub}"
            return new, f"the FFN weights of {want} of {n_layers} layers in RAM (was {old_n})"
    if "-ub" in new and int(new[new.index("-ub") + 1]) >= 256:
        ub = int(new[new.index("-ub") + 1]) // 2
        new[new.index("-ub") + 1] = str(ub)
        if "-b" in new:
            new[new.index("-b") + 1] = str(max(ub, 2048))
        return new, f"-ub {ub} (was {ub * 2})"
    if "-fitt" in new:
        mib = int(new[new.index("-fitt") + 1]) + 512
        new[new.index("-fitt") + 1] = str(mib)
        return new, f"a --fit margin of {mib} MiB"
    return None, "nothing left to shrink"


CPU_GPU_SPLIT_VRAM_GIB = 0.15


def cpu_gpu_split_stage(a, env, cmd, ls, dense=True):
    """DECODE-917 (patch 0091), opt-in: for the weights of a dense model that stay in RAM (the FFN plan's -ot, or the layers --fit leaves there) the GPU also computes a
    share of their rows, fed by DMA from page-locked RAM (CUDA) or by reading the host rows in place (Vulkan), while the CPU computes the rest. Needs the kit's patched llama.cpp with patch 0091 and CUDA; the server then loads the model without mmap and without repacking (a file-backed
    mapping cannot be page-locked, a repacked tensor is not what the GPU kernels read). Returns the command; adds GGML_CPU_GPU_SPLIT to env."""
    want = getattr(a, "cpu_gpu_split", None)
    if not want:
        return cmd
    if want != "auto":
        try:
            ok = 0.05 <= float(want) <= 0.6
        except ValueError:
            ok = False
        if not ok:
            raise SystemExit("--cpu-gpu-split takes auto or a number from 0.05 to 0.6 (the GPU's share of the rows)")
    cpu_only = arg_after(cmd, "--device", "-dev") == "none" or arg_after(cmd, "-ngl", "--n-gpu-layers") == "0"
    if not dense or cpu_only:
        say("moe-cache: --cpu-gpu-split: only a dense model on a GPU uses it (the FFN-in-RAM plan, or the layers --fit leaves in RAM); not applied here")
        return cmd
    line = llama_patch_line(ls) or ""
    m = re.search(r"patches ([0-9,]+)", line)
    if not m or "0091" not in m.group(1).split(","):
        say("moe-cache: --cpu-gpu-split: this llama-server does not carry patch 0091 (the kit's patched llama.cpp does); not applied")
        return cmd
    cmd = list(cmd)
    if "--load-mode" in cmd:
        cmd[cmd.index("--load-mode") + 1] = "none"
    else:
        cmd += ["--load-mode", "none"]
    if "--no-repack" not in cmd:
        cmd.append("--no-repack")
    env.append(("GGML_CPU_GPU_SPLIT", want))
    say(f"moe-cache: --cpu-gpu-split {want}: the GPU computes part of the rows of the FFN weights that stay in RAM (patch 0091); "
        "the model loads without mmap and without repacking, and about 0.15 GiB of GPU memory is kept for it")
    return cmd


def command_stage(a, ls, model, ctx, port, ncmoe, extra, pinargs, vram_user, dense, method, profile, machine, kv_q8=False, mm_bytes=0):
    """ENGINE-029, stage "command": CPU-only mode (--device none, -ngl 0), the llama-server command line with the placement and the pinning, --auto-fit for a dense model, and the line that says what the cache will manage. Returns (the command, the arguments for llama-server)."""
    cpu_only, cpu_why = cpu_only_decision(a.cpu_only, extra, machine.list_devices(ls) if not a.cpu_only else "", bool(vram_user))
    if cpu_only:
        extra = ["--device", "none"] + extra          # -ngl 0 alone still lets the KV cache land on a loaded GPU backend
        say(f"moe-cache: CPU-only ({cpu_why}): --device none and -ngl 0, nothing is put on a GPU")
    cmd = [ls, "-m", model, "-ngl", "0" if cpu_only else "99", "--n-cpu-moe", ncmoe, "-c", str(ctx), "-fa", "on", "--load-mode", "mmap", "--no-warmup", "--port", str(port)] + pinargs + extra
    user_ngl = any(x in ("-ngl", "--n-gpu-layers", "--gpu-layers") or x.startswith(("--n-gpu-layers=", "--gpu-layers=")) for x in extra)
    if dense and not cpu_only and not user_ngl and (a.auto_fit or not a.no_auto_fit):
        # LAUNCHER-732 (laptop, RTX 4060 8 GB, Qwen3-14B Q4_K_M 8.4 GiB): -ngl 99 aborts at load ("n_gpu_layers already set by user to 99"); llama.cpp's own --fit
        # keeps 1 GiB free by default (6.3 tok/s), --fit-target 256 MiB places more layers (8.2 tok/s, same text). With -ngl set, --fit is off (fit.cpp).
        del cmd[cmd.index("--n-cpu-moe"):cmd.index("--n-cpu-moe") + 2]
        ot = dense_ffn_override(a, model, ctx, kv_q8, vram_user, machine, extra, mm_bytes)
        if ot:
            cmd += ["-ot", ot]
            return cmd, extra
        del cmd[cmd.index("-ngl"):cmd.index("-ngl") + 2]
        margin = ""
        try:
            if not any(x in ("-fitt", "--fit-target", "--fit") or x.startswith(("--fit-target=",)) for x in extra) and "--fit-target" in ls_help(ls, machine.run):
                mib = int(a.gpu_margin * 1024) if a.gpu_margin is not None else DENSE_FIT_TARGET_MIB
                cmd += ["-fitt", str(mib)]; margin = f" and keeps {mib} MiB of GPU memory free (--fit-target; --gpu-margin GiB changes it)"
        except Exception:
            pass
        say("moe-cache: dense model, no experts: no -ngl 99 (it makes a file bigger than the GPU abort at load); llama.cpp's --fit places the layers by free GPU memory" + margin
            + ("" if a.auto_fit else ". --no-auto-fit brings -ngl 99 back"))
    # LAUNCHER-983: tell the log scan whether work on the CPU is part of this plan (experts on the CPU by design, FFN weights in RAM, CPU-only, a dense model the estimate says is over the GPU)
    expect = bool(cpu_only or (not dense and ncmoe not in (None, "0")) or (dense and "-ot" in cmd))
    if dense and not expect:
        pair = dense_gpu_fit_pair(a, model, ctx, machine, with_q4=True)
        kv_name = arg_after(extra, "-ctk", "--cache-type-k") or "f16"
        r = pair[min(kv_fit_index(kv_name, arg_after(extra, "-ctv", "--cache-type-v")), len(pair) - 1)] if pair else None
        expect = bool(r and not r["fits"])
    EXPECT_CPU_WORK[0] = expect
    if dense and not cpu_only and not expect and "--load-mode" in cmd:
        why = load_mode_fit_none(a, cmd, extra, machine, ctx, dense_gpu_fit_pair(a, model, ctx, machine, with_q4=True), arg_after(extra, "-ctk", "--cache-type-k") or "f16",
                                 arg_after(extra, "-ctv", "--cache-type-v"))
        if why:
            cmd[cmd.index("--load-mode") + 1] = "none"
            say("moe-cache: " + why)
    if method == "none (stock)":
        pass
    elif ncmoe == "0":
        say(f"moe-cache: all experts fit in your GPU memory; the cache has nothing to manage for this model/context (profile {profile}; plugin loaded by {method})")
    else:
        say(f"moe-cache: experts of {ncmoe} layers stay on the CPU and are managed by the cache; profile {profile}; plugin loaded by {method}")
    return cmd, extra


def template_stage(a, extra, model, dry=False):
    """AGENTS-992, stage "chat template": a model whose own template raises "System message must be at the beginning." (Qwen3.8: 27B, Flash-Next) answers HTTP 500 to
    Claude Code, which sends system messages mid-conversation. Gives llama-server a copy fixed by one line (a later system message becomes a user turn; made from the
    model's own template, planner/chat_template_fix.py) with --chat-template-file, plus --jinja when the template file needs it. Your own --chat-template[-file] / --no-jinja
    after -- win. Returns the flags to add."""
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        import chat_template_fix as ctf
        if ctf.user_chose_template(extra):
            return []
        path, _ = ctf.fixed_template_file(model, CACHE_DIR, write=not dry)
        if not path and ctf.needs_fix(ctf.model_template(model)):
            say("moe-cache: WARNING: this model's chat template refuses system messages in the middle of a conversation (Claude Code sends them: HTTP 500), and the automatic "
                "fix does not recognise its shape; pass your own template: -- --chat-template-file FIXED.jinja", err=True)
    except Exception:
        return []
    if not path:
        return []
    say("moe-cache: this model's chat template refuses system messages in the middle of a conversation (some clients, Claude Code among them, send them: such a request would fail with HTTP 500 "
        f"\"System message must be at the beginning\"); using a copy fixed by one line, a later system message becomes a user turn: {path} "
        "(--chat-template-file FILE after -- uses your own)")
    return ["--chat-template-file", path] + ([] if "--jinja" in extra else ["--jinja"])


# ---- LAUNCHER-994: measured auto-tune (logic and cache file: bin/moe_cache_autotune.py) ------------------------------------------------------------------------
def autotune_machine_sig(machine, ls):
    """What makes a tuned pick valid on this computer and no other: CPU, cores, RAM (rounded to GiB), the GPUs and their total memory (free memory is left out:
    it changes every minute), the OS and the llama-server build (size + mtime)."""
    try:
        devs = re.sub(r"\(\s*(\d+) MiB,\s*\d+ MiB free\s*\)", lambda m: f"({int(m.group(1)) // 1024} GiB)", machine.list_devices(ls) or "")
    except Exception:
        devs = ""
    try:
        st = os.stat(ls)
        build = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        build = ""
    try:
        mem = round(machine.mem_total() / 2**30)
    except Exception:
        mem = 0
    return {"cpu": cpu_name(), "cores": machine.physical_cores(), "ram_gib": mem, "gpus": sorted(set(l.strip() for l in devs.splitlines() if "MiB" in l or "GiB" in l)),
            "os": "windows" if machine.windows else sys.platform, "llama_server": build}


def autotune_model_id(model):
    try:
        st = os.stat(model)
        return f"{os.path.basename(model)}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return os.path.basename(model)


AUTOTUNE_PROMPT = ("The lighthouse keeper had kept a logbook for thirty-one years, and in all that time he had never once written down a lie. "
                   "So when the ship appeared on the horizon on the first calm morning of October, with no flag, no lights and no one on deck, "
                   "he sat at his small wooden table for a long while before he picked up the pen. ")


def autotune_probe(cmd, env, left_s, decode_n):
    """(see _autotune_probe) plus LAUNCHER-1042: a probe the memory guard stopped says so in "guard_stop", which aborts the whole tune."""
    holder = {}
    r = _autotune_probe(cmd, env, left_s, decode_n, holder)
    g = holder.get("guard")
    if not r.get("ok") and g is not None and getattr(g, "stopped", None):
        r["guard_stop"] = str(g.stopped)
    return r


def _autotune_probe(cmd, env, left_s, decode_n, holder):
    """The real runner for moe_cache_autotune.tune: start llama-server with `cmd` on a free local port, wait until it is healthy (its load time), send one request that
    reads a ~1.5k-token prompt (prompt tok/s) and one that writes decode_n tokens (decode tok/s), stop the server. Never raises; a failure is {"ok": False, "error": ...}."""
    import moe_cache_autotune as at
    t0 = time.monotonic()
    left = lambda: left_s - (time.monotonic() - t0)
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]
    cmd = at.set_flag(at.set_flag(list(cmd), "--port", port), "--host", "127.0.0.1")
    key = dict(env).get("LLAMA_API_KEY")
    errf = tempfile.TemporaryFile()       # stderr is kept (never shown) to recognise llama.cpp's known load abort (kv_abort below); closed in the finally
    kw = {"env": env, "stdout": subprocess.DEVNULL, "stderr": errf}
    if sys.platform.startswith("linux"):
        kw["preexec_fn"] = _die_with_parent_linux
    try:
        p = subprocess.Popen(cmd, **kw)
    except OSError as e:
        errf.close()
        return {"ok": False, "error": f"could not start llama-server: {e}"}
    try:
        if WINDOWS:
            try:
                _kill_with_parent_windows(p, None)
            except Exception:
                pass
        try:
            guard = start_commit_guard(p, env)
        except Exception:
            guard = None
        holder["guard"] = guard
        while not healthy(port, key):
            if p.poll() is not None:
                r = {"ok": False, "error": f"llama-server stopped while loading (exit {p.returncode})" + (": " + str(guard.stopped) if guard is not None and guard.stopped else "")}
                try:
                    errf.seek(0, 2); errf.seek(max(0, errf.tell() - 65536))
                    if GGML_ABORT_RE.search(errf.read().decode("utf-8", "replace")):
                        r["kv_abort"] = True          # the known abort (Granite 1B with a q8_0 KV cache): the caller may measure the same plan with f16 KV
                except Exception:
                    pass
                return r
            if left() <= 0:
                return {"ok": False, "error": "the load did not finish in the time left"}
            time.sleep(0.5)
        load_s = time.monotonic() - t0
        if left() <= 0:           # healthy, but after the time was up (a slow poll): the budget is the owner's limit, so this plan is not measured
            return {"ok": False, "load_s": load_s, "error": "the load did not finish in the time left"}
        url = f"http://127.0.0.1:{port}/completion"
        if left() > 20:      # one untimed pass of the same prompt first: the first pass pages the experts in and wakes the GPU, which would make the first plan look slow
            http(url, {"prompt": AUTOTUNE_PROMPT * 11, "n_predict": 1, "cache_prompt": False, "temperature": 0}, key, timeout=max(5, left()))
        st, body = http(url, {"prompt": AUTOTUNE_PROMPT * 11, "n_predict": 1, "cache_prompt": False, "temperature": 0}, key, timeout=max(5, left()))
        pp = json.loads(body).get("timings", {}) if st == 200 else {}
        if not pp.get("prompt_per_second"):
            return {"ok": False, "load_s": load_s, "error": f"the prompt probe failed (HTTP {st})"}
        st, body = http(url, {"prompt": "Write a long story about a lighthouse.", "n_predict": int(decode_n), "ignore_eos": True, "cache_prompt": False, "temperature": 0.7}, key,
                        timeout=max(5, left()))
        dd = json.loads(body).get("timings", {}) if st == 200 else {}
        if not dd.get("predicted_per_second"):
            return {"ok": False, "load_s": load_s, "error": f"the decode probe failed (HTTP {st})"}
        return {"ok": True, "load_s": load_s, "prompt_tps": pp["prompt_per_second"], "prompt_n": pp.get("prompt_n"),
                "decode_tps": dd["predicted_per_second"], "decode_n": dd.get("predicted_n")}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            p.terminate(); p.wait(timeout=20)
        except Exception:
            try:
                p.kill(); p.wait(timeout=10)
            except Exception:
                pass
        try:
            errf.close()
        except Exception:
            pass


def commit_fits(a, cmd, env, argv, model, ctx, ncmoe, ub, kv_q8, gpu, machine, mm_bytes=0):
    """LAUNCHER-1042: (True, "") when the LAUNCHER-991 commit check would start `cmd` as it is; (False, why) when it would refuse it or shrink it (a probe of the unshrunk plan is
    the load that crashed the desktop). Same decision as commit_stage, said nowhere. Without a Windows commit figure, with --no-commit-check or MOE_CACHE_COMMIT_CHECK=0
    commit_stage changes nothing, and so everything fits."""
    old_notes, old_quiet, old_cpu = getattr(_TL, "notes", None), getattr(_TL, "quiet", 0), EXPECT_CPU_WORK[0]
    _TL.notes, _TL.quiet = [], old_quiet + 1
    try:
        new_cmd, _n, _u, _k, refusal = commit_stage(a, list(cmd), list(env), argv, model, ctx, ncmoe, ub, kv_q8, gpu, machine, mm_bytes)
    finally:
        _TL.quiet, EXPECT_CPU_WORK[0] = old_quiet, old_cpu
        if old_notes is None:
            del _TL.notes
        else:
            _TL.notes = old_notes
    if refusal:
        return False, refusal
    if list(new_cmd) != list(cmd):
        return False, "the commit check would shrink this plan (smaller batch, mmap, more weights in RAM)"
    return True, ""


_VRAM_AS_GIVEN = object()     # autotune_stage: no vram_user passed (tests, older callers): a.vram is taken as the user's own value


def autotune_stage(a, argv, cmd, env, key_env, model, ctx, ls, machine, engine, n_layers, say_fn=None, runner=None, size=None, avail=None, fits=None, vram_user=_VRAM_AS_GIVEN):
    """LAUNCHER-994: returns (cmd, env, info). info is None when the tuner does not apply (silent: --exact / --stock / MOE_CACHE_NO_AUTOTUNE), else a dict for the plan:
    {"state": cached|tuned|too_slow|pending|..., "lines": [...], "picked", "rows", ...}. Reads the stored pick for this model + machine + context; measures only when
    a.autotune_probe is set (a real start, not --dry-run / --plan-json / the control page's plan) and there is no stored pick or --retune was given."""
    import moe_cache_autotune as at
    say_fn = say_fn or say
    if not model or not os.path.isfile(model):
        return cmd, env, None
    off = "--exact" if a.exact else "--stock" if a.stock else "MOE_CACHE_NO_AUTOTUNE is set" if os.environ.get("MOE_CACHE_NO_AUTOTUNE", "") not in ("", "0") else None
    if off:            # said, not silent: a user who looks for the tuner finds out why it is not there
        line = f"moe-cache: auto-tune: off ({off})"
        say_fn(line)
        return cmd, env, {"state": "off", "lines": [line], "why": f"off ({off})"}
    locked = at.locked_dims(a, argv, os.environ, windows=machine.windows, agent=bool(a.agent), project=bool(a.project), engine=engine)
    # what else shapes the plan a pick was measured on (LAUNCHER-1041): the user's memory caps, the modes (--fast, --mtp, --approx, long-prompt, --hybrid) and the settings the
    # tuner does not touch but the speed depends on (the prompt cache size the USER asked for - the planner's own value is not keyed: on Windows it follows the free commit of the moment -,
    # the KV type of the plan, flash attention)
    limits = {k: str(getattr(a, k)) for k in ('ram', 'vram', 'max_ram', 'gpu_margin', 'leave_room', 'fast', 'mtp', 'approx', 'long_prompt', 'no_long_prompt', 'hybrid', 'always_plugin') if getattr(a, k, None)}
    limits.update({k: str(v) for k, v in (("cache_ram", arg_after(argv, "--cache-ram", "-cram")), ("ctk", arg_after(cmd, "-ctk", "--cache-type-k")), ("fa", arg_after(cmd, "-fa", "--flash-attn"))) if v is not None})
    if vram_user is not _VRAM_AS_GIVEN:
        # gpu_stage / multi_gpu_start overwrite a.vram with the free memory of the moment (Windows AMD / Intel: "14.65"); keyed on that, a stored pick was found again only when free
        # GPU memory was the same to 0.01 GiB, so nearly every start measured anew (up to 180 s). Only the --vram the USER gave is a limit of the key. (Found by the outside review of 7c3d024.)
        limits.pop("vram", None)
        if vram_user:
            limits["vram"] = str(vram_user)
    parts = at.key_parts(autotune_model_id(model), ctx, autotune_machine_sig(machine, ls), locked, a.agent, engine, limits)
    def apply_stored(stored):
        if stored.get("state") == "too_slow":
            info = dict(stored, state="too_slow", lines=[f"moe-cache: auto-tune: not tried again: {stored.get('why')}"], locked=locked)
            say_fn(info["lines"][0])
            return cmd, env, info
        if stored.get("bad"):           # LAUNCHER-1045: this pick did not load once; the plan as it is runs until --retune measures again
            line = (f"moe-cache: auto-tune: the stored pick {stored.get('picked')} failed to load on {stored['bad'].get('at')} ({stored['bad'].get('why', '')[:120]}) and is not used again: "
                    "the plan as it is runs; --retune measures again")
            say_fn(line)
            return cmd, env, dict(stored, state="bad", lines=[line], locked=locked, key=at.key_of(parts), changed=False)
        edits, held = at.replay_edits(cmd, stored, autotune_free_vram(a, machine))      # LAUNCHER-994 review: a part that puts more on the GPU needs as much free GPU memory as when it was measured
        new_cmd, new_env, skipped = at.apply_edits(cmd, env, edits, locked)
        lines = at.summary_lines(stored, cached=True, skipped=skipped, locked=locked)
        lines = [lines[0]] + [l for l in lines[1:] if "picked" in l or "stays" in l or "left alone" in l or "not applied" in l or "--retune" in l]       # the table is in the control page / --plan-json
        if held:
            lines.append("moe-cache: auto-tune: " + held)
        for l in lines:
            say_fn(l)
        # "before": the plan as the planner made it (for the fall-back when this pick does not load); "key_parts": where the file lives (to mark it bad)
        info = dict(stored, state="cached", lines=lines, locked=locked, key=at.key_of(parts), key_parts=parts, before=(list(cmd), list(env)))
        if held:
            info["held_back"] = held
            info["changed"] = bool(edits)       # nothing of the pick applied: a failed load is then not the pick's fault (tune_retry leaves the file alone)
        return new_cmd, new_env, info
    expired = []
    def fresh(stored):
        """None for a stored pick older than PICK_MAX_AGE_DAYS (said once; the plan as it is runs and the next real start measures again)."""
        age = at.pick_age_days(stored) if stored else None
        if age is not None and age > at.PICK_MAX_AGE_DAYS and stored.get("state") != "too_slow":
            expired.append(f"moe-cache: auto-tune: the stored pick is {age:.0f} days old (tuned {stored.get('tuned_at')}; picks expire after {at.PICK_MAX_AGE_DAYS} days): measured again at the next real start")
            say_fn(expired[-1])
            return None
        if (stored and stored.get("state") in ("tuned", "cached") and not stored.get("bad") and not stored.get("free_vram_mib")
                and at.adds_gpu_use(cmd, at.pick_edits(stored)) and autotune_free_vram(a, machine)):
            # a result stored before the free GPU memory was recorded: a pick that adds GPU use cannot be compared with anything, and would stay unusable for good (a stored result is
            # never measured again). It counts as stale: the next real start measures again and records the figure.
            expired.append("moe-cache: auto-tune: the stored result has no record of the free GPU memory it was measured with, and its pick puts more on the GPU: measured again at the next real start")
            say_fn(expired[-1])
            return None
        return stored
    stored = None if a.retune else fresh(at.load(CACHE_DIR, parts))
    if stored:
        return apply_stored(stored)
    # LAUNCHER-1044: a model that cannot be loaded 3-5 times in the budget is not tuned (the default 75 GB model: one load alone is longer than 180 s). Measured nowhere: this
    # is a size rule: more than 40 % of this computer's RAM, or more than the memory the server may use (the planner expects spill).
    try:
        ram = float(machine.mem_total())
    except Exception:
        ram = 0.0
    if size and ((ram and float(size) > 0.40 * ram) or (avail and float(size) > float(avail))):
        line = (f"moe-cache: auto-tune: skipped: the model ({float(size) / 1e9:.1f} GB) is " + (f"more than 40 % of this computer's RAM ({ram / 1e9:.0f} GB)" if ram and float(size) > 0.40 * ram
                else f"bigger than the memory the server may use ({float(avail) / 1e9:.1f} GB)") +
                f": loading it up to 5 times does not fit the {at.BUDGET_S:.0f} s budget, so the plan as it is runs untuned")
        if getattr(a, "autotune_probe", False):
            say_fn(line)             # a plan that only describes (--dry-run, --plan-json, the control page) shows it in plan.autotune and stays as it was
        return cmd, env, {"state": "skipped", "lines": [line], "locked": locked}
    if not getattr(a, "autotune_probe", False):
        line = "moe-cache: auto-tune: " + ("--retune: the plans are measured again on the next real start" if a.retune else
                                           (expired[0].split("auto-tune: ", 1)[1] if expired else
                                            f"no stored pick for this model, machine and context yet: the first real start measures up to {at.BUDGET_S:.0f} s (--exact skips it)"))
        return cmd, env, {"state": "pending", "lines": [line], "locked": locked}      # not said: a plan that only describes stays as it was (the goldens); plan.autotune carries it
    why_not = None
    try:
        others = machine.other_llama_servers()
    except Exception:
        others = []
    if others:
        why_not = "another llama-server is running (its memory and speed would change the probe)"
    elif port_busy(a.port):
        why_not = f"port {a.port} is busy"
    if why_not:
        line = f"moe-cache: auto-tune: not measured now: {why_not}; it is tried again at the next start"
        say_fn(line)
        return cmd, env, {"state": "skipped", "lines": [line], "locked": locked}
    got, holder = at.acquire_lock(CACHE_DIR)             # LAUNCHER-1041: one tuning at a time
    if not got:
        line = (f"moe-cache: auto-tune: another start (pid {holder}) is measuring right now: the plan as it is runs untuned this time and the pick is looked up again at the next start"
                if holder else "moe-cache: auto-tune: the tuning lock could not be taken: the plan as it is runs untuned this time")
        say_fn(line)
        return cmd, env, {"state": "skipped", "lines": [line], "locked": locked}
    try:
        if not a.retune:                                  # the other start may have finished while this one waited for its turn
            stored = fresh(at.load(CACHE_DIR, parts))
            if stored:
                return apply_stored(stored)
        return _autotune_measure(a, cmd, env, key_env, ctx, n_layers, locked, parts, say_fn, runner, fits, free_vram=autotune_free_vram(a, machine))
    finally:
        at.release_lock(CACHE_DIR)


def autotune_free_vram(a, machine):
    """Free GPU memory in bytes now (--vram, else the machine's own reading), None when unknown."""
    try:
        v = getattr(a, "vram", None)
        if v:
            return float(v) * 2**30
        return machine.free_vram_bytes() or None
    except Exception:
        return None


def _autotune_measure(a, cmd, env, key_env, ctx, n_layers, locked, parts, say_fn, runner, fits, free_vram=None):
    import moe_cache_autotune as at
    cands = at.build_candidates(cmd, env, locked, ctx, n_layers)
    if fits is not None and len(cands) > 1:          # LAUNCHER-1042: every command that would be loaded has passed the commit check first
        ok, why = fits(cmd, env)
        if not ok:
            line = f"moe-cache: auto-tune: skipped: the commit check does not start the plan as it is unchanged ({why[:200]}), so nothing is loaded to measure"
            say_fn(line)
            return cmd, env, {"state": "skipped", "lines": [line], "locked": locked}
        kept, dropped = [cands[0]], []
        for c in cands[1:]:
            c_cmd, c_env, _ = at.apply_edits(cmd, env, c.edits)
            ok, why = fits(c_cmd, c_env)
            (kept if ok else dropped).append(c if ok else c.id)
        if dropped:
            say_fn(f"moe-cache: auto-tune: not tried (the commit check would refuse or shrink them): {', '.join(dropped)}")
        cands = kept
    if len(cands) < 2:
        res = {"rows": [{"id": "C0", "label": "the plan as it is", "status": "not measured", "edits": []}], "total_s": 0, "budget_s": at.BUDGET_S, "picked": "C0", "changed": False,
               "state": "tuned", "why": "every setting that could be tuned is yours or does not apply here; nothing to try", "tuned_at": time.strftime("%Y-%m-%d %H:%M")}
        at.save(CACHE_DIR, parts, res)
        lines = at.summary_lines(res, locked=locked)
        for l in lines:
            say_fn(l)
        return cmd, env, dict(res, lines=lines, locked=locked)
    if a.retune:       # not "first start": the stored pick is being replaced
        say_fn(f"moe-cache: auto-tune: --retune: measuring again, trying {len(cands)} plans for at most {at.BUDGET_S:.0f} s (the stored result for this model, machine and context is replaced)")
    else:
        say_fn(f"moe-cache: auto-tune: first start for this model, machine and context: trying {len(cands)} plans for at most {at.BUDGET_S:.0f} s (stored afterwards; --retune measures again)")
    kv_f16 = [False]         # after llama.cpp's known load abort with the launcher's q8_0 KV: this and the later plans are measured with f16 KV (the start's own retry ends there too)
    def run(c, left_s, n):
        c_cmd, c_env, _ = at.apply_edits(cmd, env, c.edits)
        if kv_f16[0]:
            c_cmd = kv_retry_cmd(c_cmd)[0]
            if fits is not None:        # the plans above passed the commit check with the KV types they had; f16 needs more, so this command is checked again (LAUNCHER-1042)
                ok, why = fits(c_cmd, c_env)
                if not ok:
                    return {"ok": False, "error": f"not loaded: the commit check does not pass this plan with the f16 KV cache ({why[:120]})"}
        t_start = time.monotonic()
        r = (runner or autotune_probe)(c_cmd, dict(os.environ, **dict(c_env + key_env)), left_s, n)
        if not r.get("ok") and r.get("kv_abort") and not kv_f16[0] and "kv" not in locked:       # a KV type the user gave (or --exact) is never changed
            new_cmd, was = kv_retry_cmd(c_cmd)
            if was and (fits is None or fits(new_cmd, c_env)[0]):
                say_fn(f"moe-cache: auto-tune: llama.cpp aborted at load with the {was} KV cache (known: stock llama-server on Granite 1B): the plans are measured with the f16 KV cache from here, like the start's own retry")
                kv_f16[0] = True
                r = (runner or autotune_probe)(new_cmd, dict(os.environ, **dict(c_env + key_env)), left_s - (time.monotonic() - t_start), n)
        return r
    res = at.tune(cands, run, agent=bool(a.agent), say=lambda t: say_fn("moe-cache: auto-tune:   " + t), progress=lambda t: say_fn("moe-cache: auto-tune: " + t))
    if free_vram:
        res["free_vram_mib"] = int(free_vram / 2**20)       # what a pick that puts more on the GPU is later compared with (apply_stored)
    try:
        if res.get("state") != "aborted":          # LAUNCHER-1042: a guard stop stores nothing
            at.save(CACHE_DIR, parts, res)
    except OSError as e:
        say_fn(f"moe-cache: auto-tune: could not store the result ({e}); it is measured again at the next start", err=True)
    new_cmd, new_env, skipped = at.apply_edits(cmd, env, at.pick_edits(res), locked)
    lines = [l for l in at.summary_lines(res, locked=locked) if not l.startswith("moe-cache: auto-tune:   ")]
    for l in lines:
        say_fn(l)
    return new_cmd, new_env, dict(res, lines=lines, locked=locked, key=at.key_of(parts), key_parts=parts, before=(list(cmd), list(env)))


def cuda_backend(ls):
    """True when the CUDA backend of ggml sits next to this llama-server (ggml-cuda.dll / libggml-cuda.so)."""
    try:
        d = os.path.dirname(which(ls) or ls)
        return any(os.path.isfile(os.path.join(d, n)) for n in ("ggml-cuda.dll", "libggml-cuda.so"))
    except Exception:
        return False


def commit_stage(a, cmd, env, argv, model, ctx, ncmoe, ub, kv_q8, gpu, machine, mm_bytes=0):
    """LAUNCHER-991, stage "commit": Windows charges GPU allocations (AMD / Intel Vulkan drivers), copied weights, the KV cache of a CPU-only run, the prompt cache and
    llama-server's own buffers to the commit limit (RAM + page file); when it runs out Windows kills programs (the PC, 2026-11-09: claude.exe, dwm.exe, codex.exe).
    Estimates the whole server (planner/commit_guard.py), and when free commit after the load would be under max(6 GB, 12.5 % of the limit) tries smaller layouts (mmap instead of
    copies, smaller -ub, smaller prompt cache, q8_0 KV, more weights in RAM; never a shorter context; anything the user fixed stays) and otherwise refuses.
    Returns (cmd, ncmoe, ub, kv_q8, refusal text or None). Linux and macOS: nothing here (the run-time guard watches MemAvailable)."""
    if not machine.windows or a.no_commit_check or os.environ.get("MOE_CACHE_COMMIT_CHECK") == "0":
        return cmd, ncmoe, ub, kv_q8, None
    limit, free = machine.win_commit()
    if not limit or free is None:         # free commit of exactly 0 is a figure (the PC is out of commit: the worst case to refuse for), not "not reported"
        say("moe-cache: warning: Windows did not report the commit figures: the commit check is skipped for this start", err=True)
        return cmd, ncmoe, ub, kv_q8, None
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        import commit_guard as cgd
        f = cgd.facts_from_model(model)
    except Exception as e:
        say(f"moe-cache: warning: the commit check is skipped for this start (the model's layout could not be read: {type(e).__name__}: {e})", err=True)
        return cmd, ncmoe, ub, kv_q8, None
    vram = gpu[2] * 2**20 if gpu and gpu[2] > 0 else (float(a.vram) * 2**30 if a.vram else None)
    L = cgd.layout_from_cmd(cmd, env, f, vram_free=vram, mm_bytes=mm_bytes)
    ev = dict(env)
    user = " ".join(argv)
    flags = lambda *names: any(re.search(r"(^| )" + re.escape(n) + r"( |=|$)", user) for n in names)
    locked = set()
    if a.ub != "auto" or flags("-ub", "--ubatch-size", "-b", "--batch-size"): locked.add("ub")
    if a.exact or flags("-ctk", "--cache-type-k", "-ctv", "--cache-type-v"): locked.add("kv")
    if flags("--cache-ram", "-cram") or arg_after(cmd, "--cache-ram") is None: locked.add("cache")      # no --cache-ram in the command: this llama-server may not have the option
    if ev.get("GGML_CPU_GPU_SPLIT") or flags("-ot", "--override-tensor", "-ngl", "--n-gpu-layers", "--gpu-layers", "--n-cpu-moe") or a.n_cpu_moe is not None or a.hybrid:
        locked.add("weights")
    # AMD / Intel Vulkan drivers charge GPU allocations to commit (measured: PC 2026-11-09). NVIDIA through the CUDA backend is not known to; an NVIDIA card on the Vulkan
    # build (the kit's llama-server) is not known either way, so it counts like AMD (errs high) unless the CUDA backend sits next to llama-server (LAUNCHER-991 review).
    gpu_counts = not (machine.nvidia_reports() and cuda_backend(cmd[0]))
    d = cgd.decide(f, L, free, limit, locked=locked, allow_q8=not a.exact and not a.stock, windows=gpu_counts)
    if d["ok"] and not d["steps"]:
        return cmd, ncmoe, ub, kv_q8, None
    if machine.nvidia_reports():
        # LAUNCHER-991 review: whether an NVIDIA card's memory is charged to Windows commit is NOT measured (CUDA or Vulkan): nothing here may refuse a start, or change the layout, on
        # that guess. It says so, with the estimate, and starts as planned; the run-time guard stops llama-server before Windows runs out of commit.
        say(f"moe-cache: WARNING: free commit may be tight for this plan (estimate: {cgd.parts_words(d['est'])}; {free / 1e9:.1f} GB free of {limit / 1e9:.1f} GB; the estimate is for AMD / Intel "
            "GPUs). Whether an NVIDIA card's memory counts as Windows commit is not measured, so this is a warning, not a refusal and the plan is not changed: starting as planned; "
            "the memory guard stops llama-server if commit runs out. Close big programs if you can", err=True)
        return cmd, ncmoe, ub, kv_q8, None
    if not d["ok"]:
        txt = cgd.refusal_text(d, free, limit, (machine.win_memory() or (None, None))[0] or machine.mem_total())
        if "weights" in locked:
            txt += " (-ot / -ngl / --n-cpu-moe given by you are not changed)"
        return cmd, ncmoe, ub, kv_q8, txt
    new_cmd, n_moe, new_ub, new_q8, ffn = cgd.apply_layout_to_cmd(cmd, L, d["layout"], f)
    say(cgd.apply_plan_text(d["steps"]) + f" (estimate: {cgd.parts_words(d['est'])}; {free / 1e9:.1f} GB free of {limit / 1e9:.1f} GB; --no-commit-check keeps the first plan)", err=True)
    if ffn or (not f["dense"] and n_moe and n_moe > (L.get("n_cpu_moe") or 0)):
        EXPECT_CPU_WORK[0] = True
    return new_cmd, (str(n_moe) if (n_moe is not None and not f["dense"]) else ncmoe), (str(new_ub) if new_ub != L["ub"] else ub), new_q8, None


def _launch_plan(a, extra, argv, machine):
    win = machine.windows
    dry = a.dry_run or a.plan_json         # a plan that only describes (or prints) what would run: no files written, llama-server need not exist
    if not a.model:
        from moe_cache_pick import picked       # bin/moe_cache_pick.py
        p = picked()
        if not p:
            from moe_cache_pick import pick_missing
            raise SystemExit("usage: moe-cache-server MODEL.gguf [options]  (no model given: " +
                             (pick_missing() or "none picked: `moe_cache.py pick` recommends one for this machine") + ")")
        a.model = p[0]
        a.ctx = a.ctx or p[1]
        a.agent = a.agent or p[2]
        say(f"moe-cache: no model given: using the picked one, {a.model} (context {a.ctx}" + (", agent mode" if a.agent else "") + "; moe_cache.py pick)")
    if a.max_ram:
        _TL.max_ram = a.max_ram          # the launcher's planning reads it (this thread, this plan); llama-server gets it through the plan's env
    if not win and any(x == "--api-key" or x.startswith("--api-key=") for x in argv):
        # llama-server gets it through the environment, but this launcher's own command line stays readable (/proc) for the server's lifetime
        say("moe-cache: note: a key on the command line is visible to other users of this machine (ps); LLAMA_API_KEY or the stored key "
            "(moe_cache.py key) keep it private", err=True)
    if not win:
        os.umask(0o077)       # the usage profile is private to this user
    if a.leave_room:
        if a.ram:
            say(f"moe-cache: --leave-room: --ram {a.ram} GB as given")
        elif (lr := leave_room_ram_gb(machine)):
            a.ram = lr
            say(f"moe-cache: --leave-room: the server may use {lr} GB of RAM (the rest stays for your own programs) and runs at a lower priority: smoother PC, slower model "
                "(the GPU memory and CPU cores it leaves free are reported below when they were applied)")
    if a.share_ram:
        share_ram_start(a, machine)
    user_flags, extra = adopt_user_flags(extra)
    user_extra = list(extra)             # LAUNCHER-994: what the user gave after `--` (the stages below add their own flags to extra); the auto-tune leaves those settings alone
    if "port" in user_flags and user_flags["port"] != a.port:
        say(f"moe-cache: port {user_flags['port']} from the flags after '--' (the health checks, warm sessions and agent snippets use it)")
        a.port = user_flags["port"]
    if "ctx" in user_flags and user_flags["ctx"] != a.ctx:
        say(f"moe-cache: context {user_flags['ctx']} from the flags after '--' (the GPU plan uses it)")
        a.ctx = user_flags["ctx"]
    if "ncmoe" in user_flags and a.n_cpu_moe is None:
        a.n_cpu_moe = user_flags["ncmoe"]
    ls, model, port = a.llama_server, a.model, a.port
    ctx = a.ctx or (65536 if a.agent else 8192)     # agents send 5-20k-token preambles (tools, instructions); Hermes needs at least 64000
    gcache = a.gpu_cache
    if a.hybrid and gcache != "off":
        say("moe-cache: --hybrid replaces --gpu-cache: the GPU memory goes to the hybrid bank"); gcache = "off"
    if not os.path.isfile(model):
        raise SystemExit(f"usage: moe-cache-server MODEL.gguf [options]  (model file not found: '{model}')")
    approx_stage(a, model)
    if a.stock and (a.fast or a.hybrid or a.always_plugin):
        raise SystemExit("--stock runs without the plugin: it cannot go with --fast, --hybrid or --always-plugin")
    stock, size, avail = decide(model, a.ram, plugin_wanted=bool(a.fast or a.hybrid or a.always_plugin), stock_wanted=a.stock, machine=machine)
    mm_file, mm_bytes, dense = None, 0, False
    mm_file, mm_bytes, dense, stock, size = vision_dense_start(a, extra, model, stock, size, avail)
    if stock and not dense and not a.stock:
        why = plugin_for_partial_offload(a, ls, model, ctx, machine)
        if why:
            say(why); stock = False
    lib = plugin_lib_stage(ls, dry, stock, win, machine)
    vram_user, listing, gpu = gpu_stage(a, ls, win, machine)
    try:
        ag = auto_gpu_device(extra, listing if listing is not None else machine.list_devices(ls), a.cpu_only, a.multi_gpu)
    except Exception:
        ag = None
    if ag:
        extra += ["--device", ag[0]]
        say(f"moe-cache: GPUs: {', '.join(ag[2])}; using {ag[0]} ({ag[1]}): the integrated GPU is left out (it shares system RAM and is far slower); --device after -- chooses otherwise")
    ncmoe_user = a.n_cpu_moe
    mg = multi_gpu_start(a, extra, machine, ls, listing, vram_user, win) if a.multi_gpu else None       # PROTOTYPE (--multi-gpu); None = off
    ncmoe = a.n_cpu_moe
    ncmoe, planner_info, planner_error = plan_placement(a, model, ctx, ncmoe, machine)
    # API key (DECISION_LLAMA_SERVER_KEY, option C): a key given on our command line or after `--` goes to llama-server through the environment
    # (LLAMA_API_KEY), not its argument list; agent mode and --project always get the stable per-user key. Plain chat servers get none, but a CORS
    # allow-list so a normal web page cannot read their answers.
    lan_addr = lan_stage(a, extra)
    api_key, key_env = api_key_stage(a, extra, lan_addr)
    extra += amd_windows_checkpoint_args(gpu, model, extra, agent=a.agent)
    alias, slotdir = agent_stage(a, extra, model)
    extra += template_stage(a, extra, model, dry)
    extra += mtp_args(a.mtp, a.agent or getattr(a, "approx_spec", False), ls, model, extra, exact=a.exact, dense=dense)
    extra += sampling_args(a.no_sampling_defaults, a.agent, model, extra)
    host = arg_after(extra, "--host") or ""
    if not (host == "" or host.startswith("127.") or host in ("localhost", "::1")) and not api_key:
        say(f"WARNING: llama-server will listen on '{host}' without --api-key: anyone who can reach this machine can use the model and the web UI. "
            "Add --api-key KEY after the -- or bind to 127.0.0.1.", err=True)
    pname = a.profile or ""
    if pname and not NAME_RE.fullmatch(pname):
        raise SystemExit("--profile NAME may only contain letters, digits, _ and -")
    if a.project:
        if not NAME_RE.fullmatch(a.project):
            raise SystemExit("--project NAME may only contain letters, digits, _ and -")
        pname = pname or a.project
        slotdir = os.path.join(CACHE_DIR, "slots", a.project); os.makedirs(slotdir, exist_ok=True)
        extra += ["--slot-save-path", slotdir]
    profile = profile_stage(a, model, pname, dry)
    load, method, gcache, env, cap = memory_env_stage(a, ls, lib, model, profile, stock, gcache, machine)
    # --fast also stores the KV cache as q8_0 (idea I6: KL 0.0045 on its own, RESULTS 53): half the KV memory, which at agent-sized contexts is the
    # VRAM a bigger micro-batch needs. Your own -ctk / -ctv after the -- win.
    # at 128k+ context the q8_0 KV cache is the default (owner 2026-10-30): the quality gate passed it at 4k context (KL 0.0079, tasks = stock,
    # RESULTS 72) and at 256k on a 16 GB card the smaller KV leaves room for a bigger GPU plan (prompts about 2x). Not with --stock (that compares
    # against plain llama.cpp) and not without a GPU (q8_0 KV aborted on the CPU-only build in CI; nothing to gain there). --exact (or your own
    # -ctk/-ctv after --) keeps llama.cpp's f16 KV
    kv_q8 = kv_stage(a, extra, ctx, ls, listing, machine, stock, dense)
    fast_env_stage(a, env, stock, model, machine)
    pinargs = cpu_pin_stage(a, env, extra, stock, win, machine, dense, model)
    # micro-batch size and GPU cache size: one shared tuner (planner/gguf_plan.py: tune), so the launcher and the benchmark tool decide the same way
    # (a micro-batch the user chose, with --ub N or llama-server's own -ub, goes into the same decision: the layers must still leave the long-prompt
    # block its room; before, a chosen -ub skipped the tuner and long prompts ran on the CPU, Flash-Next at 34k tokens 121-141 tok/s, RESULTS 61)
    tuned, desk_kw, ub_asked = tune_launch(a, extra, ls, model, ctx, machine, gpu, gcache, kv_q8, mm_bytes)
    kv_q8, tuned, desk_kw, ub_asked = long_prompt_stage(a, extra, ls, model, ctx, machine, gpu, gcache, kv_q8, mm_bytes, tuned, desk_kw, ub_asked)
    ub, ncmoe, no_vram = layers_stage(a, extra, model, ctx, ncmoe, ncmoe_user, gcache, kv_q8, tuned, desk_kw, machine)
    ncmoe = gpu_cache_stage(a, extra, env, ls, model, listing, gcache, ncmoe, tuned, stock, profile, machine)
    # short prompts: in the approximate modes, batches under 64 tokens stay on the CPU (the exact default keeps llama.cpp's own threshold, 32)
    # DELIBERATE (owner decision 2026-11-08): the long-prompt default sets a.fast = "kv", so this 64-token threshold applies to every default launch of a big model at 32k+ context,
    # not only to --fast users. It is part of what the +77 % prompt speed (claim v038-longprompt-default) measured; do not drop it for the long-prompt default without a new measurement.
    if (a.hybrid or a.fast) and not os.environ.get("MOE_CACHE_GPU_MIN_TOKENS"):
        env.append(("MOE_CACHE_GPU_MIN_TOKENS", "64"))
    # llama-server keeps finished prompts in host RAM (--cache-ram, 8 GiB by default), which competes with the expert cache: capped at 2 GiB here
    # (the plugin keeps the same amount free). Pass --cache-ram yourself to change it.
    cram = prompt_cache_stage(a, extra, env, ls, machine, win, dry)
    env, method = stock_env_stage(a, env, stock, ncmoe, size, avail, method)
    if not env_has(env, "ZES_ENABLE_SYSMAN"):
        env += sycl_sysman_env(ls, listing)
    extra = leave_room_stage(a, env, extra, pinargs, machine)
    ncmoe, extra = multi_gpu_stage(a, env, extra, mg, model, ctx, kv_q8, ub, gcache, ncmoe, ncmoe_user, tuned, stock, machine, win)
    cmd, extra = command_stage(a, ls, model, ctx, port, ncmoe, extra, pinargs, vram_user, dense, method, profile, machine, kv_q8, mm_bytes)
    cmd = cpu_gpu_split_stage(a, env, cmd, ls, dense)
    # LAUNCHER-994: the measured pick (stored per model + machine + context) edits the finished command; the commit check below still has the last word on memory
    ncmoe0, ub0, kv_q80 = ncmoe, ub, kv_q8
    cmd, env_t, tune_info = autotune_stage(a, list(argv) + user_extra, cmd, env, key_env, model, ctx, ls, machine, "stock" if method == "none (stock)" else "plugin", (planner_info or {}).get("layers"),
                                          size=size, avail=avail, vram_user=vram_user,
                                          fits=lambda c_cmd, c_env: commit_fits(a, c_cmd, c_env, argv, model, ctx, arg_after(c_cmd, "--n-cpu-moe", "-ncmoe") or ncmoe, arg_after(c_cmd, "-ub", "--ubatch-size") or ub,
                                                                                 kv_quantized(c_cmd), gpu, machine, mm_bytes))
    env[:] = env_t                          # the same list object the plan keeps
    if tune_info and tune_info.get("changed"):
        ub = arg_after(cmd, "-ub", "--ubatch-size") or ub
        n_ = arg_after(cmd, "--n-cpu-moe", "-ncmoe")
        ncmoe = n_ if n_ is not None else ncmoe
        kv_q8 = kv_quantized(cmd)        # q8_0, the dense ladder's K q8_0 / V q4_0 and q4_0/q4_0 all count (a plain == "q8_0" read q4_0 as f16 and let the commit check put q8_0 back)
    cmd_picked = list(cmd)
    cmd, ncmoe, ub, kv_q8, commit_refusal = commit_stage(a, cmd, env, argv, model, ctx, ncmoe, ub, kv_q8, gpu, machine, mm_bytes)
    tune_fallback = None
    if tune_info and tune_info.get("changed") and tune_info.get("before") and tune_info.get("state") in ("cached", "tuned"):
        # LAUNCHER-1041: the commit check may have changed the picked plan after all: the end state is said, not the pick alone
        diff = [f"{n} {arg_after(cmd_picked, *f)} -> {arg_after(cmd, *f)}" for n, f in (("-ub", ("-ub", "--ubatch-size")), ("--n-cpu-moe", ("--n-cpu-moe", "-ncmoe")), ("-np", ("-np", "--parallel")))
                if arg_after(cmd_picked, *f) != arg_after(cmd, *f)]
        if diff:
            line = f"moe-cache: auto-tune: the commit check changed the picked plan; the server starts with: {', '.join(diff)}"
            say(line)
            tune_info["lines"] = list(tune_info.get("lines", [])) + [line]
            tune_info["final_differs"] = diff
        before_cmd, before_env = tune_info["before"]
        tuned_env = {k for k, v in env_t if (k, v) not in before_env} | {k for k, v in before_env if (k, v) not in env_t}
        def tune_fallback():
            """LAUNCHER-1045: the plan as the planner made it (through the same commit check), for the one retry when the picked plan does not load -> (cmd, env list) or None."""
            c2, _n, _u, _k, refusal = commit_stage(a, list(before_cmd), list(before_env), argv, model, ctx, ncmoe0, ub0, kv_q80, gpu, machine, mm_bytes)
            if refusal:
                return None
            return c2, [(k, v) for k, v in env if k not in tuned_env] + [(k, v) for k, v in before_env if k in tuned_env]
    new_cram = arg_after(cmd, "--cache-ram")
    if new_cram is not None and cram is not None and re.fullmatch(r"[0-9]+", str(new_cram)) and str(new_cram) != str(cram):
        cram = new_cram                     # the stage shrank the prompt cache: the plan and the plugin's host reserve follow
        env[:] = [(k, str(host_reserve_mib(new_cram, a.agent)) if k == "MOE_CACHE_HOST_RESERVE_MIB" else v) for k, v in env]
    if commit_refusal:
        say(commit_refusal, err=True)       # the plan still describes the launch (--dry-run, the control page); run_launch_plan refuses to start it
    say(f"moe-cache: web UI and API will be at http://127.0.0.1:{port}")
    # Windows has no cgroups: --ram became a job memory limit on llama-server when the plugin runs (WP3). Stock runs are not limited by it (their
    # model lives in the file cache, which job limits do not count), so it is left off there. A job limit caps committed memory, and Windows
    # commits Vulkan's GPU allocations too: with the file-view cache (the default since RESULTS 59) the experts use no commit, so the limit no
    # longer bounds the cache at all, only llama.cpp's own GPU memory (--ram 16 on the PC: "ggml_vulkan: Device memory allocation ... failed",
    # RESULTS 60). A Linux cgroup does not count GPU memory either. So the job limit is set only for the private-memory cache
    # (MOE_CACHE_WIN_VIEW=0); --ram still sizes the cache (MOE_CACHE_RAM_GIB).
    win_view = os.environ.get("MOE_CACHE_WIN_VIEW", "1") != "0"
    win_limit = float(a.ram) * 1e9 if (win and a.ram and method != "none (stock)" and not win_view) else None
    if a.project and not dry:
        write_server_id(slotdir, model, ctx, cmd)        # after the long-prompt stage: the KV types in cmd are final
    plan = LaunchPlan()
    plan.a, plan.extra, plan.argv_in = a, extra, argv
    plan.model, plan.ls, plan.lib, plan.port, plan.ctx = model, ls, lib, port, ctx
    plan.stock, plan.size, plan.avail, plan.method = stock, size, avail, method
    plan.engine = "stock" if method == "none (stock)" else "plugin"
    plan.n_cpu_moe, plan.ncmoe_user, plan.ub, plan.kv_q8 = ncmoe, ncmoe_user, ub, kv_q8
    plan.cache_ram, plan.profile, plan.alias, plan.slotdir, plan.host = cram, profile, alias, slotdir, host
    plan.api_key, plan.cmd, plan.env, plan.key_env, plan.win_limit = api_key, cmd, env, key_env, win_limit
    plan.machine, plan.commit_refusal = machine, commit_refusal
    plan.tuned, plan.planner_info, plan.planner_error = tuned, planner_info, planner_error
    plan.autotune = tune_info
    plan.tune_fallback = tune_fallback
    return plan


def run_launch_plan(plan):
    """--dry-run: print the plan's command; otherwise start llama-server with it and keep it running (warm-keeper, snippets, OOM retry)."""
    a, extra, machine = plan.a, plan.extra, plan.machine
    model, ls, port, ctx, host = plan.model, plan.ls, plan.port, plan.ctx, plan.host
    cmd, env, key_env, method, win_limit = plan.cmd, plan.env, plan.key_env, plan.method, plan.win_limit
    api_key, alias, slotdir, kv_q8, ncmoe_user = plan.api_key, plan.alias, plan.slotdir, plan.kv_q8, plan.ncmoe_user
    if plan.commit_refusal and not a.dry_run:
        raise SystemExit(plan.commit_refusal)
    if a.dry_run:
        if plan.job_limit_note():
            say(plan.job_limit_note())
        say(plan.command_line())
        return 0
    # audit 2026-10-29 L2: checked before any helper thread starts, as learn and check do: with another server on the port the restore thread
    # sent it our project state (with the stored key) and the plugin check called it ours (a missing llama-server is refused further up)
    if port_busy(port):
        raise SystemExit(f"moe-cache: port {port} is already in use (another llama-server, or on-demand: moe_cache.py on-demand --status); "
                         f"stop it or pick another with --port")
    say(loading_line(model))      # CACHE-181: the first load of a big model is silent for minutes
    pl = llama_patch_line(ls)     # the kit's patched llama.cpp says so (every patch built in and off); an official build prints nothing
    if pl:
        say(f"moe-cache: {pl}")
    if a.project and os.path.isfile(os.path.join(slotdir, a.project + ".bin")) and not project_id_ok(slotdir, a.project):
        say(f"moe-cache: not restoring the saved state of project {a.project}: it has no record of the model, context and cache types it was made with, "
            "or they differ from this server's (a state is only valid for the same model file); save it again with `moe_cache.py project save`", err=True)
    elif a.project and os.path.isfile(os.path.join(slotdir, a.project + ".bin")):
        def restore():
            for _ in range(900):
                if healthy(port, api_key):
                    st, body = http(f"http://127.0.0.1:{port}/slots/0?action=restore", {"filename": a.project + ".bin"}, api_key, timeout=600)
                    if st == 200:
                        say(f"moe-cache: restored the saved prompt state of project {a.project}", err=True)
                    else:
                        say(f"moe-cache: could not restore project {a.project} (HTTP {st}): {body[:200]}", err=True)
                    return
                time.sleep(1)
        threading.Thread(target=restore, daemon=True).start()
    if a.agent and not a.no_warm and not a.project and slotdir:
        # a saved state is only valid for the same model file, context and KV settings: all of them go into the name
        ident = model_identity(model, ctx, arg_after(extra, "-ctk", "--cache-type-k"), arg_after(extra, "-ctv", "--cache-type-v"))
        tag = re.sub(r"[^A-Za-z0-9_-]", "_", f"{alias}-{ident}")      # file name + size + content fingerprint + ctx + KV types: files of another model never match
        if ident: threading.Thread(target=warm_keeper, args=(port, api_key, slotdir, tag.rstrip("-_")), daemon=True).start()
    if a.agent:
        def snippets():
            for _ in range(1800):
                if healthy(port, api_key):
                    say(agent_snippets(port, alias, ctx, api_key, host, hide_key=True), err=True)
                    return
                time.sleep(1)
        threading.Thread(target=snippets, daemon=True).start()
    if a.open:
        def opener():
            for _ in range(600):
                if healthy(port, api_key):
                    import webbrowser
                    webbrowser.open(f"http://127.0.0.1:{port}")
                    return
                time.sleep(1)
        threading.Thread(target=opener, daemon=True).start()
    def oom_retry():
        """llama-server could not allocate GPU memory at load: measure the free memory again, plan again with it (never fewer CPU expert layers than
        before + 1) and return the new command; None when nothing can be changed (the user fixed --n-cpu-moe, or there is no GPU figure)."""
        if "--n-cpu-moe" not in cmd and ncmoe_user is None:       # LAUNCHER-850: a dense model
            free = float(a.vram) * 2**30 if a.vram else None
            if free is None:
                try:
                    free = machine.free_vram_bytes()
                except Exception:
                    free = None
            if free is None and WINDOWS:
                g = pick_gpu_full(list_devices(ls))
                free = g[3] * 2**20 if g and g[3] > 0 else None
            margin = a.gpu_margin if a.gpu_margin is not None else 0.25
            user_ot = any(x in ("-ot", "--override-tensor") or x.startswith("--override-tensor=") for x in extra)
            kvq = kv_plan_type(extra, kv_q8)
            new, what = dense_oom_replan(cmd, model, ctx, kvq, free, margin, user_ot)
            if not new:
                say(f"moe-cache: llama-server ran out of GPU memory while loading ({what}): free some GPU memory (a game or browser may have taken it), or use a smaller --ctx / -ub / --gpu-margin", err=True)
                return None
            say("moe-cache: llama-server ran out of GPU memory while loading (another program took some after the plan" + (f", now {free / 2**30:.1f} GiB free" if free else "") + f"): trying again once with {what}")
            return new
        if ncmoe_user is not None or "--n-cpu-moe" not in cmd:
            say("moe-cache: llama-server ran out of GPU memory while loading: free some GPU memory, or give a larger --n-cpu-moe / smaller -ub / --gpu-margin", err=True)
            return None
        try:
            sys.path.insert(0, os.path.join(HERE, "planner"))
            from gguf_plan import tune, desk_reserve_gb, gpu_total_free_bytes, model_layout
            if WINDOWS:
                g = pick_gpu_full(list_devices(ls))
                tf = (g[3] * 2**20, g[2] * 2**20) if g and g[3] > 0 else None
            else:
                tf = gpu_total_free_bytes()
            if not tf:
                return None
            reserve = (a.gpu_margin * 2**30 / 1e9) if a.gpu_margin is not None else max(desk_reserve_gb(*tf)[0], 1.0, LEAVE_ROOM_VRAM_GB if a.leave_room else 0)
            t = tune(model, ctx, free=tf[1], want_cache=False, kv_q8=kv_q8, reserve_gb=reserve)
            lay = model_layout(model)
            old_n = int(cmd[cmd.index("--n-cpu-moe") + 1])
            new_n = min(lay["n_layers"] or 999, max(int(t["n_cpu_moe"]), old_n + 1)) if lay["n_layers"] else old_n + 1
            new = list(cmd)
            new[new.index("--n-cpu-moe") + 1] = str(new_n)
            new_ub = None
            if "-ub" in new and t["ub"] < int(new[new.index("-ub") + 1]):
                new_ub = str(t["ub"])
                new[new.index("-ub") + 1] = new_ub
                if "-b" in new:
                    new[new.index("-b") + 1] = str(max(int(new_ub), 2048))
            say(f"moe-cache: llama-server ran out of GPU memory while loading (now {tf[1] / 2**30:.1f} GiB free, another program took some after the plan): "
                f"trying again once with {new_n} layers of experts on the CPU (was {old_n})" + (f" and -ub {new_ub}" if new_ub else ""))
            return new
        except Exception as e:
            say(f"moe-cache: llama-server ran out of GPU memory while loading; could not plan again ({e})", err=True)
            return None
    def kv_retry():
        """llama.cpp aborted at load (the scheduler's hash set is full) with the quantized K and V cache this launcher chose (--fast, the default from 128k on, or the dense KV ladder:
        q8_0/q8_0, K q8_0 / V q4_0, q4_0/q4_0): start once more with the f16 cache. Not when -ctk / -ctv came from the user (kv_q8 is False then)."""
        if not kv_q8:
            return None
        new, was = kv_retry_cmd(cmd)
        if not was:
            return None
        if a.project and slotdir:
            write_server_id(slotdir, model, ctx, new)      # the retry runs with the f16 KV cache: server.id must say so
        say(f"moe-cache: llama.cpp aborted at load with the {was} KV cache (ggml-impl.h fatal error: seen with stock llama-server b12064 on Granite 1B with q8_0, any context): "
            "trying again once with the f16 KV cache (--exact does that from the start)", err=True)
        return new
    def tune_retry(rc, seen):
        """LAUNCHER-1045: llama-server stopped before it was ready with a plan the auto-tune picked (a stored pick is applied at every start, so a pick that fails once would fail
        at every start): mark the stored pick bad, start once with the plan as the planner made it. None when no pick was applied or the plan cannot be rebuilt."""
        nonlocal cmd, env
        t = plan.autotune
        if not (t and t.get("changed") and t.get("key_parts") and plan.tune_fallback):
            return None
        new = plan.tune_fallback()
        if not new:
            return None
        import moe_cache_autotune as at
        try:
            at.mark_bad(CACHE_DIR, t["key_parts"], f"llama-server stopped before it was ready (exit {rc}) with {t.get('picked')}")
        except OSError:
            pass
        say(f"moe-cache: auto-tune: llama-server did not come up with the picked plan {t.get('picked')} (exit {rc}): the pick is marked bad and the plan as it is starts once; "
            "--retune measures again", err=True)
        cmd, env = new
        return new[0], dict(os.environ, **dict(env + key_env))
    return run_child(cmd, dict(os.environ, **dict(env + key_env)), port=port, expect_plugin=method != "none (stock)", mem_limit=win_limit,
                     allow_sleep=a.allow_sleep, oom_retry=oom_retry, kv_retry=kv_retry, tune_retry=tune_retry)


def warm_save_wait(n_tokens, last_tokens, bytes_per_token, poll=10, idle_polls=2, per_gb=20, small_wait=180, small_per_gb=30, small_cap=600):
    """Idea N27: seconds a slot must stay unchanged before warm_keeper saves it. A save rewrites the whole slot (~7 GB at 256k on Flash-Next) and
    llama-server serves nothing else while it writes, so the bigger the file, the longer the wait (20 s + 20 s per GB). A slot that only grew a little
    since its last save (less than max(4096 tokens, a quarter)) waits much longer (180 s + 30 s per GB, at most 10 min): the file on disk is an older
    prefix of the same conversation, which a restore picks up exactly (llama-server processes only the tokens after it). A new or shorter
    conversation counts as a big change. bytes_per_token comes from the last save (None before the first: the plain 20 s)."""
    gb = n_tokens * bytes_per_token / 1e9 if bytes_per_token else 0.0
    base = poll * idle_polls
    if last_tokens is None or n_tokens < last_tokens or n_tokens - last_tokens >= max(4096, last_tokens // 4):
        return base + per_gb * gb
    return min(small_wait + small_per_gb * gb, small_cap)


def clean_stale_saves(slotdir, tag, older_than=120):
    """LAUNCHER-005 (A3-28): deletes <tag>-sN.new.bin files a server died in the middle of saving (multi-GB, never counted in disk checks). A file touched in the last
    `older_than` seconds may be a save in progress (another server on this tag) and stays. Returns the number deleted."""
    n = 0
    for f in glob.glob(os.path.join(glob.escape(slotdir), tag + "-s*.new.bin")):
        try:
            if time.time() - os.path.getmtime(f) > older_than:
                os.unlink(f)
                n += 1
        except OSError:
            pass
    return n


def warm_keeper(port, api_key, slotdir, tag, poll=10, idle_polls=2, min_tokens=2048, keep=2):
    """Ideas I29 / I34: at start, restore the saved conversations (one file per slot, at most `keep`, longest first) into the slots they came from; afterwards
    save every idle slot of min_tokens or more (with its context checkpoints) once it has been unchanged for warm_save_wait() seconds (N27: longer for
    big files and for small changes). Each agent keeps
    its own slot (llama-server picks the slot whose prompt matches best), so Hermes and OpenCode both start warm after a restart. Saves go to a
    temporary name first, so a stop during a save never leaves a broken file. When the shared KV pool fills, llama-server clears idle slots itself."""
    # idea S09 (MOE_CACHE_WARM_KEEP=N, default 2): restore only the N longest saved conversations at start (0 = none), not the two largest states every time
    kv = os.environ.get("MOE_CACHE_WARM_KEEP", "")
    keep = int(kv) if kv.isdigit() else keep
    url = f"http://127.0.0.1:{port}"
    for _ in range(3600):
        if healthy(port, api_key):
            break
        time.sleep(1)
    else:
        return
    clean_stale_saves(slotdir, tag)
    saved_tokens = {}           # slot -> token count of its file on disk (N27)
    bytes_per_token = None      # from the last save or restore (file size / tokens); sets the waits of warm_save_wait()
    files = [f for f in glob.glob(os.path.join(glob.escape(slotdir), tag + "-s*.bin")) if re.search(r"-s(\d+)\.bin$", f) and not f.endswith(".new.bin")]
    for f in sorted(files, key=os.path.getsize, reverse=True)[:keep]:
        to = int(re.search(r"-s(\d+)\.bin$", f).group(1))      # back into the slot it came from, so later saves replace this file
        t0 = time.time()
        st, body = http(f"{url}/slots/{to}?action=restore", {"filename": os.path.basename(f)}, api_key, timeout=600)
        try:
            n = json.loads(body).get("n_restored") if st == 200 else None
        except ValueError:
            n = None
        if n:
            saved_tokens[to] = n     # the file on disk holds n tokens: the slot's next save is judged against it (N27)
            bytes_per_token = os.path.getsize(f) / n
        say(f"moe-cache: warm session: restored a saved agent conversation ({n} tokens) into slot {to} in {time.time() - t0:.1f} s" if st == 200
            else f"moe-cache: warm session: could not restore {os.path.basename(f)} (HTTP {st}): {body[:160]}", err=True)
    saved, seen, tries = {}, {}, {}
    while True:
        time.sleep(poll)
        st, body = http(f"{url}/slots", api_key=api_key)
        if st != 200:
            if st == 0 and not healthy(port, api_key):
                return       # the server is gone
            continue
        try:
            slots = json.loads(body)
        except ValueError:
            continue
        if any(x.get("is_processing") for x in slots):
            seen = {}
            continue
        for x in slots:
            if (x.get("n_prompt_tokens") or 0) < min_tokens:
                continue
            sid = x["id"]
            key = (x.get("id_task"), x.get("n_prompt_tokens"))
            seen[(sid, key)] = seen.get((sid, key), 0) + 1
            if saved.get(sid) == key:
                continue
            wait = warm_save_wait(x["n_prompt_tokens"], saved_tokens.get(sid), bytes_per_token, poll, idle_polls)
            if seen[(sid, key)] * poll < wait:
                continue
            fname, tmp = f"{tag}-s{sid}.bin", f"{tag}-s{sid}.new.bin"
            t0 = time.time()
            st, body = http(f"{url}/slots/{sid}?action=save", {"filename": tmp}, api_key, timeout=600)
            tries[(sid, key)] = tries.get((sid, key), 0) + 1
            if tries[(sid, key)] >= 3:
                saved[sid] = key       # three failures for the same state: stop trying until the conversation changes
            if st != 200:
                say(f"moe-cache: warm session: save failed (HTTP {st}): {body[:160]}", err=True)
                continue
            try:
                os.replace(os.path.join(slotdir, tmp), os.path.join(slotdir, fname))
                saved[sid] = key       # only a save that really landed counts (a failed one is tried again at the next idle moment)
                saved_tokens[sid] = x["n_prompt_tokens"]
                bytes_per_token = os.path.getsize(os.path.join(slotdir, fname)) / x["n_prompt_tokens"]
                say(f"moe-cache: warm session: saved slot {sid} ({x.get('n_prompt_tokens')} tokens, "
                    f"{os.path.getsize(os.path.join(slotdir, fname)) / 1e6:.0f} MB) in {time.time() - t0:.1f} s", err=True)
            except OSError as e:
                say(f"moe-cache: warm session: save failed ({e})", err=True)


def server_info(url, api_key=None):
    """(model name, context) of a running llama-server, from /v1/models and /props; (None, None) when it cannot be reached."""
    st, body = http(url.rstrip("/") + "/v1/models", api_key=api_key)
    if st != 200:
        return None, None
    try:
        name = json.loads(body)["data"][0]["id"]
    except (ValueError, KeyError, IndexError):
        name = None
    ctx = None
    st, body = http(url.rstrip("/") + "/props", api_key=api_key)
    if st == 200:
        try:
            ctx = int(json.loads(body)["default_generation_settings"]["n_ctx"])
        except (ValueError, KeyError, TypeError):
            pass
    return name, ctx


def server_models(url, api_key=None):
    """Every model id /v1/models lists (one for llama-server; all of them for an on-demand endpoint with several models)."""
    st, body = http(url.rstrip("/") + "/v1/models", api_key=api_key)
    try:
        return [m["id"] for m in json.loads(body)["data"]] if st == 200 else []
    except (ValueError, KeyError, TypeError):
        return []


def opencode_header_timeout(ctx):
    """OpenCode waits headerTimeout ms (default 300000, 5 min) for a reply to start. A first turn that fills a 64k window took 289 s on the PC
    (RESULTS 64), longer at 128k+, so for windows of 64k and more connect sets 30 min; None keeps OpenCode's default (scout 14, opencode.ai docs)."""
    return 1800000 if ctx and int(ctx) >= 65536 else None


def opencode_config_target(cfg):
    """The global opencode.json to write, and warnings. OpenCode (and OpenCode Desktop, whose server reads the same global folder) merges config.json,
    opencode.json, then opencode.jsonc, later files winning, and creates opencode.jsonc on first run: write into that one when it is plain JSON,
    else into opencode.json and warn if the .jsonc sets its own model. OPENCODE_CONFIG (a file merged after these) is reported too."""
    warn = []
    jc = os.path.splitext(cfg)[0] + ".jsonc"
    if os.path.isfile(jc):
        text = open(jc, encoding="utf-8").read()
        try:
            json.loads(text)
            cfg = jc
        except ValueError:
            if '"model"' in text:
                warn.append(f"{jc} (has comments, so it is not edited here) sets its own \"model\" and wins over {os.path.basename(cfg)}: "
                            "change it there to moe-cache/NAME, or paste the block 'moe_cache.py connect opencode' prints")
    if os.environ.get("OPENCODE_CONFIG"):
        warn.append(f"OPENCODE_CONFIG={os.environ['OPENCODE_CONFIG']} is merged after this file and wins where they differ")
    return cfg, warn


HERMES_MARK = os.path.join(CACHE_DIR, "hermes-connected.json")      # {config.yaml path: the base URL connect last set}


def hermes_connected(h, base):
    """Remember that connect set this base URL in this Hermes config, so a later connect can tell its own setting from a local server the user set up themselves."""
    try:
        r = subprocess.run([h, "config", "path"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        path = (r.stdout.strip().splitlines() or [""])[-1].strip() if r.returncode == 0 else ""
        if not path:
            return
        try:
            marks = json.load(open(HERMES_MARK, encoding="utf-8"))
            marks = marks if isinstance(marks, dict) else {}
        except (OSError, ValueError):
            marks = {}
        marks[path] = base
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = stage_private(HERMES_MARK, json.dumps(marks))
        os.replace(tmp, HERMES_MARK)
    except (OSError, subprocess.SubprocessError):
        pass


HERMES_KEY_ENV = "MOE_CACHE_API_KEY"      # the variable in Hermes' own .env that config.yaml's model.api_key points at


def hermes_env_file(h):
    """Hermes' .env (next to its config.yaml, where Hermes loads its own keys from), or None when `hermes config path` does not answer."""
    try:
        r = subprocess.run([h, "config", "path"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        path = (r.stdout.strip().splitlines() or [""])[-1].strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return None
    return os.path.join(os.path.dirname(path), ".env") if path else None


def put_env_line(env_file, name, value):
    """AGENTS-036: set NAME=value in a .env file (mode 0600, other lines kept, one line for NAME) without the value ever being on a command line."""
    try:
        lines = open(env_file, encoding="utf-8").read().splitlines()
    except OSError:
        lines = []
    shown = value if re.fullmatch(r"[A-Za-z0-9._~+/=:@-]+", value) else '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    mine = re.compile(r"\s*(export\s+)?" + re.escape(name) + r"\s*=")
    out, done = [], False
    for l in lines:
        if mine.match(l):
            if not done:
                out.append(f"{name}={shown}")
            done = True
        else:
            out.append(l)
    if not done:
        out.append(f"{name}={shown}")
    write_private(env_file, "\n".join(out) + "\n")


def hermes_backup(h):
    """Before 'hermes config set' replaces the main model: copy config.yaml to config.yaml.bak and describe the model it had. A backup is never overwritten:
    when config.yaml.bak exists the copy gets a time stamp (config.yaml.bak.YYYYmmdd-HHMMSS). Nothing is copied when the config is our own setting from an earlier
    connect (this base URL was recorded by it) and a backup exists: a second connect must not bury the user's own setup, e.g. OpenRouter, under ours.
    Returns (backup path, "provider P, model M, URL U") or (None, "") when nothing was kept."""
    def get(k):
        r = subprocess.run([h, "config", "get", k], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return r.stdout.strip() if r.returncode == 0 else ""
    try:
        r = subprocess.run([h, "config", "path"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        path = (r.stdout.strip().splitlines() or [""])[-1].strip() if r.returncode == 0 else ""
        old = {k: get("model." + k) for k in ("provider", "default", "base_url")}
    except (OSError, subprocess.SubprocessError):
        return None, ""
    if not path or not os.path.isfile(path):
        return None, ""
    try:
        marks = json.load(open(HERMES_MARK, encoding="utf-8"))
    except (OSError, ValueError):
        marks = {}
    ours = isinstance(marks, dict) and old["provider"] == "custom" and marks.get(path) == old["base_url"]
    if ours and glob.glob(glob.escape(path) + ".bak*"):
        return None, ""
    bak = path + ".bak"
    if os.path.lexists(bak):
        bak = f"{path}.bak.{time.strftime('%Y%m%d-%H%M%S')}"
        n = 1
        while os.path.lexists(bak):
            n += 1
            bak = f"{path}.bak.{time.strftime('%Y%m%d-%H%M%S')}-{n}"
    shutil.copy2(path, bak)
    if not WINDOWS:
        os.chmod(bak, 0o600)               # it may hold the old provider's key
    return bak, f"provider {old['provider'] or '(not set)'}, model {old['default'] or '(not set)'}, URL {old['base_url'] or '(not set)'}"


def claude_env(url, model, api_key):
    """The environment Claude Code needs to talk to llama-server's Anthropic-compatible endpoint (/v1/messages, also accepts count_tokens; checked
    on llama.cpp b11966, Windows PC 2026-10-29: 200 with x-api-key and with Authorization: Bearer). ANTHROPIC_AUTH_TOKEN is sent as Bearer and does
    not trigger Claude Code's 'use this API key?' question. The base URL has no /v1: Claude Code adds /v1/messages itself."""
    env = {"ANTHROPIC_BASE_URL": url.rstrip("/"), "ANTHROPIC_AUTH_TOKEN": api_key or "none"}      # Claude Code wants some token even for a server without a key
    env["ANTHROPIC_MODEL"] = model
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"      # a block at the start of the system prompt changes per request, so llama-server would read the whole prompt again each turn
    env["ANTHROPIC_SMALL_FAST_MODEL"] = model       # its background calls (titles, summaries) would otherwise ask for a Haiku the server does not have
    return env


def claude_snippets(env, ctx, hide_key=False):
    shown = {k: (KEY_HINT if hide_key and k == "ANTHROPIC_AUTH_TOKEN" and v != "none" else v) for k, v in env.items()}
    return "\n".join(["  Claude Code, bash / zsh (this terminal only):"] + [f"    export {k}={json.dumps(v)}" for k, v in shown.items()]
                      + ["  Claude Code, PowerShell (this window only):"] + [f"    $env:{k} = {json.dumps(v)}" for k, v in shown.items()]
                      + [f"  then run: claude   (context {ctx}: Claude Code's own system prompt is long; the server should run with --agent or --ctx 65536 or more)"])


def claude_backup_path(cfg):
    """Where the first backup of a Claude Code settings file is kept: ~/.cache/moe-cache/backups, not beside it. The project file lives in a repository
    folder and its backup can hold the user's own ANTHROPIC key: a .bak next to it would be committed by a blanket add."""
    import hashlib
    real = os.path.realpath(cfg)
    return os.path.join(CACHE_DIR, "backups", hashlib.sha1(real.encode("utf-8", "replace")).hexdigest()[:12] + "-" + os.path.basename(real) + ".bak")


def write_claude_settings(cfg, env):
    """Merge env into the "env" object of a Claude Code settings file (project-local by default), private file (it holds the key). The first version of an
    existing file is kept (claude_backup_path); returns that backup's path, None when there is none."""
    cfg = os.path.realpath(cfg)
    data, eol, bak = {}, "\n", None
    if os.path.isfile(cfg):
        try:
            data = json.load(open(cfg, encoding="utf-8"))
        except ValueError:
            raise SystemExit(f"{cfg} is not plain JSON: add the lines printed by 'moe_cache.py connect claude' by hand")
        if not isinstance(data, dict):
            raise SystemExit(f"{cfg} does not hold a JSON object")
        eol = file_eol(cfg)
        bak = claude_backup_path(cfg)
        if not os.path.isfile(bak):
            os.makedirs(os.path.dirname(bak), mode=0o700, exist_ok=True)
            shutil.copy2(cfg, bak)                          # the first backup is kept: a second connect does not replace what the user had
            if not WINDOWS:
                os.chmod(bak, 0o600)
    cur = data.get("env") if isinstance(data.get("env"), dict) else {}
    cur.update(env)
    data["env"] = cur
    write_private(cfg, json_text(data), eol)
    return bak


def gitignore_settings_local(folder):
    """AGENTS-714: .claude/settings.local.json holds the API key; in a git project it is added to .gitignore when no line already covers it.
    Returns True when a line was added. Never touches a folder that is not a git project."""
    if not os.path.isdir(os.path.join(folder, ".git")):
        return False
    gi = os.path.join(folder, ".gitignore")
    try:
        old = open(gi, encoding="utf-8").read() if os.path.isfile(gi) else ""
    except OSError:
        return False
    names = (".claude/settings.local.json", "/.claude/settings.local.json", "settings.local.json", ".claude/", "/.claude/", ".claude", "/.claude", ".claude/*")
    if any(l.strip() in names for l in old.splitlines()):
        return False
    try:
        with open(gi, "a", encoding="utf-8") as f:
            f.write(("" if not old or old.endswith("\n") else "\n") + ".claude/settings.local.json\n")
    except OSError:
        return False
    return True


def project_folder_hint(cwd):
    """A warning when cwd is a sub-folder of a project (a parent holds .git or .claude) and has no .claude of its own: Claude Code started in the project root
    would not read a settings file written here. None otherwise."""
    if os.path.isdir(os.path.join(cwd, ".claude")):
        return None
    d = os.path.dirname(cwd)
    while d and d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, ".git")) or os.path.isdir(os.path.join(d, ".claude")):
            return (f"this folder is inside the project {d}: Claude Code started there does not read {os.path.join(cwd, '.claude')}; "
                    f"run the command from {d} (or give --config {os.path.join(d, '.claude', 'settings.local.json')}) to cover the whole project")
        d = os.path.dirname(d)
    return None


# ---- connect: Codex CLI, Pi, OpenClaw, Aider, Qwen Code ----
# Formats read from each agent's own docs on 2026-11-03 (links in docs/AGENTS.md). Every writer keeps the first backup as .bak, never replaces it,
# writes the file private (it holds the API key) and refuses a file that is not the expected format instead of guessing.

QWEN_KEY_VAR = "MOE_CACHE_API_KEY"
CODEX_BEGIN = "# >>> moe-cache (written by 'moe_cache.py connect codex'; remove this block to undo)"
CODEX_END = "# <<< moe-cache"
CODEX_TOP = ("model", "model_provider", "model_context_window", "web_search")
AIDER_MARK = "# written by 'moe_cache.py connect aider'; delete this file to undo. Run: aider --config THIS_FILE"


KEY_HINT = "<your API key: python3 bin/moe_cache.py key>"      # shown in a preview instead of the key (terminal scrollback, logs, shared screens)


def agent_window(ctx):
    """The window handed to an agent: the server's context minus 12.5 % headroom (docs/AGENTS.md 'Keep the agent's own context limit below the
    server's'), in whole 1024s, so the agent's own compaction starts before llama-server refuses a request. 65536 -> 57344."""
    return max(4096, (int(ctx) - int(ctx) // 8) // 1024 * 1024)


def agent_output(ctx):
    return min(32768, max(4096, int(ctx) // 4))


def bak_path(cfg):
    """Where the first backup of cfg lives: next to the real file (a symlinked config is a link into a dotfiles repo or a store: the backup belongs beside what is replaced)."""
    return os.path.realpath(cfg) + ".bak"


def keep_first_backup(cfg):
    """Copy an existing file to cfg.bak unless a backup is already there (the first one is the user's own setup). A symlink is followed: the backup sits beside the
    real file, which is also the one write_private replaces. Returns the backup path, None when there was nothing to copy."""
    cfg = os.path.realpath(cfg)
    if os.path.isfile(cfg) and not os.path.isfile(cfg + ".bak"):
        shutil.copy2(cfg, cfg + ".bak")
        if not WINDOWS:
            os.chmod(cfg + ".bak", 0o600)
    elif os.path.isfile(cfg):
        keep_numbered_backup(cfg)
    return cfg + ".bak" if os.path.isfile(cfg + ".bak") else None


NUMBERED_BACKUPS = 5


def keep_numbered_backup(cfg):
    """AGENTS-714: besides the first backup (.bak), the file as it was before this write is kept as cfg.bak.1, .2, ... when it differs from the newest copy
    (the last NUMBERED_BACKUPS stay), so a second connect that changed something can be walked back. Returns the new path, None when nothing was copied."""
    nums = sorted((int(m.group(1)), f) for f in glob.glob(glob.escape(cfg) + ".bak.*") for m in [re.search(r"\.bak\.(\d+)$", f)] if m)
    newest = nums[-1][1] if nums else cfg + ".bak"
    try:
        if open(newest, "rb").read() == open(cfg, "rb").read():
            return None
    except OSError:
        return None
    new = f"{cfg}.bak.{(nums[-1][0] + 1) if nums else 1}"
    shutil.copy2(cfg, new)
    if not WINDOWS:
        os.chmod(new, 0o600)
    for _n, f in nums[:max(0, len(nums) + 1 - NUMBERED_BACKUPS)]:
        try:
            os.unlink(f)
        except OSError:
            pass
    return new


def agent_config_paths(agent, config=None):
    """The files `connect AGENT --write` changes by default (for --undo); the second one of aider is its model metadata file."""
    home = os.path.expanduser("~")
    if config:
        cfg = config
    elif agent == "codex":
        cfg = os.path.join(os.environ.get("CODEX_HOME") or os.path.join(home, ".codex"), "config.toml")
    elif agent == "pi":
        cfg = os.path.join(os.environ.get("PI_CODING_AGENT_DIR") or os.path.join(home, ".pi", "agent"), "models.json")
    elif agent == "openclaw":
        cfg = os.path.join(home, ".openclaw", "openclaw.json")
    elif agent == "qwen":
        cfg = os.path.join(home, ".qwen", "settings.json")
    elif agent == "aider":
        cfg = os.path.join(home, ".aider.moe-cache.conf.yml")
    elif agent == "claude":
        cfg = os.path.join(os.getcwd(), ".claude", "settings.local.json")
    else:
        cfg = opencode_config_target(os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config"), "opencode", "opencode.json"))[0]
    out = [cfg]
    if agent == "pi":
        out.append(os.path.join(os.path.dirname(cfg), "settings.json"))
    if agent == "aider":
        out.append(os.path.join(os.path.dirname(os.path.abspath(cfg)), ".aider.moe-cache.model.metadata.json"))
    return out


def connect_undo(a):
    """connect AGENT --undo: put back what the agent's config was before the first connect (the .bak copy; Claude Code's lives in the cache folder), or, with no
    backup, remove what connect added where that is exact (Codex's marked block; Aider's files, which connect made). The file as it is now is kept first
    as cfg.undone so nothing is lost. Hermes keeps its own config: the .bak path is in the message connect printed."""
    if a.agent == "hermes":
        raise SystemExit("moe-cache: Hermes: copy config.yaml.bak (kept beside Hermes' config.yaml by connect) back over config.yaml; `hermes config path` shows where it is")
    n = 0
    for cfg in agent_config_paths(a.agent, a.config):
        real = os.path.realpath(cfg)
        bak = claude_backup_path(cfg) if a.agent == "claude" else real + ".bak"
        if os.path.isfile(bak):
            if os.path.isfile(real):
                shutil.copy2(real, real + ".undone")
            shutil.copy2(bak, real)
            say(f"moe-cache: {a.agent}: {real} is the file from before the first connect again (from {bak}); the file as it was is kept as {real}.undone")
            n += 1
        elif a.agent == "aider" and os.path.isfile(real):
            os.replace(real, real + ".undone")
            say(f"moe-cache: aider: {real} moved to {real}.undone (connect had made it)")
            n += 1
        elif a.agent == "codex" and os.path.isfile(real) and CODEX_BEGIN in open(real, encoding="utf-8").read():
            text = open(real, encoding="utf-8").read()
            shutil.copy2(real, real + ".undone")
            b = text.index(CODEX_BEGIN)
            e = text.find(CODEX_END, b)
            e = len(text) if e < 0 else (text.find("\n", e) + 1 or len(text))
            write_private(real, text[:b] + text[e:])
            say(f"moe-cache: codex: the moe-cache block is removed from {real}; the file as it was is kept as {real}.undone (top-level model lines connect added may stay: check model_provider)")
            n += 1
    if not n:
        say(f"moe-cache: {a.agent}: nothing to undo: no backup and no marked block for {', '.join(agent_config_paths(a.agent, a.config))}", err=True)
        return 1
    return 0


def file_eol(cfg):
    """The line end of the existing file: CRLF when it uses Windows line ends (a rewrite keeps them), else LF."""
    try:
        with open(cfg, "rb") as f:
            return "\r\n" if b"\r\n" in f.read() else "\n"
    except OSError:
        return "\n"


def json_text(data):
    """The text of a JSON config we write: indent 2, non-ASCII kept as it is (a model or folder name stays readable), a final newline."""
    try:
        return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    except ValueError as e:
        raise SystemExit(f"cannot write the config: it holds a value JSON does not allow ({e}); nothing written")


def stage_private(cfg, text, eol="\n"):
    """Write text (UTF-8, `eol` line ends) to a temporary file of its own beside cfg, mode 0600, flushed to disk; returns its path. Never through a planted link."""
    tmp = f"{cfg}.{os.getpid()}.tmp"
    if os.path.lexists(tmp):
        os.unlink(tmp)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text if eol == "\n" else text.replace("\n", eol))
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return tmp


def write_private(cfg, text, eol="\n"):
    """Replace cfg with text: mode 0600 (the file holds the API key), through a temporary file. A symlinked cfg keeps being a link: the file it points at is replaced."""
    cfg = os.path.realpath(cfg)
    os.makedirs(os.path.dirname(cfg) or ".", exist_ok=True)
    tmp = stage_private(cfg, text, eol)
    try:
        os.replace(tmp, cfg)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json_object(cfg, agent, kind="config"):
    """The JSON object in cfg ({} when the file is missing); SystemExit naming the file when it is not plain JSON or not an object."""
    if not os.path.isfile(cfg):
        return {}
    try:
        data = json.load(open(cfg, encoding="utf-8"))
    except ValueError:
        raise SystemExit(f"{cfg} is not plain JSON (comments? JSON5?): nothing written; add the settings printed by 'moe_cache.py connect {agent}' by hand")
    if not isinstance(data, dict):
        raise SystemExit(f"{cfg} is not a {kind} object: nothing written; add the settings printed by 'moe_cache.py connect {agent}' by hand")
    return data


def want_dict(data, path, cfg, agent):
    """data[path[0]][path[1]]... as a dict, created when missing; SystemExit when something on the way is not an object."""
    cur = data
    for k in path:
        nxt = cur.setdefault(k, {})
        if not isinstance(nxt, dict):
            raise SystemExit(f"{cfg}: '{'.'.join(path)}' is not an object (it is a {type(nxt).__name__}): nothing written; "
                             f"add the settings printed by 'moe_cache.py connect {agent}' by hand")
        cur = nxt
    return cur


def model_rows(base, names, ctx, key):
    """Pi's models.json and OpenClaw's models.providers entry share this shape (pi.dev models docs; docs.openclaw.ai custom providers):
    baseUrl, api, apiKey, models[{id, name, reasoning, input, contextWindow, maxTokens}]."""
    return {"baseUrl": base, "api": "openai-completions", "apiKey": key or "none",
            "models": [{"id": n, "name": n, "reasoning": False, "input": ["text"], "contextWindow": agent_window(ctx), "maxTokens": agent_output(ctx)} for n in names]}


def qwen_entries(base, names, ctx):
    """Qwen Code's modelProviders.openai entries (envKey NAMES the variable; the key itself goes in settings.json's env block)."""
    return [{"id": n, "name": n, "envKey": QWEN_KEY_VAR, "baseUrl": base, "generationConfig": {"contextWindowSize": agent_window(ctx)}} for n in names]


def codex_blocks(base, name, ctx, key):
    """(top-level block, provider table block) for ~/.codex/config.toml. Codex speaks the Responses API (wire_api "responses" is the only value),
    which llama-server answers on /v1/responses; web_search is turned off because that tool is not a function tool and llama-server answers 400
    to it ('type' of tool must be 'function')."""
    q = json.dumps
    top = [CODEX_BEGIN, f"model = {q(name)}", 'model_provider = "moe-cache"', f"model_context_window = {agent_window(ctx)}", 'web_search = "disabled"', CODEX_END]
    tab = [CODEX_BEGIN, "[model_providers.moe-cache]", 'name = "moe-cache"', f"base_url = {q(base)}", 'wire_api = "responses"']
    if key:
        tab.append(f"experimental_bearer_token = {q(key)}")
    return "\n".join(top) + "\n", "\n".join(tab + [CODEX_END]) + "\n"


def codex_merge(text, top, tab):
    """Merge the two blocks into the text of config.toml: older blocks of ours are dropped, top-level keys of the same names that the user had
    are commented out ('#moe-cache-off# ' prefix, so they can be restored by hand), the keys go to the top and the table to the end.
    SystemExit when the user already defines the moe-cache provider themselves."""
    out, skipping = [], False
    for ln in text.splitlines():
        if ln.startswith("# >>> moe-cache"):
            skipping = True
        elif skipping and ln.startswith("# <<< moe-cache"):
            skipping = False
        elif not skipping:
            out.append(ln)
    if skipping:
        raise SystemExit("config.toml has a moe-cache block without its end line: nothing written; fix or remove it")
    body = "\n".join(out)
    if re.search(r'^\s*\[\s*model_providers\s*\.\s*"?moe-cache"?\s*\]', body, re.M) or re.search(r"^\s*model_providers\s*=", body, re.M):
        raise SystemExit("config.toml already defines [model_providers.moe-cache] (or model_providers inline) outside our block: nothing written; "
                         "edit it by hand with the settings printed by 'moe_cache.py connect codex'")
    res, in_top = [], True
    for ln in out:
        if ln.lstrip().startswith("["):
            in_top = False
        if in_top and re.match(r"^(%s)\s*=" % "|".join(CODEX_TOP), ln):
            ln = "#moe-cache-off# " + ln
        res.append(ln)
    rest = "\n".join(res).strip("\n")
    return top + ("\n" + rest + "\n" if rest else "") + "\n" + tab


def connect_extra(a, name, ctx, base, names):
    """connect codex|pi|openclaw|aider|qwen: print the settings, or with --write merge them into the agent's config."""
    key = a.api_key
    shown = key if a.write else (KEY_HINT if key else None)     # a preview goes to terminals and logs
    win = agent_window(ctx)
    home = os.path.expanduser("~")
    q = json.dumps

    def block(title, obj):
        return f"  {title}\n    " + json.dumps(obj, indent=2).replace("\n", "\n    ")

    def done(who, cfg, extra=""):
        return (f"{who} uses {name} at {base} (context {win} of the server's {ctx}); config {cfg}" + (" (old file: .bak)" if os.path.isfile(bak_path(cfg)) else "") + extra)
    if a.agent == "codex":
        cfg = a.config or os.path.join(os.environ.get("CODEX_HOME") or os.path.join(home, ".codex"), "config.toml")
        top, tab = codex_blocks(base, name, ctx, shown)
        if not a.write:
            say(f"  Codex CLI, {cfg}:\n" + "\n".join("    " + l for l in (top + tab).splitlines()))
            say(f"apply with: moe_cache.py connect codex --write  (merges into {cfg}; then run: codex)")
            return 0
        try:
            import tomllib      # Python 3.11+: the old file must parse, and so must the new one
        except ImportError:
            tomllib = None
        if tomllib is None:
            raise SystemExit(f"moe-cache: writing {cfg} needs Python 3.11 or newer (this is {sys.version_info[0]}.{sys.version_info[1]}): the merged file is checked as TOML there, and without the "
                             "check a config Codex refuses to start with could be written. Nothing written. Add the settings printed by 'moe_cache.py connect codex' by hand, "
                             "or run connect with a newer Python.")
        old, eol = "", "\n"
        if os.path.isfile(cfg):
            try:
                raw = open(cfg, "rb").read().decode("utf-8")
            except UnicodeDecodeError:
                raise SystemExit(f"{cfg} is not UTF-8 text: nothing written; add the settings printed by 'moe_cache.py connect codex' by hand")
            eol = "\r\n" if "\r\n" in raw else "\n"
            old = raw.replace("\r\n", "\n")
        if tomllib and old:
            try:
                tomllib.loads(old)
            except ValueError as e:
                raise SystemExit(f"{cfg} is not valid TOML ({e}): nothing written; add the settings printed by 'moe_cache.py connect codex' by hand")
        new = codex_merge(old, top, tab)
        if tomllib:
            try:
                got = tomllib.loads(new)
            except ValueError as e:
                raise SystemExit(f"{cfg}: the merged file would not be valid TOML ({e}): nothing written; add the settings printed by 'moe_cache.py connect codex' by hand")
            if got.get("model_provider") != "moe-cache" or got.get("model") != name:
                raise SystemExit(f"{cfg}: could not place the settings at the top level: nothing written; add the settings printed by 'moe_cache.py connect codex' by hand")
        keep_first_backup(cfg)
        write_private(cfg, new, eol)
        say("Codex CLI now " + done("", cfg, "; run: codex").lstrip())
        if re.search(r"^\s*profile\s*=", old, re.M):
            say("moe-cache: warning: config.toml sets a 'profile': keys in that profile win over the ones just written", err=True)
        return 0
    if a.agent == "pi":
        d = os.environ.get("PI_CODING_AGENT_DIR") or os.path.join(home, ".pi", "agent")
        cfg = a.config or os.path.join(d, "models.json")
        settings = os.path.join(os.path.dirname(cfg), "settings.json")
        prov = model_rows(base, names, ctx, shown)
        if not a.write:
            say(block(f"Pi, {cfg}:", {"providers": {"moe-cache": prov}}))
            say(block(f"Pi, {settings} (default model):", {"defaultProvider": "moe-cache", "defaultModel": name}))
            say("apply with: moe_cache.py connect pi --write  (merges into both files; then run: pi)")
            return 0
        data, sdata = load_json_object(cfg, "pi", "Pi models.json"), load_json_object(settings, "pi", "Pi settings")
        want_dict(data, ["providers"], cfg, "pi")["moe-cache"] = prov
        sdata["defaultProvider"], sdata["defaultModel"] = "moe-cache", name
        # both files or neither: written to temporary files first, then moved into place; when the second move fails the first file is put back
        pair = [(f, dd, file_eol(f), os.path.isfile(f)) for f, dd in ((cfg, data), (settings, sdata))]
        for f, dd, eol, existed in pair:
            keep_first_backup(f)
        olds = {os.path.realpath(f): (open(f, "rb").read() if existed else None) for f, dd, eol, existed in pair}
        staged = []
        try:
            for f, dd, eol, existed in pair:
                f = os.path.realpath(f)
                os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
                staged.append((f, stage_private(f, json_text(dd), eol)))
            moved = []
            try:
                for f, tmp in staged:
                    os.replace(tmp, f)
                    moved.append(f)
            except OSError:
                for f in moved:             # put the first file back as it was
                    o = olds.get(f)
                    try:
                        if o is None:
                            os.unlink(f)
                        else:
                            with open(f, "wb") as fh:
                                fh.write(o)
                    except OSError:
                        pass
                raise
        finally:
            for f, tmp in staged:
                if os.path.lexists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        say("Pi now " + done("", cfg, f"; default model in {settings}; run: pi").lstrip())
        return 0
    if a.agent == "openclaw":
        cfg = a.config or os.path.join(home, ".openclaw", "openclaw.json")
        prov = model_rows(base, names, ctx, shown)
        if not a.write:
            say(block(f"OpenClaw, {cfg} (JSON5 file; plain JSON is written):", {"models": {"providers": {"moe-cache": prov}},
                                                                                "agents": {"defaults": {"model": {"primary": f"moe-cache/{name}"}}}}))
            say("apply with: moe_cache.py connect openclaw --write  (merges into that file; a file with comments or JSON5 syntax is refused, not rewritten)")
            return 0
        data = load_json_object(cfg, "openclaw", "OpenClaw config")
        want_dict(data, ["models", "providers"], cfg, "openclaw")["moe-cache"] = prov
        dflt = want_dict(data, ["agents", "defaults"], cfg, "openclaw")
        if isinstance(dflt.get("model"), dict):
            dflt["model"]["primary"] = f"moe-cache/{name}"
        elif dflt.get("model") is None or isinstance(dflt.get("model"), str):      # the docs allow a plain string for the primary model
            dflt["model"] = {"primary": f"moe-cache/{name}"}
        else:
            say(f"moe-cache: warning: agents.defaults.model in {cfg} is not an object, so it is left alone: choose moe-cache/{name} in OpenClaw", err=True)
        eol = file_eol(cfg)
        keep_first_backup(cfg)
        write_private(cfg, json_text(data), eol)
        say("OpenClaw now " + done("", cfg, f"; restart its gateway, then use model moe-cache/{name}").lstrip())
        return 0
    if a.agent == "qwen":
        cfg = a.config or os.path.join(home, ".qwen", "settings.json")
        ents = qwen_entries(base, names, ctx)
        if not a.write:
            say(block(f"Qwen Code, {cfg}:", {"env": {QWEN_KEY_VAR: shown or "none"}, "modelProviders": {"openai": ents},
                                              "security": {"auth": {"selectedType": "openai"}}, "model": {"name": name}}))
            say("apply with: moe_cache.py connect qwen --write  (merges into that file; then run: qwen)")
            return 0
        data = load_json_object(cfg, "qwen", "Qwen Code settings")
        want_dict(data, ["env"], cfg, "qwen")[QWEN_KEY_VAR] = key or "none"
        mp = want_dict(data, ["modelProviders"], cfg, "qwen")
        cur = mp.setdefault("openai", [])
        if not isinstance(cur, list):
            raise SystemExit(f"{cfg}: modelProviders.openai is not a list: nothing written; add the settings printed by 'moe_cache.py connect qwen' by hand")
        mp["openai"] = [e for e in cur if not (isinstance(e, dict) and e.get("envKey") == QWEN_KEY_VAR)] + ents      # ours replaced, the user's own kept
        want_dict(data, ["security", "auth"], cfg, "qwen")["selectedType"] = "openai"
        want_dict(data, ["model"], cfg, "qwen")["name"] = name
        eol = file_eol(cfg)
        keep_first_backup(cfg)
        write_private(cfg, json_text(data), eol)
        say("Qwen Code now " + done("", cfg, "; run: qwen").lstrip())
        return 0
    # aider: its YAML cannot be merged without a YAML library, so it gets a file of its own (and a model-metadata file for the window)
    cfg = a.config or os.path.join(home, ".aider.moe-cache.conf.yml")
    meta = os.path.join(os.path.dirname(os.path.abspath(cfg)), ".aider.moe-cache.model.metadata.json")
    entries = {f"openai/{n}": {"max_tokens": agent_output(ctx), "max_input_tokens": win, "max_output_tokens": agent_output(ctx),
                              "input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"} for n in names}
    yml = "\n".join([AIDER_MARK, f"openai-api-base: {q(base)}", f"openai-api-key: {q(shown or 'none')}",
                     f"model: {q('openai/' + name)}", f"model-metadata-file: {q(meta)}"]) + "\n"
    if not a.write:
        say(f"  Aider, {cfg}:\n" + "\n".join("    " + l for l in yml.splitlines()) + "\n" + block(f"and {meta}:", entries))
        say(f"apply with: moe_cache.py connect aider --write  (writes those two files of its own, your .aider.conf.yml is not touched; then run: aider --config {cfg})")
        return 0
    old_meta = load_json_object(meta, "aider", "Aider model metadata")
    old_meta.update(entries)
    meta_eol = file_eol(meta)
    keep_first_backup(cfg)
    keep_first_backup(meta)
    write_private(cfg, yml, file_eol(cfg))
    write_private(meta, json_text(old_meta), meta_eol)
    say(f"Aider settings for {name} at {base} (context {win} of the server's {ctx}) written to {cfg} and {meta}; run: aider --config {cfg}")
    return 0


def connect(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py connect", description="point a coding agent at a running moe-cache server (started with --agent). "
                                 "Prints the config; --write applies it (Hermes through 'hermes config set'; OpenCode, Codex CLI, Pi, OpenClaw and Qwen Code by merging their config file, old file kept as .bak; "
                                 "aider = its own two files, run it with --config; "
                                 "claude = Claude Code through llama-server's Anthropic-compatible /v1/messages: prints the ANTHROPIC_* lines, --write merges them into this folder's .claude/settings.local.json)")
    ap.add_argument("agent", choices=["hermes", "opencode", "claude", "codex", "pi", "openclaw", "aider", "qwen"])
    ap.add_argument("--url", help="the server (default: the on-demand endpoint's port when `on-demand` has a model, else http://127.0.0.1:8080)")
    ap.add_argument("--api-key", default=None, help="the server's API key (default: LLAMA_API_KEY for a server on this machine, else the stored key when the server asks for one)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--undo", action="store_true", help="put the agent's config back as it was before the first connect (the .bak copy; the file as it is now is kept as .undone); no server needed")
    ap.add_argument("--config", help="file to merge into (opencode: ~/.config/opencode/opencode.json; claude: ./.claude/settings.local.json of this folder; codex: ~/.codex/config.toml; pi: ~/.pi/agent/models.json; openclaw: ~/.openclaw/openclaw.json; qwen: ~/.qwen/settings.json; aider: ~/.aider.moe-cache.conf.yml)")
    ap.add_argument("--check", action="store_true", help="prove the agent's setup works end to end (config, /v1/models, a streamed chat, a tool-call round trip, the window); "
                    "with --write, after writing; exit 0/1")
    ap.add_argument("--json", action="store_true", help="with --check: the result as JSON")
    ap.add_argument("--timeout", type=float, default=180, help="with --check: seconds to wait for each model reply (default 180)")
    a = ap.parse_args(argv)
    if a.json and not a.check:
        ap.error("--json goes with --check")
    a.url_given = a.url
    if a.undo:
        if a.write or a.check:
            ap.error("--undo does not go with --write or --check")
        return connect_undo(a)
    if a.check and not a.write:
        return connect_check(a)
    try:
        rc = connect_apply(a)
    except OSError as e:        # a read-only folder, a file the agent holds open (Windows), a directory where a file should be, no permission
        raise SystemExit(f"moe-cache: cannot {'write' if a.write else 'read'} {e.filename2 or e.filename or 'the agent config'}: {e.strerror or e} - {a.agent} was not connected")
    return connect_check(a) if a.check and not rc else rc


def ambient_key(url):
    """LLAMA_API_KEY from the environment, for a server on this machine only: a --url that names another host never gets a key the user did not type for it."""
    return os.environ.get("LLAMA_API_KEY") if loopback_url(url) else None


def connect_check(a):
    """connect NAME --check: bin/moe_cache_check.py runs the five steps against the server the agent's config points to (or the one connect would use)."""
    from moe_cache_check import check_agent
    return check_agent(a.agent, a.url_given, a.config, a.json, a.timeout, default_agent_url(), lambda url: key_for(url, a.api_key or ambient_key(url) or None, launcher_owns(url)), agent_window)


def connect_apply(a):
    a.url = a.url or default_agent_url()
    a.api_key = key_for(a.url, a.api_key or ambient_key(a.url), launcher_owns(a.url))
    name, ctx = server_info(a.url, a.api_key)
    if not name:
        st = http(a.url.rstrip("/") + "/v1/models", api_key=a.api_key, timeout=3)[0]
        if st in (401, 403):     # the server is there but refuses our key (e.g. HOME points at a folder without the stored key)
            raise SystemExit(f"the server at {a.url} answers {st}: it wants its API key. Give it with --api-key KEY (or LLAMA_API_KEY), "
                             "or run connect as the user who started the server: the stored key is `moe_cache.py key`")
        from moe_cache_pick import picked
        p = picked()
        raise SystemExit(f"no llama-server answers at {a.url} (start one with: moe-cache-server MODEL.gguf --agent"
                         + (f"; or `moe_cache.py server --agent` / `on-demand` for your picked model, {os.path.basename(p[0])}" if p else
                            "; `moe_cache.py pick` recommends a model for this machine") + ")")
    ctx = ctx or 65536
    base = a.url.rstrip("/") + "/v1"
    if a.agent in ("codex", "pi", "openclaw", "aider", "qwen"):
        return connect_extra(a, name, ctx, base, [name] + [n for n in server_models(a.url, a.api_key) if n != name])      # on-demand with several models: all of them
    if a.agent == "hermes":
        if ctx < 64000:
            say(f"moe-cache: warning: the server's context is {ctx}; Hermes Agent refuses less than 64000 (restart with --agent or --ctx 65536)", err=True)
        # AGENTS-036: the key goes into Hermes' .env (0600) and config.yaml names it as ${MOE_CACHE_API_KEY}; `hermes config set` would put the key
        # itself on its command line, readable by other users of the machine in the process list
        sets = [("model.provider", "custom"), ("model.base_url", base), ("model.default", name), ("model.context_length", str(ctx)),
                ("model.api_key", "${" + HERMES_KEY_ENV + "}" if a.api_key else "none")]
        if not a.write:
            say(agent_snippets(None, name, ctx, a.api_key, base=base, only="hermes", hide_key=True))
            say("apply with: " + " && ".join(f"hermes config set {k} {v}" for k, v in sets)
                + "   (or: moe_cache.py connect hermes --write)")
            if a.api_key:
                say(f"and put the key in Hermes' .env as {HERMES_KEY_ENV}={KEY_HINT} (--write does this, mode 0600)")
            return 0
        h = shutil.which("hermes")
        if not h:
            raise SystemExit("hermes is not on PATH (install: https://github.com/NousResearch/hermes-agent)")
        if a.api_key:
            env_file = hermes_env_file(h)
            if not env_file:
                raise SystemExit("`hermes config path` did not answer, so the launcher cannot tell where Hermes keeps its .env for the API key; nothing was changed. "
                                 "Type the key in Hermes instead: Settings -> Model.")
            put_env_line(env_file, HERMES_KEY_ENV, a.api_key)
        bak, prev = hermes_backup(h)
        done_keys = []
        for k, v in sets:
            try:
                r = subprocess.run([h, "config", "set", k, v], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                bad = (r.stderr or r.stdout).strip()[-300:] if r.returncode else ""
            except (OSError, subprocess.SubprocessError) as e:
                r, bad = None, f"{type(e).__name__}: {e}"
            if r is None or r.returncode:
                what = ("; the old config is kept as " + bak + " (copy it back over config.yaml to undo)") if bak else ""
                raise SystemExit(f"hermes config set {k} failed: {bad}\n  Hermes is only partly set up now: done {', '.join(done_keys) or 'nothing'}, not done "
                                 f"{', '.join(x for x, _ in sets if x not in done_keys)}{what}. Run the command again, or finish by hand with the lines it prints without --write.")
            done_keys.append(k)
        hermes_connected(h, base)
        say(f"Hermes Agent now uses {name} at {base} (context {ctx}); run: hermes")
        if prev:
            say(f"  before: {prev}; old config kept as {bak} (copy it back over config.yaml to undo)")
        if a.api_key:
            say(f"  the API key is in Hermes' .env as {HERMES_KEY_ENV} (mode 0600); config.yaml holds only ${{{HERMES_KEY_ENV}}}. It was never on a command line.")
        return 0
    if a.agent == "claude":
        env = claude_env(a.url, name, a.api_key)
        if not a.write:
            say(claude_snippets(env, ctx, hide_key=bool(a.api_key)))
            say("apply to this folder only with: moe_cache.py connect claude --write  (merges into ./.claude/settings.local.json; nothing is written without --write)")
            return 0
        cfg = a.config or os.path.join(os.getcwd(), ".claude", "settings.local.json")
        bak = write_claude_settings(cfg, env)
        say(f"Claude Code in {os.path.dirname(os.path.dirname(cfg)) or '.'} now uses {name} at {a.url} (context {ctx}); config {os.path.realpath(cfg)}"
            + (f" (old file kept as {bak}, outside the project)" if bak else "") + "; run: claude")
        if not a.config and gitignore_settings_local(os.getcwd()):
            say("moe-cache: added .claude/settings.local.json to .gitignore (the file holds the API key)")
        hint = None if a.config else project_folder_hint(os.getcwd())
        if hint:
            say("moe-cache: warning: " + hint, err=True)
        if ctx < 64000:
            say(f"moe-cache: warning: the server's context is {ctx}; Claude Code sends a long system prompt (restart with --agent or --ctx 65536)", err=True)
        return 0
    cfg = a.config or os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config"), "opencode", "opencode.json")
    warn = []
    if not a.config:
        cfg, warn = opencode_config_target(cfg)
    names = [name] + [n for n in server_models(a.url, a.api_key) if n != name]       # on-demand with several models (N9): all of them
    entry = {"npm": "@ai-sdk/openai-compatible", "name": "moe-cache", "options": dict({"baseURL": base}, **({"apiKey": a.api_key} if a.api_key else {}),
                                                                                  **({"headerTimeout": opencode_header_timeout(ctx)} if opencode_header_timeout(ctx) else {})),
             "models": {n: {"name": n, "tool_call": True, "limit": {"context": ctx, "output": min(32768, max(4096, ctx // 4))}} for n in names}}
    if not a.write:
        say(agent_snippets(None, name, ctx, a.api_key, base=base, only="opencode", hide_key=True))
        say(f"apply with: moe_cache.py connect opencode --write  (merges into {cfg})")
        return 0
    data = {"$schema": "https://opencode.ai/config.json"}
    eol = "\n"
    if os.path.isfile(cfg):
        eol = file_eol(cfg)
        try:
            data = json.load(open(cfg, encoding="utf-8"))
        except ValueError:
            raise SystemExit(f"{cfg} is not plain JSON (comments?): add the block printed by 'moe_cache.py connect opencode' by hand")
        if not isinstance(data, dict) or not isinstance(data.get("provider", {}), dict):
            raise SystemExit(f"{cfg} is not an OpenCode config object (a JSON object with a 'provider' object): add the block printed by 'moe_cache.py connect opencode' by hand")
        keep_first_backup(cfg)       # the first backup is the user's own setup: a second connect, with or without our provider in the file, never replaces it
    data.setdefault("provider", {})["moe-cache"] = entry
    data["model"] = f"moe-cache/{name}"
    write_private(cfg, json_text(data), eol)       # the file holds the API key: private to this user, through a temporary file, a symlinked config stays a link
    say(f"OpenCode now uses moe-cache/{name} at {base} (context {ctx}); config {os.path.realpath(cfg)}" + (" (old file: .bak)" if os.path.isfile(bak_path(cfg)) else ""))
    for w in warn:
        say("moe-cache: warning: " + w, err=True)
    return 0


def default_agent_url():
    """The agents' URL: the on-demand endpoint's port when one is set up (moe_cache.py on-demand), else the launcher's default 8080."""
    try:
        port = int(json.load(open(os.path.join(CACHE_DIR, "on-demand.json"), encoding="utf-8")).get("port", 8080))
    except (OSError, ValueError, TypeError, AttributeError):
        port = 8080
    return f"http://127.0.0.1:{port}"


def pick_cmd(argv):
    from moe_cache_pick import pick      # bin/moe_cache_pick.py
    return pick(argv)


def on_demand_cmd(argv):
    from moe_cache_on_demand import on_demand      # bin/moe_cache_on_demand.py
    return on_demand(argv)


def agent_snippets(port, model, ctx, api_key, host="", base=None, only=None, hide_key=False):
    """What an agent needs to use this server: the OpenAI-compatible base URL, the model name and the context, as config blocks for Hermes Agent
    (~/.hermes/config.yaml) and OpenCode (opencode.json)."""
    if base is None:
        h = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
        base = f"http://{h}:{port}/v1"
    if api_key and hide_key:      # the launcher's own printout goes to terminals and logs: point at the key instead of showing it
        api_key = "<your API key: python3 bin/moe_cache.py key>"
    key = api_key or "none"
    out = min(32768, max(4096, ctx // 4))     # room left for the answer
    head = ["moe-cache: agent mode ready. OpenAI-compatible endpoint:",
            f"  base URL {base}   model {model}   context {ctx}   API key {'(set)' if api_key else 'none'}"]
    hermes = [
        "  Hermes Agent, ~/.hermes/config.yaml:",
        "    model:",
        "      provider: custom",
        f"      base_url: {base}",
        f"      default: {model}",
        f"      context_length: {ctx}",
        f"      api_key: {key}"]
    opencode = [
        "  OpenCode, opencode.json:",
        '    {"$schema": "https://opencode.ai/config.json", "provider": {"moe-cache": {"npm": "@ai-sdk/openai-compatible", "name": "moe-cache",',
        f'      "options": {{"baseURL": "{base}"' + (f', "apiKey": "{api_key}"' if api_key else "") +
        (f', "headerTimeout": {opencode_header_timeout(ctx)}' if opencode_header_timeout(ctx) else "") + '},',
        f'      "models": {{"{model}": {{"name": "{model}", "tool_call": true, "limit": {{"context": {ctx}, "output": {out}}}}}}}}}}},',
        f'     "model": "moe-cache/{model}"}}']
    if only == "hermes":
        return "\n".join(hermes)
    if only == "opencode":
        return "\n".join(opencode)
    return "\n".join(head + hermes + opencode + ["  or let the launcher write them: moe_cache.py connect hermes|opencode --write" + (f" --url http://127.0.0.1:{port}" if port and port != 8080 else "")])


# ---- learn ----

def learn_default_ram():
    """LAUNCHER-005 (A3-25): --ram for `learn` when none is given: what this machine can give the server, at most 20 GB (before: always 20, even on a 16 GB machine)."""
    mem = mem_for_server(None)
    return f"{min(20.0, mem / 1e9):.1f}" if mem else "20"


def learn(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py learn", description="build a warm-start profile by running a set of prompts through the model with the plugin; "
                                 "moe-cache-server loads it automatically (same default path)")
    ap.add_argument("model"); ap.add_argument("--name", default=""); ap.add_argument("--ram", type=gb_arg, help="GB the server may use (default: what this machine can give, at most 20)"); ap.add_argument("--n-cpu-moe", default="999")
    ap.add_argument("--tokens", type=int, default=150); ap.add_argument("--prompts", default=os.path.join(HERE, "tools", "learn_prompts.txt"))
    ap.add_argument("--out"); ap.add_argument("--llama-server", default=default_llama_server()); ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--load-method", default="auto", choices=["auto", "preload", "path"])
    a = ap.parse_args(argv)
    if not WINDOWS:
        os.umask(0o077)
    if not os.path.isfile(a.model):
        raise SystemExit("usage: moe-cache-learn MODEL.gguf [options]")
    a.ram = a.ram or learn_default_ram()
    lib = os.environ.get("MOE_CACHE_LIB") or pick_lib(a.llama_server)
    if not os.path.isfile(lib):
        raise SystemExit(f"plugin not built: run ./build.sh first (looked for {lib})")
    if a.name and not NAME_RE.fullmatch(a.name):
        raise SystemExit("--name may only contain letters, digits, _ and -")
    out = a.out or model_profile(a.model, a.name or "")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if port_busy(a.port):
        raise SystemExit(f"port {a.port} is already in use (another server?): stop it or pass --port, so the prompts go to the right server")
    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    before = os.path.getmtime(out) if os.path.isfile(out) else None
    import tempfile
    logf = tempfile.NamedTemporaryFile(prefix="moe-cache-learn.", suffix=".log", delete=False)
    say(f"learning from {len(prompts)} prompts, {a.tokens} tokens each; profile -> {out}")
    load, _ = load_env(a.llama_server, lib, a.load_method)
    env = dict(os.environ, **dict(load + [("MOE_CACHE_SIZE_GIB", "auto"), ("MOE_CACHE_RAM_GIB", ram_gib(a.ram)), ("MOE_CACHE_GGUF", a.model), ("MOE_CACHE_PROFILE", out)]))
    cmd = [a.llama_server, "-m", a.model, "-ngl", "99", "--n-cpu-moe", a.n_cpu_moe, "-c", "4096", "-fa", "on", "--load-mode", "mmap", "--no-warmup",
           "--port", str(a.port), "--seed", "42"]
    cmd += amd_windows_checkpoint_args(pick_gpu(list_devices(a.llama_server)) if WINDOWS else None, a.model, cmd)
    kw = {"preexec_fn": _die_with_parent_linux} if sys.platform.startswith("linux") else {}      # a SIGTERM / closed terminal must not leave llama-server behind
    p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, **kw)
    if WINDOWS:
        try:
            _kill_with_parent_windows(p)
        except Exception:
            pass
    try:
        for _ in range(300):
            if p.poll() is not None:
                raise SystemExit(f"llama-server stopped before it was ready (exit {p.returncode}); log: {logf.name}")
            if healthy(a.port):
                break
            time.sleep(2)
        else:
            raise SystemExit(f"llama-server did not become ready; log: {logf.name}")
        for i, q in enumerate(prompts, 1):
            if p.poll() is not None:
                raise SystemExit(f"llama-server stopped during prompt {i} (exit {p.returncode}); log: {logf.name}")
            st, body = http(f"http://127.0.0.1:{a.port}/completion", {"prompt": q, "n_predict": a.tokens, "temperature": 0, "seed": 42, "cache_prompt": False}, timeout=3000)
            if st != 200:
                raise SystemExit(f"prompt {i}: HTTP {st}: {body[:200]}; log: {logf.name}")
            say(f"  prompt {i}/{len(prompts)} done")
    finally:
        if p.poll() is None:
            if WINDOWS and not win_request_save(p.pid, out):     # terminate() is TerminateProcess there: the plugin's save at exit never runs
                say("moe-cache: warning: the plugin did not confirm a profile save before the stop", err=True)
            p.terminate()
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
    if not os.path.isfile(out) or (before is not None and os.path.getmtime(out) <= before):
        raise SystemExit(f"no new profile was written to {out} (server log: {logf.name})")
    logf.close()
    try:
        os.unlink(logf.name)          # LAUNCHER-005 (A3-25): kept only when something went wrong (the messages above name it)
    except OSError:
        pass
    say(f"done: moe-cache-server will warm-start from {out} ({os.path.getsize(out)} bytes)")
    return 0


# ---- project ----

def server_model_differs(url, api_key, ident):
    """The model file name the server at `url` reports (/props model_path) when it clearly is not the one in the identity `ident` (slot_id: sanitised file name first);
    "" when it matches or the server does not say (then the identity is trusted as before)."""
    st, body = http(url.rstrip("/") + "/props", api_key=api_key, timeout=5)
    try:
        mp = json.loads(body).get("model_path") if st == 200 else None
    except (ValueError, AttributeError):
        mp = None
    if not isinstance(mp, str) or not mp:
        return ""
    base = os.path.basename(mp.replace("\\", "/"))
    return "" if re.sub(r"[^A-Za-z0-9_]", "_", base)[:40] == ident.split("-")[0] else base


def project(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py project", description="save or restore the prompt state (KV cache of slot 0) of a server started with --project NAME. "
                                 "The state is only valid for the same model file and the same -c / cache type settings.")
    ap.add_argument("cmd", choices=["save", "restore", "list"]); ap.add_argument("name", nargs="?")
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"))
    a = ap.parse_args(argv)
    d = os.path.join(CACHE_DIR, "slots")
    if a.cmd == "list":
        files = sorted(glob.glob(os.path.join(glob.escape(d), "*", "*.bin")))
        for f in files:
            say(f"{os.path.getsize(f) / 1e6:10.1f} MB  {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f)))}  {f}")
        if not files:
            say("no saved projects")
        return 0
    if not a.name or not NAME_RE.fullmatch(a.name):
        raise SystemExit("usage: moe-cache-project save|restore NAME [--url URL] [--api-key KEY]")
    sdir = os.path.join(d, a.name)
    if a.cmd == "restore" and os.path.isfile(os.path.join(sdir, a.name + ".bin")) and not project_id_ok(sdir, a.name):
        raise SystemExit(f"restore refused: {a.name}.bin has no record of the model it was saved with, or it differs from the server started with "
                         f"--project {a.name}; a state only fits the same model file. Save it again with `project save`.")
    st, body = http(f"{a.url}/slots/0?action={a.cmd}", {"filename": a.name + ".bin"}, key_for(a.url, a.api_key, launcher_owns(a.url)), timeout=600)
    if a.cmd == "save" and st == 200:
        try:
            ident = open(os.path.join(sdir, "server.id")).read().strip()
            other = server_model_differs(a.url, key_for(a.url, a.api_key, launcher_owns(a.url)), ident)
            if other:
                say(f"moe-cache: note: saved, but the server at {a.url} runs {other}, not the model the server started with --project {a.name} recorded; "
                    "the model identity was not recorded, so a restore will be refused", err=True)
            else:
                write_text_atomic(os.path.join(sdir, a.name + ".bin.id"), ident + "\n")
        except OSError:
            say("moe-cache: note: saved, but the model identity could not be recorded (is the server started with --project " + a.name + "?); a restore will be refused", err=True)
    say(body.strip())
    if st != 200:
        raise SystemExit(f"{a.cmd} failed: HTTP {st}" if st else f"{a.cmd} failed: no server at {a.url}")
    return 0


# ---- check ----

def cpu_name():
    """The CPU's model name for the overview; "" when unknown."""
    try:
        if WINDOWS:
            return os.environ.get("PROCESSOR_IDENTIFIER", "")
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def meminfo_total_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return None


def gpu_driver(name):
    """Driver version of the GPU called `name`: Windows registry; Linux nvidia-smi for NVIDIA cards. "" when unknown."""
    try:
        if WINDOWS:
            return win_driver_version(name)
        if "NVIDIA" in name and shutil.which("nvidia-smi"):
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10).stdout
            return out.strip().splitlines()[0].strip() if out.strip() else ""
    except Exception:
        pass
    return ""


def gpu_summary(listing, ok):
    """moe-cache-check, MG3: when llama.cpp lists more than one device, say which ones it will use and which it leaves out (one line, nothing for one device)."""
    sys.path.insert(0, os.path.join(HERE, "planner"))
    import multigpu
    listed = multigpu.DEVICE_RE.findall(listing or "")
    if len(listed) < 2:
        return
    cards = multigpu.list_cards(listing, win_dedicated_vram() if WINDOWS else None)
    left = multigpu.skipped_igpus(listing, cards)
    ok("gpus", ", ".join(f"{c['name']} ({c['free_mib'] / 1024:.1f} GiB free)" for c in cards) + (" (llama.cpp splits layers across them; `--multi-gpu` plans the split, prototype)" if len(cards) > 1 else "")
       + (f"; left out: {', '.join(left)}" if left else ""))


def check(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py check", description="check that this machine, your llama-server and your model are ready for moe-cache. Changes nothing.")
    ap.add_argument("model", nargs="?"); ap.add_argument("--llama-server", default=default_llama_server()); ap.add_argument("--ram")
    ap.add_argument("--run", action="store_true", help="also start llama-server with the plugin and a small cache for one short request (about a minute): "
                    "proves the cache really computes on this machine")
    a = ap.parse_args(argv)
    run_load = None
    n = {"fail": 0, "warn": 0}
    facts, todo = {}, []             # the overview table at the end (N34, "doctor") and the lines that need a fix
    def ok(k, m): say(f"  [ OK ] {k:<22} {m}")
    def warn(k, m): say(f"  [WARN] {k:<22} {m}"); n["warn"] += 1; todo.append(("WARN", k, m))
    def fail(k, m): say(f"  [FAIL] {k:<22} {m}"); n["fail"] += 1; todo.append(("FAIL", k, m))
    say(f"moe-cache {VERSION} check")
    import platform
    if sys.platform.startswith("linux"):
        ok("system", "Linux " + platform.release())
    elif WINDOWS:
        warn("system", f"Windows {platform.release()}: experimental (identity-tested in CI on a Windows runner; use --run to try the cache here)")
    else:
        fail("system", f"{platform.system()}: only Linux (and experimental Windows) are supported")
    facts["system"] = ("Linux " if sys.platform.startswith("linux") else "Windows " if WINDOWS else platform.system() + " ") + platform.release()
    facts["cpu"] = cpu_name()
    if platform.machine().lower() in ("x86_64", "amd64"):
        ok("cpu", "x86-64")
    else:
        warn("cpu", f"{platform.machine()}: only x86-64 was tested")
    if os.access("/sys/devices/cpu_core/cpus", os.R_OK):
        ok("cpu pinning", f"hybrid Intel CPU (performance cores {open('/sys/devices/cpu_core/cpus').read().strip()}): threads are pinned to the performance cores automatically (about +15 %)")
    elif not WINDOWS:
        ok("cpu pinning", "not a hybrid Intel CPU: no pinning needed")
    else:
        try:
            hm = hybrid_masks(win_cores())
        except Exception:
            hm = None
        if hm:
            ok("cpu pinning", "hybrid CPU: threads are pinned to the performance cores automatically")
        elif getattr(win_cores, "groups", 1) > 1:
            ok("cpu pinning", WIN_GROUPS_NOTE)
        else:
            ok("cpu pinning", "no hybrid cores: no pinning")
    lsp = shutil.which(a.llama_server)
    if not lsp:
        fail("llama-server", f"not found ('{a.llama_server}'); looked: {ls_looked()}; pass --llama-server PATH" +
             (r' (or unpack the official llama.cpp release to C:\llama, or set it once: setx LLAMA_SERVER "D:\path\to\llama-server.exe" and open a new window)' if WINDOWS else " (or run ./install.sh, or put llama-server on the PATH)"))
    else:
        try:
            v = subprocess.run([lsp, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            vtxt = v.stdout + v.stderr
            m = re.search(r"commit [0-9a-f]+", vtxt)
        except Exception:
            m = None; vtxt = ""
        ver = m.group(0) if m else ""
        ok("llama-server", f"{lsp} ({ver or 'version unknown'})")
        _pl = llama_patch_line(lsp)
        if _pl:
            ok("llama.cpp patches", _pl)
            facts["llama.cpp patches"] = _pl
        facts["llama-server"] = f"{ver or 'version unknown'} ({lsp})"
        if tested_commit(ver) or (WINDOWS and re.search(r"620d222|\b11669\b", vtxt)):
            ok("ggml version", f"matches a tested commit ({ver or 'b11669'})")
        else:
            warn("ggml version", "this llama.cpp build is not one we tested" + (f" ({ver})" if ver else "") + ": we tested " +
                 ("b11669, b11966, b12064 and b12207 on Windows" if WINDOWS else "five llama.cpp commits on Linux (7fe450e19, 23b0202a1, ff0d396e3 = b11966, 51fe42c5d = b12064 and 998200491 = b12207)") +
                 ". Other builds usually work; if the plugin does not load, the lines below say so (moe_cache.py update --check shows the newest build)")
        if not WINDOWS:
            if "libggml-base" in ldd(lsp):
                ok("shared ggml", "libggml-base is a shared library")
            else:
                fail("shared ggml", "this llama-server has ggml built in; the plugin cannot be loaded (use a build with BUILD_SHARED_LIBS=ON)")
    lib = os.environ.get("MOE_CACHE_LIB") or (pick_lib(lsp) if lsp else default_lib())
    if not os.path.isfile(lib):
        if not lsp:
            warn("plugin", "not checked: which build fits depends on llama-server, which was not found (see above)")
        elif WINDOWS and shipped_apis() and not ls_api(lsp):
            fail("plugin", f"could not tell which ggml backend API {lsp} uses (no known version in its ggml-base.dll), so no plugin DLL was picked: "
                 f"set MOE_CACHE_LIB to the kit's ggml-moe-cache-api3.dll (API {'/'.join(sorted(shipped_apis()))} ship here)")
        elif WINDOWS:
            fail("plugin", "ggml-moe-cache-api3.dll (or -api2.dll) not found next to the launcher: unzip the whole kit and run from its folder")
        elif shutil.which("g++"):
            warn("plugin", "not built yet: run ./build.sh (g++ found)")
        else:
            fail("plugin", "not built and no g++ found; use the prebuilt release or install g++")
    else:
        ok("plugin", lib)
        facts["moe-cache"] = f"{VERSION}, plugin {os.path.basename(lib)}"
        want, have = (ls_api(lsp), lib_api(lib)) if lsp else (None, None)
        if want and have and want != have:      # the docs send tarball / kit users here to learn which build fits (audit 2026-10-29 V1)
            fail("plugin build", f"made for ggml backend API {have}, this llama-server uses API {want}: {api_advice(want)}")
        elif want and have:
            ok("plugin build", f"ggml backend API {have}, the same as this llama-server")
        if lsp:
            load, method = load_env(lsp, lib)
            probe = ""
            try:
                out = subprocess.run([lsp, "--list-devices"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=dict(os.environ, **dict(load)))
                probe = out.stdout + out.stderr
            except Exception:
                pass
            listed = "MOE_CACHE" in probe
            if listed:
                run_load = load
                ok("plugin loads", f"yes, via {method}" + ("" if a.run else " (loaded and listed; the cache itself first runs with the first request: "
                   "add --run to try it)"))
            else:
                why = [l.strip() for l in probe.splitlines() if "load_backend" in l or "moe-cache:" in l][:3]
                fail("plugin loads", f"llama-server did not list the MOE_CACHE device (method tried: {method})" + ("; llama-server said: " + " | ".join(why) if why else "")
                     + ("; on Windows an empty 'failed to load' reason usually means a missing DLL dependency or a DLL built for another ggml" if WINDOWS else ""))
            devlist = list_devices(lsp)
            g = pick_gpu(devlist)
            if g:
                ok("gpu", f"{g[0]}: {g[1]}" + (f", {g[2]} MiB free" if g[2] else ""))
                gpu_summary(devlist, ok)
                drv = gpu_driver(g[1])
                facts["gpu"] = f"{g[1]} ({g[0]})" + (f", {g[2]} MiB free" if g[2] else "") + (f", driver {drv}" if drv else "")
                if "AMD" in g[1] and WINDOWS and drv and driver_older(drv):
                    warn("gpu driver", f"AMD driver {drv} is older than 32.0.32015.7029: Flash-Next checkpoints crash it (the launcher turns them off); update Adrenalin")
            else:
                facts["gpu"] = "none usable by this llama-server (CPU only)"
                hip = hip_missing_hint(lsp, devlist)
                if hip:
                    warn("gpu (HIP)", hip)
                warn("gpu", "no GPU backend in this llama-server: CPU-only mode (works, slower). For a GPU: " +
                     ("the llama.cpp release zip ...-bin-win-vulkan-x64.zip" if WINDOWS else "./install.sh --vulkan, or a llama.cpp built with -DGGML_VULKAN=ON / -DGGML_CUDA=ON"))
    memav = cg = None
    if WINDOWS:
        t, av = win_memory()
        memav = av / 1e9 if av else None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                memav = int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    lim = cgroup_mem_limit()
    cg = lim / 1e9 if lim else None
    tot = win_memory()[0] / 1e9 if WINDOWS and win_memory()[0] else meminfo_total_gb()
    facts["ram"] = (f"{tot:.0f} GB total, " if tot else "") + (f"{memav:.1f} GB available" if memav else "available unknown") + (f", cgroup limit {cg:.1f} GB" if cg else "")
    ok("memory", (f"--ram {a.ram} GB; " if a.ram else "") + (f"available {memav:.1f} GB" if memav else "available: unknown") + (f", cgroup limit {cg:.1f} GB" if cg else ""))
    ctot, cfree = win_commit()
    if ctot and cfree is not None:      # XA-3: Windows refuses memory when commit (RAM + page file) is used up, even with RAM free; llama-server then stops
        facts["ram"] += f"; commit {cfree / 1e9:.1f} GB free of {ctot / 1e9:.1f} GB"
        (ok if cfree >= 20e9 else warn)("commit", f"{cfree / 1e9:.1f} GB free of {ctot / 1e9:.1f} GB (RAM + page file)" + ("" if cfree >= 20e9 else
            ": tight for a 16 GB GPU plus llama-server's buffers; close big programs or let Windows manage the page file size (docs/TROUBLESHOOTING.md)"))
    try:
        sw = next(int(l.split()[1]) * 1024 / 1e9 for l in open("/proc/meminfo") if l.startswith("SwapTotal"))
        if sw >= 1:
            warn("swap", f"swap is on ({sw:.1f} GB): if memory runs out the system may swap (very slow) instead of killing the server")
    except (OSError, StopIteration):
        pass
    try:
        import gguf  # noqa: F401
        ok("planner", "python gguf package found (the planner also has its own header reader)")
    except ImportError:
        ok("planner", "built-in GGUF header reader (the python gguf package is not needed)")
    d = CACHE_DIR
    while d and not os.path.exists(d) and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    if os.access(d, os.W_OK):
        ok("profile dir", CACHE_DIR + ("" if os.path.isdir(CACHE_DIR) else " (will be created)"))
    else:
        warn("profile dir", "cache directory not writable: profiles will not be saved")
    if a.model:
        if not os.access(a.model, os.R_OK):
            fail("model", f"cannot read '{a.model}'")
        else:
            size = sum(os.path.getsize(p) for p in model_parts(a.model)) / 1e9
            head = open(a.model, "rb").read(4 << 20)
            if head[:4] == b"GGUF":
                ok("model", f"GGUF, {size:.1f} GB")
            else:
                fail("model", "not a GGUF file")
            facts["model"] = f"{os.path.basename(a.model)}, {size:.1f} GB"
            try:
                sys.path.insert(0, os.path.join(HERE, "planner"))
                from gguf_plan import sampling_meta
                march = (sampling_meta(a.model).get("arch") or "").lower()
            except Exception:
                march = ""
            if march in KNOWN_UNSUPPORTED_ARCH:
                fail("architecture", unsupported_arch_message(march, KNOWN_UNSUPPORTED_ARCH[march]))
            if b"expert_count" in head or header_has_experts(a.model):
                ok("MoE model", "has experts (expert_count found)")
            else:
                warn("MoE model", "no expert_count in the header: this may not be a MoE model, moe-cache will not help")
            if not WINDOWS:
                try:
                    os.close(os.open(a.model, os.O_RDONLY | os.O_DIRECT))
                    ok("direct reads", "O_DIRECT works on this file system")
                except OSError:
                    warn("direct reads", "O_DIRECT is not supported here (tmpfs/overlay/network disk?): the plugin reads through the page cache instead, which works but is slower; a normal local disk is faster")
            try:
                du = shutil.disk_usage(os.path.dirname(os.path.abspath(a.model)))
                facts["disk"] = f"{du.free / 1e9:.0f} GB free of {du.total / 1e9:.0f} GB where the model is"
            except OSError:
                pass
            gbs = drive_read_speed(max(model_parts(a.model), key=os.path.getsize))
            if gbs is not None:
                facts["disk"] = facts.get("disk", "the model's drive") + f"; reads {gbs:.1f} GB/s"
            if gbs is None:
                warn("drive speed", "not measured: reads that bypass the file cache are not possible on this file")
            elif gbs < SLOW_DRIVE_GBS:
                warn("drive speed", f"{gbs:.1f} GB/s (random 1 MiB reads, 4 s): a slow drive (SATA, hard disk or USB?); experts that miss the cache "
                     f"are read from it, so start-up, long prompts and models far bigger than RAM get slower. An NVMe drive reads 3 GB/s or more")
            else:
                ok("drive speed", f"{gbs:.1f} GB/s (random 1 MiB reads, 4 s)")
            lim = float(a.ram) if a.ram else (cg or memav)
            if lim and size > lim:
                ok("fit", f"model ({size:.1f} GB) is larger than the memory limit ({lim:.1f} GB): this is where moe-cache helps most")
            elif lim:
                warn("fit", f"model ({size:.1f} GB) fits in {lim:.1f} GB: you get about stock speed, the gain is when memory is short")
    else:
        say("  (give a model path to also check the model: moe-cache-check MODEL.gguf)")
    if a.run:
        if not (a.model and run_load and lsp):
            fail("cache runs", "needs a model, a llama-server and a plugin that loads (see above)")
        else:
            r = run_check(lsp, a.model, run_load)
            (ok if r[0] else fail)("cache runs", r[1])
    say("")
    say("this machine at a glance")
    for k in ("system", "cpu", "ram", "gpu", "disk", "llama-server", "llama.cpp patches", "moe-cache", "model"):
        if k in facts:
            say(f"  {k:<13} {facts[k]}")
    if todo:
        say("what to fix (FAIL first)")
        for lvl, k, m in sorted(todo, key=lambda t: t[0] != "FAIL"):
            say(f"  [{lvl}] {k:<14} {m}")
    say(f"summary: {n['fail']} failed, {n['warn']} warnings")
    return 1 if n["fail"] else 0


def header_has_experts(path):
    """LAUNCHER-005 (A3-27): True when the GGUF header has an <arch>.expert_count key, read as keys and not as a text search in the first 4 MiB (a model with a big
    tokenizer written before the architecture keys pushed it out of the window)."""
    try:
        sys.path.insert(0, os.path.join(HERE, "planner"))
        from gguf_plan import read_meta, shards
        return any(k.endswith(".expert_count") and isinstance(v, int) and v > 0 for k, v in read_meta(shards(path)[0]).items())
    except Exception:
        return False


# LAUNCHER-006: architectures llama.cpp itself cannot load yet. A newer build does not help until upstream has merged support, so the doctor says that instead of
# only "exited before it was ready".
KNOWN_UNSUPPORTED_ARCH = {"clover": "upstream llama.cpp pull request 31306 is still open"}
UNKNOWN_ARCH_RE = re.compile(r"unknown model architecture:?\s*['\"]?([\w.+-]+)['\"]?", re.I)


def unsupported_arch_message(arch, why=None):
    """One plain sentence for an architecture this llama-server cannot load; `why` = what is known about it (None: only that this build refused it)."""
    return (f"llama.cpp does not support the model architecture '{arch}' yet" + (f" ({why})" if why else "") +
            ": no llama-server build loads this model until upstream adds it, and a newer build only helps once that has happened. Nothing is wrong with your setup")


def arch_in_log(lines):
    """The architecture llama-server named in 'unknown model architecture: X' in its output, or None."""
    for l in lines:
        m = UNKNOWN_ARCH_RE.search(l)
        if m:
            return m.group(1)
    return None


def run_check(lsp, model, load, port=8097):
    """Start llama-server with the plugin in cache mode (0.25 GiB cache, experts on the CPU, the rest on the CPU too, small context) for one 8-token
    request (idea I26: the Claw's check said "plugin loads: yes" while cache mode crashed at the first request). (passed, message)."""
    if port_busy(port):
        return False, f"port {port} is in use: stop what runs there and try again"
    env = dict(os.environ, **dict(load), MOE_CACHE_SIZE_GIB="0.25", MOE_CACHE_GGUF=model, MOE_CACHE_STATS="1")
    env.pop("MOE_CACHE_PROFILE", None)
    cmd = [lsp, "-m", model, "-ngl", "0", "--n-cpu-moe", "999", "-c", "512", "--no-warmup", "--port", str(port), "-np", "1"]
    kw = {"env": env, "stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
    if sys.platform.startswith("linux"):
        kw["preexec_fn"] = _die_with_parent_linux
    p = subprocess.Popen(cmd, **kw)
    if WINDOWS:
        try:
            _kill_with_parent_windows(p)
        except Exception:
            pass
    lines = []
    threading.Thread(target=lambda: [lines.append(l.decode(errors="replace").rstrip()) for l in iter(p.stderr.readline, b"")], daemon=True).start()
    try:
        ready = False
        for _ in range(600):
            if p.poll() is not None:
                break
            if healthy(port):
                ready = True
                break
            time.sleep(1)
        st, body = (http(f"http://127.0.0.1:{port}/completion", {"prompt": "The capital of France is", "n_predict": 8, "temperature": 0}, timeout=600)
                    if ready else (0, ""))
        time.sleep(1)
        mc = [l for l in lines if "moe-cache:" in l]
        if p.poll() is not None:
            arch = arch_in_log(lines)
            if arch and not ready:
                return False, unsupported_arch_message(arch, KNOWN_UNSUPPORTED_ARCH.get(arch.lower())) + "; " + exit_reason(p.returncode, ready)
            return False, exit_reason(p.returncode, ready) + (f"; last plugin line: {mc[-1]}" if mc else "")
        if not any("cache mode" in l for l in mc):
            return False, "the server ran but the plugin's cache was not on (no 'moe-cache: cache mode' line): " + (mc[-1] if mc else "no plugin lines at all")
        if st != 200:
            return False, f"the request failed (HTTP {st})"
        try:
            text = json.loads(body).get("content", "")
        except ValueError:
            text = ""
        return True, f"yes: one request in cache mode with a 0.25 GiB cache, answer {text.strip()[:40]!r}"
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()


def default_diffusion_cli():
    """llama-diffusion-cli when --cli is not given: LLAMA_DIFFUSION_CLI, else the one on the PATH, else next to llama-server."""
    if os.environ.get("LLAMA_DIFFUSION_CLI"):
        return os.environ["LLAMA_DIFFUSION_CLI"]
    if shutil.which("llama-diffusion-cli"):
        return "llama-diffusion-cli"
    d = os.path.dirname(shutil.which(default_llama_server()) or "")
    for n in ("llama-diffusion-cli", "llama-diffusion-cli.exe"):
        if d and os.path.isfile(os.path.join(d, n)):
            return os.path.join(d, n)
    return "llama-diffusion-cli"


def diffusion_plan(a, extra, cli, lib, stock, size, avail):
    """Command, environment and notes for one llama-diffusion-cli run (no side effects, so tests can call it). A diffusion step is a prefill of
    the whole canvas with no KV cache (every expert of a layer is used in every step: cache hit rate ~0 %, RESULTS 70); what the plugin still
    gives is merged expert reads and the GPU block that computes a layer's experts after one upload. llama-diffusion-cli ignores --n-cpu-moe
    (examples/diffusion/diffusion-cli.cpp fills only n_gpu_layers), so experts are kept on the CPU side with -ngl 0 (default; --ngl N changes it)."""
    notes = []
    if not a.eps:       # block schedule (LLaDA): llama.cpp asserts both
        if a.canvas % a.block:
            raise SystemExit(f"moe-cache: --canvas {a.canvas} must be a multiple of --block {a.block} (llama.cpp asserts it)")
        if a.steps % (a.canvas // a.block):
            raise SystemExit(f"moe-cache: --steps {a.steps} must be a multiple of the number of blocks ({a.canvas // a.block} = canvas / block; llama.cpp asserts it)")
    if any(x in ("--n-cpu-moe", "-ncmoe", "--cpu-moe", "--override-tensor", "-ot") for x in extra):
        notes.append("moe-cache: WARNING: llama-diffusion-cli ignores --n-cpu-moe / --override-tensor; use --ngl 0 (the default) to keep the experts on the CPU side")
    env = []
    method = "none (stock)"
    if not stock:
        load, method = load_env(cli, lib, a.load_method)
        env = load + [("MOE_CACHE_SIZE_GIB", "auto"), ("MOE_CACHE_GGUF", a.model)]
        if a.ram:
            env.append(("MOE_CACHE_RAM_GIB", ram_gib(a.ram)))
        if a.max_ram:
            env.append(("MOE_CACHE_MAX_RAM_GB", a.max_ram))
    cmd = [cli, "-m", a.model, "-ngl", str(a.ngl), "-ub", str(a.canvas), "-b", str(a.canvas), "-c", str(a.canvas), "-fa", "on", "-lm", "mmap",
           "-s", str(a.seed), "--temp", str(a.temp), "--diffusion-steps", str(a.steps),
           "--diffusion-algorithm", str(a.algorithm)] + (["--diffusion-eps", str(a.eps)] if a.eps else ["--diffusion-block-length", str(a.block)])
    if a.prompt is not None:
        cmd += ["-p", a.prompt]
    cmd += extra
    sched = ("timestep schedule eps " + str(a.eps)) if a.eps else ("blocks of " + str(a.block))
    notes.insert(0, f"moe-cache: diffusion: canvas {a.canvas} tokens (-ub; prompt + answer must fit), {a.steps} steps, {sched}; each step is a prefill of the "
                    f"whole canvas, so every expert of every layer is read each step (the model is {size / 1e9:.1f} GB, memory it may use {avail / 1e9:.1f} GB)"
                    if size and avail else f"moe-cache: diffusion: canvas {a.canvas} tokens, {a.steps} steps")
    notes.append("moe-cache: " + ("running stock llama-diffusion-cli (the model fits in the memory it may use; --always-plugin to use the plugin anyway)" if stock and not a.stock
                                  else "--stock: running without the plugin, as asked" if stock else f"plugin loaded by {method}"))
    return cmd, env, notes, method


def diffusion(argv):
    ap = argparse.ArgumentParser(prog="moe_cache.py diffusion", description="run llama-diffusion-cli (diffusion language models: LLaDA-MoE, RND1) with the expert-cache plugin")
    ap.add_argument("model"); ap.add_argument("-p", "--prompt"); ap.add_argument("--canvas", type=int, default=256, help="tokens per step (prompt + answer), default 256")
    ap.add_argument("--steps", type=int, default=64); ap.add_argument("--block", type=int, default=32, help="block length of the LLaDA schedule (default 32)")
    ap.add_argument("--eps", type=float, default=0.0, help="timestep schedule instead of blocks (RND1, Dream: 0.001)")
    ap.add_argument("--algorithm", type=int, default=4, help="0 origin, 1 entropy (RND1), 2 margin, 3 random, 4 confidence (default)")
    ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ngl", type=int, default=0, help="GPU layers; default 0: experts stay on the CPU side (llama-diffusion-cli ignores --n-cpu-moe)")
    ap.add_argument("--ram", type=gb_arg); ap.add_argument("--max-ram", type=lambda v: "0" if v.strip() in ("0", "0.0") else gb_arg(v))
    ap.add_argument("--cli", default=default_diffusion_cli()); ap.add_argument("--load-method", default="auto", choices=["auto", "preload", "path"])
    ap.add_argument("--stock", action="store_true"); ap.add_argument("--always-plugin", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    if not os.path.isfile(a.model):
        raise SystemExit(f"usage: moe-cache-diffusion MODEL.gguf -p PROMPT [options]  (model file not found: '{a.model}')")
    if a.stock and a.always_plugin:
        raise SystemExit("--stock runs without the plugin: it cannot go with --always-plugin")
    if a.canvas < 8 or a.block < 1 or a.steps < 1:
        raise SystemExit("--canvas, --block and --steps must be positive (canvas at least 8)")
    if a.max_ram:
        os.environ["MOE_CACHE_MAX_RAM_GB"] = a.max_ram
    stock, size, avail = decide(a.model, a.ram, plugin_wanted=a.always_plugin, stock_wanted=a.stock)
    lib = os.environ.get("MOE_CACHE_LIB") or pick_lib(a.cli)
    if not a.dry_run and not (os.path.isfile(a.cli) or shutil.which(a.cli)):
        raise SystemExit(f"moe-cache: llama-diffusion-cli not found: {a.cli}. Pass --cli PATH or set LLAMA_DIFFUSION_CLI "
                         "(official llama.cpp Windows zips do not contain it: build the diffusion example, or use a Linux package that does)")
    if not stock and not a.dry_run:
        want, have = ls_api(a.cli), lib_api(lib)
        if want and have and want != have:
            say(f"moe-cache: WARNING: {lib} was built for ggml backend API {have} but {a.cli} uses API {want}: the plugin will refuse to load; " + api_advice(want), err=True)
        if not os.path.isfile(lib):
            raise SystemExit(f"plugin not built: run ./build.sh first (looked for {lib})")
    cmd, env, notes, method = diffusion_plan(a, extra, a.cli, lib, stock, size, avail)
    for n in notes:
        say(n)
    if a.dry_run:
        shown = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        say(("".join(f'$env:{k}="{v}"; ' for k, v in env) + "& " + shown) if WINDOWS else (("env " + " ".join(f"{k}={v}" for k, v in env) + " ") if env else "") + shown)
        return 0
    return subprocess.call(cmd, env=dict(os.environ, **dict(env)))


def pack_cmd(argv):
    """moe_cache.py pack MODEL.gguf: write MODEL.moepack (one block per expert; the plugin reads a missing expert with one large unbuffered read)."""
    sys.path.insert(0, os.path.join(HERE, "bin"))
    from moe_cache_pack import pack_main
    return pack_main(argv)


REPO_API = "https://api.github.com/repos/IbrahimSiddiqui007/moe-cache/releases?per_page=10"     # list: /latest skips pre-releases and drafts
LLAMA_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"     # newest bNNNNN build tag (the /latest one is an old v0.x tag)


def version_tuple(v):
    """'v0.3.2' / '0.3.2-beta' -> (0, 3, 2); unknown parts -> 0."""
    nums = re.findall(r"\d+", str(v).split("-")[0])
    return tuple(int(x) for x in nums[:4]) or (0,)


def update_advice(mine, latest_tag, ls_b, llama_tag):
    """Lines for `update --check` (pure, tested): is a newer moe-cache out, and how far behind is this llama-server."""
    out = []
    if latest_tag:
        if version_tuple(latest_tag) > version_tuple(mine):
            out.append(f"moe-cache {latest_tag.lstrip('v')} is out (you have {mine}): https://github.com/IbrahimSiddiqui007/moe-cache/releases/latest")
        elif version_tuple(latest_tag) == version_tuple(mine):
            out.append(f"moe-cache {mine} is the newest release")
        else:
            out.append(f"moe-cache {mine} is newer than the newest release ({latest_tag.lstrip('v')}): a development copy")
    else:
        out.append("moe-cache: could not read the newest release (offline?)")
    m = re.fullmatch(r"b(\d+)", llama_tag or "")
    if ls_b and m:
        behind = int(m.group(1)) - ls_b
        out.append(f"llama.cpp: your llama-server is build {ls_b}, the newest release is {llama_tag}" +
                   (f" ({behind} builds newer; newer builds are supported: the plugin checks ggml's layout at start)" if behind > 0 else " (up to date)"))
    elif ls_b:
        out.append(f"llama.cpp: your llama-server is build {ls_b}; the newest release could not be read")
    return out


def update_cmd(argv):
    """moe_cache.py update --check: tells whether a newer moe-cache or llama.cpp release exists. Downloads and installs nothing; asks GitHub only now. --apply: bin/moe_cache_update.py."""
    ap = argparse.ArgumentParser(prog="moe_cache.py update", description="check for a newer moe-cache and llama.cpp release (nothing is downloaded or installed)")
    ap.add_argument("--check", action="store_true", help="the only mode: ask GitHub's release API once and print what is newer")
    ap.add_argument("--llama-server", default=default_llama_server())
    ap.add_argument("--apply", action="store_true", help="install the newest release with the release's own installer (only for a copy the installer put there; see bin/moe_cache_update.py)")
    ap.add_argument("--tag", help="with --apply: a release tag (vX.Y.Z) instead of the newest stable one")
    ap.add_argument("--yes", action="store_true", help="with --apply: do not ask")
    ap.add_argument("--allow-downgrade", action="store_true", help="with --apply: install a release older than the running one (refused without this)")
    ap.add_argument("--yes-unverified", action="store_true", help="with --apply: go on although the build attestation cannot be checked (gh missing / not logged in); sha256 is always checked")
    ap.add_argument("--dry-run", action="store_true", help="with --apply: say what would happen, download nothing")
    a = ap.parse_args(argv)
    if a.apply:
        from moe_cache_update import apply as apply_update
        return apply_update(VERSION, HERE, tag=a.tag, yes=a.yes, dry=a.dry_run, allow_downgrade=a.allow_downgrade, unverified=a.yes_unverified)
    def tag(url):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": f"moe-cache/{VERSION}"})
            with urllib.request.urlopen(req, timeout=10) as r:
                j = json.loads(r.read().decode())
            if isinstance(j, list):         # newest published release, pre-releases included; llama.cpp: its bNNNNN build tags
                j = next((x for x in j if not x.get("draft") and ("llama.cpp" not in url or re.fullmatch(r"b\d+", x.get("tag_name", "")))), {})
            return j.get("tag_name")
        except Exception:
            return None
    lsp = shutil.which(a.llama_server)
    for line in update_advice(VERSION, tag(REPO_API), ls_build(lsp) if lsp else None, tag(LLAMA_API)):
        say(line)
    say("nothing was downloaded or installed")
    return 0


def key_cmd(argv):
    """moe_cache.py key: print the stable API key of agent / project servers (made on first use); the file is KEY_FILE."""
    ap = argparse.ArgumentParser(prog="moe_cache.py key", description="print the API key that agent mode and --project servers use")
    ap.add_argument("--path", action="store_true", help="print where the key is stored instead of the key")
    a = ap.parse_args(argv)
    print(KEY_FILE if a.path else stored_key(create=True))
    return 0


def main():
    cmds = {"server": server, "pick": pick_cmd, "check": check, "doctor": check, "learn": learn, "project": project, "connect": connect, "key": key_cmd, "on-demand": on_demand_cmd, "diffusion": diffusion, "pack": pack_cmd, "update": update_cmd}
    if sys.argv[1:2] == ["ui"] or (sys.argv[1:2] == ["server"] and "--ui" in sys.argv[2:]):     # GUI1: the control page
        from moe_cache_ui import ui
        return ui([x for x in sys.argv[2:] if x != "--ui"]) or 0
    if sys.argv[1:2] == ["server"] and "--top" in sys.argv[2:]:      # GUI-047: start a model and watch it in the terminal monitor
        from moe_cache_top import top_start
        return top_start(sys.argv[2:]) or 0
    if sys.argv[1:2] == ["top"]:        # GUI-045: the Monitor view as one live terminal screen (bin/moe_cache_top.py)
        from moe_cache_top import top
        return top(sys.argv[2:]) or 0
    if sys.argv[1:2] == ["models"]:     # the model catalogue for this PC (bin/moe_cache_models.py)
        from moe_cache_models import models
        return models(sys.argv[2:]) or 0
    if sys.argv[1:2] == ["launch"]:     # the guided start (bin/moe_cache_launch.py)
        from moe_cache_launch import launch
        return launch(sys.argv[2:]) or 0
    if sys.argv[1:2] == ["bench-report"]:     # R2-1: stock vs moe-cache on this machine, a report to paste into a GitHub issue
        from moe_cache_bench import bench_report
        return bench_report(sys.argv[2:]) or 0
    if sys.argv[1:2] == ["bench-record"]:     # the benchmark record's tools: collect, monitor, assemble, validate, card (docs/BENCHMARK_TEMPLATE.md)
        from moe_cache_record import main as record_main
        return record_main(sys.argv[2:]) or 0
    if sys.argv[1:2] in (["--version"], ["-V"], ["version"]):
        say(f"moe-cache {VERSION}")
        return 0
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        say(__doc__.strip())
        return 0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 1
    return cmds[sys.argv[1]](sys.argv[2:]) or 0


def exit_status(rc):
    """LAUNCHER-005 (A3-21): a child killed by a signal comes back as -N from subprocess; sys.exit(-9) would be 247, scripts and systemd expect 128+N (137)."""
    return 128 - rc if isinstance(rc, int) and rc < 0 and not WINDOWS else rc


if __name__ == "__main__":
    sys.exit(exit_status(main()))
