#!/usr/bin/env python3
"""find_llama_server: a llama-server on the PATH whose ggml API no shipped plugin supports gives way to a default-folder one that has a supported API (the kit's
bundled llama.cpp), with a note; explicit settings still win; with no supported one anywhere the PATH one stays. v0.3.6-beta2: the -ggml0.26 tarball (API 3 plugin,
bundled b12064) ran `check` against /usr/bin/llama-server (API 2) and advised another download. Fake llama-servers, no real llama.cpp.
   python3 tests/llama_search_test.py"""
import os, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache as mc  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

TMP = tempfile.mkdtemp(prefix="moe-lssearch-")
def touch(*parts):
    p = os.path.join(TMP, *parts); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").close(); return p

kit = os.path.join(TMP, "kit")
mc.HERE = kit
EXE = "llama-server"
bundled = touch("kit", "llama", EXE)                  # the tarball's bundled llama.cpp (API 3)
distro = touch("usr", "bin", EXE)                     # the distro's (API 2)
home_build = touch("home", "llama.cpp", "build", "bin", EXE)   # a user's own build
api_of = {bundled: "3", distro: "2", home_build: "2"}
builds = {bundled: 12064, distro: 0, home_build: 9000}

def setup(path_ls=None, shipped=("3",), folders=(os.path.join(TMP, "home", "llama.cpp", "build", "bin"), kit, os.path.join(kit, "llama")), windows=False,
          registry_path=None, env=None, saved=None):
    mc.WINDOWS = windows
    mc.shipped_apis = lambda: set(shipped)
    mc.ls_api = lambda p: api_of.get(os.path.abspath(p) if p else p)
    mc.ls_build = lambda p: builds.get(os.path.abspath(p))
    mc.llama_search_dirs = lambda: list(folders)
    mc.llama_exe = lambda: EXE
    mc.win_registry_env = lambda name: ([("the user", registry_path)] if (name == "PATH" and registry_path) else [])
    mc.saved_llama_server = lambda: saved
    mc.shutil.which = lambda n, *a, **k: path_ls if n == "llama-server" else (n if os.path.isfile(n) else None)
    os.environ.pop("LLAMA_SERVER", None)
    if env:
        os.environ["LLAMA_SERVER"] = env

r = None
# 1. the beta2 case: PATH has API 2, the kit bundles API 3
setup(path_ls=distro)
r = mc.find_llama_server()
check(r["path"] == bundled and r["source"] == "a default folder" and r["api"] == "3", f"PATH API 2, kit API 3: the kit's wins ({r['path'][-28:]}, {r['source']}, API {r['api']})")
check(r["note"] and "PATH llama-server (ggml API 2) skipped: no shipped plugin for it" in r["note"] and "the kit's bundled llama-server (API 3)" in r["note"], f"the note says why ({r['note']})")
check(r["note"] in r["looked"], "the note is in the places that were looked at (check prints them)")
check(r["build"] == 12064 and home_build in r["others"], "the other folder candidate is listed as not used")
mc._LS.clear(); mc._LS.update(r, done=True)
line = mc.ls_found_line(r["path"])
check(line and "skipped" in line and line.count("(") >= 1, f"the start-up line carries the note")

# 2. a supported PATH llama-server stays first
setup(path_ls=bundled)
r = mc.find_llama_server()
check(r["path"] == bundled and r["source"] == "PATH" and not r["note"], "PATH API 3 (supported): the PATH one (as an absolute path), no note")

# 3. no supported llama-server anywhere: today's behaviour
setup(path_ls=distro, shipped=("3",), folders=(os.path.join(TMP, "home", "llama.cpp", "build", "bin"),))
r = mc.find_llama_server()
check(r["path"] == distro and r["source"] == "PATH" and not r["note"], "PATH API 2 and only API 2 in the folders: the PATH one stays (today's behaviour and advice)")

# 4. no plugin shipped at all (a source checkout): nothing to compare, the PATH one stays
setup(path_ls=distro, shipped=())
r = mc.find_llama_server()
check(r["path"] == distro and r["source"] == "PATH" and not r["note"], "no shipped plugin build: the PATH one stays")

# 5. the API of the PATH one is unknown: it stays
api_of_save = dict(api_of); api_of.pop(distro)
setup(path_ls=distro)
r = mc.find_llama_server()
check(r["path"] == bundled and "unreadable" in (r["note"] or ""), f"PATH llama-server of unreadable API loses to the kit's with a known API ({r['path'][-28:]}, {r['note']})")
setup(path_ls=distro, folders=(os.path.join(TMP, "home", "llama.cpp", "build", "bin"),))
r = mc.find_llama_server()
check(r["path"] == distro and not r["note"], "PATH llama-server of unreadable API and no supported one in the folders: it stays, as an absolute path")
setup(path_ls=distro, shipped=())
r = mc.find_llama_server()
check(r["path"] == distro and not r["note"], "PATH llama-server of unreadable API, no shipped plugin to compare: it stays")
api_of.update(api_of_save)

# 6. explicit settings win, even when their API has no plugin
setup(path_ls=bundled, env=distro)
r = mc.find_llama_server()
check(r["path"] == distro and r["source"] == "LLAMA_SERVER", "LLAMA_SERVER wins over everything")
setup(path_ls=distro, saved=distro)
r = mc.find_llama_server()
check(r["path"] == distro and "saved path" in r["source"], "the control page's saved path wins")

# 7. the kit's own copy is older than the user's other API-3 build: the newest supported one of the folders
api_of[home_build] = "3"; builds[home_build] = 13000
setup(path_ls=distro)
r = mc.find_llama_server()
check(r["path"] == home_build and r["build"] == 13000 and "using " + home_build in r["note"], "two supported folder candidates: the newest build, named in the note (not 'the kit's')")
api_of[home_build] = "2"; builds[home_build] = 9000

# 8. Windows: llama-server in the PATH (process) and in the registry's PATH
setup(path_ls=distro, windows=True)
r = mc.find_llama_server()
check(r["path"] == bundled, "Windows, process PATH API 2: the kit's API 3 wins")
regdir = os.path.dirname(distro)
setup(path_ls=None, windows=True, registry_path=regdir)
mc.resolve_llama_server = lambda v: (os.path.join(v, EXE) if os.path.isfile(os.path.join(v, EXE)) else None)
r = mc.find_llama_server()
check(r["path"] == bundled and r["note"] and "PATH in the user llama-server (ggml API 2) skipped" in r["note"], f"Windows, registry PATH API 2: the kit's wins, note names the registry ({(r['note'] or '')[:70]})")
setup(path_ls=None, windows=True, registry_path=os.path.dirname(bundled))
r = mc.find_llama_server()
check(r["path"] == bundled and r["source"] == "PATH in the user" and not r["note"], "Windows, registry PATH API 3: the registry one, no note")
sys.exit(1 if fails else 0)
