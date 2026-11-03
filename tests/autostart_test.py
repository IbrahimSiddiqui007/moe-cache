#!/usr/bin/env python3
"""`moe_cache.py on-demand --install-autostart / --remove-autostart` (AUD-092) with a FAKE systemctl and a FAKE powershell: no unit is really enabled, no
shortcut is really made. Checks the unit file (ExecStart quoting, Restart, WantedBy), the systemctl calls in order and their failure handling, the
Windows Startup-shortcut call (environment, no console, no second start when the listener runs), and the removal paths. Linux and Windows code paths
both run here (the Windows one with the module's WINDOWS flag set and powershell replaced)."""
import os, stat, sys, tempfile, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="autostart-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
os.environ["XDG_CONFIG_HOME"] = os.path.join(TMP, "config")
os.environ["APPDATA"] = os.path.join(TMP, "appdata")
sys.path.insert(0, os.path.join(ROOT, "bin"))
import moe_cache as mc            # noqa: E402
import moe_cache_on_demand as od  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

LOG = os.path.join(TMP, "systemctl.log")
BIN = os.path.join(TMP, "bin")
os.makedirs(BIN)
def fake_systemctl(fail_on=None):
    p = os.path.join(BIN, "systemctl")
    with open(p, "w") as f:
        f.write('#!/bin/sh\necho "$@" >> "%s"\n' % LOG)
        if fail_on:
            f.write('case "$*" in *%s*) echo "boom" >&2; exit 5;; esac\n' % fail_on)
        f.write("exit 0\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
def calls():
    return open(LOG).read().splitlines() if os.path.isfile(LOG) else []
def run(fn, *a):
    out = []
    old = od.say
    od.say = lambda m="": out.append(str(m))
    try:
        return fn(*a), "\n".join(out), None
    except SystemExit as e:
        return None, "\n".join(out), str(e.code)
    finally:
        od.say = old

stops = []
od.stop_running = lambda: stops.append(1)         # the real one signals a listener process
od.WINDOWS = False
os.environ["PATH"] = BIN + os.pathsep + os.environ.get("PATH", "")

# ---- Linux: systemd --user unit ----
fake_systemctl()
rc, out, err = run(od.install_autostart)
unit = od.unit_path()
check(rc == 0 and err is None and os.path.isfile(unit), f"install: unit written to {unit}")
text = open(unit).read() if os.path.isfile(unit) else ""
check("ExecStart=" in text and "on-demand" in text and "moe_cache.py" in text and "Restart=on-failure" in text and "WantedBy=default.target" in text,
      "unit: ExecStart runs `moe_cache.py on-demand`, Restart=on-failure, WantedBy=default.target")
check(f'"{sys.executable}"' in text, "unit: the Python path is quoted (paths with spaces)")
check("StartLimitBurst=5" in text, "unit: a crash loop is limited")
check(calls() == ["--user daemon-reload", f"--user enable --now {od.UNIT}"], f"systemctl: daemon-reload, then enable --now (saw {calls()})")
check("enabled and started" in out, "install prints that it was enabled and started")

open(LOG, "w").close()
fake_systemctl(fail_on="enable")
rc, out, err = run(od.install_autostart)
check(rc is None and err is not None and "enable" in err and "boom" in err, f"systemctl enable failing stops with its message (got {err!r})")

# dry run writes nothing
os.remove(unit)
rc, out, err = run(od.install_autostart, True)
check(rc == 0 and not os.path.exists(unit), "dry run: no unit written")

# XDG_CACHE_HOME is carried into the unit (so the service finds the same key / choice)
check(f"XDG_CACHE_HOME={os.environ['XDG_CACHE_HOME']}" in od.unit_text(), "unit: XDG_CACHE_HOME is passed on")

# ---- Linux: removal ----
fake_systemctl(); open(LOG, "w").close()
run(od.install_autostart); open(LOG, "w").close()
rc, out, err = run(od.remove_autostart)
check(rc == 0 and not os.path.exists(unit), "remove: unit deleted")
check(calls() == [f"--user disable --now {od.UNIT}", "--user daemon-reload"], f"remove: disable --now, then daemon-reload (saw {calls()})")
check(len(stops) == 1, "remove: the running listener is stopped")
rc, out, err = run(od.remove_autostart)
check(rc == 0 and "there was no systemd unit" in out, "remove twice: says there was no unit")

# ---- Windows: Startup shortcut through powershell ----
od.WINDOWS = True
link = od.startup_link()
check(link.startswith(os.environ["APPDATA"]) and link.endswith(od.LINK), "windows: the shortcut lives in the user's Startup folder (no admin rights)")
ps_calls, popens = [], []
def fake_run(cmd, **k):
    ps_calls.append((cmd, k.get("env")))
    if cmd[0] == "powershell" and not FAIL_PS[0]:
        open(k["env"]["MC_LINK"], "w").close()           # what CreateShortcut(...).Save() does
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return types.SimpleNamespace(returncode=1, stdout="", stderr="access denied")
FAIL_PS = [False]
class FakePopen:
    def __init__(self, cmd, **k): popens.append((cmd, k))
real_run, real_popen = od.subprocess.run, od.subprocess.Popen
od.subprocess.run, od.subprocess.Popen = fake_run, FakePopen
od.running_pid = lambda: None
try:
    rc, out, err = run(od.install_autostart)
    check(rc == 0 and os.path.isfile(link), "windows install: shortcut created")
    cmd, env = ps_calls[0]
    check(cmd[:3] == ["powershell", "-NoProfile", "-NonInteractive"] and env["MC_SCRIPT"].endswith("moe_cache.py") and env["MC_LINK"] == link,
          "windows install: powershell gets the link, python and script through the environment (no quoting through the command line)")
    check(len(popens) == 1 and popens[0][0][-1] == "on-demand" and popens[0][1].get("creationflags") == (0x8 | 0x200 | 0x08000000),
          "windows install: starts the listener now, detached with no console window")
    check("started it now too" in out, "windows install: says it also started it")

    od.running_pid = lambda: 4242
    popens.clear()
    rc, out, err = run(od.install_autostart)
    check(rc == 0 and not popens, "windows install while the listener runs: no second listener")

    FAIL_PS[0] = True
    os.remove(link)
    rc, out, err = run(od.install_autostart)
    check(rc is None and err and "could not create the Startup shortcut" in err and "access denied" in err, "windows install: powershell failing stops with its message")
    FAIL_PS[0] = False

    open(link, "w").close(); stops.clear()
    rc, out, err = run(od.remove_autostart)
    check(rc == 0 and not os.path.exists(link) and len(stops) == 1, "windows remove: shortcut deleted, listener stopped")
    rc, out, err = run(od.remove_autostart)
    check(rc == 0 and "there was no Startup shortcut" in out, "windows remove twice: says there was no shortcut")

    rc, out, err = run(od.install_autostart, True)
    check(rc == 0 and not os.path.exists(link), "windows dry run: nothing created")
finally:
    od.subprocess.run, od.subprocess.Popen = real_run, real_popen
    od.WINDOWS = mc.WINDOWS

print("autostart_test: " + ("PASS" if not fails else "FAILED: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
