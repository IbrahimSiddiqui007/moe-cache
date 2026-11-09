#!/usr/bin/env python3
"""GUI-1052: the folders models are looked for in without any setting. One implementation (bin/moe_cache_pick.py default_folders) for the page and the picker:
~/models and, on Windows, <drive>:\\models on every FIXED drive (the owner moved the models from C:\\models to D:\\models; the page listed only C:, the picker only C: and D:).
A fake drive list stands in for Windows; nothing here touches a real drive.
   python3 tests/model_folders_test.py"""
import json, os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-folders-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_pick as pk   # noqa: E402
import moe_cache_ui as ui     # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

home = os.path.join(os.path.expanduser("~"), "models")
check(pk.default_folders(windows=False, drives=["C", "D"]) == [home], "not Windows: only ~/models, whatever the drive list says")
got = pk.default_folders(windows=True, drives=["C", "D", "E", "G"])
check(got == [home, "C:\\models", "D:\\models", "E:\\models", "G:\\models"], f"Windows with four fixed drives: ~/models + one <drive>:\\models each, in drive order ({got})")
check("C:\\models" not in pk.default_folders(windows=True, drives=["D"]), "the owner's case (C:\\models deleted, models on D:): D:\\models is listed, C: is only listed when it is a fixed drive")
check(pk.default_folders(windows=True, drives=["E"]) == [home, "E:\\models"], "a PC whose only fixed drive is E: gets E:\\models (not a hard-coded C and D)")

print("the page and the picker use the same function")
check(ui.default_folders is pk.default_folders, "ui.default_folders IS picker.default_folders (one implementation)")
saved = (pk.WINDOWS, pk.fixed_drives, pk.SETTINGS_FILE)
try:
    pk.WINDOWS = True
    pk.fixed_drives = lambda: ["C", "D", "F"]
    settings = os.path.join(TMP, "ui.json")
    json.dump({"folders": ["Z:\\my models", "D:\\models"]}, open(settings, "w"))
    pk.SETTINGS_FILE = settings
    st = ui.Store.__new__(ui.Store); st.saved = {"folders": ["Z:\\my models", "Q:\\other"]}
    page = st.folders()
    check(all(x in page for x in (home, "C:\\models", "D:\\models", "F:\\models")), f"page: ~/models and every fixed drive's models folder ({page})")
    check("Z:\\my models" in page and "Q:\\other" in page, "page: folders the user added are kept")
    check(len(page) == len(set(page)), "page: no folder twice")
    pick = pk.model_folders(["X:\\extra"])
    check(pick[0] == "X:\\extra" and all(x in pick for x in (home, "C:\\models", "D:\\models", "F:\\models", "Z:\\my models")), f"picker: the same defaults plus saved and extra folders ({pick})")
    check(len(pick) == len(set(pick)), "picker: D:\\models is in the defaults and in the saved folders: listed once")
finally:
    pk.WINDOWS, pk.fixed_drives, pk.SETTINGS_FILE = saved

print("fixed_drives")
check(pk.fixed_drives() == [] if not pk.WINDOWS else all(len(x) == 1 for x in pk.fixed_drives()), "fixed_drives(): [] off Windows; letters on Windows")
if pk.WINDOWS:      # the real GetLogicalDrives / GetDriveTypeW query (Windows CI): the system drive is a fixed drive, and it comes from the query, not the fallback
    real = pk.fixed_drives()
    sysdrive = (os.environ.get("SystemDrive") or "C:")[0].upper()
    check(sysdrive in real and real == sorted(real), f"Windows: the system drive {sysdrive}: is in the fixed drives, sorted ({real})")
    check(f"{sysdrive}:\\models" in pk.default_folders(), "Windows: the real default list holds the system drive's models folder")
sv = pk.WINDOWS
try:
    pk.WINDOWS = True
    import ctypes
    had = hasattr(ctypes, "windll")
    check(pk.fixed_drives() == ["C", "D"] if not had else True, "Windows cannot be asked (no ctypes.windll): C and D, the old behaviour")
finally:
    pk.WINDOWS = sv

print("FAIL" if fails else "all ok")
sys.exit(1 if fails else 0)
