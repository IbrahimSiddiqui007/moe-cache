"""win_tool(name): the full path of a Windows helper program (powershell, tasklist, taskkill, ...).

A bare name makes Windows look in the current folder first, so an .exe dropped into Downloads would run instead of the real tool (AUD-034).
On Windows this returns the file under %SystemRoot%\\System32 (powershell: WindowsPowerShell\\v1.0); anywhere else the bare name, unchanged.
"""
import os, sys


def win_tool(name, windows=None):
    if not (sys.platform == "win32" if windows is None else windows):
        return name
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    sysdir = os.path.join(root, "System32")
    if name.lower() in ("powershell", "powershell.exe"):
        p = os.path.join(sysdir, "WindowsPowerShell", "v1.0", "powershell.exe")
    else:
        p = os.path.join(sysdir, name if name.lower().endswith(".exe") else name + ".exe")
    return p if os.path.isfile(p) else name
