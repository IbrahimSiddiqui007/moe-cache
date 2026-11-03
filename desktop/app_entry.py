#!/usr/bin/env python3
"""Entry point the desktop shell runs instead of `moe_cache.py ui` (design: docs/DESKTOP_APP_DESIGN.md, section 8 items 2-4).

It is the control page, unchanged (bin/moe_cache_ui.py), plus three things the shell needs and the page does not do by itself:

1. one stable line for the shell to read, printed as soon as the page listens:
   {"moe_cache_ui": {"url": "http://127.0.0.1:PORT/?t=TOKEN", "port": PORT, "pid": N}}
   (the human line "control page at ..." still follows, for the terminal and for older shells);
2. a parent-gone watch: the shell keeps this process's stdin open; end of input (the shell quit, crashed or was killed, on any system) stops the
   model and ends the page, so no llama-server stays behind;
3. SIGTERM / SIGBREAK end the page the same clean way as Ctrl-C (the page's own exit code stops the model).

Staged next to bin/ (resources/moe-cache/app_entry.py); in a checkout it sits in desktop/ and the kit is its parent folder.
Usage: python -u app_entry.py [--ui-port 0] [--llama-server PATH]   (the shell passes --ui-port 0)
"""
import json, os, signal, sys, threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isfile(os.path.join(HERE, "bin", "moe_cache_ui.py")) else os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "bin"))

import moe_cache_ui as ui  # noqa: E402

KILL_AFTER = 90        # seconds: the clean stop waits for llama-server; past this the process ends anyway


def main(argv):
    argv = [a for a in argv if a != "--no-browser"] + ["--no-browser"]
    started = {}
    real_make = ui.make_server

    def make_server(*a, **k):
        srv = real_make(*a, **k)
        started["srv"] = srv
        port = srv.server_address[1]
        url = f"http://127.0.0.1:{port}/?t={srv.RequestHandlerClass.token}"
        print(json.dumps({"moe_cache_ui": {"url": url, "port": port, "pid": os.getpid()}}), flush=True)
        return srv

    ui.make_server = make_server

    def stop(*_):
        """End the page: serve_forever returns and ui()'s finally block stops the model. Threaded: shutdown() waits for the serving thread."""
        srv = started.get("srv")
        if srv is not None:
            threading.Thread(target=srv.shutdown, daemon=True).start()
        t = threading.Timer(KILL_AFTER, os._exit, args=(1,))
        t.daemon = True
        t.start()

    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)

    if os.environ.get("MOE_CACHE_APP") == "1":
        def watch_parent():
            try:
                while sys.stdin.buffer.read(4096):
                    pass
            except (OSError, ValueError):
                pass
            stop()
        threading.Thread(target=watch_parent, daemon=True).start()
    rc = ui.ui(argv) or 0
    if os.environ.get("MOE_CACHE_APP") == "1":
        # the watch thread may still sit in a stdin read: a normal interpreter exit would abort on its lock ("could not acquire lock for <stdin>")
        for f in (sys.stdout, sys.stderr):
            try:
                f.flush()
            except (OSError, ValueError):
                pass
        os._exit(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
