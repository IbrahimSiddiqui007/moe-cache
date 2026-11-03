"""moe_cache.py update --apply: install the newer moe-cache release with the release's own installer (RELEASE-616).

It does not unpack anything itself. It downloads the release's install.sh (Linux) or install.ps1 (Windows), checks that file's build attestation
with `gh` when gh is usable, and runs it for the same folders the running copy was installed into. The installer then does the rest: it checks the
tarball / zip against the release's SHA256SUMS and its build attestation, unpacks beside the old version, swaps by rename and rolls back on any
failure. On Linux the previous version stays in versions/ (MOE_CACHE_KEEP_PREVIOUS=1) and `current` can be pointed back at it; on Windows the
installer deletes the old kit once the new one is complete (no previous version kept yet).
A copy that was not put there by the installer (a git checkout, an unpacked tarball or kit) is not touched: the command says so and stops.
Standard library only."""
import json, os, re, shutil, subprocess, sys, tempfile, urllib.request

REPO = "IbrahimSiddiqui007/moe-cache"
LIST_API = f"https://api.github.com/repos/{REPO}/releases?per_page=10"
WINDOWS = sys.platform == "win32"


def say(msg):
    print(msg, flush=True)


def newest_stable_tag(fetch_json):
    """Tag of the newest published release that is not a draft or pre-release; None when there is none or GitHub cannot be read."""
    try:
        for r in fetch_json(LIST_API):
            if not r.get("draft") and not r.get("prerelease") and re.fullmatch(r"v\d+(\.\d+)*", r.get("tag_name", "")):
                return r["tag_name"]
    except Exception:
        return None
    return None


def http_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "moe-cache-update"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def installed_layout(here, windows=WINDOWS, env=None):
    """How the running copy was installed by install.sh / install.ps1, as a dict, or None when it was not.
    Linux: here = <prefix>/versions/moe-cache-v*, and <prefix>/.installed names the prefix and the command. Windows: here = <root>\\kit and
    <root>\\install.json exists."""
    env = os.environ if env is None else env
    here = os.path.realpath(here)
    if windows:
        root = os.path.dirname(here)
        rec = os.path.join(root, "install.json")
        if os.path.basename(here).lower() == "kit" and os.path.isfile(rec):
            return {"kind": "windows", "root": root, "record": rec}
        return None
    parent = os.path.dirname(here)
    if os.path.basename(parent) == "versions" and re.fullmatch(r"moe-cache-v.+-linux-x86_64", os.path.basename(here)):
        prefix = os.path.dirname(parent)
        rec = os.path.join(prefix, ".installed")
        if os.path.isfile(rec):
            vals = dict(l.rstrip("\n").split("=", 1) for l in open(rec, encoding="utf-8") if "=" in l)
            return {"kind": "linux", "prefix": prefix, "command": vals.get("command", ""), "record": rec, "version": vals.get("version", ""),
                    "previous": os.path.basename(here)}
    return None


def installer_command(layout, script, tag, env=None, base=None):
    """(argv, env) that runs the downloaded installer for the folders of `layout`."""
    env = dict(os.environ if env is None else env)
    if layout["kind"] == "linux":
        env["MOE_CACHE_REF"] = tag
        env["MOE_CACHE_KEEP_PREVIOUS"] = "1"
        if base:
            env["MOE_CACHE_BASE_URL"] = base
        argv = ["bash", script, "--prefix", layout["prefix"], "--no-apt"]
        if layout.get("command"):
            argv += ["--bin-dir", os.path.dirname(layout["command"])]
        return argv, env
    ps = os.path.join(env.get("SystemRoot", r"C:\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Root", layout["root"]] + (["-BaseUrl", base] if base else []), env


def attestation(path, base_url_is_default, env=None):
    """('verified' | 'failed' | 'skipped', note). gh can only check when it is installed and logged in, and not for a mirror / test folder."""
    env = os.environ if env is None else env
    gh = shutil.which("gh")
    if not base_url_is_default or not gh or subprocess.run([gh, "auth", "status"], capture_output=True).returncode:
        return "skipped", "gh is not usable here, so only HTTPS from github.com vouches for the installer (the installer still checks the package's SHA256SUMS)"
    r = subprocess.run([gh, "attestation", "verify", path, "-R", REPO], capture_output=True, text=True)
    return ("verified", "build attestation of the installer verified (gh)") if r.returncode == 0 else ("failed", (r.stderr or r.stdout).strip()[-300:])


def apply(version, here, tag=None, yes=False, dry=False, fetch_json=http_json, base_url=None, env=None, windows=WINDOWS, tty=None):
    """Returns a process exit code. version = the running moe-cache version; here = its folder (the repository root of the copy)."""
    env = os.environ if env is None else env
    layout = installed_layout(here, windows, env)
    if not layout:
        say("moe-cache: this copy was not put here by the moe-cache installer (a git checkout, or an unpacked tarball / kit), so `update --apply` leaves it alone.")
        say("  Install the release the normal way (README, 'Quickest: one line'), or update this copy by hand. Nothing was changed.")
        return 1
    tag = tag or newest_stable_tag(fetch_json)
    if not tag:
        say("moe-cache: could not read the newest release from GitHub (offline? rate limit?). Nothing was changed.")
        return 1
    if not re.fullmatch(r"v\d+(\.\d+)*([.-][A-Za-z0-9]+)?", tag):
        say(f"moe-cache: '{tag}' does not look like a release tag. Nothing was changed.")
        return 1
    if tag == "v" + version:
        say(f"moe-cache {version} is already the newest release. Nothing to do.")
        return 0
    default_base = base_url is None and not env.get("MOE_CACHE_BASE_URL")
    base = base_url or env.get("MOE_CACHE_BASE_URL") or f"https://github.com/{REPO}/releases/download/{tag}"
    name = "install.ps1" if layout["kind"] == "windows" else "install.sh"
    say(f"moe-cache {version} -> {tag}: download {base}/{name}, run it for {layout.get('prefix') or layout.get('root')}")
    say("  the installer checks the package against the release's SHA256SUMS and its build attestation (when gh can), unpacks beside the old version and rolls back on a failure"
        + ("; the version now installed stays in versions/ for a roll back" if layout["kind"] == "linux" else "; the old kit is deleted once the new one is complete"))
    if dry:
        say("  --dry-run: nothing was downloaded.")
        return 0
    if not yes:
        if not (sys.stdin.isatty() if tty is None else tty):
            say("moe-cache: no terminal to ask on: add --yes to go ahead. Nothing was changed.")
            return 1
        if input("go ahead? [y/N] ").strip().lower() not in ("y", "yes"):
            say("nothing was changed")
            return 1
    tmp = tempfile.mkdtemp(prefix="moe-cache-update-")
    try:
        script = os.path.join(tmp, name)
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{base}/{name}", headers={"User-Agent": "moe-cache-update"}), timeout=30) as r:
                data = r.read(2_000_000)
        except Exception as e:
            say(f"moe-cache: could not download {base}/{name}: {e}. Nothing was changed.")
            return 1
        if len(data) < 200 or data[:20].lstrip().lower().startswith((b"<!doctype", b"<html")):
            say(f"moe-cache: {name} did not download as a script. Nothing was changed.")
            return 1
        with open(script, "wb") as f:
            f.write(data)
        state, note = attestation(script, default_base, env)
        if state == "failed" and env.get("MOE_CACHE_SKIP_ATTESTATION") != "1":
            say(f"moe-cache: REFUSED: gh could not verify the build attestation of {name} ({note}). Nothing was changed. MOE_CACHE_SKIP_ATTESTATION=1 goes on anyway.")
            return 1
        say("  " + note)
        argv, run_env = installer_command(layout, script, tag, env, None if default_base else base)
        rc = subprocess.call(argv, env=run_env)
        if rc:
            say(f"moe-cache: the installer stopped (exit {rc}); it puts the previous install back, so you still have {version}.")
            return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if layout["kind"] == "linux":
        say(f"updated to {tag}. Back to {version}: ln -sfn versions/{layout['previous']} {layout['prefix']}/current")
    else:
        say(f"updated to {tag}.")
    return 0
