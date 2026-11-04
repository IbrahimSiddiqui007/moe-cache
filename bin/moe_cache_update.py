"""moe_cache.py update --apply: install the newer moe-cache release with the release's own installer (RELEASE-616).

It does not unpack anything itself. It downloads the release's install.sh (Linux) or install.ps1 (Windows), checks that file's build attestation
with `gh` when gh is usable, and runs it for the same folders the running copy was installed into. The installer then does the rest: it checks the
tarball / zip against the release's SHA256SUMS and its build attestation, unpacks beside the old version, swaps by rename and rolls back on any
failure. On Linux the previous version stays in versions/ (MOE_CACHE_KEEP_PREVIOUS=1) and `current` can be pointed back at it; on Windows the previous
kit stays as kit.prev. After the swap it runs the new command's --version and imports the launcher modules from the new folder (no model); a failure
puts the previous version back.
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


def version_key(v):
    """Sortable key of a release version or tag ('v0.3.6', '0.3.6-beta2'): numbers first, a final release sorts after its own pre-releases.
    None when it is not a version (then nothing is compared)."""
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)(?:[.-]([A-Za-z0-9.]+))?", str(v).strip())
    if not m:
        return None
    nums = tuple(int(x) for x in m.group(1).split("."))
    nums += (0,) * (4 - len(nums))
    pre = m.group(2)
    tail = tuple((0, int(x), "") if x.isdigit() else (1, 0, x.lower()) for x in re.findall(r"\d+|[A-Za-z]+", pre)) if pre else ()
    return (nums, 0 if pre else 1, tail)


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


def on_path(folder, env):
    """True when `folder` is in PATH. An update leaves the PATH alone when the command folder is not there (an install made with -NoPath stays that way)."""
    norm = lambda x: os.path.normcase(os.path.normpath(x))
    return any(norm(x) == norm(folder) for x in (env.get("PATH") or env.get("Path") or "").split(os.pathsep) if x)


def installer_command(layout, script, tag, env=None, base=None):
    """(argv, env) that runs the downloaded installer for the folders of `layout`."""
    env = dict(os.environ if env is None else env)
    env["MOE_CACHE_KEEP_PREVIOUS"] = "1"
    if layout["kind"] == "linux":
        env["MOE_CACHE_REF"] = tag
        if base:
            env["MOE_CACHE_BASE_URL"] = base
        argv = ["bash", script, "--prefix", layout["prefix"], "--no-apt"]
        if layout.get("command"):
            argv += ["--bin-dir", os.path.dirname(layout["command"])]
        return argv, env
    ps = os.path.join(env.get("SystemRoot", r"C:\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Root", layout["root"]] + (["-BaseUrl", base] if base else []) + ([] if on_path(os.path.join(layout["root"], "bin"), env) else ["-NoPath"]), env


KIT_MODULES = "moe_cache, moe_cache_ui, moe_cache_on_demand, moe_cache_pick, moe_cache_pack, moe_cache_model, moe_cache_launch"


def new_bin_dir(layout):
    return os.path.join(layout["prefix"], "current", "bin") if layout["kind"] == "linux" else os.path.join(layout["root"], "kit", "bin")


def verify_new(layout, tag, command, run=subprocess.run):
    """After the swap, without a model: the moe-cache command prints the new version, and the launcher modules import from the new folder alone
    (the same check tools/make_windows_kit.sh makes on the kit). Returns '' when fine, else what is wrong."""
    want = "moe-cache " + tag.lstrip("v")
    try:
        r = run([command, "--version"], capture_output=True, text=True, timeout=60, shell=False)
        got = (r.stdout or "").strip().splitlines()[:1]
        if r.returncode or not got or got[0] != want:
            return f"`{command} --version` printed {got or r.stderr.strip()[-120:]}, expected '{want}'"
        r = run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, sys.argv[1]); import {KIT_MODULES}", new_bin_dir(layout)],
                capture_output=True, text=True, timeout=120)
        if r.returncode:
            return "a launcher module does not import from the new folder: " + (r.stderr.strip().splitlines() or ["?"])[-1]
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not run the new version: {e}"
    return ""


def roll_back(layout):
    """Make the version that was installed before current again. Linux: `current` points back (atomic rename of a new link), but only when that
    folder exists: a link to a missing folder would leave no moe-cache at all. Windows: kit and kit.prev trade places (the failed kit is kept as
    kit.failed, nothing is deleted; when the second rename fails the first one is undone). Returns '' or what went wrong; nothing is changed then."""
    try:
        if layout["kind"] == "linux":
            target = os.path.join(layout["prefix"], "versions", layout["previous"])
            if not os.path.isdir(target):
                return f"{target} does not exist any more, so `current` was not changed (it still points at the version that was just installed)"
            tmp = os.path.join(layout["prefix"], ".current.back")
            if os.path.lexists(tmp):
                os.remove(tmp)
            os.symlink(os.path.join("versions", layout["previous"]), tmp)
            os.replace(tmp, os.path.join(layout["prefix"], "current"))
        else:
            kit, prev, bad = (os.path.join(layout["root"], n) for n in ("kit", "kit.prev", "kit.failed"))
            if not os.path.isdir(prev):
                return "no kit.prev to go back to"
            if os.path.exists(bad):
                return f"{bad} is in the way: move it away and go back by hand (rename kit to kit.failed, kit.prev to kit)"
            os.rename(kit, bad)
            try:
                os.rename(prev, kit)
            except OSError:
                os.rename(bad, kit)         # put the kit back: never leave the root without one
                raise
    except OSError as e:
        return str(e)
    return ""


MAX_SCRIPT = 2_000_000      # the installer scripts are a few tens of KB; anything bigger is not one
SIGNER_WORKFLOW = f"{REPO}/.github/workflows/release.yml"      # only an attestation made by this repository's release workflow counts (as packages.yml)
# gh output that means "gh could not reach GitHub / the service said later", not "the file does not match its attestation"
_NETWORK = re.compile(r"no such host|dial tcp|i/o timeout|timed out|timeout|connection (refused|reset)|network is unreachable|could not resolve|temporary failure|tls handshake|no route to host|offline|HTTP 5\d\d|rate limit|unexpected EOF", re.I)


def attestation(path, env=None):
    """('verified' | 'failed' | 'skipped', note). RELEASE-686: whenever gh is installed and logged in the file is verified, also when it came from a mirror
    (the attestation is bound to the file's content, not to where it was fetched). 'skipped' = the check could not run (gh missing, not logged in, GitHub not reachable):
    --yes-unverified may go on. 'failed' = gh ran and said no: nothing overrides that."""
    env = os.environ if env is None else env
    gh = shutil.which("gh", path=env.get("PATH"))
    if not gh:
        return "skipped", "gh (the GitHub command line tool) is not installed"
    if subprocess.run([gh, "auth", "status"], capture_output=True, env=dict(env)).returncode:
        return "skipped", "gh is installed but not logged in (run: gh auth login)"
    r = subprocess.run([gh, "attestation", "verify", path, "-R", REPO, "--signer-workflow", SIGNER_WORKFLOW], capture_output=True, text=True, env=dict(env))
    if r.returncode == 0:
        return "verified", "build attestation of the installer verified (gh)"
    out = (r.stderr or r.stdout).strip()
    if _NETWORK.search(out):
        return "skipped", "gh could not reach GitHub to verify: " + out[-200:]
    return "failed", "gh attestation verify FAILED: " + out[-300:]


def apply(version, here, tag=None, yes=False, dry=False, allow_downgrade=False, unverified=False, fetch_json=http_json, base_url=None, env=None, windows=WINDOWS, tty=None):
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
    mine, theirs = version_key(version), version_key(tag)
    if mine and theirs and theirs < mine and not allow_downgrade:
        say(f"moe-cache: REFUSED: {tag} is older than the running {version}, so this would be a downgrade. Nothing was changed. If you mean it, add --allow-downgrade.")
        return 1
    default_base = base_url is None and not env.get("MOE_CACHE_BASE_URL")
    base = base_url or env.get("MOE_CACHE_BASE_URL") or f"https://github.com/{REPO}/releases/download/{tag}"
    name = "install.ps1" if layout["kind"] == "windows" else "install.sh"
    say(f"moe-cache {version} -> {tag}: download {base}/{name}, run it for {layout.get('prefix') or layout.get('root')}")
    say("  the installer checks the package against the release's SHA256SUMS and its build attestation (when gh can), unpacks beside the old version and rolls back on a failure"
        + ("; the version now installed stays in versions/ for a roll back" if layout["kind"] == "linux" else "; the kit now installed stays as kit.prev for a roll back"))
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
                data = r.read(MAX_SCRIPT + 1)
                want = r.headers.get("Content-Length")
            if len(data) > MAX_SCRIPT:
                say(f"moe-cache: {name} is larger than {MAX_SCRIPT} bytes: not an installer script. Nothing was changed.")
                return 1
            if want is not None and want.isdigit() and int(want) != len(data):       # a cut-off download is refused, not run
                say(f"moe-cache: {name} downloaded incompletely ({len(data)} of {want} bytes). Nothing was changed.")
                return 1
        except Exception as e:
            say(f"moe-cache: could not download {base}/{name}: {e}. Nothing was changed.")
            return 1
        if len(data) < 200 or data[:20].lstrip().lower().startswith((b"<!doctype", b"<html")):
            say(f"moe-cache: {name} did not download as a script. Nothing was changed.")
            return 1
        with open(script, "wb") as f:
            f.write(data)
        state, note = attestation(script, env)
        allow = unverified or env.get("MOE_CACHE_ALLOW_UNVERIFIED") == "1" or env.get("MOE_CACHE_SKIP_ATTESTATION") == "1"
        if state == "failed":
            say(f"  NOT checked: who built {name} (build attestation): {note}")
            say("moe-cache: REFUSED: the build attestation was checked and FAILED, so nothing was run or changed. --yes-unverified / MOE_CACHE_ALLOW_UNVERIFIED=1 only cover a check that could not run (gh missing, not logged in, GitHub unreachable); they never override a failed one.")
            return 1
        if state == "verified":
            say("  checked: " + note + "; the installer checks the package's build attestation and sha256 the same way")
        else:
            say(f"  checked:     nothing about who built {name} (only HTTPS to the server it came from); the installer will check the package against the release's SHA256SUMS (finds a damaged download)")
            say(f"  NOT checked: who built {name} (build attestation): {note}")
            if not allow:
                say("moe-cache: REFUSED: the build attestation could not be checked, so nothing was run or changed. Log in with gh and try again, or accept the risk with"
                    " --yes-unverified (or MOE_CACHE_ALLOW_UNVERIFIED=1). Check by hand: gh attestation verify FILE -R " + REPO)
                return 1
            say("  WARNING: going on without the build attestation because --yes-unverified / MOE_CACHE_ALLOW_UNVERIFIED=1 was given.")
        ibase = None if default_base else base
        if ibase and ibase.startswith("file://") and layout["kind"] == "windows":      # install.ps1 takes a folder, not a file:// address
            ibase = urllib.request.url2pathname(ibase[len("file://"):]).lstrip("\\") if re.match(r"file:///[A-Za-z]:", ibase) else urllib.request.url2pathname(ibase[len("file://"):])
        argv, run_env = installer_command(layout, script, tag, env, ibase)
        if state != "verified" and allow:
            run_env["MOE_CACHE_ALLOW_UNVERIFIED"] = "1"      # the installer checks the package itself and would refuse the same way
        rc = subprocess.call(argv, env=run_env)
        if rc:
            say(f"moe-cache: the installer stopped (exit {rc}); it puts the previous install back, so you still have {version}.")
            return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    command = layout.get("command") or (os.path.join(layout["root"], "bin", "moe-cache.cmd") if layout["kind"] == "windows" else "")
    if windows and layout["kind"] == "windows":
        try:
            command = json.load(open(layout["record"], encoding="utf-8-sig")).get("command") or command
        except (OSError, ValueError):
            pass
    bad = verify_new(layout, tag, command)
    if bad:
        err = roll_back(layout)
        say(f"moe-cache: the new version failed its check ({bad}); " + (f"going back to {version} failed too: {err}" if err else f"went back to {version}."))
        return 1
    if layout["kind"] == "linux":
        say(f"updated to {tag}; its command and modules checked (no model needed). Back to {version}: ln -sfn versions/{layout['previous']} {layout['prefix']}/current")
    else:
        say(f"updated to {tag}; its command and modules checked (no model needed). The kit that was installed stays as {layout['root']}\\kit.prev (rename kit to kit.failed and kit.prev to kit to go back)")
    return 0
