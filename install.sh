#!/bin/bash
# moe-cache setup for a fresh Ubuntu / Debian machine: installs the build tools, builds the matching llama.cpp (CPU, optionally Vulkan)
# and the moe-cache plugin, creates a Python environment for the planner and prints the commands to run a model.
#   ./install.sh [--vulkan] [--prefix DIR] [--jobs N] [--no-apt]
#     --vulkan   also build the Vulkan backend (use an AMD / Intel / NVIDIA GPU or integrated GPU)
#     --prefix   where everything goes (default: ~/moe-cache-setup)
#     --no-apt   do not install packages (you already have cmake, g++, git, python3-venv)
#     --own-llama  build llama.cpp even when this folder has the official one in llama/ (the -ggml0.26 release tarball does)
#     --dry-run  print what it would do and stop
# In a release tarball with llama/ (llama.cpp b12207: the official source plus moe-cache's patches, all off, since v0.3.8) it builds nothing: no compiler, no apt, about a minute. LLAMA_SHA=... or --own-llama builds instead.
# Takes about 10-20 minutes, almost all of it compiling llama.cpp. Needs about 3 GB of disk. Run it as a normal user (it asks for sudo only for apt).
# Downloaded on its own (the release's install.sh, not inside a checkout or tarball) it installs the release instead of building: it fetches the
# ggml0.26 tarball (official llama.cpp inside), checks it against SHA256SUMS (refuses on a mismatch; that catches a damaged download, not a tampered release: see the attestation below), unpacks it under --prefix (default
# ~/.local/share/moe-cache) and adds a `moe-cache` command in ~/.local/bin. No apt, no compiler, no root, nothing sent anywhere. Run it again to update.
#     --uninstall  remove exactly what a release install added (the versions folder, the moe-cache command); models and ~/.cache/moe-cache stay
#     --build      build from source even when downloaded on its own (the old behaviour)
#     --bin-dir D  where the moe-cache command goes (default ~/.local/bin)
#     --yes-unverified  install even though the build attestation could not be checked (gh missing / not logged in / GitHub unreachable; a verification that ran and FAILED is never overridden); same as MOE_CACHE_ALLOW_UNVERIFIED=1. The sha256 check is never skipped.
[ -n "$BASH_VERSION" ] || { command -v bash >/dev/null && exec bash "$0" "$@"; echo "install.sh needs bash"; exit 1; }   # `sh install.sh` is the documented way
set -e
# The llama.cpp commit moe-cache was built and tested against. Default: 7fe450e19 (ggml 0.25.1, backend API 2).
# LLAMA_SHA=23b0202a1... (any full sha starting 23b0202a1, ggml 0.26.0, API 3: GLM-5.3-Flash and MTP support) selects the matching plugin build.
LLAMA_SHA_GIVEN="${LLAMA_SHA:-}"                     # set by the user: build that commit, never the bundled build
LLAMA_SHA="${LLAMA_SHA:-7fe450e19305b828c199d602c23a8337aaa1f03b}"
case "$LLAMA_SHA" in 23b0202a1*) export GGML_API=3;; esac
MOE_CACHE_REF="${MOE_CACHE_REF:-v0.3.7.1}"            # tag of moe-cache used without a checkout (newest release; the release workflow checks it matches the tag)
GGUF_PY_VERSION=0.19.0                                # python 'gguf' package version the planner was tested with
GGUF_PY_SHA256=70bcd10edfe697fb2dad6e40af2234b9d8ece9a41a99761405121ebda1c3c1cd   # sha256 of gguf-$GGUF_PY_VERSION-py3-none-any.whl (RELEASE-008); pip is refused to install anything else
UNVERIFIED_OK=0; [ "${MOE_CACHE_ALLOW_UNVERIFIED:-}" = 1 ] || [ "${MOE_CACHE_SKIP_ATTESTATION:-}" = 1 ] && UNVERIFIED_OK=1
VULKAN=0; PREFIX=""; JOBS=$(nproc 2>/dev/null || echo 2); APT=1; OWN=0; DRY=0; UNINSTALL=0; BUILD=0; BINDIR="$HOME/.local/bin"
while [ $# -gt 0 ]; do case "$1" in --yes-unverified) UNVERIFIED_OK=1; shift;; --vulkan) VULKAN=1; shift;; --prefix) PREFIX=$2; shift 2;; --jobs) JOBS=$2; shift 2;; --no-apt) APT=0; shift;; --own-llama) OWN=1; shift;; --dry-run) DRY=1; shift;; --uninstall) UNINSTALL=1; shift;; --build) BUILD=1; shift;; --bin-dir) BINDIR=$2; shift 2;;
  -h|--help) sed -n 2,20p "$0"; exit 0;; *) echo "unknown option: $1"; exit 1;; esac; done
say() { printf '\n== %s\n' "$*"; }

# ---- release install: this file was downloaded on its own (no build.sh, no llama/ next to it) ----
SELF_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
if [ $BUILD = 0 ] && [ -z "$LLAMA_SHA_GIVEN" ] && [ $OWN = 0 ] && [ ! -f "$SELF_DIR/build.sh" ] && [ ! -x "$SELF_DIR/llama/llama-server" ]; then
  REL_PREFIX="${PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/moe-cache}"
  REL_BASE="${MOE_CACHE_BASE_URL:-https://github.com/IbrahimSiddiqui007/moe-cache/releases/download/$MOE_CACHE_REF}"   # MOE_CACHE_BASE_URL: tests, mirrors
  case "$REL_PREFIX" in /*) ;; *) REL_PREFIX="$PWD/$REL_PREFIX";; esac; case "$BINDIR" in /*) ;; *) BINDIR="$PWD/$BINDIR";; esac
  REL_VER="${MOE_CACHE_REF#v}"; REL_NAME="moe-cache-v$REL_VER-ggml0.26-linux-x86_64"
  REL_SHIM="$BINDIR/moe-cache"; REL_MARK="# moe-cache command written by install.sh"; REL_REC="$REL_PREFIX/.installed"
  rec() { [ -f "$REL_REC" ] && sed -n "s/^$1=//p" "$REL_REC" | tail -1; }
  if [ $UNINSTALL = 1 ]; then
    [ -f "$REL_REC" ] || { echo "moe-cache: nothing to remove: $REL_REC does not exist (this installer added nothing under $REL_PREFIX)"; exit 0; }
    if [ $DRY = 1 ]; then echo "install.sh: would remove the moe-cache-v* folders in $REL_PREFIX/versions, $REL_PREFIX/current, $REL_REC and $REL_SHIM (if it is ours)"; exit 0; fi
    REC_CMD="$(rec command || true)"
    for d in "$REL_PREFIX"/versions/moe-cache-v*-ggml0.26-linux-x86_64*; do [ -d "$d" ] && rm -rf "$d"; done     # only the folders this installer makes (the prefix may be shared)
    rmdir "$REL_PREFIX/versions" 2>/dev/null || true
    rm -f "$REL_PREFIX/current" "$REL_REC"
    for c in "$REL_SHIM" "$REC_CMD"; do      # the command given now, and the one the install recorded (it may have used another --bin-dir)
      if [ -n "$c" ] && [ -f "$c" ] && grep -qF "$REL_MARK" "$c"; then rm -f "$c"; echo "removed $c"; fi
    done
    rmdir "$REL_PREFIX" 2>/dev/null || true
    echo "removed what the moe-cache installer added. Kept: your models and ~/.cache/moe-cache (settings, learned profiles); delete them by hand if you want."
    exit 0
  fi
  [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || { echo "the release install is for Linux x86-64 (this is $(uname -s) $(uname -m)); build from source: sh install.sh --build"; exit 1; }
  [ "$(id -u)" != 0 ] || { echo "run as a normal user, not root"; exit 1; }
  echo "moe-cache $REL_VER: release install into $REL_PREFIX (command: $REL_SHIM)"
  if [ $DRY = 1 ]; then echo "install.sh: would download $REL_BASE/$REL_NAME.tar.gz and SHA256SUMS, check the hash, unpack under $REL_PREFIX/versions and add $REL_SHIM"; exit 0; fi
  command -v python3 >/dev/null && python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' \
    || { echo "Python 3.8 or newer is needed (python3 not found or too old). Ubuntu / Debian: sudo apt install python3; Fedora: sudo dnf install python3; Arch: sudo pacman -S python. Then run this again."; exit 1; }
  if command -v curl >/dev/null; then dl() { curl -fsSL -o "$2" "$1"; }; elif command -v wget >/dev/null; then dl() { wget -q -O "$2" "$1"; }
  else echo "curl or wget is needed to download the release"; exit 1; fi
  command -v sha256sum >/dev/null || { echo "sha256sum is needed to check the download"; exit 1; }
  # preflight: everything that can refuse runs BEFORE anything is downloaded or touched (AUD-049)
  if [ -e "$REL_SHIM" ] && ! grep -qF "$REL_MARK" "$REL_SHIM" 2>/dev/null; then echo "$REL_SHIM exists and is not ours: left alone (use --bin-dir). Nothing was installed."; exit 1; fi
  mkdir -p "$REL_PREFIX/versions" "$BINDIR" 2>/dev/null && [ -w "$REL_PREFIX/versions" ] && [ -w "$BINDIR" ] || { echo "cannot write to $REL_PREFIX/versions and $BINDIR: choose other folders with --prefix / --bin-dir. Nothing was installed."; exit 1; }
  failpoint() { [ "${MOE_CACHE_INSTALL_FAIL_AT:-}" != "$1" ] || { echo "install.sh: test failure injected at $1" >&2; exit 1; }; return 0; }   # tests only
  # Nothing is committed until the end: the new version is unpacked beside the old one, swapped in by renames, and if any step fails
  # the trap puts the previous install back and removes the half-done pieces.
  TMP="$(mktemp -d)"; STAGE="$REL_PREFIX/versions/$REL_NAME.part"; BACKUP="$REL_PREFIX/versions/$REL_NAME.old"; NEWSHIM="$BINDIR/.moe-cache.new.$$"; NEWLINK="$REL_PREFIX/.current.new.$$"; NEWREC="$REL_PREFIX/.installed.new.$$"; OLDSHIM="$BINDIR/.moe-cache.old.$$"
  SWAPPED=0; SHIMMED=0; COMMITTED=0; PREV_LINK="$(readlink "$REL_PREFIX/current" 2>/dev/null || true)"
  cleanup() {
    rc=$?
    if [ $COMMITTED = 0 ] && [ $rc != 0 ]; then
      if [ $SWAPPED = 1 ]; then rm -rf "$REL_PREFIX/versions/$REL_NAME"; fi
      if [ -d "$BACKUP" ]; then rm -rf "$REL_PREFIX/versions/$REL_NAME"; mv "$BACKUP" "$REL_PREFIX/versions/$REL_NAME"; fi
      if [ $SHIMMED = 1 ]; then if [ -f "$OLDSHIM" ]; then mv -f "$OLDSHIM" "$REL_SHIM"; else rm -f "$REL_SHIM"; fi; fi
      if [ -z "$PREV_LINK" ]; then rm -f "$REL_PREFIX/current"; fi
      if [ -n "$PREV_LINK" ]; then ln -sfn "$PREV_LINK" "$REL_PREFIX/.current.back.$$" && mv -T "$REL_PREFIX/.current.back.$$" "$REL_PREFIX/current"; fi
      [ $rc = 0 ] || echo "install.sh: stopped; the previous install (if any) is untouched and the half-done files were removed." >&2
    fi
    rm -rf "$TMP" "$STAGE" "$NEWSHIM" "$NEWLINK" "$NEWREC" "$OLDSHIM" "$REL_PREFIX/.current.back.$$"; rmdir "$REL_PREFIX/versions" "$REL_PREFIX" 2>/dev/null || true
    return $rc
  }
  trap cleanup EXIT
  say "1/3 download (nothing is installed until the hash matches)"
  dl "$REL_BASE/SHA256SUMS" "$TMP/SHA256SUMS" || { echo "could not download $REL_BASE/SHA256SUMS (offline? no release $MOE_CACHE_REF?)"; exit 1; }
  WANT="$(awk -v n="$REL_NAME.tar.gz" '$2 == n || $2 == "*" n {print $1}' "$TMP/SHA256SUMS" | head -1)"
  [ -n "$WANT" ] || { echo "SHA256SUMS has no line for $REL_NAME.tar.gz: refusing to install"; exit 1; }
  dl "$REL_BASE/$REL_NAME.tar.gz" "$TMP/$REL_NAME.tar.gz" || { echo "could not download $REL_BASE/$REL_NAME.tar.gz"; exit 1; }
  GOT="$(sha256sum "$TMP/$REL_NAME.tar.gz" | cut -d' ' -f1)"
  [ "$GOT" = "$WANT" ] || { echo "REFUSED: the download's sha256 ($GOT) is not the one in SHA256SUMS ($WANT). Nothing was installed."; exit 1; }
  echo "sha256 matches SHA256SUMS ($WANT)   (SUMS come from the same release: this finds a damaged download, it does not prove who built it)"
  # RELEASE-686: the checksum alone only finds a damaged download (SHA256SUMS come from the same place as the file). Who built the file is checked
  # with `gh attestation verify`; when that cannot be done the install stops unless the user asked for it (--yes-unverified / MOE_CACHE_ALLOW_UNVERIFIED=1).
  ATT_WHY=""
  if ! command -v gh >/dev/null; then ATT_WHY="gh (the GitHub command line tool) is not installed"
  else
    # not gated on `gh auth status`: it exits 1 when ANY saved account has a bad token, although the active one works (seen on the PC: a second, expired
    # account). The verify call itself decides; its exit code is all that counts, its text only names the reason.
    # --signer-workflow: only an attestation made by this repository's release workflow counts (packages.yml pins it the same way)
    VOUT="$(gh attestation verify "$TMP/$REL_NAME.tar.gz" -R IbrahimSiddiqui007/moe-cache --signer-workflow IbrahimSiddiqui007/moe-cache/.github/workflows/release.yml 2>&1)" && ATT_OK=1 || {
      if printf '%s' "$VOUT" | grep -qiE 'no such host|dial tcp|i/o timeout|timed out|timeout|connection (refused|reset)|network is unreachable|could not resolve|temporary failure|tls handshake|no route to host|offline|HTTP 5[0-9][0-9]|rate limit|unexpected EOF'; then
        ATT_WHY="gh could not reach GitHub to verify (network): $(printf '%s' "$VOUT" | tail -c 200)"      # the check could not run: --yes-unverified may go on
      elif printf '%s' "$VOUT" | grep -qiE 'gh auth login|not logged in|authentication required|requires authentication|HTTP 401|bad credentials|GH_TOKEN'; then
        ATT_WHY="gh has no working login (not logged in, or its token is invalid or expired): the attestation check needs a working \`gh auth login\` (see: gh auth status)"
      else ATT_FAILED=1; ATT_WHY="gh attestation verify FAILED: $(printf '%s' "$VOUT" | tail -c 300)"; fi; }
  fi
  if [ "${ATT_FAILED:-0}" = 1 ]; then
    echo "checked:     sha256 of the download against SHA256SUMS of the same release (finds a damaged download)"
    echo "NOT checked: who built the file (build attestation): $ATT_WHY"
    echo "REFUSED: the build attestation was checked and FAILED. Nothing was installed. --yes-unverified / MOE_CACHE_ALLOW_UNVERIFIED=1 only cover a check that could not run (gh missing, not logged in, GitHub unreachable); they never override a failed one."; exit 1
  fi
  if [ "${ATT_OK:-0}" = 1 ]; then
    echo "checked: build attestation verified (gh): built by this repository's release workflow"
  else
    echo "checked:     sha256 of the download against SHA256SUMS of the same release (finds a damaged download)"
    echo "NOT checked: who built the file (build attestation): $ATT_WHY"
    if [ $UNVERIFIED_OK = 1 ]; then echo "WARNING: continuing without the build attestation because --yes-unverified / MOE_CACHE_ALLOW_UNVERIFIED=1 was given."
    else echo "REFUSED: the build attestation could not be checked. Nothing was installed. Log in with gh and run this again, or accept the risk with --yes-unverified (or MOE_CACHE_ALLOW_UNVERIFIED=1). Check by hand: gh attestation verify FILE -R IbrahimSiddiqui007/moe-cache"; exit 1; fi
  fi
  say "2/3 unpack (into a temporary folder; the old version stays until the new one is complete)"
  tar -tzf "$TMP/$REL_NAME.tar.gz" | awk -v n="$REL_NAME/" 'index($0, n) != 1 || $0 ~ /\.\.\//{bad=1} END{exit bad}' || { echo "REFUSED: the tarball has files outside $REL_NAME/"; exit 1; }
  OLD="$(rec version || true)"
  if [ -f "$REL_PREFIX/versions/$REL_NAME/bin/moe_cache.py" ] && [ "$OLD" = "$REL_VER" ]; then echo "moe-cache $REL_VER is already unpacked here: refreshing the command only"
  else
    rm -rf "$STAGE" "$BACKUP"; mkdir "$STAGE"
    tar -xzf "$TMP/$REL_NAME.tar.gz" -C "$STAGE" --strip-components=1
    failpoint unpack
    [ -f "$STAGE/bin/moe_cache.py" ] || { echo "REFUSED: the tarball has no bin/moe_cache.py"; exit 1; }
    [ -d "$REL_PREFIX/versions/$REL_NAME" ] && mv "$REL_PREFIX/versions/$REL_NAME" "$BACKUP"   # same version, damaged or not recorded: kept until the swap is done
    mv "$STAGE" "$REL_PREFIX/versions/$REL_NAME"; SWAPPED=1
    failpoint swap
  fi
  say "3/3 the moe-cache command"
  # single-quoted path: bash's %q writes $'..' for non-ASCII names, which /bin/sh (dash) cannot read (AUD-055)
  QPY="'$(printf %s "$REL_PREFIX/current/bin/moe_cache.py" | sed "s/'/'\\\\''/g")'"
  printf '#!/bin/sh\n%s\nexec "${MOE_CACHE_PYTHON:-python3}" %s "$@"\n' "$REL_MARK (remove: sh install.sh --uninstall)" "$QPY" > "$NEWSHIM"
  chmod +x "$NEWSHIM"
  ln -sfn "versions/$REL_NAME" "$NEWLINK"
  printf 'version=%s\nprefix=%s\ncommand=%s\n' "$REL_VER" "$REL_PREFIX" "$REL_SHIM" > "$NEWREC"
  failpoint link
  mv -T "$NEWLINK" "$REL_PREFIX/current"            # atomic: current names the old version or the new one, never nothing
  failpoint shim
  if [ -f "$REL_SHIM" ]; then cp -p "$REL_SHIM" "$OLDSHIM"; fi
  SHIMMED=1; mv -f "$NEWSHIM" "$REL_SHIM"
  failpoint record
  mv -f "$NEWREC" "$REL_REC"
  COMMITTED=1
  rm -rf "$BACKUP"
  for d in "$REL_PREFIX"/versions/moe-cache-v*-ggml0.26-linux-x86_64; do [ -d "$d" ] && [ "$d" != "$REL_PREFIX/versions/$REL_NAME" ] && { [ "${MOE_CACHE_KEEP_PREVIOUS:-}" = 1 ] && [ "$d" = "$REL_PREFIX/$PREV_LINK" ] || rm -rf "$d"; }; done     # older versions this installer unpacked, deleted last; MOE_CACHE_KEEP_PREVIOUS=1 (set by `moe_cache.py update --apply`) keeps the one that was current, for a roll back
  echo "installed: $("$REL_SHIM" --version 2>&1 | head -1)${OLD:+ (was $OLD)}"
  echo "  files:   $REL_PREFIX (versions/, current, .installed)"
  echo "  command: $REL_SHIM"
  echo "  undo:    sh install.sh --uninstall   (removes exactly the two things above; your models stay)"
  if ! ldconfig -p 2>/dev/null | grep -q libvulkan.so.1; then
    echo "note: the bundled llama.cpp uses Vulkan and no libvulkan.so.1 was found."
    if [ $APT = 1 ] && command -v apt-get >/dev/null && [ -t 0 ]; then
      printf 'install libvulkan1 and mesa-vulkan-drivers with sudo apt now? [y/N] '; read -r A || A=n
      case "$A" in y|Y|yes) sudo apt-get install -y libvulkan1 mesa-vulkan-drivers;; *) echo "  skipped: sudo apt install libvulkan1 mesa-vulkan-drivers";; esac
    else echo "  Ubuntu / Debian: sudo apt install libvulkan1 mesa-vulkan-drivers (CPU-only still works without it)"; fi
  fi
  case ":$PATH:" in *":$BINDIR:"*) ;; *) echo "note: $BINDIR is not on your PATH; add it (bash: echo 'export PATH=\"$BINDIR:\$PATH\"' >> ~/.bashrc) or run $REL_SHIM"; esac
  echo; echo "Next: moe-cache launch     (a guided start: checks this PC, suggests a model, starts the server)"
  exit 0
fi
PREFIX="${PREFIX:-$HOME/moe-cache-setup}"
# the official llama.cpp bundled in this folder (release tarball): used instead of building one
BUNDLED=""
if [ $OWN = 0 ] && [ -z "$LLAMA_SHA_GIVEN" ]; then
  HERE_="$(cd "$(dirname "$0")" && pwd)"
  for d in "$HERE_/llama" "$HERE_/llama/bin"; do [ -x "$d/llama-server" ] && { BUNDLED="$d"; break; }; done
fi
if [ $DRY = 1 ]; then
  if [ -n "$BUNDLED" ]; then echo "install.sh: would use the bundled llama.cpp in $BUNDLED (no build, no apt); plugin: the prebuilt one if present"
  else echo "install.sh: would build llama.cpp $LLAMA_SHA (CPU$([ $VULKAN = 1 ] && echo ' + Vulkan')) and the plugin"; fi
  exit 0
fi
[ "$(uname -s)" = Linux ] || { echo "Linux only"; exit 1; }
[ "$(id -u)" != 0 ] || { echo "run as a normal user, not root"; exit 1; }
[ -n "$BUNDLED" ] && VULKAN=1                      # the bundled build is the Vulkan one: the device check at the end applies
mkdir -p "$PREFIX"; PREFIX="$(cd "$PREFIX" && pwd)"     # absolute: env.sh / env.fish are sourced from other folders (a relative --prefix broke them)

if [ -n "$BUNDLED" ]; then
  say "1/5 packages: not needed (this tarball has llama.cpp and the plugin ready)"
elif [ $APT = 1 ]; then
  say "1/5 packages (sudo needed)"
  command -v apt-get >/dev/null || { echo "apt-get not found: install cmake, g++, git, python3-venv yourself and use --no-apt"; exit 1; }
  PK="build-essential cmake git curl ca-certificates python3 python3-venv python3-pip ninja-build"
  if [ $VULKAN = 1 ]; then PK="$PK libvulkan-dev mesa-vulkan-drivers vulkan-tools spirv-headers"; fi
  sudo apt-get update -y
  sudo apt-get install -y $PK
  if [ $VULKAN = 1 ]; then sudo apt-get install -y glslc || sudo apt-get install -y glslang-tools shaderc || true; fi
else
  say "1/5 packages: skipped (--no-apt)"
fi
NEED="cmake g++ git python3"; [ -n "$BUNDLED" ] && NEED=python3
for c in $NEED; do command -v $c >/dev/null || { echo "missing: $c"; exit 1; }; done

if [ -n "$BUNDLED" ]; then
  BIN="$BUNDLED"; say "2/5 and 3/5 llama.cpp: using the bundled official build in $BIN"
else
say "2/5 llama.cpp at the tested commit"
L="$PREFIX/llama.cpp"
if [ ! -d "$L/.git" ]; then mkdir -p "$L"; git -C "$L" init -q; git -C "$L" remote add origin https://github.com/ggml-org/llama.cpp.git; fi
git -C "$L" fetch -q --depth 1 origin "$LLAMA_SHA"
git -C "$L" checkout -q FETCH_HEAD

say "3/5 building llama.cpp ($JOBS jobs; this is the long step)"
FLAGS=(-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)
[ $VULKAN = 1 ] && FLAGS+=(-DGGML_VULKAN=ON)
GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
cmake -S "$L" -B "$L/build" "${GEN[@]}" "${FLAGS[@]}" >/dev/null
cmake --build "$L/build" --target llama-server llama-bench llama-perplexity -j "$JOBS"
BIN="$L/build/bin"; [ -x "$BIN/llama-server" ] || { echo "llama-server was not built"; exit 1; }
fi

say "4/5 moe-cache plugin"
if [ -f "$(dirname "$0")/build.sh" ] && [ -d "$(dirname "$0")/src" ]; then M="$(cd "$(dirname "$0")" && pwd)"; else
  M="$PREFIX/moe-cache"
  if [ -d "$M/.git" ]; then git -C "$M" fetch -q --tags origin && git -C "$M" checkout -q "$MOE_CACHE_REF"
  else git clone -q --branch "$MOE_CACHE_REF" https://github.com/IbrahimSiddiqui007/moe-cache "$M"; fi
fi
if [ -n "$BUNDLED" ] && [ -f "$M/build/libggml-moe-cache.so" ]; then echo "plugin: the prebuilt one in $M/build (nothing to compile)"
else GGML_LIB_DIR="$BIN" "$M/build.sh"; fi

say "5/5 python environment for the planner"
# the wheel is fetched alone, its sha256 compared with the pinned one, and only then installed (its dependencies come from pip as usual)
gguf_wheel() {
  W="$(mktemp -d)"; "$PREFIX/venv/bin/pip" download -q --no-deps --only-binary :all: -d "$W" "gguf==$GGUF_PY_VERSION" || return 1
  F="$W/gguf-$GGUF_PY_VERSION-py3-none-any.whl"; [ -f "$F" ] || return 1
  if [ "$(sha256sum "$F" | cut -d' ' -f1)" != "$GGUF_PY_SHA256" ]; then echo "REFUSED: the gguf wheel's sha256 is not the pinned $GGUF_PY_SHA256; not installed"; return 1; fi
  "$PREFIX/venv/bin/pip" install -q "$F"
}
if python3 -m venv "$PREFIX/venv" >/dev/null 2>&1 && gguf_wheel; then PY="$PREFIX/venv/bin/python3"
else
  echo "note: no python venv or no network for pip (Ubuntu: sudo apt install python3-venv): using the system python3; the planner has its own GGUF reader, so this is fine"
  PY="$(command -v python3)"
fi

{
  echo "# source this file:  . $(printf '%q' "$PREFIX/env.sh")"
  printf 'export LLAMA_SERVER=%q\n' "$BIN/llama-server"
  printf 'export MOE_CACHE_PYTHON=%q\n' "$PY"
  printf 'export PATH=%q:"$PATH"\n' "$M/bin"
} > "$PREFIX/env.sh"
{
  echo "# fish:  source $PREFIX/env.fish"
  printf 'set -gx LLAMA_SERVER %q\n' "$BIN/llama-server"
  printf 'set -gx MOE_CACHE_PYTHON %q\n' "$PY"
  printf 'set -gx PATH %q $PATH\n' "$M/bin"
} > "$PREFIX/env.fish"
if [ $VULKAN = 1 ]; then
  # Vulkan falls back to a software device (llvmpipe) when the user may not open the GPU's render node, e.g. on a headless server (T7)
  DEVS="$("$BIN/llama-server" --list-devices 2>&1)"
  if ! echo "$DEVS" | grep -q "Vulkan[0-9]" || echo "$DEVS" | grep -q -i "llvmpipe"; then
    say "WARNING: llama-server does not list a real Vulkan GPU:"; echo "$DEVS" | sed -n '/Available devices/,$p'
    for r in /dev/dri/renderD*; do
      if [ -e "$r" ] && [ ! -r "$r" ]; then
        echo "  $r is not readable by $(id -un): run  sudo usermod -aG render,video $(id -un)  and log in again"
        break
      fi
    done
  fi
fi
say "setup check"
LLAMA_SERVER="$BIN/llama-server" MOE_CACHE_PYTHON="$PY" "$M/bin/moe-cache-check" --llama-server "$BIN/llama-server" || true
cat <<EOT

Done. Next steps (bash or zsh; for fish use: source $PREFIX/env.fish):
  . $PREFIX/env.sh
  # get any MoE model in GGUF format, for example (12 GB):
  #   curl -L -o gpt-oss-20b.gguf https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf
  moe-cache-check gpt-oss-20b.gguf
  $M/tests/identity.sh gpt-oss-20b.gguf --llama-server $BIN/llama-server     # PASS = same tokens as stock
  moe-cache-server gpt-oss-20b.gguf --ram 20 --open                          # chat UI at http://localhost:8080
  python3 $M/bin/moe_cache.py bench-report gpt-oss-20b.gguf                   # stock vs moe-cache on your machine (or Share my numbers on the control page)
EOT
