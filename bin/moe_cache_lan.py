"""Serving to the home network (SERVER-016 `--lan`, GUI-012 `--ui-lan`): the address rules, in plain functions a test can fake. Standard library only.
docs/LAN_ACCESS.md has the threat table; every rule there that is about an address is here.

  classify(addr)            "loopback" | "private" | "public" | "any" (0.0.0.0 / ::) | "bad"
  local_addresses()         the IPv4 addresses of this PC (best effort; tests pass their own)
  pick_address(addrs, want) the one private address `--lan` binds, or an error sentence
  check_bind(host, addrs, allow_public)  None when the bind is allowed, else the refusal sentence"""
import ipaddress, os, socket

# What counts as "home network": the three RFC 1918 ranges, link-local, 100.64/10 (Tailscale and similar VPNs) and the IPv6 equivalents. Spelled out because Python's
# is_private also covers documentation and other reserved ranges, and "anything else is public" is the safe side for this check.
PRIVATE_NETS = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "100.64.0.0/10", "fc00::/7", "fe80::/10")]


def is_loopback_host(host):
    """True for a --host value that keeps llama-server on this PC only (empty = llama-server's own default 127.0.0.1)."""
    h = str(host or "").strip().strip("[]").lower()
    return h == "" or h == "localhost" or classify(h) == "loopback"


def classify(addr):
    try:
        ip = ipaddress.ip_address(str(addr).strip().strip("[]"))
    except ValueError:
        return "bad"
    if ip.is_unspecified:
        return "any"
    if ip.is_loopback:
        return "loopback"
    if any(ip.version == n.version and ip in n for n in PRIVATE_NETS):
        return "private"
    return "public"


def local_addresses():
    """IPv4 addresses of this PC: the one the system would use to reach a LAN address (no packet is sent) and what the host name resolves to."""
    if os.environ.get("MOE_CACHE_LAN_ADDRS") is not None:            # tests: "192.168.1.20,203.0.113.5" instead of asking the system
        return [a.strip() for a in os.environ["MOE_CACHE_LAN_ADDRS"].split(",") if a.strip()]
    found = []
    for target in ("192.168.0.1", "10.255.255.255", "172.16.0.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((target, 9))
            found.append(s.getsockname()[0])
            s.close()
        except OSError:
            pass
    try:
        found += [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    return [a for a in dict.fromkeys(found) if classify(a) not in ("any", "bad")]


def pick_address(addrs, want=None):
    """The private address `--lan` tells the user to use (the server itself listens on 0.0.0.0, see check_bind): `want` when given (it must be one of this PC's private addresses), else the first private one (192.168 and 10
    before the rest). Returns (address, None) or (None, sentence)."""
    priv = [a for a in addrs if classify(a) == "private"]
    priv.sort(key=lambda a: (not a.startswith(("192.168.", "10.")), a))
    if want:
        if classify(want) != "private":
            return None, f"--lan-address {want} is not a private (home network) address of this PC; private ones here: {', '.join(priv) or 'none found'}"
        if want not in addrs:
            return None, f"{want} is not an address of this PC (found: {', '.join(addrs) or 'none'})"
        return want, None
    if not priv:
        return None, ("no private (home network) address found on this PC, so --lan has nothing safe to listen on. Connect to the network, or give the address "
                      "yourself with --lan-address ADDR")
    return priv[0], None


def check_bind(host, addrs, allow_public=False):
    """May a server listen on `host`? None = yes. Loopback and private addresses: yes. A public address, or 0.0.0.0 / :: on a PC that has a public address:
    only with allow_public."""
    kind = classify(host)
    if kind == "bad":
        return f"'{host}' is not an IP address"
    if kind in ("loopback", "private"):
        return None
    if allow_public:
        return None
    if kind == "any" and ipaddress.ip_address(str(host).strip().strip("[]")).version == 6:
        # "::" listens on every IPv6 card (and, on most systems, every IPv4 one too); local_addresses() only knows IPv4 addresses, so a global IPv6 address
        # of this PC cannot be seen: treat it as public
        return f"{host} would listen on every network card, including IPv6 ones this check cannot see: refused. Use --lan (one home-network address) or add --allow-public"
    if kind == "public":
        return f"{host} is a public (internet) address: refused. Use a home-network address, a VPN address, or add --allow-public if you really mean it"
    pub = [a for a in addrs if classify(a) == "public"]
    if pub:
        return f"{host} would listen on every network card, and this PC has a public address ({pub[0]}): refused. Use --lan (one home-network address) or add --allow-public"
    return None


def risk_lines(url, key_file=None):
    """What the launcher prints when it opens a network address, in plain words (the key itself is never printed)."""
    out = [f"moe-cache: LAN: listening on {url}"]
    if key_file:
        out.append(f"moe-cache: LAN: the API key is stored in {key_file} (`python3 bin/moe_cache.py key` prints it); every request needs it")
    out.append("moe-cache: LAN: this is plain http: anyone who can watch your network can read the key and your chats. Use it on a network you trust "
               "(or over a VPN); do not forward this port on your router (docs/LAN_ACCESS.md)")
    return out
