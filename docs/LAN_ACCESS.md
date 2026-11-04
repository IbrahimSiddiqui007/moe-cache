# Reaching moe-cache from another computer (`--lan`, `--ui-lan`)

Status 2026-11-04: **built on the branch `lan-016`, not merged, not run on a second computer.** Off by default: nothing listens beyond this PC unless you
ask. The tests use fake addresses and fake clients (`tests/lan_test.py`).

## In plain words

By default everything moe-cache starts listens on `127.0.0.1`: only programs on this PC can reach it. Two options open one door to your home network:

| Option | What it opens | What you get |
|---|---|---|
| `moe-cache-server MODEL --lan` | The model server's API on this PC's home-network address (for example `192.168.1.20:8080`) | Another computer's agent or chat program uses the model on this PC. The API key is always on; the exact address is printed; the key is never printed, `python3 bin/moe_cache.py key` shows it |
| `moe-cache-server --ui --ui-lan` | The control page on the same kind of address | You open the page from a laptop and chat, start and stop. Actions that change this PC's files or settings stay on this PC (table below) |

**The risk, in plain words.** The network is plain `http`: anyone who can watch your home network (a guest on your Wi-Fi, an infected device) can read what
is sent, including the key and the page token, and anyone who gets the key can use the model on your PC and make it work. Use it only on a network you trust.
Do not forward the port on your router. For use from outside the house use a VPN (WireGuard, Tailscale) and keep `--lan` for the VPN's address; moe-cache
does not set that up.

## Rules the code keeps

| # | Rule | Where it is enforced |
|---|---|---|
| 1 | Nothing listens beyond this PC unless `--lan` or `--ui-lan` is given | default host stays `127.0.0.1` |
| 2 | `--lan` needs a **private address on this PC** (home ranges 10/8, 172.16/12, 192.168/16, link-local, 100.64/10 used by Tailscale) and prints it; the server listens on every network card (the launcher and the page talk to it through 127.0.0.1), so rule 3 is what keeps a public card closed | `moe_cache_lan.pick_address` |
| 3 | A public (internet) address, or listening on every card on a PC that has a public address, is refused unless `--allow-public` is also given; the same for an explicit `--host` | `moe_cache_lan.check_bind` |
| 4 | The model server never runs open on a network address: `--lan` forces an API key (yours after `--`, `LLAMA_API_KEY`, or the stored per-user key, made on first use) | launcher |
| 5 | The control page keeps the per-run token header (`X-MoeCache-Token`); there is no cookie and no way in without it | page server |
| 6 | The page answers only for the exact address it was bound to (plus `127.0.0.1` and `localhost`); a different `Host` is refused (no DNS rebinding) | page server `allowed()` |
| 7 | The address and the exact URL (with the page token) are printed once at start; the API key is never printed, only where it is stored | launcher / page |
| 8 | Requests from another computer cannot do what changes this PC outside the model: set the llama-server path, add a model folder, write an agent's config, start a download, move files to the Trash, run a share report | page server `REMOTE_BLOCKED` (403) |
| 9 | Repeated wrong tokens from one remote address are answered 429 for a minute | page server |

## Threat table

| Threat | How it could happen | What stops it | What is left |
|---|---|---|---|
| A stranger on the internet uses your model | The PC has a public address, or the port is forwarded | rule 3 (refused when the PC has a public address; a router forward cannot be seen from here: do not do it); key always on | `--allow-public` removes the guard: do not use it without a VPN or a TLS proxy |
| A neighbour or guest on your Wi-Fi uses the model | They reach the address | API key (random, 0600 file); page token | Anyone who has the key. Put guests on a guest network |
| Someone on the network reads the key or the chat | Plain `http`, no encryption | none in moe-cache | **Open risk.** A trusted network, or a VPN. TLS is not built (a reverse proxy such as Caddy can add it) |
| A web page in your browser attacks the page | Cross-site request to the LAN address | token header, `Sec-Fetch-Site` and `Origin` checks, `Host` check | none known |
| DNS rebinding | A hostile site resolves its name to your LAN address | rule 6: only the exact `ip:port` and loopback names are accepted | none known |
| Guessing the page token | Remote guessing | 24 random bytes (192 bits) plus rule 9 | none |
| A remote visitor runs code on the PC | Setting the llama-server path to any program, writing an agent's config, adding a folder | rule 8 | a visitor with the token can still start and stop models and read chats |
| A remote visitor fills the disk | Model download | rule 8 (downloads stay local) | long chats are stored on the PC |
| The key leaks through the process list | `--api-key KEY` on a command line | the launcher already takes the key from `LLAMA_API_KEY` / the stored file | a key typed after `--` is visible to other users of this PC |
| A second user of this PC | They can reach `127.0.0.1` anyway | token and key (SERVER-011) | same as before |

## Use

```bash
moe-cache-server MODEL --lan                 # model API on the home-network address, key forced
moe-cache-server --ui --ui-lan               # control page too (prints the URL with its token)
python3 bin/moe_cache.py key                 # the key to type into the other computer
```

On the other computer, OpenCode or Hermes use `http://ADDRESS:PORT/v1` and the key; `connect` writes that for a server on this PC only (the remote
computer is configured by hand).
