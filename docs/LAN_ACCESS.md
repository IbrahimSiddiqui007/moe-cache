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
| 8 | A request from another computer (any non-loopback peer, GET and POST alike) may use only the routes on an allow-list; every other route, and any new one, is 403. The list is in the table below | page server `REMOTE_GET`, `REMOTE_POST`, checked in `do_GET` and `do_POST`; `tests/lan_test.py` fails when a route has no decision |
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
| A remote visitor runs code on the PC | Setting the llama-server path to any program, writing an agent's config, adding a folder | rule 8 (allow-list; `/api/llama`, `/api/folders`, `/api/connect`, `/api/command` and every start, stop, bench and download route are 403) | a visitor with the token can read chats and status and send chat messages to the running model, nothing else |
| A remote visitor fills the disk | Model download, or a chat saved over and over | rule 8 (downloads stay local); saved chats are size-limited per file | long chats are stored on the PC |
| The key leaks through the process list | `--api-key KEY` on a command line | the launcher already takes the key from `LLAMA_API_KEY` / the stored file | a key typed after `--` is visible to other users of this PC |
| A second user of this PC | They can reach `127.0.0.1` anyway | token and key (SERVER-011) | same as before |

## What another computer may do with `--ui-lan`

The page token is needed for all of it. Anything not listed is 403 (so is any route added later until it is decided).

| Route | From another computer |
|---|---|
| `GET /`, `/index.html`, `/api/system`, `/api/busy`, `/api/pick`, `/api/downloads`, `/api/download`, `/api/models`, `/api/plan`, `/api/status`, `/api/bench`, `/api/servers`, `/api/chats`, `/api/log` | allowed (read only) |
| `POST /api/chat` | allowed: the point of `--ui-lan`. It only sends a message to a model server that is already running on this PC (the one the page started, one of its extra servers, or a keyless one on 127.0.0.1:8080-8089); the API key is added on the PC and never sent to the browser. It cannot start, load or configure anything |
| `POST /api/chats/save` | allowed (saves the conversation next to the others, size-limited) |
| `GET /api/key` | **403**: the key is typed by hand on the other computer (`moe_cache.py key` on this PC) |
| `GET /api/agents`, `/api/sessions`, `/api/peek`, `/api/share` | 403 (agent config files, saved agent conversations, probing ports, reports) |
| `POST /api/start`, `/api/stop`, `/api/bench`, `/api/servers/start`, `/api/servers/stop`, `/api/servers/group`, `/api/servers/plan` (the two-model split panel starts servers), `/api/pick/use`, `/api/command` | 403 (they start, stop, load or change settings) |
| `POST /api/llama`, `/api/folders`, `/api/connect`, `/api/download/start`, `/api/download/cancel`, `/api/chats/trash`, `/api/sessions/trash`, `/api/share`, `/api/share/cancel` | 403 (paths, files, downloads, Trash, reports) |

Which computers count as "home network": the server listens on every network card (`0.0.0.0`) so that 127.0.0.1 keeps working, and the ranges 172.16/12 and
100.64/10 are counted private. That includes **Docker bridge networks** (172.17/16 and similar) and VPN ranges such as **Tailscale**: a container on this PC or
a device on your tailnet can reach the page the same as a laptop on your Wi-Fi. They still need the page token and must use the exact address printed at start
(any other `Host` is refused), and they get the same short route list above. If you do not want containers or a VPN to reach it, do not use `--ui-lan` on
that PC. The token and the key travel over plain `http` (no encryption): see the third row of the threat table.

## Use

```bash
moe-cache-server MODEL --lan                 # model API on the home-network address, key forced
moe-cache-server --ui --ui-lan               # control page too (prints the URL with its token)
python3 bin/moe_cache.py key                 # the key to type into the other computer
```

On the other computer, OpenCode or Hermes use `http://ADDRESS:PORT/v1` and the key; `connect` writes that for a server on this PC only (the remote
computer is configured by hand).
