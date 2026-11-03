#!/usr/bin/env python3
"""GUI-009 on the page server, without a model: images in chat requests (content parts, data URLs only), saved chats with images, the vision flag from
llama-server's /props, and the page's own code for images and the context-full warning.
   python3 tests/chat_images_test.py"""
import base64, http.server, json, os, re, shutil, sys, tempfile, threading, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="moe-img-test-")
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP, "cache")
sys.path.insert(0, os.path.join(HERE, "bin"))
import moe_cache_ui as ui  # noqa: E402

fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 64).decode()
JPG = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff" + b"\0" * 64).decode()

# ---- chat_request ----
r = ui.chat_request({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "what is this?", "images": [PNG, JPG]}, {"role": "assistant", "content": "a"}]})
u = r["messages"][1]["content"]
check(isinstance(u, list) and u[0] == {"type": "text", "text": "what is this?"} and [p["type"] for p in u] == ["text", "image_url", "image_url"] and u[1]["image_url"]["url"] == PNG,
      "a user message with images becomes text + image_url parts")
check(r["messages"][0]["content"] == "s" and r["messages"][2]["content"] == "a", "messages without images stay plain strings")
check(ui.chat_request({"messages": [{"role": "user", "content": "", "images": [PNG]}]})["messages"][0]["content"] == [{"type": "image_url", "image_url": {"url": PNG}}], "an image-only message has no empty text part")
for bad, what in (([{"role": "user", "content": "x", "images": ["http://evil/x.png"]}], "an http URL"), ([{"role": "user", "content": "x", "images": ["data:text/html;base64,AAAA"]}], "a non-image data URL"),
                  ([{"role": "assistant", "content": "x", "images": [PNG]}], "images on an assistant message"), ([{"role": "user", "content": "x", "images": [PNG] * 9}], "nine images"),
                  ([{"role": "user", "content": "x", "images": "file:///etc/passwd"}], "images that are not a list"), ([{"role": "user", "content": "x", "images": ["data:image/png;base64,AA\"><x"]}], "a URL with markup")):
    try:
        ui.chat_request({"messages": bad}); check(False, "refused: " + what)
    except ValueError:
        check(True, "refused: " + what)
check(ui.chat_request({"messages": [{"role": "user", "content": "x"}]})["messages"][0] == {"role": "user", "content": "x"}, "a plain chat is unchanged")

# ---- saved chats keep the images, only valid ones ----
ui.CHAT_DIR = os.path.join(TMP, "chats")
c = {"id": "img-1", "title": "t", "model": "m", "system": "", "messages": [{"role": "user", "content": "hi", "shown": "hi", "images": [PNG], "files": [{"name": "a.png", "tokens": 100, "image": True}]}], "tokens": 5, "files": []}
ui.save_chat(c)
back = json.load(open(ui.chat_path("img-1"), encoding="utf-8"))
check(back["messages"][0]["images"] == [PNG] and back["messages"][0]["files"][0]["image"] is True, "a saved chat keeps its images and their file entry")
c["messages"][0]["images"] = ["https://example.com/x.png"]
try:
    ui.save_chat(c); check(False, "a chat with a web URL as image is refused")
except ValueError:
    check(True, "a chat with a web URL as image is refused")

# ---- through the server: vision flag and the request llama-server receives ----
SEEN, PROPS = [], {"modalities": {"vision": True}}
class Up(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = json.dumps(PROPS).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        SEEN.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Up)
threading.Thread(target=up.serve_forever, daemon=True).start()
TOKEN = "img-token"
srv = ui.make_server(0, "llama-server", token=TOKEN)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"
run = srv.RequestHandlerClass.runner
def req(path, body=None):
    h = {"X-MoeCache-Token": TOKEN}
    if body is not None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, h)
    try:
        with urllib.request.urlopen(r, timeout=30) as x:
            return x.status, x.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
check(json.loads(req("/api/status")[1])["vision"] is None, "status: vision unknown while no model runs")
run.state, run.settings = "ready", {"model": "m.gguf", "port": up.server_address[1], "ctx": 8192}
check(json.loads(req("/api/status")[1])["vision"] is True, "status: vision true when llama-server's /props says modalities.vision")
run.reset({}); run.state, run.settings = "ready", {"model": "m.gguf", "port": up.server_address[1], "ctx": 8192}
PROPS["modalities"] = {"vision": False}; run.vision_at = 0
check(json.loads(req("/api/status")[1])["vision"] is False, "status: vision false for a text-only model")
st, body = req("/api/chat", {"messages": [{"role": "user", "content": "describe", "images": [PNG]}]})
check(st == 200 and SEEN and SEEN[-1]["messages"][0]["content"][1]["image_url"]["url"] == PNG, "/api/chat passes the image to llama-server as an image_url part")
check(req("/api/chat", {"messages": [{"role": "user", "content": "x", "images": ["http://127.0.0.1:1/x"]}]})[0] == 400, "/api/chat refuses a web URL as image")
big = {"messages": [{"role": "user", "content": "x", "images": [PNG]}], "pad": "x" * (ui.MAX_CHAT_BODY + 10)}
try:
    st = req("/api/chat", big)[0]
except Exception:
    st = 413
check(st in (413, 400), "a body over the chat limit is refused")

# ---- the page's own code ----
page = open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8").read()
check("async function addImage" in page and "createImageBitmap" in page and 'S.status.vision === false' in page and "--mmproj" in page, "page: images are shrunk in the browser and refused with the --mmproj hint when the model has no encoder")
check("images.length ? { images }" in page and "m.images && m.images.length" in page, "page: images go into the sent message and are kept in the chat")
check('id="ctxwarn"' in page and "The chat is full" in page and "% full" in page, "page: a warning above the input at 85 % and a refusal when the chat is full")
check("@media (max-width: 760px)" in page and "@media (max-width: 1180px)" in page, "page: the narrow layouts are in the stylesheet")

check("#switch, #start { min-width:0; max-width:100%; overflow:hidden; text-overflow:ellipsis" in page, "page: on a phone the Start / Switch button shrinks with an ellipsis (a long model name pushed the page to 560 px)")

shutil.rmtree(TMP, ignore_errors=True)
print("\nFAILED: " + "; ".join(fails) if fails else "\nchat image tests passed")
sys.exit(1 if fails else 0)
