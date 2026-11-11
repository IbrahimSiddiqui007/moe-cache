// moe-cache-top: the native Monitor (R15-03, phase 0 spike). One Overview screen from llama-server's /metrics and /slots:
//   decode and prompt tokens/s (live, from the counters' change), a braille history of the decode speed, the context in use, requests, one row per slot.
// Optional: --monitor-url URL --token T also reads the control page's /api/monitor?layers=0 and shows its `schema` number (R15-01).
// Linux and Windows (mingw UCRT, static). C++17, FTXUI 7.0.3 (third_party/ftxui), no other dependency. Text modes: --once (one frame to stdout), --json.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sock_t;
#define CLOSESOCK close
#define INVALID_SOCKET (-1)
#endif

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/dom/table.hpp"
#include "ftxui/screen/screen.hpp"

static const char* VERSION = "moe-cache-top 0.0.1-spike (FTXUI 7.0.3)";

// ---------------------------------------------------------------- HTTP GET over a plain socket (a monitor reads one machine; no TLS, no redirects)

static bool http_get(const std::string& host, int port, const std::string& path, const std::string& extra_header, int timeout_ms, int& status, std::string& body) {
#ifdef _WIN32
    static bool wsa = false;
    if (!wsa) { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); wsa = true; }
#endif
    struct addrinfo hints; std::memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) return false;
    sock_t s = INVALID_SOCKET;
    for (struct addrinfo* a = res; a; a = a->ai_next) {
        s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == INVALID_SOCKET) continue;
#ifdef _WIN32
        DWORD tv = timeout_ms; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv); setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof tv);
#else
        struct timeval tv; tv.tv_sec = timeout_ms / 1000; tv.tv_usec = (timeout_ms % 1000) * 1000; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv); setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
        if (connect(s, a->ai_addr, (int)a->ai_addrlen) == 0) break;
        CLOSESOCK(s); s = INVALID_SOCKET;
    }
    freeaddrinfo(res);
    if (s == INVALID_SOCKET) return false;
    std::string req = "GET " + path + " HTTP/1.1\r\nHost: " + host + ":" + std::to_string(port) + "\r\n" + extra_header + "Connection: close\r\nAccept: */*\r\n\r\n";
    if (send(s, req.data(), (int)req.size(), 0) < 0) { CLOSESOCK(s); return false; }
    std::string raw; char buf[16384];
    for (;;) { int n = recv(s, buf, sizeof buf, 0); if (n <= 0) break; raw.append(buf, buf + n); if (raw.size() > (64u << 20)) break; }
    CLOSESOCK(s);
    size_t he = raw.find("\r\n\r\n");
    if (raw.compare(0, 5, "HTTP/") != 0 || he == std::string::npos) return false;
    status = std::atoi(raw.c_str() + raw.find(' ') + 1);
    std::string head = raw.substr(0, he);
    for (auto& c : head) c = (char)std::tolower((unsigned char)c);
    body = raw.substr(he + 4);
    if (head.find("transfer-encoding: chunked") != std::string::npos) {             // llama-server answers with a length; a proxy in between may chunk
        std::string out; size_t p = 0;
        while (p < body.size()) {
            size_t e = body.find("\r\n", p); if (e == std::string::npos) break;
            size_t len = std::strtoul(body.substr(p, e - p).c_str(), nullptr, 16);
            if (!len) break;
            out.append(body, e + 2, len); p = e + 2 + len + 2;
        }
        body = out;
    }
    return true;
}

// ---------------------------------------------------------------- JSON: just enough (numbers, strings, bools, null, arrays, objects)

struct J {
    enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false; double n = 0; std::string s; std::vector<J> a; std::vector<std::pair<std::string, J>> o;
    const J* get(const std::string& k) const { for (auto& kv : o) if (kv.first == k) return &kv.second; return nullptr; }
    double num(double dflt = NAN) const { return t == Num ? n : (t == Bool ? (b ? 1 : 0) : dflt); }
};
struct JP {
    const std::string& x; size_t i = 0; bool ok = true;
    explicit JP(const std::string& s) : x(s) {}
    void ws() { while (i < x.size() && std::isspace((unsigned char)x[i])) i++; }
    J value(int depth = 0) {
        J v; ws();
        if (i >= x.size() || depth > 64) { ok = false; return v; }
        char c = x[i];
        if (c == '{') { v.t = J::Obj; i++; ws(); if (i < x.size() && x[i] == '}') { i++; return v; }
            for (;;) { ws(); J k = value(depth + 1); if (!ok || k.t != J::Str) { ok = false; return v; } ws(); if (i >= x.size() || x[i] != ':') { ok = false; return v; } i++;
                J e = value(depth + 1); if (!ok) return v; v.o.emplace_back(k.s, std::move(e)); ws();
                if (i < x.size() && x[i] == ',') { i++; continue; } if (i < x.size() && x[i] == '}') { i++; return v; } ok = false; return v; } }
        if (c == '[') { v.t = J::Arr; i++; ws(); if (i < x.size() && x[i] == ']') { i++; return v; }
            for (;;) { J e = value(depth + 1); if (!ok) return v; v.a.push_back(std::move(e)); ws();
                if (i < x.size() && x[i] == ',') { i++; continue; } if (i < x.size() && x[i] == ']') { i++; return v; } ok = false; return v; } }
        if (c == '"') { v.t = J::Str; i++; while (i < x.size() && x[i] != '"') { if (x[i] == '\\' && i + 1 < x.size()) { i++; char e = x[i]; v.s += e == 'n' ? '\n' : e == 't' ? '\t' : e == 'u' ? '?' : e; if (e == 'u') i += 4; } else v.s += x[i]; i++; }
            if (i >= x.size()) ok = false; else i++; return v; }
        if (x.compare(i, 4, "true") == 0) { v.t = J::Bool; v.b = true; i += 4; return v; }
        if (x.compare(i, 5, "false") == 0) { v.t = J::Bool; i += 5; return v; }
        if (x.compare(i, 4, "null") == 0) { i += 4; return v; }
        char* end = nullptr; double d = std::strtod(x.c_str() + i, &end);
        if (end == x.c_str() + i) { ok = false; return v; }
        v.t = J::Num; v.n = d; i = (size_t)(end - x.c_str()); return v;
    }
};
static bool parse_json(const std::string& text, J& out) { JP p(text); out = p.value(); return p.ok; }

// ---------------------------------------------------------------- Prometheus text from /metrics ("llamacpp:name value", labels ignored)

static std::map<std::string, double> parse_prom(const std::string& text) {
    std::map<std::string, double> m; std::istringstream in(text); std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t sp = line.find_last_of(' '); if (sp == std::string::npos) continue;
        std::string name = line.substr(0, sp); size_t br = name.find('{'); if (br != std::string::npos) name = name.substr(0, br);
        if (name.compare(0, 9, "llamacpp:") == 0) name = name.substr(9);
        m[name] = std::atof(line.c_str() + sp + 1);
    }
    return m;
}

// ---------------------------------------------------------------- state

struct Slot { int id = 0; bool busy = false; double n_ctx = NAN, decoded = NAN, remain = NAN; };
struct State {
    bool reachable = false; std::string error; double schema = NAN; std::string schema_note;
    double decode_tps = NAN, prompt_tps = NAN, decode_avg = NAN, prompt_avg = NAN, kv_ratio = NAN, processing = NAN, deferred = NAN;
    double ctx_total = NAN; std::vector<Slot> slots; std::deque<double> history; double polled_at = 0; long polls = 0;
};
struct Prev { bool have = false; double t = 0, pt = 0, ps = 0, dt = 0, ds = 0; };

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void poll_once(const std::string& host, int port, const std::string& mhost, int mport, const std::string& token, State& st, Prev& pv) {
    int code = 0; std::string body;
    st.polls++;
    if (!http_get(host, port, "/metrics", "", 1500, code, body)) { st.reachable = false; st.error = "llama-server is not answering on " + host + ":" + std::to_string(port); return; }
    if (code != 200) { st.reachable = false; st.error = "/metrics answered HTTP " + std::to_string(code) + " (start llama-server with --metrics)"; return; }
    st.reachable = true; st.error.clear();
    auto m = parse_prom(body); auto g = [&](const char* k) { auto it = m.find(k); return it == m.end() ? NAN : it->second; };
    double t = now_s(), pt = g("prompt_tokens_total"), ps = g("prompt_seconds_total"), dt = g("tokens_predicted_total"), ds = g("tokens_predicted_seconds_total");
    if (pv.have) {
        if (ds - pv.ds > 1e-6 && dt - pv.dt >= 0) st.decode_tps = (dt - pv.dt) / (ds - pv.ds); else st.decode_tps = 0;        // no decode time passed: idle
        if (ps - pv.ps > 1e-6 && pt - pv.pt >= 0) st.prompt_tps = (pt - pv.pt) / (ps - pv.ps); else st.prompt_tps = 0;
        st.history.push_back(std::isnan(st.decode_tps) ? 0 : st.decode_tps);
        while (st.history.size() > 240) st.history.pop_front();
    }
    pv = {true, t, pt, ps, dt, ds};
    st.decode_avg = g("predicted_tokens_seconds"); st.prompt_avg = g("prompt_tokens_seconds"); st.kv_ratio = g("kv_cache_usage_ratio");
    st.processing = g("requests_processing"); st.deferred = g("requests_deferred");
    st.slots.clear(); st.ctx_total = NAN;
    if (http_get(host, port, "/slots", "", 1500, code, body) && code == 200) {
        J j; if (parse_json(body, j) && j.t == J::Arr) for (auto& e : j.a) {
            Slot s; if (auto* v = e.get("id")) s.id = (int)v->num(0); if (auto* v = e.get("is_processing")) s.busy = v->num(0) != 0;
            if (auto* v = e.get("n_ctx")) { s.n_ctx = v->num(); if (std::isnan(st.ctx_total)) st.ctx_total = 0; st.ctx_total += s.n_ctx; }
            if (auto* nt = e.get("next_token")) { if (auto* v = nt->get("n_decoded")) s.decoded = v->num(); if (auto* v = nt->get("n_remain")) s.remain = v->num(); }
            st.slots.push_back(s);
        }
    }
    if (!mhost.empty()) {
        std::string hdr = token.empty() ? "" : "X-MoeCache-Token: " + token + "\r\n";
        if (http_get(mhost, mport, "/api/monitor?since=0&log=0&layers=0&req=0&tail=1", hdr, 1500, code, body) && code == 200) {
            J j; if (parse_json(body, j) && j.t == J::Obj) { if (auto* v = j.get("schema")) { st.schema = v->num(); st.schema_note = ""; } else { st.schema = NAN; st.schema_note = "no schema field (older control page)"; } }
        } else { st.schema = NAN; st.schema_note = "control page not answering"; }
    }
    st.polled_at = t;
}

// ---------------------------------------------------------------- the Overview screen

using namespace ftxui;

static std::string fmt1(double v, const char* unit = "") { if (std::isnan(v)) return "n/a"; char b[64]; std::snprintf(b, sizeof b, v >= 100 ? "%.0f%s" : "%.1f%s", v, unit); return b; }

static Element overview(const State& st, const std::string& host, int port) {
    std::string hdr = "moe-cache top  " + host + ":" + std::to_string(port);
    if (!st.reachable) return vbox({text(hdr) | bold, separator(), text(st.error.empty() ? "connecting..." : st.error) | color(Color::Red), filler()}) | border;
    std::vector<double> hist(st.history.begin(), st.history.end());
    double peak = 1; for (double v : hist) peak = std::max(peak, v);
    auto spark = graph([hist, peak](int w, int h) {
        std::vector<int> out(w, 0); size_t n = hist.size();
        for (int x = 0; x < w; x++) { long idx = (long)n - w + x; if (idx >= 0 && idx < (long)n) out[x] = (int)std::lround(hist[idx] / peak * (h - 1)); }
        return out; }) | color(Color::Cyan) | size(HEIGHT, EQUAL, 4);
    double used = (!std::isnan(st.kv_ratio) && !std::isnan(st.ctx_total)) ? st.kv_ratio * st.ctx_total : NAN;
    Elements rows;
    rows.push_back(hbox({text(hdr) | bold, filler(), text(st.slots.empty() ? "" : std::to_string(st.slots.size()) + " slot" + (st.slots.size() == 1 ? "" : "s")) | dim}));
    rows.push_back(separator());
    rows.push_back(hbox({vbox({text("decode") | dim, text(fmt1(st.decode_tps, " tok/s")) | bold | color(Color::Green), text("avg " + fmt1(st.decode_avg, " tok/s")) | dim}) | size(WIDTH, EQUAL, 22),
                         vbox({text("prompt") | dim, text(fmt1(st.prompt_tps, " tok/s")) | bold | color(Color::Yellow), text("avg " + fmt1(st.prompt_avg, " tok/s")) | dim}) | size(WIDTH, EQUAL, 22),
                         vbox({text("requests") | dim, text(fmt1(st.processing) + " running") | bold, text(fmt1(st.deferred) + " waiting") | dim}) | flex}));
    rows.push_back(separator());
    rows.push_back(hbox({text("decode speed  ") | dim, text("peak " + fmt1(peak, " tok/s")) | dim}));
    rows.push_back(spark);
    rows.push_back(separator());
    double ratio = std::isnan(st.kv_ratio) ? 0 : std::min(1.0, std::max(0.0, st.kv_ratio));
    rows.push_back(hbox({text("context  ") | dim, gauge((float)ratio) | flex | color(ratio > 0.9 ? Color::Red : Color::Blue),
                         text("  " + (std::isnan(used) ? std::string("n/a") : std::to_string((long)std::lround(used))) + " / " + (std::isnan(st.ctx_total) ? std::string("n/a") : std::to_string((long)st.ctx_total)) + " tokens")}));
    std::vector<std::vector<std::string>> cells = {{" slot ", " state ", " context ", " decoded ", " left "}};
    auto pad = [](const std::string& x) { return " " + x + " "; };
    for (auto& s : st.slots) cells.push_back({pad(std::to_string(s.id)), pad(s.busy ? "working" : "idle"), pad(fmt1(s.n_ctx)), pad(fmt1(s.decoded)), pad(s.remain < 0 || std::isnan(s.remain) ? "-" : fmt1(s.remain))});
    if (st.slots.empty()) cells.push_back({pad("-"), pad("no /slots"), pad("-"), pad("-"), pad("-")});
    Table tb(cells); tb.SelectRow(0).Decorate(bold); tb.SelectAll().Border(LIGHT); tb.SelectAll().SeparatorVertical(LIGHT);
    rows.push_back(tb.Render());
    std::string foot = "q quit    schema " + (std::isnan(st.schema) ? std::string(st.schema_note.empty() ? "-" : st.schema_note) : std::to_string((long)st.schema)) + "    poll #" + std::to_string(st.polls);
    rows.push_back(filler());
    rows.push_back(text(foot) | dim);
    return vbox(rows) | border;
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    std::string host = "127.0.0.1", mhost, token; int port = 8080, mport = 0, interval_ms = 500, width = 100, height = 30; bool once = false, json = false, no_color = std::getenv("NO_COLOR") != nullptr;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto next = [&](const char* what) -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); } return argv[++i]; };
        if (a == "--version") { std::puts(VERSION); return 0; }
        else if (a == "--port") port = std::atoi(next("--port").c_str());
        else if (a == "--host") host = next("--host");
        else if (a == "--interval") interval_ms = std::max(100, std::atoi(next("--interval").c_str()));
        else if (a == "--once") once = true;
        else if (a == "--json") json = true;
        else if (a == "--no-color") no_color = true;
        else if (a == "--width") width = std::atoi(next("--width").c_str());
        else if (a == "--height") height = std::atoi(next("--height").c_str());
        else if (a == "--token") token = next("--token");
        else if (a == "--monitor-url") { std::string u = next("--monitor-url"); size_t p = u.find("://"); if (p != std::string::npos) u = u.substr(p + 3); size_t sl = u.find('/'); if (sl != std::string::npos) u = u.substr(0, sl);
            size_t c = u.rfind(':'); mhost = c == std::string::npos ? u : u.substr(0, c); mport = c == std::string::npos ? 80 : std::atoi(u.c_str() + c + 1); }
        else if (a == "--help" || a == "-h") { std::puts("moe-cache-top [--port 8080] [--host 127.0.0.1] [--interval MS] [--once [--width N --height N] [--no-color]] [--json] [--monitor-url http://127.0.0.1:8070 --token T] [--version]\n"
                                                         "Reads llama-server's /metrics and /slots (start it with --metrics). q quits."); return 0; }
        else { std::fprintf(stderr, "unknown option %s (try --help)\n", a.c_str()); return 2; }
    }
    State st; Prev pv;
    if (once || json) {                                           // two polls a moment apart, so the rates are real
        poll_once(host, port, mhost, mport, token, st, pv);
        if (st.reachable) { std::this_thread::sleep_for(std::chrono::milliseconds(std::min(interval_ms, 1000))); poll_once(host, port, mhost, mport, token, st, pv); }
        if (json) {
            std::printf("{\"reachable\":%s,\"decode_tps\":%s,\"prompt_tps\":%s,\"kv_cache_usage_ratio\":%s,\"slots\":%zu,\"schema\":%s}\n", st.reachable ? "true" : "false",
                        std::isnan(st.decode_tps) ? "null" : fmt1(st.decode_tps).c_str(), std::isnan(st.prompt_tps) ? "null" : fmt1(st.prompt_tps).c_str(),
                        std::isnan(st.kv_ratio) ? "null" : fmt1(st.kv_ratio).c_str(), st.slots.size(), std::isnan(st.schema) ? "null" : std::to_string((long)st.schema).c_str());
            return st.reachable ? 0 : 1;
        }
        Screen screen = Screen::Create(Dimension::Fixed(width), Dimension::Fixed(height));
        Render(screen, overview(st, host, port));
        std::string frame = screen.ToString();
        if (no_color) { std::string o; for (size_t i = 0; i < frame.size(); i++) { if (frame[i] == '\x1b' && i + 1 < frame.size() && frame[i + 1] == '[') { while (i < frame.size() && !std::isalpha((unsigned char)frame[i])) i++; } else o += frame[i]; } frame = o; }
        std::cout << frame << std::endl;
        return st.reachable ? 0 : 1;
    }
    std::mutex mu; std::atomic<bool> stop{false};
    auto screen = ScreenInteractive::Fullscreen();
    std::thread poller([&] {
        while (!stop) {
            { State copy; Prev p2; { std::lock_guard<std::mutex> l(mu); copy = st; p2 = pv; }
              poll_once(host, port, mhost, mport, token, copy, p2);
              { std::lock_guard<std::mutex> l(mu); st = copy; pv = p2; } }
            screen.PostEvent(Event::Custom);
            for (int w = 0; w < interval_ms && !stop; w += 50) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } });
    auto ui = Renderer([&] { std::lock_guard<std::mutex> l(mu); return overview(st, host, port); });
    ui = CatchEvent(ui, [&](Event e) { if (e == Event::Character('q') || e == Event::Escape) { screen.ExitLoopClosure()(); return true; } return false; });
    screen.Loop(ui);
    stop = true; poller.join();
    return 0;
}
