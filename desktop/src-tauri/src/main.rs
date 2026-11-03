// moe-cache desktop app: a thin Tauri 2 shell around the control page (bin/moe_cache_ui.py + ui/).
//
// What it does, and nothing more:
//   1. starts the control page as a child process (bundled Python on Windows, the system Python elsewhere) on a free port,
//   2. reads the page's address (with its per-run token) from the child's output,
//   3. points the window at that address,
//   4. gives a tray icon (Open / Start model / Stop model / Quit) and keeps one instance,
//   5. on Quit stops the model through the page's own API, then the child.
// The child is desktop/app_entry.py (staged as app_entry.py): the control page plus a stable address line and a parent-gone watch on its stdin, so
// the shell vanishing (crash, kill) still stops the model. Without it the shell falls back to `moe_cache.py ui` and the human address line.
// Everything the user sees is the control page, so a change there lands in the browser page and the app alike.
// Design: docs/DESKTOP_APP_DESIGN.md.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::webview::NewWindowResponse;
use tauri::{AppHandle, Manager, RunEvent, Url, WebviewUrl, WebviewWindowBuilder, WindowEvent};

const MARKER: &str = "control page at ";

/// Where the control page listens and the token that opens it.
#[derive(Clone)]
struct Ctl {
    port: u16,
    token: String,
}

#[derive(Default)]
struct Inner {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>, // kept open while the app runs; closing it tells app_entry.py to stop the model and exit
    ctl: Mutex<Option<Ctl>>,
    log: Mutex<Vec<String>>,
    splash: Mutex<Option<Url>>,
    quitting: AtomicBool,
    run: AtomicU32, // counts page starts: a watcher of an older run must not report
}

type State = Arc<Inner>;

// ---- finding the kit (bin/, ui/, planner/, models.json) and Python ----

fn has_kit(dir: &Path) -> bool {
    dir.join("bin").join("moe_cache.py").is_file() && dir.join("ui").join("index.html").is_file()
}

/// MOE_CACHE_HOME, else the bundled resources, else (development) a parent folder of the executable that is the repository.
fn kit_root(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MOE_CACHE_HOME") {
        let p = PathBuf::from(p);
        if has_kit(&p) {
            return Some(p);
        }
    }
    if let Ok(res) = app.path().resource_dir() {
        let p = res.join("moe-cache");
        if has_kit(&p) {
            return Some(p);
        }
    }
    let mut d = std::env::current_exe().ok()?;
    for _ in 0..8 {
        if !d.pop() {
            break;
        }
        if has_kit(&d) {
            return Some(d);
        }
    }
    None
}

/// MOE_CACHE_PYTHON, else the bundled interpreter (Windows kit), else python3 / python on the PATH.
fn python_cmd(app: &AppHandle) -> PathBuf {
    if let Ok(p) = std::env::var("MOE_CACHE_PYTHON") {
        return PathBuf::from(p);
    }
    if let Ok(res) = app.path().resource_dir() {
        for rel in ["python/python.exe", "python/bin/python3"] {
            let p = res.join(rel);
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

// ---- the child process ----

/// (script, arguments): app_entry.py when it is there (staged next to bin/, or desktop/ in a checkout), else the plain launcher.
fn page_command(root: &Path) -> (PathBuf, Vec<&'static str>) {
    for rel in [Path::new("app_entry.py"), Path::new("desktop").join("app_entry.py").as_path()] {
        let p = root.join(rel);
        if p.is_file() {
            return (p, vec!["--ui-port", "0"]);
        }
    }
    (root.join("bin").join("moe_cache.py"), vec!["ui", "--ui-port", "0", "--no-browser"])
}

fn spawn_page(app: &AppHandle, state: &State, tx: mpsc::Sender<String>) -> Result<(), String> {
    let root = kit_root(app).ok_or("the moe-cache files (bin/moe_cache.py, ui/index.html) were not found; set MOE_CACHE_HOME")?;
    let py = python_cmd(app);
    let (script, args) = page_command(&root);
    let mut cmd = Command::new(&py);
    cmd.arg("-u")
        .arg(script)
        .args(args)
        .current_dir(&root)
        .env("PYTHONIOENCODING", "utf-8")
        .env("MOE_CACHE_APP", "1")
        .stdin(Stdio::piped()) // kept open: app_entry.py treats end of input as "the app is gone" and stops its model
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| format!("could not start {}: {e}", py.display()))?;
    *state.stdin.lock().unwrap() = child.stdin.take();
    let out = child.stdout.take();
    let err = child.stderr.take();
    *state.child.lock().unwrap() = Some(child);
    for stream in [out.map(|s| Box::new(s) as Box<dyn Read + Send>), err.map(|s| Box::new(s) as Box<dyn Read + Send>)].into_iter().flatten() {
        let (tx, state) = (tx.clone(), state.clone());
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                {
                    let mut log = state.log.lock().unwrap();
                    log.push(line.clone());
                    if log.len() > 60 {
                        log.remove(0);
                    }
                }
                if let Some(url) = json_url(&line).or_else(|| marker_url(&line)) {
                    let _ = tx.send(url);
                }
            }
        });
    }
    Ok(())
}

/// The address in the stable line app_entry.py prints: {"moe_cache_ui": {"url": "http://127.0.0.1:PORT/?t=TOKEN", ...}}
fn json_url(line: &str) -> Option<String> {
    if !line.starts_with("{\"moe_cache_ui\"") {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    v["moe_cache_ui"]["url"].as_str().map(String::from)
}

/// The address in the line "moe-cache: control page at http://127.0.0.1:PORT/?t=TOKEN (this machine ...)".
fn marker_url(line: &str) -> Option<String> {
    let i = line.find(MARKER)?;
    let url = line[i + MARKER.len()..].split_whitespace().next()?;
    Some(url.to_string())
}

/// "http://127.0.0.1:PORT/?t=TOKEN" -> Ctl
fn parse_url(url: &str) -> Option<Ctl> {
    let rest = url.strip_prefix("http://127.0.0.1:")?;
    let (port, tail) = rest.split_once('/')?;
    let token = tail.split_once("t=")?.1.to_string();
    Some(Ctl { port: port.parse().ok()?, token })
}

// ---- talking to the page's API (plain HTTP on 127.0.0.1, token in the header; no extra crates) ----

fn api(ctl: &Ctl, method: &str, path: &str, body: &str, secs: u64) -> Option<String> {
    let addr: SocketAddr = ([127, 0, 0, 1], ctl.port).into();
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(secs)));
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nX-MoeCache-Token: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        ctl.port,
        ctl.token,
        body.len()
    );
    s.write_all(req.as_bytes()).ok()?;
    let mut resp = String::new();
    s.read_to_string(&mut resp).ok()?;
    Some(resp.split_once("\r\n\r\n").map(|x| x.1).unwrap_or("").to_string())
}

fn ctl_of(state: &State) -> Option<Ctl> {
    state.ctl.lock().unwrap().clone()
}

fn model_state(ctl: &Ctl) -> Option<String> {
    let b = api(ctl, "GET", "/api/status", "", 3)?;
    serde_json::from_str::<serde_json::Value>(&b).ok()?["state"].as_str().map(String::from)
}

/// Stop the model and wait (at most `secs`) until the page says it is no longer starting / ready / stopping.
fn stop_model(state: &State, secs: u64) {
    let Some(ctl) = ctl_of(state) else { return };
    let _ = api(&ctl, "POST", "/api/stop", "{}", 5);
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(secs) {
        match model_state(&ctl).as_deref() {
            Some("starting") | Some("ready") | Some("stopping") => std::thread::sleep(Duration::from_millis(300)),
            _ => break,
        }
    }
}

/// End the page process: close its stdin (app_entry.py stops the model and exits), wait up to `secs`, then kill what is left.
fn end_child(state: &State, secs: u64) {
    drop(state.stdin.lock().unwrap().take());
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(secs) {
        match state.child.lock().unwrap().as_mut().map(|c| c.try_wait()) {
            None | Some(Ok(Some(_))) | Some(Err(_)) => break,
            Some(Ok(None)) => {}
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    if let Some(mut c) = state.child.lock().unwrap().take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

// ---- window ----

fn show_main(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

fn navigate(app: &AppHandle, url: &str) {
    if let Some(w) = app.get_webview_window("main") {
        let js = format!("location.replace({})", serde_json::to_string(url).unwrap());
        let _ = w.eval(&js);
    }
}

/// A log line without the run's token ("?t=..." in the address lines): the text goes on screen, into screenshots and bug reports.
fn redact(line: &str) -> String {
    match line.find("t=") {
        Some(i) if line[..i].ends_with('?') => {
            let end = line[i..].find(|c: char| c.is_whitespace() || c == '"').map_or(line.len(), |n| i + n);
            format!("{}t=…{}", &line[..i], &line[end..])
        }
        _ => line.to_string(),
    }
}

/// Show `text` on the splash page (the page may still be loading: say it three times).
fn fail(app: &AppHandle, state: &State, why: &str) {
    fail_titled(app, state, why, "moe-cache could not start its control page.")
}

fn fail_titled(app: &AppHandle, state: &State, why: &str, title: &str) {
    let tail = state.log.lock().unwrap().iter().rev().take(15).rev().map(|l| redact(l)).collect::<Vec<_>>().join("\n");
    let js = format!("window.__moeFail && window.__moeFail({}, {})", serde_json::to_string(&format!("{why}\n\n{tail}")).unwrap(), serde_json::to_string(title).unwrap());
    let app = app.clone();
    std::thread::spawn(move || {
        for ms in [100, 700, 1500] {
            std::thread::sleep(Duration::from_millis(ms));
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.eval(&js);
            }
        }
    });
}

/// Back to the splash page (when the control page process has ended).
fn show_splash(app: &AppHandle, state: &State) {
    let splash = state.splash.lock().unwrap().clone();
    if let (Some(w), Some(u)) = (app.get_webview_window("main"), splash) {
        let _ = w.navigate(u);
    }
}

fn quit(app: &AppHandle, state: &State) {
    state.quitting.store(true, Ordering::SeqCst);
    stop_model(state, 20); // through the page's API: works with any page process
    end_child(state, 20);
    app.exit(0);
}

// ---- links: the window shows the control page only; every other address opens in the system browser ----

/// May the window itself go to this address? Its own pages (the splash and the control page on this run's port) only.
fn is_internal(url: &Url, ctl_port: Option<u16>) -> bool {
    match url.scheme() {
        "tauri" | "about" | "data" | "blob" => true,
        "http" | "https" => {
            let host = url.host_str().unwrap_or("");
            host == "tauri.localhost" || (url.scheme() == "http" && host == "127.0.0.1" && ctl_port.is_some() && url.port() == ctl_port)
        }
        _ => false,
    }
}

/// Only web addresses are handed to the system: no file:, no custom schemes.
fn safe_external(url: &Url) -> bool {
    matches!(url.scheme(), "http" | "https") && url.host_str().is_some()
}

fn open_external(url: &Url) {
    if !safe_external(url) {
        return;
    }
    let u = url.as_str();
    #[cfg(windows)]
    let mut cmd = {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("rundll32");
        c.args(["url.dll,FileProtocolHandler", u]).creation_flags(0x0800_0000);
        c
    };
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = Command::new("open");
        c.arg(u);
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = {
        let mut c = Command::new("xdg-open");
        c.arg(u);
        c
    };
    let _ = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
}

// ---- starting the control page, and noticing when it ends ----

/// Start the page process; the window follows once it reports its address. A watcher tells the user when the process ends on its own.
fn start_page(app: &AppHandle, state: &State) {
    let run = state.run.fetch_add(1, Ordering::SeqCst) + 1;
    *state.ctl.lock().unwrap() = None;
    state.log.lock().unwrap().clear();
    let (tx, rx) = mpsc::channel::<String>();
    if let Err(e) = spawn_page(app, state, tx) {
        fail(app, state, &e);
        return;
    }
    let (app, st) = (app.clone(), state.clone());
    std::thread::spawn(move || {
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(url) => match parse_url(&url) {
                Some(ctl) => {
                    *st.ctl.lock().unwrap() = Some(ctl);
                    navigate(&app, &url);
                }
                None => return fail(&app, &st, "the control page printed an address this app does not understand"),
            },
            Err(_) => return fail(&app, &st, "the control page did not report an address within 60 seconds"),
        }
        loop {
            std::thread::sleep(Duration::from_secs(1));
            if st.quitting.load(Ordering::SeqCst) || st.run.load(Ordering::SeqCst) != run {
                return;
            }
            let ended = match st.child.lock().unwrap().as_mut().map(|c| c.try_wait()) {
                Some(Ok(Some(status))) => Some(status.to_string()),
                None => Some("no process".to_string()),
                _ => None,
            };
            if let Some(how) = ended {
                *st.ctl.lock().unwrap() = None;
                show_splash(&app, &st);
                return fail_titled(&app, &st, &format!("The control page process ended ({how}). Use the tray icon: Restart control page. A model it started has been stopped by it, or shows under Other servers after a restart."), "moe-cache's control page stopped.");
            }
        }
    });
}

fn restart_page(app: &AppHandle, state: &State) {
    stop_model(state, 10);
    end_child(state, 15);
    show_splash(app, state);
    start_page(app, state);
}

fn main() {
    let state: State = Arc::new(Inner::default());
    let st_setup = state.clone();
    let st_run = state.clone();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| show_main(app)))
        .on_window_event(|w, ev| {
            // the close button hides to the tray; Quit in the tray menu ends the app
            if let WindowEvent::CloseRequested { api, .. } = ev {
                api.prevent_close();
                let _ = w.hide();
            }
        })
        .setup(move |app| {
            let handle = app.handle().clone();

            // the window is built here (not in tauri.conf.json) so it can refuse to leave the control page: other addresses open in the system browser
            let (st_nav, st_new) = (st_setup.clone(), st_setup.clone());
            let win = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("moe-cache")
                .inner_size(1280.0, 820.0)
                .min_inner_size(480.0, 520.0)
                .on_navigation(move |url| {
                    if is_internal(url, ctl_of(&st_nav).map(|c| c.port)) {
                        return true;
                    }
                    open_external(url);
                    false
                })
                .on_new_window(move |url, _| {
                    // target=_blank and window.open: llama-server's own page on our port too, which the system browser shows better than a second app window
                    let _ = &st_new;
                    open_external(&url);
                    NewWindowResponse::Deny
                })
                .build()?;
            *st_setup.splash.lock().unwrap() = win.url().ok();

            // tray
            let open = MenuItem::with_id(app, "open", "Open moe-cache", true, None::<&str>)?;
            let start = MenuItem::with_id(app, "start", "Start model", true, None::<&str>)?;
            let stop = MenuItem::with_id(app, "stop", "Stop model", true, None::<&str>)?;
            let restart = MenuItem::with_id(app, "restart", "Restart control page", true, None::<&str>)?;
            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let menu = Menu::with_items(app, &[&open, &sep, &start, &stop, &restart, &sep, &quit_i])?;
            let st_menu = st_setup.clone();
            TrayIconBuilder::new()
                .icon(app.default_window_icon().ok_or("no window icon")?.clone())
                .tooltip("moe-cache")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(move |app, ev| match ev.id().as_ref() {
                    "open" => show_main(app),
                    "start" => {
                        // the page's last settings (model, context, mode); an error shows in the page
                        show_main(app);
                        if let Some(ctl) = ctl_of(&st_menu) {
                            std::thread::spawn(move || {
                                let _ = api(&ctl, "POST", "/api/start", "{}", 10);
                            });
                        }
                    }
                    "stop" => {
                        let s = st_menu.clone();
                        std::thread::spawn(move || stop_model(&s, 20));
                    }
                    "restart" => {
                        let (a, s) = (app.clone(), st_menu.clone());
                        std::thread::spawn(move || restart_page(&a, &s));
                    }
                    "quit" => {
                        let (a, s) = (app.clone(), st_menu.clone());
                        std::thread::spawn(move || quit(&a, &s));
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, ev| {
                    if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = ev {
                        show_main(tray.app_handle());
                    }
                })
                .build(app)?;

            start_page(&handle, &st_setup);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the moe-cache app");

    app.run(move |app_handle, ev| match ev {
        // closing the window must not end the app: the tray keeps it (and the model) running
        RunEvent::ExitRequested { api, code, .. } if code.is_none() => api.prevent_exit(),
        RunEvent::Exit => {
            st_run.quitting.store(true, Ordering::SeqCst);
            stop_model(&st_run, 10);
            end_child(&st_run, 10);
            let _ = app_handle;
        }
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn finds_the_address_in_the_launchers_line() {
        let line = "moe-cache: control page at http://127.0.0.1:41233/?t=abc123 (this machine and this user only; Ctrl-C stops it)";
        assert_eq!(marker_url(line).as_deref(), Some("http://127.0.0.1:41233/?t=abc123"));
        assert_eq!(marker_url("moe-cache: something else"), None);
        assert_eq!(marker_url("control page at "), None);
    }

    #[test]
    fn parses_port_and_token() {
        let c = parse_url("http://127.0.0.1:41233/?t=abc123").unwrap();
        assert_eq!((c.port, c.token.as_str()), (41233, "abc123"));
        assert!(parse_url("http://example.com:80/?t=x").is_none());
        assert!(parse_url("http://127.0.0.1:99999/?t=x").is_none());
        assert!(parse_url("http://127.0.0.1:8070/").is_none());
    }

    #[test]
    fn api_sends_the_token_header_and_returns_the_body() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let t = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = [0u8; 2048];
            let n = s.read(&mut buf).unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = r#"{"state":"ready"}"#;
            write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            req
        });
        let ctl = Ctl { port, token: "tok".into() };
        let got = api(&ctl, "GET", "/api/status", "", 3).unwrap();
        let req = t.join().unwrap();
        assert!(req.starts_with("GET /api/status HTTP/1.1\r\n"));
        assert!(req.contains(&format!("Host: 127.0.0.1:{port}\r\n")));
        assert!(req.contains("X-MoeCache-Token: tok\r\n"));
        assert_eq!(got, r#"{"state":"ready"}"#);
    }

    #[test]
    fn api_returns_none_when_nothing_listens() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        assert!(api(&Ctl { port, token: "t".into() }, "GET", "/", "", 1).is_none());
    }

    #[test]
    fn the_token_never_reaches_the_screen() {
        assert_eq!(redact("moe-cache: control page at http://127.0.0.1:5000/?t=abc_DEF-1 (this machine"), "moe-cache: control page at http://127.0.0.1:5000/?t=… (this machine");
        assert_eq!(redact(r#"{"moe_cache_ui": {"url": "http://127.0.0.1:5000/?t=abc", "port": 5000}}"#), r#"{"moe_cache_ui": {"url": "http://127.0.0.1:5000/?t=…", "port": 5000}}"#);
        assert_eq!(redact("plain line, nothing to hide"), "plain line, nothing to hide");
    }

    #[test]
    fn reads_the_stable_json_line() {
        let l = r#"{"moe_cache_ui": {"url": "http://127.0.0.1:5000/?t=abc", "port": 5000, "pid": 7}}"#;
        assert_eq!(json_url(l).as_deref(), Some("http://127.0.0.1:5000/?t=abc"));
        assert_eq!(json_url("moe-cache: control page at http://127.0.0.1:5000/?t=abc"), None);
        assert_eq!(json_url(r#"{"moe_cache_ui": "broken""#), None);
    }

    #[test]
    fn the_window_stays_on_its_own_pages() {
        let u = |s: &str| Url::parse(s).unwrap();
        assert!(is_internal(&u("tauri://localhost/"), None));
        assert!(is_internal(&u("http://tauri.localhost/index.html"), None));
        assert!(is_internal(&u("http://127.0.0.1:5000/?t=abc"), Some(5000)));
        assert!(!is_internal(&u("http://127.0.0.1:5000/"), None));
        assert!(!is_internal(&u("http://127.0.0.1:8080/"), Some(5000))); // llama-server's own page
        assert!(!is_internal(&u("https://huggingface.co/x"), Some(5000)));
        assert!(!is_internal(&u("file:///etc/passwd"), Some(5000)));
    }

    #[test]
    fn only_web_addresses_go_to_the_system() {
        let u = |s: &str| Url::parse(s).unwrap();
        assert!(safe_external(&u("https://github.com/ggml-org/llama.cpp/releases")));
        assert!(safe_external(&u("http://127.0.0.1:8080/")));
        assert!(!safe_external(&u("file:///C:/Windows/System32/calc.exe")));
        assert!(!safe_external(&u("ms-msdt:/id")));
        assert!(!safe_external(&u("javascript:alert(1)")));
    }

    #[test]
    fn picks_app_entry_when_present() {
        let d = std::env::temp_dir().join(format!("moe-desk-test-{}", std::process::id()));
        std::fs::create_dir_all(d.join("bin")).unwrap();
        let (p, args) = page_command(&d);
        assert!(p.ends_with("bin/moe_cache.py") || p.ends_with("bin\\moe_cache.py"));
        assert_eq!(args.first(), Some(&"ui"));
        std::fs::write(d.join("app_entry.py"), "").unwrap();
        let (p, args) = page_command(&d);
        assert!(p.ends_with("app_entry.py"));
        assert_eq!(args, vec!["--ui-port", "0"]);
        let _ = std::fs::remove_dir_all(&d);
    }
}
