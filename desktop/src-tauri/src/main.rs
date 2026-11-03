// moe-cache desktop app: a thin Tauri 2 shell around the control page (bin/moe_cache_ui.py + ui/).
//
// What it does, and nothing more:
//   1. starts the control page as a child process (bundled Python on Windows, the system Python elsewhere) on a free port,
//   2. reads the page's address (with its per-run token) from the child's output,
//   3. points the window at that address,
//   4. gives a tray icon (Open / Start model / Stop model / Quit) and keeps one instance,
//   5. on Quit stops the model through the page's own API, then the child.
// Everything the user sees is the control page, so a change there lands in the browser page and the app alike.
// Design: docs/DESKTOP_APP_DESIGN.md.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, RunEvent, WindowEvent};

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
    ctl: Mutex<Option<Ctl>>,
    log: Mutex<Vec<String>>,
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

fn spawn_page(app: &AppHandle, state: &State, tx: mpsc::Sender<String>) -> Result<(), String> {
    let root = kit_root(app).ok_or("the moe-cache files (bin/moe_cache.py, ui/index.html) were not found; set MOE_CACHE_HOME")?;
    let py = python_cmd(app);
    let mut cmd = Command::new(&py);
    cmd.arg("-u")
        .arg(root.join("bin").join("moe_cache.py"))
        .args(["ui", "--ui-port", "0", "--no-browser"])
        .current_dir(&root)
        .env("PYTHONIOENCODING", "utf-8")
        .env("MOE_CACHE_APP", "1")
        .stdin(Stdio::piped()) // kept open: the page can later treat EOF as "the app is gone" and stop its model
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| format!("could not start {}: {e}", py.display()))?;
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
                if let Some(i) = line.find(MARKER) {
                    let _ = tx.send(line[i + MARKER.len()..].split_whitespace().next().unwrap_or("").to_string());
                }
            }
        });
    }
    Ok(())
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

fn kill_child(state: &State) {
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

fn fail(app: &AppHandle, state: &State, why: &str) {
    let tail = state.log.lock().unwrap().iter().rev().take(15).rev().cloned().collect::<Vec<_>>().join("\n");
    let text = format!("{why}\n\n{tail}");
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.eval(&format!("window.__moeFail && window.__moeFail({})", serde_json::to_string(&text).unwrap()));
    }
}

fn quit(app: &AppHandle, state: &State) {
    stop_model(state, 20);
    kill_child(state);
    app.exit(0);
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

            // tray
            let open = MenuItem::with_id(app, "open", "Open moe-cache", true, None::<&str>)?;
            let start = MenuItem::with_id(app, "start", "Start model", true, None::<&str>)?;
            let stop = MenuItem::with_id(app, "stop", "Stop model", true, None::<&str>)?;
            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let menu = Menu::with_items(app, &[&open, &sep, &start, &stop, &sep, &quit_i])?;
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

            // the control page as a child process; the window follows once it prints its address
            let (tx, rx) = mpsc::channel::<String>();
            if let Err(e) = spawn_page(&handle, &st_setup, tx) {
                fail(&handle, &st_setup, &e);
                return Ok(());
            }
            let st = st_setup.clone();
            std::thread::spawn(move || {
                match rx.recv_timeout(Duration::from_secs(60)) {
                    Ok(url) => match parse_url(&url) {
                        Some(ctl) => {
                            *st.ctl.lock().unwrap() = Some(ctl);
                            navigate(&handle, &url);
                        }
                        None => fail(&handle, &st, "the control page printed an address this app does not understand"),
                    },
                    Err(_) => fail(&handle, &st, "the control page did not report an address within 60 seconds"),
                }
                // crash handling (M2 in the design): if the child ends on its own, say so and offer a restart
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the moe-cache app");

    app.run(move |app_handle, ev| match ev {
        // closing the window must not end the app: the tray keeps it (and the model) running
        RunEvent::ExitRequested { api, code, .. } if code.is_none() => api.prevent_exit(),
        RunEvent::Exit => {
            stop_model(&st_run, 10);
            kill_child(&st_run);
            let _ = app_handle;
        }
        _ => {}
    });
}
