// 调试:复刻 live_gui 里的语音 spawn 逻辑,打印每一步结果
use std::process::{Command, Stdio};

fn main() {
    let exe_root = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("../..")));
    println!("exe_root = {:?}", exe_root);
    let resolve = |rel: &str| -> Option<String> {
        let abs = |p: &std::path::Path| {
            p.canonicalize()
                .ok()
                .map(|c| c.to_string_lossy().into_owned())
        };
        let cwd_p = std::path::Path::new(rel);
        if cwd_p.exists() {
            abs(cwd_p)
        } else {
            exe_root.as_ref().map(|r| r.join(rel)).and_then(|p| abs(&p))
        }
    };
    let model = resolve("stt/ggml-base.bin");
    let stt_bin = resolve("mac_stt");
    println!("model = {:?}", model);
    println!("stt_bin = {:?}", stt_bin);

    let server = model.as_ref().and_then(|m| {
        let args = [
            "-m", m.as_str(), "-l", "zh", "--host", "127.0.0.1", "--port", "8391",
        ];
        Command::new("whisper-server")
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .or_else(|_| {
                Command::new("/opt/homebrew/bin/whisper-server")
                    .args(args)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
            })
            .ok()
    });
    println!("server spawned = {}", server.is_some());

    let recorder = stt_bin.as_ref().and_then(|b| {
        println!("spawning recorder: {:?}", b);
        match Command::new(b)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => {
                println!("recorder pid = {}", c.id());
                Some(c)
            }
            Err(e) => {
                println!("recorder spawn ERR: {:?}", e);
                None
            }
        }
    });
    println!("recorder = {}", recorder.is_some());

    std::thread::sleep(std::time::Duration::from_secs(2));
    if let Some(mut r) = recorder {
        let _ = r.kill();
    }
    if let Some(mut s) = server {
        let _ = s.kill();
    }
}
