use opencv::{core, highgui, imgcodecs, imgproc, prelude::*};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use tnt_comput::net::{decode_ws_message, tunnel, Tracker, TunnelEvent};
use tnt_comput::proto::GameEvent;

/// 隧道回传的实时战斗状态。世界坐标系:x 向右,y 向下。
#[derive(Default, Clone)]
struct NetHud {
    /// 每收到一次 Init 自增;渲染线程用它检测"新对局"并重置手动标记。
    battle_id: u64,
    /// 最近一次对局的地图世界尺寸(单位=服务器世界坐标)
    map_w: i64,
    map_h: i64,
    /// 我方 uid:名字匹配(PlayerDataNotify / INIT 名字)或首炮关联得到
    my_id: Option<u64>,
    pos: HashMap<u64, (i64, i64)>,
    names: HashMap<u64, String>,
    round: u64,
    /// 本回合出手者(order[0])
    active: Option<u64>,
    /// 回合风原始字段(解码尚未确认,仅诊断)
    wind_seed: i64,
    /// 局内风密钥候选(每次报风反解一个)。生效值=众数,防个别抖动回合污染;
    /// wind10 = (|raw|&0xff) ^ C ^ round
    wind_key_votes: Vec<i64>,
    /// 持久化的 基数→密钥 映射(raw>>8 → C);同房连打基数不变密钥也不变
    key_by_base: HashMap<i64, i64>,
    /// 最近一次确认过的密钥(跨进程重启的兜底先验)
    last_key: Option<i64>,
    /// 每个 WindSeed 自增,渲染线程据此判定"新回合"并应用自动风
    turn_seq: u64,
    /// 本回合解码出的风速大小(×10,无符号);None=密钥未知
    auto_wind10: Option<i32>,
    /// 最近一发实际炮弹: (射手id, 角度)
    last_fire: Option<(u64, Option<f64>)>,
    /// 最近一条客户端 FireReq: (角度, 力度) — 一定是我们的
    last_fire_req: Option<(u64, f64)>,
    /// 上一条 FireReq 还没等到对应 Fire,用它反推 my_id
    pending_firereq: bool,
    /// 确定不是我们的 uid(对方开炮时记录,用于中途加入时反推 my_id)
    not_mine: std::collections::HashSet<u64>,
    /// 多敌模式:用户在小地图上点选的目标 uid;None=自动取第一个
    target_id: Option<u64>,
    /// 当前活着的隧道连接数(0=隧道断开)
    conns: i32,
    /// 最近一条记录的游戏端→本地投递延迟(排队导致的滞后)
    last_lag_ms: i64,
}

/// 追加一行配对样本到 wind_pairs.csv(运行时 cwd=项目目录)。
/// kind=SEED 记录每回合 raw;kind=VOICE/MANUAL 记录用户标定的屏幕风速。
/// 攒够 (raw, 实际风速) 配对后用于破解字段编码。
fn log_wind_pair(
    kind: &str,
    battle: u64,
    round: u64,
    active: Option<u64>,
    raw: i64,
    wind: Option<f64>,
    applied: Option<f64>,
) {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("wind_pairs.csv")
    {
        use std::io::Write;
        if f.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            let _ = writeln!(f, "ms,kind,battle,round,active,raw,wind,applied");
        }
        let _ = writeln!(
            f,
            "{},{},{},{},{},{},{},{}",
            ms,
            kind,
            battle,
            round,
            active.map(|a| a.to_string()).unwrap_or_default(),
            raw,
            wind.map(|w| format!("{w:.1}")).unwrap_or_default(),
            applied.map(|w| format!("{w:.1}")).unwrap_or_default()
        );
    }
}

/// 相对风向提示: +1=顺风(箭头与投掷方向一致) -1=逆风。
/// raw 符号=世界风向(已验证);投掷方向=敌人相对我方 x 方位。
/// 信息不足(无种子/无坐标/站位重合)时返回 None。
/// 一次报风/纠错 = 一票 + 更新持久化映射并存盘
fn push_wind_vote(st: &mut NetHud, c: i64) {
    st.wind_key_votes.push(c);
    let base = st.wind_seed.abs() >> 8;
    if base > 0 {
        st.key_by_base.insert(base, c);
    }
    st.last_key = Some(c);
    save_wind_keys(st);
}

fn save_wind_keys(st: &NetHud) {
    let mut s = String::new();
    if let Some(l) = st.last_key {
        s.push_str(&format!("last {l}\n"));
    }
    for (b, c) in &st.key_by_base {
        s.push_str(&format!("base {b} {c}\n"));
    }
    let _ = std::fs::write("wind_key.txt", s);
}

/// 启动时载入上次会话确认的密钥 → 重启后第一回合即可预测
fn load_wind_keys(st: &mut NetHud) {
    if let Ok(txt) = std::fs::read_to_string("wind_key.txt") {
        for line in txt.lines() {
            let p: Vec<&str> = line.split_whitespace().collect();
            match p.as_slice() {
                ["last", v] => {
                    // 只作兜底猜测,不占票:避免上局旧密钥压制本局真实密钥
                    st.last_key = v.parse().ok();
                }
                ["base", b, c] => {
                    if let (Ok(b), Ok(c)) = (b.parse::<i64>(), c.parse::<i64>()) {
                        st.key_by_base.insert(b, c);
                    }
                }
                _ => {}
            }
        }
    }
}

/// 密钥众数(平票取最新):单次抖动/误报不污染已确认的密钥,连续重报可翻票
fn voted_key(votes: &[i64]) -> Option<i64> {
    let mut best: Option<(i64, usize, usize)> = None; // (key, count, last_idx)
    for (i, &v) in votes.iter().enumerate() {
        let n = votes.iter().filter(|&&x| x == v).count();
        let better = match best {
            None => true,
            Some((_, bn, bi)) => n > bn || (n == bn && i > bi),
        };
        if better {
            best = Some((v, n, i));
        }
    }
    best.map(|(v, _, _)| v)
}

fn wind_dir_hint(net: &NetHud) -> Option<i32> {
    if net.wind_seed == 0 {
        return None;
    }
    let my = net.my_id?;
    let (mx, _) = *net.pos.get(&my)?;
    let (ex, _) = net
        .target_id
        .and_then(|t| net.pos.get(&t))
        .or_else(|| {
            net.pos
                .iter()
                .find(|(id, _)| **id != my && !net.not_mine.contains(*id))
                .map(|(_, p)| p)
        })?;
    let aim = (ex - mx).signum() as i32;
    if aim == 0 {
        return None;
    }
    let arrow = net.wind_seed.signum() as i32;
    Some(arrow * aim)
}

/// 输入按"风速大小"解释,符号=顺/逆风自动附加(hint 缺失时原样返回)。
fn apply_auto_wind_sign(net: &NetHud, w: f64, flip_ui: bool) -> f64 {
    let flip = std::env::var("TNT_WIND_SIGN")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(1)
        * if flip_ui { -1 } else { 1 };
    match wind_dir_hint(net) {
        Some(h) => w.abs() * (h * flip) as f64,
        None => w,
    }
}

fn apply_net_event(st: &mut NetHud, ev: &GameEvent, my_name: &str) {
    match ev {
        GameEvent::Init {
            map_w,
            map_h,
            players,
        } => {
            let ps: Vec<String> = players
                .iter()
                .map(|p| format!("{} \"{}\"@({},{})", p.id, p.name, p.x, p.y))
                .collect();
            println!("📡 NET INIT map={}x{} players=[{}]", map_w, map_h, ps.join(", "));
            st.battle_id += 1;
            st.map_w = *map_w as i64;
            st.map_h = *map_h as i64;
            st.pos.clear();
            st.names.clear();
            st.round = 0;
            st.active = None;
            st.wind_seed = 0;
            // 新局清空会话票;先验预测由 key_by_base/last_key 兜底,不占票——
            // 同房连打基数不变→命中历史映射自动预测;基数变了→用户报一次即确立
            st.wind_key_votes.clear();
            st.auto_wind10 = None;
            st.last_fire = None;
            st.last_fire_req = None;
            st.pending_firereq = false;
            st.not_mine.clear();
            st.target_id = None;
            st.my_id = None;
            for p in players {
                st.pos.insert(p.id, (p.x, p.y));
                st.names.insert(p.id, p.name.clone());
                if p.name == my_name {
                    st.my_id = Some(p.id);
                }
            }
        }
        GameEvent::WindSeed {
            round,
            order,
            seed,
            ..
        } => {
            st.round = *round;
            st.wind_seed = *seed;
            st.active = order.first().copied();
            st.turn_seq += 1;
            // 密钥来源: 本会话投票众数 → 同基数历史映射(持久化)。
            // 跨局旧密钥不兜底——不同房间C不同,乱猜只会填垃圾值。
            let base = (*seed).abs() >> 8;
            let c = voted_key(&st.wind_key_votes)
                .or_else(|| st.key_by_base.get(&base).copied());
            // 未确认时(本会话无票)过滤离谱值(>15风),等用户报一次校准
            st.auto_wind10 = c
                .map(|c| (((*seed).abs() & 0xff) as i64 ^ c ^ (*round as i64)) as i32)
                .filter(|&m| !st.wind_key_votes.is_empty() || m <= 150);
            println!(
                "📡 NET WIND raw={seed} active={:?} auto={:?}",
                st.active, st.auto_wind10
            );
            log_wind_pair(
                "SEED",
                st.battle_id,
                st.round,
                st.active,
                st.wind_seed,
                None,
                st.auto_wind10.map(|w| w as f64 / 10.0),
            );
        }
        GameEvent::PlayerUpdate { id, pos, .. } => {
            if let Some(p) = pos {
                st.pos.insert(*id, *p);
            }
        }
        GameEvent::Move { id, x, y } => {
            st.pos.insert(*id, (*x, *y));
        }
        GameEvent::MoveReq { x } => {
            if let Some(id) = st.my_id {
                if let Some((_, y)) = st.pos.get(&id).copied() {
                    st.pos.insert(id, (*x, y));
                }
            }
        }
        GameEvent::Flight { id, path } => {
            if st.pending_firereq {
                if st.my_id.is_none() {
                    st.my_id = Some(*id);
                }
                st.pending_firereq = false;
            }
            if let Some(&destination) = path.last() {
                st.pos.insert(*id, destination);
            }
        }
        GameEvent::Fire { id, x, y, angle, .. } => {
            st.pos.insert(*id, (*x, *y));
            // FireReq(c2s) 之后紧跟的 Fire 一定是我们的 → 反推 my_id
            if st.pending_firereq {
                if st.my_id.is_none() {
                    st.my_id = Some(*id);
                }
                st.pending_firereq = false;
            } else if st.my_id.is_none() {
                st.not_mine.insert(*id);
            }
            // 双人局:另一个不是我们的就是我
            if st.my_id.is_none() && st.pos.len() == 2 {
                if let Some((&k, _)) = st.pos.iter().find(|(k, _)| !st.not_mine.contains(*k)) {
                    st.my_id = Some(k);
                }
            }
            st.last_fire = Some((*id, *angle));
        }
        GameEvent::FireReq { angle, power, .. } => {
            st.pending_firereq = true;
            st.last_fire_req = Some((*angle, *power));
            println!("📡 NET FIREREQ angle={angle} power={power:.2}");
        }
        GameEvent::Other { name, seq, .. } => {
            // PlayerDataNotify 的 seq 字段实测就是我方 uid
            if name == "PlayerDataNotify" && st.my_id.is_none() {
                st.my_id = *seq;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EditMode {
    None,
    P1,
    E1,
    DrawRuler1,
    DrawRuler2,
}

#[derive(Debug, Clone, Copy)]
struct AppState {
    edit_mode: EditMode,
    manual_p1: Option<core::Point>,
    manual_e1: Option<core::Point>,
    manual_cam_rect: Option<core::Rect>,
    drag_start: Option<core::Point>,
    current_angle: f64,
    wind: f64,
    locked_px_per_unit: Option<f64>,
    /// 大地图模式:整张地图的宽度代表多少游戏距离(12/16/18/20/22)
    map_units: f64,
    /// 当前是否大地图模式(运行时可切换)
    big_map: bool,
    /// 显示缩放系数:canvas 像素 / 截图像素,切换模式时重算
    disp_scale: f64,
    /// 当前地图源尺寸(截图像素)
    src_w: i32,
    src_h: i32,
    auto_angle: bool,
    is_fixed_angle: bool,
    /// F 键切换:自动风向附加时强制反向(极端大风反向抛等手动修正场景)
    wind_flip: bool,
    /// 当前风速来自网络密钥自动解码(显示 [N] 标记)
    wind_net: bool,
    exit_requested: bool,
    switch_requested: bool,
    /// 按钮防误触:记录按下位置,松开时确认仍在同一按钮内才生效
    btn_press: Option<(i32, i32)>,
}

use tnt_comput::physics::*;

fn compute_fixed_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64) -> Option<(f64, f64)> {
    let mut eff_angle = angle_deg;
    if eff_angle > 90.0 {
        eff_angle = 180.0 - eff_angle;
    }

    let wind_power = wind; // 用户输入的是相对风力（正=顺风，负=逆风），物理引擎统一按向右打计算，所以直接传入即可
    let dist = dx_units.abs();

    // 调用全新的底层物理引擎，同时传入高低差 dy_units
    let final_power = power_for_angle(eff_angle, dist, dy_units, wind_power)?;
    Some((final_power, angle_deg))
}

fn compute_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64) -> Option<(f64, f64)> {
    let mut eff_angle = angle_deg;
    if eff_angle > 90.0 {
        eff_angle = 180.0 - eff_angle;
    }

    let is_reverse = dx_units < 0.0;
    let wind_power = wind; // 用户输入的是相对风力（正=顺风，负=逆风）
    let dist = dx_units.abs();

    // 使用新的 power_for_angle 和 calc_angle 传递 dy_units
    let base_power = power_for_angle(eff_angle, dist, dy_units, 0.0)?;
    let mut final_angle = calc_angle(dist, dy_units, base_power, wind_power, eff_angle);

    final_angle = final_angle.clamp(15.0, 89.0);

    if is_reverse {
        final_angle = 180.0 - final_angle;
    }

    Some((base_power, final_angle))
}

fn draw_btn(
    canvas: &mut core::Mat,
    rect: core::Rect,
    label: &str,
    is_active: bool,
) -> opencv::Result<()> {
    let color = if is_active {
        core::Scalar::new(0.0, 200.0, 255.0, 0.0)
    } else {
        core::Scalar::new(80.0, 80.0, 80.0, 0.0)
    };
    let text_color = if is_active {
        core::Scalar::new(0.0, 0.0, 0.0, 0.0)
    } else {
        core::Scalar::new(255.0, 255.0, 255.0, 0.0)
    };

    imgproc::rectangle(canvas, rect, color, -1, imgproc::LINE_8, 0)?;
    imgproc::rectangle(
        canvas,
        rect,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    let mut baseline = 0;
    let size = imgproc::get_text_size(label, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, 1, &mut baseline)?;
    let text_x = rect.x + (rect.width - size.width) / 2;
    let text_y = rect.y + (rect.height + size.height) / 2;

    imgproc::put_text(
        canvas,
        label,
        core::Point::new(text_x, text_y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

fn is_inside(x: i32, y: i32, rect: core::Rect) -> bool {
    x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height
}

fn allow_plain_manual_mark(net_active: bool) -> bool {
    !net_active
}

/// SenseVoice 常驻识别服务子进程(stdin 喂 wav 路径,stdout 回 TEXT:)
#[cfg(target_os = "macos")]
struct SttServer {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: std::process::ChildStdout,
}

#[cfg(target_os = "macos")]
fn find_stt_rec() -> Option<String> {
    // stt_rec 在可执行文件同级或项目根
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let cands = [
        "./stt_rec".to_string(),
        exe_dir
            .as_ref()
            .map(|d| d.join("../../stt_rec").to_string_lossy().into_owned())
            .unwrap_or_default(),
    ];
    for b in cands {
        if b.is_empty() || !std::path::Path::new(&b).exists() {
            continue;
        }
        return Some(b);
    }
    None
}

#[cfg(target_os = "macos")]
fn spawn_stt_server(model: &str, tokens: &str) -> Option<SttServer> {
    let b = find_stt_rec()?;
    if let Ok(mut c) = Command::new(&b)
        .arg(model)
        .arg(tokens)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        if let (Some(si), Some(so)) = (c.stdin.take(), c.stdout.take()) {
            return Some(SttServer {
                child: c,
                stdin: si,
                stdout: so,
            });
        }
    }
    None
}

/// 常听模式: stt_rec vad 的 stdin 直接吃 mac_stt stream 的 PCM stdout
#[cfg(target_os = "macos")]
fn spawn_stt_vad(model: &str, tokens: &str, vad: &str, audio_in: std::process::ChildStdout) -> Option<(Child, std::process::ChildStdout)> {
    let b = find_stt_rec()?;
    Command::new(&b)
        .arg("vad")
        .arg(model)
        .arg(tokens)
        .arg(vad)
        .stdin(Stdio::from(audio_in))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()
        .and_then(|mut c| c.stdout.take().map(|so| (c, so)))
}

/// whisper 兜底:SenseVoice 对快语速/口音的短口令吐不出可解析结果时,
/// 用同一条 wav 跑 whisper-cli(large-v3-turbo)。只在按住说话模式启用——
/// 常听模式下环境音每几秒一条,全部兜底会把 CPU 打满。
/// 返回清理后的文本(剥掉 [BLANK_AUDIO]/时间戳行),识别不出返回 None。
#[cfg(target_os = "macos")]
fn whisper_fallback(wav: &str, model: &str) -> Option<String> {
    let bin = if std::path::Path::new("/opt/homebrew/bin/whisper-cli").exists() {
        "/opt/homebrew/bin/whisper-cli"
    } else {
        "whisper-cli"
    };
    let out = Command::new(bin)
        .args(["-m", model, "-l", "zh", "-nt", "-f", wav])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let cleaned: String = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.contains("BLANK_AUDIO") && !l.starts_with('['))
        .collect();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// whisper-server 常驻:模型只加载一次,之后每条 wav POST 一次 ~1.3s。
/// 等端口就绪(模型加载 1-3s),超时杀掉并返回 None 让上层回退 SenseVoice。
#[cfg(target_os = "macos")]
fn spawn_whisper_server(model: &str, port: u16) -> Option<Child> {
    let bin = if std::path::Path::new("/opt/homebrew/bin/whisper-server").exists() {
        "/opt/homebrew/bin/whisper-server"
    } else {
        "whisper-server"
    };
    let mut c = Command::new(bin)
        .args([
            "-m",
            model,
            "-l",
            "zh",
            "--port",
            &port.to_string(),
            "--host",
            "127.0.0.1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let addr: std::net::SocketAddr = format!("127.0.0.1:{}", port).parse().ok()?;
    for _ in 0..40 {
        if std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(200)).is_ok()
        {
            return Some(c);
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    let _ = c.kill();
    None
}

/// 常驻 whisper-server 转写:wav → {"text":"..."}。
/// 没有 serde_json,手写最简取值——whisper 的输出这里不会有嵌套引号。
#[cfg(target_os = "macos")]
fn whisper_server_transcribe(wav: &str, port: u16) -> Option<String> {
    let out = Command::new("curl")
        .args([
            "-s",
            "-m",
            "8",
            "-F",
            &format!("file=@{}", wav),
            "-F",
            "response-format=json",
            &format!("http://127.0.0.1:{}/inference", port),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let key = "\"text\":\"";
    let start = body.find(key)? + key.len();
    let end = body[start..].rfind('"')? + start;
    let raw = body[start..end].replace("\\n", " ").replace("\\\"", "\"");
    let t = raw.trim().to_string();
    if t.is_empty() || t.contains("BLANK_AUDIO") {
        None
    } else {
        Some(t)
    }
}

/// 用户交互式框选截图后，通过全屏截图 + 模板匹配反推出实际的屏幕坐标
fn find_screen_position(crop_img: &core::Mat) -> Option<(i32, i32, i32, i32)> {
    let full_path = "/tmp/tnt_full_screen.png";
    // 静默全屏截图
    #[cfg(target_os = "macos")]
    let _ = Command::new("screencapture").arg("-x").arg(full_path).status();
    #[cfg(target_os = "linux")]
    let _ = Command::new("sh").arg("-c").arg(format!("grim {}", full_path)).status();

    let full_img = imgcodecs::imread(full_path, imgcodecs::IMREAD_COLOR).ok()?;
    if full_img.empty() || crop_img.cols() > full_img.cols() || crop_img.rows() > full_img.rows() {
        return None;
    }

    let mut match_result = core::Mat::default();
    imgproc::match_template(&full_img, crop_img, &mut match_result, imgproc::TM_CCOEFF_NORMED, &core::no_array()).ok()?;
    let mut max_val = 0.0;
    let mut max_loc = core::Point::new(0, 0);
    core::min_max_loc(&match_result, None, Some(&mut max_val), None, Some(&mut max_loc), &core::no_array()).ok()?;

    if max_val > 0.5 {
        println!("✅ 模板匹配成功 (score={:.2})，屏幕坐标: ({},{}) {}x{}", max_val, max_loc.x, max_loc.y, crop_img.cols(), crop_img.rows());
        Some((max_loc.x, max_loc.y, crop_img.cols(), crop_img.rows()))
    } else {
        println!("⚠️  模板匹配得分过低 ({:.2})，使用 (0,0) 作为默认坐标", max_val);
        None
    }
}

#[cfg(target_os = "linux")]
fn select_crop_interactive(path: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("grim -g \"$(slurp)\" {}", path))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(target_os = "macos")]
fn select_crop_interactive(path: &str) -> bool {
    Command::new("screencapture")
        .arg("-i")
        .arg(path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn capture_rect_to_file(geo: (i32, i32, i32, i32), path: &str) {
    let _ = Command::new("sh")
        .arg("-c")
        .arg(format!("grim -g \"{},{} {}x{}\" -t ppm {}.tmp && mv {}.tmp {}", geo.0, geo.1, geo.2, geo.3, path, path, path))
        .status();
}

#[cfg(target_os = "macos")]
fn capture_rect_to_file(geo: (i32, i32, i32, i32), path: &str) {
    let _ = Command::new("screencapture")
        .arg("-R")
        .arg(format!("{},{},{},{}", geo.0, geo.1, geo.2, geo.3))
        .arg("-x")
        .arg("-t")
        .arg("png")
        .arg(path)
        .status();
}

fn main() -> opencv::Result<()> { // Recognizer moved to bg thread
    #[cfg(target_os = "macos")]
    println!("=== 🍎 Mac OS 环境检测成功，已自动切换原生 screencapture 截图引擎 ===");

    // ===== 网络实况隧道:libtntsniff.so → adb reverse → 127.0.0.1:19001 =====
    // 尽早绑定:用户还在框选时隧道就能连进来,开局 INIT 不容易错过。
    let net_state: Arc<Mutex<NetHud>> = Arc::new(Mutex::new(NetHud::default()));
    load_wind_keys(&mut net_state.lock().unwrap());
    {
        let ns = net_state.clone();
        thread::spawn(move || {
            let my_name =
                std::env::var("TNT_MY_NAME").unwrap_or_else(|_| "1111rust".to_string());
            let port: u16 = std::env::var("TNT_NET_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(19001);
            let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) else {
                println!("⚠️ NET: 端口 {port} 被占用(可能在跑 net_live),网络定位不可用");
                return;
            };
            let trace_path = std::env::var("TNT_TRACE_PATH")
                .unwrap_or_else(|_| "/tmp/tnt-tunnel-records.bin".to_string());
            let trace = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&trace_path)
                .ok()
                .map(|f| Arc::new(Mutex::new(f)));
            println!("📡 NET: 监听 127.0.0.1:{port} — adb reverse tcp:{port} tcp:{port}");
            println!("📡 NET: 原始记录保存到 {trace_path}");
            for conn in listener.incoming() {
                let Ok(mut s) = conn else { continue };
                let ns = ns.clone();
                let my_name = my_name.clone();
                let trace = trace.clone();
                thread::spawn(move || {
                    ns.lock().unwrap().conns += 1;
                    println!("📡 NET: 隧道已连接");
                    s.set_nodelay(true).ok();
                    let mut parser = tunnel::Parser::default();
                    let mut tracker = Tracker::default();
                    let mut buf = [0u8; 65536];
                    // (游戏端mono_ms, 本地收到时刻)基准对 → lag>0 说明隧道内排队延迟
                    let mut t0: Option<(u64, std::time::Instant)> = None;
                    let mut n_rec = 0u64;
                    loop {
                        let n = match s.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => n,
                        };
                        for rec in parser.feed(&buf[..n]) {
                            if let Some(trace) = &trace {
                                if let Ok(mut f) = trace.lock() {
                                    let len = rec.payload.len() as u32;
                                    let _ = f.write_all(&[rec.kind]);
                                    let _ = f.write_all(&rec.pid.to_le_bytes());
                                    let _ = f.write_all(&rec.fd.to_le_bytes());
                                    let _ = f.write_all(&rec.ms.to_le_bytes());
                                    let _ = f.write_all(&len.to_le_bytes());
                                    let _ = f.write_all(&rec.payload);
                                }
                            }
                            n_rec += 1;
                            let lag = match t0 {
                                None => {
                                    t0 = Some((rec.ms, std::time::Instant::now()));
                                    0i64
                                }
                                Some((a0, i0)) => rec.ms.wrapping_sub(a0) as i64
                                    - i0.elapsed().as_millis() as i64,
                            };
                            let mut st = ns.lock().unwrap();
                            st.last_lag_ms = lag;
                            let mut ev_names: Vec<String> = Vec::new();
                            for ev in tracker.on_record(&rec) {
                                if let TunnelEvent::Msg { msg, .. } = ev {
                                    for gev in decode_ws_message(&msg) {
                                        let name = match &gev {
                                            GameEvent::Init { .. } => "INIT",
                                            GameEvent::WindSeed { .. } => "WIND",
                                            GameEvent::PlayerUpdate { .. } => "UPDATE",
                                            GameEvent::Move { .. } => "MOVE",
                                            GameEvent::MoveReq { .. } => "MOVE_REQ",
                                            GameEvent::Flight { .. } => "FLIGHT",
                                            GameEvent::Fire { .. } => "FIRE",
                                            GameEvent::FireReq { .. } => "FIREREQ",
                                            GameEvent::Other { name, .. } => name.as_str(),
                                        };
                                        ev_names.push(name.to_string());
                                        apply_net_event(&mut st, &gev, &my_name);
                                    }
                                }
                            }
                            if !ev_names.is_empty() {
                                println!("📡 lag={lag}ms rec#{n_rec} evts={ev_names:?}");
                            }
                        }
                    }
                    ns.lock().unwrap().conns -= 1;
                    println!("📡 NET: 隧道连接断开,等待重连");
                });
            }
        });
    }

    println!("👉 [模式选择] 回车 = 小地图模式(实时刷新); 输入 2 = 大地图模式(框选整张静态地图):");
    let mut mode_buf = String::new();
    let _ = std::io::stdin().read_line(&mut mode_buf);
    let big_map_mode = mode_buf.trim() == "2";

    if big_map_mode {
        println!("👉 [步骤 1/2] 请框选【整张游戏地图】区域(整屏宽度 = 12/16/18/20/22 按钮)
    提示: 画面实时刷新;点我方/敌方后直接出力度");
    } else {
        println!("👉 [步骤 1/2] 请在屏幕上框选【左上角小地图】区域...");
    }
    let map_crop_path = "/tmp/tnt_selected_map.png";
    select_crop_interactive(map_crop_path);

    let initial_img = match imgcodecs::imread(map_crop_path, imgcodecs::IMREAD_COLOR) {
        Ok(m) if !m.empty() => m,
        _ => {
            println!("❌ 抓取【小地图】区域失败或取消！");
            return Ok(());
        }
    };
    let t_w = initial_img.cols();
    let t_h = initial_img.rows();
    // 通过模板匹配反推屏幕坐标，回退到 (0,0)
    let map_geo = find_screen_position(&initial_img).unwrap_or((0, 0, t_w, t_h));
    println!("📍 小地图屏幕区域: ({},{}) {}x{}", map_geo.0, map_geo.1, map_geo.2, map_geo.3);

    println!("👉 [步骤 2/2] 请框选【左下角圆盘里的角度数字】——框要紧贴数字本身,别把圆盘边缘/箭头/其他数字框进来...");
    let power_crop_path = "/tmp/tnt_selected_power.png";
    select_crop_interactive(power_crop_path);

    let power_img = imgcodecs::imread(power_crop_path, imgcodecs::IMREAD_COLOR)
        .ok()
        .filter(|m| !m.empty());
    let power_geo = power_img
        .as_ref()
        .and_then(|img| find_screen_position(img))
        // 外扩 10px:框紧贴数字时,换值后宽字形(8/9)会贴到截图边被"贴边否决"误伤,
        // 多抓一圈边距让否决只作用于真被裁的帧
        .map(|(x, y, w, h)| (x.saturating_sub(10), y.saturating_sub(10), w + 20, h + 20));
    if let Some(pg) = power_geo {
        println!("📍 数值区域屏幕坐标: ({},{}) {}x{}", pg.0, pg.1, pg.2, pg.3);
    }

    let window_name = "TNT Assistant HUD";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    // 显示缩放:大地图缩到 ~452px 宽(和小地图显示尺寸一致),小地图沿用放大规则
    let scale = if big_map_mode {
        (452.0 / t_w as f64).min(2.0)
    } else if t_w > 600 {
        1.0
    } else {
        2.0
    };

    let app_state = Arc::new(Mutex::new(AppState {
        edit_mode: EditMode::None,
        manual_p1: None,
        manual_e1: None,
        manual_cam_rect: None,
        drag_start: None,
        current_angle: 45.0,
        wind: 0.0,
        locked_px_per_unit: None,
        map_units: 12.0,
        big_map: big_map_mode,
        disp_scale: scale,
        src_w: t_w,
        src_h: t_h,
        auto_angle: true,
        is_fixed_angle: true,
        wind_flip: false,
        wind_net: false,
        exit_requested: false,
        switch_requested: false,
        btn_press: None,
    }));

    let map_w_display = (t_w as f64 * scale) as i32;

    // 大地图模式:整屏宽默认 12 距,启动即自动锁尺(12/16/18/20/22 按钮可切换)
    if big_map_mode {
        app_state.lock().unwrap().locked_px_per_unit = Some(t_w as f64 / 12.0);
    }

    let btn_p1 = core::Rect::new(map_w_display + 20, 30, 110, 40);
    let btn_e1 = core::Rect::new(map_w_display + 140, 30, 110, 40);
    // 大/小地图模式切换(点击后弹交互框选新区域)
    let btn_mode_switch = core::Rect::new(map_w_display + 260, 30, 110, 40);

    let btn_lock_ruler = core::Rect::new(map_w_display + 20, 80, 230, 40);
    let btn_draw_ruler = core::Rect::new(map_w_display + 20, 130, 230, 35);

    let btn_exit = core::Rect::new(map_w_display + 150, 5, 100, 30);
    // 多人局目标轮换:点击在敌人列表里循环选中(等效于右键点标记)
    let btn_target = core::Rect::new(map_w_display + 260, 5, 110, 30);

    let btn_clear = core::Rect::new(map_w_display + 20, 210, 230, 25);

    // 大地图模式:整屏距离档位(小地图模式下仅占位,点击无效)
    let btn_u12 = core::Rect::new(map_w_display + 20, 175, 52, 30);
    let btn_u16 = core::Rect::new(map_w_display + 77, 175, 52, 30);
    let btn_u18 = core::Rect::new(map_w_display + 134, 175, 52, 30);
    let btn_u20 = core::Rect::new(map_w_display + 191, 175, 52, 30);
    let btn_u22 = core::Rect::new(map_w_display + 248, 175, 52, 30);

    // Preset Angle Buttons
    let btn_a20 = core::Rect::new(map_w_display + 20, 240, 50, 30);
    let btn_a30 = core::Rect::new(map_w_display + 80, 240, 50, 30);
    let btn_a45 = core::Rect::new(map_w_display + 140, 240, 50, 30);
    let btn_a50 = core::Rect::new(map_w_display + 200, 240, 50, 30);

    let btn_a60 = core::Rect::new(map_w_display + 20, 275, 50, 30);
    let btn_a65 = core::Rect::new(map_w_display + 80, 275, 50, 30);
    let btn_a70 = core::Rect::new(map_w_display + 140, 275, 50, 30);
    let btn_a75 = core::Rect::new(map_w_display + 200, 275, 50, 30);

    // Fine-tune buttons
    let btn_ang_m5 = core::Rect::new(map_w_display + 20, 315, 40, 35);
    let btn_ang_minus = core::Rect::new(map_w_display + 65, 315, 35, 35);
    let rect_ang_text = core::Rect::new(map_w_display + 105, 315, 60, 35);
    let btn_ang_plus = core::Rect::new(map_w_display + 170, 315, 35, 35);
    let btn_ang_p5 = core::Rect::new(map_w_display + 210, 315, 40, 35);

    let btn_wind_m1 = core::Rect::new(map_w_display + 20, 355, 45, 30);
    let btn_wind_m01 = core::Rect::new(map_w_display + 75, 355, 45, 30);
    let rect_wind_text = core::Rect::new(map_w_display + 125, 355, 60, 30);
    let btn_wind_p01 = core::Rect::new(map_w_display + 190, 355, 45, 30);
    let btn_wind_p1 = core::Rect::new(map_w_display + 245, 355, 45, 30);

    let btn_auto_angle = core::Rect::new(map_w_display + 20, 395, 230, 30);

    // 全部按钮矩形:用于"按下和松开都在同一按钮内才生效"的防误触判定
    let all_btns = vec![
        btn_p1, btn_e1, btn_mode_switch, btn_lock_ruler, btn_draw_ruler,
        btn_exit, btn_target, btn_clear, btn_a20, btn_a30, btn_a45, btn_a50, btn_a60, btn_a65, btn_a70,
        btn_a75, btn_ang_m5, btn_ang_minus, btn_ang_plus, btn_ang_p5, btn_auto_angle,
        btn_wind_m1, btn_wind_m01, btn_wind_p01, btn_wind_p1,
        btn_u12, btn_u16, btn_u18, btn_u20, btn_u22,
    ];

    let state_cb = app_state.clone();
    let net_state_cb = net_state.clone();
    highgui::set_mouse_callback(
        window_name,
        Some(Box::new(move |event, x, y, _flags| {
            let net_active = net_state_cb
                .lock()
                .map(|n| n.battle_id > 0 && !n.pos.is_empty())
                .unwrap_or(false);
            let mut st = state_cb.lock().unwrap();
            let map_h_display = (st.src_h as f64 * st.disp_scale) as i32;
            let on_map = x >= 0 && x < map_w_display && y >= 0 && y < map_h_display;

            if event == highgui::EVENT_LBUTTONDOWN {
                // 点在按钮上:只记录位置,等松开时确认仍在同一按钮内再触发
                if all_btns.iter().any(|b| is_inside(x, y, *b)) {
                    st.btn_press = Some((x, y));
                    return;
                }
                // Map click
                if on_map {
                    let pt = core::Point::new(x, y);
                    match st.edit_mode {
                        EditMode::P1 => {
                            st.manual_p1 = Some(pt);
                            st.edit_mode = EditMode::None;
                        }
                        EditMode::E1 => {
                            st.manual_e1 = Some(pt);
                            st.edit_mode = EditMode::None;
                        }
                        EditMode::DrawRuler1 => {
                            st.drag_start = Some(pt);
                            st.edit_mode = EditMode::DrawRuler2;
                        }
                        EditMode::DrawRuler2 => {
                            if let Some(start) = st.drag_start {
                                let min_x = start.x.min(pt.x);
                                let max_x = start.x.max(pt.x);
                                let src_x = ((min_x as f64 / st.disp_scale).round() as i32)
                                    .clamp(0, st.src_w.saturating_sub(1));
                                let src_right = ((max_x as f64 / st.disp_scale).round() as i32)
                                    .clamp(src_x + 1, st.src_w);
                                st.manual_cam_rect =
                                    Some(core::Rect::new(src_x, 0, src_right - src_x, st.src_h));
                            }
                            st.drag_start = None;
                            st.edit_mode = EditMode::None;
                            // Auto-lock the ruler with the newly drawn box (0.0 triggers evaluation in drawing loop)
                            st.locked_px_per_unit = Some(0.0);
                        }
                        EditMode::None => {
                            // In a network battle, plain clicks must not silently override
                            // automatic coordinates. Use the explicit My/Enemy buttons first.
                            if allow_plain_manual_mark(net_active) {
                                st.manual_p1 = Some(pt);
                            }
                        }
                    }
                }
            } else if event == highgui::EVENT_RBUTTONDOWN {
                // 右键:网络局随时点敌人标记选目标;空地落点当手动敌方标记
                // (无网时随便标,有网时须先按"敌方"进 E1 模式才允许手动点)
                // 标尺框定过程中忽略右键,免打扰两次左键标定
                if on_map
                    && !matches!(st.edit_mode, EditMode::DrawRuler1 | EditMode::DrawRuler2)
                {
                    // 多人局:点在敌人标记附近 → 选中该玩家为目标(用网络精确坐标);
                    // 点在空地 → 照旧当手动敌方点
                    let mut picked = false;
                    if net_active {
                        if let Ok(mut n) = net_state_cb.lock() {
                            if n.map_w > 0 && n.map_h > 0 {
                                let mw = (st.src_w as f64 * st.disp_scale) as f64;
                                let mh = (st.src_h as f64 * st.disp_scale) as f64;
                                let mut best: Option<(u64, f64)> = None;
                                for (k, &(wx, wy)) in n.pos.iter() {
                                    if Some(*k) == n.my_id {
                                        continue;
                                    }
                                    let sx = wx as f64 * mw / n.map_w as f64;
                                    let sy = wy as f64 * mh / n.map_h as f64;
                                    let d = ((sx - x as f64).powi(2)
                                        + (sy - y as f64).powi(2))
                                    .sqrt();
                                    if d < 30.0 && best.map_or(true, |(_, bd)| d < bd) {
                                        best = Some((*k, d));
                                    }
                                }
                                if let Some((uid, _)) = best {
                                    n.target_id = Some(uid);
                                    st.manual_e1 = None;
                                    picked = true;
                                }
                            }
                        }
                    }
                    if !picked && (!net_active || st.edit_mode == EditMode::E1) {
                        st.manual_e1 = Some(core::Point::new(x, y));
                        st.edit_mode = EditMode::None;
                    }
                    if picked {
                        st.edit_mode = EditMode::None;
                    }
                }
            } else if event == highgui::EVENT_LBUTTONUP {
                // 按钮生效:按下和松开都在同一个按钮内,才算真点
                if let Some((px, py)) = st.btn_press.take() {
                    if all_btns
                        .iter()
                        .any(|b| is_inside(px, py, *b) && is_inside(x, y, *b))
                    {
                        if is_inside(x, y, btn_p1) {
                            st.edit_mode = if st.edit_mode == EditMode::P1 {
                                EditMode::None
                            } else {
                                EditMode::P1
                            };
                        } else if is_inside(x, y, btn_e1) {
                            st.edit_mode = if st.edit_mode == EditMode::E1 {
                                EditMode::None
                            } else {
                                EditMode::E1
                            };
                        } else if is_inside(x, y, btn_target) {
                            // 目标轮换:按 id 排序稳定循环,当前目标的下一个成为新目标
                            if let Ok(mut n) = net_state_cb.lock() {
                                let mut ids: Vec<u64> = n
                                    .pos
                                    .keys()
                                    .copied()
                                    .filter(|k| Some(*k) != n.my_id)
                                    .collect();
                                ids.sort_unstable();
                                if !ids.is_empty() {
                                    n.target_id = match n.target_id {
                                        Some(t) => {
                                            let i = ids.iter().position(|&k| k == t);
                                            Some(ids[(i.map_or(0, |i| i + 1)) % ids.len()])
                                        }
                                        None => Some(ids[0]),
                                    };
                                    st.manual_e1 = None;
                                }
                            }
                        } else if is_inside(x, y, btn_lock_ruler) {
                            if st.locked_px_per_unit.is_some() {
                                st.locked_px_per_unit = None;
                            } else {
                                st.locked_px_per_unit = Some(0.0);
                            }
                        } else if is_inside(x, y, btn_draw_ruler) {
                            st.edit_mode = if st.edit_mode == EditMode::DrawRuler1
                                || st.edit_mode == EditMode::DrawRuler2
                            {
                                EditMode::None
                            } else {
                                EditMode::DrawRuler1
                            };
                        } else if is_inside(x, y, btn_clear) {
                            // 只清标记。比例尺锁定和标记无关,保留(清了就会出现
                            // "明明锁定了却提示请先锁定"的问题)
                            st.manual_p1 = None;
                            st.manual_e1 = None;
                            st.manual_cam_rect = None;
                            st.edit_mode = EditMode::None;
                        } else if is_inside(x, y, btn_a20) {
                            st.current_angle = 20.0;
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_a30) {
                            st.current_angle = 30.0;
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_a45) {
                            st.current_angle = 45.0;
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_a50) {
                            st.current_angle = 50.0;
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_a60) {
                            st.current_angle = 60.0;
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_a65) {
                            st.current_angle = 65.0;
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_a70) {
                            st.current_angle = 70.0;
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_a75) {
                            st.current_angle = 75.0;
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_ang_m5) {
                            st.current_angle = (st.current_angle - 5.0).max(0.0);
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_ang_minus) {
                            st.current_angle = (st.current_angle - 1.0).max(0.0);
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_ang_plus) {
                            st.current_angle = (st.current_angle + 1.0).min(180.0);
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_ang_p5) {
                            st.current_angle = (st.current_angle + 5.0).min(180.0);
                            st.auto_angle = false;
                        } else if is_inside(x, y, btn_auto_angle) {
                            st.auto_angle = true;
                        } else if is_inside(x, y, btn_wind_m1) {
                            st.wind -= 1.0;
                        } else if is_inside(x, y, btn_wind_m01) {
                            st.wind -= 0.1;
                        } else if is_inside(x, y, btn_wind_p01) {
                            st.wind += 0.1;
                        } else if is_inside(x, y, btn_wind_p1) {
                            st.wind += 1.0;
                        } else if is_inside(x, y, btn_u12)
                            || is_inside(x, y, btn_u16)
                            || is_inside(x, y, btn_u18)
                            || is_inside(x, y, btn_u20)
                            || is_inside(x, y, btn_u22)
                        {
                            // 大地图模式:整屏宽度 = 所选距离,立刻按新档位重锁尺子。
                            // 小地图模式不响应(比例尺走手动标尺),只吞掉这次点击。
                            let units = if is_inside(x, y, btn_u12) {
                                12.0
                            } else if is_inside(x, y, btn_u16) {
                                16.0
                            } else if is_inside(x, y, btn_u18) {
                                18.0
                            } else if is_inside(x, y, btn_u20) {
                                20.0
                            } else {
                                22.0
                            };
                            if st.big_map {
                                st.map_units = units;
                                st.locked_px_per_unit = Some(st.src_w as f64 / units);
                            }
                        } else if is_inside(x, y, btn_mode_switch) {
                            // 运行时切换大/小地图:标记 switch_requested,
                            // 由 UI 线程弹交互框选(回调里不能阻塞,会死锁)
                            st.switch_requested = true;
                        } else if is_inside(x, y, btn_exit) {
                            st.exit_requested = true;
                        }
                    }
                }
            } else if event == highgui::EVENT_MOUSEMOVE {
                if st.edit_mode == EditMode::DrawRuler2 {
                    if let Some(start) = st.drag_start {
                        let min_x = start.x.min(x);
                        let max_x = start.x.max(x);
                        // 预览只看左右边界,纵向铺满整幅源图
                        let src_x = ((min_x as f64 / st.disp_scale).round() as i32)
                            .clamp(0, st.src_w.saturating_sub(1));
                        let src_right = ((max_x as f64 / st.disp_scale).round() as i32)
                            .clamp(src_x + 1, st.src_w);
                        st.manual_cam_rect =
                            Some(core::Rect::new(src_x, 0, src_right - src_x, st.src_h));
                    }
                }
            }
        })),
    )?;

    let is_running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r_clone = is_running.clone();

    let cap_time_ms = Arc::new(std::sync::Mutex::new(0u128));
    let cap_time_ms_clone = cap_time_ms.clone();

    // 地图截图区域放共享变量:运行时切换模式后,后台线程下一轮就用新区域
    let map_geo_shared = Arc::new(std::sync::Mutex::new(map_geo));
    let map_geo_shared_clone = map_geo_shared.clone();
    let power_geo_clone = power_geo.clone();

    let shared_map = Arc::new(std::sync::Mutex::new(None::<core::Mat>));
    let shared_map_clone = shared_map.clone();
    let shared_recognized_angle = Arc::new(std::sync::Mutex::new(None::<i32>));
    let shared_recognized_angle_clone = shared_recognized_angle.clone();

    // 回合切换信号:新回合清空角度投票+显示值,防止旧角度给出错误力度
    let angle_reset_gen =
        Arc::new(std::sync::atomic::AtomicU64::new(0));
    let angle_reset_gen_clone = angle_reset_gen.clone();

    let app_state_bg = app_state.clone();

    thread::spawn(move || {
        let recognizer =
            tnt_comput::ui::UiRecognizer::new("src/templates").expect("Failed to init recognizer");
        // 角度 OCR:表盘法读数本身很准(±1°),单帧直读直用——
        // 画面没变跳过识别,变了就识别就生效,基本即时。
        let mut last_angle_gen = 0u64;
        // 地图截图降频:大图抓取贵(~百ms级),每帧都抓会把角度ROI采样率压到个位数fps。
        // 地图内容变化慢,250ms刷一次完全够,角度框仍然每轮都抓。
        let mut last_map_cap = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let mut last_size_warn = std::time::Instant::now();
        let mut last_roi_sig = (0i64, 0i64, 0i64);
        let mut last_misread_dbg = std::time::Instant::now();
        let mut last_none_dbg = std::time::Instant::now();

        #[cfg(target_os = "linux")]
        let (map_path, power_path) = ("/tmp/tnt_map.ppm", "/tmp/tnt_power.ppm");
        #[cfg(target_os = "macos")]
        let (map_path, power_path) = ("/tmp/tnt_map.png", "/tmp/tnt_power.png");

        while r_clone.load(std::sync::atomic::Ordering::Relaxed) {
            let t0 = std::time::Instant::now();

            // 回合切换:只清投票缓冲(上回合读数不拖累新角度);
            // 显示值保留——旧角度先顶着,比一直"监听中"强
            let gen_now = angle_reset_gen_clone.load(std::sync::atomic::Ordering::Relaxed);
            if gen_now != last_angle_gen {
                last_angle_gen = gen_now;
            }

            // 实测(2560x1440):单次小区域截图 ~51ms,而大图的 PNG 编解码要贵得多,
            // 所以"并集成一张大图"反而不如"各自抓小图"快。
            // 地图 250ms 一帧(内容基本不变,降频把带宽让给角度ROI);
            // 角度框每轮都抓:区域只有几十像素,截图很快,去掉100ms节流后同步延迟减半。
            if last_map_cap.elapsed().as_millis() > 250 {
                last_map_cap = std::time::Instant::now();
                capture_rect_to_file(*map_geo_shared_clone.lock().unwrap(), map_path);
                if let Ok(m) = imgcodecs::imread(map_path, imgcodecs::IMREAD_COLOR) {
                    if !m.empty() {
                        if let Ok(mut lock) = shared_map_clone.lock() {
                            *lock = Some(m);
                        }
                    }
                }
            }

            if let Some(pg) = power_geo_clone {
                {
                    capture_rect_to_file(pg, power_path);
                    if let Ok(p) = imgcodecs::imread(power_path, imgcodecs::IMREAD_COLOR) {
                        if !p.empty() {
                            // 框得离谱大才警告:表盘法要框住整个表盘(~300px),
                            // 老数字OCR的小框(~60px)也照常工作
                            if p.cols() > 500 || p.rows() > 400 {
                                if last_size_warn.elapsed().as_millis() > 5000 {
                                    last_size_warn = std::time::Instant::now();
                                    println!(
                                        "⚠️ 角度区域 {}x{} 超过 500x400,识别跳过 —— 框整个表盘即可,别框半屏",
                                        p.cols(),
                                        p.rows()
                                    );
                                }
                            } else {
                                // TNT_DUMP_FAIL=1 时把原始 ROI 也落盘,方便事后分析误读帧
                                if std::env::var("TNT_DUMP_FAIL").is_ok() {
                                    let _ = imgcodecs::imwrite(
                                        "/tmp/tnt_roi_last.png",
                                        &p,
                                        &core::Vector::new(),
                                    );
                                }
                                // 画面逐像素没变 → 跳过识别直接下一帧
                                // (转场/静止期省掉OCR开销,提高每秒尝试次数)
                                let s = core::sum_elems(&p).unwrap_or_default();
                                let sig = (s[0] as i64, s[1] as i64, s[2] as i64);
                                if sig == last_roi_sig {
                                    if let Ok(mut lock) = cap_time_ms_clone.lock() {
                                        *lock = t0.elapsed().as_millis();
                                    }
                                    continue;
                                }
                                last_roi_sig = sig;
                                // 纯表盘几何法:读虚线方向,不走数字OCR
                                let val = recognizer.recognize_angle_dial(&p);
                                match val {
                                    Ok(Some(stable)) => {
                                        // 表盘法单帧即准,直读直用
                                        if last_misread_dbg.elapsed().as_millis() > 2000 {
                                            last_misread_dbg = std::time::Instant::now();
                                            println!("🔍 角度OCR: 读到 {}", stable);
                                        }
                                        if let Ok(mut lock) = shared_recognized_angle_clone.lock()
                                        {
                                            *lock = Some(stable);
                                        }
                                        if stable >= 3 && stable <= 90 {
                                            if let Ok(mut m_state) = app_state_bg.lock() {
                                                if m_state.auto_angle {
                                                    m_state.current_angle = stable as f64;
                                                }
                                            }
                                        }
                                    }
                                    Ok(None) => {
                                        // 失败帧在这套识别器里很常见(ui.rs 设计就是整帧作废、
                                        // 沿用上一次),所以不清投票——只跳过本帧,
                                        // 否则成功帧永远攒不够 2 帧,同步会卡死
                                        if last_none_dbg.elapsed().as_millis() > 1000 {
                                            last_none_dbg = std::time::Instant::now();
                                            println!("🔍 角度OCR: 本帧未识别(保留已有投票)");
                                        }
                                    }
                                    Err(_) => {}
                                }
                            }
                        }
                    }
                }
            }

            if let Ok(mut lock) = cap_time_ms_clone.lock() {
                *lock = t0.elapsed().as_millis();
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    });

    // TNT_WIND_SIGN_AUTO=1: 风速输入只报大小,顺/逆风符号自动按 网络箭头×敌人方位 附加
    let auto_wind_sign =
        std::env::var("TNT_WIND_SIGN_AUTO").ok().as_deref() == Some("1");
    // 1 距对应的服务器世界坐标数(视野宽=12距≈960单位→80)。TNT_WPU 可微调。
    let world_per_unit: f64 = std::env::var("TNT_WPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80.0);
    let mut last_battle_id: u64 = 0;
    let mut last_wind_turn: u64 = 0;
    /// 上一个我方回合(局,回合,raw,回合开始时的风值),翻篇时结算进配对日志
    let mut prev_my_turn: Option<(u64, u64, i64, f64)> = None;

    let mut first_show = true;

    // 立即显示初始化画面，防止 Mac 窗口引擎死锁
    let mut init_canvas = core::Mat::new_rows_cols_with_default(
        400,
        800,
        core::CV_8UC3,
        core::Scalar::new(30.0, 30.0, 30.0, 0.0),
    )?;
    imgproc::put_text(
        &mut init_canvas,
        "Initializing Background Capture...",
        core::Point::new(50, 200),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.0,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;
    highgui::imshow(window_name, &init_canvas)?;
    highgui::set_window_property(window_name, highgui::WND_PROP_TOPMOST, 1.0)?;
    highgui::wait_key(100)?;

    let mut wind_input_buf = String::new();

    // ===== 语音风速: 按住右 Option 说话 → ./mac_stt 录音写 wav →
    // whisper-server(常驻, ~150ms)转文字 → 解析写 st.wind =====
    #[allow(dead_code)] // 非 macOS 平台只收不发
    enum VoiceMsg {
        RecStart,
        RecEnd,
        Text(String),
        Err(String),
    }
    let (voice_tx, voice_rx) = mpsc::channel::<VoiceMsg>();
    #[allow(unused_assignments)]
    let mut voice_status = String::new();
    let mut voice_children: Vec<Child> = Vec::new();
    // 常听模式标志:为 true 时文本必须含"风/速"才改风速
    #[allow(unused_variables)]
    let voice_always = std::env::var("TNT_VOICE_ALWAYS").ok().as_deref() == Some("1");

    #[cfg(target_os = "macos")]
    {
        // 兼容从任意 cwd 启动:资源先找 ./,再找可执行文件上两级(项目根)。
        // 注意必须返回带分隔符的路径(绝对/带./),裸文件名会被当成 PATH 查找
        let exe_root = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("../..")));
        let resolve = |rel: &str| -> Option<String> {
            let abs = |p: &std::path::Path| {
                p.canonicalize().ok().map(|c| c.to_string_lossy().into_owned())
            };
            let cwd_p = std::path::Path::new(rel);
            if cwd_p.exists() {
                abs(cwd_p)
            } else {
                exe_root.as_ref().map(|r| r.join(rel)).and_then(|p| abs(&p))
            }
        };
        // 优先大模型(更准),没有就回退 base
        // SenseVoice 模型(sherpa-onnx int8):stt/<模型目录>/model.int8.onnx + tokens.txt
        let sv_dir = resolve("stt/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17");
        let vad_model = resolve("stt/silero_vad.onnx");
        let stt_bin = resolve("mac_stt");
        // TNT_VOICE_ALWAYS=1 → 常听:mac_stt stream 的 PCM 直接喂 stt_rec vad,
        // 不用按键;文本必须含"风/速"关键词才认(防环境音误改风速)
        if voice_always {
            let streamer = stt_bin.as_ref().and_then(|b| {
                Command::new(b)
                    .arg("stream")
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .ok()
            });
            let setup = match (sv_dir.as_ref(), vad_model.as_ref(), streamer) {
                (Some(d), Some(v), Some(mut s)) if s.stdout.is_some() => {
                    spawn_stt_vad(
                        &format!("{}/model.int8.onnx", d),
                        &format!("{}/tokens.txt", d),
                        v,
                        s.stdout.take().unwrap(),
                    )
                    .map(|x| (x, s))
                }
                _ => None,
            };
            match setup {
                Some(((srv_child, srv_out), mac_child)) => {
                    voice_children.push(srv_child);
                    voice_children.push(mac_child);
                    voice_status = "语音: 常听中 说'风速±xx'".to_string();
                    thread::spawn(move || {
                        for line in BufReader::new(srv_out).lines() {
                            let Ok(l) = line else { break };
                            if let Some(t) = l.strip_prefix("TEXT:") {
                                if !t.is_empty() {
                                    let _ = voice_tx.send(VoiceMsg::Text(t.to_string()));
                                }
                            } else if let Some(e) = l.strip_prefix("ERR:") {
                                let _ = voice_tx.send(VoiceMsg::Err(e.to_string()));
                            }
                        }
                    });
                }
                None => {
                    voice_status = "语音不可用: 缺 stt_rec/silero_vad.onnx/模型".to_string();
                }
            }
        } else {
        // whisper 兜底/turbo 模型(可选,缺了就等于没装,行为不变)
        let whisper_model = resolve("stt/ggml-large-v3-turbo-q8_0.bin");
        // 最近一次录音的 wav 路径:按键线程写,结果线程读(兜底要用)
        let latest_wav = Arc::new(std::sync::Mutex::new(None::<String>));
        // TNT_VOICE_TURBO=1:全部走常驻 whisper-server(大模型,每条 ~1.3s);
        // 起不来自动回退 SenseVoice 小模型
        let voice_turbo = std::env::var("TNT_VOICE_TURBO").ok().as_deref() == Some("1");
        let turbo_srv = if voice_turbo {
            whisper_model
                .as_deref()
                .and_then(|m| spawn_whisper_server(m, 18124))
        } else {
            None
        };
        let server = if turbo_srv.is_none() {
            sv_dir.as_ref().and_then(|d| {
                spawn_stt_server(
                    &format!("{}/model.int8.onnx", d),
                    &format!("{}/tokens.txt", d),
                )
            })
        } else {
            None
        };
        let recorder = stt_bin.as_ref().and_then(|b| {
            // stdin 必须 piped 保持打开:null 会让 mac_stt 的"父进程退出自杀"
            // 看门狗读到 /dev/null EOF 立刻退出,按键监听根本没起来
            Command::new(b)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .ok()
        });
        match (turbo_srv, server, recorder) {
            (Some(tsrv), _, Some(mut stt)) if stt.stdout.is_some() => {
                // turbo 全量:whisper-server 常驻,WAV → curl → Text
                let so = stt.stdout.take().unwrap();
                voice_children.push(tsrv);
                voice_children.push(stt);
                voice_status = "语音: 按住Shift说话 [turbo]".to_string();
                thread::spawn(move || {
                    for line in BufReader::new(so).lines() {
                        let Ok(l) = line else { break };
                        if l == "REC:START" {
                            let _ = voice_tx.send(VoiceMsg::RecStart);
                        } else if l == "REC:END" {
                            let _ = voice_tx.send(VoiceMsg::RecEnd);
                        } else if let Some(p) = l.strip_prefix("WAV:") {
                            if p.starts_with('(') {
                                let _ = voice_tx.send(VoiceMsg::Err("按太短没录上".into()));
                                continue;
                            }
                            // curl 是同步阻塞(~1.3s),另开线程别卡按键事件
                            let p = p.to_string();
                            let tx3 = voice_tx.clone();
                            thread::spawn(move || {
                                match whisper_server_transcribe(&p, 18124) {
                                    Some(txt) => {
                                        let _ = tx3.send(VoiceMsg::Text(txt));
                                    }
                                    None => {
                                        let _ = tx3.send(VoiceMsg::Err("没听清".into()));
                                    }
                                }
                            });
                        } else if let Some(e) = l.strip_prefix("ERR:") {
                            let _ = voice_tx.send(VoiceMsg::Err(e.to_string()));
                        }
                    }
                });
            }
            (None, Some(srv), Some(mut stt)) if stt.stdout.is_some() => {
                let so = stt.stdout.take().unwrap();
                let rec_in = Arc::new(Mutex::new(srv.stdin));
                voice_children.push(srv.child);
                voice_children.push(stt);
                voice_status = "语音: 按住Shift说话".to_string();
                // stt_rec 结果线程:TEXT:<识别文本>
                let tx2 = voice_tx.clone();
                let whisper_model2 = whisper_model.clone();
                let latest_wav_res = latest_wav.clone();
                thread::spawn(move || {
                    for line in BufReader::new(srv.stdout).lines() {
                        let Ok(l) = line else { break };
                        if let Some(t) = l.strip_prefix("TEXT:") {
                            // SenseVoice 空/解析不出数字 → whisper turbo 兜底同一条 wav
                            let ok = !t.is_empty()
                                && tnt_comput::voice::parse_wind_speech(t).is_some();
                            if ok {
                                let _ = tx2.send(VoiceMsg::Text(t.to_string()));
                                continue;
                            }
                            let fb = latest_wav_res
                                .lock()
                                .unwrap()
                                .clone()
                                .zip(whisper_model2.clone())
                                .and_then(|(w, m)| whisper_fallback(&w, &m));
                            match fb {
                                Some(txt) => {
                                    println!("🎙 whisper兜底: \"{}\" (SenseVoice: \"{}\")", txt, t);
                                    let _ = tx2.send(VoiceMsg::Text(txt));
                                }
                                None => {
                                    let _ = tx2.send(if t.is_empty() {
                                        VoiceMsg::Err("没听清".into())
                                    } else {
                                        VoiceMsg::Text(t.to_string())
                                    });
                                }
                            }
                        } else if let Some(e) = l.strip_prefix("ERR:") {
                            let _ = tx2.send(VoiceMsg::Err(e.to_string()));
                        }
                    }
                });
                // mac_stt 按键线程:REC/WAV 事件,WAV 路径喂给 stt_rec
                let latest_wav_key = latest_wav.clone();
                thread::spawn(move || {
                    for line in BufReader::new(so).lines() {
                        let Ok(l) = line else { break };
                        if l == "REC:START" {
                            let _ = voice_tx.send(VoiceMsg::RecStart);
                        } else if l == "REC:END" {
                            let _ = voice_tx.send(VoiceMsg::RecEnd);
                        } else if let Some(p) = l.strip_prefix("WAV:") {
                            if p.starts_with('(') {
                                let _ = voice_tx.send(VoiceMsg::Err("按太短没录上".into()));
                                continue;
                            }
                            *latest_wav_key.lock().unwrap() = Some(p.to_string());
                            use std::io::Write;
                            let res = rec_in
                                .lock()
                                .unwrap()
                                .write_all(format!("{}\n", p).as_bytes());
                            if res.is_err() {
                                let _ = voice_tx.send(VoiceMsg::Err("识别服务断了".into()));
                            }
                        } else if let Some(e) = l.strip_prefix("ERR:") {
                            let _ = voice_tx.send(VoiceMsg::Err(e.to_string()));
                        }
                    }
                });
            }
            (t, _, _) => {
                // turbo 已拉起但 mac_stt 没起 → 杀掉,别留孤儿 whisper-server
                if let Some(mut t) = t {
                    let _ = t.kill();
                }
                voice_status =
                    "语音不可用: 需要 ./mac_stt + ./stt_rec + stt/sense-voice 模型目录"
                        .to_string();
            }
        }
        }
    }

    // 弹道缓存：点位/角度/风没变就不重算。定角模式约 70 次全程仿真，
    // 不可达时更是 600+ 次采样，每帧重算是纯浪费。
    let mut traj_memo: Option<((f64, f64, f64, f64, bool), Option<(f64, f64)>)> = None;
    let mut fps_t0 = std::time::Instant::now();
    let mut last_toggle_time = std::time::Instant::now() - std::time::Duration::from_secs(1); // 防抖时间戳

    loop {
        let loop_t0 = std::time::Instant::now();

        // 语音消息: 录音状态/识别文本→风速
        while let Ok(msg) = voice_rx.try_recv() {
            match msg {
                VoiceMsg::RecStart => voice_status = "🎙 录音中...(松开识别)".to_string(),
                VoiceMsg::RecEnd => voice_status = "识别中...".to_string(),
                VoiceMsg::Text(t) => {
                    // 带"度/角"→角度(显式值优先,暂停OCR同步防止被旧读数顶回)
                    if let Some(a) = tnt_comput::voice::parse_angle_speech(&t) {
                        let a = a.clamp(0.0, 180.0);
                        let mut stg = app_state.lock().unwrap();
                        stg.current_angle = a;
                        stg.auto_angle = false;
                        drop(stg);
                        voice_status = format!("✅ 语音角 {:.0}° (\"{}\")", a, t);
                        continue;
                    }
                    // 常听模式必须带"风/速"关键词,否则忽略(防环境音误触发)
                    let parsed = if voice_always {
                        tnt_comput::voice::parse_wind_gated(&t)
                    } else {
                        tnt_comput::voice::parse_wind_speech(&t)
                    };
                    match parsed {
                        Some(w) => {
                            let flip_ui = app_state.lock().unwrap().wind_flip;
                            let mut net = net_state.lock().unwrap();
                            // 报一次风=反解局密钥候选 C;众数生效,之后所有回合自动解码
                            if net.wind_seed != 0 {
                                let w10 = (w.abs() * 10.0).round() as i64;
                                let c = (net.wind_seed.abs() & 0xff) ^ w10 ^ net.round as i64;
                                push_wind_vote(&mut net, c);
                            }
                            let applied = if auto_wind_sign {
                                apply_auto_wind_sign(&net, w, flip_ui)
                            } else {
                                w
                            };
                            log_wind_pair(
                                "VOICE",
                                net.battle_id,
                                net.round,
                                net.active,
                                net.wind_seed,
                                Some(w),
                                Some(applied),
                            );
                            drop(net);
                            {
                                let mut stg = app_state.lock().unwrap();
                                stg.wind = applied;
                                stg.wind_net = false;
                            }
                            wind_input_buf.clear();
                            voice_status = format!("✅ 语音风 {:+.1} (\"{}\")", applied, t);
                        }
                        None if voice_always => {
                            voice_status = format!("语音: 忽略 \"{}\" (缺风速关键词)", t)
                        }
                        None => voice_status = format!("❓ 没听清 (\"{}\")", t),
                    }
                }
                VoiceMsg::Err(e) => voice_status = format!("语音: {}", e),
            }
        }

        let img = {
            let lock = shared_map.lock().unwrap();
            lock.as_ref().and_then(|m| m.try_clone().ok())
        };

        let t_io = loop_t0.elapsed().as_millis();
        let t1 = std::time::Instant::now();

        let power_recognized_val = *shared_recognized_angle.lock().unwrap();

        let t_recog = t1.elapsed().as_millis();
        let t2 = std::time::Instant::now();

        let canvas_w = map_w_display + 310;
        let st = *app_state.lock().unwrap();
        // 模式切换后这些会变,每帧从状态里取(遮蔽外层的启动值)
        let t_w = st.src_w;
        let t_h = st.src_h;
        let scale = st.disp_scale;
        let map_h_display = (t_h as f64 * scale) as i32;
        let canvas_h_target = map_h_display.max(580);
        let mut canvas = core::Mat::new_rows_cols_with_default(
            canvas_h_target,
            canvas_w,
            core::CV_8UC3,
            core::Scalar::new(30.0, 30.0, 30.0, 0.0),
        )?;

        if let Some(minimap) = img {
            let mut map_display = core::Mat::default();
            imgproc::resize(
                &minimap,
                &mut map_display,
                core::Size::new(map_w_display, (t_h as f64 * scale) as i32),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            for y in 0..map_display.rows() {
                if let (Ok(src_row), Ok(dst_row)) = (
                    map_display.at_row::<core::Vec3b>(y),
                    canvas.at_row_mut::<core::Vec3b>(y),
                ) {
                    let len = src_row.len();
                    dst_row[0..len].copy_from_slice(src_row);
                }
            }



            let cam_rect = st
                .manual_cam_rect
                .unwrap_or(core::Rect::new(0, 0, t_w, t_h));

            let to_scr = |p: core::Point| {
                core::Point::new((p.x as f64 * scale) as i32, (p.y as f64 * scale) as i32)
            };

            // 绘制摄像机框（黄框=手动标尺/识别框）
            let cam_p1 = to_scr(core::Point::new(cam_rect.x, cam_rect.y));
            let cam_p2 = to_scr(core::Point::new(
                cam_rect.x + cam_rect.width,
                cam_rect.y + cam_rect.height,
            ));
            if st.manual_cam_rect.is_some() {
                // 未锁=黄框(待定),已锁=绿框(比例尺生效中)
                let box_color = if st.locked_px_per_unit.is_some() {
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0)
                } else {
                    core::Scalar::new(0.0, 255.0, 255.0, 0.0)
                };
                let _ = imgproc::rectangle(
                    &mut canvas,
                    core::Rect::new(cam_p1.x, cam_p1.y, cam_p2.x - cam_p1.x, cam_p2.y - cam_p1.y),
                    box_color,
                    2,
                    imgproc::LINE_8,
                    0,
                );

                let ruler_width = cam_p2.x - cam_p1.x;
                for i in 1..12 {
                    let tick_x = cam_p1.x + (ruler_width as f64 * (i as f64 / 12.0)) as i32;
                    let _ = imgproc::line(&mut canvas, core::Point::new(tick_x, cam_p1.y), core::Point::new(tick_x, cam_p1.y + 10), box_color, 1, imgproc::LINE_AA, 0);
                    let _ = imgproc::line(&mut canvas, core::Point::new(tick_x, cam_p2.y - 10), core::Point::new(tick_x, cam_p2.y), box_color, 1, imgproc::LINE_AA, 0);
                }

                let cam_txt = format!(
                    "CAMERA {}x{}",
                    cam_rect.width,
                    cam_rect.height
                );
                let _ = imgproc::put_text(&mut canvas, &cam_txt, core::Point::new(cam_p1.x, (cam_p1.y - 5).max(10)), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, imgproc::LINE_8, false);
            }

            let mut current_px_per_unit = cam_rect.width as f64 / 12.0;
            if let Some(locked) = st.locked_px_per_unit {
                if locked == 0.0 {
                    // 大地图且没画过手动标尺:整屏宽 = map_units 距;
                    // 画了标尺就仍按标尺算,两种都能用
                    let v = if st.big_map && st.manual_cam_rect.is_none() {
                        t_w as f64 / st.map_units
                    } else {
                        current_px_per_unit
                    };
                    current_px_per_unit = v;
                    if let Ok(mut m_state) = app_state.lock() {
                        m_state.locked_px_per_unit = Some(v);
                    }
                } else {
                    current_px_per_unit = locked;
                }
            }
            let px_per_unit = current_px_per_unit;

            // ===== 网络实况:世界坐标 → 小地图 canvas 点 =====
            let net = net_state.lock().unwrap().clone();

            // 局密钥已知 → 我方新回合自动解码风速写进 st.wind(符号=箭头×方位×F翻转)
            if net.turn_seq != last_wind_turn {
                last_wind_turn = net.turn_seq;
                // 通知 OCR 线程:新回合清票,快速锁定新角度
                angle_reset_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // 结算上一个我方回合:st.wind=用户最终认定值(没改=预测正确,改了=修正)
                // → 每个我方回合都有标注数据,纠错样本额外给密钥投票
                if let Some((pb, pr, praw, pw)) = prev_my_turn.take() {
                    let settled = app_state.lock().unwrap().wind;
                    log_wind_pair(
                        "SETTLED",
                        pb,
                        pr,
                        net.my_id,
                        praw,
                        Some(settled),
                        Some(settled),
                    );
                    if (settled - pw).abs() > 0.01
                        && settled.abs() > 0.05
                        && pb == net.battle_id
                    {
                        let w10 = (settled.abs() * 10.0).round() as i64;
                        let c = (praw.abs() & 0xff) ^ w10 ^ pr as i64;
                        push_wind_vote(&mut net_state.lock().unwrap(), c);
                    }
                }
                if net.active.is_some() && net.active == net.my_id {
                    let mut stg = app_state.lock().unwrap();
                    if let Some(mag10) = net.auto_wind10 {
                        let flip_n = if stg.wind_flip { -1 } else { 1 };
                        let dir = wind_dir_hint(&net)
                            .map(|h| h * flip_n)
                            .unwrap_or(net.wind_seed.signum() as i32);
                        stg.wind = mag10 as f64 / 10.0 * dir as f64;
                        stg.wind_net = true;
                    }
                    prev_my_turn =
                        Some((net.battle_id, net.round, net.wind_seed, stg.wind));
                }
            }

            // 小地图渲染横纵密度不同(世界 1688x1206 压进 233x274 选区时,
            // 纵向像素密度比横向密约1.65倍)。dy/弧线纵向必须乘这个比,
            // 否则高差被放大、弧线被压扁。无网络地图尺寸时保持原行为。
            let px_per_unit_y = if net.map_w > 0 && net.map_h > 0 {
                px_per_unit * (st.src_h as f64 * net.map_w as f64)
                    / (st.src_w as f64 * net.map_h as f64)
            } else {
                px_per_unit
            };

            // 新对局:清掉上一局的手动标记,按世界尺度自动锁尺
            if net.battle_id != 0 && net.battle_id != last_battle_id && net.map_w > 0 {
                last_battle_id = net.battle_id;
                if let Ok(mut stg) = app_state.lock() {
                    stg.manual_p1 = None;
                    stg.manual_e1 = None;
                    stg.manual_cam_rect = None;
                    stg.locked_px_per_unit =
                        Some(st.src_w as f64 / net.map_w as f64 * world_per_unit);
                }
            }

            let net_p1;
            let net_e1;
            // 所有非我玩家: (uid, 显示坐标) —— 多人局全部画出来供点选
            let mut net_all: Vec<(u64, core::Point)> = Vec::new();
            if net.map_w > 0 && net.map_h > 0 {
                let to_map = |x: i64, y: i64| core::Point::new(
                    (x as f64 * map_w_display as f64 / net.map_w as f64).round() as i32,
                    (y as f64 * map_h_display as f64 / net.map_h as f64).round() as i32,
                );
                net_p1 = net
                    .my_id
                    .and_then(|id| net.pos.get(&id))
                    .map(|&(x, y)| to_map(x, y));
                net_all = net
                    .pos
                    .iter()
                    .filter(|(k, _)| Some(**k) != net.my_id)
                    .map(|(k, &(x, y))| (*k, to_map(x, y)))
                    .collect();
                // 目标:点选的优先,否则第一个非我玩家
                net_e1 = net
                    .target_id
                    .and_then(|t| net.pos.get(&t))
                    .or_else(|| {
                        net.pos
                            .iter()
                            .find(|(k, _)| Some(**k) != net.my_id)
                            .map(|(_, p)| p)
                    })
                    .map(|&(x, y)| to_map(x, y));
            } else {
                net_p1 = None;
                net_e1 = None;
            }

            // 手动点击优先(覆盖),网络点位兜底
            let p1 = st.manual_p1.or(net_p1);
            let e1 = st.manual_e1.or(net_e1);

            // 网络点位:空心环标(青色=我方,品红=敌方),被手动点覆盖时仍画出
            let draw_ring = |c: &mut core::Mat, pt: core::Point, is_red: bool| {
                let color = if is_red {
                    core::Scalar::new(255.0, 0.0, 255.0, 0.0)
                } else {
                    core::Scalar::new(255.0, 255.0, 0.0, 0.0)
                };
                let _ = imgproc::circle(c, pt, 9, color, 2, imgproc::LINE_AA, 0);
                let _ = imgproc::circle(c, pt, 2, color, -1, imgproc::LINE_AA, 0);
            };
            if let Some(p) = net_p1 {
                draw_ring(&mut canvas, p, false);
            }
            // 所有非我玩家都画出来:选中的目标亮品红+名字,其余暗色(右键点选切换目标)
            let sel_uid = net.target_id.or_else(|| {
                net.pos
                    .iter()
                    .find(|(k, _)| Some(**k) != net.my_id)
                    .map(|(k, _)| *k)
            });
            for (uid, p) in &net_all {
                let is_sel = sel_uid == Some(*uid);
                let color = if is_sel {
                    core::Scalar::new(255.0, 0.0, 255.0, 0.0)
                } else {
                    core::Scalar::new(140.0, 80.0, 140.0, 0.0)
                };
                let thick = if is_sel { 2 } else { 1 };
                let _ = imgproc::circle(&mut canvas, *p, 9, color, thick, imgproc::LINE_AA, 0);
                let _ = imgproc::circle(&mut canvas, *p, 2, color, -1, imgproc::LINE_AA, 0);
                if let Some(nm) = net.names.get(uid) {
                    let short: String = nm.chars().take(5).collect();
                    let _ = imgproc::put_text(
                        &mut canvas,
                        &short,
                        core::Point::new(p.x - 12, p.y - 13),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.35,
                        color,
                        1,
                        imgproc::LINE_AA,
                        false,
                    );
                }
            }

            let draw_pt =
                |c: &mut core::Mat, pt: core::Point, label: &str, is_red: bool, is_manual: bool| {
                    // 手动点击的坐标已经是 canvas 像素，直接画。
                    let cx = pt.x;
                    let cy = pt.y;
                    let color = if is_red {
                        core::Scalar::new(50.0, 50.0, 255.0, 0.0)
                    } else {
                        core::Scalar::new(255.0, 255.0, 0.0, 0.0)
                    };
                    let thickness = if is_manual { -1 } else { 2 };
                    // radius 6 for smaller dots
                    let _ = imgproc::circle(
                        c,
                        core::Point::new(cx, cy),
                        6,
                        color,
                        thickness,
                        imgproc::LINE_AA,
                        0,
                    );
                    let _ = imgproc::put_text(
                        c,
                        label,
                        core::Point::new(cx - 15, cy - 12),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_AA,
                        false,
                    );
                };

            if let Some(p) = p1 {
                let label = if st.manual_p1.is_some() { "我方 [手动]" } else { "我方 (My)" };
                draw_pt(&mut canvas, p, label, false, st.manual_p1.is_some());
            }
            if let Some(e) = e1 {
                let label = if st.manual_e1.is_some() {
                    "敌方 [手动]".to_string()
                } else {
                    let nm = sel_uid
                        .and_then(|u| net.names.get(&u))
                        .map(|s| s.as_str())
                        .unwrap_or("Enemy");
                    format!("敌方 ({})", nm)
                };
                draw_pt(&mut canvas, e, &label, true, st.manual_e1.is_some());
            }

            let mut y_offset = 480;
            let mut draw_result = |p: core::Point, e: core::Point| {
                // Ensure ruler is locked
                if st.locked_px_per_unit.is_none() {
                    imgproc::rectangle(
                        &mut canvas,
                        core::Rect::new(map_w_display + 10, y_offset - 35, 250, 80),
                        core::Scalar::new(0.0, 0.0, 50.0, 0.0),
                        -1,
                        imgproc::LINE_8,
                        0,
                    )
                    .unwrap();
                    imgproc::rectangle(
                        &mut canvas,
                        core::Rect::new(map_w_display + 10, y_offset - 35, 250, 80),
                        core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_8,
                        0,
                    )
                    .unwrap();
                    let _ = imgproc::put_text(
                        &mut canvas,
                        "请先锁定距离尺!",
                        core::Point::new(map_w_display + 20, y_offset + 10),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_AA,
                        false,
                    );
                    return;
                }

                // Convert pixel diff to original minimap scale, then apply px_per_unit
                let orig_dx = (e.x - p.x) as f64 / scale;
                let dx = orig_dx / px_per_unit;

                let orig_dy = -(e.y - p.y) as f64 / scale;
                let dy = orig_dy / px_per_unit_y;

                // 相对风(正=顺风),纯手动输入(按钮/语音)
                let eff_wind = st.wind;

                let key = (dx, dy, st.current_angle, eff_wind, st.is_fixed_angle);
                let trajectory_res = match traj_memo {
                    Some((k, v)) if k == key => v,
                    _ => {
                        let r = if st.is_fixed_angle {
                            compute_fixed_trajectory(dx, dy, st.current_angle, eff_wind)
                        } else {
                            compute_trajectory(dx, dy, st.current_angle, eff_wind)
                        };
                        traj_memo = Some((key, r));
                        r
                    }
                };

                match trajectory_res {
                    Some((force, final_angle)) => {
                        // Draw a beautiful background box for the force recommendation
                        imgproc::rectangle(
                            &mut canvas,
                            core::Rect::new(map_w_display + 10, y_offset - 35, 300, 80),
                            core::Scalar::new(0.0, 50.0, 0.0, 0.0),
                            -1,
                            imgproc::LINE_8,
                            0,
                        )
                        .unwrap();
                        imgproc::rectangle(
                            &mut canvas,
                            core::Rect::new(map_w_display + 10, y_offset - 35, 300, 80),
                            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                            2,
                            imgproc::LINE_8,
                            0,
                        )
                        .unwrap();

                        let mode_str = if st.is_fixed_angle {
                            "[定角打法]"
                        } else {
                            "[变角打法]"
                        };
                        let title = format!(
                            "{}  风力: {:.1}  X距: {:.1}  Y高: {:.1}",
                            mode_str,
                            eff_wind,
                            dx.abs(),
                            dy
                        );
                        let _ = imgproc::put_text(
                            &mut canvas,
                            &title,
                            core::Point::new(map_w_display + 15, y_offset - 10),
                            imgproc::FONT_HERSHEY_SIMPLEX,
                            0.45,
                            core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                            1,
                            imgproc::LINE_AA,
                            false,
                        );

                        let res_txt = if st.is_fixed_angle {
                            format!("锁定: {:.0}° 力度: {:.1} 2/3: {:.1}", final_angle, force, force * 2.0 / 3.0)
                        } else {
                            format!("推荐: {:.0}° 力度: {:.1} 2/3: {:.1}", final_angle, force, force * 2.0 / 3.0)
                        };
                        let _ = imgproc::put_text(
                            &mut canvas,
                            &res_txt,
                            core::Point::new(map_w_display + 15, y_offset + 15),
                            imgproc::FONT_HERSHEY_SIMPLEX,
                            0.55,
                            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                            2,
                            imgproc::LINE_AA,
                            false,
                        );

                        // Draw trajectory dots on the minimap
                        let mut draw_angle = final_angle;
                        let is_reverse = e.x < p.x;
                        if is_reverse && draw_angle <= 90.0 {
                            draw_angle = 180.0 - draw_angle;
                        } else if !is_reverse && draw_angle > 90.0 {
                            draw_angle = 180.0 - draw_angle;
                        }

                        // 物理引擎原生支持真实的物理世界坐标系（角度>90代表向左，风向带符号）
                        // 用户输入的是相对风力（正=顺风），但在画图时，我们要把它转成绝对世界的风向。
                        // 如果向左打，顺风就是向左吹（绝对世界里的负风向）。
                        let sim_wind = if is_reverse { -eff_wind } else { eff_wind };
                        let path = tnt_comput::physics::simulate_path(draw_angle, force, sim_wind);
                        for (sim_x, sim_y) in path {
                            let img_x = p.x as f64 + sim_x * scale * px_per_unit;
                            let img_y = p.y as f64 - sim_y * scale * px_per_unit_y;

                            // Stop if out of bounds of the minimap
                            if img_x < 0.0 || img_x > map_w_display as f64 || img_y > map_h_display as f64 {
                                break;
                            }
                            if img_y >= 0.0 {
                                let _ = imgproc::circle(
                                    &mut canvas,
                                    core::Point::new(img_x as i32, img_y as i32),
                                    2,
                                    core::Scalar::new(255.0, 255.0, 0.0, 0.0),
                                    -1,
                                    imgproc::LINE_AA,
                                    0,
                                );
                            }
                        }
                    }
                    None => {
                        // Draw Unreachable box
                        imgproc::rectangle(
                            &mut canvas,
                            core::Rect::new(map_w_display + 10, y_offset - 35, 300, 80),
                            core::Scalar::new(0.0, 0.0, 50.0, 0.0),
                            -1,
                            imgproc::LINE_8,
                            0,
                        )
                        .unwrap();
                        let _ = imgproc::put_text(
                            &mut canvas,
                            "❌ 目标不可达 (Unreachable)",
                            core::Point::new(map_w_display + 20, y_offset + 5),
                            imgproc::FONT_HERSHEY_SIMPLEX,
                            0.7,
                            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                            2,
                            imgproc::LINE_AA,
                            false,
                        );
                    }
                }

                if let Some(pval) = power_recognized_val {
                    let power_txt = format!("右下角实时读数: {}", pval);
                    let _ = imgproc::put_text(
                        &mut canvas,
                        &power_txt,
                        core::Point::new(map_w_display + 15, y_offset + 38),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.55,
                        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_AA,
                        false,
                    );
                }

                y_offset += 100;

                let pt_p = core::Point::new(p.x, p.y);
                let pt_e = core::Point::new(e.x, e.y);
                let pt_corner = core::Point::new(e.x, p.y);
                // Draw horizontal line (X distance)
                let _ = imgproc::line(
                    &mut canvas,
                    pt_p,
                    pt_corner,
                    core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                    1,
                    imgproc::LINE_AA,
                    0,
                );
                // Draw vertical line (Y distance)
                let _ = imgproc::line(
                    &mut canvas,
                    pt_corner,
                    pt_e,
                    core::Scalar::new(255.0, 100.0, 0.0, 0.0),
                    1,
                    imgproc::LINE_AA,
                    0,
                );
            };

            if let (Some(p), Some(e)) = (p1, e1) {
                draw_result(p, e);
            } else if net.map_w == 0 && net.pos.len() >= 2 && st.manual_p1.is_none() && st.manual_e1.is_none() {
                // 中途加入的对局:错过了 INIT,拿不到地图尺寸,无法在小地图标点。
                // 但世界坐标差值是完整数据——直接出距离和力度,不画地图。
                let mine = net.my_id.and_then(|id| net.pos.get(&id));
                let enemy = net
                    .pos
                    .iter()
                    .find(|(k, _)| Some(**k) != net.my_id)
                    .map(|(_, v)| *v);
                if let (Some(&(px, py)), Some((ex, ey))) = (mine, enemy) {
                    let dx = (ex - px) as f64 / world_per_unit;
                    let dy = (py - ey) as f64 / world_per_unit;
                    let res = if st.is_fixed_angle {
                        compute_fixed_trajectory(dx, dy, st.current_angle, st.wind)
                    } else {
                        compute_trajectory(dx, dy, st.current_angle, st.wind)
                    };
                    imgproc::rectangle(
                        &mut canvas,
                        core::Rect::new(map_w_display + 10, 445, 300, 80),
                        core::Scalar::new(0.0, 40.0, 0.0, 0.0),
                        -1,
                        imgproc::LINE_8,
                        0,
                    )?;
                    imgproc::rectangle(
                        &mut canvas,
                        core::Rect::new(map_w_display + 10, 445, 300, 80),
                        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                        2,
                        imgproc::LINE_8,
                        0,
                    )?;
                    let _ = imgproc::put_text(
                        &mut canvas,
                        &format!("X距: {:.1}  Y高: {:.1}  风: {:.1}", dx.abs(), dy, st.wind),
                        core::Point::new(map_w_display + 15, 465),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                        1,
                        imgproc::LINE_AA,
                        false,
                    );
                    let txt = match res {
                        Some((force, ang)) => format!("力度: {:.1}  ({} {:.0}°)", force, if st.is_fixed_angle {"锁定"} else {"推荐"}, ang),
                        None => "❌ 目标不可达".to_string(),
                    };
                    let _ = imgproc::put_text(
                        &mut canvas,
                        &txt,
                        core::Point::new(map_w_display + 15, 495),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.55,
                        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                        2,
                        imgproc::LINE_AA,
                        false,
                    );
                }
            }
        } else if img.is_none() {
            imgproc::put_text(
                &mut canvas,
                "Waiting for Capture...",
                core::Point::new(50, 50),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.8,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )?;
        }

        let pval_str = if st.auto_angle {
            if let Some(val) = power_recognized_val {
                format!("实机同步: 开启 ({})", val)
            } else {
                "实机同步: 监听中...".to_string()
            }
        } else {
            "实机同步: 已暂停 (点击恢复)".to_string()
        };
        let _ = draw_btn(
            &mut canvas,
            btn_auto_angle,
            &pval_str,
            st.auto_angle,
        );

        draw_btn(
            &mut canvas,
            btn_p1,
            "我方 (My)",
            st.edit_mode == EditMode::P1,
        )?;
        draw_btn(
            &mut canvas,
            btn_e1,
            "敌方 (Enemy)",
            st.edit_mode == EditMode::E1,
        )?;
        draw_btn(
            &mut canvas,
            btn_mode_switch,
            if st.big_map { "MINI MAP" } else { "BIG MAP" },
            false,
        )?;

        let lock_label = if st.locked_px_per_unit.is_some() {
            "[已锁定] 解锁尺子"
        } else {
            "[未锁定] 锁定距离尺"
        };
        draw_btn(
            &mut canvas,
            btn_lock_ruler,
            lock_label,
            st.locked_px_per_unit.is_some(),
        )?;

        let ruler_lbl = if st.edit_mode == EditMode::DrawRuler1 {
            "[步骤1] 点击左边线"
        } else if st.edit_mode == EditMode::DrawRuler2 {
            "[步骤2] 点击右边线"
        } else {
            "C: 框定 1屏幕宽"
        };
        draw_btn(
            &mut canvas,
            btn_draw_ruler,
            ruler_lbl,
            st.edit_mode == EditMode::DrawRuler1 || st.edit_mode == EditMode::DrawRuler2,
        )?;
        draw_btn(&mut canvas, btn_clear, "清空手动标记", false)?;

        // 大地图整屏距离档位(小地图模式下灰显不响应)
        draw_btn(
            &mut canvas,
            btn_u12,
            "12",
            st.big_map && (st.map_units - 12.0).abs() < 0.1,
        )?;
        draw_btn(
            &mut canvas,
            btn_u16,
            "16",
            st.big_map && (st.map_units - 16.0).abs() < 0.1,
        )?;
        draw_btn(
            &mut canvas,
            btn_u18,
            "18",
            st.big_map && (st.map_units - 18.0).abs() < 0.1,
        )?;
        draw_btn(
            &mut canvas,
            btn_u20,
            "20",
            st.big_map && (st.map_units - 20.0).abs() < 0.1,
        )?;
        draw_btn(
            &mut canvas,
            btn_u22,
            "22",
            st.big_map && (st.map_units - 22.0).abs() < 0.1,
        )?;
        // Draw preset angle buttons
        draw_btn(
            &mut canvas,
            btn_a20,
            "20°",
            (st.current_angle - 20.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a30,
            "30°",
            (st.current_angle - 30.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a45,
            "45°",
            (st.current_angle - 45.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a50,
            "50°",
            (st.current_angle - 50.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a60,
            "60°",
            (st.current_angle - 60.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a65,
            "65°",
            (st.current_angle - 65.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a70,
            "70°",
            (st.current_angle - 70.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a75,
            "75°",
            (st.current_angle - 75.0).abs() < 0.5,
        )?;

        draw_btn(&mut canvas, btn_ang_m5, "-5", false)?;
        draw_btn(&mut canvas, btn_ang_minus, "-1", false)?;
        draw_btn(
            &mut canvas,
            rect_ang_text,
            &format!("{:.0}°", st.current_angle),
            true,
        )?;
        draw_btn(&mut canvas, btn_ang_plus, "+1", false)?;
        draw_btn(&mut canvas, btn_ang_p5, "+5", false)?;

        draw_btn(&mut canvas, btn_wind_m1, "-1.0", false)?;
        draw_btn(&mut canvas, btn_wind_m01, "-0.1", false)?;

        let wind_str = if !wind_input_buf.is_empty() {
            format!("缓冲: {}_", wind_input_buf)
        } else {
            let net_tag = if st.wind_net { "N" } else { "" };
            let flip_tag = if st.wind_flip { " [翻转]" } else { "" };
            format!("风: {:.1}{} {}", st.wind, net_tag, flip_tag)
        };
        let _ = imgproc::put_text(
            &mut canvas,
            &wind_str,
            core::Point::new(rect_wind_text.x - 10, rect_wind_text.y + 20),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.45,
            core::Scalar::new(0.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        );
        draw_btn(&mut canvas, btn_wind_p01, "+0.1", false)?;
        draw_btn(&mut canvas, btn_wind_p1, "+1.0", false)?;

        let hint_txt = "网络自动; 手动先点我方/敌方 Shift=语音";
        let _ = imgproc::put_text(
            &mut canvas,
            hint_txt,
            core::Point::new(map_w_display + 5, 420),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.38,
            core::Scalar::new(180.0, 255.0, 180.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        );

        imgproc::rectangle(
            &mut canvas,
            btn_exit,
            core::Scalar::new(0.0, 0.0, 220.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::put_text(
            &mut canvas,
            "X EXIT",
            core::Point::new(btn_exit.x + 15, btn_exit.y + 20),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        // 目标轮换钮:标签直接显示当前目标名(网络局才可用)
        {
            let net = net_state.lock().unwrap().clone();
            let sel_uid = net.target_id.or_else(|| {
                net.pos
                    .iter()
                    .find(|(k, _)| Some(**k) != net.my_id)
                    .map(|(k, _)| *k)
            });
            let tgt_label = sel_uid
                .and_then(|u| net.names.get(&u))
                .map(|n| {
                    let s: String = n.chars().take(4).collect();
                    format!("▶{}", s)
                })
                .unwrap_or_else(|| "▶--".to_string());
            // 醒目:敌人标记同色的品红底 + 亮字,一眼锁定当前目标
            let has_tgt = sel_uid.is_some();
            imgproc::rectangle(
                &mut canvas,
                btn_target,
                if has_tgt {
                    core::Scalar::new(220.0, 0.0, 220.0, 0.0)
                } else {
                    core::Scalar::new(80.0, 80.0, 80.0, 0.0)
                },
                -1,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::rectangle(
                &mut canvas,
                btn_target,
                core::Scalar::new(255.0, 180.0, 255.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::put_text(
                &mut canvas,
                &tgt_label,
                core::Point::new(btn_target.x + 8, btn_target.y + 21),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.55,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )?;
        }

        let t_ui = t2.elapsed().as_millis();
        let bg_cap_ms = *cap_time_ms.lock().unwrap();
        let fps = 1000.0 / fps_t0.elapsed().as_millis().max(1) as f64;
        fps_t0 = std::time::Instant::now();
        
        let perf_txt = format!("FPS:{:.0} | IO:{} Rec:{} UI:{} BG:{}", fps, t_io, t_recog, t_ui, bg_cap_ms);
        let _ = imgproc::put_text(
            &mut canvas,
            &perf_txt,
            core::Point::new(map_w_display + 5, 435),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.55,
            core::Scalar::new(255.0, 100.0, 100.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        );

        if !voice_status.is_empty() {
            let _ = imgproc::put_text(
                &mut canvas,
                &voice_status,
                core::Point::new(map_w_display + 5, 455),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.45,
                core::Scalar::new(0.0, 255.0, 200.0, 0.0),
                1,
                imgproc::LINE_AA,
                false,
            );
        }

        // 网络实况行:隧道连着或有对局数据才显示
        {
            let net = net_state.lock().unwrap();
            if net.battle_id > 0 || net.conns > 0 {
                let link = if net.conns > 0 { "NET" } else { "NET(断)" };
                let turn_s = match net.active {
                    Some(a) if Some(a) == net.my_id => "我方回合",
                    Some(_) => "敌方回合",
                    None => "",
                };
                // raw 符号=风向已验证(6/6):正→向右吹,负→向左;大小仍未解码
                let wind_dir = if net.wind_seed > 0 {
                    "→"
                } else if net.wind_seed < 0 {
                    "←"
                } else {
                    "-"
                };
                let flip_n = if st.wind_flip { -1 } else { 1 };
                let rel = match wind_dir_hint(&net).map(|h| h * flip_n) {
                    Some(1) => "顺",
                    Some(-1) => "逆",
                    _ => "",
                };
                let auto_s = net
                    .auto_wind10
                    .map(|m| format!(" 风~{:.1}", m as f64 / 10.0))
                    .unwrap_or_default();
                let mut info = format!(
                    "{link} 回合{} {turn_s} 风向{wind_dir}{rel}{auto_s} raw:{} lag:{}ms",
                    net.round, net.wind_seed, net.last_lag_ms
                );
                // 采集信息:实时识别角度(和屏幕数字直接对照)
                if let Some(val) = power_recognized_val {
                    info += &format!(" 读{}°", val);
                }
                if let Some((id, Some(ang))) = net.last_fire {
                    let who = if Some(id) == net.my_id { "我" } else { "敌" };
                    info += &format!(" 上炮{who}:{:.0}°", ang);
                }
                if let Some((a, pw)) = net.last_fire_req {
                    info += &format!(" REQ:{:.0}°/{:.1}", a, pw);
                }
                let info_y = (canvas_h_target - 12).max(470);
                let _ = imgproc::put_text(
                    &mut canvas,
                    &info,
                    core::Point::new(map_w_display + 5, info_y),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.4,
                    core::Scalar::new(120.0, 220.0, 255.0, 0.0),
                    1,
                    imgproc::LINE_AA,
                    false,
                );
            }
        }

        highgui::imshow(window_name, &canvas)?;
        if first_show {
            highgui::set_window_property(window_name, highgui::WND_PROP_TOPMOST, 1.0)?;
            first_show = false;
        }

        let key = highgui::wait_key(15)?;
        let debounce_ok = last_toggle_time.elapsed() > std::time::Duration::from_millis(300);
        let is_visible =
            highgui::get_window_property(window_name, highgui::WND_PROP_VISIBLE).unwrap_or(1.0);
        let exit_req = app_state.lock().unwrap().exit_requested;
        if key == 27 || key == 'q' as i32 || is_visible < 1.0 || exit_req {
            break;
        } else if key == 13 || key == 10 || key == 3 {
            // Enter: Set Wind (macOS 小键盘回车发的是 \x03=3)
            if !wind_input_buf.is_empty() {
                if let Ok(w) = wind_input_buf.parse::<f64>() {
                    let flip_ui = app_state.lock().unwrap().wind_flip;
                    let mut net = net_state.lock().unwrap();
                    if net.wind_seed != 0 {
                        let w10 = (w.abs() * 10.0).round() as i64;
                        let c = (net.wind_seed.abs() & 0xff) ^ w10 ^ net.round as i64;
                        push_wind_vote(&mut net, c);
                    }
                    let applied = if auto_wind_sign {
                        apply_auto_wind_sign(&net, w, flip_ui)
                    } else {
                        w
                    };
                    log_wind_pair(
                        "MANUAL",
                        net.battle_id,
                        net.round,
                        net.active,
                        net.wind_seed,
                        Some(w),
                        Some(applied),
                    );
                    drop(net);
                    let mut st = app_state.lock().unwrap();
                    st.wind = applied;
                    st.wind_net = false;
                }
                wind_input_buf.clear();
            }
        } else if key == 32 {
            // Space: Set Angle
            if !wind_input_buf.is_empty() {
                if let Ok(a) = wind_input_buf.parse::<f64>() {
                    let mut st = app_state.lock().unwrap();
                    st.current_angle = a.clamp(0.0, 180.0);
                    st.auto_angle = false;
                }
                wind_input_buf.clear();
            }
        } else if key == 8 || key == 127 {
            // Backspace
            wind_input_buf.pop();
        } else if debounce_ok && (key == 'z' as i32 || key == 'Z' as i32) {
            // 快捷键 Z: 切换我方标注模式 (EditMode::P1)
            let mut st = app_state.lock().unwrap();
            st.edit_mode = if st.edit_mode == EditMode::P1 { EditMode::None } else { EditMode::P1 };
            last_toggle_time = std::time::Instant::now();
        } else if debounce_ok && (key == 'f' as i32 || key == 'F' as i32) {
            // 快捷键 F: 风向翻转——极端大风等手动修正场景,强制反转自动附加的顺/逆符号
            let mut st = app_state.lock().unwrap();
            st.wind_flip = !st.wind_flip;
            last_toggle_time = std::time::Instant::now();
        } else if debounce_ok && (key == 'x' as i32 || key == 'X' as i32) {
            // 快捷键 X: 切换敌方标注模式 (EditMode::E1)
            let mut st = app_state.lock().unwrap();
            st.edit_mode = if st.edit_mode == EditMode::E1 { EditMode::None } else { EditMode::E1 };
            last_toggle_time = std::time::Instant::now();
        } else if debounce_ok && (key == 'c' as i32 || key == 'C' as i32) {
            // 快捷键 C: 进入标尺框定(等效点"框定"按钮) → 先点左边界、再点右边界。
            // 注意不回退:按住不放会产生重复按键,框定过程中忽略后续 C
            let mut st = app_state.lock().unwrap();
            if st.edit_mode != EditMode::DrawRuler1 && st.edit_mode != EditMode::DrawRuler2 {
                st.edit_mode = EditMode::DrawRuler1;
                last_toggle_time = std::time::Instant::now();
            }
        } else if debounce_ok && (key == 'n' as i32 || key == 'N' as i32) {
            // 快捷键 N: 清空手动标记
            let mut st = app_state.lock().unwrap();
            st.manual_p1 = None;
            st.manual_e1 = None;
            st.edit_mode = EditMode::None;
            last_toggle_time = std::time::Instant::now();
        } else if debounce_ok && (key == 'r' as i32 || key == 'R' as i32) {
            // 快捷键 R: 锁定/解锁距离尺
            let mut st = app_state.lock().unwrap();
            if st.locked_px_per_unit.is_some() {
                st.locked_px_per_unit = None;
            } else {
                st.locked_px_per_unit = Some(0.0);
            }
            last_toggle_time = std::time::Instant::now();
        } else if debounce_ok && (key == 'm' as i32 || key == 'M' as i32) {
            let mut st = app_state.lock().unwrap();
            st.is_fixed_angle = !st.is_fixed_angle;
            last_toggle_time = std::time::Instant::now();
        } else if key == 65362 || key == 0x260000 || key == 82 {
            // Up arrow
            if let Ok(mut m_state) = app_state.lock() {
                m_state.current_angle = (m_state.current_angle + 1.0).min(180.0);
                m_state.auto_angle = false;
            }
        } else if key == 65364 || key == 0x280000 || key == 84 {
            // Down arrow
            if let Ok(mut m_state) = app_state.lock() {
                m_state.current_angle = (m_state.current_angle - 1.0).max(0.0);
                m_state.auto_angle = false;
            }
        } else if key > 0 {
            let ch = (key & 0xFF) as u8 as char;
            if ch.is_ascii_digit() || ch == '.' || ch == '-' {
                wind_input_buf.push(ch);
            } else {
                // macOS waitKey 可能返回虚拟键码或全角字符(中文输入法):
                // 主键盘数字行 0x12-0x1D,小键盘 0x52-0x5C,全角数字 0xFF10-0xFF19
                let w = (key & 0xFFFF) as u32;
                let mapped = match w {
                    0xFF10..=0xFF19 => char::from_u32(w - 0xFF10 + 0x30),
                    // 主键盘数字行(mac 虚拟键码)
                    0x12 => Some('1'), 0x13 => Some('2'), 0x14 => Some('3'),
                    0x15 => Some('4'), 0x17 => Some('5'), 0x16 => Some('6'),
                    0x1A => Some('7'), 0x1C => Some('8'), 0x19 => Some('9'),
                    0x1D => Some('0'),
                    // 小键盘数字
                    0x52 => Some('0'), 0x53 => Some('1'), 0x54 => Some('2'),
                    0x55 => Some('3'), 0x56 => Some('4'), 0x57 => Some('5'),
                    0x58 => Some('6'), 0x59 => Some('7'), 0x5B => Some('8'),
                    0x5C => Some('9'),
                    // 符号键
                    0x1B => Some('-'), 0x2F => Some('.'),
                    0x41 => Some('.'), 0x43 => Some('-'),
                    _ => None,
                };
                match mapped {
                    Some(c) => wind_input_buf.push(c),
                    None => println!("⌨️ 未识别按键 key={key:#06x}"),
                }
            }
        }

        // 运行时切换大/小地图模式:交互框选新区域 → 换图源 → 重算缩放和比例尺。
        // 点位属于旧图,全部作废;大地图自动按档位锁尺,小地图重新画尺子。
        if app_state.lock().unwrap().switch_requested {
            app_state.lock().unwrap().switch_requested = false;
            let to_big = !app_state.lock().unwrap().big_map;
            println!(
                "👉 [切换模式] 请框选【{}】区域...",
                if to_big { "整张游戏地图" } else { "左上角小地图" }
            );
            let crop_path = "/tmp/tnt_mode_switch.png";
            if select_crop_interactive(crop_path) {
                if let Ok(new_img) = imgcodecs::imread(crop_path, imgcodecs::IMREAD_COLOR) {
                    if !new_img.empty() {
                        let nw = new_img.cols();
                        let nh = new_img.rows();
                        let geo = find_screen_position(&new_img).unwrap_or((0, 0, nw, nh));
                        *map_geo_shared.lock().unwrap() = geo;
                        *shared_map.lock().unwrap() = None; // 丢弃旧区域画面,等新图
                        let new_scale = (map_w_display as f64 / nw as f64).min(2.0);
                        let mut stg = app_state.lock().unwrap();
                        stg.big_map = to_big;
                        stg.src_w = nw;
                        stg.src_h = nh;
                        stg.disp_scale = new_scale;
                        stg.manual_p1 = None;
                        stg.manual_e1 = None;
                        stg.manual_cam_rect = None;
                        if to_big {
                            stg.locked_px_per_unit = Some(nw as f64 / stg.map_units);
                        } else {
                            stg.locked_px_per_unit = None;
                        }
                        println!(
                            "✅ 已切换到{} ({}x{}, 显示缩放 {:.2})",
                            if to_big { "大地图" } else { "小地图" },
                            nw,
                            nh,
                            new_scale
                        );
                    }
                }
            } else {
                println!("⚠️ 未完成框选,保持原模式");
            }
        }
    }

    is_running.store(false, std::sync::atomic::Ordering::Relaxed);
    for mut c in voice_children {
        let _ = c.kill();
    }
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tnt_comput::proto::PlayerInit;

    #[test]
    fn flight_consumes_pending_fire_without_reassigning_identity() {
        let mut st = NetHud {
            my_id: Some(74108),
            pending_firereq: true,
            ..Default::default()
        };
        st.pos.insert(74108, (421, 882));
        st.pos.insert(72574, (1704, 718));

        apply_net_event(
            &mut st,
            &GameEvent::Flight {
                id: 74108,
                path: vec![(685, 463)],
            },
            "1111rust",
        );
        assert_eq!(st.my_id, Some(74108));
        assert!(!st.pending_firereq);
        assert_eq!(st.pos.get(&74108), Some(&(685, 463)));

        apply_net_event(
            &mut st,
            &GameEvent::Fire {
                id: 72574,
                x: 1704,
                y: 718,
                angle: None,
                shots: 1,
            },
            "1111rust",
        );
        assert_eq!(st.my_id, Some(74108));
        assert_eq!(st.pos.get(&72574), Some(&(1704, 718)));
    }

    #[test]
    fn init_clears_stale_identity_when_name_is_absent() {
        let mut st = NetHud {
            my_id: Some(72574),
            ..Default::default()
        };
        apply_net_event(
            &mut st,
            &GameEvent::Init {
                map_w: 2000,
                map_h: 1000,
                players: vec![PlayerInit {
                    id: 72574,
                    name: "someone_else".to_string(),
                    x: 100,
                    y: 200,
                    angle_hint: 0,
                }],
            },
            "1111rust",
        );
        assert_eq!(st.my_id, None);
    }

    #[test]
    fn plain_click_cannot_override_active_network_battle() {
        assert!(!allow_plain_manual_mark(true));
    }

    #[test]
    fn plain_click_still_works_without_network_battle() {
        assert!(allow_plain_manual_mark(false));
    }
}
