use opencv::{core, highgui, imgcodecs, imgproc, prelude::*};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use tnt_comput::net::{decode_ws_message, tunnel, Tracker, TunnelEvent};
use tnt_comput::proto::GameEvent;
#[path = "../hud_window.rs"]
mod hud_window;
#[path = "../hud_theme.rs"]
mod hud_theme;

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
    /// 回合风原始字段 f4(客户端 XOR 输入,仅诊断)
    wind_seed: i64,
    /// 每个 WindSeed 自增,渲染线程据此判定"新回合"并应用自动风
    turn_seq: u64,
    /// 本回合客户端公式解出的世界风速(×10,带符号);None=Action 缺字段
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

const MAP_PANE_WIDTH: i32 = 440;
const MAP_MAX_WIDTH: i32 = 392;
const MAP_MAX_HEIGHT: i32 = 398;
const SIDEBAR_WIDTH: i32 = 390;
const HUD_HEIGHT: i32 = 680;
const MAX_TARGETS: usize = 8;

fn compact_map_scale(width: i32, height: i32) -> f64 {
    (MAP_MAX_WIDTH as f64 / width.max(1) as f64)
        .min(MAP_MAX_HEIGHT as f64 / height.max(1) as f64).min(2.0)
}

fn map_display_rect(width: i32, height: i32, scale: f64) -> core::Rect {
    let size = map_display_size(width, height, scale);
    core::Rect::new(24 + (MAP_MAX_WIDTH - size.width)/2,
        52 + (MAP_MAX_HEIGHT - size.height)/2, size.width, size.height)
}

fn source_ruler_bounds(left: i32, right: i32, width: i32, map_left: i32, scale: f64) -> (i32,i32) {
    let x = (((left - map_left) as f64 / scale).round() as i32).clamp(0,width.saturating_sub(1));
    let end = (((right - map_left) as f64 / scale).round() as i32).clamp(x+1,width);
    (x,end)
}

fn map_display_size(width: i32, height: i32, scale: f64) -> core::Size {
    core::Size::new((width as f64 * scale).round().max(1.0) as i32,
                    (height as f64 * scale).round().max(1.0) as i32)
}

fn opponent_ids(net: &NetHud) -> Vec<u64> {
    if net.my_id.is_none() { return Vec::new(); }
    let mut ids: Vec<_> = net.pos.keys().copied().filter(|id| Some(*id) != net.my_id).collect();
    ids.sort_unstable();
    ids
}

fn selected_target_id(net: &NetHud) -> Option<u64> {
    net.target_id.filter(|id| Some(*id) != net.my_id && net.pos.contains_key(id))
        .or_else(|| opponent_ids(net).first().copied())
}

fn target_position(net: &NetHud) -> Option<(i64, i64)> {
    net.pos.get(&selected_target_id(net)?).copied()
}

fn target_slot_rect(index: usize) -> Option<core::Rect> {
    (index < MAX_TARGETS).then(|| core::Rect::new(
        MAP_PANE_WIDTH + 16 + (index % 2) as i32 * 184,
        386 + (index / 2) as i32 * 48, 174, 40))
}

fn target_at_sidebar(net: &NetHud, x: i32, y: i32) -> Option<u64> {
    target_from_rows(&opponent_ids(net), x, y)
}

fn target_from_rows(ids: &[u64], x: i32, y: i32) -> Option<u64> {
    ids.iter().copied().take(MAX_TARGETS).enumerate()
        .find_map(|(i, uid)| is_inside(x, y, target_slot_rect(i)?).then_some(uid))
}

fn released_target(net: &NetHud, pressed: u64, x: i32, y: i32) -> Option<u64> {
    (target_at_sidebar(net, x, y) == Some(pressed)).then_some(pressed)
}

fn target_at_map(net: &NetHud, x: i32, y: i32, size: core::Size) -> Option<u64> {
    if net.map_w <= 0 || net.map_h <= 0 { return None; }
    let mut best = None;
    let mut best_distance = 30.0_f64.powi(2);
    for uid in opponent_ids(net) {
        let &(wx, wy) = net.pos.get(&uid)?;
        let dx = wx as f64 * size.width as f64 / net.map_w as f64 - x as f64;
        let dy = wy as f64 * size.height as f64 / net.map_h as f64 - y as f64;
        let d = dx * dx + dy * dy;
        if d < best_distance { best = Some(uid); best_distance = d; }
    }
    best
}

fn hud_controls() -> [core::Rect; 32] {
    let sx = MAP_PANE_WIDTH;
    [
        (28,594,118,26), (154,594,118,26), (302,570,110,22),
        (28,628,150,26), (188,628,224,26), (sx+282,350,92,24), (280,594,132,26),
        (196,462,38,22), (240,462,38,22), (284,462,38,22), (328,462,38,22), (372,462,38,22),
        (28,516,42,26), (76,516,42,26), (124,516,42,26), (172,516,42,26),
        (220,516,42,26), (268,516,42,26), (316,516,42,26), (364,516,42,26),
        (sx+26,272,32,22), (sx+66,272,32,22), (sx+28,236,150,30), (sx+106,272,32,22), (sx+146,272,32,22),
        (sx+210,272,34,22), (sx+250,272,34,22), (sx+212,236,150,30), (sx+290,272,34,22), (sx+330,272,34,22),
        (sx+16,314,296,28), (sx+320,314,54,28),
    ].map(|(x,y,w,h)| core::Rect::new(x,y,w,h))
}

fn fit_text(text: &str, width: i32, scale: f64, thickness: i32) -> opencv::Result<String> {
    hud_theme::fit(text,width,scale,thickness)
}

fn draw_target_list(canvas: &mut core::Mat, net: &NetHud) -> opencv::Result<()> {
    use hud_theme as t;
    let ids=opponent_ids(net);
    t::text(canvas,&format!("选择对手  {}",ids.len().min(MAX_TARGETS)),MAP_PANE_WIDTH+16,370,250,0.47,t::TEXT)?;
    let selected=selected_target_id(net);
    for i in 0..MAX_TARGETS {
        let r=target_slot_rect(i).unwrap();
        let uid=ids.get(i).copied();
        let active=uid.is_some() && uid==selected;
        t::rounded(canvas,r,9,if active {t::ACCENT} else {t::BORDER})?;
        t::rounded(canvas,core::Rect::new(r.x+1,r.y+1,r.width-2,r.height-2),8,
            if active {t::ACCENT_DARK} else {t::SURFACE})?;
        t::rounded(canvas,core::Rect::new(r.x+10,r.y+10,20,20),5,
            if active {t::ACCENT} else {t::RAISED})?;
        t::text(canvas,&(i+1).to_string(),r.x+16,r.y+25,18,0.37,
            if active {t::BACKGROUND} else {t::MUTED})?;
        let name=uid.and_then(|id|net.names.get(&id).filter(|s|!s.trim().is_empty()).cloned()
            .or_else(||Some(format!("ID {id}")))).unwrap_or_else(||"等待对手".into());
        t::text(canvas,&name,r.x+40,r.y+25,r.width-50,0.42,
            if active {t::ACCENT} else if uid.is_some() {t::TEXT} else {t::MUTED})?;
    }
    Ok(())
}

fn sidebar_text(canvas: &mut core::Mat, text: &str, y: i32, scale: f64, color: core::Scalar) -> opencv::Result<()> {
    let text=fit_text(text,SIDEBAR_WIDTH-56,scale,1)?;
    imgproc::put_text(canvas,&text,core::Point::new(MAP_PANE_WIDTH+28,y),
        imgproc::FONT_HERSHEY_SIMPLEX,scale,color,1,imgproc::LINE_AA,false)
}

fn aim_card_rect() -> core::Rect { core::Rect::new(MAP_PANE_WIDTH+16,52,SIDEBAR_WIDTH-32,130) }

fn draw_aim_detail(canvas: &mut core::Mat, detail: &str) -> opencv::Result<()> {
    let r=aim_card_rect();
    imgproc::rectangle(canvas,core::Rect::new(r.x,184,r.width,22),hud_theme::color(hud_theme::BACKGROUND),-1,imgproc::LINE_8,0)?;
    hud_theme::text(canvas,detail,r.x+4,198,r.width-8,0.39,hud_theme::MUTED)
}

fn draw_aim_pending(canvas: &mut core::Mat, title: &str, detail: &str) -> opencv::Result<()> {
    use hud_theme as t;
    let r=aim_card_rect();
    t::card(canvas,r)?;
    t::text(canvas,"推荐力度",r.x+18,r.y+25,200,0.42,t::MUTED)?;
    t::text(canvas,"—",r.x+18,r.y+77,150,1.4,t::MUTED)?;
    t::text(canvas,title,r.x+18,r.y+105,r.width-36,0.48,t::TEXT)?;
    draw_aim_detail(canvas,detail)
}

fn draw_aim_ready(canvas: &mut core::Mat, force: f64, angle: f64, dx: f64, dy: f64, fixed: bool, label: &str) -> opencv::Result<()> {
    use hud_theme as t;
    let r=aim_card_rect();
    t::card(canvas,r)?;
    t::text(canvas,label,r.x+18,r.y+25,r.width-36,0.42,t::MUTED)?;
    t::text(canvas,&format!("{force:.1}"),r.x+18,r.y+83,214,1.75,t::POWER)?;
    t::text(canvas,&format!("{angle:.0}°"),r.x+254,r.y+79,86,0.94,t::TEXT)?;
    t::text(canvas,&format!("{}  ·  2/3 力度 {:.1}",if fixed {"固定角度"} else {"推荐角度"},force*2.0/3.0),
        r.x+18,r.y+112,r.width-36,0.43,t::MUTED)?;
    draw_aim_detail(canvas,&format!("水平距离 {:.2}     高差 {:+.2}",dx.abs(),dy))
}

fn draw_hud_background(canvas: &mut core::Mat, net: &NetHud) -> opencv::Result<()> {
    use hud_theme as t;
    t::text(canvas,"TNT",20,29,80,0.72,t::ACCENT)?;
    t::text(canvas,"战况地图",86,28,230,0.43,t::MUTED)?;
    t::text(canvas,"瞄准助手",MAP_PANE_WIDTH+18,30,240,0.6,t::TEXT)?;
    let online=net.conns>0;
    t::rounded(canvas,core::Rect::new(MAP_PANE_WIDTH+300,12,74,24),12,
        if online {t::ACCENT_DARK} else {t::RAISED})?;
    t::text(canvas,if online {"已连接"} else {"等待连接"},MAP_PANE_WIDTH+310,29,58,0.36,
        if online {t::ACCENT} else {t::MUTED})?;
    for r in [core::Rect::new(16,44,408,414),core::Rect::new(16,490,408,62),
              core::Rect::new(16,562,408,102),core::Rect::new(MAP_PANE_WIDTH+16,210,174,92),
              core::Rect::new(MAP_PANE_WIDTH+200,210,174,92),core::Rect::new(MAP_PANE_WIDTH+16,590,358,72)] {
        t::card(canvas,r)?;
    }
    t::text(canvas,"角度预设",28,507,240,0.38,t::MUTED)?;
    t::text(canvas,"地图与标尺",28,582,250,0.42,t::TEXT)?;
    draw_aim_pending(canvas,"等待对局数据","进入对局后自动同步")
}

fn draw_hud_controls(canvas: &mut core::Mat, st: &AppState, live_angle: Option<f64>, ocr: Option<i32>, wind_input: &str) -> opencv::Result<()> {
    use hud_theme as t;
    let [p1,e1,mode,lock,ruler,next,clear,u12,u16,u18,u20,u22,a20,a30,a45,a50,a60,a65,a70,a75,
        m5,minus,angle_value,plus,p5,wm1,wm01,wind_value,wp01,wp1,auto,source]=hud_controls();
    for (r,label,active) in [(p1,"标记我方",st.edit_mode==EditMode::P1),
        (e1,"标记对手",st.edit_mode==EditMode::E1),(clear,"清除手动标记",false),
        (mode,if st.big_map {"切换小地图"} else {"切换大地图"},false),
        (lock,if st.locked_px_per_unit.is_some() {"标尺已锁定"} else {"锁定标尺"},st.locked_px_per_unit.is_some()),
        (ruler,match st.edit_mode {EditMode::DrawRuler1=>"点击左边界",EditMode::DrawRuler2=>"点击右边界",_=>"框定一屏宽度"},
            matches!(st.edit_mode,EditMode::DrawRuler1|EditMode::DrawRuler2)),(next,"下一位",false)] {
        draw_btn(canvas,r,label,active)?;
    }
    if st.big_map {
        t::text(canvas,"地图宽度",28,478,155,0.4,t::MUTED)?;
        for (r,n) in [(u12,12.0),(u16,16.0),(u18,18.0),(u20,20.0),(u22,22.0)] {
            draw_btn(canvas,r,&format!("{n:.0}"),(st.map_units-n).abs()<0.1)?;
        }
    } else {
        t::text(canvas,"小地图 · 精确坐标同步",28,478,360,0.39,t::MUTED)?;
    }
    for (r,n) in [(a20,20.0),(a30,30.0),(a45,45.0),(a50,50.0),(a60,60.0),(a65,65.0),(a70,70.0),(a75,75.0)] {
        draw_btn(canvas,r,&format!("{n:.0}°"),(st.current_angle-n).abs()<0.5)?;
    }
    t::text(canvas,"角度",angle_value.x,229,74,0.42,t::MUTED)?;
    t::text(canvas,if st.auto_angle {"自动"} else {"手动"},angle_value.x+111,229,45,0.34,t::ACCENT)?;
    let value=if !st.auto_angle || live_angle.is_some() || ocr.is_some() {format!("{:.0}°",st.current_angle)} else {"—".into()};
    t::text(canvas,&value,angle_value.x,263,angle_value.width,0.96,t::TEXT)?;
    t::text(canvas,if !wind_input.is_empty() {"风速 · 输入中"} else if st.wind_flip {"风速 · 翻转"} else if st.wind_net {"风速 · 自动"} else {"风速 · 手动"},
        wind_value.x,229,wind_value.width,0.4,t::MUTED)?;
    let wind=if wind_input.is_empty() {format!("{:+.1}",st.wind)} else {format!("{wind_input}_")};
    t::text(canvas,&wind,wind_value.x,263,wind_value.width,0.96,t::TEXT)?;
    for (r,label) in [(m5,"-5"),(minus,"-1"),(plus,"+1"),(p5,"+5"),(wm1,"-1.0"),(wm01,"-0.1"),(wp01,"+0.1"),(wp1,"+1.0")] {
        draw_btn(canvas,r,label,false)?;
    }
    draw_btn(canvas,auto,if !st.auto_angle {"恢复自动同步角度"} else if live_angle.is_some() || ocr.is_some() {"角度自动同步中"} else {"等待读取角度"},st.auto_angle)?;
    draw_btn(canvas,source,if st.prefer_memory_angle {"MEM"} else {"OCR"},st.prefer_memory_angle)
}

fn render_hud_preview(path: &str, variant: &str) -> opencv::Result<()> {
    use hud_theme as t;
    let mut canvas=core::Mat::new_rows_cols_with_default(HUD_HEIGHT,MAP_PANE_WIDTH+SIDEBAR_WIDTH,core::CV_8UC3,t::color(t::BACKGROUND))?;
    let mut net=NetHud {my_id:Some(1),target_id:Some(4),map_w:2000,map_h:1000,conns:1,..Default::default()};
    net.pos.insert(1,(200,650));
    for (i,name) in ["远山听雨","Opponent Beta","宁静致远7","很长很长的对手名字测试","Player Five","同名玩家","同名玩家","最后一位对手"].iter().enumerate() {
        let uid=i as u64+2;
        net.names.insert(uid,(*name).into());net.pos.insert(uid,(750+i as i64*120,500+(i%3) as i64*100));
    }
    let st=AppState {edit_mode:EditMode::None,manual_p1:None,manual_e1:None,manual_cam_rect:None,drag_start:None,
        current_angle:51.0,wind:-7.7,locked_px_per_unit:Some(8.0),map_units:12.0,big_map:false,
        disp_scale:compact_map_scale(335,387),src_w:335,src_h:387,auto_angle:true,prefer_memory_angle:true,
        ocr_roi_requested:false,is_fixed_angle:true,wind_flip:false,wind_net:true,switch_requested:false,
        btn_press:None,target_press:None};
    draw_hud_background(&mut canvas,&net)?;
    let map=map_display_rect(st.src_w,st.src_h,st.disp_scale);
    t::rounded(&mut canvas,map,5,[32.0,29.0,24.0])?;
    for y in (map.y+20..map.y+map.height-12).step_by(40) {
        imgproc::line(&mut canvas,core::Point::new(map.x+6,y),core::Point::new(map.x+map.width-6,y),t::color([44.0,40.0,34.0]),1,imgproc::LINE_AA,0)?;
    }
    for x in (map.x+20..map.x+map.width-12).step_by(40) {
        imgproc::line(&mut canvas,core::Point::new(x,map.y+6),core::Point::new(x,map.y+map.height-6),t::color([44.0,40.0,34.0]),1,imgproc::LINE_AA,0)?;
    }
    let point=|x:f64,y:f64|core::Point::new(map.x+(x*map.width as f64) as i32,map.y+(y*map.height as f64) as i32);
    for i in 0..65 {
        let x=i as f64/64.0;
        let p=point(0.1+x*0.79,0.68-1.6*x*(1.0-x));
        if i%2==0 {imgproc::circle(&mut canvas,p,1,t::color(t::ACCENT),-1,imgproc::LINE_AA,0)?;}
    }
    for i in 0..8 {
        let pt=point(0.56+(i%3) as f64*0.09,0.59+(i/3) as f64*0.09);
        let c=if i==2 {t::ACCENT} else {t::MUTED};
        imgproc::circle(&mut canvas,pt,6,t::color(c),2,imgproc::LINE_AA,0)?;
        t::text(&mut canvas,&(i+1).to_string(),pt.x+9,pt.y-6,20,0.36,c)?;
    }
    let me=point(0.1,0.68);
    imgproc::circle(&mut canvas,me,6,t::color(t::TEXT),2,imgproc::LINE_AA,0)?;
    t::text(&mut canvas,"我方",me.x-12,me.y+23,60,0.36,t::TEXT)?;
    match variant {
        "scale"=>draw_aim_pending(&mut canvas,"体型变化中","稳定后自动切换弹道模型")?,
        "expanded"=>draw_aim_ready(&mut canvas,power_for_angle_for_scale(ShotScale::Expanded,51.0,13.9,-3.8,-7.7).unwrap(),51.0,13.9,-3.8,true,"推荐力度 · 放大炮弹")?,
        "waiting"=>draw_aim_pending(&mut canvas,"读数更新中","稍后自动恢复建议")?,
        "recent"=>draw_aim_ready(&mut canvas,54.4,51.0,13.9,-3.8,true,"上次建议 · 读数更新中")?,
        _=>draw_aim_ready(&mut canvas,54.4,51.0,13.9,-3.8,true,"推荐力度")?,
    }
    draw_hud_controls(&mut canvas,&st,Some(51.0),None,"")?;
    draw_target_list(&mut canvas,&net)?;
    sidebar_text(&mut canvas,"按住 Shift 说出风速",615,0.39,t::color(t::MUTED))?;
    sidebar_text(&mut canvas,"在线 · 第 5 回合 · 我方回合",644,0.39,t::color(t::ACCENT))?;
    t::text(&mut canvas,"地图和名单均可点选",20,677,330,0.34,t::MUTED)?;
    imgcodecs::imwrite(path,&canvas,&core::Vector::new())?;
    Ok(())
}

/// 只读 Android 进程中 BattleRole 的逻辑坐标，仅作为网络坐标的短期覆盖层。
/// 不写入 NetHud，避免污染历史抓包记录及原有网络定位状态。
#[derive(Clone)]
struct LiveXY {
    updated: std::time::Instant,
    pos: HashMap<u64, (i64, i64)>,
}

/// 默认启用本地人物图形坐标；用户可以通过 TNT_LIVE_XY=0 关闭。
/// 大小写不敏感，兼容常见环境变量表示；未设置时也启用。
fn live_xy_enabled_from_env(value: Option<&str>) -> bool {
    !value.is_some_and(|v| {
        ["0", "false", "off", "no"]
            .iter()
            .any(|disabled| v.trim().eq_ignore_ascii_case(disabled))
    })
}

/// 独立只读 Python 探针的严格行协议；None 表示非协议行，Some(None) 表示清空。
fn parse_live_xy_line(line: &str) -> Option<Option<HashMap<u64, (i64, i64)>>> {
    if line.trim() == "TNTXY-" {
        return Some(None);
    }
    let mut it = line.split_whitespace();
    if it.next()? != "TNTXY" {
        return None;
    }
    let pid = it.next()?.parse::<u32>().ok()?;
    if pid == 0 {
        return None;
    }
    let count = it.next()?.parse::<usize>().ok()?;
    if count == 0 || count > 128 {
        return None;
    }
    let mut pos = HashMap::new();
    for _ in 0..count {
        let mut parts = it.next()?.split(':');
        let id = parts.next()?.parse::<u64>().ok()?;
        let x = parts.next()?.parse::<i64>().ok()?;
        let y = parts.next()?.parse::<i64>().ok()?;
        if parts.next().is_some() || id == 0 || !(-100000..=100000).contains(&x)
            || !(-100000..=100000).contains(&y)
            || pos.insert(id, (x, y)).is_some()
        {
            return None;
        }
    }
    if it.next().is_some() {
        return None;
    }
    Some(Some(pos))
}

/// 只覆盖【我方】在 Qt 图形对象上的当前坐标。
/// 敌方由原有网络事件驱动，保持不变；不创建未知 ID。
fn overlay_live_xy(net: &mut NetHud, live: Option<&LiveXY>) -> usize {
    if net.conns <= 0 || net.map_w <= 0 || net.map_h <= 0 {
        return 0;
    }
    let (Some(live), Some(my_id)) = (live, net.my_id) else {
        return 0;
    };
    if live.updated.elapsed() > std::time::Duration::from_secs(3) {
        return 0;
    }
    if let (Some(&xy), Some(original)) = (live.pos.get(&my_id), net.pos.get_mut(&my_id)) {
        *original = xy;
        1
    } else {
        0
    }
}

/// Optional, short-lived candidate read from the local game's own aiming control.
/// The probe's Qt scene-transform cache may lag the actual game; this experimental
/// source is opt-in only and must never override an explicit manual angle.
#[derive(Clone)]
struct LiveAngle {
    updated: std::time::Instant,
    uid: u64,
    fire_angle: u16,
}

#[derive(Clone)]
struct LiveOrigin {
    updated: std::time::Instant,
    age_ms: u64,
    uid: u64,
    fire_angle: u16,
    player: (f64, f64),
    origin: (f64, f64),
    scale: ShotScale,
}

#[derive(Clone)]
struct OriginHint {
    updated: std::time::Instant,
    uid: u64,
    code: String,
    scale: Option<ShotScale>,
}

fn parse_origin_hint(line: &str) -> Option<OriginHint> {
    let mut p=line.split_whitespace();
    if p.next()? != "TNTORIGINSTATUS" { return None; }
    let pid=p.next()?.parse::<u32>().ok()?;
    let uid=p.next()?.parse::<u64>().ok()?;
    let sampled=p.next()?.parse::<u64>().ok()?;
    let code=p.next()?;
    let scale=p.next()?.parse::<f64>().ok()?;
    let now=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok()?.as_millis() as u64;
    if pid==0 || uid==0 || p.next().is_some() || now.checked_sub(sampled)? > 600
        || !scale.is_finite() || !(0.01..=10.0).contains(&scale)
        || !["ready","scale","angle","moving","read"].contains(&code) { return None; }
    Some(OriginHint {updated:std::time::Instant::now().checked_sub(std::time::Duration::from_millis(now-sampled))?,uid,code:code.into(),scale:ShotScale::from_factor(scale)})
}

fn apply_origin_hint(sample: &mut Option<LiveOrigin>, hint: &OriginHint) {
    // A ready status precedes its pose. Keep the previous timestamp untouched
    // when the source is continuous; do not create an empty frame every poll.
    if hint.code!="ready" || sample.as_ref().is_some_and(|s|s.uid!=hint.uid || Some(s.scale)!=hint.scale) {
        *sample=None;
    }
}

fn origin_pending_message(sample: Option<&LiveOrigin>, hint: Option<&OriginHint>, net: &NetHud, angle: f64) -> (&'static str,&'static str) {
    if let Some(hint)=hint.filter(|h|Some(h.uid)==net.my_id && h.updated.elapsed().as_millis()<=600) {
        match hint.code.as_str() {
            "scale" => return ("体型变化中","稳定后自动切换弹道模型"),
            "angle" => return ("角度同步中","正在读取实际发射角度"),
            "moving" => return ("人物状态变化中","位置稳定后恢复建议"),
            _ => (),
        }
    }
    let Some(s)=sample else { return ("读数更新中","稍后自动恢复建议"); };
    if s.updated.elapsed().as_millis()+u128::from(s.age_ms)>600 { return ("读数更新中","稍后自动恢复建议"); }
    if Some(s.uid)!=net.my_id { return ("等待读取对局身份","进入对局后自动同步"); }
    if (f64::from(s.fire_angle.min(180-s.fire_angle))-angle).abs()>0.01 { return ("角度同步中","正在读取实际发射角度"); }
    ("人物位置同步中","正在对齐实时位置")
}

fn parse_live_origin_line(line: &str) -> Option<Option<LiveOrigin>> {
    if line.trim() == "TNTORIGIN-" { return Some(None); }
    let mut p = line.split_whitespace();
    if p.next()? != "TNTORIGIN" { return None; }
    let pid = p.next()?.parse::<u32>().ok()?;
    let uid = p.next()?.parse::<u64>().ok()?;
    let sampled_ms = p.next()?.parse::<u64>().ok()?;
    let fire_angle = p.next()?.parse::<u16>().ok()?;
    match p.next()? { "0" | "1" => (), _ => return None }; // protocol metadata, never an aiming gate
    let px = p.next()?.parse::<f64>().ok()?;
    let py = p.next()?.parse::<f64>().ok()?;
    let x = p.next()?.parse::<f64>().ok()?;
    let y = p.next()?.parse::<f64>().ok()?;
    let scale=match p.next() {
        Some(value)=>ShotScale::from_factor(value.parse::<f64>().ok()?)?,
        None=>ShotScale::Normal,
    };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok()?.as_millis() as u64;
    let age_ms = now.checked_sub(sampled_ms)?;
    if p.next().is_some() || pid == 0 || uid == 0 || fire_angle > 180 || age_ms > 600
        || ![px, py, x, y].iter().all(|v| v.is_finite() && v.abs() <= 100000.0)
        || (x - px).hypot(y - py) > 40.0*scale.factor() {
        return None;
    }
    Some(Some(LiveOrigin { updated: std::time::Instant::now(), age_ms, uid,
        fire_angle, player: (px, py), origin: (x, y),scale }))
}

// A correction is useful only for the same local pose and current firing angle.
// Body facing is metadata, not a condition on where a projectile may be aimed.
fn live_origin_for_hud(sample: Option<&LiveOrigin>, net: &NetHud, angle: f64) -> Option<(f64, f64)> {
    let s = sample?;
    if net.map_w <= 0 || net.map_h <= 0 || net.my_id != Some(s.uid)
        || s.updated.elapsed().as_millis() + u128::from(s.age_ms) > 600
        || !angle.is_finite() || (f64::from(s.fire_angle.min(180 - s.fire_angle)) - angle).abs() > 0.01 {
        return None;
    }
    let &(px, py) = net.pos.get(&s.uid)?;
    if (s.player.0 - px as f64).abs() > 1.0 || (s.player.1 - py as f64).abs() > 1.0 {
        return None;
    }
    Some(s.origin)
}

#[derive(Debug, PartialEq)]
enum AimOrigin {
    Muzzle((f64, f64)),
    Player,
    Verifying,
}

#[derive(Clone, PartialEq)]
struct AimInputs {
    battle: u64,
    mine: u64,
    target: u64,
    player: (i64,i64),
    enemy: (i64,i64),
    angle: f64,
    wind: f64,
    units: (f64,f64),
    source: (i32,i32),
    fixed: bool,
    scale: ShotScale,
    world_angle: Option<u16>,
}

fn aim_inputs(net: &NetHud, angle: f64, wind: f64, units: (f64,f64), source: (i32,i32), fixed: bool) -> Option<AimInputs> {
    let mine=net.my_id?;
    let target=selected_target_id(net)?;
    Some(AimInputs {battle:net.battle_id,mine,target,player:*net.pos.get(&mine)?,enemy:*net.pos.get(&target)?,
        angle,wind,units,source,fixed,scale:ShotScale::Normal,world_angle:None})
}

struct RecentAim {
    inputs: AimInputs,
    sampled: std::time::Instant,
    force: f64,
    angle: f64,
    dx: f64,
    dy: f64,
}
impl RecentAim {
    fn can_reference(&self, inputs: Option<&AimInputs>, hint: Option<&OriginHint>) -> bool {
        inputs==Some(&self.inputs) && self.sampled.elapsed().as_millis()<=1200
            && hint.is_some_and(|h|h.uid==self.inputs.mine && h.code=="ready" && h.updated.elapsed().as_millis()<=1200)
    }
}

fn resolve_aim_origin(sample: Option<&LiveOrigin>, net: &NetHud, angle: f64, required: bool) -> AimOrigin {
    match live_origin_for_hud(sample, net, angle) {
        Some(xy) => AimOrigin::Muzzle(xy),
        None if required => AimOrigin::Verifying,
        None => AimOrigin::Player,
    }
}

fn update_shot_scale(latched: &mut Option<(u64,u64,ShotScale)>, net: &NetHud, sample: Option<&LiveOrigin>, hint: Option<&OriginHint>) -> ShotScale {
    let Some(uid)=net.my_id else {*latched=None;return ShotScale::Normal;};
    if latched.is_some_and(|(battle,mine,_)|battle!=net.battle_id || mine!=uid) {*latched=None;}
    let observed=sample.filter(|s|s.uid==uid && s.updated.elapsed().as_millis()+u128::from(s.age_ms)<=600).map(|s|s.scale)
        .or_else(||hint.filter(|h|h.uid==uid && h.code=="ready" && h.updated.elapsed().as_millis()<=600).and_then(|h|h.scale));
    if let Some(scale)=observed {*latched=Some((net.battle_id,uid,scale));}
    latched.map(|(_,_,s)|s).unwrap_or(ShotScale::Normal)
}

/// Strict, bounded one-line protocol from stream_live_angle_readonly.py.
/// The reconstructed fireAngle formula is checked again before accepting data.
fn parse_live_angle_line(line: &str) -> Option<Option<LiveAngle>> {
    if line.trim() == "TNTANGLE-" {
        return Some(None);
    }
    let mut parts = line.split_whitespace();
    if parts.next()? != "TNTANGLE" {
        return None;
    }
    let pid = parts.next()?.parse::<u32>().ok()?;
    let uid = parts.next()?.parse::<u64>().ok()?;
    let fire = parts.next()?.parse::<u16>().ok()?;
    let control = parts.next()?.parse::<i32>().ok()?;
    let rotation = parts.next()?.parse::<f64>().ok()?;
    let flipped = match parts.next()? {
        "0" => false,
        "1" => true,
        _ => return None,
    };
    if parts.next().is_some()
        || pid == 0
        || uid == 0
        || fire > 180
        || !(0..=180).contains(&control)
        || !rotation.is_finite()
        || rotation.abs() > 360.0
    {
        return None;
    }
    let expected = if flipped {
        (180.0 - rotation - control as f64).trunc() as i32
    } else {
        (control as f64 - rotation).trunc() as i32
    };
    if expected != i32::from(fire) {
        return None;
    }
    Some(Some(LiveAngle {
        updated: std::time::Instant::now(),
        uid,
        fire_angle: fire,
    }))
}

/// The Python probe verifies a *BattleSpritePlayerSelf* vtable plus its
/// linked role UID and current battle panel before emitting a TNTANGLE row.
/// The network tunnel is a separate optional source: its absence is NOT an
/// identity mismatch. Reject only when a known network identity contradicts
/// the verified local self identity, or when the sample has expired.
/// FireReq 110 degrees folds to the HUD's 70-degree elevation.
fn live_angle_for_hud(sample: Option<&LiveAngle>, my_id: Option<u64>) -> Option<f64> {
    let sample = sample?;
    if my_id.is_some_and(|network_uid| network_uid != sample.uid)
        || sample.updated.elapsed() > std::time::Duration::from_secs(2)
    {
        return None;
    }
    let folded = sample.fire_angle.min(180 - sample.fire_angle);
    Some(f64::from(folded))
}

/// Local angle is the preferred source by default. Explicit environment opt-out
/// keeps the old OCR-only path without spawning the memory reader.
fn live_angle_enabled_from_env(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some(s) if ["0", "false", "off", "no"].iter()
            .any(|off| s.eq_ignore_ascii_case(off)) => false,
        _ => true,
    }
}

fn selected_memory_angle(enabled: bool, prefer_memory: bool,
                         sample: Option<&LiveAngle>, my_id: Option<u64>) -> Option<f64> {
    if enabled && prefer_memory {
        live_angle_for_hud(sample, my_id)
    } else {
        None
    }
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
/// 箭头取客户端解出的世界风符号;投掷方向=敌人相对我方 x 方位。
/// 信息不足(无解码风/零风/无坐标/站位重合)时返回 None。
fn wind_dir_hint(net: &NetHud) -> Option<i32> {
    let arrow = net.auto_wind10?.signum();
    if arrow == 0 {
        return None;
    }
    let my = net.my_id?;
    let (mx, _) = *net.pos.get(&my)?;
    let (ex, _) = target_position(net)?;
    let aim = (ex - mx).signum() as i32;
    if aim == 0 {
        return None;
    }
    Some(arrow * aim)
}

/// 我方新回合自动风的呈现值: |world10|/10 × (hint×F翻转),
/// 方位缺失时退回世界风符号。与渲染循环内联表达式保持一致。
fn auto_wind_present(world10: i32, hint: Option<i32>, flip: bool) -> f64 {
    let flip_n = if flip { -1 } else { 1 };
    let dir = hint.map(|h| h * flip_n).unwrap_or(world10.signum());
    (world10 as i64).abs() as f64 / 10.0 * dir as f64
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
            speed,
            seed,
            wind10,
        } => {
            st.round = *round;
            st.wind_seed = *seed;
            // 客户端同款活动 uid = abs(i32(order[0]));order 全量仍由 XOR 使用
            st.active = order
                .first()
                .map(|&id| (id as i32 as i64).unsigned_abs())
                .filter(|&id| id != 0);
            st.turn_seq += 1;
            // 客户端 BattleCommandAction::init 已在 proto 层解码;缺字段=None
            st.auto_wind10 = *wind10;
            println!(
                "📡 NET WIND raw={seed} speed={speed} order={order:?} active={:?} wind10={:?} world={:?}",
                st.active,
                wind10,
                wind10.map(|w| w as f64 / 10.0)
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
    /// Source preference: memory first (automatic OCR fallback) vs OCR only.
    prefer_memory_angle: bool,
    /// A click on OCR requests a one-time ROI picker on the UI thread.
    ocr_roi_requested: bool,
    is_fixed_angle: bool,
    /// F 键切换:自动风向附加时强制反向(极端大风反向抛等手动修正场景)
    wind_flip: bool,
    /// 当前风速来自网络密钥自动解码(显示 [N] 标记)
    wind_net: bool,
    switch_requested: bool,
    /// 按钮防误触:记录按下位置,松开时确认仍在同一按钮内才生效
    btn_press: Option<(i32, i32)>,
    target_press: Option<(u64, u64)>,
}

use tnt_comput::physics::*;

// Calculate from continuous world coordinates before mini-map rounding. Keep
// the selected ruler (including manual scale) and the existing target choice.
fn network_aim_offsets(
    net: &NetHud,
    manual_points: bool,
    source: core::Size,
    units: (f64, f64),
    origin: Option<(f64, f64)>,
) -> Option<(f64, f64)> {
    if manual_points || net.map_w <= 0 || net.map_h <= 0
        || source.width <= 0 || source.height <= 0
        || ![units.0, units.1].iter().all(|v| v.is_finite() && *v > 0.0) {
        return None;
    }
    let &(px, py) = net.pos.get(&net.my_id?)?;
    let (px, py) = origin.unwrap_or((px as f64, py as f64));
    let (ex, ey) = target_position(net)?;
    Some((
        (ex as f64 - px) * source.width as f64 / net.map_w as f64 / units.0,
        (py - ey as f64) * source.height as f64 / net.map_h as f64 / units.1,
    ))
}

fn compute_fixed_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64, scale: ShotScale) -> Option<(f64, f64)> {
    let mut eff_angle = angle_deg;
    if eff_angle > 90.0 {
        eff_angle = 180.0 - eff_angle;
    }

    let wind_power = wind; // 用户输入的是相对风力（正=顺风，负=逆风），物理引擎统一按向右打计算，所以直接传入即可
    let dist = dx_units.abs();

    // 调用全新的底层物理引擎，同时传入高低差 dy_units
    let final_power = power_for_angle_for_scale(scale, eff_angle, dist, dy_units, wind_power)?;
    Some((final_power, angle_deg))
}

fn compute_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64, scale: ShotScale) -> Option<(f64, f64)> {
    let mut eff_angle = angle_deg;
    if eff_angle > 90.0 {
        eff_angle = 180.0 - eff_angle;
    }

    let is_reverse = dx_units < 0.0;
    let wind_power = wind; // 用户输入的是相对风力（正=顺风，负=逆风）
    let dist = dx_units.abs();

    // 使用新的 power_for_angle 和 calc_angle 传递 dy_units
    let base_power = power_for_angle_for_scale(scale, eff_angle, dist, dy_units, 0.0)?;
    let mut final_angle = calc_angle_for_scale(scale, dist, dy_units, base_power, wind_power, eff_angle);

    final_angle = final_angle.clamp(15.0, 89.0);

    if is_reverse {
        final_angle = 180.0 - final_angle;
    }

    Some((base_power, final_angle))
}

// Keep only the latest exact inputs. No rounding: a changed position, wind,
// angle or mode must immediately produce a fresh result (including None).
struct LastValue<K, V>(Option<(K, V)>);

impl<K: PartialEq, V> LastValue<K, V> {
    fn new() -> Self { Self(None) }

    fn get_or_compute(&mut self, key: K, compute: impl FnOnce() -> V) -> &V {
        if self.0.as_ref().is_none_or(|(previous, _)| previous != &key) {
            self.0 = Some((key, compute()));
        }
        &self.0.as_ref().unwrap().1
    }
}

fn target_relative_to_world_wind(net: &NetHud, wind: f64, fallback_dx: f64) -> f64 {
    // UI wind is relative to player→target. A muzzle can lie beyond a nearby
    // target, so its dx must not silently reverse the actual wind direction.
    let dx=net.my_id.and_then(|uid|net.pos.get(&uid)).zip(target_position(net))
        .map(|(&(px,_),(ex,_))|(ex-px) as f64).filter(|dx|*dx!=0.0).unwrap_or(fallback_dx);
    if dx<0.0 {-wind} else {wind}
}

fn compute_world_trajectory(dx: f64, dy: f64, world_angle: u16, world_wind: f64, fixed: bool, scale: ShotScale) -> Option<(f64,f64)> {
    let angle=f64::from(world_angle);
    let launch_left=angle>90.0;
    let forward_dx=if launch_left {-dx} else {dx};
    if fixed || forward_dx<=0.0 || !(15.0..=89.0).contains(&angle.min(180.0-angle)) {
        return power_for_world_angle_for_scale(scale,angle,dx,dy,world_wind).map(|p|(p,angle));
    }
    let elevation=angle.min(180.0-angle);
    let relative_wind=if launch_left {-world_wind} else {world_wind};
    let (power,adjusted)=compute_trajectory(forward_dx,dy,elevation,relative_wind,scale)?;
    Some((power,if launch_left {180.0-adjusted} else {adjusted}))
}

fn trajectory_draw_angle(angle: f64, dx: f64, actual_world_angle: bool) -> f64 {
    if actual_world_angle {angle}
    else if (dx<0.0 && angle<=90.0) || (dx>=0.0 && angle>90.0) {180.0-angle}
    else {angle}
}

type TrajectoryKey = (f64, f64, f64, f64, bool, ShotScale, Option<u16>);

fn cached_trajectory(
    memo: &mut LastValue<TrajectoryKey, Option<(f64, f64)>>,
    key: TrajectoryKey,
) -> Option<(f64, f64)> {
    *memo.get_or_compute(key, || {
        let (dx, dy, angle, wind, fixed, scale, world_angle) = key;
        if let Some(world_angle)=world_angle {return compute_world_trajectory(dx,dy,world_angle,wind,fixed,scale);}
        if fixed { compute_fixed_trajectory(dx, dy, angle, wind, scale) }
        else { compute_trajectory(dx, dy, angle, wind, scale) }
    })
}

fn same_angle_roi(a: &core::Mat, b: &core::Mat) -> opencv::Result<bool> {
    if a.rows() != b.rows() || a.cols() != b.cols() || a.typ() != b.typ() {
        return Ok(false);
    }
    Ok(core::norm2(a, b, core::NORM_INF, &core::no_array())? == 0.0)
}

// A screenshot is immutable after publication. Retain its Arc along with the
// resized image so pointer identity cannot be reused while this entry is live.
struct MapDisplayCache(Option<(Arc<core::Mat>, core::Size, core::Mat)>);

impl MapDisplayCache {
    fn get(&mut self, source: &Arc<core::Mat>, size: core::Size) -> opencv::Result<&core::Mat> {
        if self.0.as_ref().is_none_or(|(previous, previous_size, _)| {
            !Arc::ptr_eq(previous, source) || *previous_size != size
        }) {
            let mut resized = core::Mat::default();
            imgproc::resize(source.as_ref(), &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;
            self.0 = Some((Arc::clone(source), size, resized));
        }
        Ok(&self.0.as_ref().unwrap().2)
    }
}

fn draw_btn(canvas: &mut core::Mat, rect: core::Rect, label: &str, is_active: bool) -> opencv::Result<()> {
    hud_theme::button(canvas,rect,label,is_active)
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

/// Last successful mini-map crop, reused on future launches to avoid
/// interrupting every session with the OS rectangle picker. Clear with
/// TNT_RESELECT_MINIMAP=1 to deliberately recalibrate.
fn parse_cached_map_geo(s: &str) -> Option<(i32, i32, i32, i32)> {
    let values: Vec<i32> = s.trim().split(',')
        .map(|v| v.trim().parse::<i32>())
        .collect::<Result<_, _>>().ok()?;
    if values.len() != 4 {
        return None;
    }
    let (x, y, w, h) = (values[0], values[1], values[2], values[3]);
    (x >= 0 && y >= 0 && (60..=2000).contains(&w) && (60..=1600).contains(&h)
        && x.checked_add(w).is_some() && y.checked_add(h).is_some())
        .then_some((x, y, w, h))
}

fn minimap_cache_path() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".config").join("tnt-comput").join("minimap_region.txt")
}

fn cached_minimap_geo() -> Option<(i32, i32, i32, i32)> {
    std::fs::read_to_string(minimap_cache_path()).ok()
        .and_then(|s| parse_cached_map_geo(&s))
}

fn save_minimap_geo(geo: (i32, i32, i32, i32)) {
    let path = minimap_cache_path();
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() { return; }
    }
    let (x, y, w, h) = geo;
    let _ = std::fs::write(path, format!("{x},{y},{w},{h}\n"));
}

/// A memory-first run must not spend CPU grabbing and recognizing any angle
/// pixels. The old OCR path runs only when explicitly selected AND calibrated.
fn should_capture_angle_ocr(prefer_memory: bool, roi_set: bool) -> bool {
    !prefer_memory && roi_set
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
    let args: Vec<_> = std::env::args().collect();
    #[cfg(target_os = "macos")]
    if args.get(1).map(String::as_str) == Some("--check-window-close") {
        return hud_window::check_native_close();
    }
    if args.get(1).map(String::as_str) == Some("--preview-layout") {
        let path = args.get(2).ok_or_else(|| opencv::Error::new(core::StsBadArg,"preview PNG path required"))?;
        return render_hud_preview(path,args.get(3).map(String::as_str).unwrap_or("ready"));
    }
    eprintln!("[PHYSICS] model={} world_units_per_hud_unit={}", model_name(), world_units_per_hud_unit());
    #[cfg(target_os = "macos")]
    println!("=== 🍎 Mac OS 环境检测成功，已自动切换原生 screencapture 截图引擎 ===");

    // ===== 网络实况隧道:libtntsniff.so → adb reverse → 127.0.0.1:19001 =====
    // 尽早绑定:用户还在框选时隧道就能连进来,开局 INIT 不容易错过。
    let net_state: Arc<Mutex<NetHud>> = Arc::new(Mutex::new(NetHud::default()));
    // One private session joins network records and pre-shot local pose samples.
    let recording_root = std::env::var_os("TNT_RECORD_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("recordings"));
    let raw_override = std::env::var_os("TNT_TRACE_PATH").map(std::path::PathBuf::from);
    let trace = match tnt_comput::capture::Capture::open(&recording_root, raw_override.as_deref()) {
        Ok(capture) => {
            println!("📡 NET: 本次数据目录 {}（含炮口姿态记录）", capture.dir.display());
            Some(Arc::new(Mutex::new(capture)))
        }
        Err(e) => { eprintln!("⚠️ 无法开启数据录制（不会覆盖旧文件）: {e}"); None }
    };
    // 默认开启我方 QGraphicsItem 动态 XY；可 TNT_LIVE_XY=0 关闭。
    // 失败自动回退网络数据；敌方仍走网络解析，不污染原始抓包。
    let live_xy_state: Arc<Mutex<Option<LiveXY>>> = Arc::new(Mutex::new(None));
    let live_xy_enabled = live_xy_enabled_from_env(std::env::var("TNT_LIVE_XY").ok().as_deref());
    if live_xy_enabled {
        let state = live_xy_state.clone();
        thread::spawn(move || {
            let script = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/scripts/position/stream_live_xy_readonly.py"
            );
            let child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn();
            match child {
                Ok(mut child) => {
                    println!("📍 XY: 已启动默认的我方 QGraphicsItem 走路/飞行实时坐标采样；失败自动回退网络坐标（TNT_LIVE_XY=0 可关闭）");
                    if let Some(stdout) = child.stdout.take() {
                        for line in BufReader::new(stdout).lines() {
                            let Ok(line) = line else { break };
                            if let Some(update) = parse_live_xy_line(&line) {
                                if let Ok(mut st) = state.lock() {
                                    *st = update.map(|pos| LiveXY {
                                        updated: std::time::Instant::now(),
                                        pos,
                                    });
                                }
                            }
                        }
                    }
                    if let Ok(mut st) = state.lock() {
                        *st = None;
                    }
                    let _ = child.kill();
                    let _ = child.wait();
                    eprintln!("⚠️ XY: 只读探针已结束，自动回退网络坐标");
                }
                Err(err) => eprintln!("⚠️ XY: 无法启动只读探针: {err}"),
            }
        });
    }
    // The verified local aiming source is enabled by default, independent of
    // network/XY/wind. TNT_LIVE_ANGLE=0 provides a hard OCR-only opt-out.
    let live_angle_state: Arc<Mutex<Option<LiveAngle>>> = Arc::new(Mutex::new(None));
    let live_origin_state: Arc<Mutex<Option<LiveOrigin>>> = Arc::new(Mutex::new(None));
    let origin_hint_state: Arc<Mutex<Option<OriginHint>>> = Arc::new(Mutex::new(None));
    let live_origin_enabled = live_xy_enabled_from_env(std::env::var("TNT_FIRE_ORIGIN").ok().as_deref());
    let live_angle_enabled = live_angle_enabled_from_env(
        std::env::var("TNT_LIVE_ANGLE").ok().as_deref()
    );
    if live_angle_enabled {
        let state = live_angle_state.clone();
        let origin_state = live_origin_state.clone();
        let hint_state = origin_hint_state.clone();
        let pose_trace = trace.clone();
        thread::spawn(move || {
            let script = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/scripts/position/stream_live_angle_readonly.py"
            );
            let mut command = Command::new("python3");
            command.arg("-u").arg(script);
            if live_origin_enabled {
                command.arg("--fire-origin");
                if pose_trace.is_some() { command.arg("--record-origin"); }
            }
            let child = command
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn();
            match child {
                Ok(mut child) => {
                    println!("🎯 ANGLE: 内存角度默认启用；失效回退表盘。UI 可切 MEM/OCR，TNT_LIVE_ANGLE=0 强制禁用内存读数");
                    if let Some(stdout) = child.stdout.take() {
                        for line in BufReader::new(stdout).lines() {
                            let Ok(line) = line else { break };
                            if line.starts_with("TNTPOSE ") || line.starts_with("TNTORIGIN") {
                                if let Some(trace) = &pose_trace {
                                    if let Ok(mut capture) = trace.lock() { capture.origin_sample(&line); }
                                }
                            }
                            if let Some(update) = parse_live_angle_line(&line) {
                                if let Ok(mut st) = state.lock() {
                                    *st = update;
                                }
                            }
                            if let Some(update) = parse_live_origin_line(&line) {
                                if let Ok(mut st) = origin_state.lock() { *st = update; }
                            }
                            if let Some(hint)=parse_origin_hint(&line) {
                                if let Ok(mut st)=origin_state.lock() { apply_origin_hint(&mut st,&hint); }
                                if let Ok(mut st)=hint_state.lock() { *st=Some(hint); }
                            }
                            if line=="TNTORIGINSTATUS-" {
                                if let Ok(mut st)=hint_state.lock() { *st=None; }
                            }
                        }
                    }
                    if let Ok(mut st) = state.lock() {
                        *st = None;
                    }
                    if let Ok(mut st) = origin_state.lock() { *st = None; }
                    if let Ok(mut st) = hint_state.lock() { *st = None; }
                    let _ = child.kill();
                    let _ = child.wait();
                    eprintln!("⚠️ ANGLE: 只读角度探针结束，已回退表盘识别");
                }
                Err(err) => eprintln!("⚠️ ANGLE: 无法启动只读探针: {err}"),
            }
        });
    }
    {
        let ns = net_state.clone();
        let trace = trace.clone();
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
            let mut connection_seq = 0u64;
            println!("📡 NET: 监听 127.0.0.1:{port} — adb reverse tcp:{port} tcp:{port}");
            for conn in listener.incoming() {
                let Ok(mut s) = conn else { continue };
                connection_seq += 1;
                let connection_id = connection_seq;
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
                    let mut tls_seen = false;
                    loop {
                        let n = match s.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => n,
                        };
                        for rec in parser.feed(&buf[..n]) {
                            if !tls_seen && matches!(rec.kind, tunnel::KIND_TLS_TX | tunnel::KIND_TLS_RX) {
                                tls_seen = true;
                                println!("📡 NET: 已收到 SSL 明文，开始解析加密连接内的 WebSocket");
                            }
                            let raw_offset = trace.as_ref().and_then(|trace| {
                                trace.lock().ok().and_then(|mut capture| capture.record(&rec))
                            });
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
                                if let TunnelEvent::Msg { msg, c2s, .. } = ev {
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
                                        if let Some(trace) = &trace {
                                            if let Ok(mut capture) = trace.lock() {
                                                let ctx = tnt_comput::capture::Context {
                                                    battle: st.battle_id, round: st.round,
                                                    turn: st.turn_seq, wind10: st.auto_wind10,
                                                    my_id: st.my_id,
                                                };
                                                capture.event(connection_id, &rec, raw_offset, c2s, &gev, &ctx);
                                            }
                                        }
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

    // No startup mode prompt: mini-map is the common default.
    // Existing mode-switch button still offers optional big-map calibration.
    let big_map_mode = std::env::var("TNT_MAP_MODE").ok()
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("big"));
    let map_crop_path = "/tmp/tnt_selected_map.png";
    let force_reselect = std::env::var("TNT_RESELECT_MINIMAP").ok().as_deref() == Some("1");
    let reused = if !big_map_mode && !force_reselect {
        cached_minimap_geo().and_then(|geo| {
            // A failed screenshot must not accidentally reuse an old file.
            let _ = std::fs::remove_file("/tmp/tnt_cached_minimap.png");
            capture_rect_to_file(geo, "/tmp/tnt_cached_minimap.png");
            imgcodecs::imread("/tmp/tnt_cached_minimap.png", imgcodecs::IMREAD_COLOR)
                .ok().filter(|m| !m.empty() && m.cols() == geo.2 && m.rows() == geo.3)
                .map(|m| (m, geo))
        }).or_else(|| {
            // Upgrade path from previous HUD versions: match the old saved
            // crop once against a live screenshot, no picker when possible.
            imgcodecs::imread(map_crop_path, imgcodecs::IMREAD_COLOR)
                .ok().filter(|m| !m.empty())
                .and_then(|m| find_screen_position(&m).map(|geo| (m, geo)))
        })
    } else { None };

    let (initial_img, map_geo) = if let Some(found) = reused {
        println!("📍 自动复用小地图区域；如屏幕布局有变，设置 TNT_RESELECT_MINIMAP=1 重新框选");
        found
    } else {
        println!("👉 首次校准：请框选【{}】区域，之后会记住位置，不用每次选",
                 if big_map_mode { "整张游戏地图" } else { "左上角小地图" });
        if !select_crop_interactive(map_crop_path) {
            println!("❌ 没有完成地图区域选择");
            return Ok(());
        }
        let initial = match imgcodecs::imread(map_crop_path, imgcodecs::IMREAD_COLOR) {
            Ok(m) if !m.empty() => m,
            _ => { println!("❌ 地图截图读取失败"); return Ok(()); }
        };
        let geo = match find_screen_position(&initial) {
            Some(geo) => geo,
            None => {
                println!("❌ 未能定位地图区域，请重新框选");
                return Ok(());
            }
        };
        (initial, geo)
    };
    if !big_map_mode { save_minimap_geo(map_geo); }
    let t_w = initial_img.cols();
    let t_h = initial_img.rows();
    println!("📍 {}区域: ({},{}) {}x{}", if big_map_mode {"大地图"} else {"小地图"},
             map_geo.0, map_geo.1, map_geo.2, map_geo.3);

    // Angle OCR is OFF by default: no screenshot and no picker at startup.
    // Selecting OCR inside the HUD requests its ROI only when first needed.
    let power_geo: Option<(i32, i32, i32, i32)> = None;

    let window_name = "TNT Assistant HUD";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    // Preserve source aspect ratio within a bounded map pane.
    let scale = compact_map_scale(t_w, t_h);

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
        prefer_memory_angle: live_angle_enabled,
        ocr_roi_requested: !live_angle_enabled,
        is_fixed_angle: true,
        wind_flip: false,
        wind_net: false,
        switch_requested: false,
        btn_press: None,
        target_press: None,
    }));

    let map_w_display = MAP_PANE_WIDTH;

    // 大地图模式:整屏宽默认 12 距,启动即自动锁尺(12/16/18/20/22 按钮可切换)
    if big_map_mode {
        app_state.lock().unwrap().locked_px_per_unit = Some(t_w as f64 / 12.0);
    }

    let [btn_p1, btn_e1, btn_mode_switch, btn_lock_ruler, btn_draw_ruler,
        btn_target, btn_clear, btn_u12, btn_u16, btn_u18, btn_u20, btn_u22,
        btn_a20, btn_a30, btn_a45, btn_a50, btn_a60, btn_a65, btn_a70, btn_a75,
        btn_ang_m5, btn_ang_minus, _rect_ang_text, btn_ang_plus, btn_ang_p5,
        btn_wind_m1, btn_wind_m01, _rect_wind_text, btn_wind_p01, btn_wind_p1,
        btn_auto_angle, btn_angle_source] = hud_controls();

    // 全部按钮矩形:用于"按下和松开都在同一按钮内才生效"的防误触判定
    let all_btns = vec![
        btn_p1, btn_e1, btn_mode_switch, btn_lock_ruler, btn_draw_ruler,
        btn_target, btn_clear, btn_a20, btn_a30, btn_a45, btn_a50, btn_a60, btn_a65, btn_a70,
        btn_a75, btn_ang_m5, btn_ang_minus, btn_ang_plus, btn_ang_p5, btn_auto_angle,
        btn_angle_source, btn_wind_m1, btn_wind_m01, btn_wind_p01, btn_wind_p1,
        btn_u12, btn_u16, btn_u18, btn_u20, btn_u22,
    ];

    let state_cb = app_state.clone();
    let net_state_cb = net_state.clone();
    let target_rows: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let target_rows_cb = target_rows.clone();
    let angle_reader_available = live_angle_enabled;
    highgui::set_mouse_callback(
        window_name,
        Some(Box::new(move |event, x, y, _flags| {
            let net_active = net_state_cb
                .lock()
                .map(|n| n.battle_id > 0 && !n.pos.is_empty())
                .unwrap_or(false);
            let mut st = state_cb.lock().unwrap();
            let map_rect = map_display_rect(st.src_w, st.src_h, st.disp_scale);
            let map_size = map_rect.size();
            let on_map = map_rect.contains(core::Point::new(x,y));

            if event == highgui::EVENT_LBUTTONDOWN {
                st.target_press = None;
                st.btn_press = None;
                if let Ok(n) = net_state_cb.lock() {
                    if let Ok(rows) = target_rows_cb.lock() {
                        if let Some(uid) = target_from_rows(&rows, x, y) {
                            if n.pos.contains_key(&uid) && Some(uid) != n.my_id {
                                st.target_press = Some((n.battle_id, uid));
                                return;
                            }
                        }
                    }
                }
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
                                let (src_x,src_right) = source_ruler_bounds(min_x,max_x,st.src_w,map_rect.x,st.disp_scale);
                                st.manual_cam_rect =
                                    Some(core::Rect::new(src_x, 0, src_right - src_x, st.src_h));
                            }
                            st.drag_start = None;
                            st.edit_mode = EditMode::None;
                            // Auto-lock the ruler with the newly drawn box (0.0 triggers evaluation in drawing loop)
                            st.locked_px_per_unit = Some(0.0);
                        }
                        EditMode::None => {
                            if net_active {
                                if let Ok(mut n) = net_state_cb.lock() {
                                    if let Some(uid) = target_at_map(&n, x-map_rect.x, y-map_rect.y, map_size) {
                                        n.target_id = Some(uid);
                                        st.manual_e1 = None;
                                    }
                                }
                            }
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
                                if let Some(uid) = target_at_map(&n, x-map_rect.x, y-map_rect.y, map_size) {
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
                if let Some((battle, uid)) = st.target_press.take() {
                    if let Ok(mut n) = net_state_cb.lock() {
                        if let Some(uid) = released_target(&n, uid, x, y).filter(|_| n.battle_id == battle) {
                            n.target_id = Some(uid);
                            st.manual_e1 = None;
                            st.edit_mode = EditMode::None;
                        }
                    }
                    return;
                }
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
                                let ids = opponent_ids(&n);
                                if !ids.is_empty() {
                                    n.target_id = match selected_target_id(&n) {
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
                        } else if is_inside(x, y, btn_angle_source) {
                            // Source switch is independent of manual angle and
                            // never changes the established XY/wind/network paths.
                            if angle_reader_available {
                                st.prefer_memory_angle = !st.prefer_memory_angle;
                                if !st.prefer_memory_angle {
                                    st.ocr_roi_requested = true;
                                }
                            }
                        } else if is_inside(x, y, btn_wind_m1) {
                            st.wind -= 1.0;
                            st.wind_net = false;
                        } else if is_inside(x, y, btn_wind_m01) {
                            st.wind -= 0.1;
                            st.wind_net = false;
                        } else if is_inside(x, y, btn_wind_p01) {
                            st.wind += 0.1;
                            st.wind_net = false;
                        } else if is_inside(x, y, btn_wind_p1) {
                            st.wind += 1.0;
                            st.wind_net = false;
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
                        }
                    }
                }
            } else if event == highgui::EVENT_MOUSEMOVE {
                if st.edit_mode == EditMode::DrawRuler2 {
                    if let Some(start) = st.drag_start {
                        let min_x = start.x.min(x);
                        let max_x = start.x.max(x);
                        // 预览只看左右边界,纵向铺满整幅源图
                        let (src_x,src_right) = source_ruler_bounds(min_x,max_x,st.src_w,map_rect.x,st.disp_scale);
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
    // The OCR crop is absent until requested explicitly by the user.
    let power_geo_shared = Arc::new(std::sync::Mutex::new(power_geo));
    let power_geo_clone = power_geo_shared.clone();

    let shared_map = Arc::new(std::sync::Mutex::new(None::<Arc<core::Mat>>));
    let shared_map_clone = shared_map.clone();
    let shared_recognized_angle = Arc::new(std::sync::Mutex::new(None::<(i32, std::time::Instant)>));
    let shared_recognized_angle_clone = shared_recognized_angle.clone();

    // 回合切换信号:新回合清空角度投票+显示值,防止旧角度给出错误力度
    let angle_reset_gen =
        Arc::new(std::sync::atomic::AtomicU64::new(0));
    let angle_reset_gen_clone = angle_reset_gen.clone();

    let app_state_bg = app_state.clone();
    let live_angle_capture = live_angle_state.clone();
    let net_angle_capture = net_state.clone();

    thread::spawn(move || {
        // Do NOT initialize OCR templates or allocate an OCR reader in normal
        // memory-angle mode. The first explicit OCR switch initializes it.
        let mut recognizer: Option<tnt_comput::ui::UiRecognizer> = None;
        // On-demand angle OCR: do not run in default MEM mode.
        let mut last_angle_gen = 0u64;
        // The mini-map background can be refreshed less often; its screenshots
        // do not determine the network XY / memory angle sampling rates.
        let mut last_map_cap = std::time::Instant::now() - std::time::Duration::from_secs(1);
        // Map background only: less frequent screenshot and PNG parsing.
        // Player markers and memory angle remain independently refreshed.
        let map_capture_ms: u128 = std::env::var("TNT_MAP_CAPTURE_MS")
            .ok().and_then(|v| v.parse::<u128>().ok())
            .filter(|&ms| (250..=3000).contains(&ms)).unwrap_or(500);
        let mut last_size_warn = std::time::Instant::now();
        let mut last_roi: Option<core::Mat> = None;
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
            // 500ms/default map background; angle screen never captured in
            // MEM mode, only when OCR has been explicitly chosen.
            if last_map_cap.elapsed().as_millis() >= map_capture_ms {
                last_map_cap = std::time::Instant::now();
                capture_rect_to_file(*map_geo_shared_clone.lock().unwrap(), map_path);
                if let Ok(m) = imgcodecs::imread(map_path, imgcodecs::IMREAD_COLOR) {
                    if !m.empty() {
                        if let Ok(mut lock) = shared_map_clone.lock() {
                            *lock = Some(Arc::new(m));
                        }
                    }
                }
            }

            let prefer_memory = app_state_bg.lock().ok()
                .is_none_or(|st| st.prefer_memory_angle);
            let selected_power_geo = power_geo_clone.lock().ok().and_then(|g| *g);
            let ocr_active = should_capture_angle_ocr(
                prefer_memory, selected_power_geo.is_some()
            );
            if let Some(pg) = selected_power_geo.filter(|_| ocr_active) {
                if recognizer.is_none() {
                    recognizer = tnt_comput::ui::UiRecognizer::new("src/templates").ok();
                }
                if recognizer.is_some() {
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
                                // Compare pixels, not channel sums: a moving dial
                                // can keep exactly the same total brightness.
                                if last_roi.as_ref().is_some_and(|previous| {
                                    same_angle_roi(&p, previous).unwrap_or(false)
                                }) {
                                    if let Ok(mut lock) = cap_time_ms_clone.lock() {
                                        *lock = t0.elapsed().as_millis();
                                    }
                                    // This early exit must keep the normal OCR
                                    // throttle, even while the dial is stationary.
                                    std::thread::sleep(std::time::Duration::from_millis(20));
                                    continue;
                                }
                                // 纯表盘几何法:读虚线方向,不走数字OCR
                                let val = recognizer.as_ref().unwrap().recognize_angle_dial(&p);
                                last_roi = Some(p);
                                match val {
                                    Ok(Some(stable)) => {
                                        // 表盘法单帧即准,直读直用
                                        if last_misread_dbg.elapsed().as_millis() > 2000 {
                                            last_misread_dbg = std::time::Instant::now();
                                            println!("🔍 角度OCR: 读到 {}", stable);
                                        }
                                        if let Ok(mut lock) = shared_recognized_angle_clone.lock()
                                        {
                                            *lock = Some((stable, std::time::Instant::now()));
                                        }
                                        if stable >= 3 && stable <= 90 {
                                            // Do not overwrite a fresh, opt-in memory angle with
                                            // an OCR observation from a stale or partial dial.
                                            let my_id = net_angle_capture.lock().ok().and_then(|n| n.my_id);
                                            let memory_fresh = live_angle_capture.lock().ok()
                                                .and_then(|l| live_angle_for_hud(l.as_ref(), my_id))
                                                .is_some();
                                            if let Ok(mut m_state) = app_state_bg.lock() {
                                                // Manually selected OCR must not be suppressed
                                                // just because memory samples are available.
                                                if m_state.auto_angle &&
                                                    (!m_state.prefer_memory_angle || !memory_fresh) {
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
            // MEM mode only needs an occasional mini-map refresh. Avoid
            // waking 50 times/sec to discover OCR is still disabled.
            std::thread::sleep(std::time::Duration::from_millis(
                if ocr_active { 20 } else { 60 }
            ));
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
    /// 上次自动写风时用的 (方位hint, F翻转);回合内换目标/按F/站位变化都会改它
    let mut last_wind_key: Option<(Option<i32>, bool)> = None;
    /// 上一个我方回合(局,回合,raw,回合开始时的风值),翻篇时结算进配对日志
    let mut prev_my_turn: Option<(u64, u64, i64, f64)> = None;
    // 每 5 秒汇报一次 XY 端到端状态，诊断「抓到坐标却没真正画出来」。
    let mut last_xy_diag = std::time::Instant::now() - std::time::Duration::from_secs(6);

    let mut first_show = true;

    // 立即显示初始化画面，防止 Mac 窗口引擎死锁
    let mut init_canvas = core::Mat::new_rows_cols_with_default(
        HUD_HEIGHT,
        MAP_PANE_WIDTH+SIDEBAR_WIDTH,
        core::CV_8UC3,
        core::Scalar::new(30.0, 30.0, 30.0, 0.0),
    )?;
    imgproc::put_text(
        &mut init_canvas,
        "正在连接游戏…",
        core::Point::new(50, HUD_HEIGHT/2),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.0,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;
    highgui::imshow(window_name, &init_canvas)?;
    let window_lifecycle = hud_window::HudWindow::attach(window_name);
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
    let mut traj_memo = LastValue::new();
    let mut path_memo = LastValue::new();
    let mut map_memo = MapDisplayCache(None);
    let mut recent_aim: Option<RecentAim> = None;
    let mut latched_scale: Option<(u64,u64,ShotScale)> = None;
    let mut fps_t0 = std::time::Instant::now();
    let mut last_angle_diag = std::time::Instant::now() - std::time::Duration::from_secs(6);
    let mut last_toggle_time = std::time::Instant::now() - std::time::Duration::from_secs(1); // 防抖时间戳

    loop {
        if window_lifecycle.is_closed() { break; }
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
                            let net = net_state.lock().unwrap();
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
            lock.clone()
        };

        let t_io = loop_t0.elapsed().as_millis();
        let t1 = std::time::Instant::now();

        let prefer_memory_angle = app_state.lock().unwrap().prefer_memory_angle;
        // OCR readings are never consulted in MEM mode. No OCR screenshot,
        // recognition or stale fallback happens unless OCR is explicitly chosen.
        let power_recognized_val = if prefer_memory_angle {
            None
        } else {
            shared_recognized_angle.lock().unwrap().as_ref()
                .and_then(|&(value, seen)| {
                    (seen.elapsed() <= std::time::Duration::from_secs(2)).then_some(value)
                })
        };
        // Source preference is an independent HUD toggle (defaults to MEM).
        // Identity remains verified locally and checked against network UID.
        let (live_angle_used,live_fire_angle_used) = if live_angle_enabled && prefer_memory_angle {
            let my_id = net_state.lock().unwrap().my_id;
            let live = live_angle_state.lock().unwrap();
            let angle=selected_memory_angle(live_angle_enabled, prefer_memory_angle, live.as_ref(), my_id);
            (angle,angle.and_then(|_|live.as_ref().map(|s|s.fire_angle)))
        } else {
            (None,None)
        };
        if live_angle_enabled && last_angle_diag.elapsed() >= std::time::Duration::from_secs(5) {
            let my_id = net_state.lock().unwrap().my_id;
            let state = live_angle_state.lock().unwrap();
            let (status, sample_uid, age_ms, fire_angle) = match state.as_ref() {
                None => ("no_verified_sample", None, None, None),
                Some(sample) if my_id.is_some_and(|network_uid| network_uid != sample.uid) =>
                    ("uid_mismatch", Some(sample.uid), Some(sample.updated.elapsed().as_millis()), Some(sample.fire_angle)),
                Some(sample) if sample.updated.elapsed() > std::time::Duration::from_secs(2) =>
                    ("sample_stale", Some(sample.uid), Some(sample.updated.elapsed().as_millis()), Some(sample.fire_angle)),
                Some(sample) =>
                    (if my_id.is_some() { "sample_ready" } else { "sample_ready_local" },
                     Some(sample.uid), Some(sample.updated.elapsed().as_millis()), Some(sample.fire_angle)),
            };
            println!(
                "🎯 ANGLE HUD: {status} my_id={my_id:?} sample_uid={sample_uid:?} age_ms={age_ms:?} fire_angle={fire_angle:?}"
            );
            last_angle_diag = std::time::Instant::now();
        }
        // Sync immediately after MEM/OCR switching, without waiting for
        // the image ROI to change again. Manual angles remain untouched.
        if let Ok(mut state) = app_state.lock() {
            if state.auto_angle {
                if let Some(angle) = live_angle_used {
                    state.current_angle = angle;
                } else if let Some(ocr_value) = power_recognized_val {
                    state.current_angle = f64::from(ocr_value);
                }
            }
        }

        let t_recog = t1.elapsed().as_millis();
        let t2 = std::time::Instant::now();

        let canvas_w = map_w_display + SIDEBAR_WIDTH;
        let st = *app_state.lock().unwrap();
        // 模式切换后这些会变,每帧从状态里取(遮蔽外层的启动值)
        let t_w = st.src_w;
        let t_h = st.src_h;
        let scale = st.disp_scale;
        let map_rect = map_display_rect(t_w, t_h, scale);
        let map_size = map_rect.size();
        let map_h_display = map_size.height;
        let canvas_h_target = HUD_HEIGHT;
        let mut canvas = core::Mat::new_rows_cols_with_default(
            canvas_h_target,
            canvas_w,
            core::CV_8UC3,
            hud_theme::color(hud_theme::BACKGROUND),
        )?;

        let frame_net = net_state.lock().unwrap().clone();
        let frame_scale={
            let origin=live_origin_state.lock().ok();
            let hint=origin_hint_state.lock().ok();
            update_shot_scale(&mut latched_scale,&frame_net,origin.as_ref().and_then(|x|x.as_ref()),hint.as_ref().and_then(|x|x.as_ref()))
        };
        draw_hud_background(&mut canvas,&frame_net)?;
        if let Some(minimap) = img {
            let map_display = map_memo.get(
                &minimap, map_size
            )?;

            for y in 0..map_display.rows() {
                if let (Ok(src_row), Ok(dst_row)) = (
                    map_display.at_row::<core::Vec3b>(y),
                    canvas.at_row_mut::<core::Vec3b>(y+map_rect.y),
                ) {
                    let len = src_row.len();
                    dst_row[map_rect.x as usize..map_rect.x as usize+len].copy_from_slice(src_row);
                }
            }



            let cam_rect = st
                .manual_cam_rect
                .unwrap_or(core::Rect::new(0, 0, t_w, t_h));

            let to_scr = |p: core::Point| {
                core::Point::new(map_rect.x+(p.x as f64 * scale) as i32, map_rect.y+(p.y as f64 * scale) as i32)
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
            // 只修改渲染快照；原始网络坐标与录制数据保持不变。
            let mut net = frame_net.clone();
            let mut xy_seen = 0usize;
            let mut xy_age_ms: Option<u128> = None;
            let mut xy_applied = 0usize;
            if let Ok(live) = live_xy_state.lock() {
                if let Some(sample) = live.as_ref() {
                    xy_seen = sample.pos.len();
                    xy_age_ms = Some(sample.updated.elapsed().as_millis());
                }
                xy_applied = overlay_live_xy(&mut net, live.as_ref());
            }
            if live_xy_enabled && last_xy_diag.elapsed() >= std::time::Duration::from_secs(5) {
                let mut coords: Vec<_> = net.pos.iter()
                    .map(|(&id, &(x, y))| (id, x, y))
                    .collect();
                coords.sort_unstable_by_key(|p| p.0);
                println!(
                    "📍 XY HUD: mem_roles={} sample_age_ms={:?} applied={} net_roles={} conns={} map={}x{} my_id={:?} manual_self={} manual_enemy={} positions={:?}",
                    xy_seen, xy_age_ms, xy_applied, net.pos.len(), net.conns,
                    net.map_w, net.map_h, net.my_id,
                    st.manual_p1.is_some(), st.manual_e1.is_some(), coords
                );
                last_xy_diag = std::time::Instant::now();
            }

            // 客户端解出的世界风 → 我方新回合自动写进 st.wind(符号=箭头×方位×F翻转)
            if net.turn_seq != last_wind_turn {
                last_wind_turn = net.turn_seq;
                // 通知 OCR 线程:新回合清票,快速锁定新角度
                angle_reset_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // 结算上一个我方回合(纯诊断):st.wind=用户最终认定值
                // (没改≠用户确认,改了只是修正记录)
                if let Some((pb, pr, praw, _pw)) = prev_my_turn.take() {
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
                }
                if net.active.is_some() && net.active == net.my_id {
                    let mut stg = app_state.lock().unwrap();
                    if let Some(world10) = net.auto_wind10 {
                        stg.wind =
                            auto_wind_present(world10, wind_dir_hint(&net), stg.wind_flip);
                        stg.wind_net = true;
                        last_wind_key = Some((wind_dir_hint(&net), stg.wind_flip));
                    } else {
                        // 本回合 Action 缺字段:保留手动风,但不再标网络真值
                        stg.wind_net = false;
                    }
                    prev_my_turn =
                        Some((net.battle_id, net.round, net.wind_seed, stg.wind));
                }
            }

            // 回合内换目标/按F翻转/敌人移动导致顺逆符号变化 → 重算网络风;
            // 手动输入/语音/±按钮改过的风(wind_net=false)不覆盖
            if net.active.is_some() && net.active == net.my_id {
                let mut stg = app_state.lock().unwrap();
                let key = (wind_dir_hint(&net), stg.wind_flip);
                if last_wind_key != Some(key) {
                    last_wind_key = Some(key);
                    if stg.wind_net {
                        if let Some(w10) = net.auto_wind10 {
                            stg.wind = auto_wind_present(w10, key.0, stg.wind_flip);
                        }
                    }
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
                    map_rect.x+(x as f64 * map_size.width as f64 / net.map_w as f64).round() as i32,
                    map_rect.y+(y as f64 * map_h_display as f64 / net.map_h as f64).round() as i32,
                );
                net_p1 = net
                    .my_id
                    .and_then(|id| net.pos.get(&id))
                    .map(|&(x, y)| to_map(x, y));
                net_all = opponent_ids(&net).into_iter().filter_map(|uid| {
                    net.pos.get(&uid).map(|&(x, y)| (uid, to_map(x, y)))
                }).collect();
                net_e1 = target_position(&net).map(|(x, y)| to_map(x, y));
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
            // 多人局:未选中敌人=高亮黄色、选中目标=亮品红。
            // 名字加黑色描边，在明亮或杂色地图背景上都能看清。
            let sel_uid = selected_target_id(&net);
            for (index, (uid, p)) in net_all.iter().enumerate() {
                let is_sel = sel_uid == Some(*uid);
                let marker_color = hud_theme::color(if is_sel {hud_theme::ACCENT} else {hud_theme::MUTED});
                let black = core::Scalar::new(0.0, 0.0, 0.0, 0.0);
                // 深色外环保证标记不会淹没在地图背景里。
                let _ = imgproc::circle(&mut canvas, *p, 10, black, 4, imgproc::LINE_AA, 0);
                let _ = imgproc::circle(&mut canvas, *p, 9, marker_color, 3, imgproc::LINE_AA, 0);
                let _ = imgproc::circle(&mut canvas, *p, 2, marker_color, -1, imgproc::LINE_AA, 0);
                let label = format!("{}", index + 1);
                let label_pt = core::Point::new((p.x + 10).clamp(map_rect.x+3, (map_rect.x+map_size.width-26).max(map_rect.x+3)),
                    (p.y - 12).clamp(map_rect.y+15, (map_rect.y+map_size.height-4).max(map_rect.y+15)));
                let _ = imgproc::put_text(&mut canvas, &label, label_pt,
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.48, black, 4, imgproc::LINE_AA, false);
                let _ = imgproc::put_text(&mut canvas, &label, label_pt,
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.48, marker_color, 1, imgproc::LINE_AA, false);

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
            if let Some(e) = st.manual_e1 {
                draw_pt(&mut canvas, e, "敌方 [手动]", true, true);
            }

            let mut draw_result = |p: core::Point, e: core::Point| {
                if st.locked_px_per_unit.is_none() {
                    let _=draw_aim_pending(&mut canvas,"请先锁定标尺","可在左下方锁定或重新框定");
                    return;
                }

                // Network points keep world-coordinate precision. Hand-picked
                // points still use their actual mini-map position and ruler.
                let manual_points = st.manual_p1.is_some() || st.manual_e1.is_some();
                let use_live_origin = live_origin_enabled && is_recorded_model() && !manual_points
                    && prefer_memory_angle && st.auto_angle;
                let mut inputs=aim_inputs(&net,st.current_angle,st.wind,(px_per_unit,px_per_unit_y),(t_w,t_h),st.is_fixed_angle);
                let world_angle=if !manual_points && prefer_memory_angle && st.auto_angle {live_fire_angle_used} else {None};
                if let Some(i)=inputs.as_mut() {i.scale=frame_scale;i.world_angle=world_angle;}
                let mut source_sampled=None;
                let mut pending=("正在读取炮口","本次采样未确认，稍后自动重试");
                let origin_choice = if use_live_origin {
                    let sample = live_origin_state.lock().ok();
                    let live = if live_angle_used.is_some() { sample.as_ref().and_then(|s| s.as_ref()) } else { None };
                    source_sampled=live.and_then(|s|s.updated.checked_sub(std::time::Duration::from_millis(s.age_ms)));
                    let hint=origin_hint_state.lock().ok();
                    pending=if live.is_some_and(|s|Some(s.fire_angle)!=world_angle) {
                        ("角度同步中","正在读取实际发射角度")
                    } else {origin_pending_message(live,hint.as_ref().and_then(|s|s.as_ref()),&net,st.current_angle)};
                    resolve_aim_origin(live.filter(|s|s.scale==frame_scale && world_angle==Some(s.fire_angle)), &net, st.current_angle, true)
                } else { AimOrigin::Player };
                let origin = match origin_choice {
                    AimOrigin::Muzzle(xy) => Some(xy),
                    AimOrigin::Player => None,
                    AimOrigin::Verifying => {
                        if pending.0=="读数更新中" {
                            let hint=origin_hint_state.lock().ok();
                            if let Some(last)=recent_aim.as_ref().filter(|last|last.can_reference(inputs.as_ref(),hint.as_ref().and_then(|h|h.as_ref()))) {
                                let _=draw_aim_ready(&mut canvas,last.force,last.angle,last.dx,last.dy,st.is_fixed_angle,
                                    if last.inputs.scale==ShotScale::Expanded {"上次放大建议 · 更新中"} else {"上次建议 · 读数更新中"});
                                return;
                            }
                        }
                        let _=draw_aim_pending(&mut canvas,pending.0,pending.1);
                        return;
                    }
                };
                let (dx, dy) = network_aim_offsets(
                    &net, manual_points,
                    core::Size::new(t_w, t_h), (px_per_unit, px_per_unit_y),
                    origin,
                ).unwrap_or_else(|| (
                    (e.x - p.x) as f64 / scale / px_per_unit,
                    -(e.y - p.y) as f64 / scale / px_per_unit_y,
                ));

                // 相对风(正=顺风),纯手动输入(按钮/语音)
                let eff_wind = if world_angle.is_some() {target_relative_to_world_wind(&net,st.wind,dx)} else {st.wind};

                let key = (dx, dy, st.current_angle, eff_wind, st.is_fixed_angle,frame_scale,world_angle);
                let trajectory_res = cached_trajectory(&mut traj_memo, key);

                match trajectory_res {
                    Some((force, final_angle)) => {
                        let label=match (origin.is_some(),frame_scale) {
                            (true,ShotScale::Normal)=>"推荐力度 · 普通炮弹",
                            (true,ShotScale::Expanded)=>"推荐力度 · 放大炮弹",
                            (false,ShotScale::Normal)=>"参考力度 · 普通炮弹",
                            (false,ShotScale::Expanded)=>"参考力度 · 放大炮弹",
                        };
                        let display_angle=if world_angle.is_some() {final_angle.min(180.0-final_angle)} else {final_angle};
                        let _=draw_aim_ready(&mut canvas,force,display_angle,dx,dy,st.is_fixed_angle,label);
                        if origin.is_some() {
                            if let (Some(inputs),Some(sampled))=(inputs,source_sampled) {
                                recent_aim=Some(RecentAim {inputs,sampled,force,angle:display_angle,dx,dy});
                            }
                        }

                        // Draw trajectory dots on the minimap
                        let is_reverse = dx < 0.0;
                        let draw_angle = trajectory_draw_angle(final_angle,dx,world_angle.is_some());

                        // 物理引擎原生支持真实的物理世界坐标系（角度>90代表向左，风向带符号）
                        // 用户输入的是相对风力（正=顺风），但在画图时，我们要把它转成绝对世界的风向。
                        // 如果向左打，顺风就是向左吹（绝对世界里的负风向）。
                        let sim_wind = if world_angle.is_some() {eff_wind} else if is_reverse { -eff_wind } else { eff_wind };
                        let path = path_memo.get_or_compute((draw_angle, force, sim_wind,frame_scale), || {
                            tnt_comput::physics::simulate_path_for_scale(frame_scale,draw_angle,force,sim_wind)
                        });
                        let draw_start = origin.map(|(x, y)| (
                            map_rect.x as f64 + x * map_size.width as f64 / net.map_w as f64,
                            map_rect.y as f64 + y * map_h_display as f64 / net.map_h as f64,
                        )).unwrap_or((p.x as f64, p.y as f64));
                        for &(sim_x, sim_y) in path.iter() {
                            let img_x = draw_start.0 + sim_x * scale * px_per_unit;
                            let img_y = draw_start.1 - sim_y * scale * px_per_unit_y;

                            // Stop if out of bounds of the minimap
                            if img_x < map_rect.x as f64 || img_x > (map_rect.x+map_size.width) as f64 || img_y > (map_rect.y+map_h_display) as f64 {
                                break;
                            }
                            if img_y >= map_rect.y as f64 {
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
                        let _=draw_aim_pending(&mut canvas,"当前角度无法到达","调整角度或目标后重试");
                    }
                }

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
                let enemy = target_position(&net);
                if let (Some(&(px, py)), Some((ex, ey))) = (mine, enemy) {
                    let dx = (ex - px) as f64 / world_per_unit;
                    let dy = (py - ey) as f64 / world_per_unit;
                    let res = cached_trajectory(
                        &mut traj_memo, (dx, dy, st.current_angle, if st.auto_angle && prefer_memory_angle && live_fire_angle_used.is_some() {target_relative_to_world_wind(&net,st.wind,dx)} else {st.wind}, st.is_fixed_angle,frame_scale,if st.auto_angle && prefer_memory_angle {live_fire_angle_used} else {None})
                    );
                    match res {
                        Some((force,angle)) => draw_aim_ready(&mut canvas,force,if st.auto_angle && prefer_memory_angle && live_fire_angle_used.is_some() {angle.min(180.0-angle)} else {angle},dx,dy,st.is_fixed_angle,"参考力度")?,
                        None => draw_aim_pending(&mut canvas,"当前角度无法到达","地图尺寸暂未同步")?,
                    }
                }
            }
        } else if img.is_none() {
            hud_theme::text(&mut canvas,"等待读取地图",62,220,310,0.6,hud_theme::MUTED)?;
        }

        draw_hud_controls(&mut canvas, &st, live_angle_used, power_recognized_val, &wind_input_buf)?;

        let t_ui = t2.elapsed().as_millis();
        let bg_cap_ms = *cap_time_ms.lock().unwrap();
        let fps = 1000.0 / fps_t0.elapsed().as_millis().max(1) as f64;
        fps_t0 = std::time::Instant::now();
        
        let perf_txt = format!("FPS {:.0}", fps);
        imgproc::put_text(&mut canvas, &perf_txt, core::Point::new(352, HUD_HEIGHT - 4),
            imgproc::FONT_HERSHEY_SIMPLEX, 0.38, core::Scalar::all(130.0), 1, imgproc::LINE_AA, false)?;
        sidebar_text(&mut canvas, if voice_status.is_empty() {"按住 Shift 说出风速"} else {&voice_status}, 615, 0.39, hud_theme::color(hud_theme::MUTED))?;
        let _ = (t_io, t_recog, t_ui, bg_cap_ms);
        {
            let net = &frame_net;
            draw_target_list(&mut canvas, &net)?;
            if let Ok(mut rows) = target_rows.lock() { *rows = opponent_ids(&net); }
        }

        // 网络实况行:隧道连着或有对局数据才显示
        {
            let net = &frame_net;
            if net.battle_id > 0 || net.conns > 0 {
                let link = if net.conns > 0 { "NET" } else { "NET(断)" };
                let turn_s = match net.active {
                    Some(a) if Some(a) == net.my_id => "我方回合",
                    Some(_) => "敌方回合",
                    None => "",
                };
                // 客户端已解出带符号世界风:正→向右吹,负→向左
                let wind_dir = match net.auto_wind10.map(|w| w.signum()) {
                    Some(s) if s > 0 => "→",
                    Some(s) if s < 0 => "←",
                    _ => "-",
                };
                let flip_n = if st.wind_flip { -1 } else { 1 };
                let rel = match wind_dir_hint(&net).map(|h| h * flip_n) {
                    Some(1) => "顺",
                    Some(-1) => "逆",
                    _ => "",
                };
                let auto_s = net
                    .auto_wind10
                    .map(|m| format!(" 风{:.1}", (m as i64).abs() as f64 / 10.0))
                    .unwrap_or_default();
                let info = format!("{link} 回合{} {turn_s} 风向{wind_dir}{rel}{auto_s}", net.round);
                sidebar_text(&mut canvas, &info, 644, 0.39,
                    core::Scalar::new(120.0, 220.0, 255.0, 0.0))?;
            }
        }

        // Check before imshow: imshow is allowed to recreate a missing window.
        if window_lifecycle.is_closed() { break; }
        highgui::imshow(window_name, &canvas)?;
        if first_show {
            highgui::set_window_property(window_name, highgui::WND_PROP_TOPMOST, 1.0)?;
            first_show = false;
        }

        let key = match highgui::wait_key(15) {
            Ok(key) => key,
            Err(_) if window_lifecycle.is_closed() => break,
            Err(e) => return Err(e),
        };
        let debounce_ok = last_toggle_time.elapsed() > std::time::Duration::from_millis(300);
        if key == 27 || key == 'q' as i32 || window_lifecycle.is_closed() {
            break;
        } else if key == 13 || key == 10 || key == 3 {
            // Enter: Set Wind (macOS 小键盘回车发的是 \x03=3)
            if !wind_input_buf.is_empty() {
                if let Ok(w) = wind_input_buf.parse::<f64>() {
                    let flip_ui = app_state.lock().unwrap().wind_flip;
                    let net = net_state.lock().unwrap();
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

        // OCR is off by default. Only an explicit click on the OCR source
        // triggers a one-time crop picker here on the UI thread (never in a
        // mouse callback). The memory-angle sampling thread is unaffected.
        let request_ocr_roi = {
            let mut st = app_state.lock().unwrap();
            let req = st.ocr_roi_requested;
            st.ocr_roi_requested = false;
            req
        };
        if request_ocr_roi && power_geo_shared.lock().unwrap().is_none() {
            println!("👉 OCR 首次启用：请框选左下角角度表盘（仅此时需要）");
            let crop_path = "/tmp/tnt_selected_power.png";
            if select_crop_interactive(crop_path) {
                let candidate = imgcodecs::imread(crop_path, imgcodecs::IMREAD_COLOR)
                    .ok().filter(|m| !m.empty())
                    .and_then(|img| find_screen_position(&img))
                    .map(|(x, y, w, h)| {
                        (x.saturating_sub(10), y.saturating_sub(10), w + 20, h + 20)
                    });
                if let Some(geo) = candidate {
                    *power_geo_shared.lock().unwrap() = Some(geo);
                    *shared_recognized_angle.lock().unwrap() = None;
                    println!("✅ OCR 区域设置完成；只有切到 OCR 才会截图识别");
                } else {
                    println!("⚠️ OCR 区域无法定位；可点击 MEM 返回内存角");
                }
            } else {
                println!("⚠️ 未选择 OCR 区域；仍可点击 MEM 使用内存角");
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
                        if !to_big { save_minimap_geo(geo); }
                        *shared_map.lock().unwrap() = None; // 丢弃旧区域画面,等新图
                        let new_scale = compact_map_scale(nw, nh);
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
        let _ = c.wait();
    }
    let _ = highgui::destroy_window(window_name);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_layout_bounds_and_aspect_survive_map_switches() {
        for (w,h) in [(335,387),(1920,1080),(120,2000),(2000,120),(233,274)] {
            let scale = compact_map_scale(w,h);
            let size = map_display_size(w,h,scale);
            assert!(size.width <= MAP_MAX_WIDTH && size.height <= MAP_MAX_HEIGHT);
            let map=map_display_rect(w,h,scale);
            assert!(map.x>=24 && map.x+map.width<=MAP_PANE_WIDTH-24);
            assert!(map.y>=52 && map.y+map.height<=52+MAP_MAX_HEIGHT);
            assert!((size.width as f64 - w as f64 * scale).abs() <= 0.5);
            assert!((size.height as f64 - h as f64 * scale).abs() <= 0.5);
        }
        let mut rectangles = hud_controls().to_vec();
        rectangles.extend((0..MAX_TARGETS).map(|i| target_slot_rect(i).unwrap()));
        for r in &rectangles {
            assert!(r.x >= 0 && r.x + r.width <= MAP_PANE_WIDTH + SIDEBAR_WIDTH);
            assert!(r.y >= 0 && r.y + r.height < HUD_HEIGHT);
        }
        for (i,a) in rectangles.iter().enumerate() {
            for b in rectangles.iter().skip(i+1) {
                assert!(a.x+a.width <= b.x || b.x+b.width <= a.x || a.y+a.height <= b.y || b.y+b.height <= a.y);
            }
        }
    }

    #[test]
    fn centered_map_keeps_clicks_and_ruler_in_source_coordinates() {
        let mut net=NetHud {my_id:Some(1),map_w:2000,map_h:1000,..Default::default()};
        net.pos.insert(1,(200,500));net.pos.insert(2,(1600,600));
        for (w,h) in [(670,774),(1920,1080),(300,1800)] {
            let scale=compact_map_scale(w,h);
            let map=map_display_rect(w,h,scale);
            let x=map.x+(0.8*map.width as f64).round() as i32;
            let y=map.y+(0.6*map.height as f64).round() as i32;
            assert_eq!(target_at_map(&net,x-map.x,y-map.y,map.size()),Some(2));
            let left=map.x+(w as f64*0.25*scale).round() as i32;
            let right=map.x+(w as f64*0.75*scale).round() as i32;
            let (a,b)=source_ruler_bounds(left,right,w,map.x,scale);
            let tolerance=(0.5/scale).ceil() as i32+1;
            assert!((a-w/4).abs()<=tolerance && (b-3*w/4).abs()<=tolerance);
            let units=(w as f64/2000.0*80.0,h as f64/1000.0*80.0);
            let offsets=network_aim_offsets(&net,false,core::Size::new(w,h),units,None).unwrap();
            assert!((offsets.0-17.5).abs()<1e-10 && (offsets.1+1.25).abs()<1e-10);
        }
    }

    #[test]
    fn transient_scale_has_specific_message() {
        let net=NetHud {my_id:Some(1),..Default::default()};
        let now=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
        let hint=parse_origin_hint(&format!("TNTORIGINSTATUS 123 1 {now} scale 2.0")).unwrap();
        assert_eq!(origin_pending_message(None,Some(&hint),&net,51.0).0,"体型变化中");
        let other=OriginHint {uid:2,..hint.clone()};
        assert_eq!(origin_pending_message(None,Some(&other),&net,51.0).0,"读数更新中");
        let stale=OriginHint {updated:std::time::Instant::now()-std::time::Duration::from_secs(2),..hint};
        assert_eq!(origin_pending_message(None,Some(&stale),&net,51.0).0,"读数更新中");
        assert!(parse_origin_hint(&format!("TNTORIGINSTATUS 123 1 {} scale 2.0",now-1000)).is_none());
        assert!(parse_origin_hint(&format!("TNTORIGINSTATUS 123 1 {now} scale NaN")).is_none());
    }

    #[test]
    fn recent_advice_is_bounded_and_never_follows_changed_inputs_or_invalid_pose() {
        let mut net=NetHud {my_id:Some(1),battle_id:3,..Default::default()};
        net.pos.insert(1,(200,400));net.pos.insert(2,(1200,500));
        let input=aim_inputs(&net,51.0,7.7,(8.0,8.0),(500,300),true).unwrap();
        let now=std::time::Instant::now();
        let mut last=RecentAim {inputs:input.clone(),sampled:now-std::time::Duration::from_millis(900),
            force:54.4,angle:51.0,dx:12.5,dy:-1.25};
        let hint=OriginHint {updated:now-std::time::Duration::from_millis(900),uid:1,code:"ready".into(),scale:Some(ShotScale::Normal)};
        assert!(last.can_reference(Some(&input),Some(&hint)));
        assert!(!last.can_reference(Some(&input),None));
        assert!(!last.can_reference(Some(&AimInputs {world_angle:Some(129),..input.clone()}),Some(&hint)));
        assert!(!last.can_reference(Some(&AimInputs {scale:ShotScale::Expanded,..input.clone()}),Some(&hint)));
        assert!(!last.can_reference(Some(&AimInputs {wind:7.8,..input.clone()}),Some(&hint)));
        assert!(!last.can_reference(Some(&AimInputs {target:3,..input.clone()}),Some(&hint)));
        assert!(!last.can_reference(Some(&AimInputs {battle:4,..input.clone()}),Some(&hint)));
        assert!(!last.can_reference(Some(&AimInputs {units:(9.0,8.0),..input.clone()}),Some(&hint)));
        for code in ["scale","angle","moving","read"] {
            assert!(!last.can_reference(Some(&input),Some(&OriginHint {code:code.into(),..hint.clone()})));
        }
        last.sampled=now-std::time::Duration::from_millis(1300);
        assert!(!last.can_reference(Some(&input),Some(&hint)));
    }

    #[test]
    fn target_list_uses_stable_identity_and_cancels_roster_changes_and_drags() {
        let mut net = NetHud { my_id: Some(99), map_w: 2000, map_h: 1000, ..Default::default() };
        for id in [99,8,3,2,9,7,4,6,5,10] { net.pos.insert(id,(id as i64 * 100, 300)); }
        net.names.insert(2,"同名玩家".into());
        net.names.insert(3,"同名玩家".into());
        assert_eq!(opponent_ids(&net),vec![2,3,4,5,6,7,8,9,10]);
        let r = target_slot_rect(1).unwrap();
        assert_eq!(target_at_sidebar(&net,r.x+10,r.y+10),Some(3));
        assert_eq!(released_target(&net,3,r.x+10,r.y+10),Some(3));
        assert_eq!(released_target(&net,3,r.x-40,r.y+10),None);
        assert_eq!(target_slot_rect(8),None);
        let displayed = opponent_ids(&net);
        net.pos.remove(&2);
        assert_eq!(target_from_rows(&displayed,r.x+10,r.y+10),Some(3));
        assert_eq!(released_target(&net,3,r.x+10,r.y+10),None);
        net.pos.remove(&3);
        assert_ne!(selected_target_id(&net),Some(3));
    }

    #[test]
    fn sidebar_and_map_selection_feed_the_same_position_and_wind() {
        let mut net = NetHud { my_id: Some(1), map_w: 2000, map_h: 1000,
            auto_wind10: Some(50), ..Default::default() };
        net.pos.insert(1,(1000,500));
        net.pos.insert(2,(200,500));
        net.pos.insert(3,(1800,500));
        // A known other player must stay selectable after firing.
        net.not_mine.insert(2);
        assert_eq!(selected_target_id(&net),Some(2));
        assert_eq!(wind_dir_hint(&net),Some(-1));
        let size=core::Size::new(400,200);
        let picked=target_at_map(&net,360,100,size).unwrap();
        assert_eq!(picked,3);
        net.target_id=Some(picked);
        assert_eq!(target_position(&net),Some((1800,500)));
        assert_eq!(wind_dir_hint(&net),Some(1));
        assert_eq!(network_aim_offsets(&net,false,size,(16.0,16.0),None),Some((10.0,0.0)));
        assert_eq!(target_at_map(&net,200,100,size),None);
        net.target_id=Some(999);
        assert_eq!(selected_target_id(&net),Some(2));
    }

    #[test]
    fn long_target_names_fit_without_drawing_into_neighbours() -> opencv::Result<()> {
        let text=fit_text("重复名字很长很长\n🙂 opponent with long name",166,0.5,1)?;
        let mut baseline=0;
        assert!(imgproc::get_text_size(&text,imgproc::FONT_HERSHEY_SIMPLEX,0.5,1,&mut baseline)?.width <=166);
        assert!(!text.contains('\n'));
        let mut net=NetHud { my_id:Some(1),..Default::default() };
        net.pos.insert(1,(10,10));
        for uid in 2..=9 { net.pos.insert(uid,(20,20)); net.names.insert(uid,"很长的对手名字".into()); }
        let mut canvas=core::Mat::new_rows_cols_with_default(HUD_HEIGHT,MAP_PANE_WIDTH+SIDEBAR_WIDTH,
            core::CV_8UC3,core::Scalar::all(0.0))?;
        draw_target_list(&mut canvas,&net)?;
        assert_eq!(*canvas.at_2d::<core::Vec3b>(406,MAP_PANE_WIDTH-1)?,core::Vec3b::all(0));
        Ok(())
    }

    #[test]
    fn live_origin_protocol_and_pose_guards() {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
        let line = format!("TNTORIGIN 123 1 {now} 120 1 2070 501 2046 486");
        let row = parse_live_origin_line(&line).unwrap().unwrap();
        let mut net = NetHud { map_w: 3000, map_h: 2000, my_id: Some(1), ..Default::default() };
        net.pos.insert(1, (2070, 501));
        net.pos.insert(2, (1000, 500));
        assert_eq!(live_origin_for_hud(Some(&row), &net, 60.0), Some((2046.0, 486.0)));
        assert_eq!(live_origin_for_hud(Some(&row), &net, 61.0), None);
        net.my_id = Some(2);
        assert_eq!(live_origin_for_hud(Some(&row), &net, 60.0), None);
        net.my_id = Some(1);
        net.pos.insert(1, (2072, 501));
        assert_eq!(live_origin_for_hud(Some(&row), &net, 60.0), None);
        net.pos.insert(1, (2070, 501));
        net.pos.insert(2, (2500, 500));
        assert_eq!(live_origin_for_hud(Some(&row), &net, 60.0), Some((2046.0,486.0)));
        net.pos.insert(2, (1000, 500));
        let stale = LiveOrigin { age_ms: 601, ..row.clone() };
        assert_eq!(live_origin_for_hud(Some(&stale), &net, 60.0), None);
        // Loss/expiry must not silently switch an automatic recommendation
        // from the muzzle back to the player's different coordinates.
        assert_eq!(resolve_aim_origin(Some(&row), &net, 60.0, true), AimOrigin::Muzzle((2046.0,486.0)));
        assert_eq!(resolve_aim_origin(Some(&stale), &net, 60.0, true), AimOrigin::Verifying);
        assert_eq!(resolve_aim_origin(None, &net, 60.0, true), AimOrigin::Verifying);
        assert_eq!(resolve_aim_origin(None, &net, 60.0, false), AimOrigin::Player);
        for invalid in [line.replace(" 486", " NaN"), line.replace(" 2046 ", " 9999 "),
                        line.replace(" 120 1 ", " 181 1 "), format!("{line} extra"),
                        line.replace(&now.to_string(), &(now - 1000).to_string())] {
            assert!(parse_live_origin_line(&invalid).is_none());
        }
        assert!(matches!(parse_live_origin_line("TNTORIGIN-"), Some(None)));
    }

    #[test]
    fn backshots_use_actual_angle_without_body_facing_gate_or_forced_mirror() {
        let now=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
        for (angle,body_left,target_x) in [(60,1,2500),(120,0,1000)] {
            let row=parse_live_origin_line(&format!("TNTORIGIN 123 1 {now} {angle} {body_left} 2070 501 2046 486")).unwrap().unwrap();
            let mut net=NetHud {my_id:Some(1),map_w:3000,map_h:2000,..Default::default()};
            net.pos.insert(1,(2070,501));net.pos.insert(2,(target_x,500));
            assert_eq!(resolve_aim_origin(Some(&row),&net,60.0,true),AimOrigin::Muzzle((2046.0,486.0)));
            for scale in [ShotScale::Normal,ShotScale::Expanded] {
                let dx=if angle<90 {10.0} else {-10.0};
                let (force,world_angle)=compute_world_trajectory(dx,-2.0,angle,if dx<0.0 {-7.7} else {7.7},true,scale).unwrap();
                assert_eq!(world_angle,f64::from(angle));
                let expected=power_for_angle_for_scale(scale,60.0,10.0,-2.0,7.7).unwrap();
                assert!((force-expected).abs()<1e-9);
            }
        }
        // A wind-returning shot must keep its real initial direction on the map.
        assert_eq!(trajectory_draw_angle(80.0,-10.0,true),80.0);
        assert_eq!(trajectory_draw_angle(100.0,10.0,true),100.0);
        // Explicit manual elevations retain the existing target-relative convention.
        assert_eq!(trajectory_draw_angle(60.0,-10.0,false),120.0);
        let mut net=NetHud {my_id:Some(1),..Default::default()};
        net.pos.insert(1,(100,500));net.pos.insert(2,(110,500));
        // Target is right of the player but left of a right-side muzzle.
        assert_eq!(target_relative_to_world_wind(&net,7.7,-0.1),7.7);
        net.pos.insert(2,(90,500));
        assert_eq!(target_relative_to_world_wind(&net,7.7,0.1),-7.7);
    }

    #[test]
    fn continuous_ready_status_keeps_timestamp_but_invalid_state_clears_origin() {
        let now=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
        let row=parse_live_origin_line(&format!("TNTORIGIN 123 1 {now} 120 1 2070 501 2046 486")).unwrap().unwrap();
        let ready=parse_origin_hint(&format!("TNTORIGINSTATUS 123 1 {now} ready 1.0")).unwrap();
        let mut sample=Some(row.clone());
        apply_origin_hint(&mut sample,&ready);
        assert_eq!(sample.as_ref().unwrap().updated,row.updated);
        assert_eq!(sample.as_ref().unwrap().age_ms,row.age_ms);
        apply_origin_hint(&mut sample,&OriginHint {code:"moving".into(),..ready.clone()});
        assert!(sample.is_none());
        sample=Some(row.clone());
        apply_origin_hint(&mut sample,&OriginHint {scale:Some(ShotScale::Expanded),..ready.clone()});
        assert!(sample.is_none());
        sample=Some(row);
        apply_origin_hint(&mut sample,&OriginHint {uid:2,..ready});
        assert!(sample.is_none());
    }

    #[test]
    fn expanded_scale_switches_before_shot_and_survives_only_same_battle_gaps() {
        let now=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
        let expanded=parse_live_origin_line(&format!("TNTORIGIN 123 1 {now} 120 1 2070 501 2022 471 2.0")).unwrap().unwrap();
        assert_eq!(expanded.scale,ShotScale::Expanded);
        assert!(parse_live_origin_line(&format!("TNTORIGIN 123 1 {now} 120 1 2070 501 2022 471 1.0")).is_none());
        assert!(parse_live_origin_line(&format!("TNTORIGIN 123 1 {now} 120 1 2070 501 2046 486 1.5")).is_none());
        let mut net=NetHud {my_id:Some(1),battle_id:3,map_w:3000,map_h:2000,..Default::default()};
        net.pos.insert(1,(2070,501));net.pos.insert(2,(1000,500));
        assert_eq!(live_origin_for_hud(Some(&expanded),&net,60.0),Some((2022.0,471.0)));
        let mut latch=None;
        assert_eq!(update_shot_scale(&mut latch,&net,Some(&expanded),None),ShotScale::Expanded);
        assert_eq!(update_shot_scale(&mut latch,&net,None,None),ShotScale::Expanded);
        let normal=LiveOrigin {scale:ShotScale::Normal,origin:(2046.0,486.0),..expanded.clone()};
        assert_eq!(update_shot_scale(&mut latch,&net,Some(&normal),None),ShotScale::Normal);
        let ready=parse_origin_hint(&format!("TNTORIGINSTATUS 123 1 {now} ready 2.0")).unwrap();
        assert_eq!(update_shot_scale(&mut latch,&net,None,Some(&ready)),ShotScale::Expanded);
        net.battle_id=4;
        assert_eq!(update_shot_scale(&mut latch,&net,None,None),ShotScale::Normal);
        let stale=LiveOrigin {age_ms:601,..expanded.clone()};
        assert_eq!(update_shot_scale(&mut latch,&net,Some(&stale),None),ShotScale::Normal);
        net.my_id=Some(2);
        assert_eq!(update_shot_scale(&mut latch,&net,Some(&expanded),Some(&ready)),ShotScale::Normal);
        net.my_id=None;
        assert_eq!(update_shot_scale(&mut latch,&net,None,None),ShotScale::Normal);
        assert!(latch.is_none());
    }

    #[test]
    fn network_aim_uses_corrected_origin_and_respects_manual_points() {
        let mut net = NetHud { map_w: 3000, map_h: 2000, my_id: Some(1), ..Default::default() };
        net.pos.insert(1, (2070, 501));
        net.pos.insert(2, (1000, 500));
        let offset = network_aim_offsets(&net, false, core::Size::new(300, 200), (8.0, 8.0), Some((2046.0, 486.0))).unwrap();
        assert!((offset.0 - (-1046.0 / 80.0)).abs() < 1e-12);
        assert!((offset.1 - (-14.0 / 80.0)).abs() < 1e-12);
        assert_eq!(network_aim_offsets(&net, true, core::Size::new(300, 200), (8.0, 8.0), Some((2046.0, 486.0))), None);
    }

    #[test]
    fn network_aim_keeps_distinct_world_points_in_the_same_display_pixel() {
        let mut net = NetHud { map_w: 3000, map_h: 2000, my_id: Some(1), ..Default::default() };
        net.pos.insert(1, (1000, 800));
        net.pos.insert(2, (1001, 799));
        // Both positions round to (150,120) on a 450x300 display.
        assert_eq!((1000.0_f64 * 0.15).round(), (1001.0_f64 * 0.15).round());
        assert_eq!((800.0_f64 * 0.15).round(), (799.0_f64 * 0.15).round());
        assert_eq!(network_aim_offsets(&net, false, core::Size::new(300, 200), (8.0, 8.0), None),
                   Some((0.0125, 0.0125)));
    }

    #[test]
    fn network_aim_preserves_aspect_ruler_target_and_direction() {
        let mut net = NetHud { map_w: 3000, map_h: 1000, my_id: Some(1), target_id: Some(3), ..Default::default() };
        net.pos.insert(1, (1000, 600));
        net.pos.insert(2, (1800, 200));
        net.pos.insert(3, (200, 800));
        let source = core::Size::new(300, 250);
        assert_eq!(network_aim_offsets(&net, false, source, (8.0, 20.0), None), Some((-10.0, -2.5)));
        assert_eq!(network_aim_offsets(&net, false, source, (16.0, 40.0), None), Some((-5.0, -1.25)));
        net.target_id = Some(2);
        assert_eq!(network_aim_offsets(&net, false, source, (8.0, 20.0), None), Some((10.0, 5.0)));
    }

    #[test]
    fn network_aim_leaves_manual_points_and_missing_context_to_existing_path() {
        let mut net = NetHud { map_w: 3000, map_h: 1000, my_id: Some(1), ..Default::default() };
        net.pos.insert(1, (1000, 600));
        net.pos.insert(2, (1800, 200));
        let source = core::Size::new(300, 250);
        assert_eq!(network_aim_offsets(&net, true, source, (8.0, 20.0), None), None);
        assert_eq!(network_aim_offsets(&net, false, source, (f64::NAN, 20.0), None), None);
        net.my_id = None;
        assert_eq!(network_aim_offsets(&net, false, source, (8.0, 20.0), None), None);
        net.my_id = Some(1);
        net.map_w = 0;
        assert_eq!(network_aim_offsets(&net, false, source, (8.0, 20.0), None), None);
    }

    #[test]
    fn map_cache_updates_for_new_screenshot_and_display_size() -> opencv::Result<()> {
        let first = Arc::new(core::Mat::new_rows_cols_with_default(
            2, 2, core::CV_8UC3, core::Scalar::all(10.0)
        )?);
        let second = Arc::new(core::Mat::new_rows_cols_with_default(
            2, 2, core::CV_8UC3, core::Scalar::all(20.0)
        )?);
        let mut memo = MapDisplayCache(None);
        let size = core::Size::new(4, 4);
        let data = memo.get(&first, size)?.data();
        assert_eq!(memo.get(&Arc::clone(&first), size)?.data(), data);
        assert_eq!(*memo.get(&second, size)?.at_2d::<core::Vec3b>(0, 0)?,
                   core::Vec3b::from([20, 20, 20]));
        assert_eq!(memo.get(&second, core::Size::new(8, 6))?.size()?, core::Size::new(8, 6));
        assert_eq!(*first.at_2d::<core::Vec3b>(0, 0)?, core::Vec3b::from([10, 10, 10]));
        Ok(())
    }

    #[test]
    fn trajectory_cache_invalidates_every_input_and_retains_unreachable() {
        let mut memo = LastValue::new();
        let mut calls = 0;
        let cases = [
            (12.0, 2.0, 65.0, 7.1, true,ShotScale::Normal,None),
            (f64::from_bits(12.0_f64.to_bits() + 1), 2.0, 65.0, 7.1, true,ShotScale::Normal,None),
            (12.0, 3.0, 65.0, 7.1, true,ShotScale::Normal,None),
            (12.0, 3.0, 66.0, 7.1, true,ShotScale::Normal,None),
            (12.0, 3.0, 66.0, -7.1, true,ShotScale::Normal,None),
            (12.0, 3.0, 66.0, -7.1, false,ShotScale::Normal,None),
            (-12.0, 3.0, 66.0, -7.1, false,ShotScale::Normal,None),
            (-12.0, 3.0, 66.0, -7.1, false,ShotScale::Expanded,None),
            (-12.0, 3.0, 66.0, -7.1, false,ShotScale::Expanded,Some(114)),
        ];
        for (i, key) in cases.into_iter().enumerate() {
            for _ in 0..2 {
                let result = *memo.get_or_compute(key, || {
                    calls += 1;
                    None::<(f64, f64)>
                });
                assert_eq!(result, None);
            }
            assert_eq!(calls, i + 1);
        }
        // Returning to an older target must recompute; the cache is bounded.
        let expected = compute_fixed_trajectory(12.0, 2.0, 65.0, 7.1,ShotScale::Normal);
        assert!(expected.is_some());
        assert_eq!(cached_trajectory(&mut memo, cases[0]), expected);
    }

    #[test]
    fn path_cache_keeps_world_direction_and_exact_parameters() {
        let mut memo = LastValue::new();
        let mut calls = 0;
        for key in [(65.0,60.0,7.1,ShotScale::Normal), (115.0,60.0,-7.1,ShotScale::Normal),
                    (115.0,61.0,-7.1,ShotScale::Normal), (115.0,61.0,7.1,ShotScale::Normal),
                    (115.0,61.0,7.1,ShotScale::Expanded)] {
            let expected = simulate_path_for_scale(key.3,key.0,key.1,key.2);
            for _ in 0..2 {
                assert_eq!(memo.get_or_compute(key, || {
                    calls += 1;
                    simulate_path_for_scale(key.3,key.0,key.1,key.2)
                }), &expected);
            }
        }
        assert_eq!(calls, 5);
    }

    #[test]
    fn angle_roi_rejects_moved_pixels_with_identical_channel_sums() -> opencv::Result<()> {
        let mut a = core::Mat::zeros(2, 2, core::CV_8UC3)?.to_mat()?;
        let mut b = a.try_clone()?;
        *a.at_2d_mut::<core::Vec3b>(0, 0)? = core::Vec3b::from([10, 20, 30]);
        *b.at_2d_mut::<core::Vec3b>(1, 1)? = core::Vec3b::from([10, 20, 30]);
        assert_eq!(core::sum_elems(&a)?, core::sum_elems(&b)?);
        assert!(!same_angle_roi(&a, &b)?);
        assert!(same_angle_roi(&a, &a.try_clone()?)?);
        assert!(!same_angle_roi(&a, &core::Mat::zeros(1, 4, core::CV_8UC3)?.to_mat()?)?);
        Ok(())
    }

    #[test]
    fn live_xy_default_on_and_explicit_off() {
        assert!(live_xy_enabled_from_env(None));
        assert!(live_xy_enabled_from_env(Some("1")));
        assert!(live_xy_enabled_from_env(Some("yes")));
        assert!(!live_xy_enabled_from_env(Some("0")));
        assert!(!live_xy_enabled_from_env(Some("false")));
        assert!(!live_xy_enabled_from_env(Some(" OFF ")));
        assert!(!live_xy_enabled_from_env(Some("No")));
    }

    #[test]
    fn memory_angle_defaults_on_and_explicit_off() {
        assert!(live_angle_enabled_from_env(None));
        assert!(live_angle_enabled_from_env(Some("1")));
        assert!(live_angle_enabled_from_env(Some("yes")));
        assert!(live_angle_enabled_from_env(Some("")));
        for val in ["0", "false", "OFF", " No "] {
            assert!(!live_angle_enabled_from_env(Some(val)));
        }
    }

    #[test]
    fn saved_minimap_geo_validation_and_default_ocr_off() {
        assert_eq!(parse_cached_map_geo("42,97,327,288\n"), Some((42, 97, 327, 288)));
        assert_eq!(parse_cached_map_geo("46, 101, 359, 338"), Some((46, 101, 359, 338)));
        assert_eq!(parse_cached_map_geo("0,0,300"), None);
        assert_eq!(parse_cached_map_geo("-20,97,327,288"), None);
        assert_eq!(parse_cached_map_geo("20,97,0,288"), None);
        assert_eq!(parse_cached_map_geo("a,97,327,288"), None);
        assert!(!should_capture_angle_ocr(true, false));
        assert!(!should_capture_angle_ocr(true, true));
        assert!(!should_capture_angle_ocr(false, false));
        assert!(should_capture_angle_ocr(false, true));
    }

    #[test]
    fn angle_source_button_draws_within_existing_sidebar() -> opencv::Result<()> {
        let total_width = MAP_PANE_WIDTH + SIDEBAR_WIDTH;
        let auto_btn = hud_controls()[30];
        let source_btn = hud_controls()[31];
        assert!(auto_btn.x + auto_btn.width < source_btn.x);
        assert!(source_btn.x + source_btn.width <= total_width);
        assert_eq!(source_btn.y, auto_btn.y);
        let mut canvas = core::Mat::new_rows_cols_with_default(
            580, total_width, core::CV_8UC3, core::Scalar::all(0.0)
        )?;
        draw_btn(&mut canvas, auto_btn, "Sync", true)?;
        draw_btn(&mut canvas, source_btn, "MEM", true)?;
        let mem_pixel = canvas.at_2d::<core::Vec3b>(source_btn.y + 3, source_btn.x + 3)?;
        assert!(mem_pixel[0] > 0 || mem_pixel[1] > 0 || mem_pixel[2] > 0);
        draw_btn(&mut canvas, source_btn, "OCR", false)?;
        let ocr_pixel = canvas.at_2d::<core::Vec3b>(source_btn.y + 3, source_btn.x + 3)?;
        assert!(ocr_pixel[0] > 0 || ocr_pixel[1] > 0 || ocr_pixel[2] > 0);
        Ok(())
    }

    #[test]
    fn optional_live_angle_protocol_validates_client_formula_and_source() {
        let row = parse_live_angle_line("TNTANGLE 3056 74108 110 65 5.0 1")
            .unwrap().unwrap();
        assert_eq!(row.uid, 74108);
        assert_eq!(row.fire_angle, 110);
        assert_eq!(live_angle_for_hud(Some(&row), Some(74108)), Some(70.0));
        assert_eq!(selected_memory_angle(true, true, Some(&row), Some(74108)), Some(70.0));
        assert_eq!(selected_memory_angle(true, false, Some(&row), Some(74108)), None);
        assert_eq!(selected_memory_angle(false, true, Some(&row), Some(74108)), None);
        // No network stream: accept the probe's separately verified self UID.
        assert_eq!(live_angle_for_hud(Some(&row), None), Some(70.0));
        // A real disagreement between network and local self still rejects.
        assert_eq!(live_angle_for_hud(Some(&row), Some(12345)), None);
        assert!(matches!(parse_live_angle_line("TNTANGLE-"), Some(None)));
        assert!(parse_live_angle_line("TNTANGLE 3056 74108 111 65 5.0 1").is_none());
        assert!(parse_live_angle_line("TNTANGLE 3056 74108 110 65 NaN 1").is_none());
        assert!(parse_live_angle_line("TNTANGLE 3056 74108 110 65 5.0 2").is_none());
        assert!(parse_live_angle_line("TNTANGLE 3056 74108 110 65 5.0 1 extra").is_none());
        let unflipped = parse_live_angle_line("TNTANGLE 3056 74108 50 48 -2.0 0")
            .unwrap().unwrap();
        assert_eq!(live_angle_for_hud(Some(&unflipped), Some(74108)), Some(50.0));
        let stale = LiveAngle {
            updated: std::time::Instant::now() - std::time::Duration::from_secs(3),
            ..row
        };
        assert_eq!(live_angle_for_hud(Some(&stale), Some(74108)), None);
    }

    #[test]
    fn live_xy_protocol_parses_and_rejects_invalid_rows() {
        let parsed = parse_live_xy_line("TNTXY 10161 2 55437:2125:614 74108:465:615")
            .unwrap().unwrap();
        assert_eq!(parsed.get(&55437), Some(&(2125, 614)));
        assert_eq!(parsed.get(&74108), Some(&(465, 615)));
        assert!(matches!(parse_live_xy_line("TNTXY-"), Some(None)));
        assert!(parse_live_xy_line("TNTXY 10161 2 1:3:4").is_none());
        assert!(parse_live_xy_line("TNTXY 10161 2 1:3:4 1:5:6").is_none());
        assert!(parse_live_xy_line("TNTXY 10161 1 1:3:4 junk").is_none());
        assert!(parse_live_xy_line("TNTXY 10161 1 1:999999:4").is_none());
    }

    #[test]
    fn live_xy_only_overlays_my_own_graphics_pos_when_fresh() {
        let mut net = NetHud {
            map_w: 3000,
            map_h: 1500,
            conns: 1,
            my_id: Some(74108),
            ..Default::default()
        };
        net.pos.insert(55437, (2125, 1));
        net.pos.insert(74108, (465, 1));
        let mut pos = HashMap::new();
        pos.insert(55437, (2125, 614));
        pos.insert(74108, (465, 615));
        pos.insert(111111, (1, 2)); // 未经网络确认的角色不能插入
        let fresh = LiveXY { updated: std::time::Instant::now(), pos: pos.clone() };
        assert_eq!(overlay_live_xy(&mut net, Some(&fresh)), 1);
        assert_eq!(net.pos.get(&55437), Some(&(2125, 1))); // 敌方网络坐标必须不变
        assert_eq!(net.pos.get(&74108), Some(&(465, 615))); // 仅自己的显示坐标跟走路
        assert!(!net.pos.contains_key(&111111));

        let stale = LiveXY {
            updated: std::time::Instant::now() - std::time::Duration::from_secs(5),
            pos,
        };
        net.pos.insert(74108, (465, 70));
        assert_eq!(overlay_live_xy(&mut net, Some(&stale)), 0);
        assert_eq!(net.pos.get(&74108), Some(&(465, 70)));

        net.my_id = None;
        assert_eq!(overlay_live_xy(&mut net, Some(&fresh)), 0);
        net.my_id = Some(74108);
        net.conns = 0;
        assert_eq!(overlay_live_xy(&mut net, Some(&fresh)), 0);
    }
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

    fn wind_seed_ev(seed: i64, wind10: Option<i32>) -> GameEvent {
        GameEvent::WindSeed {
            round: 2,
            order: vec![74108, 0, 73884],
            speed: 100,
            seed,
            wind10,
        }
    }

    #[test]
    fn init_clears_auto_wind() {
        let mut st = NetHud {
            wind_seed: 449,
            auto_wind10: Some(71),
            ..Default::default()
        };
        apply_net_event(
            &mut st,
            &GameEvent::Init {
                map_w: 2000,
                map_h: 1000,
                players: vec![],
            },
            "1111rust",
        );
        assert_eq!(st.auto_wind10, None);
        assert_eq!(st.wind_seed, 0);
    }

    #[test]
    fn first_wind_seed_decodes_without_calibration() {
        let mut st = NetHud::default();
        apply_net_event(&mut st, &wind_seed_ev(449, Some(71)), "1111rust");
        assert_eq!(st.auto_wind10, Some(71));
        assert_eq!(st.active, Some(74108));
        assert_eq!(st.wind_seed, 449);
        assert_eq!(st.turn_seq, 1);
    }

    #[test]
    fn wind_dir_hint_uses_decoded_sign() {
        let mut st = NetHud {
            my_id: Some(1),
            wind_seed: -447,
            auto_wind10: Some(-60),
            ..Default::default()
        };
        st.pos.insert(1, (100, 100));
        // 世界风向左,敌人在右 → 逆风
        st.pos.insert(2, (500, 100));
        assert_eq!(wind_dir_hint(&st), Some(-1));
        // 敌人在左 → 顺风
        st.pos.insert(2, (50, 100));
        assert_eq!(wind_dir_hint(&st), Some(1));
    }

    #[test]
    fn wind_dir_hint_none_for_zero_or_missing_wind() {
        let mut st = NetHud {
            my_id: Some(1),
            auto_wind10: Some(0),
            ..Default::default()
        };
        st.pos.insert(1, (100, 100));
        st.pos.insert(2, (500, 100));
        assert_eq!(wind_dir_hint(&st), None);
        st.auto_wind10 = None;
        assert_eq!(wind_dir_hint(&st), None);
    }

    #[test]
    fn wind_dir_hint_works_when_raw_is_zero() {
        // raw=0 的完整 Action 也能解出真风(如 10.0),方向提示照常工作
        let mut st = NetHud {
            my_id: Some(1),
            wind_seed: 0,
            auto_wind10: Some(100),
            ..Default::default()
        };
        st.pos.insert(1, (100, 100));
        st.pos.insert(2, (500, 100));
        assert_eq!(wind_dir_hint(&st), Some(1));
    }

    #[test]
    fn wind_dir_hint_fallback_is_deterministic() {
        // 多人局未选目标:默认敌人 = id 最小者,与"目标"按钮轮换首个一致;
        // 若靠 HashMap 遍历顺序,每次插入顺序不同结果会飘
        let mut st = NetHud {
            my_id: Some(1),
            auto_wind10: Some(50), // 世界风向右
            ..Default::default()
        };
        st.pos.insert(1, (100, 100));
        st.pos.insert(9, (500, 100));
        st.pos.insert(3, (50, 100)); // id 最小,在我方左侧
        assert_eq!(wind_dir_hint(&st), Some(-1)); // 向右风打左侧 = 逆
        // 显式目标优先于默认
        st.target_id = Some(9);
        assert_eq!(wind_dir_hint(&st), Some(1));
    }

    #[test]
    fn auto_wind_presentation_matches_client_formula() {
        // |world10|/10 × (hint×F翻转);hint 缺失退回世界风符号
        assert_eq!(auto_wind_present(71, Some(1), false), 7.1);
        assert_eq!(auto_wind_present(71, Some(-1), false), -7.1);
        assert_eq!(auto_wind_present(-60, Some(1), false), 6.0);
        assert_eq!(auto_wind_present(-60, None, false), -6.0);
        assert_eq!(auto_wind_present(71, Some(1), true), -7.1);
        assert_eq!(auto_wind_present(0, None, false), 0.0);
    }

    #[test]
    fn manual_override_sign_still_applies() {
        let mut st = NetHud {
            my_id: Some(1),
            auto_wind10: Some(-60),
            ..Default::default()
        };
        st.pos.insert(1, (100, 100));
        st.pos.insert(2, (500, 100));
        // 逆风 → 手动输入的大小被附负号
        assert_eq!(apply_auto_wind_sign(&st, 5.0, false), -5.0);
        // F 翻转 → 顺风
        assert_eq!(apply_auto_wind_sign(&st, 5.0, true), 5.0);
        // 无 hint → 原样返回
        let bare = NetHud::default();
        assert_eq!(apply_auto_wind_sign(&bare, 5.0, false), 5.0);
    }
}
