//! Standalone live decoder for the battle protocol.
//!
//!   net_live pcap <file|->          decode a pcap file or `tcpdump -w -` stdin
//!   net_live listen [port]          accept sniffer-tunnel records (adb reverse)
//!
//! Every decoded event is printed as one line.

use std::collections::HashMap;
use std::io::{self, Read};
use std::net::TcpListener;

use tnt_comput::net::tunnel;
use tnt_comput::net::{decode_ws_message, FlowReasm, PcapReader, Tracker, TunnelEvent};
use tnt_comput::proto::GameEvent;

fn fmt_event(prefix: &str, ev: &GameEvent) -> String {
    match ev {
        GameEvent::Init {
            map_w,
            map_h,
            players,
        } => {
            let ps: Vec<String> = players
                .iter()
                .map(|p| {
                    format!(
                        "{} \"{}\"@({},{}) a9={}",
                        p.id, p.name, p.x, p.y, p.angle_hint
                    )
                })
                .collect();
            format!("{prefix} INIT map={map_w}x{map_h} players=[{}]", ps.join(", "))
        }
        GameEvent::WindSeed {
            round,
            order,
            speed,
            seed,
            wind10,
        } => format!(
            "{prefix} WIND round={round} order={order:?} speed={speed} seed={seed} wind10={:?} world={:?}",
            wind10,
            wind10.map(|w| w as f64 / 10.0)
        ),
        GameEvent::PlayerUpdate {
            id,
            pos,
            hp,
            attr8,
        } => {
            let pos_s = pos.map(|(x, y)| format!("({x},{y})")).unwrap_or("-".into());
            format!("{prefix} UPDATE id={id} pos={pos_s} hp={hp:?} a8={attr8:?}")
        }
        GameEvent::Move { id, x, y } => format!("{prefix} MOVE id={id} ->({x},{y})"),
        GameEvent::MoveReq { x } => format!("{prefix} MOVEREQ x={x}"),
        GameEvent::Flight { id, path } => format!(
            "{prefix} FLIGHT id={id} points={} dest={:?}",
            path.len(),
            path.last()
        ),
        GameEvent::Fire {
            id,
            x,
            y,
            angle,
            shots,
        } => format!("{prefix} FIRE id={id} from=({x},{y}) angle={angle:?} shots={shots}"),
        GameEvent::FireReq {
            weapon,
            angle,
            power,
        } => format!("{prefix} FIREREQ weapon={weapon} angle={angle} power={power:.2}"),
        GameEvent::Other { name, ntype, seq } => {
            format!("{prefix} MSG {name} type={ntype:?} seq={seq:?}")
        }
    }
}

fn run_pcap(path: &str) -> io::Result<()> {
    let reader: Box<dyn Read> = if path == "-" {
        Box::new(io::stdin().lock())
    } else {
        Box::new(std::fs::File::open(path)?)
    };
    let mut pcap = PcapReader::new(reader)?;

    // flow key -> per-direction reassembler
    let mut flows: HashMap<String, (FlowReasm, FlowReasm)> = HashMap::new();
    let mut n_pkts = 0u64;
    let mut n_msgs = 0u64;

    while let Some(p) = pcap.next_packet() {
        n_pkts += 1;
        let a = format!("{}.{}.{}.{}:{}", p.src[0], p.src[1], p.src[2], p.src[3], p.sport);
        let b = format!("{}.{}.{}.{}:{}", p.dst[0], p.dst[1], p.dst[2], p.dst[3], p.dport);
        let (key, c2s) = if a < b { (format!("{a}|{b}"), true) } else { (format!("{b}|{a}"), false) };
        let entry = flows.entry(key.clone()).or_default();
        let dir = if c2s { &mut entry.0 } else { &mut entry.1 };
        let msgs = dir.push(p.seq, &p.payload);
        for m in msgs {
            n_msgs += 1;
            let tag = if c2s { "c2s" } else { "s2c" };
            let prefix = format!("[{:.3} {tag}]", p.ts);
            let evs = decode_ws_message(&m);
            if evs.is_empty() {
                println!("{prefix} ws op{} len={} (no events)", m.opcode, m.payload.len());
            }
            for ev in evs {
                println!("{}", fmt_event(&prefix, &ev));
            }
        }
    }
    eprintln!("done: packets={n_pkts} ws_msgs={n_msgs} flows={}", flows.len());
    Ok(())
}

fn handle_tunnel(stream: std::net::TcpStream) -> io::Result<()> {
    let mut stream = stream;
    stream.set_nodelay(true).ok();
    let mut parser = tunnel::Parser::default();
    let mut tracker = Tracker::default();
    let mut buf = [0u8; 65536];

    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        for rec in parser.feed(&buf[..n]) {
            for ev in tracker.on_record(&rec) {
                match ev {
                    TunnelEvent::Connect { pid, fd, peer } => {
                        println!("[tun] connect pid={pid} fd={fd} peer={peer}");
                    }
                    TunnelEvent::Close { pid, fd, peer } => {
                        println!("[tun] close pid={pid} fd={fd} peer={peer}");
                    }
                    TunnelEvent::Dead { pid, fd, peer } => {
                        eprintln!("[tun] ws decoder dead pid={pid} fd={fd} peer={peer}");
                    }
                    TunnelEvent::Msg {
                        pid,
                        fd,
                        c2s,
                        msg: m,
                    } => {
                        let tag = if c2s { "c2s" } else { "s2c" };
                        let prefix = format!("[tun {tag} {pid}:{fd}]");
                        let evs = decode_ws_message(&m);
                        if evs.is_empty() && m.opcode == 2 {
                            println!("{prefix} ws len={} undecoded", m.payload.len());
                        }
                        for ev in evs {
                            println!("{}", fmt_event(&prefix, &ev));
                        }
                    }
                    TunnelEvent::Hello { pid, text } => {
                        println!("[tun] hello pid={pid} {text}");
                    }
                    TunnelEvent::Rand { pid, text } => {
                        println!("[tun] RAND pid={pid} {text}");
                    }
                    TunnelEvent::Log { pid, text } => {
                        println!("[tun] log pid={pid} {text}");
                    }
                }
            }
        }
    }
}

fn run_listen(port: u16) -> io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    println!("[tun] listening on 127.0.0.1:{port} (adb reverse tcp:{port} tcp:{port})");
    for conn in listener.incoming() {
        match conn {
            Ok(s) => {
                println!("[tun] tunnel connection from {:?}", s.peer_addr());
                std::thread::spawn(move || {
                    if let Err(e) = handle_tunnel(s) {
                        eprintln!("[tun] conn error: {e}");
                    }
                    println!("[tun] tunnel connection ended");
                });
            }
            Err(e) => eprintln!("[tun] accept: {e}"),
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("pcap") => {
            let path = args.get(2).map(String::as_str).unwrap_or("-");
            run_pcap(path)
        }
        Some("listen") => {
            let port: u16 = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(19001);
            run_listen(port)
        }
        _ => {
            eprintln!("usage: net_live pcap <file|-> | net_live listen [port]");
            Ok(())
        }
    }
}
