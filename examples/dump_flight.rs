// FLIGHT 前后各玩家 MOVE/FIRE 位置,验证 path 语义
use tnt_comput::net::{tunnel, Tracker, TunnelEvent, decode_ws_message};
use tnt_comput::proto::GameEvent;
use std::collections::HashMap;

fn main() {
    let data = std::fs::read("/tmp/tnt-tunnel-records.bin").unwrap();
    let mut parser = tunnel::Parser::default();
    let mut tracker = Tracker::default();
    let mut pos: HashMap<u64, (i64,i64)> = HashMap::new();
    for rec in parser.feed(&data) {
        for ev in tracker.on_record(&rec) {
            let TunnelEvent::Msg { c2s, msg, .. } = ev else { continue };
            if c2s { continue; }
            for e in decode_ws_message(&msg) {
                match e {
                    GameEvent::Init { players, .. } => {
                        for p in players { pos.insert(p.id, (p.x, p.y)); }
                        println!("INIT {:?}", pos);
                    }
                    GameEvent::Move { id, x, y } => {
                        let old = pos.insert(id, (x,y));
                        println!("MOVE id={id} ({x},{y}) old={old:?}");
                    }
                    GameEvent::Flight { id, path } => {
                        println!("FLIGHT id={id} start={:?} end={:?} npts={} cur={:?}",
                            path.first(), path.last(), path.len(), pos.get(&id));
                        if let Some(&d) = path.last() { pos.insert(id, d); }
                    }
                    GameEvent::Fire { id, x, y, .. } => {
                        println!("FIRE id={id} at=({x},{y}) tracked={:?}", pos.get(&id));
                        pos.insert(id, (x,y));
                    }
                    _ => {}
                }
            }
        }
    }
}
