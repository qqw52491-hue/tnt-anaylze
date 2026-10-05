// 对照 UPDATE块字段 与 MOVE/FLIGHT 坐标,找位置字段
use tnt_comput::net::{tunnel, Tracker, TunnelEvent, decode_ws_message};
use tnt_comput::proto::{parse_envelope, parse_fields, FieldSlice, GameEvent};
use flate2::read::ZlibDecoder;
use std::io::Read;

fn main() {
    let data = std::fs::read("/tmp/tnt-tunnel-records.bin").unwrap();
    let mut parser = tunnel::Parser::default();
    let mut tracker = Tracker::default();
    for rec in parser.feed(&data) {
        for ev in tracker.on_record(&rec) {
            let TunnelEvent::Msg { c2s, msg, .. } = ev else { continue };
            if c2s || msg.opcode != 2 || msg.payload.len() < 6 { continue; }
            let mut z = ZlibDecoder::new(&msg.payload[4..]);
            let mut plain = Vec::new();
            if z.read_to_end(&mut plain).is_err() { continue; }
            let Some(env) = parse_envelope(&plain) else { continue };
            let Some(fields) = parse_fields(&env.body) else { continue };
            let ntype = fields.varint(2).unwrap_or(0);
            if ntype == 9 {
                let upd = fields.subs(9).into_iter().next().unwrap_or_default();
                for blk in upd.subs(1) {
                    let id = blk.varint(1).unwrap_or(0);
                    let f7 = blk.subs(7).into_iter().next().unwrap_or_default();
                    let f7_1 = f7.varint(1).unwrap_or(0);
                    let f7_2 = f7.subs(2).into_iter().next().unwrap_or_default();
                    let f7_2_1 = f7_2.varint(1).map(|v| v as i64).unwrap_or(0);
                    let f18 = blk.varint(18);
                    let f19 = blk.subs(19).into_iter().next()
                        .and_then(|s| s.varint(1));
                    let f26 = blk.varint(26);
                    let f29 = blk.varint(29);
                    println!("UPD id={id} f7_1={f7_1} f7_2_1={f7_2_1} f18={f18:?} f19_1={f19:?} f26={f26:?} f29={f29:?}");
                }
            } else {
                for e in decode_ws_message(&msg) {
                    match e {
                        GameEvent::Move { id, x, y } => println!("  MOVE id={id} x={x} y={y}"),
                        GameEvent::Flight { id, path } => println!("  FLIGHT id={id} last={:?}", path.last()),
                        GameEvent::Fire { id, x, y, .. } => println!("  FIRE id={id} x={x} y={y}"),
                        _ => {}
                    }
                }
            }
        }
    }
}
