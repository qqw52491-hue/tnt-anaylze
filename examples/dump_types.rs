use tnt_comput::net::{tunnel, Tracker, TunnelEvent};
use tnt_comput::proto::{parse_envelope, parse_fields, FieldSlice};
use std::collections::HashMap;
use flate2::read::ZlibDecoder;
use std::io::Read;

fn main() {
    let data = std::fs::read("/tmp/tnt-tunnel-records.bin").unwrap();
    let mut parser = tunnel::Parser::default();
    let mut tracker = Tracker::default();
    let mut types: HashMap<u64, u64> = HashMap::new();
    let mut envs = 0u64;
    for rec in parser.feed(&data) {
        for ev in tracker.on_record(&rec) {
            let TunnelEvent::Msg { c2s, msg, .. } = ev else { continue };
            if c2s || msg.opcode != 2 || msg.payload.len() < 6 { continue; }
            let mut z = ZlibDecoder::new(&msg.payload[4..]);
            let mut plain = Vec::new();
            if z.read_to_end(&mut plain).is_err() { continue; }
            let Some(env) = parse_envelope(&plain) else { continue };
            envs += 1;
            let Some(fields) = parse_fields(&env.body) else { continue };
            *types.entry(fields.varint(2).unwrap_or(0)).or_default() += 1;
        }
    }
    eprintln!("envs={envs} types={types:?}");
}
