use std::io::Read;

use flate2::read::ZlibDecoder;
use tnt_comput::net::tunnel;
use tnt_comput::net::{Tracker, TunnelEvent};
use tnt_comput::proto::{
    decode_envelope, parse_envelope, parse_fields, Field, FieldSlice, GameEvent, Wire,
};

fn hex_prefix(bytes: &[u8], limit: usize) -> String {
    bytes
        .iter()
        .take(limit)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

fn print_fields(fields: &[Field], depth: usize) {
    let pad = "  ".repeat(depth);
    for field in fields {
        match &field.wire {
            Wire::Varint(value) => {
                println!("{pad}field {} varint {} (signed {})", field.no, value, *value as i64);
            }
            Wire::Fixed32(bits) => {
                println!(
                    "{pad}field {} fixed32 0x{bits:08x} f32={}",
                    field.no,
                    f32::from_bits(*bits)
                );
            }
            Wire::Fixed64(bits) => {
                println!(
                    "{pad}field {} fixed64 0x{bits:016x} f64={}",
                    field.no,
                    f64::from_bits(*bits)
                );
            }
            Wire::Bytes(bytes) => {
                println!(
                    "{pad}field {} bytes len={} hex={}",
                    field.no,
                    bytes.len(),
                    hex_prefix(bytes, 96)
                );
                if depth < 4 {
                    if let Some(sub) = parse_fields(bytes) {
                        print_fields(&sub, depth + 1);
                    }
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/tnt-tunnel-records.bin".to_string());
    let bytes = std::fs::read(&path)?;
    let mut parser = tunnel::Parser::default();
    let mut tracker = Tracker::default();
    let mut found = 0usize;

    for rec in parser.feed(&bytes) {
        for event in tracker.on_record(&rec) {
            let TunnelEvent::Msg { c2s, msg, .. } = event else {
                continue;
            };
            if msg.opcode != 2 || msg.payload.len() < 6 {
                continue;
            }
            let mut decoder = ZlibDecoder::new(&msg.payload[4..]);
            let mut plain = Vec::new();
            if decoder.read_to_end(&mut plain).is_err() {
                continue;
            }
            let Some(env) = parse_envelope(&plain) else {
                continue;
            };
            if env.name != "BattleNotify" {
                continue;
            }
            let Some(fields) = parse_fields(&env.body) else {
                continue;
            };
            if fields.varint(2) != Some(8) {
                continue;
            }

            found += 1;
            println!(
                "\n=== turn #{found} ms={} dir={} envelope_body={} ===",
                rec.ms,
                if c2s { "c2s" } else { "s2c" },
                hex_prefix(&env.body, 256)
            );
            print_fields(&fields, 0);
            // 客户端公式解出的世界风(复用 decode_envelope 的字段完整性门槛)
            for gev in decode_envelope(&env) {
                if let GameEvent::WindSeed {
                    wind10: Some(w), ..
                } = gev
                {
                    println!("decoded wind10={w} world_wind={:.1}", w as f64 / 10.0);
                }
            }
        }
    }

    println!("\nfound {found} BattleNotify type-8 messages");
    Ok(())
}
