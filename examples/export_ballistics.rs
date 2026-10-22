//! Offline-only export of exact protobuf bodies for ballistics research.
//! Usage: export_ballistics <snapshot-root> <new-output.jsonl>
//! Contains private game data. Never opens a device or runs the HUD/game.
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use tnt_comput::net::{tunnel, Tracker, TunnelEvent};
use tnt_comput::proto::{parse_envelope, parse_fields, EnvelopeFraming, FieldSlice};

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err(io::Error::other("usage: export_ballistics <snapshot-root> <new-output.jsonl>"));
    }
    let mut files: Vec<_> = std::fs::read_dir(&args[1])?
        .collect::<io::Result<Vec<_>>>()?
        .into_iter()
        .map(|entry| entry.path().join("raw.bin"))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    let file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&args[2])?;
    let mut out = BufWriter::new(file);
    let mut messages = 0;
    let mut dead = 0;
    let mut trailing = 0;
    for path in &files {
        let source = path.parent().unwrap().file_name().unwrap().to_str()
            .ok_or_else(|| io::Error::other("non-UTF8 snapshot name"))?;
        if !source.bytes().all(|b| b.is_ascii_digit() || b == b'-') {
            return Err(io::Error::other("snapshot names must contain only digits and hyphens"));
        }
        let bytes = std::fs::read(path)?;
        let records = tunnel::Parser::default().feed(&bytes);
        let mut tracker = Tracker::default();
        let mut offset = 0usize;
        for (index, record) in records.iter().enumerate() {
            for event in tracker.on_record(record) {
                let (pid, fd, c2s, msg) = match event {
                    TunnelEvent::Dead { .. } => { dead += 1; continue; }
                    TunnelEvent::Msg { pid, fd, c2s, msg } => (pid, fd, c2s, msg),
                    _ => continue,
                };
                if msg.opcode != 2 || msg.payload.len() < 6 { continue; }
                let mut plain = Vec::new();
                if flate2::read::ZlibDecoder::new(&msg.payload[4..]).read_to_end(&mut plain).is_err() {
                    continue;
                }
                let declared = u32::from_be_bytes(msg.payload[..4].try_into().unwrap()) as usize;
                if declared != plain.len() { continue; }
                let Some(env) = parse_envelope(&plain) else { continue };
                let Some(fields) = parse_fields(&env.body) else { continue };
                if env.name != "FireReq" && !(env.name == "BattleNotify" &&
                    matches!(fields.varint(2), Some(0 | 8 | 9 | 11 | 14 | 15))) { continue; }
                let framing = if env.framing == EnvelopeFraming::UpdatedBe16 { "be16" } else { "le32" };
                let hex: String = env.body.iter().map(|b| format!("{b:02x}")).collect();
                writeln!(out,
                    "{{\"source\":\"{source}\",\"record\":{index},\"raw_offset\":{offset},\"message\":{messages},\"pid\":{pid},\"stream\":{fd},\"game_ms\":{},\"record_kind\":{},\"c2s\":{c2s},\"framing\":\"{framing}\",\"name\":\"{}\",\"body_hex\":\"{hex}\"}}",
                    record.ms, record.kind, env.name)?;
                messages += 1;
            }
            offset += tunnel::HDR_LEN + record.payload.len();
        }
        trailing += bytes.len() - offset;
    }
    out.flush()?;
    eprintln!("files={} messages={messages} dead_streams={dead} trailing_bytes={trailing}", files.len());
    Ok(())
}
