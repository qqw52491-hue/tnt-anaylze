//! Append-only offline research recordings. No changes to game/physics behavior.
use crate::net::tunnel;
use crate::proto::GameEvent;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn optional<T: ToString>(v: Option<T>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "null".into())
}
fn float(v: f64) -> String {
    if v.is_finite() {
        v.to_string()
    } else {
        "null".into()
    }
}
fn xy(p: Option<(i64, i64)>) -> String {
    p.map(|(x, y)| format!("[{x},{y}]"))
        .unwrap_or_else(|| "null".into())
}

#[derive(Default)]
pub struct Context {
    pub battle: u64,
    pub round: u64,
    pub turn: u64,
    pub wind10: Option<i32>,
    pub my_id: Option<u64>,
}

pub struct Capture {
    pub dir: PathBuf,
    raw: File,
    events: File,
    offset: u64,
    seq: u64,
    failed: bool,
}
impl Capture {
    pub fn open(root: &Path, raw_override: Option<&Path>) -> io::Result<Self> {
        std::fs::create_dir_all(root)?;
        // create_dir is atomic; even same-ms / same-PID starts cannot overwrite.
        let mut index = 0u64;
        let dir = loop {
            let dir = root.join(format!("{}-{}-{index}", epoch_ms(), std::process::id()));
            match std::fs::create_dir(&dir) {
                Ok(()) => break dir,
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => index += 1,
                Err(e) => return Err(e),
            }
        };
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        let raw_path = raw_override
            .map(Path::to_path_buf)
            .unwrap_or_else(|| dir.join("raw.bin"));
        let raw = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&raw_path)?;
        let events = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(dir.join("events.jsonl"))?;
        let manifest = format!(
            "{{\"schema\":1,\"created_unix_ms\":{},\"raw_path\":{},\"raw_format\":\"tnt tunnel records v1, 21-byte header\",\"tls_plaintext_kinds\":{{\"tx\":11,\"rx\":12}},\"tls_stream_ids\":\"fd bit 31 set: SSL lifetime ID, not OS socket; peer may be unknown with Qt memory BIO\",\"physics_changed\":{},\"physics_model\":{},\"world_units_per_hud_unit\":{},\"fit_ready\":false,\"note\":\"Events are not automatically paired; decoded defaults require raw-field validation; Fire xy semantics and projectile paths unresolved. HUD context is a snapshot, not verified ground truth. Raw packets may contain private game data.\"}}\n",
            epoch_ms(),
            quote(&raw_path.to_string_lossy()),
            crate::physics::is_recorded_model(),
            quote(crate::physics::model_name()),
            crate::physics::world_units_per_hud_unit()
        );
        std::fs::write(dir.join("manifest.json"), manifest)?;
        Ok(Self {
            dir,
            raw,
            events,
            offset: 0,
            seq: 0,
            failed: false,
        })
    }
    // On a disk error, stop this recording and warn once, rather than pretend it succeeded.
    pub fn record(&mut self, rec: &tunnel::Record) -> Option<u64> {
        if self.failed {
            return None;
        }
        let offset = self.offset;
        let result = (|| -> io::Result<()> {
            let len = u32::try_from(rec.payload.len())
                .map_err(|_| io::Error::other("record too large"))?;
            // One header write instead of five small writes. Preserve the exact
            // tunnel layout and immediate visibility to offline/live readers.
            let mut header = [0u8; tunnel::HDR_LEN];
            header[0] = rec.kind;
            header[1..5].copy_from_slice(&rec.pid.to_le_bytes());
            header[5..9].copy_from_slice(&rec.fd.to_le_bytes());
            header[9..17].copy_from_slice(&rec.ms.to_le_bytes());
            header[17..21].copy_from_slice(&len.to_le_bytes());
            self.raw.write_all(&header)?;
            self.raw.write_all(&rec.payload)?;
            self.raw.flush()?;
            Ok(())
        })();
        if let Err(e) = result {
            self.fail(e);
            return None;
        }
        self.offset += tunnel::HDR_LEN as u64 + rec.payload.len() as u64;
        Some(offset)
    }
    fn fail(&mut self, e: io::Error) {
        self.failed = true;
        eprintln!(
            "⚠️ 数据录制已停止（HUD 继续运行）: {}: {e}",
            self.dir.display()
        );
    }
    pub fn event(
        &mut self,
        connection: u64,
        rec: &tunnel::Record,
        raw_offset: Option<u64>,
        c2s: bool,
        ev: &GameEvent,
        ctx: &Context,
    ) {
        if self.failed {
            return;
        }
        self.seq += 1;
        let data = match ev {
            GameEvent::Init {
                map_w,
                map_h,
                players,
            } => format!(
                "{{\"kind\":\"battle_init\",\"map_w\":{map_w},\"map_h\":{map_h},\"players\":[{}]}}",
                players
                    .iter()
                    .map(|p| format!(
                        "{{\"id\":{},\"name\":{},\"xy\":[{},{}],\"angle_hint\":{}}}",
                        p.id,
                        quote(&p.name),
                        p.x,
                        p.y,
                        p.angle_hint
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            GameEvent::FireReq {
                round,
                left,
                angle,
                power,
            } => format!(
                "{{\"kind\":\"fire_request\",\"request_round\":{round},\"left\":{left},\"angle\":{angle},\"power\":{},\"field_presence_verified\":true}}",
                float(*power)
            ),
            GameEvent::Fire {
                id,
                x,
                y,
                angle,
                shots,
            } => format!(
                "{{\"kind\":\"fire_notification\",\"shooter\":{id},\"notification_xy\":[{x},{y}],\"angle\":{},\"shots\":{shots},\"xy_semantics\":\"unverified\",\"projectile_path\":null,\"impact_xy\":null,\"field_presence_verified\":false}}",
                angle.map(float).unwrap_or_else(|| "null".into())
            ),
            GameEvent::WindSeed {
                round,
                order,
                speed,
                seed,
                wind10,
            } => format!(
                "{{\"kind\":\"wind\",\"round\":{round},\"order\":{order:?},\"speed\":{speed},\"raw\":{seed},\"world_wind10\":{}}}",
                optional(*wind10)
            ),
            GameEvent::Move { id, x, y } => {
                format!("{{\"kind\":\"player_move\",\"player\":{id},\"xy\":[{x},{y}]}}")
            }
            GameEvent::MoveReq { x } => format!("{{\"kind\":\"move_request\",\"x\":{x}}}"),
            GameEvent::PlayerUpdate { id, pos, hp, attr8 } => format!(
                "{{\"kind\":\"player_update\",\"player\":{id},\"xy\":{},\"hp\":{},\"attr8\":{}}}",
                xy(*pos),
                optional(*hp),
                optional(*attr8)
            ),
            GameEvent::Flight { id, path } => format!(
                "{{\"kind\":\"player_transport\",\"player\":{id},\"path\":[{}],\"not_projectile\":true}}",
                path.iter()
                    .map(|(x, y)| format!("[{x},{y}]"))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            _ => format!(
                "{{\"kind\":\"other\",\"decoded_debug\":{}}}",
                quote(&format!("{ev:?}"))
            ),
        };
        let line = format!(
            "{{\"schema\":1,\"event_seq\":{},\"received_unix_ms\":{},\"game_ms\":{},\"connection\":{connection},\"pid\":{},\"fd\":{},\"direction\":{},\"raw_record_kind\":{},\"raw_record_offset\":{},\"hud_context_unverified\":{{\"battle\":{},\"round\":{},\"turn\":{},\"world_wind10\":{},\"my_id\":{}}},\"fit_ready\":false,\"data\":{data}}}",
            self.seq,
            epoch_ms(),
            rec.ms,
            rec.pid,
            rec.fd,
            quote(if c2s { "c2s" } else { "s2c" }),
            rec.kind,
            optional(raw_offset),
            ctx.battle,
            ctx.round,
            ctx.turn,
            optional(ctx.wind10),
            optional(ctx.my_id)
        );
        let result = writeln!(self.events, "{line}").and_then(|_| self.events.flush());
        if let Err(e) = result {
            self.fail(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static TEST_DIR_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    fn root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "tnt-capture-test-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            TEST_DIR_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }
    #[test]
    fn json_escape_and_nonfinite() {
        assert_eq!(quote("a\"\\\n\t\u{1}"), "\"a\\\"\\\\\\n\\t\\u0001\"");
        assert_eq!(float(f64::NAN), "null");
        assert_eq!(optional::<u64>(None), "null");
    }
    #[test]
    fn unique_recordings_and_replay() {
        let root = root();
        let mut c = Capture::open(&root, None).unwrap();
        let c2 = Capture::open(&root, None).unwrap();
        assert_ne!(c.dir, c2.dir);
        let rec = tunnel::Record {
            kind: tunnel::KIND_TX,
            pid: 9,
            fd: 0x8000_1234,
            ms: 0x0102_0304_0506_0708,
            payload: vec![1, 2, 3],
        };
        let offset = c.record(&rec);
        assert_eq!(offset, Some(0));
        c.event(
            1,
            &rec,
            offset,
            true,
            &GameEvent::FireReq {
                round: 1,
                left: false,
                angle: 45,
                power: 60.123456789,
            },
            &Context::default(),
        );
        assert_eq!(c.record(&rec), Some(24));
        let bytes = std::fs::read(c.dir.join("raw.bin")).unwrap();
        let replay = tunnel::Parser::default().feed(&bytes);
        assert_eq!(replay.len(), 2);
        for item in &replay {
            assert_eq!(item.kind, rec.kind);
            assert_eq!(item.pid, rec.pid);
            assert_eq!(item.fd, rec.fd);
            assert_eq!(item.ms, rec.ms);
            assert_eq!(item.payload, rec.payload);
        }
        let text = std::fs::read_to_string(c.dir.join("events.jsonl")).unwrap();
        assert!(text.contains("60.123456789"));
        assert!(text.contains("\"world_wind10\":null"));
        assert!(text.contains("\"fit_ready\":false"));
        let manifest = std::fs::read_to_string(c.dir.join("manifest.json")).unwrap();
        assert!(manifest.contains(&format!("\"physics_model\":{}", quote(crate::physics::model_name()))));
        assert!(manifest.contains("\"world_units_per_hud_unit\":"));
        drop(c);
        drop(c2);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn explicit_raw_path_never_overwrites() {
        let root = root();
        std::fs::create_dir_all(&root).unwrap();
        let raw = root.join("existing.bin");
        std::fs::write(&raw, b"keep").unwrap();
        assert!(Capture::open(&root, Some(&raw)).is_err());
        assert_eq!(std::fs::read(&raw).unwrap(), b"keep");
        std::fs::remove_dir_all(root).unwrap();
    }
}
