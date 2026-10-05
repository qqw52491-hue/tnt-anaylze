//! Generic protobuf wire-format parser plus the game's message-envelope
//! decoding. No schema is required: fields are emitted in wire order and the
//! game-specific extractors walk the field tree by number.

#[derive(Clone, Debug)]
pub enum Wire {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    Bytes(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct Field {
    pub no: u32,
    pub wire: Wire,
}

pub fn read_varint(buf: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    while pos < buf.len() && shift < 70 {
        let b = buf[pos];
        pos += 1;
        value |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((value, pos));
        }
        shift += 7;
    }
    None
}

/// Parse a buffer as a flat protobuf field list. Returns None on truncation or
/// illegal wire types; callers use this as a "does this look like protobuf"
/// probe too.
pub fn parse_fields(buf: &[u8]) -> Option<Vec<Field>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < buf.len() {
        let (tag, ni) = read_varint(buf, i)?;
        i = ni;
        let no = (tag >> 3) as u32;
        if no == 0 || no > 0x1fff_ffff {
            return None;
        }
        let wire = match tag & 7 {
            0 => {
                let (v, ni) = read_varint(buf, i)?;
                i = ni;
                Wire::Varint(v)
            }
            1 => {
                if i + 8 > buf.len() {
                    return None;
                }
                let v = u64::from_le_bytes(buf[i..i + 8].try_into().ok()?);
                i += 8;
                Wire::Fixed64(v)
            }
            2 => {
                let (l, ni) = read_varint(buf, i)?;
                i = ni;
                let l = l as usize;
                if l > buf.len() - i {
                    return None;
                }
                let b = buf[i..i + l].to_vec();
                i += l;
                Wire::Bytes(b)
            }
            5 => {
                if i + 4 > buf.len() {
                    return None;
                }
                let v = u32::from_le_bytes(buf[i..i + 4].try_into().ok()?);
                i += 4;
                Wire::Fixed32(v)
            }
            _ => return None,
        };
        out.push(Field { no, wire });
    }
    Some(out)
}

pub trait FieldSlice {
    fn varint(&self, no: u32) -> Option<u64>;
    /// All varints for `no`, plus varints decoded from length-delimited
    /// "packed" representations.
    fn varints(&self, no: u32) -> Vec<u64>;
    fn fixed64(&self, no: u32) -> Option<u64>;
    fn f64(&self, no: u32) -> Option<f64>;
    fn bytes(&self, no: u32) -> Option<&[u8]>;
    fn subs(&self, no: u32) -> Vec<Vec<Field>>;
    fn text(&self, no: u32) -> Option<String>;
}

impl FieldSlice for [Field] {
    fn varint(&self, no: u32) -> Option<u64> {
        self.iter().find_map(|f| match (&f.wire, f.no == no) {
            (Wire::Varint(v), true) => Some(*v),
            _ => None,
        })
    }

    fn varints(&self, no: u32) -> Vec<u64> {
        let mut out = Vec::new();
        for f in self.iter().filter(|f| f.no == no) {
            match &f.wire {
                Wire::Varint(v) => out.push(*v),
                Wire::Bytes(b) => {
                    let mut pos = 0;
                    while let Some((v, ni)) = read_varint(b, pos) {
                        if ni == pos {
                            break;
                        }
                        pos = ni;
                        out.push(v);
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn fixed64(&self, no: u32) -> Option<u64> {
        self.iter().find_map(|f| match (&f.wire, f.no == no) {
            (Wire::Fixed64(v), true) => Some(*v),
            _ => None,
        })
    }

    fn f64(&self, no: u32) -> Option<f64> {
        self.fixed64(no).map(f64::from_bits)
    }

    fn bytes(&self, no: u32) -> Option<&[u8]> {
        self.iter().find_map(|f| match (&f.wire, f.no == no) {
            (Wire::Bytes(b), true) => Some(b.as_slice()),
            _ => None,
        })
    }

    fn subs(&self, no: u32) -> Vec<Vec<Field>> {
        self.iter()
            .filter(|f| f.no == no)
            .filter_map(|f| match &f.wire {
                Wire::Bytes(b) => parse_fields(b),
                _ => None,
            })
            .collect()
    }

    fn text(&self, no: u32) -> Option<String> {
        self.bytes(no)
            .and_then(|b| String::from_utf8(b.to_vec()).ok())
    }
}

/// Message framing: `[u32 LE body_len][u32 LE name_len][name][protobuf]`
/// where body_len counts everything after the first word.
pub struct Envelope {
    pub name: String,
    pub body: Vec<u8>,
}

pub fn parse_envelope(buf: &[u8]) -> Option<Envelope> {
    if buf.len() < 8 {
        return None;
    }
    let body_len = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
    if body_len + 4 != buf.len() {
        return None;
    }
    let name_len = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
    let name_end = 8usize.checked_add(name_len)?;
    if name_end > buf.len() {
        return None;
    }
    let name = String::from_utf8(buf[8..name_end].to_vec()).ok()?;
    Some(Envelope {
        name,
        body: buf[name_end..].to_vec(),
    })
}

fn sub_xy(msg: &[Field], field: u32) -> Option<(i64, i64)> {
    let sub = msg.subs(field).into_iter().next()?;
    Some((sub.varint(1)? as i64, sub.varint(2)? as i64))
}

#[derive(Clone, Debug)]
pub struct PlayerInit {
    pub id: u64,
    pub name: String,
    pub x: i64,
    pub y: i64,
    pub angle_hint: u64,
}

/// Client BattleCommandAction::init uses signed 32-bit XORs before display.
pub fn decode_wind10(raw: i64, round: u64, speed: u64, order: &[u64]) -> i32 {
    order.iter().fold(
        raw as i32 ^ round as i32 ^ speed as i32,
        |wind, &id| wind ^ id as i32,
    )
}

#[derive(Clone, Debug)]
pub enum GameEvent {
    /// Battle init: players + map world size.
    Init {
        map_w: u64,
        map_h: u64,
        players: Vec<PlayerInit>,
    },
    /// Turn boundary: f1=round, f2=repeated order, f3=speed, f4=raw seed.
    /// `wind10` is the signed world wind in tenths decoded by
    /// [`decode_wind10`]; None when any required field is missing.
    WindSeed {
        round: u64,
        order: Vec<u64>,
        speed: u64,
        seed: i64,
        wind10: Option<i32>,
    },
    /// Per-player attribute block inside a type-9 update.
    PlayerUpdate {
        id: u64,
        pos: Option<(i64, i64)>,
        hp: Option<u64>,
        attr8: Option<u64>,
    },
    /// Type-11 move notification.
    Move { id: u64, x: i64, y: i64 },
    /// Client -> server local horizontal movement. Field 2 is absolute world x.
    MoveReq { x: i64 },
    /// Type-14 player flight/transport with the complete server trajectory.
    Flight { id: u64, path: Vec<(i64, i64)> },
    /// Type-15 projectile event.
    Fire {
        id: u64,
        x: i64,
        y: i64,
        angle: Option<f64>,
        shots: u32,
    },
    /// Client -> server fire request (only exists when the user shoots).
    FireReq { weapon: u64, angle: u64, power: f64 },
    /// Anything else we only count, e.g. loading progress / ready / chat.
    Other { name: String, ntype: Option<u64>, seq: Option<u64> },
}

fn parse_player_block(msg: &[Field]) -> Option<PlayerInit> {
    let id = msg.varint(1)?;
    let name = msg.text(2).unwrap_or_default();
    let (x, y) = sub_xy(msg, 8)?;
    Some(PlayerInit {
        id,
        name,
        x,
        y,
        angle_hint: msg.varint(9).unwrap_or(0),
    })
}

/// Decode one decompressed envelope body into game events.
pub fn decode_envelope(env: &Envelope) -> Vec<GameEvent> {
    let mut out = Vec::new();
    let Some(fields) = parse_fields(&env.body) else {
        return out;
    };

    match env.name.as_str() {
        "BattleNotify" => {
            let seq = fields.varint(1).unwrap_or(0);
            let ntype = fields.varint(2).unwrap_or(0);
            match ntype {
                // Battle/room init. Outer field 3 holds room data; player
                // blocks sit at its fields 3 and 4, map size at 13/14.
                0 => {
                    if let Some(room) = fields.subs(3).into_iter().next() {
                        let players: Vec<PlayerInit> = room
                            .subs(3)
                            .into_iter()
                            .chain(room.subs(4))
                            .filter_map(|p| parse_player_block(&p))
                            .collect();
                        out.push(GameEvent::Init {
                            map_w: room.varint(13).unwrap_or(0),
                            map_h: room.varint(14).unwrap_or(0),
                            players,
                        });
                    }
                }
                // Turn start: {1: round, 2: repeated order, 3: speed, 4: seed}
                8 => {
                    if let Some(w) = fields.subs(8).into_iter().next() {
                        let round = w.varint(1);
                        let speed = w.varint(3);
                        let seed = w.varint(4).map(|raw| raw as i32 as i64);
                        let order = w.varints(2);
                        let wind10 = match (round, speed, seed, order.is_empty()) {
                            (Some(round), Some(speed), Some(seed), false) => {
                                Some(decode_wind10(seed, round, speed, &order))
                            }
                            _ => None,
                        };
                        out.push(GameEvent::WindSeed {
                            round: round.unwrap_or(0),
                            order,
                            speed: speed.unwrap_or(0),
                            seed: seed.unwrap_or(0),
                            wind10,
                        });
                    }
                }
                // Per-player attr blocks at repeated field 1.
                9 => {
                    if let Some(upd) = fields.subs(9).into_iter().next() {
                        for blk in upd.subs(1) {
                            let Some(id) = blk.varint(1) else { continue };
                            out.push(GameEvent::PlayerUpdate {
                                id,
                                pos: sub_xy(&blk, 22),
                                hp: blk.varint(29),
                                attr8: blk.varint(8),
                            });
                        }
                    }
                }
                // Move: {1: id, 2: {x, y}}
                11 => {
                    if let Some(m) = fields.subs(10).into_iter().next() {
                        if let (Some(id), Some((x, y))) = (m.varint(1), sub_xy(&m, 2)) {
                            out.push(GameEvent::Move {
                                id,
                                x,
                                y,
                            });
                        }
                    }
                }
                // Player flight/transport: {1: id, repeated 4: {x, y}}.
                14 => {
                    if let Some(flight) = fields.subs(13).into_iter().next() {
                        let id = flight.varint(1).unwrap_or(0);
                        let path = flight
                            .subs(4)
                            .into_iter()
                            .filter_map(|p| Some((p.varint(1)? as i64, p.varint(2)? as i64)))
                            .collect();
                        out.push(GameEvent::Flight { id, path });
                    }
                }
                // Projectile: {1: shooter, 3: {x,y}, 10: {repeated shots}}
                15 => {
                    if let Some(f) = fields.subs(14).into_iter().next() {
                        let id = f.varint(1).unwrap_or(0);
                        let (x, y) = sub_xy(&f, 3).unwrap_or((0, 0));
                        let mut angle = None;
                        let mut shots = 0u32;
                        for grp in f.subs(10) {
                            for shot in grp.subs(1) {
                                shots += 1;
                                if angle.is_none() {
                                    angle = shot.f64(3);
                                }
                            }
                        }
                        out.push(GameEvent::Fire {
                            id,
                            x,
                            y,
                            angle,
                            shots,
                        });
                    }
                }
                _ => {
                    out.push(GameEvent::Other {
                        name: env.name.clone(),
                        ntype: Some(ntype),
                        seq: Some(seq),
                    });
                }
            }
        }
        "MoveReq" => {
            if let Some(x) = fields.varint(2) {
                out.push(GameEvent::MoveReq { x: x as i64 });
            }
        }
        "FireReq" => {
            out.push(GameEvent::FireReq {
                weapon: fields.varint(1).unwrap_or(0),
                angle: fields.varint(3).unwrap_or(0),
                power: fields.f64(4).unwrap_or(0.0),
            });
        }
        _ => {
            out.push(GameEvent::Other {
                name: env.name.clone(),
                ntype: None,
                seq: fields.varint(1),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_req_decodes_absolute_x() {
        // {1: 3, 2: 1636} — field2 is the local player's absolute world x.
        let env = Envelope {
            name: "MoveReq".to_string(),
            body: vec![0x08, 0x03, 0x10, 0xe4, 0x0c],
        };
        let events = decode_envelope(&env);
        assert_eq!(events.len(), 1);
        match &events[0] {
            GameEvent::MoveReq { x } => assert_eq!(*x, 1636),
            other => panic!("expected MoveReq, got {other:?}"),
        }
    }

    #[test]
    fn flight_decodes_server_path() {
        // BattleNotify type=14: {1: id, repeated 4: {x, y}, ...} at field 13.
        let env = Envelope {
            name: "BattleNotify".to_string(),
            body: vec![
                0x08, 0x43, 0x10, 0x0e, 0x6a, 0x1c, 0x08, 0xfc, 0xc2, 0x04,
                0x10, 0x00, 0x18, 0x00, 0x22, 0x06, 0x08, 0xbc, 0x03, 0x10, 0xe1,
                0x06, 0x22, 0x06, 0x08, 0xf0, 0x05, 0x10, 0x9e, 0x03, 0x38, 0x27,
                0x40, 0x01,
            ],
        };
        let events = decode_envelope(&env);
        assert_eq!(events.len(), 1);
        match &events[0] {
            GameEvent::Flight { id, path } => {
                assert_eq!(*id, 74108);
                assert_eq!(*path, vec![(444, 865), (752, 414)]);
            }
            other => panic!("expected Flight, got {other:?}"),
        }
    }

    // ---- wind decode regression (frozen wind_decode_validation.json) ----

    /// (round, speed, raw, order, expected wind10) — frozen offline records.
    const FROZEN: &[(u64, u64, i64, &[u64], i32)] = &[
        (1, 100, -447, &[74108, 73884, 0], -60),
        (1, 100, 447, &[73884, 0, 74108], 58),
        (2, 100, 469, &[73884, 74108, 0], 83),
        (2, 100, 449, &[74108, 0, 73884], 71),
        (3, 100, 486, &[73884, 74108, 0], 97),
        (3, 100, 484, &[74108, 0, 73884], 99),
        (4, 100, 465, &[73884, 74108, 0], 81),
        (4, 100, -434, &[74108, 0, 73884], -50),
        (5, 100, -470, &[73884, 74108, 0], -85),
        (5, 100, 467, &[74108, 0, 73884], 82),
        (1, 100, -130217, &[56817, 74108, 0], -65),
        (1, 100, 130226, &[74108, 0, 56817], 90),
        (2, 100, 130260, &[74108, 56817, 0], 63),
        (1, 100, 125170, &[65432, 48235, 26895, 13375, 55164, 74108, 0], 84),
        (1, 100, 125160, &[48235, 26895, 13375, 55164, 74108, 0, 65432], 78),
        (1, 100, -125160, &[26895, 13375, 55164, 74108, 0, 48235, 65432], -66),
        (1, 100, 87282, &[13375, 55164, 74108, 0, 65432, 26895], 63),
        (1, 100, 109417, &[55164, 74108, 0, 26895, 13375], 60),
        (1, 100, -109338, &[74108, 0, 55164, 26895, 13375], -77),
        (2, 100, 109332, &[55164, 26895, 13375, 74108, 0], 66),
        (2, 100, -109319, &[26895, 13375, 74108, 0, 55164], -81),
        (2, 100, -109419, &[13375, 74108, 0, 55164, 26895], -61),
        (2, 100, 109338, &[74108, 0, 55164, 26895, 13375], 76),
        (3, 100, 109412, &[55164, 26895, 13375, 74108, 0], 51),
        (3, 100, 109409, &[26895, 13375, 74108, 0, 55164], 54),
        (3, 100, 35454, &[13375, 0, 55164, 26895], 85),
        (1, 100, 8690, &[65717, 74108, 0], 94),
        (1, 100, 8674, &[74108, 0, 65717], 78),
        (2, 100, 8605, &[74108, 65717, 0], 50),
        (2, 100, 8699, &[65717, 0, 74108], 84),
        (3, 100, 8688, &[74108, 65717, 0], 94),
        (3, 100, -8602, &[65717, 0, 74108], -56),
    ];

    #[test]
    fn decode_wind10_matches_frozen_records() {
        for &(round, speed, raw, order, expected) in FROZEN {
            assert_eq!(
                decode_wind10(raw, round, speed, order),
                expected,
                "round={round} speed={speed} raw={raw} order={order:?}"
            );
        }
    }

    fn push_varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn push_tag(out: &mut Vec<u8>, no: u64, wire: u64) {
        push_varint(out, (no << 3) | wire);
    }

    /// Build a BattleNotify type-8 envelope; `packed` switches the order list
    /// between length-delimited packed and repeated-varint encodings.
    /// raw is encoded as a varint of its sign-extended u64 (proto int32/64).
    fn wind_seed_env(
        round: Option<u64>,
        order: &[u64],
        speed: Option<u64>,
        raw: Option<i64>,
        packed: bool,
    ) -> Envelope {
        let mut w = Vec::new();
        if let Some(r) = round {
            push_tag(&mut w, 1, 0);
            push_varint(&mut w, r);
        }
        if packed {
            let mut p = Vec::new();
            for &o in order {
                push_varint(&mut p, o);
            }
            push_tag(&mut w, 2, 2);
            push_varint(&mut w, p.len() as u64);
            w.extend(p);
        } else {
            for &o in order {
                push_tag(&mut w, 2, 0);
                push_varint(&mut w, o);
            }
        }
        if let Some(s) = speed {
            push_tag(&mut w, 3, 0);
            push_varint(&mut w, s);
        }
        if let Some(r) = raw {
            push_tag(&mut w, 4, 0);
            push_varint(&mut w, r as u64);
        }
        let mut body = Vec::new();
        push_tag(&mut body, 1, 0);
        push_varint(&mut body, 7); // seq
        push_tag(&mut body, 2, 0);
        push_varint(&mut body, 8); // ntype
        push_tag(&mut body, 8, 2);
        push_varint(&mut body, w.len() as u64);
        body.extend(w);
        Envelope {
            name: "BattleNotify".to_string(),
            body,
        }
    }

    fn decode_wind(env: &Envelope) -> (u64, Vec<u64>, u64, i64, Option<i32>) {
        let events = decode_envelope(env);
        assert_eq!(events.len(), 1);
        match &events[0] {
            GameEvent::WindSeed {
                round,
                order,
                speed,
                seed,
                wind10,
            } => (*round, order.clone(), *speed, *seed, *wind10),
            other => panic!("expected WindSeed, got {other:?}"),
        }
    }

    #[test]
    fn wind_seed_decodes_positive_wind() {
        let env = wind_seed_env(Some(2), &[74108, 0, 73884], Some(100), Some(449), false);
        let (round, order, speed, seed, wind10) = decode_wind(&env);
        assert_eq!(round, 2);
        assert_eq!(order, vec![74108, 0, 73884]);
        assert_eq!(speed, 100);
        assert_eq!(seed, 449);
        assert_eq!(wind10, Some(71));
    }

    #[test]
    fn wind_seed_decodes_negative_raw_sign_extended() {
        // -447 on the wire is the 10-byte sign-extended varint of u64.
        let env = wind_seed_env(Some(1), &[74108, 73884, 0], Some(100), Some(-447), false);
        let (_, _, _, seed, wind10) = decode_wind(&env);
        assert_eq!(seed, -447);
        assert_eq!(wind10, Some(-60));
    }

    #[test]
    fn wind_seed_packed_order_matches_unpacked() {
        let a = wind_seed_env(Some(1), &[56817, 74108, 0], Some(100), Some(-130217), true);
        let b = wind_seed_env(Some(1), &[56817, 74108, 0], Some(100), Some(-130217), false);
        assert_eq!(decode_wind(&a), decode_wind(&b));
        assert_eq!(decode_wind(&a).4, Some(-65));
    }

    #[test]
    fn wind_seed_order_reorder_is_invariant() {
        let a = wind_seed_env(Some(2), &[74108, 0, 73884], Some(100), Some(449), false);
        let b = wind_seed_env(Some(2), &[73884, 0, 74108], Some(100), Some(449), false);
        assert_eq!(decode_wind(&a).4, Some(71));
        assert_eq!(decode_wind(&b).4, Some(71));
    }

    #[test]
    fn wind_seed_six_player_order() {
        let env = wind_seed_env(
            Some(1),
            &[65432, 48235, 26895, 13375, 55164, 74108, 0],
            Some(100),
            Some(125170),
            false,
        );
        assert_eq!(decode_wind(&env).4, Some(84));
        let env = wind_seed_env(
            Some(1),
            &[26895, 13375, 55164, 74108, 0, 48235, 65432],
            Some(100),
            Some(-125160),
            false,
        );
        assert_eq!(decode_wind(&env).4, Some(-66));
    }

    #[test]
    fn wind_seed_raw_zero_is_valid_complete_action() {
        // raw=0 is a real field value, not "missing": order [1] decodes 10.0.
        let env = wind_seed_env(Some(1), &[1], Some(100), Some(0), false);
        let (_, _, _, seed, wind10) = decode_wind(&env);
        assert_eq!(seed, 0);
        assert_eq!(wind10, Some(100));
    }

    #[test]
    fn wind_seed_true_zero_wind() {
        let env = wind_seed_env(Some(1), &[74108, 73884, 0], Some(100), Some(389), false);
        assert_eq!(decode_wind(&env).4, Some(0));
    }

    #[test]
    fn wind_seed_decodes_above_old_25_5_limit() {
        let env = wind_seed_env(Some(1), &[74108, 73884, 0], Some(100), Some(169), false);
        assert_eq!(decode_wind(&env).4, Some(300));
    }

    #[test]
    fn wind_seed_missing_fields_decode_none() {
        // Each incomplete action still emits WindSeed but never guesses wind.
        for env in [
            wind_seed_env(None, &[74108, 73884, 0], Some(100), Some(389), false),
            wind_seed_env(Some(1), &[74108, 73884, 0], None, Some(389), false),
            wind_seed_env(Some(1), &[74108, 73884, 0], Some(100), None, false),
            wind_seed_env(Some(1), &[], Some(100), Some(389), false),
        ] {
            assert_eq!(decode_wind(&env).4, None);
        }
    }
}
