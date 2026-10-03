//! Live capture plumbing: WebSocket stream decoder, zlib message layer,
//! pcap/TCP reassembly and the in-process sniffer tunnel record format.
//!
//! Two feeders produce the same `Vec<u8>` message payloads:
//!   * pcap mode   — `PcapReader` + `FlowReasm` rebuild the TCP streams
//!   * tunnel mode — `libtntsniff.so` inside the game process sends records
//!     over `adb reverse` (`TunnelParser`); each record already arrives in
//!     order, so only the WS layer runs on top.

use crate::proto::{decode_envelope, parse_envelope, GameEvent};
use flate2::read::ZlibDecoder;
use std::collections::BTreeMap;
use std::io::Read;

/// One fully-reassembled WebSocket message payload (data opcodes only).
#[derive(Debug)]
pub struct WsMessage {
    pub opcode: u8,
    pub payload: Vec<u8>,
}

enum Resync {
    Found(usize),
    Wait,
    None,
}

/// Incremental WebSocket frame parser for one direction of one connection.
/// Handles the optional HTTP upgrade prefix, masking and continuation frames.
#[derive(Default)]
pub struct WsDecoder {
    buf: Vec<u8>,
    handshake_done: bool,
    frag_opcode: Option<u8>,
    frag: Vec<u8>,
    pub dead: bool,
    pub dropped_prefix: usize,
}

impl WsDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw stream bytes; returns every complete data message seen.
    pub fn feed(&mut self, data: &[u8]) -> Vec<WsMessage> {
        if self.dead {
            return Vec::new();
        }
        self.buf.extend_from_slice(data);
        let mut out = Vec::new();

        if !self.handshake_done {
            // HTTP upgrade lines ("GET ..." / "HTTP/1.1 101") precede frames.
            if self.buf.starts_with(b"GET ") || self.buf.starts_with(b"HTTP") {
                match find_subslice(&self.buf, b"\r\n\r\n") {
                    Some(end) => {
                        self.dropped_prefix = end + 4;
                        self.buf.drain(..end + 4);
                        self.handshake_done = true;
                    }
                    None => {
                        if self.buf.len() > 8192 {
                            self.dead = true;
                        }
                        return out;
                    }
                }
            } else {
                // Capture started mid-connection or there is no handshake.
                self.handshake_done = true;
            }
        }

        loop {
            let Some(result) = self.parse_frame() else {
                break;
            };
            let Ok(((fin, opcode, payload), consumed)) = result else {
                // 中途接入/丢包导致错位:扫描下一个可疑帧头重新对齐,
                // 签名是 [0x8x 帧头][可选mask][u32BE 原始长度][0x78 zlib]。
                match self.resync() {
                    Resync::Found(off) => {
                        self.buf.drain(..off);
                        continue;
                    }
                    // 帧头像但负载还没收全:保留缓冲等下一批字节
                    Resync::Wait => break,
                    // 完全没有候选:丢空缓冲,下一批数据再试
                    Resync::None => {
                        self.buf.clear();
                        break;
                    }
                }
            };
            self.buf.drain(..consumed);
            match opcode {
                0 | 1 | 2 => {
                    if opcode != 0 {
                        self.frag_opcode = Some(opcode);
                        self.frag.clear();
                    }
                    self.frag.extend_from_slice(&payload);
                    if fin {
                        let op = self.frag_opcode.take().unwrap_or(2);
                        out.push(WsMessage {
                            opcode: op,
                            payload: std::mem::take(&mut self.frag),
                        });
                    }
                }
                8 => {
                    // close frame: keep parsing whatever else arrived
                }
                9 | 10 => {} // ping/pong
                _ => {
                    self.dead = true;
                    break;
                }
            }
        }
        if self.buf.len() > 16 * 1024 * 1024 {
            self.dead = true;
        }
        out
    }

    /// Look for the next plausible binary-data frame header in the buffer.
    /// Every game message payload starts with `[u32 BE declared_len][zlib]`,
    /// so byte 4 of the payload must be the zlib magic 0x78 (or the first
    /// bytes unmasked to it). This is a strong resync signature.
    fn resync(&self) -> Resync {
        let b = &self.buf;
        let mut i = 0usize;
        while i + 3 <= b.len() {
            // binary frame, fin bit set, no rsv bits
            if b[i] & 0x7f != 0x02 {
                i += 1;
                continue;
            }
            let masked = b[i + 1] & 0x80 != 0;
            let mut len = (b[i + 1] & 0x7f) as usize;
            let mut pos = i + 2;
            if len == 126 {
                if b.len() < pos + 2 {
                    return Resync::Wait;
                }
                len = u16::from_be_bytes(b[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;
            } else if len == 127 {
                if b.len() < pos + 8 {
                    return Resync::Wait;
                }
                len = u64::from_be_bytes(b[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8;
            }
            if len < 6 || len > 1 << 20 {
                i += 1;
                continue;
            }
            let mask = if masked {
                if b.len() < pos + 4 {
                    return Resync::Wait;
                }
                let k = [b[pos], b[pos + 1], b[pos + 2], b[pos + 3]];
                pos += 4;
                Some(k)
            } else {
                None
            };
            // need at least 5 payload bytes to check the zlib signature
            if b.len() < pos + 5 {
                return Resync::Wait;
            }
            let mut head = [0u8; 5];
            for j in 0..5 {
                head[j] = b[pos + j];
            }
            if let Some(k) = mask {
                for j in 0..5 {
                    head[j] ^= k[j & 3];
                }
            }
            if head[4] == 0x78 {
                return Resync::Found(i);
            }
            i += 1;
        }
        Resync::None
    }

    /// None = incomplete; Some(Err) = corrupt stream; Some(Ok) = frame.
    fn parse_frame(&self) -> Option<Result<((bool, u8, Vec<u8>), usize), ()>> {
        let b = &self.buf;
        if b.len() < 2 {
            return None;
        }
        let fin = b[0] & 0x80 != 0;
        let opcode = b[0] & 0x0f;
        if b[0] & 0x70 != 0 || opcode > 0x0a {
            return Some(Err(()));
        }
        let masked = b[1] & 0x80 != 0;
        let mut len = (b[1] & 0x7f) as u64;
        let mut pos = 2;
        if len == 126 {
            if b.len() < pos + 2 {
                return None;
            }
            len = u16::from_be_bytes(b[pos..pos + 2].try_into().unwrap()) as u64;
            pos += 2;
        } else if len == 127 {
            if b.len() < pos + 8 {
                return None;
            }
            len = u64::from_be_bytes(b[pos..pos + 8].try_into().unwrap());
            pos += 8;
        }
        if len > 64 * 1024 * 1024 {
            return Some(Err(()));
        }
        let mask_key = if masked {
            if b.len() < pos + 4 {
                return None;
            }
            let k = [b[pos], b[pos + 1], b[pos + 2], b[pos + 3]];
            pos += 4;
            Some(k)
        } else {
            None
        };
        let len = len as usize;
        if b.len() < pos + len {
            return None;
        }
        let mut payload = b[pos..pos + len].to_vec();
        if let Some(k) = mask_key {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= k[i & 3];
            }
        }
        Some(Ok(((fin, opcode, payload), pos + len)))
    }
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// WS message -> zlib inflate -> envelope -> game events.
/// Binary payload layout: `[u32 BE declared_len][zlib stream]`.
pub fn decode_ws_message(msg: &WsMessage) -> Vec<GameEvent> {
    if msg.opcode != 2 || msg.payload.len() < 6 {
        return Vec::new();
    }
    let declared = u32::from_be_bytes(msg.payload[0..4].try_into().unwrap()) as usize;
    let mut z = ZlibDecoder::new(&msg.payload[4..]);
    let mut plain = Vec::with_capacity(declared.min(1 << 20));
    if z.read_to_end(&mut plain).is_err() {
        return Vec::new();
    }
    let Some(env) = parse_envelope(&plain) else {
        return Vec::new();
    };
    decode_envelope(&env)
}

// ---------------- pcap + TCP reassembly ----------------

#[derive(Default)]
pub struct FlowReasm {
    next_seq: Option<u32>,
    pending: BTreeMap<u32, Vec<u8>>,
    pub ws: WsDecoder,
}

impl FlowReasm {
    /// Feed one TCP payload; returns completed WS messages in order.
    pub fn push(&mut self, seq: u32, data: &[u8]) -> Vec<WsMessage> {
        if data.is_empty() {
            return Vec::new();
        }
        match self.next_seq {
            None => {
                self.next_seq = Some(seq.wrapping_add(data.len() as u32));
                self.ws.feed(data)
            }
            Some(next) => {
                if seq == next {
                    let mut out = self.ws.feed(data);
                    self.next_seq = Some(next.wrapping_add(data.len() as u32));
                    self.flush_pending(&mut out);
                    out
                } else if seq.wrapping_sub(next) as i32 > 0 {
                    // future segment: stash
                    self.pending.insert(seq, data.to_vec());
                    Vec::new()
                } else {
                    // retransmit/overlap
                    let overlap = next.wrapping_sub(seq) as usize;
                    if overlap >= data.len() {
                        Vec::new()
                    } else {
                        let fresh = &data[overlap..];
                        let mut out = self.ws.feed(fresh);
                        self.next_seq = Some(next.wrapping_add(fresh.len() as u32));
                        self.flush_pending(&mut out);
                        out
                    }
                }
            }
        }
    }

    fn flush_pending(&mut self, out: &mut Vec<WsMessage>) {
        loop {
            let Some(next) = self.next_seq else { return };
            let Some((&seq, _)) = self.pending.range(..=next).next_back() else {
                return;
            };
            if seq > next {
                return;
            }
            let data = self.pending.remove(&seq).unwrap();
            let overlap = next.wrapping_sub(seq) as usize;
            if overlap < data.len() {
                let fresh = &data[overlap..];
                out.extend(self.ws.feed(fresh));
                self.next_seq = Some(next.wrapping_add(fresh.len() as u32));
            }
        }
    }
}

/// Streaming pcap packet source (classic libpcap format).
pub struct PcapReader<R: Read> {
    r: R,
    endian_le: bool,
    link_type: u32,
    ts_div: u64,
}

pub struct PcapPacket {
    pub ts: f64,
    pub src: [u8; 4],
    pub dst: [u8; 4],
    pub sport: u16,
    pub dport: u16,
    pub seq: u32,
    pub payload: Vec<u8>,
}

impl<R: Read> PcapReader<R> {
    pub fn new(mut r: R) -> std::io::Result<Self> {
        let mut hdr = [0u8; 24];
        r.read_exact(&mut hdr)?;
        let (le, div) = match &hdr[0..4] {
            [0xd4, 0xc3, 0xb2, 0xa1] => (true, 1_000_000u64),
            [0xa1, 0xb2, 0xc3, 0xd4] => (false, 1_000_000u64),
            [0x4d, 0x3c, 0xb2, 0xa1] => (true, 1_000_000_000u64),
            [0xa1, 0xb2, 0x3c, 0x4d] => (false, 1_000_000_000u64),
            m => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("bad pcap magic {:02x?}", m),
                ))
            }
        };
        let u32v = |b: &[u8]| -> u32 {
            if le {
                u32::from_le_bytes(b.try_into().unwrap())
            } else {
                u32::from_be_bytes(b.try_into().unwrap())
            }
        };
        let link_type = u32v(&hdr[20..24]);
        Ok(Self {
            r,
            endian_le: le,
            link_type,
            ts_div: div,
        })
    }

    /// Next TCP/IPv4 packet with payload, or None at EOF.
    pub fn next_packet(&mut self) -> Option<PcapPacket> {
        let mut ph = [0u8; 16];
        if self.r.read_exact(&mut ph).is_err() {
            return None;
        }
        let (sec, frac, caplen) = if self.endian_le {
            (
                u32::from_le_bytes(ph[0..4].try_into().unwrap()),
                u32::from_le_bytes(ph[4..8].try_into().unwrap()),
                u32::from_le_bytes(ph[8..12].try_into().unwrap()) as usize,
            )
        } else {
            (
                u32::from_be_bytes(ph[0..4].try_into().unwrap()),
                u32::from_be_bytes(ph[4..8].try_into().unwrap()),
                u32::from_be_bytes(ph[8..12].try_into().unwrap()) as usize,
            )
        };
        let mut pkt = vec![0u8; caplen];
        if self.r.read_exact(&mut pkt).is_err() {
            return None;
        }
        let ts = sec as f64 + frac as f64 / self.ts_div as f64;

        let ip_off = match self.link_type {
            0 => {
                // BSD loopback: 4-byte family header, host endianness
                if pkt.len() < 4 || u32::from_ne_bytes(pkt[0..4].try_into().unwrap()) != 2 {
                    return self.next_packet();
                }
                4
            }
            108 => {
                // PKTAP-ish hdr used by earlier captures: 4B big-endian family
                if pkt.len() < 4 || u32::from_be_bytes(pkt[0..4].try_into().unwrap()) != 2 {
                    return self.next_packet();
                }
                4
            }
            101 => 0, // raw IP
            1 => 14,  // ethernet
            _ => 0,
        };
        if pkt.len() < ip_off + 20 {
            return self.next_packet();
        }
        let ihl = ((pkt[ip_off] & 0x0f) * 4) as usize;
        if pkt[ip_off] >> 4 != 4 || pkt[ip_off + 9] != 6 {
            return self.next_packet();
        }
        let total = u16::from_be_bytes(pkt[ip_off + 2..ip_off + 4].try_into().unwrap()) as usize;
        let src: [u8; 4] = pkt[ip_off + 12..ip_off + 16].try_into().unwrap();
        let dst: [u8; 4] = pkt[ip_off + 16..ip_off + 20].try_into().unwrap();
        let t = ip_off + ihl;
        if pkt.len() < t + 20 {
            return self.next_packet();
        }
        let sport = u16::from_be_bytes(pkt[t..t + 2].try_into().unwrap());
        let dport = u16::from_be_bytes(pkt[t + 2..t + 4].try_into().unwrap());
        let seq = u32::from_be_bytes(pkt[t + 4..t + 8].try_into().unwrap());
        let thl = ((pkt[t + 12] >> 4) * 4) as usize;
        let pstart = t + thl;
        let pend = (ip_off + total).min(pkt.len());
        if pstart >= pend {
            return self.next_packet();
        }
        Some(PcapPacket {
            ts,
            src,
            dst,
            sport,
            dport,
            seq,
            payload: pkt[pstart..pend].to_vec(),
        })
    }
}

// ---------------- tunnel session tracking ----------------

/// Structured events emitted by [`Tracker`] while consuming tunnel records.
#[derive(Debug)]
pub enum TunnelEvent {
    Connect { pid: u32, fd: u32, peer: String },
    Close { pid: u32, fd: u32, peer: String },
    /// One complete WS message; `c2s` = client -> server.
    Msg { pid: u32, fd: u32, c2s: bool, msg: WsMessage },
    /// WS stream broke unrecoverably on this fd (emitted once per fd).
    Dead { pid: u32, fd: u32, peer: String },
    Hello { pid: u32, text: String },
    Rand { pid: u32, text: String },
    Log { pid: u32, text: String },
}

struct ConnState {
    peer: String,
    c2s: WsDecoder,
    s2c: WsDecoder,
    dead_noted: bool,
}

/// Tracks every (pid, fd) connection announced by the sniffer and runs the
/// per-direction WS decoders. Shared by `net_live` and the HUD listener.
#[derive(Default)]
pub struct Tracker {
    conns: std::collections::HashMap<(u32, u32), ConnState>,
}

impl Tracker {
    pub fn on_record(&mut self, rec: &tunnel::Record) -> Vec<TunnelEvent> {
        let key = (rec.pid, rec.fd);
        match rec.kind {
            tunnel::KIND_CONNECT => {
                let peer = String::from_utf8_lossy(&rec.payload).to_string();
                self.conns.insert(
                    key,
                    ConnState {
                        peer: peer.clone(),
                        c2s: WsDecoder::new(),
                        s2c: WsDecoder::new(),
                        dead_noted: false,
                    },
                );
                vec![TunnelEvent::Connect {
                    pid: rec.pid,
                    fd: rec.fd,
                    peer,
                }]
            }
            tunnel::KIND_TX | tunnel::KIND_RX => {
                let mut out = Vec::new();
                // 中途接入的连接不会有 CONNECT 记录(connect 发生在隧道建立之前),
                // 就地补建状态——WS 解码器支持从流中间起步。
                if !self.conns.contains_key(&key) {
                    self.conns.insert(
                        key,
                        ConnState {
                            peer: "?".to_string(),
                            c2s: WsDecoder::new(),
                            s2c: WsDecoder::new(),
                            dead_noted: false,
                        },
                    );
                    out.push(TunnelEvent::Connect {
                        pid: rec.pid,
                        fd: rec.fd,
                        peer: "?(mid-stream)".to_string(),
                    });
                }
                let c = self.conns.get_mut(&key).unwrap();
                let c2s = rec.kind == tunnel::KIND_TX;
                let dec = if c2s { &mut c.c2s } else { &mut c.s2c };
                if dec.dead {
                    return out;
                }
                for m in dec.feed(&rec.payload) {
                    out.push(TunnelEvent::Msg {
                        pid: rec.pid,
                        fd: rec.fd,
                        c2s,
                        msg: m,
                    });
                }
                if dec.dead && !c.dead_noted {
                    c.dead_noted = true;
                    out.push(TunnelEvent::Dead {
                        pid: rec.pid,
                        fd: rec.fd,
                        peer: c.peer.clone(),
                    });
                }
                out
            }
            tunnel::KIND_CLOSE => {
                if let Some(c) = self.conns.remove(&key) {
                    vec![TunnelEvent::Close {
                        pid: rec.pid,
                        fd: rec.fd,
                        peer: c.peer,
                    }]
                } else {
                    Vec::new()
                }
            }
            tunnel::KIND_HELLO => vec![TunnelEvent::Hello {
                pid: rec.pid,
                text: String::from_utf8_lossy(&rec.payload).to_string(),
            }],
            tunnel::KIND_RAND => vec![TunnelEvent::Rand {
                pid: rec.pid,
                text: String::from_utf8_lossy(&rec.payload).to_string(),
            }],
            tunnel::KIND_LOG => vec![TunnelEvent::Log {
                pid: rec.pid,
                text: String::from_utf8_lossy(&rec.payload).to_string(),
            }],
            _ => Vec::new(),
        }
    }
}

// ---------------- sniffer tunnel protocol ----------------

pub mod tunnel {
    /// record := u8 kind | u32 pid | u32 fd | u64 ms | u32 len | payload
    /// header is 21 bytes; len sits at offset 17.
    pub const KIND_CONNECT: u8 = 1;
    pub const KIND_TX: u8 = 2;
    pub const KIND_RX: u8 = 3;
    pub const KIND_CLOSE: u8 = 4;
    pub const KIND_HELLO: u8 = 5;
    pub const KIND_RAND: u8 = 6;
    pub const KIND_LOG: u8 = 7;
    pub const HDR_LEN: usize = 1 + 4 + 4 + 8 + 4; // = 21

    pub struct Record {
        pub kind: u8,
        pub pid: u32,
        pub fd: u32,
        pub ms: u64,
        pub payload: Vec<u8>,
    }

    #[derive(Default)]
    pub struct Parser {
        buf: Vec<u8>,
    }

    impl Parser {
        pub fn feed(&mut self, data: &[u8]) -> Vec<Record> {
            self.buf.extend_from_slice(data);
            let mut out = Vec::new();
            loop {
                if self.buf.len() < HDR_LEN {
                    break;
                }
                let len = u32::from_le_bytes(self.buf[17..21].try_into().unwrap()) as usize;
                if len > 16 * 1024 * 1024 {
                    // desync: drop one byte and retry
                    self.buf.remove(0);
                    continue;
                }
                if self.buf.len() < HDR_LEN + len {
                    break;
                }
                let rec = Record {
                    kind: self.buf[0],
                    pid: u32::from_le_bytes(self.buf[1..5].try_into().unwrap()),
                    fd: u32::from_le_bytes(self.buf[5..9].try_into().unwrap()),
                    ms: u64::from_le_bytes(self.buf[9..17].try_into().unwrap()),
                    payload: self.buf[HDR_LEN..HDR_LEN + len].to_vec(),
                };
                self.buf.drain(..HDR_LEN + len);
                out.push(rec);
            }
            out
        }
    }
}
