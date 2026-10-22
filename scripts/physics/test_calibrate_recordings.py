"""Offline guard regressions. No device or game access."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import calibrate_recordings as c


def v(value):
    value &= (1 << 64) - 1
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def integer(number, value):
    return v(number << 3) + v(value)


def blob(number, value):
    return v((number << 3) | 2) + v(len(value)) + value


def double(number, value):
    return v((number << 3) | 1) + struct.pack("<d", value)


def fixture(first_event=None):
    def row(name, body, ms, outgoing=False):
        return dict(source="1792202338110-1-0", pid=1, stream=0x80000001,
                    raw_offset=ms, game_ms=ms, c2s=outgoing,
                    framing="be16", name=name, body_hex=body.hex())

    def notice(kind, field=None, body=b"", ms=0):
        return row("BattleNotify", integer(1, ms) + integer(2, kind) +
                   (blob(field, body) if field else b""), ms)

    action = integer(1, 1) + integer(2, 1) + integer(3, 100) + integer(4, -106)
    request = integer(1, 3) + integer(2, 0) + integer(3, 65) + double(4, 60.0)
    points = [integer(1, 120 + i * 10) + integer(2, 80 - i) for i in range(15)]
    track = b"".join(blob(2, p) for p in points)
    if first_event is not None:
        track += blob(3, integer(1, first_event))
    branch = integer(1, 0) + blob(2, track) + double(3, 65.0)
    fire = integer(1, 1) + integer(2, 0) + blob(3, integer(1, 100) + integer(2, 100)) + blob(10, blob(1, branch))
    return [notice(0), notice(8, 8, action, 10), row("FireReq", request, 20, True),
            notice(15, 14, fire, 70)]


def extract(rows):
    with tempfile.TemporaryDirectory(prefix="tnt-shot-audit-") as tmp:
        path = Path(tmp) / "input.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return c.extract(path)


class WireTests(unittest.TestCase):
    def test_rejects_truncation_illegal_tags_and_overflow(self):
        for data in [b"\x00", b"\x0a\x03a", b"\x09\x00", b"\x0e", b"\x08" + b"\xff" * 10]:
            with self.assertRaises(ValueError):
                c.parse(data)

    def test_zero_is_present_and_negative_xy_is_signed(self):
        fields = c.parse(integer(1, 0) + integer(2, -3))
        self.assertEqual(c.xy(fields), (0, -3))
        self.assertIsNone(c.one(fields, 3, 0))

    def test_legacy_full_xor_and_updated_direct_wind(self):
        order = [7, 11, 13]
        raw = -106 ^ 4 ^ 70
        for uid in order:
            raw ^= uid
        fields = c.parse(integer(1, 4) + blob(2, b"".join(v(x) for x in order)) +
                         integer(3, 70) + integer(4, raw))
        self.assertEqual(c.wind_action(fields, "le32"), (-10.6, 4, 7))
        self.assertEqual(c.wind_action(fields, "be16")[0], c.signed32(raw) / 10)


class PairingTests(unittest.TestCase):
    def test_different_round_counters_still_require_a_matching_turn_and_stream(self):
        shots, counts = extract(fixture())
        self.assertEqual(counts["accepted"], 1)
        self.assertEqual(shots[0]["round"], 1)
        self.assertEqual(shots[0]["request_round"], 3)
        self.assertEqual(shots[0]["wind"], -10.6)
        self.assertEqual(c.scale_family(shots[0]), "ordinary_origin_scale")

    def test_never_pairs_across_streams_or_late_responses(self):
        for field, value in [("stream", 9), ("game_ms", 2000)]:
            rows = fixture()
            rows[-1][field] = value
            self.assertEqual(extract(rows)[0], [])

    def test_new_turn_discards_pending_request(self):
        rows = fixture()
        rows.insert(-1, copy.deepcopy(rows[1]))
        self.assertEqual(extract(rows)[0], [])

    def test_missing_force_does_not_become_zero(self):
        rows = fixture()
        rows[2]["body_hex"] = (integer(1, 3) + integer(2, 0) + integer(3, 65)).hex()
        shots, counts = extract(rows)
        self.assertEqual(shots, [])
        self.assertEqual(counts["request_missing_required"], 1)

    def test_overlapping_requests_are_ambiguous(self):
        rows = fixture()
        rows.insert(-1, copy.deepcopy(rows[2]))
        shots, counts = extract(rows)
        self.assertEqual(shots, [])
        self.assertEqual(counts["overlapping_requests"], 1)

    def test_only_pre_event_points_are_used(self):
        shots, _ = extract(fixture(first_event=12))
        self.assertEqual(len(shots[0]["points"]), 12)
        shots, counts = extract(fixture(first_event=10))
        self.assertEqual(shots, [])
        self.assertEqual(counts["short_pre_event_track"], 1)

    def test_player_flight_is_not_a_projectile(self):
        rows = fixture()
        rows[-1]["body_hex"] = (integer(1, 70) + integer(2, 14)).hex()
        self.assertEqual(extract(rows)[0], [])

    def test_duplicate_track_does_not_leak_across_session_groups(self):
        rows = fixture()
        duplicate = copy.deepcopy(rows)
        for row in duplicate:
            row["source"] = "1792414762078-2-0"
        shots, counts = extract(rows + duplicate)
        self.assertEqual(len(shots), 1)
        self.assertEqual(counts["duplicate_trajectory"], 1)

    def test_partitions_whole_sessions_by_date(self):
        for source, expected in [("1792202338110-1-0", "train"),
                                 ("1792284932816-1-0", "validation"),
                                 ("1792414762078-1-0", "test")]:
            self.assertEqual(c.partition(dict(source=source, framing="be16")), expected)


if __name__ == "__main__":
    unittest.main()
