"""Offline regression tests for the optional, read-only TNT angle stream."""
import math
import contextlib
import io
import re
import struct
import subprocess
import threading
import unittest
from unittest import mock

import probe_live_angle_readonly as angle
import stream_live_angle_readonly as stream


class TestAngleSamples(unittest.TestCase):
    def setUp(self):
        self.cache = {
            "pid": 3056, "panel": 0x1000, "sprite": 0x2000,
            "role": 0x3000, "angle": 0x4000,
            "rotation_private": 0x5000, "rotation_transform": 0x6000,
            "flip_private": 0x7000,
            "panel_expected_vptr": 0x12345678,
            "self_vptr": 0x22334455, "uid": 74108,
        }
        self.memory = {
            0x1000: struct.pack("<Q", 0x12345678),
            0x2000: struct.pack("<Q", 0x22334455),
            0x21D8: struct.pack("<Q", 0x4000),
            0x3068: struct.pack("<i", 74108),
            0x40A0: struct.pack("<iii", 65, 77, 49),
            0x5090: struct.pack("<Q", 0x6000),
            0x70A0: struct.pack("<d", -0.996),
            0x6060: struct.pack("<d", 5.0),
        }
        memory = self.memory

        class FakeMemory:
            def __init__(self, pid):
                if pid != 3056:
                    raise AssertionError("wrong pid")

            def root_cmd(self, command, timeout=8):
                pairs = re.findall(r"skip=(\d+) count=(\d+)", command)
                if len(pairs) != 8:
                    raise AssertionError(f"expected 8 ranges, got {len(pairs)}")
                blob = b"".join(memory[int(address)] for address, _ in pairs)
                if len(blob) != sum(int(n) for _, n in pairs):
                    raise AssertionError("bad field size")
                return " ".join(f"{b:02x}" for b in blob)

        patcher = mock.patch.object(angle, "Memory", FakeMemory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fast_read_and_angle_folding(self):
        result = angle.quick_sample(self.cache)
        self.assertEqual(result["angle_value"], 65)
        self.assertEqual(result["allowed_range"], [49, 77])
        self.assertEqual(result["provisional_fire_angle"], 110)
        self.assertIn("TNTANGLE 3056 74108 110 65", stream.encode_line(self.cache, result))

    def test_reversed_direction(self):
        self.memory[0x70A0] = struct.pack("<d", 0.996)
        result = angle.quick_sample(self.cache)
        self.assertEqual(result["provisional_fire_angle"], 60)

    def test_reject_wrong_identity_or_scene(self):
        self.memory[0x3068] = struct.pack("<i", 74109)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            angle.quick_sample(self.cache)
        self.memory[0x3068] = struct.pack("<i", 74108)
        self.memory[0x1000] = struct.pack("<Q", 0)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            angle.quick_sample(self.cache)

    def test_reject_invalid_transform_or_bounds(self):
        self.memory[0x5090] = struct.pack("<Q", 0x9000)
        with self.assertRaisesRegex(RuntimeError, "transform"):
            angle.quick_sample(self.cache)
        self.memory[0x5090] = struct.pack("<Q", 0x6000)
        self.memory[0x40A0] = struct.pack("<iii", 999, 77, 49)
        with self.assertRaisesRegex(ValueError, "out of range"):
            angle.quick_sample(self.cache)


class TestStreamProtocol(unittest.TestCase):
    def test_slow_background_verification_never_blocks_sampling_or_queues_duplicates(self):
        verifier=stream.BackgroundVerifier()
        release=threading.Event()
        started=threading.Event()
        def work():
            started.set()
            release.wait(timeout=2)
            return 'verified'
        try:
            self.assertTrue(verifier.start(work))
            self.assertTrue(started.wait(timeout=1))
            self.assertIsNone(verifier.take())
            self.assertFalse(verifier.start(lambda:'duplicate'))
        finally:
            release.set()
            verifier.thread.join(timeout=1)
        self.assertEqual(verifier.take(),dict(value='verified'))

    def test_pending_full_check_does_not_extend_twenty_second_cache_trust(self):
        cache=dict(pid=1,uid=7)
        completed=dict(value=dict(cache=cache,discovered=dict(status='coordinates_found'),
                                 verified_at=0.0,duration_ms=1500,recovery_attempted=False))
        verifier=mock.Mock()
        verifier.take.side_effect=[completed,None,None,None]
        clock=[0.0]
        source=dict(status='angle_value_found',provisional_fire_angle=65,angle_value=65,
                    qt_rotation_degrees=0.0,flipped_x_cached=False)
        output=io.StringIO()
        with mock.patch.object(stream,'BackgroundVerifier',return_value=verifier), \
             mock.patch.object(stream.sys,'argv',['stream','--max-samples','4']), \
             mock.patch.object(stream.os,'getppid',return_value=123), \
             mock.patch.object(stream.time,'monotonic',side_effect=lambda:clock[0]), \
             mock.patch.object(stream.time,'sleep',side_effect=lambda _:clock.__setitem__(0,clock[0]+7)), \
             mock.patch.object(stream,'sample',return_value=source) as full, \
             mock.patch.object(stream,'quick_sample',return_value=source) as quick, \
             contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(stream.main(),0)
        lines=output.getvalue().splitlines()
        self.assertEqual(len([l for l in lines if l.startswith('TNTANGLE ')]),3)
        self.assertEqual(lines[-1],'TNTANGLE-')
        self.assertEqual(full.call_count,1)
        self.assertEqual(quick.call_count,2)

    def test_full_check_is_deferred_during_long_hold(self):
        cache = {"verified": True}
        check = stream.should_full_check
        # Startup always verifies the full scene.
        self.assertTrue(check(None, 10.0, 13.0, 10.0, 10.0))
        # Still before normal interval: no complete scan.
        self.assertFalse(check(cache, 12.0, 13.0, 10.0, 10.0))
        # The angle moved within the last 0.8s: keep the fast path.
        self.assertFalse(check(cache, 13.1, 13.0, 12.8, 10.0))
        self.assertFalse(check(cache, 23.0, 13.0, 22.9, 10.0))
        # Player stopped moving: full identity/ABI verification resumes.
        self.assertTrue(check(cache, 14.0, 13.0, 13.1, 10.0))
        # Even with uninterrupted movement, hard deadline applies.
        self.assertTrue(check(cache, 30.1, 13.0, 30.0, 10.0))

    def test_transient_errors_retry_but_identity_mismatch_does_not(self):
        self.assertTrue(stream.transient_sampling_error(
            RuntimeError("quick angle read incomplete 17 bytes")))
        self.assertTrue(stream.transient_sampling_error(
            RuntimeError("root command failed: device busy")))
        self.assertTrue(stream.transient_sampling_error(
            subprocess.TimeoutExpired(cmd="adb", timeout=8)))
        self.assertFalse(stream.transient_sampling_error(
            RuntimeError("angle scene identity/transform changed")))
        self.assertFalse(stream.transient_sampling_error(
            ValueError("angle triplet is out of range")))

    def test_invalid_angle_is_never_published(self):
        source = {
            "status": "angle_value_found",
            "provisional_fire_angle": 110,
            "angle_value": 65,
            "qt_rotation_degrees": 5.0,
            "flipped_x_cached": True,
        }
        self.assertEqual(stream.encode_line({"pid": 3056, "uid": 74108}, source),
                         "TNTANGLE 3056 74108 110 65 5.0 1")
        self.assertEqual(stream.encode_line({"pid": 3056, "uid": 74108},
                                            {**source, "qt_rotation_degrees": math.nan}),
                         "TNTANGLE-")
        self.assertEqual(stream.encode_line({"pid": 3056, "uid": 74108},
                                            {**source, "provisional_fire_angle": -2}),
                         "TNTANGLE-")

    def test_adb_reconnect_only_when_transport_is_missing(self):
        available = "List of devices attached\n127.0.0.1:5555 device\n"
        missing = "List of devices attached\n"
        with mock.patch.object(stream.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, available, "")
            self.assertFalse(stream.recover_adb_transport())
            self.assertEqual(run.call_count, 1)
        with mock.patch.object(stream.subprocess, "run") as run:
            run.side_effect = [
                subprocess.CompletedProcess([], 0, missing, ""),
                subprocess.CompletedProcess([], 0, "connected", ""),
            ]
            self.assertTrue(stream.recover_adb_transport())
            self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
