"""Pure offline checks of the verified Qt coordinate reconstruction."""
import math
import json
import struct
import unittest
from unittest import mock

import probe_fire_origin_readonly as origin
import stream_live_angle_readonly as stream


def data(base=(1, 0, 0, 1, 0, 0, 0, 0, 1), scale=1, rotation=0,
         pivot=(0, 0), only=False, list_address=0xA000):
    out = bytearray(0x88)
    struct.pack_into("<9d", out, 0, *base)
    struct.pack_into("<4d", out, 0x58, scale, rotation, *pivot)
    struct.pack_into("<Q", out, 0x78, list_address)
    out[0x80] = int(only)
    return bytes(out)


class MatrixTests(unittest.TestCase):
    def test_mirror_ground_rotation_and_world_translation(self):
        mirror, _ = origin.transform_data(data(base=(-1, 0, 0, 1, 0, 0, 0, 0, 1)))
        ground, _ = origin.transform_data(data(rotation=90))
        result = origin.multiply(origin.translate(1000, 500),
                                 origin.multiply(ground, origin.multiply(mirror, origin.translate(20, -20))))
        self.assertAlmostEqual(result[0][2], 1020)
        self.assertAlmostEqual(result[1][2], 480)

    def test_pivot_scale_rotation_and_base_transform_order(self):
        result, _ = origin.transform_data(data(base=(1, 0, 0, 1, 5, 7, 0, 0, 1),
                                              scale=2, rotation=90, pivot=(3, 4)))
        self.assertAlmostEqual(result[0][2], 16)
        self.assertAlmostEqual(result[1][2], 5)

    def test_uses_actual_qtransform_memory_order(self):
        matrix, _ = origin.transform_data(data(base=(2, 3, 4, 5, 6, 7, 0, 0, 1), only=True))
        self.assertEqual(matrix, ((2, 4, 6), (3, 5, 7), (0, 0, 1)))

    def test_only_transform_flag_follows_qt_branch(self):
        matrix, _ = origin.transform_data(data(scale=2, rotation=90, only=True))
        self.assertEqual(matrix, origin.identity())

    def test_rejects_unsupported_or_nonfinite_layouts(self):
        examples = [data(base=(1, 0, 0, 1, 0, 0, .1, 0, 1)),
                    data(scale=0), data(rotation=math.nan), b"short"]
        invalid = bytearray(data())
        invalid[0x80] = 2
        examples.append(bytes(invalid))
        for raw in examples:
            with self.assertRaises(ValueError):
                origin.transform_data(raw)


class SnapshotTests(unittest.TestCase):
    def fixture(self, nonempty=False):
        u64 = lambda v: struct.pack("<Q", v)
        rows = [dict(item=0x1000, private=0x2000, parent=0x3000, transform=0, list=None),
                dict(item=0x3000, private=0x6000, parent=0x8010, transform=0x9000, list=0xA000),
                dict(item=0x8010, private=0xB000, parent=0xC000, transform=0, list=None)]
        cache = dict(pid=123, panel=0xD000, panel_expected_vptr=9,
                     sprite=0x8000, self_vptr=10, role=0xE000, uid=7,
                     angle=0xFF0, angle_vptr=11, chain=rows)
        parts = [u64(9), u64(10), u64(0xE000), struct.pack("<i", 7), u64(0xFF0), u64(11)]
        for row, xy in zip(rows, [(20, -20), (0, 0), (100, 300)]):
            private = bytearray(0x40)
            struct.pack_into("<dd", private, 0, *xy)
            struct.pack_into("<Q", private, 0x28, row["parent"])
            struct.pack_into("<Q", private, 0x38, row["transform"])
            parts.extend([u64(row["private"]), bytes(private)])
            if row["transform"]:
                parts.extend([data(base=(-1, 0, 0, 1, 0, 0, 0, 0, 1)),
                              struct.pack("<ii", 0, 1 if nonempty else 0)])
        return cache, parts

    def test_snapshot_produces_candidate_without_writing_or_invoking_qt(self):
        cache, parts = self.fixture()
        with mock.patch.object(origin, "batch", return_value=parts):
            result = origin.sample(cache)
        self.assertEqual(result["fire_pos_rounded"], [80, 280])
        self.assertEqual(result["player_graphics_xy"], [100, 300])
        self.assertEqual(result["linear_scale_xy"], [1, 1])
        self.assertTrue(result["research_only"])

    def test_expanded_snapshot_uses_real_qt_transform(self):
        cache, parts = self.fixture()
        for i, raw in enumerate(parts):
            if len(raw)==0x88:
                parts[i]=data(base=(-1,0,0,1,0,0,0,0,1),scale=2)
        with mock.patch.object(origin,'batch',return_value=parts):
            result=origin.sample(cache)
        self.assertEqual(result['fire_pos_rounded'],[60,260])
        self.assertEqual(result['linear_scale_xy'],[2,2])
        self.assertEqual(result['linear_determinant'],-4)

    def test_uid_mismatch_never_publishes_a_position(self):
        cache, parts = self.fixture()
        parts[3] = struct.pack("<i", 8)
        with mock.patch.object(origin, "batch", return_value=parts):
            with self.assertRaisesRegex(RuntimeError, "identity"):
                origin.sample(cache)

    def test_changed_parent_never_publishes_a_position(self):
        cache, parts = self.fixture()
        changed = bytearray(parts[7])
        struct.pack_into("<Q", changed, 0x28, 0xBAD0)
        parts[7] = bytes(changed)
        with mock.patch.object(origin, "batch", return_value=parts):
            with self.assertRaisesRegex(RuntimeError, "parent"):
                origin.sample(cache)

    def test_nonempty_graphics_transform_list_is_rejected(self):
        cache, parts = self.fixture(nonempty=True)
        with mock.patch.object(origin, "batch", return_value=parts):
            with self.assertRaisesRegex(RuntimeError, "graphics-transform"):
                origin.sample(cache)

    def test_aim_guard_uses_control_from_the_same_snapshot(self):
        cache, parts = self.fixture()
        cache['rotatable'] = 0x2FF0
        parts.append(struct.pack('<iii', 65, 90, 0))
        with mock.patch.object(origin, 'batch', return_value=parts):
            result = origin.sample(cache, with_aim=True)
        self.assertEqual(result['fire_angle'], 115)
        self.assertTrue(result['heading_left'])
        decoded = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(len(decoded['snapshot_ranges']), len(parts))
        self.assertEqual(bytes.fromhex(decoded['snapshot_ranges'][-1]['hex']), parts[-1])
        self.assertEqual(decoded['nodes'][1]['local_transform'][0][0], -1)


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.cache = dict(pid=123, uid=7, sprite=0x1000, panel=0x2000)
        self.reader = stream.OriginStream()
        self.reader.cache = self.cache
        self.row = dict(status='fire_origin_candidate', pid=123, uid=7,
                        fire_pos_rounded=[80, 280], player_graphics_xy=[100, 300],
                        linear_scale_xy=[1, 1], linear_determinant=-1,
                        fire_angle=115, heading_left=True, snapshot_digest='same',
                        nodes=[dict(item=1,parent=2,pos=[20,-20],local_transform=origin.identity())])
        self.clock = 100.0
        for name, target, value in [('sample', origin, lambda *a, **kw: dict(self.row)),
                                    ('monotonic', stream.time, lambda: self.clock),
                                    ('time_ns', stream.time, lambda: int(self.clock * 1e9))]:
            patcher = mock.patch.object(target, name, side_effect=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def poll(self, **kwargs):
        self.clock += .4
        return self.reader.poll(self.cache, dict(provisional_fire_angle=115), **kwargs)

    def test_two_stable_samples_required_and_period_is_bounded(self):
        self.assertEqual(self.poll(), 'TNTORIGIN-')
        self.assertFalse(self.reader.observation['source_eligible'])
        self.assertIn('sample_start_unix_ms', self.reader.observation)
        self.assertTrue(self.poll().startswith('TNTORIGIN 123 7 '))
        self.assertTrue(self.reader.observation['source_eligible'])
        self.assertIsNone(self.reader.poll(self.cache, dict(provisional_fire_angle=115)))
        self.assertIsNone(self.reader.observation)

    def test_changed_pose_and_long_gap_clear_correction(self):
        self.poll()
        self.row['nodes'][0]['pos'] = [20,-21]
        self.assertEqual(self.poll(), 'TNTORIGIN-')
        self.clock += 2
        self.assertEqual(self.poll(), 'TNTORIGIN-')

    def test_control_or_metadata_changes_do_not_invalidate_unchanged_geometry(self):
        self.poll()
        self.poll()
        self.row.update(snapshot_digest='different raw control bytes',control_value=64,fire_angle=116)
        self.clock += .4
        line=self.reader.poll(self.cache,dict(provisional_fire_angle=116))
        self.assertTrue(line.startswith('TNTORIGIN 123 7 '))
        self.assertIn(' 116 ',line)

    def test_one_missing_angle_row_does_not_require_another_full_scene_discovery(self):
        self.poll()
        self.assertEqual(self.reader.poll(self.cache,None),'TNTORIGIN-')
        self.assertIsNotNone(self.reader.cache)
        self.assertEqual(self.poll(),'TNTORIGIN-')
        self.assertTrue(self.poll().startswith('TNTORIGIN '))

    def test_intermediate_scale_shear_and_disagreeing_angle_do_not_apply(self):
        for field, value in [('linear_scale_xy', [1.5, 1.5]),
                             ('linear_determinant', .5), ('fire_angle', 116)]:
            with self.subTest(field=field):
                original = self.row[field]
                self.row[field] = value
                self.poll()
                self.assertEqual(self.poll(), 'TNTORIGIN-')
                self.assertFalse(self.reader.observation['source_eligible'])
                self.assertEqual(self.reader.observation[field], value)
                self.row[field] = original

    def test_double_scale_publishes_scaled_muzzle_and_ready_status(self):
        self.row.update(linear_scale_xy=[2,2],linear_determinant=-4,
                        fire_pos_rounded=[60,260])
        self.assertEqual(self.poll(),'TNTORIGIN-')
        line=self.poll()
        self.assertTrue(line.startswith('TNTORIGIN 123 7 '))
        self.assertTrue(line.endswith(' 60 260 2.0'))
        self.assertIn(' ready 2.000000',stream.origin_status_line(self.reader.observation))
        self.assertTrue(self.reader.observation['source_eligible'])
        # Returning to scale 1 requires a new stable geometry pair.
        self.row.update(linear_scale_xy=[1,1],linear_determinant=-1,
                        fire_pos_rounded=[80,280])
        self.assertEqual(self.poll(),'TNTORIGIN-')
        self.assertTrue(self.poll().endswith(' 80 280 1.0'))

    def test_identity_change_requires_new_discovery(self):
        self.poll()
        self.cache = dict(self.cache, sprite=0x9999)
        self.assertEqual(self.poll(), 'TNTORIGIN-')
        self.assertIsNone(self.reader.cache)

    def test_read_failure_is_recorded_as_unverified_not_old_pose(self):
        with mock.patch.object(origin, 'sample', side_effect=RuntimeError('changed parent')):
            self.assertEqual(self.poll(), 'TNTORIGIN-')
        self.assertEqual(self.reader.observation['status'], 'unverified')
        self.assertFalse(self.reader.observation['source_eligible'])
        self.assertNotIn('fire_pos_rounded', self.reader.observation)

    def test_origin_failure_keeps_angle_reader_available(self):
        with mock.patch.object(origin, 'sample', side_effect=RuntimeError('changed parent')):
            self.assertEqual(self.poll(), 'TNTORIGIN-')
        self.assertIsNone(self.reader.cache)


if __name__ == "__main__":
    unittest.main()
