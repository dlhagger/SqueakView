from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from squeakview.common import run_context
from squeakview.common.run_layout import artifact_path, initialize_layout
from squeakview.common.run_outputs import finalize_primary_tables, write_weigh_in
from squeakview.apps.inference.pose_persistence import OBJECT_HEADERS, KEYPOINT_HEADERS, PoseCsvWriter
from squeakview.common.serial import SerialHandle, SERIAL_HEADER
from squeakview.common.controller_protocol_v2 import Frame, MessageType


class PrimaryRunOutputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        initialize_layout(self.root)
        artifact_path(self.root, 'diagnostics').mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, name, rows):
        path = artifact_path(self.root, name)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def rows(self, name):
        with artifact_path(self.root, name).open(newline='') as handle:
            return list(csv.DictReader(handle))

    def write_json(self, name, value):
        path = artifact_path(self.root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def anchors(self):
        records = [dict(payload_utf8=f'{kind},count={count},timestamp_us={us}',
                        boot_id=12, session_id=9)
                   for kind, count, us in [('CAMERA_EPOCH', 40, 1000), ('CAMERA_STOP', 42, 3000)]]
        artifact_path(self.root, 'diagnostics/controller_v2.jsonl').write_text(
            ''.join(json.dumps(r)+'\n' for r in records))

    def frames(self):
        return [dict(stream_id=0, source_sequence_index=i, raw_frame_index=i,
                     camera_timestamp_ns=5000+i*1000, gst_pts_ns=i*1000,
                     host_unix_ns=10000000+i*1000, host_monotonic_ns=50000+i*1000)
                for i in range(3)]

    def test_paths_keep_only_primary_tables_at_root_and_legacy_unchanged(self):
        self.assertEqual(artifact_path(self.root, 'objects.csv'), self.root/'detections.csv')
        self.assertEqual(artifact_path(self.root, 'serial.csv'), self.root/'sensors.csv')
        self.assertEqual(artifact_path(self.root, 'run_manifest.json'), self.root/'_metadata/run_manifest.json')
        self.assertEqual(artifact_path(self.root.parent, 'serial.csv'), self.root.parent/'serial.csv')
        with self.assertRaises(ValueError):
            artifact_path(self.root, '../escape')

    def test_pose_is_one_row_with_named_keypoints_and_native_timing(self):
        writer = PoseCsvWriter(self.root, ['nose'])
        row = ['']*len(OBJECT_HEADERS)
        for field, value in dict(observation_id='o1', stream_id=0, source_sequence_index=0,
                                 camera_timestamp_ns=5000).items():
            row[OBJECT_HEADERS.index(field)] = value
        writer.write_object(row, host_unix_ns=10000000, host_monotonic_ns=50000)
        point = ['']*len(KEYPOINT_HEADERS)
        for field, value in dict(observation_id='o1', keypoint_name='nose', x_px=2,
                                 y_px=3, confidence=.9, visible=1).items():
            point[KEYPOINT_HEADERS.index(field)] = value
        writer.write_keypoints([point])
        writer.close()
        rows = self.rows('detections.csv')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['nose_x_px'], '2')
        self.assertEqual(rows[0]['camera_timestamp_ns'], '5000')
        self.assertEqual(rows[0]['host_unix_ns'], '10000000')
        self.assertFalse((self.root/'objects.csv').exists())
        self.assertFalse((self.root/'keypoints.csv').exists())

    def test_frame_interpolation_and_every_detection_gets_its_own_time(self):
        self.anchors()
        self.write_csv('frames.csv', self.frames())
        self.write_csv('detections.csv', [dict(observation_id='o1',stream_id=0,source_sequence_index=1)])
        finalize_primary_tables(self.root)
        frames = self.rows('frames.csv')
        self.assertEqual([r['controller_time_us'] for r in frames], ['1000','2000','3000'])
        self.assertEqual([r['controller_time_method'] for r in frames], ['exact_anchor','piecewise_interpolated','exact_anchor'])
        detection = self.rows('detections.csv')[0]
        self.assertEqual(detection['controller_time_us'], '2000')
        self.assertEqual(detection['camera_timestamp_ns'], '6000')
        self.assertEqual(detection['host_unix_ns'], frames[1]['host_unix_ns'])
        finalize_primary_tables(self.root)  # Recovery is repeatable.
        self.assertEqual(self.rows('frames.csv'), frames)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ['_metadata','detections.csv','frames.csv','sensors.csv','weigh_in.csv'])

    def test_native_only_run_does_not_invent_controller_time(self):
        self.write_csv('frames.csv', self.frames())
        finalize_primary_tables(self.root)
        self.assertTrue(all(r['controller_time_method']=='unavailable' for r in self.rows('frames.csv')))
        self.assertTrue(all(r['host_unix_ns'] for r in self.rows('frames.csv')))

    def test_unmapped_detection_fails_finalization(self):
        self.write_csv('frames.csv', self.frames())
        self.write_csv('detections.csv', [dict(stream_id=0,source_sequence_index=99)])
        with self.assertRaisesRegex(ValueError, 'no recorded frame'):
            finalize_primary_tables(self.root)

    def test_missing_frame_timestamp_fails(self):
        frames = self.frames()
        frames[0]['camera_timestamp_ns'] = ''
        self.write_csv('frames.csv', frames)
        with self.assertRaisesRegex(ValueError, 'native timestamp'):
            finalize_primary_tables(self.root)

    def test_weigh_in_has_mouse_bottles_and_preserves_entry_history(self):
        self.write_json('run_status.json', {'mouse_id':'mouse1'})
        data = {'mouse_weight_g':25, 'left':{'fluid':'water','initial_weight_g':100}}
        write_weigh_in(self.root, data)
        original = self.rows('weigh_in.csv')
        self.assertEqual({r['measurement_type'] for r in original}, {'bottle','mouse'})
        self.assertTrue(all(r['entered_at_unix_ns'] and r['host_monotonic_ns'] for r in original))
        self.assertTrue(all(r['mouse_id']=='mouse1' for r in original))
        write_weigh_in(self.root, data)
        self.assertEqual(self.rows('weigh_in.csv'), original)
        data['mouse_weight_g']=26
        write_weigh_in(self.root, data)
        self.assertEqual(len(self.rows('weigh_in.csv')), 3)
        self.assertEqual(self.rows('weigh_in.csv')[:2], original)

    def test_weigh_in_maps_host_entry_time_without_claiming_direct_measurement(self):
        self.write_json('diagnostics/clock_validation.json', {'result':'PASS','before_samples':[
            dict(round_trip_ns=100, jetson_send_ns=1000000, jetson_receive_ns=1000100,
                 rp2040_receive_us=100, rp2040_send_us=100)]})
        self.write_json('diagnostics/controller_v2_summary.json', {'boot_id':12, 'session_id':9})
        with patch('squeakview.common.run_outputs.time.time_ns', return_value=2000050):
            write_weigh_in(self.root, {'mouse_weight_g':25})
        row = self.rows('weigh_in.csv')[0]
        self.assertEqual(row['controller_time_us'], '1100')
        self.assertEqual(row['controller_time_method'], 'host_clock_estimate')
        self.assertEqual(row['controller_boot_id'], '12')

    def test_sensor_rows_include_native_controller_and_host_times(self):
        handle = SerialHandle('/dev/test', 115200, lambda _: None)
        handle._open_csv(self.root/'sensors.csv')
        handle._write_csv_line('POKE,1,12345,left,1')
        handle.log_marker('test')
        handle._csv_file.flush()
        rows = self.rows('sensors.csv')
        self.assertEqual(rows[0]['controller_time_us'], '12345')
        self.assertEqual(rows[0]['controller_time_method'], 'direct_event')
        self.assertTrue(all(r['hostUnixNs'] and r['hostMonotonicNs'] for r in rows))
        handle.close()

    def test_one_frame_epoch_and_stop_can_share_an_anchor(self):
        path = artifact_path(self.root, 'diagnostics/controller_v2.jsonl')
        path.write_text(''.join(json.dumps(dict(payload_utf8=f'{kind},count=40,timestamp_us=1000',
                        boot_id=12, session_id=9))+'\n' for kind in ('CAMERA_EPOCH','CAMERA_STOP')))
        self.write_csv('frames.csv', self.frames()[:1])
        finalize_primary_tables(self.root)
        self.assertEqual(self.rows('frames.csv')[0]['controller_time_us'], '1000')
        self.assertEqual(self.rows('frames.csv')[0]['controller_time_method'], 'exact_anchor')

    def test_modern_manifest_declares_real_metadata_and_primary_paths(self):
        from squeakview.apps.operator.backend.manifest import RunManifestContext, RunManifestService
        from squeakview.apps.operator.backend.contracts import RunRequest
        context = RunManifestContext(config=RunRequest(mouse_id='mouse1'),
            application_root=self.root, project_root=self.root,
            project_id='00000000-0000-0000-0000-000000000001', project_name='test',
            runs_root=self.root.parent, created_at=None, storage={}, model_snapshot=None,
            device_context=None)
        manifest = RunManifestService(lambda _: None).build(self.root, context)
        outputs = manifest['expected_outputs']
        self.assertEqual(outputs['detections'], 'detections.csv')
        self.assertEqual(outputs['sensors'], 'sensors.csv')
        self.assertEqual(outputs['weigh_in'], 'weigh_in.csv')
        self.assertEqual(outputs['manifest'], '_metadata/run_manifest.json')
        self.assertEqual(manifest['observability']['deepstream_log'], '_metadata/diagnostics/deepstream.log')
        self.assertNotIn('keypoints', outputs)

    def test_recording_evidence_finds_private_ledgers(self):
        from squeakview.common.recording_evidence import capture_recording_evidence, recording_evidence_complete
        for name in ('raw.mp4','capture_cam0.jsonl','record_admission.csv'):
            artifact_path(self.root,name).write_bytes(b'evidence')
        evidence = capture_recording_evidence(self.root,1)
        self.assertTrue(recording_evidence_complete(evidence,1))

    def test_v2_sensor_preserves_receipt_time_before_persistence_delay(self):
        handle = SerialHandle('/dev/test',115200,lambda _:None)
        handle._open_csv(self.root/'sensors.csv')
        frame = Frame(int(MessageType.EVENT), 0,12,9,7,12345,b'POKE,1,12300,left,1',b'')
        handle._publish_v2_frame(frame, 999000, 888000)
        handle._csv_file.flush()
        row = self.rows('sensors.csv')[0]
        self.assertEqual(row['hostUnixNs'],'999000')
        self.assertEqual(row['hostMonotonicNs'],'888000')
        self.assertEqual(row['controller_boot_id'],'12')
        self.assertEqual(row['controller_time_us'],'12300')
        handle.close()

    def test_analysis_reader_expands_wide_keypoints_without_another_stream(self):
        from data_viz.pose_tables import load_pose_tables
        self.write_csv('detections.csv', [dict(observation_id='one', stream_id=0,
            source_sequence_index=0, controller_time_us=1000, nose_x_px=10,
            nose_y_px=20, nose_confidence=.8, nose_visible=1)])
        objects, points = load_pose_tables(self.root)
        self.assertEqual(len(objects), 1)
        self.assertEqual(points.iloc[0]['keypoint_name'], 'nose')
        self.assertEqual(points.iloc[0]['x_px'], 10)
        self.assertEqual(points.iloc[0]['controller_time_us'], 1000)

if __name__=='__main__':
    unittest.main()
