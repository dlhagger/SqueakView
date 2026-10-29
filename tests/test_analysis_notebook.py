"""Execute the teaching notebook against real capture-only and pose tables."""
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')
from data_viz.notebook_support import selected_pose_schema
from squeakview.common.run_layout import artifact_path
from squeakview.apps.inference.pose_persistence import PoseCsvWriter, OBJECT_HEADERS, KEYPOINT_HEADERS
from tests.support.recordings import create_recording


class AnalysisNotebookTests(unittest.TestCase):
    def test_all_code_cells_execute_with_and_without_pose(self):
        import matplotlib.pyplot as plt
        notebook = json.loads((Path(__file__).parents[1] / 'data_viz/analysis_demo_viz.ipynb').read_text())
        with tempfile.TemporaryDirectory() as directory:
            for pose in (False, True):
                with self.subTest(pose=pose):
                    root = create_recording(Path(directory) / str(pose), 'v2-triggered')
                    if pose:
                        with (root / 'frames.csv').open() as stream:
                            frame = next(csv.DictReader(stream))
                        writer = PoseCsvWriter(root, ['nose'])
                        self.addCleanup(writer.close)
                        values = dict(observation_id='s0:f0:o0', stream_id=0, deepstream_frame_number=0, source_sequence_index=0, camera_frame_id=100, camera_timestamp_ns=1000000, gst_pts_ns=0, class_id=0, class_label='mouse', track_id=1, detected_this_frame=1, tracker_predicted=0, detector_confidence=.9, tracker_confidence=.8, detector_x=0, detector_y=0, detector_w=50, detector_h=50, track_x=0, track_y=0, track_w=50, track_h=50, pose_available=1, schema_version=2)
                        writer.write_object([values[k] for k in OBJECT_HEADERS], host_unix_ns=frame['host_unix_ns'], host_monotonic_ns=frame['host_monotonic_ns'])
                        point = dict(observation_id='s0:f0:o0', stream_id=0, deepstream_frame_number=0, source_sequence_index=0, camera_frame_id=100, track_id=1, class_id=0, class_label='mouse', keypoint_index=0, keypoint_name='nose', x_px=10, y_px=20, x_norm=10/64, y_norm=20/64, confidence=.8, visible=1, coordinate_space='source_px', source='flir_user_meta')
                        writer.write_keypoints([[point[k] for k in KEYPOINT_HEADERS]])
                        writer.close()
                        schema = artifact_path(root, 'config/fixture.pose.json')
                        schema.parent.mkdir(parents=True, exist_ok=True)
                        schema.write_text(json.dumps(dict(keypoint_count=1, keypoint_threshold=.5)))
                    completed = subprocess.run([sys.executable, "-m", "squeakview.apps.inference.post_run", str(root), "--align"], capture_output=True, text=True, timeout=15)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    scope = {}
                    try:
                        for index, cell in enumerate(notebook['cells']):
                            if cell['cell_type'] != 'code':
                                continue
                            source = ''.join(cell['source']).replace('RUN_DIR = None', f'RUN_DIR = Path({str(root)!r})')
                            with contextlib.redirect_stdout(io.StringIO()):
                                exec(compile(source, f'<notebook cell {index}>', 'exec'), scope)
                            scope['display'] = lambda *args, **kwargs: None
                    finally:
                        plt.close('all')
                    self.assertEqual(len(scope['frames']), 3)
                    self.assertEqual(len(scope['keypoints']), int(pose))

    def test_missing_declared_schema_is_an_error_for_pose_enabled_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_recording(Path(directory), 'v2-triggered')
            manifest = artifact_path(root, 'run_manifest.json')
            payload = json.loads(manifest.read_text())
            payload['inference']['enabled'] = True
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'pose schema'):
                selected_pose_schema(root, root)
