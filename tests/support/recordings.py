"""Small real recordings for software workflow tests; no devices or GPU jobs.

The 1,016-byte H.264/MP4 fixture contains three generated black 64x64 frames.
Generated with videotestsrc ! openh264enc ! h264parse ! qtmux (30 fps).
"""
from pathlib import Path
import base64
import csv
import json
import time
from squeakview.common.run_layout import initialize_layout, artifact_path
from squeakview.common import run_context, controller_protocol_v2 as protocol
from squeakview.apps.inference.frame_audit import FrameCsvOperator
from squeakview.apps.inference.pose_persistence import OBJECT_HEADERS
from squeakview.common.serial import SERIAL_HEADER
from squeakview.common.sensor_csv import make_event_record
MODES = ('v2-triggered', 'v2-free', 'native', 'legacy-triggered')

def write_test_video(path):
    data = Path(__file__).parents[1] / 'fixtures' / 'three_black_frames.mp4.b64'
    Path(path).write_bytes(base64.b64decode(data.read_bytes(), validate=False))

def table(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))

def create_recording(run, mode, *, recovery=False, health=False):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    initialize_layout(run, compact_metadata=True)
    base = time.time_ns()
    triggered = mode in ('v2-triggered', 'legacy-triggered')
    enabled = mode != 'native'
    manifest = {'schema_version': '3.0', 'run_id': 'fixture-' + mode, 'capture': {'trigger_on': triggered, 'num_cameras': 1, 'width': 64, 'height': 64, 'fps': 30}, 'serial': {'enabled': enabled, 'controller_protocol': 'v2' if mode.startswith('v2') else 'legacy', 'alignment_required': mode == 'legacy-triggered'}, 'inference': {'enabled': False, 'preview_enabled': False}, 'mouse_id': 'fixture'}
    write_json(artifact_path(run, 'run_manifest.json'), manifest)
    write_test_video(run / 'raw.mp4')
    rows = []
    captures = []
    admission = []
    for i in range(3):
        stamp = 1000000 + i * 33333333
        host = base + i * 33333000
        rows.append({key: val for key, val in dict(stream_id=0, camera_serial='fake', deepstream_frame_number=i, source_sequence_index=i, source='cam0', raw_frame_index=i, pts_ns=i * 33333333, dts_ns=i * 33333333, duration_ns=33333333, host_unix_ns=host, host_monotonic_ns=100000 + i * 33333333, status='ok', camera_frame_id=100 + i, camera_timestamp_ns=stamp, gst_pts_ns=i * 33333333, source_width=64, source_height=64, metadata_status='source_native').items()})
        captures.append(dict(camera_index=0, source_sequence_index=i, camera_frame_id=100 + i, camera_timestamp_ns=stamp, gst_pts_ns=i * 33333333, host_received_unix_ns=host, host_received_monotonic_ns=100000 + i * 33333333, source_width=64, source_height=64, total_incomplete=0, total_frame_gap_events=0, total_crc_failures=0, native_timing_validated=True))
        admission.append(dict(stream_id=0, record_frame_index=i, pts_ns=i * 33333333))
    table(run / 'frames.csv', FrameCsvOperator.HEADERS, rows)
    cap = artifact_path(run, 'capture_cam0.jsonl')
    cap.parent.mkdir(parents=True, exist_ok=True)
    cap.write_text(''.join((json.dumps(r) + '\n' for r in captures)))
    table(artifact_path(run, 'record_admission.csv'), ['stream_id', 'record_frame_index', 'pts_ns'], admission)
    table(run / 'detections.csv', OBJECT_HEADERS, [])
    sensor = []
    if mode.startswith('v2'):
        payloads = [(1, 1000, 'ACK_START,1,1000,nan,30,33333,69420,69420,task,nan')]
        if triggered:
            payloads.append((3, 1000, 'CAMERA_EPOCH,count=1,timestamp_us=1000,period_us=33333,pulse_us=1000,health=0x3,queue=0/64,suppressed=0,reason=Start'))
        else:
            payloads[0] = (1, 1000, 'ACK_START_BEHAVIOR,1,1000,nan,0,0,69420,69420,task,nan')
        payloads.append((1, 2000, 'DRINK_START,1,2000,LD,1,69420,69420,69420,task,nan'))
        if triggered:
            payloads.append((5, 100000, 'CAMERA_STOP,count=3,timestamp_us=67666,period_us=33333,pulse_us=1000,health=0x1,queue=0/64,suppressed=0,reason=StopCommand'))
        payloads.append((1, 100000, 'ACK_STOP,1,100000,nan,3,69420,69420,69420,task,nan'))
        if recovery:
            for side in ('LD', 'RD'):
                payloads.extend([
                    (1, 1500, f'LICK_RECOVERY_START,1,1500,{side},0,0,1,1,task,START'),
                    (1, 50000, f'LICK_RECOVERY_READY,1,50000,{side},0,48500,1,1,task,READY'),
                ])
            payloads.sort(key=lambda item: item[1])
        journal = []
        for seq, (kind, stamp, payload) in enumerate(payloads, 10):
            frame = protocol.decode_frame(protocol.encode_frame(message_type=kind, flags=1, boot_id=12345, session_id=7, sequence=seq, monotonic_us=stamp, payload=payload.encode())[:-1])
            journal.append(dict(schema_version=1, host_unix_ns=base + seq, host_monotonic_ns=100000 + seq, message_type=kind, flags=1, boot_id=12345, session_id=7, sequence=seq, monotonic_us=stamp, payload_utf8=payload, decoded_hex=frame.decoded.hex()))
            sensor.append(make_event_record(payload.split(','), payload, frame=frame, host_unix_ns=base + seq, host_monotonic_ns=100000 + seq))
        artifact_path(run, 'diagnostics').mkdir(parents=True, exist_ok=True)
        artifact_path(run, 'diagnostics/controller_v2.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in journal)))
        write_json(artifact_path(run, 'diagnostics/controller_v2_summary.json'), dict(protocol='mousehouse_v2', boot_id=12345, session_id=7, journal_start_sequence=10, integrity_latched=False, counts=dict(frames_stored=len(journal), boot_boundaries=0, crc_or_framing_errors=0, conflicting_duplicates=0)))
        write_json(artifact_path(run, 'diagnostics/clock_validation.json'), dict(result='PASS', before_samples=[dict(round_trip_ns=200, jetson_send_ns=base - 100, jetson_receive_ns=base + 100, rp2040_receive_us=1000, rp2040_send_us=1000)]))
    elif enabled:
        for event, stamp, count, reason in [('MARKER', 500, 0, 'START_SENT'), ('CAMERA_HIGH', 1000, 1, 'nan'), ('CAMERA_HIGH', 34333, 2, 'nan'), ('CAMERA_HIGH', 67666, 3, 'nan'), ('MARKER', 90000, 0, 'CAPTURE_STOP_REQUESTED'), ('MARKER', 91000, 0, 'STOP_SENT'), ('MARKER', 100000, 0, 'CAPTURE_STOP_DONE')]:
            payload = f'{event},1,{stamp},nan,{count},69420,69420,69420,task,{reason}'
            sensor.append(make_event_record(payload.split(','), payload, host_unix_ns=base + stamp * 1000, host_monotonic_ns=100000 + stamp * 1000))
    table(run / 'sensors.csv', SERIAL_HEADER, sensor)
    run_context.write_bottle_artifacts(run, {'mouse_weight_g': 25, 'left': {'fluid': 'water', 'initial_weight_g': 30, 'final_weight_g': 29}, 'right': {'fluid': 'water', 'initial_weight_g': 30, 'final_weight_g': 29}})
    run_context.write_status(run, 'capture_closed', expected_ttl_count=3 if triggered else None, ledger_frame_counts=[3, 3], capture_exit_code=0)
    if health:
        for row in rows:
            row.update(metadata_status='ok', host_received_unix_ns=row['host_unix_ns'],
                       host_received_monotonic_ns=row['host_monotonic_ns'], missing_frames_before=0)
        table(run / 'frames.csv', FrameCsvOperator.HEADERS, rows)
        cap.unlink()
        write_json(artifact_path(run, 'capture_health_cam0.json'), dict(schema_version=1,
            camera_index=0, state='closed', source_frames=3, total_timeouts=0,
            total_incomplete=0, total_frame_gap_events=0, total_crc_failures=0,
            native_timing_validated=True, host_unix_ns=base, host_monotonic_ns=100000))
    return run
