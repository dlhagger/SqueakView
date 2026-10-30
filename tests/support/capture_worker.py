"""Substitute only the physical camera in the full operator software test."""
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading

from squeakview.common.child_events import encode_child_event
from squeakview.common.run_layout import artifact_path
from tests.support.recordings import create_recording


def main():
    run = Path(sys.argv[1])
    serial_enabled = len(sys.argv) > 2 and sys.argv[2] == 'v2-free'
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    with tempfile.TemporaryDirectory() as directory:
        source = create_recording(Path(directory), 'native', health=True)
        for name in ('raw.mp4', 'frames.csv', 'sensors.csv', 'detections.csv', 'capture_health_cam0.json', 'record_admission.csv'):
            if serial_enabled and name == 'sensors.csv':
                continue  # The real SerialHandle owns this file.
            destination = artifact_path(run, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact_path(source, name), destination)
    from squeakview.common import run_context
    health_path = artifact_path(run, 'capture_health_cam0.json')
    health = run_context.read_json_required(health_path)
    health['state'] = 'recording'
    run_context.atomic_write_json(health_path, health)
    print(encode_child_event('pipeline_ready', run_dir=str(run)), flush=True)
    if not stopped.wait(30):
        raise TimeoutError('test operator never stopped fixture capture')
    health['state'] = 'closed'
    run_context.atomic_write_json(health_path, health)
    print(encode_child_event('capture_closed', run_dir=str(run), exit_code=0, close_errors=[]), flush=True)


if __name__ == '__main__':
    main()
