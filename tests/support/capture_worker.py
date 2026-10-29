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
        source = create_recording(Path(directory), 'native')
        for name in ('raw.mp4', 'frames.csv', 'sensors.csv', 'detections.csv', 'capture_cam0.jsonl', 'record_admission.csv'):
            if serial_enabled and name == 'sensors.csv':
                continue  # The real SerialHandle owns this file.
            destination = artifact_path(run, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact_path(source, name), destination)
    print(encode_child_event('pipeline_ready', run_dir=str(run)), flush=True)
    if not stopped.wait(30):
        raise TimeoutError('test operator never stopped fixture capture')
    print(encode_child_event('capture_closed', run_dir=str(run), exit_code=0, close_errors=[]), flush=True)


if __name__ == '__main__':
    main()
