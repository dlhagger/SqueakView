# Hard-power-loss qualification protocol

This is a destructive bench qualification, not a normal operating procedure.
Use a dedicated test run and storage that contains no irreplaceable data. Do
not substitute `reboot`, `shutdown`, or closing the GUI: those exercise orderly
software paths rather than loss of input power.

## Preconditions

1. Commit the exact candidate build and rebuild the model engine through
   Project Setup to produce its schema-3 manifest. Confirm ordinary short GUI
   runs qualify first.
2. Launch the GUI from a standalone terminal, with the serial controller and
   camera connected exactly as they will be in production.
3. Record the Jetson supply, storage device/filesystem, power mode, model hash,
   application commit, and selected run directory.
4. Use a switched, current-rated bench supply under local operator control.
   Expect filesystem damage; never perform this remotely or during a real
   experiment.

## Procedure

1. Start a triggered GUI run and confirm camera TTLs, recording admissions,
   MP4 growth, and system telemetry for at least five minutes.
2. Without pressing **Stop**, remove DC input power at the bench supply.
3. Restore power, boot normally, and copy the interrupted run directory before
   attempting any recovery or modification.
4. Record filesystem/kernel errors and inspect the copied `run_status.json`,
   ledgers, recording telemetry, and MP4 with the application full-decode
   validator (`ffmpeg -xerror`), not a container-header frame count.
5. Run the read-only-derived qualification report outside the copied run:

   ```bash
   .venv/bin/python scripts/qualify_run.py /path/to/copied-run \
     --output /tmp/power-loss-qualification.json
   ```

6. Confirm that the report is `failed`, including
   `terminal_success_state: false`; an interrupted `recording`, `stopping`, or
   `finalizing` state must never qualify as scientific evidence.
7. Start and cleanly stop a new short GUI run. It must acquire the process lock,
   camera, serial controller, and output storage normally after the crash.

## Acceptance evidence

- The interrupted run cannot qualify or be mislabeled production-complete.
- No frames after the last durable source/admission records are invented.
- Any readable MP4 prefix is treated as salvage only, never as a finalized run.
- The acquisition lock is released by process death and the next clean run
  validates successfully.
- The copied interrupted directory, qualification JSON, console log, kernel
  log excerpt, and next clean-run manifest are retained with the test record.
