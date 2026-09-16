# SqueakView Project Workflow

The Git checkout is the SqueakView application installation. Scientific projects
are separate sibling directories, so pulling, rebuilding, or rolling back code
does not modify project data. There is no legacy-data migration in this
clean-break release.

## Application checkout and updates

The intended default layout is:

```text
~/Documents/
├── SqueakView/           Git-managed application checkout
└── SqueakView Projects/  Durable scientific projects
```

Clone and prepare the application from `~/Documents`:

```bash
cd "$HOME/Documents"
git clone https://github.com/dlhagger/SqueakView.git
cd SqueakView
uv venv --python 3.12 --system-site-packages
uv sync
bash scripts/setup_jetson.sh
```

`setup_jetson.sh` is the one-time privileged device provisioning step. It creates
`~/Documents/SqueakView Projects` with private user permissions when absent and
leaves it completely unchanged when it already exists. Reboot after setup adds
the operator to `dialout`. Even when setup itself is invoked through `sudo`, its
native compilation runs as the desktop user so it cannot leave root-owned build
artifacts in the Git checkout. For routine application updates, first finish
any active acquisition and close SqueakView, then run:

```bash
cd "$HOME/Documents/SqueakView"
git pull --ff-only
uv sync
bash scripts/build_native.sh
.venv/bin/python -m pytest -q
bash squeakview.sh
```

Routine code updates do not require a reboot. A rollback selects a known-good
Git commit or tag in the same checkout and then repeats `uv sync`, the native
build, and the tests. Update and cleanup commands must never target
`~/Documents/SqueakView Projects`; `.venv/` and native build outputs in the
checkout are replaceable application artifacts, while everything under the
projects directory is durable data.

## Launch and project selection

Start SqueakView from the application checkout:

```bash
bash squeakview.sh
```

The project chooser appears on every launch. The most recently opened valid
project is selected by default, but SqueakView does not open it until the
operator confirms. Choose **Open Other…** for an existing project or **Create
Project…** for a new one. The default parent is:

```text
~/Documents/SqueakView Projects/
```

First launch also creates this parent when absent, so pulling the application
onto an already-provisioned Jetson does not require rerunning system setup. An
existing parent and everything below it are left unchanged.

After selection, Project Setup appears before the acquisition supervisor. It
can build a project source, select an existing published package as the default,
or continue explicitly without inference. Returning from Project Setup goes back
to the chooser. A successful build worker must exit before the recording GUI can
start.

An explicit project path is available for controlled development and scripted
launches:

```bash
bash squeakview.sh "/absolute/path/to/My Project"
```

An open project holds an ownership lock for the complete supervisor session.
The camera/controller acquisition lock is device-wide, so two projects cannot
acquire the hardware concurrently.

## New-project contents

Project creation is staged and then published with one same-filesystem rename;
an existing destination is never replaced. Each new project contains:

```text
My Project/
├── squeakview_project.json
├── .squeakview.lock
├── model_sources/
│   ├── mousehouse_v2/
│   │   ├── mousehouse_v2.pt
│   │   └── mousehouse_v2.yaml
│   └── stock_yolo26_pose/
│       ├── yolo26n-pose.pt
│       └── coco-pose.yaml
├── models/
├── tasks/default.yaml
├── profiles/
│   ├── experiments/
│   └── subjects/
├── qualification/
└── runs/
```

`.squeakview.lock` is a small ownership-token file guarded by the operating
system for the lifetime of an open supervisor session. It prevents two app
instances from writing the same project; a stale unlocked file is harmless.

The two model-source packages are independent project-owned copies. The
application never edits them. They are source checkpoints, not deployable
TensorRT packages; build a device-local package before selecting inference in
the GUI. Users may add other checkpoint/YAML pairs under `model_sources/`.

## Build the MouseHouse v2 package

Launch SqueakView normally and select the project. Before experiments and
subjects are presented, Project Setup displays its model sources. MouseHouse v2
is preselected for a new project; press **Build Model**.

The TensorRT build runs in an isolated process, publishes
`models/mousehouse_v2/` transactionally, selects it as the project default, and
exits before the acquisition supervisor starts. The setup UI never imports the
CUDA build stack, so no manual restart is required. See
[the builder guide](../build_engine/README.md) for the package contract and
diagnostic command.

## Path ownership

- Application: Python code, native plugins, scripts, builder service, and immutable
  new-project templates.
- Project: runs, model sources, generated model packages, tasks, experiment and
  subject profiles, and qualification state.
- User state: recent-project catalog, GUI layout, and pre-project launch logs
  under `~/.config/SqueakView/` and `~/.local/state/SqueakView/`.
- Runtime: supervisor sockets and the device acquisition lock under
  `$XDG_RUNTIME_DIR/squeakview/`.

Run manifests record the canonical project identity and selected project-owned
configuration. Clean-break run-manifest schema 3 distinguishes the replaceable
application root from the durable project root, and qualification rejects a
missing or invalid project identity. Stored profile paths are project-relative
so moving a complete project does not bind it to an application checkout.

## Short on-device acceptance pass

Do this before an overnight or weekend acquisition:

1. Launch with no path. Confirm the chooser appears and preselects the last
   project without opening it automatically.
2. Create a disposable project. Confirm both model-source directories, the
   default task, and all empty project-owned directories listed above exist.
3. Build MouseHouse v2 in Project Setup. Confirm the worker reports that GPU
   resources were released and the package is selected as the project default.
4. Create one experiment and subject, close SqueakView normally, relaunch, and
   verify the chooser and project-local profiles restore correctly.
5. Run a two-minute triggered recording with inference and preview enabled.
   Confirm the header frame count advances, recording gaps stay at zero, serial
   events remain live, and the project `runs/` directory receives the run.
6. Stop normally. Wait for finalization and confirm `run_status.json` is
   terminal, `run_manifest.json` identifies the project, `raw.mp4` plays, and
   the recording/camera/controller counts pass.
7. On a safe bench setup, trigger a feeder jam. Confirm the persistent warning,
   disabled feeding, exact `CLEAR_JAM` acknowledgement behavior, and successful
   later shutdown.
8. In a separate disposable run, close the GUI during acquisition. Confirm the
   supervisor stops acquisition and produces a failed, terminal run rather than
   continuing headlessly.
9. Open a second project and verify the first project's profiles, models, and
   runs are not offered as the second project's assets.

Keep the short-run directories as acceptance evidence until their manifests and
logs have been reviewed. Only then begin the long-duration qualification run.

### Current acceptance evidence

On 2026-09-16 the `Test` project completed the model-build and primary short-run
gates on the Orin Nano Super. The isolated worker built a project-owned YOLO26
FP16 TensorRT package with a 1 GiB workspace ceiling, passed ONNX and TensorRT
execution validation, published it beneath `Test/models/mousehouse/`, and set it
as the project default. The subsequent triggered inference run recorded 2,672 of
2,672 source frames with zero gaps, incomplete frames, CRC failures, or recording
evictions. MP4 sample counting, controller/camera alignment, bottle persistence,
feeder-jam clear acknowledgement, shutdown, and finalization all passed.
The next launch preselected `Test`; Project Setup recognized `mousehouse` as the
project default without rebuilding; and the `DID` experiment and `TEST` subject
were restored. A new `Isolation_Test` project then received its own unique ID,
both shipped model sources, task, and qualification templates while correctly
starting with no published models, profiles, or runs. The original `Test`
project remained intact and both projects were independently recorded in the
recent-project catalog.

This evidence validates the core project/model/capture route but does not replace
the remaining operator gates: safe-bench GUI-loss behavior and the one-hour and
full-duration runs. Run those from the final clean commit so their manifests
identify the release candidate rather than a dirty development tree.

## Failure behavior

A missing, malformed, unsupported, or locked project is rejected before the GUI
acquires hardware. A reconnect does not rewrite project metadata. If project
creation fails, its unpublished staging directory is removed and the requested
destination remains absent; existing destinations are never cleaned or reused.
