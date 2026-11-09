# Project Model Builder

Model construction is part of SqueakView's pre-acquisition Project Setup UI.
The former notebook implementation has been retired so build behavior cannot
diverge from the operator workflow.

Launch SqueakView normally:

```bash
bash squeakview.sh
```

After choosing or creating a project, Project Setup displays the checkpoint/YAML
pairs under that project's `model_sources/` directory. Select a source and press
**Build Model**. New projects contain independent MouseHouse v2 and stock YOLO
pose sources, with MouseHouse v2 preselected.

The setup application starts a separate worker process which:

1. acquires the project's exclusive writer lock;
2. removes only stale staging for the selected package;
3. exports the device-local FP16 TensorRT engine and ONNX graph;
4. validates the YOLO26 pose contract and executes one bounded `trtexec` pass;
5. writes schema-3 package provenance and artifact identities;
6. atomically publishes the complete package; and
7. selects it as the project default.

New model packages bundle the parser-only library on DeepStream 9.2 and the
legacy library on 9.1. Native setup and model packaging share the same installed
SDK-version selector. Existing model packages keep their own parser artifact.

The builder disables Ultralytics automatic dependency installation before importing
it (`YOLO_AUTOINSTALL=false`) and requires JetPack's system TensorRT rather than
a pip-installed copy inside `.venv`. A stale worker imported with automatic
installation enabled fails early and must be restarted. Install missing Python
dependencies explicitly through UV and vendor dependencies through JetPack.
The import report records the TensorRT module path as well as its version.

If a previous export installed `tensorrt-cu13`, `tensorrt-cu13-bindings`, and
`tensorrt-cu13-libs` in `.venv`, close the builder and remove only those local
packages with `uv pip uninstall --python .venv/bin/python tensorrt-cu13
tensorrt-cu13-bindings tensorrt-cu13-libs`. Verify that a fresh environment
Python imports TensorRT from `/usr/lib/python3.12/dist-packages` before rebuilding.

Exports explicitly select `nms=False`, `max_det=300` detections per image,
`agnostic_nms=False`, static dimensions, and `quantize=16` (FP16) or `32` (FP32).
Engine metadata must match the requested pose task, batch/image size, keypoint
shape, precision, and NMS-free head. The import report retains the requested
arguments alongside the exporter metadata and execution results. Compare source
checkpoint accuracy using `nms=False` too: Ultralytics' default validation uses
a different head with NMS.

The existing published package is preserved if export or validation fails.
Cancellation terminates the complete worker process group; an unpublished
staging directory is safely recovered on the next attempt.

The setup UI does not import PyTorch, Ultralytics, TensorRT, or CUDA. After a
successful build it waits for the worker to exit, reports that GPU resources
have been released, and only then allows the acquisition supervisor to start.
No manual kernel restart, cache clearing, swap manipulation, or reboot is
required. Bounded build logs are retained under
`~/.local/state/SqueakView/logs/` and can also be copied directly from Project
Setup.

The retained `build_engine.ipynb` contains only a migration notice and no
executable build path. For diagnostics or automation, the same isolated worker
can be invoked directly:

```bash
.venv/bin/python -m squeakview.apps.model_builder \
  --project "$HOME/Documents/SqueakView Projects/My Project" \
  --source mousehouse_v2 \
  --model-name mousehouse_v2
```

Normal operators should use Project Setup rather than this command.
