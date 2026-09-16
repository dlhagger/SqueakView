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
