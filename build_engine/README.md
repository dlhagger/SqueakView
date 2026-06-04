# Build Engine Notebook

Notebook-first workflow for importing YOLO26 pose models into `SqueakView`.

Primary file:

- `build_engine.ipynb`

The notebook is intentionally scoped to YOLO26 pose packages. Given a `.pt`
model and its training/data `.yaml`, it builds the files expected by the
runtime:

```text
models/<model_name>/
  model.yaml
  weights/<model>.pt
  onnx/<model>_<precision>_b<batch>.onnx
  engines/<model>_<precision>_b<batch>.engine
  labels/classes.txt
  labels/labels.txt
  configs/<model_name>.txt
  configs/<model_name>.pose.json
  validation/import_report.json
```

## What It Does

1. Copies the source `.pt` into the model package.
2. Reads class names and keypoint shape from the `.pt` and data YAML.
3. Validates the YOLO26 pose contract: class count, keypoint count, and `x/y/conf`
   keypoint layout.
4. Exports ONNX with Ultralytics using `nms=False`.
5. Builds a TensorRT engine with local TensorRT `trtexec`.
6. Writes a DeepStream `nvinfer` config for `NvDsInferParseYolo26Pose`.
7. Writes a pose sidecar JSON used by SqueakView instead of unsupported custom
   keys inside the DeepStream config.
8. Writes `model.yaml` and `validation/import_report.json` for later GUI and
   preflight integration.
9. Verifies that the generated DeepStream config paths are absolute and point to
   existing files.

## Environment

From the repo root, use the existing SqueakView environment:

```bash
uv run jupyter lab build_engine/build_engine.ipynb
```

You can also open the notebook from an already-active environment if it has
Ultralytics, PyTorch, ONNX, PyYAML, and TensorRT available.

## Runtime Assumptions

- DeepStream is installed on the Jetson.
- `trtexec` is available at `/usr/src/tensorrt/bin/trtexec`, on `PATH`, or via
  `TRTEXEC=/path/to/trtexec`.
- The generated DeepStream config uses absolute paths for the ONNX, engine,
  labels, and custom parser library. `nvinfer` resolves these keys relative to
  the config file, so absolute paths are safer than repo-relative paths.
- The generated config is for YOLO26 pose only:
  `parse-bbox-func-name=NvDsInferParseYolo26Pose`.
- Engine builds and notebooks can fragment Jetson GPU/NvMap memory. Before long
  acquisition runs, close Jupyter/build processes or reboot if preflight reports
  low `tegrastats` `lfb` memory.

## Before Selecting The Model In The GUI

Check these values in the generated package:

1. `model.yaml` batch size matches the intended camera count.
2. `configs/<model_name>.txt` points to an existing ONNX and engine.
3. `labels/classes.txt` and `labels/labels.txt` are correct.
4. `validation/import_report.json` has no failed checks.
