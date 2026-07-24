# Build Engine Notebook

Notebook-first workflow for importing YOLO26 pose models into `SqueakView`.

Source `.pt` and YAML inputs under `build_me/` are shipped with the repository.
Generated packages under `models/` are device-local, ignored by Git, and must be
built on each fresh Jetson before inference can be enabled.

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

1. Reads classes, keypoint labels, and keypoint shape from the dataset YAML.
2. Treats the dataset YAML as read-only ground truth; no SqueakView-specific fields are required or written. No model-specific labels or indices live in the notebook.
3. Validates those values against the source `.pt` checkpoint.
4. Passes the same YAML to the Ultralytics exporter with `data=...`.
5. Exports a static, end-to-end TensorRT engine directly on the target Jetson.
6. Validates the ONNX model and `(batch, 300, 6 + 3*kpts)` output contract.
7. Removes the Ultralytics JSON prefix from the engine and validates the raw
   TensorRT plan before giving it to DeepStream.
8. Writes a detector config with clustering disabled because YOLO26 end-to-end
   output already contains final detections.
9. Writes the complete pose schema v2 with tensor/input contracts, a global confidence threshold, all keypoints assigned to each class, and class zero tracked by default.
10. Writes `model.yaml` and `validation/import_report.json` with portable
    dataset provenance and a schema-v2 check.
11. Writes the completed model package under `models/<model_name>/`.

## Dataset Metadata

The builder reads standard `names`, `kpt_shape`, and any of `kp_names`, `keypoint_names`, or Ultralytics `kpt_names`. The checkpoint supplies classes, shape, and named keypoints when the YAML omits them. The source YAML is never modified.

## Environment

From the repo root, use the existing SqueakView environment:

```bash
uv run jupyter lab build_engine/build_engine.ipynb
```

You can also open the notebook from an already-active environment if it has
Ultralytics, PyTorch, ONNX, PyYAML, and TensorRT available.

## Runtime Assumptions

- DeepStream is installed on the Jetson.
- The TensorRT Python bindings installed by JetPack are importable.
- The generated DeepStream config uses paths relative to the config file for the
  ONNX, engine, labels, and custom parser library, keeping model packages
  portable across clone locations.
- The generated config is for YOLO26 pose only:
  `parse-bbox-func-name=NvDsInferParseYolo26Pose`.
- Build the native parser after system CUDA/TensorRT updates and before running
  the notebook. Close Jupyter after engine builds before long acquisition runs.

## Before Selecting The Model In The GUI

Check these values in the generated package:

1. `model.yaml` batch size matches the intended camera count.
2. `configs/<model_name>.txt` points to an existing ONNX and engine.
3. `labels/classes.txt` and `labels/labels.txt` are correct.
4. `validation/import_report.json` has no failed checks.

Then run the strict package validation used by preflight:

```bash
uv run python -m squeakview.model_package --config models/<model_name>/configs/<model_name>.txt
```
