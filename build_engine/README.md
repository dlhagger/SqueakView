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
  lib/libnvdsinfer_custom_impl_Yolo.so
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
7. Removes the Ultralytics JSON prefix from the engine, deserializes the raw
   TensorRT plan, and runs one bounded `trtexec` synthetic inference before
   giving it to DeepStream. The command, return code, timeout, truncation state,
   and bounded output are retained in `validation/import_report.json`.
8. Writes a detector config with clustering disabled because YOLO26 end-to-end
   output already contains final detections.
9. Writes the complete pose schema v2 with tensor/input contracts, a global confidence threshold, all keypoints assigned to each class, and class zero tracked by default.
10. Writes `model.yaml` and `validation/import_report.json` with portable
    dataset provenance plus the exact TensorRT, CUDA, Jetson Linux, device, and
    compute-capability identity used to build the engine. Schema-3 packages are
    rejected when that identity does not match the acquisition runtime.
11. Records path-bound SHA-256 identities for every runtime artifact, including
    `validation/import_report.json`, preventing either mixed build files or
    edited engine-execution/build-environment evidence from passing preflight.
12. Builds in a hidden staging directory on the `models/` filesystem, validates
    the complete package there, then publishes it with one atomic directory
    rename. When overwrite is enabled, Linux atomic exchange preserves the old
    complete package until the new package has passed validation; failed builds
    never modify the selected package.

Hidden `.build-*` directories are incomplete staging output, never selectable
model packages. They are cleaned when the notebook kernel exits and before a
rerun. If the machine loses power during export, they can be removed safely;
the existing non-hidden package remains intact.

For an automated, non-destructive candidate build, set a new single-component
package name before executing the notebook, for example
`SQUEAKVIEW_BUILD_MODEL_NAME=mousehouse_jp721`. Existing packages are never
replaced unless `SQUEAKVIEW_BUILD_OVERWRITE=1` is explicitly set; the only
accepted overwrite values are `0` and `1`.

Current upstream CUDA 13.2 PyTorch wheels may print an Orin compute-capability
8.7 warning even when CUDA operations work; NVIDIA has acknowledged this
warning for JetPack 7.2 upstream wheels. The notebook does not hide it. Package
acceptance instead requires the CUDA export, TensorRT deserialization, and a
bounded real `trtexec` inference to succeed. Treat an actual CUDA or `trtexec`
failure as fatal. See NVIDIA's [JetPack 7.2 Orin PyTorch guidance](https://forums.developer.nvidia.com/t/how-do-i-correctly-install-pytorch-on-jetpack-7-2/372773/5).

At model selection, SqueakView hashes `model.yaml`, the pose sidecar, ONNX
graph, DeepStream config, and TensorRT engine into the run manifest so a
qualified run identifies both its portable model inputs and device-local plan.

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
- The generated DeepStream config uses package-relative paths for the ONNX,
  engine, labels, and a package-local copy of the custom parser library, keeping
  each validated package internally consistent across clone locations.
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
uv run python -m squeakview.model_package \
  --config models/<model_name>/configs/<model_name>.txt \
  --require-engine-identity
```
