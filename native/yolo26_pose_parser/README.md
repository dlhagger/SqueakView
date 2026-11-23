# YOLO26 pose parser

This self-contained DeepStream 9.2 library builds only the
`NvDsInferParseYolo26Pose` detector callback for end-to-end pose outputs. It uses
C++17 and DeepStream's supported custom-parser ABI, with no custom TensorRT
layers, V2 plugins, CUDA kernels, or TensorRT runtime linkage.

The legacy all-model parser sources and DeepStream 9.1 build support have been
removed. Boundary tests live in `tests/test_pose_parser.cpp` and compile this
parser directly. Existing model packages and acquisition outputs are preserved.

## Build and compare

From the application checkout:

```bash
bash scripts/build_native.sh
```

For optional equivalence testing, set `SQUEAKVIEW_POSE_BASELINE` to the absolute
path of the library bundled with an accepted model package, then run
`bash scripts/build_pose_parser.sh`. There is no implicit legacy-build baseline.
`SQUEAKVIEW_DEEPSTREAM_SDK` selects an alternative 9.2 SDK root.

The comparison candidate is
`native/yolo26_pose_parser/build-comparison/libnvdsinfer_squeakview_pose.so`.
Normal setup/model packaging uses the separate `build/` directory, so a missing
comparison baseline cannot break normal setup. The selected SDK version and
headers are checked on every CMake configuration.
CTest runs the existing malformed-output boundary tests against the candidate,
plus an ABI-level comparison against the baseline: 729 deterministic cases
cover 1/17/19 keypoints, row/channel-major tensors, batch-1 layouts, multiple
source sizes, per-class thresholds, unconfigured classes, and invalid values.
Status, detection count, class, confidence, and box coordinates must match exactly.
Optimized builds disable floating-point contraction to preserve baseline
letterbox rounding; no fast-math options are used.

Only the unused second copy of decoded poses has been removed. All keypoint
values are still validated; authoritative keypoint metadata, tracking,
persistence, and overlay handling remain in the existing Python pipeline.
This phase does not unify those paths or establish a speed improvement.

## Bounded real-output comparison

For an idle development Jetson, `tests/compare_recording.py` is a small manual
diagnostic for the accepted 640x640, 19-keypoint model. It samples four frames
(0, 300, 600, 900), runs the existing engine with an explicit CUDA stream, and
passes each output tensor to both callbacks. It does not replay the complete
acquisition pipeline, compare tracker trajectories, or modify the run/model.
Video preprocessing is for common-input parser comparison, not proof of exact
pixel equivalence with live FLIR acquisition or an accuracy evaluation.

```bash
.venv/bin/python native/yolo26_pose_parser/tests/compare_recording.py \
  --video /absolute/path/to/run/raw.mp4 \
  --engine /absolute/path/to/model/engines/model.engine \
  --baseline /absolute/path/to/model/lib/libnvdsinfer_squeakview_pose.so \
  --candidate native/yolo26_pose_parser/build-comparison/libnvdsinfer_squeakview_pose.so \
  --compare native/yolo26_pose_parser/build-comparison/compare_pose_parsers
```

The helper requires a recording longer than 900 frames and FLOAT model I/O.
Tensor files are temporary. Engine compatibility warnings remain visible.
A full offline re-inference job continues to belong on a downstream machine,
as documented in the application README.

## Integration result on the development Jetson

On DeepStream 9.2, the candidate and accepted `mousehouse_v2` parser were each
loaded through `nvinfer` with the same existing engine, tensor metadata decoder,
and CUDA NvDCF configuration. A 120-frame stream-copied sample of the accepted
bench recording produced exactly matching frame/PTS values, 120 observations,
boxes, classes, confidence values, keypoints, and tracking IDs. Both pipelines
reached EOS and exited successfully. This was a bounded diagnostic, not a live
camera run or accuracy evaluation.

The separately published development-project package is named
`mousehouse_v2_parser_candidate`. Its validation directory includes
`parser_integration.json` and both pipeline logs. The accepted package and
project default were preserved and checked by hashes. This result is limited to
the tested 9.2 stack.

## Live check and default selection

A subsequent live FLIR run with the candidate finalized with 859 frames,
1,281 pose observations, no reported camera gaps/errors, and passing recording
validation. The operator confirmed the overlay looked correct. This supports
the parser-only default on 9.2. The original accepted model
package is retained. Separating/unifying pose decoding and moving
work between CPU and GPU are later changes requiring their own measurements.
