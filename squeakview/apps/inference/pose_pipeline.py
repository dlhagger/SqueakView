"""Frame-scoped YOLO26 pose decoding, persistence, and DeepStream overlays."""
from __future__ import annotations

import atexit
import csv
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pyservicemaker import BatchMetadataOperator, osd


POSE_META_DESCRIPTOR = b"SQUEAKVIEW.POSE.OBJECT_META.v1"
UNTRACKED_OBJECT_IDS = {-1, (1 << 64) - 1}


@dataclass(frozen=True, slots=True)
class PoseClass:
    class_id: int
    name: str
    threshold: float
    track: bool
    keypoint_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PoseSchema:
    version: int
    input_width: int
    input_height: int
    output_layer: str
    keypoint_names: tuple[str, ...]
    classes: tuple[PoseClass, ...]
    keypoint_threshold: float

    @property
    def row_width(self) -> int:
        return 6 + 3 * len(self.keypoint_names)

    def class_spec(self, class_id: int) -> PoseClass | None:
        return next((item for item in self.classes if item.class_id == class_id), None)


def load_pose_schema(config_path: Path, class_names: list[str]) -> PoseSchema:
    """Load the required schema-v2 pose contract localized beside the run config."""

    path = config_path.with_name(f"{config_path.stem}.pose.json")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read pose schema {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Pose schema must contain a JSON object: {path}")
    version = int(data.get("schema_version", 0))
    if version != 2:
        raise ValueError(f"Pose schema version 2 is required, found {version}: {path}")
    if data.get("postprocess") != "pyservicemaker_yolo26_pose_v1":
        raise ValueError(f"Unsupported pose postprocess contract: {data.get('postprocess')!r}")

    label_value = data.get("keypoint_labels_path")
    if not label_value:
        raise ValueError(f"Pose schema is missing keypoint_labels_path: {path}")
    label_path = Path(str(label_value)).expanduser()
    if not label_path.is_absolute():
        label_path = path.parent / label_path
    try:
        labels = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    except OSError as exc:
        raise ValueError(f"Could not read pose keypoint labels {label_path}: {exc}") from exc
    count = int(data["keypoint_count"])
    if len(labels) != count:
        raise ValueError(f"pose sidecar declares {count} keypoints but provides {len(labels)} labels")

    raw_classes = data.get("classes")
    if not isinstance(raw_classes, list):
        raise ValueError("Pose schema classes must be a list")
    classes: list[PoseClass] = []
    for class_id, name in enumerate(class_names):
        matches = [
            item for item in raw_classes
            if isinstance(item, dict) and item.get("id") == class_id and item.get("name") == name
        ]
        if len(matches) != 1:
            raise ValueError(f"Pose schema must define class {class_id} as {name!r} exactly once")
        raw = matches[0]
        classes.append(
            PoseClass(
                class_id=class_id,
                name=name,
                threshold=float(raw["threshold"]),
                track=bool(raw["track"]),
                keypoint_indices=tuple(int(index) for index in raw["keypoint_indices"]),
            )
        )
    return PoseSchema(
        version=version,
        input_width=int(data["input_width"]),
        input_height=int(data["input_height"]),
        output_layer=str(data["output_layer"]),
        keypoint_names=tuple(labels),
        classes=tuple(classes),
        keypoint_threshold=float(data["keypoint_threshold"]),
    )
def _source_point(
    x: float,
    y: float,
    *,
    source_width: int,
    source_height: int,
    input_width: int,
    input_height: int,
) -> tuple[float, float]:
    scale = min(input_width / float(source_width), input_height / float(source_height))
    pad_x = (input_width - source_width * scale) / 2.0
    pad_y = (input_height - source_height * scale) / 2.0
    return (
        min(float(source_width), max(0.0, (x - pad_x) / scale)),
        min(float(source_height), max(0.0, (y - pad_y) / scale)),
    )


def decode_yolo26_rows(
    rows: Iterable[Iterable[float]],
    schema: PoseSchema,
    *,
    source_width: int,
    source_height: int,
) -> list[dict[str, Any]]:
    """Decode Ultralytics YOLO26 one-to-one pose rows without external NMS."""

    detections: list[dict[str, Any]] = []
    for detection_index, values in enumerate(rows):
        row = [float(value) for value in values]
        if len(row) != schema.row_width or not all(math.isfinite(value) for value in row):
            continue
        x1, y1, x2, y2, confidence, raw_class_id = row[:6]
        class_id = int(round(raw_class_id))
        if abs(raw_class_id - class_id) > 1e-3:
            continue
        class_spec = schema.class_spec(class_id)
        if class_spec is None or confidence < class_spec.threshold or x2 <= x1 or y2 <= y1:
            continue
        left, top = _source_point(
            x1, y1, source_width=source_width, source_height=source_height,
            input_width=schema.input_width, input_height=schema.input_height,
        )
        right, bottom = _source_point(
            x2, y2, source_width=source_width, source_height=source_height,
            input_width=schema.input_width, input_height=schema.input_height,
        )
        if right <= left or bottom <= top:
            continue
        keypoints: list[dict[str, Any]] = []
        for index in class_spec.keypoint_indices:
            if not 0 <= index < len(schema.keypoint_names):
                continue
            offset = 6 + index * 3
            x, y = _source_point(
                row[offset], row[offset + 1], source_width=source_width,
                source_height=source_height, input_width=schema.input_width,
                input_height=schema.input_height,
            )
            score = row[offset + 2]
            keypoints.append(
                {
                    "index": index,
                    "name": schema.keypoint_names[index],
                    "x": x,
                    "y": y,
                    "confidence": score,
                    "visible": bool(score >= schema.keypoint_threshold),
                }
            )
        detections.append(
            {
                "schema": POSE_META_DESCRIPTOR.decode(),
                "schema_version": schema.version,
                "detection_index": detection_index,
                "class_id": class_id,
                "class_label": class_spec.name,
                "detector_confidence": confidence,
                "detector_bbox": {"x": left, "y": top, "w": right - left, "h": bottom - top},
                "keypoints": keypoints,
                "coordinate_space": "source_pixels",
                "source_width": source_width,
                "source_height": source_height,
            }
        )
    return detections


class FramePoseStore:
    """Bounded frame-scoped handoff between the decoder and post-tracker operator.

    PyServiceMaker 9.1 currently double-frees JSON user metadata acquired from Python
    when it is attached to ObjectMetadata. A short token in the detector label keeps
    the association exact while the actual pose stays in this process-local frame map.
    The downstream operator deletes every frame immediately after consuming it.
    """

    TOKEN = "SQPOSE"

    def __init__(self):
        self._lock = threading.Lock()
        self._frames: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}

    def put(self, stream_id: int, frame_number: int, observation: dict[str, Any]) -> str:
        index = int(observation["detection_index"])
        with self._lock:
            self._frames.setdefault((stream_id, frame_number), {})[index] = observation
        return f"{self.TOKEN}:{frame_number}:{index}"

    def get(self, stream_id: int, frame_number: int, token: str) -> dict[str, Any] | None:
        parts = str(token or "").split(":")
        if len(parts) != 3 or parts[0] != self.TOKEN:
            return None
        try:
            token_frame, index = int(parts[1]), int(parts[2])
        except ValueError:
            return None
        if token_frame != frame_number:
            return None
        with self._lock:
            return self._frames.get((stream_id, frame_number), {}).get(index)

    def discard(self, stream_id: int, frame_number: int) -> None:
        with self._lock:
            self._frames.pop((stream_id, frame_number), None)


class Yolo26PoseTensorOperator(BatchMetadataOperator):
    """Convert each frame's YOLO26 tensor into object-scoped DeepStream metadata."""

    def __init__(self, schema: PoseSchema, store: FramePoseStore):
        super().__init__()
        self.schema = schema
        self.store = store

    @staticmethod
    def _rows(layer) -> list[list[float]]:
        import torch

        tensor = torch.utils.dlpack.from_dlpack(layer).detach().to(device="cpu", dtype=torch.float32)
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"expected a two-dimensional YOLO26 output, got {tuple(tensor.shape)}")
        return tensor.tolist()

    def handle_metadata(self, batch_meta) -> None:
        for frame_meta in batch_meta.frame_items:
            stream_id = int(getattr(frame_meta, "source_id", frame_meta.pad_index))
            source_width = int(frame_meta.source_width or frame_meta.pipeline_width)
            source_height = int(frame_meta.source_height or frame_meta.pipeline_height)
            found = False
            for tensor_meta in frame_meta.tensor_items:
                layers = tensor_meta.as_tensor_output().get_layers()
                layer = layers.get(self.schema.output_layer)
                if layer is None and len(layers) == 1:
                    layer = next(iter(layers.values()))
                if layer is None:
                    continue
                found = True
                observations = decode_yolo26_rows(
                    self._rows(layer), self.schema,
                    source_width=source_width, source_height=source_height,
                )
                for observation in observations:
                    observation["detector_frame_number"] = int(frame_meta.frame_number)
                    object_meta = batch_meta.acquire_object_meta()
                    object_meta.class_id = observation["class_id"]
                    object_meta.confidence = observation["detector_confidence"]
                    # NvDsObjectMeta acquired from the pool defaults to zero, which is a
                    # valid track ID. Mark detector objects explicitly as untracked so
                    # nvtracker assigns trajectory IDs instead of treating every object
                    # as track zero.
                    object_meta.object_id = (1 << 64) - 1
                    object_meta.unique_component_id = 1
                    object_meta.label = self.store.put(
                        stream_id, int(frame_meta.frame_number), observation
                    )
                    bbox = observation["detector_bbox"]
                    object_meta.rect_params.left = bbox["x"]
                    object_meta.rect_params.top = bbox["y"]
                    object_meta.rect_params.width = bbox["w"]
                    object_meta.rect_params.height = bbox["h"]
                    frame_meta.append(object_meta)
            if not found:
                raise RuntimeError(
                    f"YOLO26 tensor layer {self.schema.output_layer!r} is missing on frame "
                    f"{frame_meta.frame_number}"
                )


class ObservationOperator(BatchMetadataOperator):
    """Persist tracked objects/keypoints and produce the matching live overlay."""

    OBJECT_HEADERS = [
        "observation_id", "stream_id", "deepstream_frame_number", "source_sequence_index",
        "camera_frame_id", "camera_timestamp_ns", "gst_pts_ns", "class_id", "class_label",
        "track_id", "detected_this_frame", "tracker_predicted", "detector_confidence",
        "tracker_confidence", "detector_x", "detector_y", "detector_w", "detector_h",
        "track_x", "track_y", "track_w", "track_h", "pose_available", "schema_version",
    ]
    KEYPOINT_HEADERS = [
        "observation_id", "stream_id", "deepstream_frame_number", "source_sequence_index",
        "camera_frame_id", "track_id", "class_id", "class_label", "keypoint_index",
        "keypoint_name", "x_px", "y_px", "x_norm", "y_norm", "confidence", "visible",
        "coordinate_space", "source",
    ]
    def __init__(
        self,
        run_dir: Path,
        schema: PoseSchema,
        *,
        store: FramePoseStore,
        flir_meta_type: int | None,
        frame_ledger: dict[int, dict[str, Any]] | None = None,
        mapping_method: str = "flir_user_meta",
        source_name: str = "flirspinsrc",
    ):
        super().__init__()
        self.schema = schema
        self.store = store
        self.flir_meta_type = None if flir_meta_type is None else int(flir_meta_type)
        self.frame_ledger = frame_ledger or {}
        self.mapping_method = str(mapping_method)
        self.source_name = str(source_name)
        self._lock = threading.Lock()
        self._closed = False
        self._files: dict[str, Any] = {}
        self._writers: dict[str, csv.writer] = {}
        for name, headers in (
            ("objects", self.OBJECT_HEADERS),
            ("keypoints", self.KEYPOINT_HEADERS),
        ):
            handle = (run_dir / f"{name}.csv").open("w", newline="", buffering=1)
            self._files[name] = handle
            self._writers[name] = csv.writer(handle)
            self._writers[name].writerow(headers)
        atexit.register(self.close)

    @staticmethod
    def _json_meta(owner, meta_type: int) -> dict[str, Any] | None:
        try:
            items = owner.user_meta_items(meta_type)
        except Exception:
            return None
        for item in items:
            try:
                value = item.get_user_data_json()
                if isinstance(value, str):
                    value = json.loads(value)
                if isinstance(value, dict):
                    return value
            except Exception:
                continue
        return None

    @staticmethod
    def _track_id(object_meta) -> int | None:
        value = int(getattr(object_meta, "object_id", -1))
        return None if value in UNTRACKED_OBJECT_IDS else value

    @staticmethod
    def _color(track_id: int | None, class_id: int):
        palette = (
            (0.13, 0.82, 0.94), (0.96, 0.45, 0.18), (0.55, 0.88, 0.24),
            (0.75, 0.42, 0.95), (0.98, 0.78, 0.16), (0.20, 0.72, 0.42),
        )
        r, g, b = palette[(track_id if track_id is not None else class_id) % len(palette)]
        return osd.Color(r, g, b, 1.0)

    def _decorate(self, batch_meta, frame_meta, object_meta, pose, track_id, predicted) -> None:
        color = self._color(track_id, int(object_meta.class_id))
        object_meta.rect_params.border_width = 1 if predicted else 2
        object_meta.rect_params.border_color = color
        spec = self.schema.class_spec(int(object_meta.class_id))
        label = spec.name if spec else f"class_{object_meta.class_id}"
        track_label = f" T{track_id}" if track_id is not None else ""
        confidence = pose.get("detector_confidence") if pose else object_meta.tracker_confidence
        try:
            text = object_meta.text_params
            text.display_text = f"{label}{track_label} {float(confidence):.2f}".encode("ascii")
            text.x_offset = max(0, int(object_meta.rect_params.left))
            text.y_offset = max(0, int(object_meta.rect_params.top) - 18)
            text.font.name = osd.FontFamily.Serif
            text.font.size = 11
            text.font.color = color
            text.set_bg_color = True
            text.bg_color = osd.Color(0.0, 0.0, 0.0, 0.65)
        except Exception:
            pass
        if not pose:
            return
        visible = {
            int(point["index"]): point for point in pose.get("keypoints", []) if point.get("visible")
        }
        circles = []
        for point in visible.values():
            circle = osd.Circle()
            circle.xc = int(round(point["x"]))
            circle.yc = int(round(point["y"]))
            circle.radius = 3
            circle.width = 2
            circle.color = color
            circle.has_bg_color = True
            circle.bg_color = color
            circles.append(circle)
        for start in range(0, len(circles), 16):
            display = batch_meta.acquire_display_meta()
            for circle in circles[start:start + 16]:
                display.add_circle(circle)
            frame_meta.append(display)

    def handle_metadata(self, batch_meta) -> None:
        with self._lock:
            for frame_meta in batch_meta.frame_items:
                frame_number = int(frame_meta.frame_number)
                stream_id = int(getattr(frame_meta, "source_id", frame_meta.pad_index))
                pts_ns = int(getattr(frame_meta, "buffer_pts", 0) or 0)
                flir = (
                    self._json_meta(frame_meta, self.flir_meta_type)
                    if self.flir_meta_type is not None
                    else None
                ) or self.frame_ledger.get(frame_number, {})
                source_sequence = flir.get("source_sequence_index")
                if source_sequence in (None, ""):
                    source_sequence = flir.get("raw_frame_index")
                camera_frame_id = flir.get("camera_frame_id")
                camera_timestamp = (
                    flir.get("transport_timestamp_ns")
                    if flir.get("transport_timestamp_ns") not in (None, "")
                    else flir.get("camera_timestamp_ns")
                )
                ledger_pts = flir.get("pts_ns")
                if ledger_pts not in (None, ""):
                    pts_ns = int(ledger_pts)
                source_width = int(frame_meta.source_width or frame_meta.pipeline_width)
                source_height = int(frame_meta.source_height or frame_meta.pipeline_height)
                for ordinal, object_meta in enumerate(frame_meta.object_items):
                    class_id = int(object_meta.class_id)
                    spec = self.schema.class_spec(class_id)
                    label = spec.name if spec else f"class_{class_id}"
                    track_id = self._track_id(object_meta)
                    pose = self.store.get(
                        stream_id, frame_number, str(getattr(object_meta, "label", ""))
                    )
                    detected = bool(pose and pose.get("detector_frame_number") == frame_number)
                    predicted = not detected
                    rect = object_meta.rect_params
                    detector_bbox = pose.get("detector_bbox", {}) if pose else {}
                    observation_id = f"s{stream_id}:f{source_sequence if source_sequence is not None else frame_number}:o{ordinal}"
                    detector_confidence = pose.get("detector_confidence", "") if pose else ""
                    tracker_confidence = float(getattr(object_meta, "tracker_confidence", -0.1))
                    self._writers["objects"].writerow(
                        [
                            observation_id, stream_id, frame_number,
                            "" if source_sequence is None else source_sequence,
                            "" if camera_frame_id is None else camera_frame_id,
                            "" if camera_timestamp is None else camera_timestamp,
                            pts_ns, class_id, label, "" if track_id is None else track_id,
                            int(detected), int(predicted), detector_confidence, tracker_confidence,
                            detector_bbox.get("x", ""), detector_bbox.get("y", ""),
                            detector_bbox.get("w", ""), detector_bbox.get("h", ""),
                            float(rect.left), float(rect.top), float(rect.width), float(rect.height),
                            int(pose is not None), pose.get("schema_version", "") if pose else "",
                        ]
                    )
                    if pose:
                        for point in pose.get("keypoints", []):
                            self._writers["keypoints"].writerow(
                                [
                                    observation_id, stream_id, frame_number,
                                    "" if source_sequence is None else source_sequence,
                                    "" if camera_frame_id is None else camera_frame_id,
                                    "" if track_id is None else track_id, class_id, label,
                                    point["index"], point["name"], point["x"], point["y"],
                                    point["x"] / source_width, point["y"] / source_height,
                                    point["confidence"], int(point["visible"]),
                                    pose.get("coordinate_space", "source_pixels"), "detector",
                                ]
                            )
                    self._decorate(batch_meta, frame_meta, object_meta, pose, track_id, predicted)
                self.store.discard(stream_id, frame_number)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for handle in self._files.values():
                handle.flush()
                handle.close()
