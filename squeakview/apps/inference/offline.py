"""Offline DeepStream replay of an immutable SqueakView ground-truth recording."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import signal
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pyservicemaker import BatchMetadataOperator, EOSMessage, Pipeline, Probe

from squeakview.apps.inference.pose_pipeline import (
    FramePoseStore,
    ObservationOperator,
    Yolo26PoseTensorOperator,
    load_pose_schema,
)
from squeakview.apps.inference.service_maker_runner import _load_class_names
from squeakview.model_package import validate_model_package


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _int_value(row: dict[str, str], name: str) -> int | None:
    raw = str(row.get(name, "")).strip()
    if not raw:
        return None
    return int(float(raw))


def _video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=False, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe did not return a frame count for {path}") from exc


def _load_frame_ledger(path: Path) -> tuple[dict[int, dict[str, Any]], int, int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"frame ledger is empty: {path}")
    streams = {_int_value(row, "stream_id") or 0 for row in rows}
    if streams != {0}:
        raise ValueError("offline replay currently requires a single-camera frame ledger")
    ledger: dict[int, dict[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        source_sequence = _int_value(row, "source_sequence_index")
        raw_index = _int_value(row, "raw_frame_index")
        ledger[ordinal] = {
            "source_sequence_index": source_sequence if source_sequence is not None else raw_index,
            "raw_frame_index": raw_index,
            "camera_frame_id": _int_value(row, "camera_frame_id"),
            "camera_timestamp_ns": _int_value(row, "camera_timestamp_ns"),
            "pts_ns": _int_value(row, "pts_ns"),
        }
    first = rows[0]
    width = _int_value(first, "source_width")
    height = _int_value(first, "source_height")
    if not width or not height:
        raise ValueError("frames.csv must contain source_width and source_height for offline replay")
    return ledger, width, height


def _resolve_default_config(run_dir: Path) -> Path:
    manifest = _read_json(run_dir / "run_manifest.json")
    try:
        raw = manifest["inference"]["model_package"]["config"]
    except (KeyError, TypeError) as exc:
        raise ValueError("run manifest has no inference.model_package.config; pass --cfg") from exc
    return Path(str(raw)).expanduser().resolve()


class FrameAuditOperator(BatchMetadataOperator):
    """Assert that decoded frame ordinals are contiguous and count every replayed frame."""

    def __init__(self, expected: int):
        super().__init__()
        self.expected = int(expected)
        self.count = 0
        self._lock = threading.Lock()

    def handle_metadata(self, batch_meta) -> None:
        with self._lock:
            for frame_meta in batch_meta.frame_items:
                actual = int(frame_meta.frame_number)
                if actual != self.count:
                    raise RuntimeError(
                        f"offline decoded-frame sequence is not contiguous: expected {self.count}, got {actual}"
                    )
                if actual >= self.expected:
                    raise RuntimeError(
                        f"offline decoder produced frame {actual} beyond the {self.expected}-row ledger"
                    )
                self.count += 1


@dataclass(slots=True)
class OfflineConfig:
    run_dir: Path
    cfg_path: Path | None = None
    out_dir: Path | None = None


class OfflineInferenceApp:
    """Replay raw.mp4 through the live TensorRT decoder and NvDCF tracker."""

    def __init__(
        self,
        config: OfflineConfig,
        *,
        pipeline_factory=Pipeline,
        probe_factory=Probe,
    ):
        self.config = config
        self.run_dir = Path(config.run_dir).expanduser().resolve()
        self.video_path = self.run_dir / "raw.mp4"
        self.frames_path = self.run_dir / "frames.csv"
        if not self.video_path.is_file() or not self.frames_path.is_file():
            raise FileNotFoundError("offline replay requires raw.mp4 and frames.csv in the run directory")
        self.cfg_path = (
            Path(config.cfg_path).expanduser().resolve()
            if config.cfg_path is not None
            else _resolve_default_config(self.run_dir)
        )
        self.model = validate_model_package(self.cfg_path)
        self.tracker_path = Path(__file__).resolve().parents[3] / "configs" / "tracker_mouse_nvdcf.yml"
        self.source_hashes = {
            "video_sha256": _sha256(self.video_path),
            "frames_sha256": _sha256(self.frames_path),
            "parser_sha256": _sha256(self.model.parser_library),
            "tracker_sha256": _sha256(self.tracker_path),
        }
        self.ledger, self.width, self.height = _load_frame_ledger(self.frames_path)
        video_frames = _video_frame_count(self.video_path)
        if video_frames != len(self.ledger):
            raise ValueError(
                f"ground-truth mismatch: raw.mp4 has {video_frames} frames but frames.csv has {len(self.ledger)}"
            )

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = (
            Path(config.out_dir).expanduser().resolve()
            if config.out_dir is not None
            else self.run_dir / "offline_inference" / f"{stamp}_{self.model.name}"
        )
        if self.out_dir.exists() and any(self.out_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output directory: {self.out_dir}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_factory = pipeline_factory
        self.probe_factory = probe_factory
        self.pipeline = None
        self.observations = None
        self.audit = FrameAuditOperator(len(self.ledger))
        self.stop_event = threading.Event()
        self.eos_received = False
        self.exit_code = 0
        self.started_at = datetime.now().astimezone().isoformat()
        self._write_manifest("starting")

    def _write_manifest(self, status: str, error: str | None = None) -> None:
        outputs = {}
        for name in ("objects.csv", "keypoints.csv"):
            path = self.out_dir / name
            if path.exists():
                with path.open(newline="") as handle:
                    count = max(0, sum(1 for _ in handle) - 1)
                outputs[name] = {"rows": count, "sha256": _sha256(path)}
        payload = {
            "schema_version": 1,
            "status": status,
            "started_at": self.started_at,
            "finished_at": datetime.now().astimezone().isoformat() if status != "starting" else None,
            "source": {
                "run_dir": str(self.run_dir),
                "video": str(self.video_path),
                "video_sha256": self.source_hashes["video_sha256"],
                "frames": str(self.frames_path),
                "frames_sha256": self.source_hashes["frames_sha256"],
                "expected_frames": len(self.ledger),
            },
            "model_package": self.model.manifest_snapshot(),
            "runtime_artifacts": {
                "parser_library": str(self.model.parser_library),
                "parser_sha256": self.source_hashes["parser_sha256"],
                "tracker_config": str(self.tracker_path),
                "tracker_sha256": self.source_hashes["tracker_sha256"],
            },
            "mapping": "decoded frame ordinal joined to authoritative frames.csv row ordinal",
            "decoded_frames": self.audit.count,
            "outputs": outputs,
            "error": error,
        }
        target = self.out_dir / "offline_manifest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)

    def build(self):
        pipeline = self.pipeline_factory("squeakview-offline")
        pipeline.add(
            "nvurisrcbin", "source",
            {"uri": self.video_path.as_uri(), "disable-audio": True, "file-loop": False,
             "dec-skip-frames": 0, "drop-frame-interval": 0, "leaky": 0,
             "drop-on-latency": False},
        )
        pipeline.add(
            "nvstreammux", "mux",
            {
                "batch-size": 1, "width": self.width, "height": self.height,
                "live-source": False, "sync-inputs": False,
            },
        )
        pipeline.link(("source", "mux"), ("", "sink_%u"))
        pipeline.attach("mux", self.probe_factory("offline_frame_audit", self.audit))

        class_names = _load_class_names(self.cfg_path)
        schema = load_pose_schema(self.cfg_path, class_names)
        store = FramePoseStore()
        pipeline.add(
            "nvinfer", "infer",
            {
                "config-file-path": str(self.cfg_path), "batch-size": 1,
                "filter-out-class-ids": ";".join(str(item.class_id) for item in schema.classes),
            },
        )
        pipeline.attach("infer", self.probe_factory("yolo26_pose", Yolo26PoseTensorOperator(schema, store)))
        pipeline.add(
            "nvtracker", "tracker",
            {
                "tracker-width": 640, "tracker-height": 480,
                "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                "ll-config-file": str(self.tracker_path),
                "operate-on-class-ids": ";".join(
                    str(item.class_id) for item in schema.classes if item.track
                ),
                "display-tracking-id": False, "tracking-id-reset-mode": 3,
            },
        )
        self.observations = ObservationOperator(
            self.out_dir, schema, store=store, flir_meta_type=None,
            frame_ledger=self.ledger,
            mapping_method="offline_video_ledger", source_name="offline_raw_mp4",
        )
        pipeline.attach("tracker", self.probe_factory("observations", self.observations))
        pipeline.add("fakesink", "sink", {"sync": False, "async": False})
        pipeline.link("mux", "infer", "tracker", "sink")
        self.pipeline = pipeline
        return pipeline

    def _on_message(self, message) -> None:
        if isinstance(message, EOSMessage):
            self.eos_received = True
            self.stop_event.set()

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> int:
        if self.pipeline is None:
            self.build()
        try:
            self.pipeline.start(self._on_message)
            while not self.stop_event.wait(0.2):
                pass
        except KeyboardInterrupt:
            self.exit_code = 130
        except Exception as exc:
            self.exit_code = 1
            self._write_manifest("failed", str(exc))
            print(f"[OFFLINE] failed: {exc}", flush=True)
        finally:
            if self.pipeline is not None:
                try:
                    if self.eos_received:
                        self.pipeline.wait()
                    else:
                        self.pipeline.stop()
                        self.pipeline.wait()
                except Exception as exc:
                    self.exit_code = self.exit_code or 1
                    self._write_manifest("failed", str(exc))
            if self.observations is not None:
                self.observations.close()

        if self.exit_code == 0 and self.audit.count != len(self.ledger):
            self.exit_code = 1
            error = f"decoded {self.audit.count} frames; expected {len(self.ledger)}"
            self._write_manifest("failed", error)
            print(f"[OFFLINE] {error}", flush=True)
        elif self.exit_code == 0:
            try:
                from scripts.align_run_outputs_streaming import build_alignment
                build_alignment(
                    self.run_dir, self.out_dir,
                    objects_path=self.out_dir / "objects.csv",
                )
            except Exception as exc:
                self.exit_code = 1
                self._write_manifest("failed", f"alignment failed: {exc}")
                print(f"[OFFLINE] alignment failed: {exc}", flush=True)
            else:
                self._write_manifest("complete")
                print(f"[OFFLINE] complete: {self.out_dir}", flush=True)
        return self.exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a SqueakView raw.mp4 through the live DeepStream model and tracker"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cfg", type=Path, default=None, help="Model config; defaults to run manifest model")
    parser.add_argument("--out-dir", type=Path, default=None, help="New, empty derived-output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = OfflineInferenceApp(
        OfflineConfig(
            run_dir=args.run_dir, cfg_path=args.cfg, out_dir=args.out_dir,
        )
    )

    def handle_signal(_signal, _frame) -> None:
        app.request_stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
