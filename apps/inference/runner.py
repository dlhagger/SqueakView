from __future__ import annotations

"""DeepStream live inference runner extracted from the legacy script."""

import atexit
import csv
import ctypes
import json
import math
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GObject", "2.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GObject, GLib, GstVideo  # type: ignore

try:
    import pyds  # DeepStream bindings
except Exception as exc:  # pragma: no cover - environment-specific
    raise RuntimeError("DeepStream Python bindings (pyds) are required") from exc

from squeakview.common import run_context

_GST_DEBUG_ENV = "SQUEAKVIEW_GST_DEBUG"
_DS_DEBUG_ENV = "SQUEAKVIEW_DS_DEBUG"
_SURF_DEBUG_ENV = "SQUEAKVIEW_SURF_DEBUG"  # set to 0 to disable surface/caps debug


def _maybe_enable_debug_logging() -> None:
    """Opt-in knobs to get deeper DeepStream/GStreamer logging for diagnosis."""
    gst_flag = os.environ.get(_GST_DEBUG_ENV)
    if gst_flag and "GST_DEBUG" not in os.environ:
        # Focus on nvinfer + surf transform; allow override by setting GST_DEBUG yourself.
        os.environ["GST_DEBUG"] = "nvinfer:5,nvbufsurftransform:5,nvstreammux:4,queue:3"
        print(f"[{ts()}] [DEBUG] GST_DEBUG enabled ({os.environ['GST_DEBUG']})", flush=True)
    ds_flag = os.environ.get(_DS_DEBUG_ENV)
    if ds_flag:
        # Default to verbose infer logging unless user already set them.
        os.environ.setdefault("NVDSINFER_LOG_LEVEL", "5")
        os.environ.setdefault("NVDSINFER_DEBUG", "1")
        print(
            f"[{ts()}] [DEBUG] NVDSINFER_LOG_LEVEL={os.environ['NVDSINFER_LOG_LEVEL']} "
            f"NVDSINFER_DEBUG={os.environ['NVDSINFER_DEBUG']}",
            flush=True,
        )


def ts() -> str:
    return time.strftime("%H:%M:%S")


def _read_rss_kb() -> int:
    """Return current process RSS in KB using /proc (no extra deps)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1])  # already in kB
    except Exception:
        pass
    return -1


def _fd_count() -> int:
    """Return number of open fds for current process."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return -1

class CSVWriter:
    def __init__(
        self,
        run_dir: Path,
        *,
        pose: bool = False,
        kpt_count: int = 0,
        kpt_dims: int = 3,
        kp_names: list[str] | None = None,
    ):
        self.path = run_dir / "detections.csv"
        self.pose_enabled = bool(pose)
        self.kpt_count = int(kpt_count)
        self.kpt_dims = int(kpt_dims)
        self.kp_names = kp_names if (kp_names and len(kp_names) >= self.kpt_count) else None
        self._f = self.path.open("w", newline="", buffering=1)
        self._writer = csv.writer(self._f)
        self._flush_every = 25
        self._row_count = 0
        headers = [
            "frame",
            "raw_frame_num",
            "ts_us",
            "stream_id",
            "source",
            "obj_id",
            "class_id",
            "class_label",
            "conf",
            "x",
            "y",
            "w",
            "h",
        ]
        if self.pose_enabled:
            headers.extend(
                [
                    "pose_schema",
                    "kpt_count",
                    "kpt_names_json",
                    "kpt_values_json",
                ]
            )
        self._writer.writerow(headers)
        self._f.flush()
        atexit.register(self.close)

    def row(self, values: list[object]) -> None:
        self._writer.writerow(values)
        self._row_count += 1
        if self._row_count % self._flush_every == 0:
            try:
                self._f.flush()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass


@dataclass(slots=True)
class PoseSchema:
    name: str
    offset: int
    names: list[str]


def _safe_object_id(ometa) -> int:
    """Return integer object_id or -1 across DS versions."""
    try:
        oid = int(getattr(ometa, "object_id", -1))
    except Exception:
        return -1
    try:
        invalid_const = getattr(pyds, "NVDS_OBJECT_ID_INVALID", -1)
    except Exception:
        invalid_const = -1
    return oid if oid != invalid_const else -1


@dataclass(slots=True)
class InferenceConfig:
    sock: str = "/tmp/cam.sock"
    socks: list[str] | None = None
    cfg_path: Path | None = field(default_factory=lambda: Path.cwd() / "config_infer_primary_11m.txt")
    capture_backend: str = "flir"
    num_cameras: int = 1
    zed_depth_enabled: bool = False
    zed_depth_mode: str = "NEURAL"
    zed_depth_socket: str = "/tmp/cam_depth.sock"
    zed_depth_record: bool = False
    zed_confidence_threshold: int = 100
    zed_texture_confidence_threshold: int = 100
    zed_depth_minimum_distance_mm: int = 300
    zed_depth_maximum_distance_mm: int = 20000
    zed_fill_mode: bool = False
    zed_depth_stabilization: int = 30
    width: int = 1280
    height: int = 720
    fps: int = 30
    bitrate: int = 4000
    window_xid: int | None = None
    enable_infer: bool = True
    run_dir: Path | None = None
    draw_skeleton: bool = False


class App:
    def __init__(self, config: InferenceConfig):
        self.config = config
        _maybe_enable_debug_logging()
        self.enable_infer = bool(config.enable_infer)
        if config.run_dir is not None:
            self.run_dir = Path(config.run_dir).expanduser()
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir = run_context.timestamped_run_dir("ds")
        self.artifacts = run_context.run_artifacts(self.run_dir)
        self.capture_backend = str(getattr(config, "capture_backend", "flir") or "flir").lower().strip()
        self.requested_cam_count = max(1, int(getattr(config, "num_cameras", 1)))
        self.zed_depth_enabled = bool(getattr(config, "zed_depth_enabled", False))
        self.zed_depth_mode = str(getattr(config, "zed_depth_mode", "NEURAL")).upper().strip()
        self.zed_depth_socket = str(getattr(config, "zed_depth_socket", "/tmp/cam_depth.sock"))
        self.zed_depth_record = bool(getattr(config, "zed_depth_record", False))
        self.zed_confidence_threshold = max(
            0, min(100, int(getattr(config, "zed_confidence_threshold", 100) or 100))
        )
        self.zed_texture_confidence_threshold = max(
            0, min(100, int(getattr(config, "zed_texture_confidence_threshold", 100) or 100))
        )
        self.zed_depth_minimum_distance_mm = max(
            100, min(3000, int(getattr(config, "zed_depth_minimum_distance_mm", 300) or 300))
        )
        self.zed_depth_maximum_distance_mm = max(
            500, min(40000, int(getattr(config, "zed_depth_maximum_distance_mm", 20000) or 20000))
        )
        self.zed_fill_mode = bool(getattr(config, "zed_fill_mode", False))
        self.zed_depth_stabilization = max(
            0, min(100, int(getattr(config, "zed_depth_stabilization", 30) or 30))
        )
        raw_socks = list(config.socks or [])
        if not raw_socks:
            raw_socks = [config.sock]
        if config.sock not in raw_socks:
            raw_socks.insert(0, config.sock)
        self.socks = [s for s in raw_socks if s] if self.capture_backend != "zed" else []
        self._extra_raw_videos: list[Path] = []
        self._depth_video: Path | None = None
        self._zed_svo_file: Path | None = None
        self._zed_svo_compression = "H264"
        self._zed_svo_enable_attempts = 0
        self._zed_svo_enabled_runtime = False
        self.preview_enabled = True
        if self.enable_infer:
            self.pose_mode, self.pose_kpt_count, self.pose_input_dims = self._detect_pose_mode(config)
        else:
            self.pose_mode, self.pose_kpt_count, self.pose_input_dims = False, 0, (float(config.width), float(config.height))
        self.pose_kpt_dims = 3 if self.pose_mode else 0
        self.draw_skeleton = bool(config.draw_skeleton)
        self.kp_names: list[str] | None = None
        self.class_names: list[str] = []
        self.pose_schemas_by_name: dict[str, PoseSchema] = {}
        self.pose_schemas_by_id: dict[int, PoseSchema] = {}
        if self.pose_mode:
            # Load labels file (pose models use labels as keypoint names)
            try:
                lines = Path(config.cfg_path).read_text().splitlines()
                label_path = None
                kp_label_path = None
                for line in lines:
                    raw = line.strip()
                    if not raw or raw.startswith("#"):
                        continue
                    if raw.startswith("labelfile-path"):
                        try:
                            label_path = raw.split("=", 1)[1].strip().strip('"')
                        except Exception:
                            label_path = None
                    if raw.startswith("pose-kpt-labels-path"):
                        try:
                            kp_label_path = raw.split("=", 1)[1].strip().strip('"')
                        except Exception:
                            kp_label_path = None
                # keypoint labels: prefer explicit pose-kpt-labels-path, fallback to labelfile-path
                if label_path:
                    lp = Path(label_path)
                    if lp.exists():
                        self.class_names = [ln.strip() for ln in lp.read_text().splitlines() if ln.strip()]
                kp_label_path = kp_label_path or label_path
                if kp_label_path:
                    self.kp_names = self._load_pose_labels(Path(kp_label_path))
            except Exception as exc:
                print(f"[{ts()}] [POSE] unable to load keypoint labels: {exc}")
        # Skeleton toggle via file
        self.skeleton_ctrl_path = self.run_dir / "skeleton_toggle.txt"
        try:
            self.skeleton_ctrl_path.write_text("on" if self.draw_skeleton else "off")
        except Exception:
            pass
        self.csv: CSVWriter | None = None
        if self.enable_infer:
            self.csv = CSVWriter(
                self.run_dir,
                pose=self.pose_mode,
                kpt_count=self.pose_kpt_count,
                kpt_dims=self.pose_kpt_dims,
                kp_names=self.kp_names if self.pose_mode else None,
            )
        self.perf_path = self.run_dir / "perf_stats.csv"
        self._perf_file = self.perf_path.open("w", newline="", buffering=1)
        self._perf_writer = csv.writer(self._perf_file)
        self._perf_writer.writerow(["timestamp", "streaming_fps", "inference_fps", "latency_ms"])
        self.pipeline = None
        self.loop = GLib.MainLoop()
        self._stopping = False
        self._preview_valve = None
        self.video_enabled = True
        self._window_handle = config.window_xid
        self._infer_starts: dict[int, float] = {}
        self._perf_window_sec = 5.0
        self._infer_history: deque[tuple[float, float]] = deque()
        self._stream_history: deque[float] = deque()
        self._stream_fps: float = float("nan")
        self._stream_frame_seq: dict[int, int] = {}
        self._raw_frame_offsets = self._load_raw_frame_offsets()
        self._single_cam_aligned_raw = False
        self._csv_probe_pre_tiler: bool = False
        self._pose_cache_seq: int = 0
        self._pose_cache_last: list[dict] | None = None
        self._pose_cache_fn = None
        self._pose_cache_lib = None
        # Draw all keypoints by default; can be overridden via pose-draw-threshold in the config
        self.pose_draw_score_thresh = self._load_pose_draw_thresh(config.cfg_path) if self.pose_mode else 0.0
        self.pose_draw_radius = 8
        # Allow forcing CPU OSD via env (default GPU)
        self.osd_cpu = os.environ.get("SQUEAKVIEW_OSD_CPU") == "1"
        # Debug inference logging is on by default; set SQUEAKVIEW_DEBUG_INFER=0 to disable.
        env_debug = os.environ.get("SQUEAKVIEW_DEBUG_INFER")
        self.debug_infer = False if env_debug == "0" else True
        if self.debug_infer:
            print(f"[{ts()}] [DEBUG] Inference debug logging enabled", flush=True)
        # Surface/caps logging (limited to a few frames).
        self._caps_logged = False
        self._pgie_caps_logged = False
        self._mux_sink_caps_logged = False
        self._surface_logs = 0
        # Enabled by default (unless SQUEAKVIEW_SURF_DEBUG=0); limit via SQUEAKVIEW_SURF_DEBUG_LIMIT.
        self._surf_debug_enabled = os.environ.get(_SURF_DEBUG_ENV, "1") != "0"
        try:
            self._surf_debug_limit = int(os.environ.get("SQUEAKVIEW_SURF_DEBUG_LIMIT", "15"))
        except Exception:
            self._surf_debug_limit = 15
        GLib.timeout_add_seconds(1, self._poll_skeleton_toggle)
        if self.pose_mode:
            self._init_pose_cache_helper()
        print(f"[{ts()}] [INFO] raw frame offsets: {self._raw_frame_offsets}", flush=True)
        print(f"[{ts()}] [INFO] run dir:      {self.run_dir}", flush=True)
        print(f"[{ts()}] [INFO] run marker:   {run_context.RUN_MARKER}", flush=True)
        atexit.register(self._atexit_cleanup)

    @staticmethod
    def _load_raw_frame_offsets() -> list[int]:
        """Return per-stream offsets mapping DeepStream frame numbers to raw video frames."""
        raw = (os.environ.get("SQUEAKVIEW_RAW_FRAME_OFFSET") or "16").strip()
        if not raw:
            return [16]
        parts = [part.strip() for part in raw.split(",")]
        offsets: list[int] = []
        for part in parts:
            if not part:
                continue
            try:
                offsets.append(int(part))
            except ValueError:
                print(f"[{ts()}] [WARN] invalid raw frame offset '{part}', using 16", flush=True)
                offsets.append(16)
        return offsets or [16]

    def _raw_frame_num_for_stream(self, ds_frame_num: int, stream_id: int) -> int:
        if self._single_cam_aligned_raw:
            return int(ds_frame_num)
        offsets = self._raw_frame_offsets
        index = max(0, min(int(stream_id), len(offsets) - 1))
        return int(ds_frame_num) + int(offsets[index])

    @staticmethod
    def _detect_pose_mode(config: InferenceConfig) -> tuple[bool, int, tuple[float, float]]:
        if not config.cfg_path:
            return False, 0, (float(config.width), float(config.height))
        try:
            text = Path(config.cfg_path).read_text().splitlines()
        except Exception:
            return False, 0, (float(config.width), float(config.height))
        parser_name = ""
        net_dims = None
        label_path = None
        for line in text:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("infer-dims"):
                try:
                    parts = [p.strip() for p in line.split("=", 1)[1].split(";")]
                    if len(parts) >= 3:
                        net_h = float(parts[-2])
                        net_w = float(parts[-1])
                        net_dims = (net_w, net_h)
                except Exception:
                    net_dims = None
            if line.startswith("parse-bbox-func-name"):
                parser_name = line.split("=", 1)[1].strip()
            if line.startswith("labelfile-path"):
                try:
                    label_path = line.split("=", 1)[1].strip().strip('"')
                except Exception:
                    label_path = None
            if parser_name and net_dims and label_path:
                break
        if not parser_name:
            return False, 0, net_dims or (float(config.width), float(config.height))
        parser_lower = parser_name.lower()
        if "pose" not in parser_lower:
            return False, 0, net_dims or (float(config.width), float(config.height))
        # Pose mode enabled; defer keypoint count to runtime decode (from parser cache).
        return True, 0, net_dims or (float(config.width), float(config.height))

    @staticmethod
    def _load_pose_draw_thresh(cfg_path: Path | None) -> float:
        if not cfg_path:
            return 0.0
        try:
            lines = Path(cfg_path).read_text().splitlines()
        except Exception:
            return 0.0
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.lower().startswith("pose-draw-threshold"):
                try:
                    return float(raw.split("=", 1)[1].strip())
                except Exception:
                    return 0.0
        return 0.0

    @staticmethod
    def _load_cfg_batch_size(cfg_path: Path | None) -> int | None:
        if not cfg_path:
            return None
        try:
            lines = Path(cfg_path).read_text().splitlines()
        except Exception:
            return None
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.lower().startswith("batch-size"):
                try:
                    return int(raw.split("=", 1)[1].strip())
                except Exception:
                    return None
        return None

    def _load_pose_labels(self, labels_path: Path) -> list[str] | None:
        try:
            lines = labels_path.read_text().splitlines()
        except Exception:
            return None

        plain_names: list[str] = []
        section_name: str | None = None
        section_names: list[str] = []
        section_offset = 0
        saw_section = False

        def _commit_section() -> None:
            nonlocal section_name, section_names, section_offset
            if not section_name or not section_names:
                return
            schema = PoseSchema(name=section_name, offset=section_offset, names=list(section_names))
            self.pose_schemas_by_name[section_name.lower()] = schema
            section_offset += len(section_names)
            section_names = []

        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("#"):
                continue
            if raw.startswith("[") and raw.endswith("]") and len(raw) > 2:
                saw_section = True
                _commit_section()
                section_name = raw[1:-1].strip()
                continue
            if saw_section:
                section_names.append(raw)
            else:
                plain_names.append(raw)

        if saw_section:
            _commit_section()
            if self.class_names:
                for idx, class_name in enumerate(self.class_names):
                    schema = self.pose_schemas_by_name.get(class_name.strip().lower())
                    if schema:
                        self.pose_schemas_by_id[idx] = schema
            ordered: list[str] = []
            for schema in sorted(self.pose_schemas_by_name.values(), key=lambda s: s.offset):
                ordered.extend(schema.names)
            return ordered or None

        return plain_names or None

    def _pose_schema_for_detection(self, class_id: int, label: str) -> PoseSchema | None:
        schema = self.pose_schemas_by_id.get(class_id)
        if schema:
            return schema
        norm = (label or "").strip().lower()
        if norm:
            schema = self.pose_schemas_by_name.get(norm)
            if schema:
                return schema
        if 0 <= class_id < len(self.class_names):
            return self.pose_schemas_by_name.get(self.class_names[class_id].strip().lower())
        return None

    def _effective_cam_count(self) -> int:
        if self.capture_backend != "zed":
            return max(1, len(self.socks))
        if self.zed_depth_enabled and self.requested_cam_count > 1:
            print(
                f"[{ts()}] [WARN] ZED depth mode currently uses one DeepStream source (left image only). "
                "Set batch-size=1 config for this mode.",
                flush=True,
            )
            return 1
        return 2 if self.requested_cam_count > 1 else 1

    @staticmethod
    def _zed_resolution_enum(width: int, height: int) -> int:
        dims = (int(width), int(height))
        if dims == (2208, 1242):
            return 0  # HD2K
        if dims == (1920, 1080):
            return 1  # HD1080
        if dims == (1920, 1200):
            return 2  # HD1200
        if dims == (1280, 720):
            return 3  # HD720
        if dims == (960, 600):
            return 4  # SVGA
        if dims == (672, 376):
            return 5  # VGA
        return 6  # auto

    @staticmethod
    def _zed_depth_mode_enum(mode: str) -> int:
        mapping = {
            "NONE": 0,
            "PERFORMANCE": 1,
            "QUALITY": 2,
            "ULTRA": 3,
            "NEURAL": 5,
            "NEURAL_PLUS": 6,
            "NEURAL_LIGHT": 1,  # best available fallback in this plugin enum set
        }
        return mapping.get(str(mode).upper().strip(), 5)

    def _build_shm_source_launch(self, cfg: InferenceConfig, cam_count: int) -> str:
        source_blocks: list[str] = []
        self._extra_raw_videos = []
        aligned_single_cam = bool(self.enable_infer and cam_count == 1)
        for idx, sock in enumerate(self.socks):
            tee_name = f"T{idx}"
            raw_out = self.artifacts.raw_video if idx == 0 else (self.run_dir / f"raw_cam{idx}.mp4")
            if idx > 0:
                self._extra_raw_videos.append(raw_out)
            if aligned_single_cam and idx == 0:
                source_blocks.append(
                    f"""
            shmsrc socket-path={sock} is-live=true do-timestamp=true !
              video/x-raw,format=GRAY8,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 !
              nvvideoconvert compute-hw=1 copy-hw=2 !
              video/x-raw(memory:NVMM),format=NV12,width={cfg.width},height={cfg.height} !
              queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! m.sink_{idx}
                    """.rstrip()
                )
            else:
                source_blocks.append(
                    f"""
            shmsrc socket-path={sock} is-live=true do-timestamp=true !
              video/x-raw,format=GRAY8,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 !
              nvvideoconvert compute-hw=1 copy-hw=2 !
              video/x-raw(memory:NVMM),format=NV12,width={cfg.width},height={cfg.height} !
              tee name={tee_name}

              {tee_name}. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream !
                   nvvideoconvert compute-hw=1 copy-hw=2 !
                   video/x-raw,format=NV12,width={cfg.width},height={cfg.height} !
                   x264enc tune=zerolatency speed-preset=ultrafast bitrate={cfg.bitrate} key-int-max={cfg.fps} !
                   h264parse ! mp4mux !
                   filesink location=\"{raw_out}\" sync=false

              {tee_name}. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! m.sink_{idx}
                    """.rstrip()
                )
        return "\n\n".join(source_blocks)

    def _build_zed_source_launch(self, cfg: InferenceConfig, cam_count: int) -> str:
        self._extra_raw_videos = []
        self._depth_video = None
        self._zed_svo_file = self.run_dir / "zed_capture.svo2"
        self._zed_svo_enable_attempts = 0
        self._zed_svo_enabled_runtime = False
        if self.zed_depth_enabled:
            try:
                if os.path.exists(self.zed_depth_socket):
                    os.unlink(self.zed_depth_socket)
            except Exception:
                pass
        zed_res = self._zed_resolution_enum(cfg.width, cfg.height)
        zed_depth_mode = self._zed_depth_mode_enum(self.zed_depth_mode)
        stream_type = 4 if self.zed_depth_enabled else 2
        is_depth = "true" if self.zed_depth_enabled else "false"
        print(
            f"[{ts()}] [INFO] ZED DeepStream source: stream-type={stream_type} "
            f"depth-mode={self.zed_depth_mode} confidence-threshold={self.zed_confidence_threshold} "
            f"texture-confidence={self.zed_texture_confidence_threshold} "
            f"depth-range-mm={self.zed_depth_minimum_distance_mm}-{self.zed_depth_maximum_distance_mm} "
            f"fill-mode={'on' if self.zed_fill_mode else 'off'} depth-stabilization={self.zed_depth_stabilization} "
            f"requested_cams={self.requested_cam_count} effective_cams={cam_count}",
            flush=True,
        )

        raw_left = self.artifacts.raw_video
        raw_aux = self.run_dir / "raw_cam1.mp4"
        if cam_count > 1:
            self._extra_raw_videos.append(raw_aux)
        preview_w = 640
        preview_h = 400
        preview_fps = min(15, int(cfg.fps))

        left_block = f"""
            zed_demux.src_left ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 !
                autovideoconvert !
                nvvideoconvert compute-hw=1 copy-hw=2 !
                video/x-raw(memory:NVMM),format=NV12,width={cfg.width},height={cfg.height} !
                tee name=T0
            T0. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream !
                nvvideoconvert compute-hw=1 copy-hw=2 !
                video/x-raw,format=NV12,width={cfg.width},height={cfg.height} !
                x264enc tune=zerolatency speed-preset=ultrafast bitrate={cfg.bitrate} key-int-max={cfg.fps} !
                h264parse ! mp4mux ! filesink location=\"{raw_left}\" sync=false
            T0. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! m.sink_0
        """.rstrip()

        aux_block = ""
        if cam_count > 1:
            aux_block = f"""
            zed_demux.src_aux ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 !
                autovideoconvert !
                nvvideoconvert compute-hw=1 copy-hw=2 !
                video/x-raw(memory:NVMM),format=NV12,width={cfg.width},height={cfg.height} !
                tee name=T1
            T1. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream !
                nvvideoconvert compute-hw=1 copy-hw=2 !
                video/x-raw,format=NV12,width={cfg.width},height={cfg.height} !
                x264enc tune=zerolatency speed-preset=ultrafast bitrate={cfg.bitrate} key-int-max={cfg.fps} !
                h264parse ! mp4mux ! filesink location=\"{raw_aux}\" sync=false
            T1. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! m.sink_1
            """.rstrip()
        elif self.zed_depth_enabled:
            aux_record_block = ""
            if self.zed_depth_record:
                depth_out = self.run_dir / "raw_depth.mp4"
                self._depth_video = depth_out
                aux_record_block = f"""
            TDEP. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream !
                videoconvert ! video/x-raw,format=I420,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 !
                x264enc tune=zerolatency speed-preset=ultrafast bitrate={cfg.bitrate} key-int-max={cfg.fps} !
                h264parse ! mp4mux ! filesink location=\"{depth_out}\" sync=false
                """.rstrip()
            aux_block = f"""
            zed_demux.src_aux ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 !
                autovideoconvert !
                video/x-raw,format=BGRA,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 !
                tee name=TDEP
            TDEP. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream !
                videorate ! videoscale !
                video/x-raw,format=BGRA,width={preview_w},height={preview_h},framerate={preview_fps}/1 !
                shmsink socket-path={self.zed_depth_socket} wait-for-connection=false sync=false async=false shm-size=20000000
            {aux_record_block}
            """.rstrip()

        return f"""
            zedsrc name=zedsrc camera-resolution={zed_res} camera-fps={cfg.fps} stream-type={stream_type} depth-mode={zed_depth_mode}
                confidence-threshold={self.zed_confidence_threshold} texture-confidence-threshold={self.zed_texture_confidence_threshold}
                depth-minimum-distance={self.zed_depth_minimum_distance_mm} depth-maximum-distance={self.zed_depth_maximum_distance_mm}
                fill-mode={'true' if self.zed_fill_mode else 'false'} depth-stabilization={self.zed_depth_stabilization}
                ctrl-saturation=4 ctrl-sharpness=1 ctrl-gamma=2 ctrl-whitebalance-auto=true ctrl-aec-agc=true
                svo-recording-filename=\"{self._zed_svo_file}\" svo-recording-compression={self._zed_svo_compression} !
                zeddemux name=zed_demux is-depth={is_depth}
            {left_block}
            {aux_block}
        """.rstrip()

    def _enable_zed_svo_runtime(self) -> bool:
        if self._stopping or self.capture_backend != "zed" or self._zed_svo_file is None:
            return False
        self._zed_svo_enable_attempts += 1
        try:
            zedsrc = self.pipeline.get_by_name("zedsrc")
            if zedsrc is None:
                raise RuntimeError("zedsrc element not found in pipeline")
            zedsrc.set_property("svo-recording-filename", str(self._zed_svo_file))
            # Force H264 because H265 SVO playback/export is failing on this stack.
            zedsrc.set_property("svo-recording-compression", 1)
            zedsrc.set_property("svo-recording-enable", True)
            active = bool(zedsrc.get_property("svo-recording-enable"))
            try:
                compression = int(zedsrc.get_property("svo-recording-compression"))
            except Exception:
                compression = -1
            if active:
                self._zed_svo_enabled_runtime = True
                print(
                    f"[{ts()}] [INFO] ZED SVO recording enabled at runtime "
                    f"(compression={compression}) -> {self._zed_svo_file}",
                    flush=True,
                )
                return False
            if self._zed_svo_enable_attempts % 5 == 0:
                print(
                    f"[{ts()}] [WARN] ZED SVO recording not active yet "
                    f"(attempt {self._zed_svo_enable_attempts})",
                    flush=True,
                )
        except Exception as exc:
            if self._zed_svo_enable_attempts % 5 == 0:
                print(
                    f"[{ts()}] [WARN] ZED SVO runtime enable failed "
                    f"(attempt {self._zed_svo_enable_attempts}): {exc}",
                    flush=True,
                )
        # Retry for a short window after PLAYING because zedsrc only starts recording once started.
        return self._zed_svo_enable_attempts < 20

    def build(self) -> None:
        cfg = self.config
        if self.enable_infer and not cfg.cfg_path:
            raise RuntimeError("DeepStream config (--cfg) is required when inference is enabled.")
        if self.capture_backend != "zed":
            for sock in self.socks:
                for _ in range(300):
                    if os.path.exists(sock):
                        break
                    time.sleep(0.01)
        cam_count = self._effective_cam_count()
        cfg_batch = self._load_cfg_batch_size(cfg.cfg_path)
        if self.enable_infer and cfg_batch is not None and cfg_batch != cam_count:
            raise RuntimeError(
                f"nvinfer config batch-size ({cfg_batch}) does not match camera count ({cam_count}). "
                f"Update {cfg.cfg_path} and rebuild/use a matching engine."
            )
        mux_timeout_us = max(10_000, int(1_000_000 / max(1, int(cfg.fps or 30))))
        self._single_cam_aligned_raw = bool(self.enable_infer and cam_count == 1)
        if self._single_cam_aligned_raw:
            print(f"[{ts()}] [INFO] single-camera raw.mp4 aligned to detection branch", flush=True)
        tile_rows = 1 if cam_count <= 2 else int(math.ceil(math.sqrt(cam_count)))
        tile_cols = int(math.ceil(cam_count / max(1, tile_rows)))
        tiler_block = ""
        if cam_count > 1:
            tiler_block = (
                f"nvmultistreamtiler rows={tile_rows} columns={tile_cols} "
                f"width={cfg.width} height={cfg.height} !"
            )

        if self.enable_infer:
            if self._single_cam_aligned_raw:
                infer_block = f"""
                  nvinfer name=pgie config-file-path=\"{cfg.cfg_path}\" batch-size={cam_count} !
                  nvvideoconvert compute-hw=1 copy-hw=2 !
                  video/x-raw(memory:NVMM),format=RGBA,width={cfg.width},height={cfg.height} !
                  tee name=raw_align_tee
                  raw_align_tee. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream !
                    nvvideoconvert compute-hw=1 copy-hw=2 !
                    video/x-raw,format=I420,width={cfg.width},height={cfg.height} !
                    x264enc tune=zerolatency speed-preset=ultrafast bitrate={cfg.bitrate} key-int-max={cfg.fps} !
                    h264parse ! mp4mux ! filesink location=\"{self.artifacts.raw_video}\" sync=false
                  raw_align_tee. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream !
                    tee name=vis_tee
                  vis_tee. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream ! vis_select.sink_0
                  videotestsrc pattern=black is-live=true !
                    video/x-raw,format=RGBA,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 !
                    nvvideoconvert compute-hw=1 copy-hw=2 !
                    video/x-raw(memory:NVMM),format=RGBA,width={cfg.width},height={cfg.height} !
                    queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! vis_select.sink_1
                  input-selector name=vis_select sync-streams=true !
                  nvdsosd name=osd process-mode=0 display-bbox=1 display-text=1 !
                  nvvideoconvert compute-hw=1 copy-hw=2 !
                  video/x-raw(memory:NVMM),format=RGBA,width={cfg.width},height={cfg.height} !
                """
            elif self.osd_cpu:
                infer_block = f"""
                  nvinfer name=pgie config-file-path=\"{cfg.cfg_path}\" batch-size={cam_count} !
                  {tiler_block}
                  nvvideoconvert compute-hw=1 copy-hw=2 !
                  video/x-raw,format=RGBA,width={cfg.width},height={cfg.height} !
                  tee name=vis_tee
                  vis_tee. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream ! vis_select.sink_0
                  videotestsrc pattern=black is-live=true !
                    video/x-raw,format=RGBA,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 !
                    queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! vis_select.sink_1
                  input-selector name=vis_select sync-streams=true !
                  nvdsosd name=osd process-mode=1 display-bbox=1 display-text=1 !
                  nvvideoconvert compute-hw=1 copy-hw=2 !
                  video/x-raw(memory:NVMM),format=RGBA,width={cfg.width},height={cfg.height} !
                """
            else:
                infer_block = f"""
                  nvinfer name=pgie config-file-path=\"{cfg.cfg_path}\" batch-size={cam_count} !
                  {tiler_block}
                  nvvideoconvert compute-hw=1 copy-hw=2 !
                  video/x-raw(memory:NVMM),format=RGBA,width={cfg.width},height={cfg.height} !
                  tee name=vis_tee
                  vis_tee. ! queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream ! vis_select.sink_0
                  videotestsrc pattern=black is-live=true !
                    video/x-raw,format=RGBA,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 !
                    nvvideoconvert compute-hw=1 copy-hw=2 !
                    video/x-raw(memory:NVMM),format=RGBA,width={cfg.width},height={cfg.height} !
                    queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! vis_select.sink_1
                  input-selector name=vis_select sync-streams=true !
                  nvdsosd name=osd process-mode=0 display-bbox=1 display-text=1 !
                  nvvideoconvert compute-hw=1 copy-hw=2 !
                  video/x-raw(memory:NVMM),format=RGBA,width={cfg.width},height={cfg.height} !
                """
        else:
            infer_block = f"""
              {tiler_block}
              nvvideoconvert compute-hw=1 copy-hw=2 !
              video/x-raw(memory:NVMM),format=RGBA,width={cfg.width},height={cfg.height} !
            """

        if self.capture_backend == "zed":
            sources_launch = self._build_zed_source_launch(cfg, cam_count)
        else:
            sources_launch = self._build_shm_source_launch(cfg, cam_count)

        launch = f"""
{sources_launch}

              nvstreammux name=m batch-size={cam_count} width={cfg.width} height={cfg.height} live-source=1 batched-push-timeout={mux_timeout_us} !
              identity name=perf_tap silent=true !
              queue max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream !
{infer_block}
              tee name=TA

              TA. ! valve name=preview_valve drop=false ! queue name=q_preview max-size-buffers=5 max-size-bytes=0 max-size-time=0 leaky=downstream !
                   nvegltransform ! nveglglessink name=preview_sink sync=false

        """
        print(f"[{ts()}] [INFO] building pipeline…")
        self.pipeline = Gst.parse_launch(launch)
        mux = self.pipeline.get_by_name("m")
        if mux and cam_count > 1:
            try:
                mux.set_property("sync-inputs", True)
            except Exception:
                pass
        self._install_caps_logging()
        tap = self.pipeline.get_by_name("perf_tap")
        if tap:
            pad = tap.get_static_pad("src")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self._on_stream_probe)

        osd = self.pipeline.get_by_name("osd") if self.enable_infer else None
        if self.enable_infer:
            self._csv_probe_pre_tiler = cam_count > 1
            if osd is None:
                raise AssertionError("nvdsosd element missing")
            pgie = self.pipeline.get_by_name("pgie")
            if pgie is None:
                print(f"[{ts()}] [WARN] nvinfer element 'pgie' not found; inference metrics disabled")
            else:
                sink_pad = pgie.get_static_pad("sink")
                src_pad = pgie.get_static_pad("src")
                if sink_pad is not None:
                    sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_pgie_sink)
                else:
                    print(f"[{ts()}] [WARN] nvinfer sink pad missing; cannot time inference")
                if src_pad is not None:
                    src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_pgie_src)
                    if self._csv_probe_pre_tiler:
                        src_pad.add_probe(Gst.PadProbeType.BUFFER, self.on_probe)
                else:
                    print(f"[{ts()}] [WARN] nvinfer src pad missing; cannot time inference")
            try:
                # GPU mode (0) works with NVMM surfaces; CPU mode would require sysmem.
                osd.set_property("process-mode", 0)
                osd.set_property("display-bbox", 1)
                osd.set_property("display-text", 1)
            except Exception as exc:
                print(f"[{ts()}] [WARN] could not set OSD props: {exc}")

        sink = self.pipeline.get_by_name("preview_sink")
        self._preview_sink = sink
        if sink is not None and self._window_handle:
            handle = int(self._window_handle)
            try:
                sink.set_property("window-xid", handle)
            except Exception:
                pass
            try:
                if hasattr(sink, "set_window_handle"):
                    sink.set_window_handle(handle)
                else:
                    GstVideo.VideoOverlay.set_window_handle(sink, handle)
            except Exception as exc:
                print(f"[{ts()}] [WARN] could not bind preview window: {exc}")
        else:
            # No preview window available; force preview disabled.
            self.preview_enabled = False

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_msg)

        if self.enable_infer and osd is not None:
            if self._csv_probe_pre_tiler:
                # In multi-camera mode, process metadata on pgie src (pre-tiler)
                # so source/frame identity is preserved for both CSV and pose draws.
                pass
            else:
                # Single-camera mode can use OSD sink directly.
                pad = osd.get_static_pad("sink")
                assert pad is not None, "osd sink pad missing"
                pad.add_probe(Gst.PadProbeType.BUFFER, self.on_probe)
        self._preview_valve = self.pipeline.get_by_name("preview_valve")
        if self._preview_valve is None:
            print(f"[{ts()}] [WARN] preview valve missing; preview toggle unavailable")
        else:
            try:
                self._preview_valve.set_property("drop", not self.preview_enabled)
            except Exception:
                pass
        self._vis_select = self.pipeline.get_by_name("vis_select")
        if self._vis_select:
            try:
                live_pad = self._vis_select.get_static_pad("sink_0")
                if live_pad:
                    self._vis_select.set_property("active-pad", live_pad)
                self._apply_video_state()
            except Exception as exc:
                print(f"[{ts()}] [WARN] could not set initial video pad: {exc}")
        else:
            if self.enable_infer:
                print(f"[{ts()}] [WARN] vis_select missing; video toggle unavailable")
        # Preview/video toggles were removed; always keep preview enabled when a window is available.

    def _collect_debug_queues(self) -> None:
        """Gather queue elements to log their levels with frame metadata."""
        self._debug_queues = []
        if not self.pipeline:
            return
        try:
            it = self.pipeline.iterate_elements()
            while True:
                try:
                    ok, elem = it.next()
                except StopIteration:
                    break
                except Exception:
                    break
                if not ok or elem is None:
                    continue
                try:
                    factory = elem.get_factory()
                    if factory and factory.get_name() == "queue":
                        self._debug_queues.append(elem)
                except Exception:
                    continue
        except Exception:
            pass

    def _install_caps_logging(self) -> None:
        """Print caps/surface details early to catch renegotiation or bad surfaces."""
        if not self._surf_debug_enabled:
            return
        mux = self.pipeline.get_by_name("m")
        if mux:
            pad = mux.get_static_pad("src")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self._on_mux_src)
        pgie = self.pipeline.get_by_name("pgie")
        if pgie:
            sink = pgie.get_static_pad("sink")
            if sink:
                sink.add_probe(Gst.PadProbeType.BUFFER, self._on_pgie_sink_caps)

    def _on_pgie_sink(self, _pad, info):
        buf = info.get_buffer()
        if buf is not None:
            self._infer_starts[hash(buf)] = time.perf_counter()
        return Gst.PadProbeReturn.OK

    def _on_pgie_src(self, _pad, info):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        key = hash(buf)
        start = self._infer_starts.pop(key, None)
        if start is None:
            return Gst.PadProbeReturn.OK
        now_perf = time.perf_counter()
        latency_ms = (now_perf - start) * 1000.0
        wall_ts = time.time()
        self._infer_history.append((wall_ts, latency_ms))
        cutoff = wall_ts - self._perf_window_sec
        while self._infer_history and self._infer_history[0][0] < cutoff:
            self._infer_history.popleft()
        fps = float('nan')
        if len(self._infer_history) >= 2:
            span = self._infer_history[-1][0] - self._infer_history[0][0]
            if span > 0:
                fps = (len(self._infer_history) - 1) / span
        avg_latency = float('nan')
        if self._infer_history:
            avg_latency = sum(lat for _, lat in self._infer_history) / len(self._infer_history)
        if getattr(self, '_perf_writer', None):
            self._write_perf_row(stream_fps=self._stream_fps, infer_fps=fps, latency_ms=avg_latency)
        return Gst.PadProbeReturn.OK

    @staticmethod
    def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _init_pose_cache_helper(self) -> None:
        cfg_path = Path(self.config.cfg_path)
        lib_path: Path | None = None
        try:
            for raw in cfg_path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("custom-lib-path"):
                    value = line.split("=", 1)[1].strip().strip('"')
                    lib_path = Path(value)
                    break
        except Exception as exc:
            print(f"[{ts()}] [POSE] unable to read {cfg_path}: {exc}")
            return
        if not lib_path or not lib_path.exists():
            print(f"[{ts()}] [POSE] custom lib missing for pose cache")
            return
        try:
            lib = ctypes.CDLL(str(lib_path))
            fn = lib.NvDsInferGetPoseCache
            fn.restype = ctypes.c_ulonglong
            fn.argtypes = [
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            pop_fn = getattr(lib, "NvDsInferPopPoseCache", None)
            if pop_fn is not None:
                pop_fn.restype = ctypes.c_ulonglong
                pop_fn.argtypes = [
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                ]
            self._pose_cache_lib = lib
            self._pose_cache_fn = fn
            self._pose_cache_pop_fn = pop_fn
            if pop_fn is not None:
                print(f"[{ts()}] [POSE] cache hook ready (queued): {lib_path}")
            else:
                print(f"[{ts()}] [POSE] cache hook ready (latest-only): {lib_path}")
        except Exception as exc:
            print(f"[{ts()}] [POSE] cache hook failed: {exc}")

    def _decode_pose_tensor(self, frame_meta, *, consume: bool = True) -> list[dict] | None:
        if not self.pose_mode:
            return None
        fn = getattr(self, "_pose_cache_fn", None)
        pop_fn = getattr(self, "_pose_cache_pop_fn", None)
        if fn is None and pop_fn is None:
            return []

        data_ptr = ctypes.POINTER(ctypes.c_float)()
        total = ctypes.c_int()
        kpts = ctypes.c_int()
        used_pop = False
        if consume and pop_fn is not None:
            seq = pop_fn(ctypes.byref(data_ptr), ctypes.byref(total), ctypes.byref(kpts))
            used_pop = True
        else:
            seq = fn(ctypes.byref(data_ptr), ctypes.byref(total), ctypes.byref(kpts))
        total_val = int(total.value)

        if seq == 0 or total_val <= 0 or not data_ptr:
            if not used_pop:
                self._pose_cache_seq = seq
                self._pose_cache_last = []
            return []

        if (not used_pop) and seq == self._pose_cache_seq and self._pose_cache_last is not None:
            return self._pose_cache_last

        kpt_count = max(0, int(kpts.value))
        stride = 5 + 3 * kpt_count
        if stride <= 5:
            print(f"[{ts()}] [POSE] cache stride invalid: stride={stride}")
            return []
        remainder = total_val % stride
        if remainder != 0:
            print(f"[{ts()}] [POSE] cache size mismatch total={total_val} stride={stride}")
            total_val -= remainder
            if total_val <= 0:
                return []

        arr = np.ctypeslib.as_array(data_ptr, shape=(total_val,))
        arr = np.array(arr, copy=True).reshape(-1, stride)

        detections: list[dict] = []
        self.pose_kpt_count = kpt_count
        self.pose_kpt_dims = 3 if kpt_count > 0 else 0
        if arr.size and not hasattr(self, "_pose_debug_raw_printed"):
            try:
                rows_to_show = arr[: min(3, arr.shape[0]), : min(stride, arr.shape[1])]
                print(f"[{ts()}] [POSE] raw sample rows:\n{rows_to_show}")
            except Exception as debug_exc:
                print(f"[{ts()}] [POSE] raw sample dump err: {debug_exc}")
            self._pose_debug_raw_printed = True

        in_w, in_h = self.pose_input_dims if hasattr(self, "pose_input_dims") else (float(self.config.width), float(self.config.height))
        net_w = float(in_w)
        net_h = float(in_h)
        frame_w = float(getattr(frame_meta, "source_frame_width", self.config.width))
        frame_h = float(getattr(frame_meta, "source_frame_height", self.config.height))
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = float(self.config.width), float(self.config.height)
        gain = min(net_w / frame_w, net_h / frame_h) if frame_w and frame_h else 1.0
        if gain <= 0:
            gain = 1.0
        pad_x = 0.5 * (net_w - frame_w * gain)
        pad_y = 0.5 * (net_h - frame_h * gain)

        for chunk in arr:
            if chunk.size < stride:
                continue
            x1, y1, x2, y2, conf = chunk[:5]
            if conf <= 0:
                continue
            # undo letterbox
            x1 = (x1 - pad_x) / gain
            y1 = (y1 - pad_y) / gain
            x2 = (x2 - pad_x) / gain
            y2 = (y2 - pad_y) / gain
            x1 = float(np.clip(x1, 0.0, frame_w - 1.0))
            y1 = float(np.clip(y1, 0.0, frame_h - 1.0))
            x2 = float(np.clip(x2, 0.0, frame_w - 1.0))
            y2 = float(np.clip(y2, 0.0, frame_h - 1.0))

            kp_vals = chunk[5 : 5 + 3 * kpt_count]
            if kp_vals.size < 3 * kpt_count:
                continue
            kpts_flat: list[float] = []
            for idx in range(kpt_count):
                kx = (kp_vals[3 * idx + 0] - pad_x) / gain
                ky = (kp_vals[3 * idx + 1] - pad_y) / gain
                ks = float(kp_vals[3 * idx + 2])
                kx = float(np.clip(kx, 0.0, frame_w - 1.0))
                ky = float(np.clip(ky, 0.0, frame_h - 1.0))
                kpts_flat.extend([kx, ky, ks])

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "kpts": kpts_flat,
            })

        detections.sort(key=lambda d: d["conf"], reverse=True)
        self._pose_cache_seq = int(seq)
        self._pose_cache_last = detections
        if detections and not hasattr(self, "_pose_debug_cache_print"):
            first = detections[0]
            kp0 = first["kpts"][0] if first["kpts"] else 0.0
            print(f"[{ts()}] [POSE] cache sample conf={first['conf']:.4f} kp0={kp0:.2f}")
            self._pose_debug_cache_print = True
        return detections

    def _match_pose_to_bbox(self, bbox: tuple[float, float, float, float], det_conf: float, pose_dets: list[dict], used: set[int]) -> dict | None:
        if not pose_dets:
            return None
        best_idx = -1
        best_score = -1e9
        dx1, dy1, dx2, dy2 = bbox
        for idx, det in enumerate(pose_dets):
            if idx in used:
                continue
            bx1, by1, bx2, by2 = det["bbox"]
            ix1 = max(dx1, bx1)
            iy1 = max(dy1, by1)
            ix2 = min(dx2, bx2)
            iy2 = min(dy2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            area_det = max(1e-6, (dx2 - dx1) * (dy2 - dy1))
            area_box = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_det + area_box - inter + 1e-6
            iou = inter / union
            conf_penalty = abs(det_conf - det["conf"])
            score = iou - 0.001 * conf_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            return None
        used.add(best_idx)
        return pose_dets[best_idx]

    def on_bus_msg(self, bus, msg):
        msg_type = msg.type
        if msg_type == Gst.MessageType.EOS:
            print(f"[{ts()}] [BUS] EOS")
            self._disable_zed_svo_recording(context="eos")
            self.stop()
        elif msg_type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[{ts()}] [BUS] ERROR: {err}  debug:{dbg}")
            self.stop()
        elif msg_type == Gst.MessageType.ELEMENT:
            st = msg.get_structure()
            name = st.get_name() if st else None
            if st and name == "prepare-window-handle" and self._window_handle:
                try:
                    if hasattr(msg.src, "set_window_handle"):
                        msg.src.set_window_handle(int(self._window_handle))
                    else:
                        GstVideo.VideoOverlay.set_window_handle(msg.src, int(self._window_handle))
                except Exception as exc:
                    print(f"[{ts()}] [WARN] could not bind preview window: {exc}")
        return True

    def _disable_zed_svo_recording(self, *, context: str) -> None:
        if self.capture_backend != "zed":
            return
        try:
            zedsrc = self.pipeline.get_by_name("zedsrc")
        except Exception:
            zedsrc = None
        if zedsrc is None:
            return
        try:
            pre = bool(zedsrc.get_property("svo-recording-enable"))
        except Exception:
            pre = False
        print(f"[{ts()}] [INFO] ZED SVO stop ({context}): pre-disable active={pre}", flush=True)
        try:
            zedsrc.set_property("svo-recording-enable", False)
        except Exception:
            return
        for _ in range(40):
            try:
                active = bool(zedsrc.get_property("svo-recording-enable"))
            except Exception:
                active = False
            if not active:
                break
            time.sleep(0.05)
        try:
            post = bool(zedsrc.get_property("svo-recording-enable"))
        except Exception:
            post = False
        print(f"[{ts()}] [INFO] ZED SVO stop ({context}): post-disable active={post}", flush=True)
        # Small grace window for encoder/container finalization.
        time.sleep(0.35)

    def on_probe(self, pad, info):
        if not self.enable_infer:
            return Gst.PadProbeReturn.OK
        parent_name = ""
        pad_name = ""
        try:
            parent = pad.get_parent_element()
            if parent is not None:
                parent_name = parent.get_name() or ""
        except Exception:
            parent_name = ""
        try:
            pad_name = pad.get_name() or ""
        except Exception:
            pad_name = ""
        is_pretiler_probe = parent_name == "pgie" and pad_name == "src"
        if self._csv_probe_pre_tiler and not is_pretiler_probe:
            return Gst.PadProbeReturn.OK
        write_csv = is_pretiler_probe or (not self._csv_probe_pre_tiler)
        draw_pose = is_pretiler_probe or (not self._csv_probe_pre_tiler)
        consume_pose = is_pretiler_probe or (not self._csv_probe_pre_tiler)
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        try:
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if not batch:
                return Gst.PadProbeReturn.OK
            l_frame = batch.frame_meta_list
            while l_frame:
                try:
                    fmeta = pyds.NvDsFrameMeta.cast(l_frame.data)
                except StopIteration:
                    break

                source_id = int(getattr(fmeta, "source_id", -1))
                pad_index = int(getattr(fmeta, "pad_index", -1))
                stream_id = source_id if source_id >= 0 else (pad_index if pad_index >= 0 else 0)
                source = f"cam{stream_id}"

                frame_num_raw = int(getattr(fmeta, "frame_num", -1))
                last_frame_num = self._stream_frame_seq.get(stream_id, -1)
                if frame_num_raw >= 0 and frame_num_raw > last_frame_num:
                    frame_num = frame_num_raw
                elif last_frame_num >= 0:
                    frame_num = last_frame_num + 1
                else:
                    frame_num = frame_num_raw if frame_num_raw >= 0 else 0
                self._stream_frame_seq[stream_id] = frame_num

                pts_value = int(getattr(fmeta, "buf_pts", 0) or 0)
                if pts_value > 0:
                    ts_us = int(pts_value / 1_000)
                else:
                    ntp_value = int(getattr(fmeta, "ntp_timestamp", 0) or 0)
                    if ntp_value > 0:
                        ts_us = int(ntp_value / 1_000)
                    else:
                        ts_us = int(time.time() * 1_000_000)

                pose_detections = self._decode_pose_tensor(fmeta, consume=consume_pose) if self.pose_mode else None
                used_pose: set[int] = set()
                frame_pose_draw: list[dict] = []

                obj_count = 0
                l_obj = fmeta.obj_meta_list
                while l_obj:
                    try:
                        ometa = pyds.NvDsObjectMeta.cast(l_obj.data)
                        rect = ometa.rect_params
                        x, y, w, h = rect.left, rect.top, rect.width, rect.height
                        try:
                            conf = float(ometa.confidence)
                        except Exception:
                            conf = -1.0
                        class_id = int(getattr(ometa, "class_id", -1))
                        label = ometa.obj_label if getattr(ometa, "obj_label", None) else ""
                        obj_id = _safe_object_id(ometa)

                        raw_frame_num = self._raw_frame_num_for_stream(frame_num, stream_id)
                        row_values: list[object] = [
                            frame_num,
                            raw_frame_num,
                            ts_us,
                            stream_id,
                            source,
                            obj_id,
                            class_id,
                            label,
                            f"{conf:.6f}",
                            f"{x:.3f}",
                            f"{y:.3f}",
                            f"{w:.3f}",
                            f"{h:.3f}",
                        ]
                        if self.pose_mode:
                            bbox = (x, y, x + w, y + h)
                            pose_entry = self._match_pose_to_bbox(bbox, conf, pose_detections or [], used_pose)
                            pose_schema = ""
                            pose_names_json = ""
                            pose_values_json = ""
                            pose_count = 0
                            if pose_entry:
                                pose_vals = pose_entry.get("kpts", [])
                            else:
                                pose_vals = None
                            if pose_vals:
                                schema = self._pose_schema_for_detection(class_id, label)
                                if schema:
                                    pose_offset = int(schema.offset)
                                    names = list(schema.names)
                                else:
                                    pose_offset = 0
                                    names = (
                                        list(self.kp_names[: self.pose_kpt_count])
                                        if self.kp_names and len(self.kp_names) >= self.pose_kpt_count
                                        else [f"kp{idx}" for idx in range(self.pose_kpt_count)]
                                    )
                                pose_count = len(names)
                                expected = (pose_offset + pose_count) * self.pose_kpt_dims
                            else:
                                pose_count = 0
                                expected = 0
                            if pose_vals and pose_count > 0 and len(pose_vals) >= expected:
                                pose_schema = f"{(label or f'class_{class_id}').strip() or f'class_{class_id}'}_v1"
                                pose_triplets: list[list[float]] = []
                                for idx in range(pose_count):
                                    base = (pose_offset + idx) * self.pose_kpt_dims
                                    pose_triplets.append(
                                        [
                                            round(float(pose_vals[base + 0]), 3),
                                            round(float(pose_vals[base + 1]), 3),
                                            round(float(pose_vals[base + 2]), 3),
                                        ]
                                    )
                                pose_names_json = json.dumps(names, separators=(",", ":"))
                                pose_values_json = json.dumps(pose_triplets, separators=(",", ":"))
                                frame_pose_draw.append({
                                    "rect": (x, y, w, h),
                                    "pose_bbox": pose_entry.get("bbox", bbox) if pose_entry else bbox,
                                    "kpts": pose_vals,
                                })
                            row_values.extend([pose_schema, pose_count, pose_names_json, pose_values_json])

                        if write_csv and self.csv:
                            self.csv.row(row_values)
                        obj_count += 1
                    except Exception as exc:
                        print(f"[{ts()}] [PROBE] warn: {exc}", flush=True)
                    finally:
                        try:
                            l_obj = l_obj.next
                        except StopIteration:
                            break

                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

                if draw_pose and self.pose_mode and frame_pose_draw:
                    display_meta = None
                    try:
                        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch)
                        try:
                            max_circles = len(display_meta.circle_params)
                        except Exception:
                            max_circles = int(getattr(pyds, "NVDS_MAX_CIRCLE_PARAMS", 128))
                        try:
                            max_lines = len(display_meta.line_params)
                        except Exception:
                            max_lines = int(getattr(pyds, "NVDS_MAX_LINE_PARAMS", 128))
                        circle_idx = 0
                        line_idx = 0
                        for entry in frame_pose_draw:
                            rect_x, rect_y, rect_w, rect_h = entry["rect"]
                            pose_x1, pose_y1, pose_x2, pose_y2 = entry["pose_bbox"]
                            pose_w = pose_x2 - pose_x1
                            pose_h = pose_y2 - pose_y1
                            if pose_w <= 1e-6 or pose_h <= 1e-6:
                                continue
                            kpts_vals = entry["kpts"]
                            draw_points: list[tuple[float, float, float]] = []
                            for kp_idx in range(self.pose_kpt_count):
                                base = kp_idx * self.pose_kpt_dims
                                try:
                                    kx = float(kpts_vals[base + 0])
                                    ky = float(kpts_vals[base + 1])
                                    ks = float(kpts_vals[base + 2])
                                except (IndexError, ValueError, TypeError):
                                    continue
                                if ks < self.pose_draw_score_thresh:
                                    continue
                                norm_x = (kx - pose_x1) / pose_w
                                norm_y = (ky - pose_y1) / pose_h
                                if circle_idx >= max_circles:
                                    break
                                draw_x = rect_x + float(np.clip(norm_x, 0.0, 1.0)) * rect_w
                                draw_y = rect_y + float(np.clip(norm_y, 0.0, 1.0)) * rect_h
                                draw_points.append((draw_x, draw_y, ks))
                                cp = display_meta.circle_params[circle_idx]
                                cp.xc = int(round(draw_x))
                                cp.yc = int(round(draw_y))
                                cp.radius = max(1, int(self.pose_draw_radius))
                                # cycle colors by keypoint index for visibility
                                colors = [
                                    (1.0, 0.2, 0.2),  # red-ish
                                    (0.2, 0.8, 1.0),  # cyan-ish
                                    (0.9, 0.7, 0.2),  # yellow-ish
                                    (0.6, 0.4, 1.0),  # purple-ish
                                    (0.2, 1.0, 0.4),  # green-ish
                                ]
                                r, g, b = colors[kp_idx % len(colors)]
                                cp.circle_color.red = r
                                cp.circle_color.green = g
                                cp.circle_color.blue = b
                                cp.circle_color.alpha = 1.0
                                cp.has_bg_color = 0
                                circle_idx += 1
                            if circle_idx >= max_circles:
                                break
                            if self.draw_skeleton and len(draw_points) >= 2:
                                for i in range(len(draw_points)):
                                    for j in range(i + 1, len(draw_points)):
                                        if line_idx >= max_lines:
                                            break
                                        pa = draw_points[i]; pb = draw_points[j]
                                        if pa[2] < self.pose_draw_score_thresh or pb[2] < self.pose_draw_score_thresh:
                                            continue
                                        lp = display_meta.line_params[line_idx]
                                        lp.x1 = int(round(pa[0])); lp.y1 = int(round(pa[1]))
                                        lp.x2 = int(round(pb[0])); lp.y2 = int(round(pb[1]))
                                        lp.line_width = max(1, int(self.pose_draw_radius // 2))
                                        lp.line_color.red = 0.1; lp.line_color.green = 0.9; lp.line_color.blue = 0.9; lp.line_color.alpha = 1.0
                                        line_idx += 1
                                    if line_idx >= max_lines:
                                        break
                        if circle_idx > 0:
                            display_meta.num_circles = circle_idx
                        if self.draw_skeleton and line_idx > 0:
                            display_meta.num_lines = line_idx
                        if circle_idx > 0 or (self.draw_skeleton and line_idx > 0):
                            pyds.nvds_add_display_meta_to_frame(fmeta, display_meta)
                            display_meta = None  # ownership transferred
                    except Exception as exc:
                        print(f"[{ts()}] [POSE] display meta error: {exc}")
                    finally:
                        if display_meta is not None:
                            try:
                                pyds.nvds_release_display_meta_to_pool(batch, display_meta)
                            except Exception:
                                pass


        except Exception as exc:
            print(f"[{ts()}] [PROBE] outer warn: {exc}", flush=True)

        return Gst.PadProbeReturn.OK

    def run(self) -> None:
        print(f"[{ts()}] [INFO] set PLAYING…")
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            print(f"[{ts()}] [FATAL] cannot set pipeline to PLAYING")
            self.stop()
            return
        print(f"[{ts()}] [READY] inference playing", flush=True)
        if self.capture_backend == "zed" and self._zed_svo_file is not None:
            GLib.timeout_add(250, self._enable_zed_svo_runtime)
        print(f"[{ts()}] [INFO] streaming (Ctrl-C to stop)…")
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print(f"[{ts()}] [INFO] Ctrl-C → EOS")
            self.pipeline.send_event(Gst.Event.new_eos())

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        print(f"[{ts()}] [INFO] stopping pipeline…")
        try:
            if self.capture_backend == "zed":
                # Fallback disable for stop paths that do not go through EOS (e.g., error path).
                self._disable_zed_svo_recording(context="stop")
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        if self.csv:
            print(f"[{ts()}] [INFO] closing CSV: {self.csv.path}")
            self.csv.close()
        if getattr(self, '_perf_file', None):
            try:
                self._perf_file.flush()
                self._perf_file.close()
                print(f"[{ts()}] [INFO] perf log:   {self.perf_path}")
            except Exception:
                pass
        try:
            self.loop.quit()
        except Exception:
            pass
        print(f"[{ts()}] [INFO] done. Files in: {self.run_dir}")
        print(f"[{ts()}] [INFO]   raw:        {self.artifacts.raw_video}")
        for path in self._extra_raw_videos:
            print(f"[{ts()}] [INFO]   raw extra:  {path}")
        if self._depth_video is not None:
            print(f"[{ts()}] [INFO]   depth:      {self._depth_video}")
        if self._zed_svo_file is not None:
            print(f"[{ts()}] [INFO]   svo:        {self._zed_svo_file}")
            if self._zed_svo_file.exists():
                try:
                    print(f"[{ts()}] [INFO]   svo size:   {self._zed_svo_file.stat().st_size} bytes")
                except Exception:
                    pass
            print(f"[{ts()}] [INFO]   svo active: {self._zed_svo_enabled_runtime}")
        if self.enable_infer:
            print(f"[{ts()}] [INFO]   detections: {self.artifacts.detections_csv}")
        print(f"[{ts()}] [INFO]   (marker):   {run_context.RUN_MARKER}")

    def _write_perf_row(self, stream_fps: float, infer_fps: float | None = None, latency_ms: float | None = None) -> None:
        if not getattr(self, "_perf_writer", None):
            return
        row = [
            time.strftime('%H:%M:%S'),
            f"{stream_fps:.4f}" if stream_fps == stream_fps else '',
            f"{infer_fps:.4f}" if infer_fps is not None and infer_fps == infer_fps else '',
            f"{latency_ms:.4f}" if latency_ms is not None and latency_ms == latency_ms else '',
        ]
        self._perf_writer.writerow(row)
        if getattr(self, "_perf_file", None):
            try:
                self._perf_file.flush()
            except Exception:
                pass

    def _on_mux_src(self, _pad, info) -> Gst.PadProbeReturn:
        # Log caps once and first few surfaces to diagnose bad buffers.
        if not self._caps_logged:
            caps = _pad.get_current_caps()
            print(f"[{ts()}] [DEBUG] mux src caps: {caps.to_string() if caps else 'N/A'}", flush=True)
            self._caps_logged = True
        buf = info.get_buffer()
        if buf is not None:
            self._log_frame_meta(buf, label="mux src")
        return Gst.PadProbeReturn.OK

    def _log_frame_meta(self, buf: Gst.Buffer, label: str) -> None:
        if not self._surf_debug_enabled or self._surface_logs >= self._surf_debug_limit:
            return
        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        except Exception as exc:
            print(f"[{ts()}] [DEBUG] {label} meta fetch failed: {exc}", flush=True)
            return
        if not batch_meta:
            print(f"[{ts()}] [DEBUG] {label} meta missing", flush=True)
            return
        l_frame = batch_meta.frame_meta_list
        while l_frame and self._surface_logs < self._surf_debug_limit:
            try:
                fmeta = pyds.NvDsFrameMeta.cast(l_frame.data)
                print(
                    f"[{ts()}] [DEBUG] {label} frame_num={fmeta.frame_num} "
                    f"batch_id={fmeta.batch_id} src_id={fmeta.source_id} "
                    f"w={fmeta.source_frame_width} h={fmeta.source_frame_height} "
                    f"buf_pts={fmeta.buf_pts} rss_kb={_read_rss_kb()} fds={_fd_count()}",
                    flush=True,
                )
                self._surface_logs += 1
            except Exception as exc:
                print(f"[{ts()}] [DEBUG] {label} frame meta parse failed: {exc}", flush=True)
            try:
                l_frame = l_frame.next
            except Exception:
                break

    def _on_pgie_sink_caps(self, pad, info) -> Gst.PadProbeReturn:
        if self._pgie_caps_logged:
            return Gst.PadProbeReturn.OK
        caps = pad.get_current_caps()
        print(f"[{ts()}] [DEBUG] pgie sink caps: {caps.to_string() if caps else 'N/A'}", flush=True)
        self._pgie_caps_logged = True
        return Gst.PadProbeReturn.OK

    def _on_stream_probe(self, _pad, info):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        now = time.time()
        self._stream_history.append(now)
        cutoff = now - self._perf_window_sec
        while self._stream_history and self._stream_history[0] < cutoff:
            self._stream_history.popleft()
        fps = float("nan")
        if len(self._stream_history) >= 2:
            span = self._stream_history[-1] - self._stream_history[0]
            if span > 0:
                fps = (len(self._stream_history) - 1) / span
        self._stream_fps = fps
        if not self.enable_infer:
            self._write_perf_row(stream_fps=fps, infer_fps=float("nan"), latency_ms=float("nan"))
        return Gst.PadProbeReturn.OK
    def _atexit_cleanup(self) -> None:
        try:
            if self.csv:
                self.csv.close()
        except Exception:
            pass
        try:
            if getattr(self, "_perf_file", None):
                self._perf_file.flush()
                self._perf_file.close()
        except Exception:
            pass

    def _poll_skeleton_toggle(self):
        try:
            text = self.skeleton_ctrl_path.read_text().strip().lower()
        except FileNotFoundError:
            text = ""
        except Exception:
            return True
        new_state = text != "off"
        if new_state != self.draw_skeleton:
            self.draw_skeleton = new_state
            print(f"[{ts()}] [POSE] skeleton {'enabled' if new_state else 'disabled'}")
        return True

    def _apply_video_state(self):
        """Ensure the live feed stays selected on the visualizer input selector."""
        if not getattr(self, "_vis_select", None):
            return
        try:
            pad = self._vis_select.get_static_pad("sink_0")
            if pad:
                self._vis_select.set_property("active-pad", pad)
        except Exception:
            pass


def run(config: InferenceConfig) -> None:
    Gst.init(None)
    GObject.threads_init()

    app = App(config)

    def _sig(sig_num, _frame):
        print(f"[{ts()}] [SIG] {signal.Signals(sig_num).name} → EOS")
        if app.pipeline:
            app.pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    app.build()
    app.run()
