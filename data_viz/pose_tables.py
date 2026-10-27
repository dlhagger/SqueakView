"""Read consolidated detections or historical object/keypoint tables."""
from pathlib import Path
import pandas as pd
from squeakview.common.run_layout import artifact_path


def load_pose_tables(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(result_dir)
    path = artifact_path(root, "objects.csv")
    objects = pd.read_csv(path) if path.exists() else pd.DataFrame()
    legacy_points = artifact_path(root, "keypoints.csv")
    if legacy_points.exists():
        return objects, pd.read_csv(legacy_points)
    names = [column[:-5] for column in objects.columns if column.endswith("_x_px")]
    identity = [name for name in ("observation_id", "stream_id", "deepstream_frame_number",
        "source_sequence_index", "camera_frame_id", "track_id", "class_id", "class_label",
        "camera_timestamp_ns", "gst_pts_ns", "host_unix_ns", "host_monotonic_ns",
        "controller_time_us", "controller_time_method", "controller_boot_id",
        "controller_session_id") if name in objects]
    tables = []
    for index, name in enumerate(names):
        columns = [f"{name}_{suffix}" for suffix in ("x_px", "y_px", "confidence", "visible")]
        points = objects[identity + columns].rename(columns={column: column[len(name)+1:] for column in columns}).copy()
        points["keypoint_name"] = name
        points["keypoint_index"] = index
        tables.append(points.dropna(subset=["x_px", "y_px"]))
    points = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=identity+["keypoint_name", "keypoint_index", "x_px", "y_px", "confidence", "visible"])
    return objects, points
