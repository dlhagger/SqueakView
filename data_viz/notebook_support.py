"""Small, testable notebook input policies; plotting never changes source data."""
from pathlib import Path

from squeakview.common.run_context import read_json_required
from squeakview.common.run_layout import artifact_path


def selected_pose_schema(run_dir: Path, result_dir: Path) -> dict:
    run_dir, result_dir = Path(run_dir), Path(result_dir)
    if result_dir != run_dir:
        manifest = read_json_required(result_dir / 'offline_manifest.json')
        declared = Path(manifest['model_package']['pose_sidecar'])
        if not declared.is_file():
            raise ValueError('Selected offline model pose schema is unavailable; copy its declared schema with the results.')
        return read_json_required(declared)
    snapshots = sorted(artifact_path(run_dir, 'config').glob('*.pose.json'))
    if len(snapshots) == 1:
        return read_json_required(snapshots[0])
    manifest = read_json_required(artifact_path(run_dir, 'run_manifest.json'))
    if not snapshots and manifest.get('inference', {}).get('enabled') is False:
        return {}
    raise ValueError(f'expected one run-local pose schema, found {len(snapshots)}')


def shade_lick_gaps(axis, gaps):
    """Mark known missing observation intervals on a controller-time plot."""
    right = axis.get_xlim()[1]
    for index, gap in enumerate(gaps):
        axis.axvspan(gap['start_time_s'], gap['end_time_s'] if gap['end_time_s'] is not None else right,
                     color='grey', alpha=.2, label='Lick sensing unavailable' if index == 0 else None)
