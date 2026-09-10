from __future__ import annotations

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

from squeakview.common.diagnostics.system_telemetry import (
    MAX_TELEMETRY_LINE_CHARS,
    SystemTelemetryRecorder,
)
from squeakview.common.diagnostics.tegrastats import (
    parse_tegrastats_line,
    read_platform_metrics,
)


R39_LINE = (
    "09-01-2026 09:25:18 RAM 5739/7546MB (lfb 1x2MB) "
    "SWAP 2047/2048MB (cached 209MB) "
    "CPU [72%@1728,75%@1728,88%@1728,79%@1728,87%@1728,78%@1728] "
    "GR3D_FREQ 33% cpu@50C/50C soc2@47.937C/47.937C "
    "soc0@48.437C/48.437C gpu@48.75C/48.75C tj@50C/50C "
    "soc1@48.5C/48.5C VDD_IN 9761mW/9761mW/9761mW "
    "VDD_CPU_GPU_CV 4238mW/4238mW/4238mW VDD_SOC 1825mW/1825mW/1825mW"
)


class TegrastatsParserTests(unittest.TestCase):
    def test_parses_current_r39_line(self) -> None:
        sample = parse_tegrastats_line(R39_LINE)

        self.assertEqual(sample["parse_status"], "ok")
        self.assertEqual(sample["ram_used_mb"], 5739)
        self.assertAlmostEqual(float(sample["ram_pct"]), 5739 / 7546 * 100)
        self.assertEqual(sample["lfb_blocks"], 1)
        self.assertEqual(sample["swap_cached_mb"], 209)
        self.assertEqual(sample["cpu_online"], 6)
        self.assertEqual(sample["cpu_total"], 6)
        self.assertAlmostEqual(float(sample["cpu_util_mean_pct"]), 79.8333333333)
        self.assertEqual(sample["cpu_util_max_pct"], 88.0)
        self.assertEqual(sample["gpu_util_pct"], 33.0)
        self.assertIsNone(sample["gpu_freq_mhz"])
        self.assertEqual(sample["temp_gpu_c"], 48.75)
        self.assertEqual(sample["temp_max_c"], 50.0)
        self.assertEqual(sample["vdd_in_current_mw"], 9761)
        self.assertEqual(sample["vdd_cpu_gpu_cv_peak_mw"], 4238)
        self.assertEqual(sample["raw_line"], R39_LINE)

    def test_parses_legacy_emc_gpu_frequency_and_offline_cpu(self) -> None:
        sample = parse_tegrastats_line(
            "RAM 100/1000MB CPU [10%@1020,off,30%@2040] "
            "EMC_FREQ 40%@1600 GR3D_FREQ 50%@918 gpu@42.5C VDD_IN 5000mW"
        )

        self.assertEqual(sample["parse_status"], "ok")
        self.assertEqual(sample["cpu_online"], 2)
        self.assertEqual(sample["cpu_total"], 3)
        self.assertEqual(sample["cpu_util_mean_pct"], 20.0)
        self.assertEqual(sample["emc_util_pct"], 40.0)
        self.assertEqual(sample["emc_freq_mhz"], 1600.0)
        self.assertEqual(sample["gpu_freq_mhz"], 918.0)
        self.assertEqual(sample["vdd_in_avg_mw"], None)

    def test_preserves_partial_and_malformed_lines(self) -> None:
        partial = parse_tegrastats_line("RAM 10/20MB random-vendor-field 7")
        malformed = parse_tegrastats_line("not a tegrastats record\n")

        self.assertEqual(partial["parse_status"], "partial")
        self.assertEqual(partial["ram_pct"], 50.0)
        self.assertEqual(malformed["parse_status"], "malformed")
        self.assertEqual(malformed["raw_line"], "not a tegrastats record")

    def test_reads_optional_emc_and_throttle_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            emc = root / "bwmgr"
            emc.mkdir()
            (emc / "cur_freq").write_text("200\n")
            (emc / "max_freq").write_text("800\n")
            cpu = root / "cpu-throttle-alert" / "thermal_trip_event"
            gpu = root / "gpu-throttle-alert" / "thermal_trip_event"
            cpu.parent.mkdir()
            gpu.parent.mkdir()
            cpu.write_text("0\n")
            gpu.write_text("1\n")

            metrics = read_platform_metrics(
                emc_root=emc,
                throttle_paths=[cpu, gpu],
            )

        self.assertEqual(metrics["emc_freq_hz"], 200)
        self.assertEqual(metrics["emc_clock_pct_of_max"], 25.0)
        self.assertTrue(metrics["thermal_throttled"])
        self.assertEqual(metrics["thermal_throttle_domains"], "gpu")


class _FakeProcess:
    def __init__(self, lines: list[str], *, timeout_once: bool = False) -> None:
        self.stdout = _FakeStdout(lines)
        self.timeout_once = timeout_once
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None if not self.terminated and not self.killed else 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        del timeout
        self.wait_calls += 1
        if self.timeout_once and self.wait_calls == 1:
            raise subprocess.TimeoutExpired("tegrastats", 0.01)
        return 0


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self.closed = False

    def readline(self, size: int = -1) -> str:
        if not self._lines:
            return ""
        line = self._lines[0]
        if size >= 0 and len(line) > size:
            self._lines[0] = line[size:]
            return line[:size]
        self._lines.pop(0)
        return line

    def close(self) -> None:
        self.closed = True


class SystemTelemetryRecorderTests(unittest.TestCase):
    def test_shutdown_timeouts_reject_nonfinite_and_cap_large_values(self) -> None:
        recorder = SystemTelemetryRecorder(
            Path("/tmp/not-opened-system.csv"),
            terminate_timeout_s=float("inf"),
            kill_timeout_s=float("nan"),
        )
        self.assertEqual(recorder.terminate_timeout_s, 2.0)
        self.assertEqual(recorder.kill_timeout_s, 1.0)

        recorder = SystemTelemetryRecorder(
            Path("/tmp/not-opened-system.csv"),
            terminate_timeout_s=10_000,
            kill_timeout_s=10_000,
        )
        self.assertEqual(recorder.terminate_timeout_s, 60.0)
        self.assertEqual(recorder.kill_timeout_s, 60.0)

    def test_streams_deterministic_rows_and_clamps_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "diagnostics" / "system.csv"
            fake = _FakeProcess([R39_LINE + "\n", "unknown\n"])
            calls: list[tuple[list[str], dict]] = []

            def spawn(command, **kwargs):
                calls.append((command, kwargs))
                return fake

            unix_values = iter([100, 200])
            monotonic_values = iter([10, 20])
            recorder = SystemTelemetryRecorder(
                output,
                interval_ms=1,
                popen_factory=spawn,
                platform_reader=lambda: {
                    "emc_freq_hz": 123,
                    "thermal_throttled": False,
                    "thermal_throttle_domains": "",
                },
                unix_ns=lambda: next(unix_values),
                monotonic_ns=lambda: next(monotonic_values),
            )

            self.assertTrue(recorder.start())
            assert recorder._thread is not None
            recorder._thread.join(timeout=2)
            recorder.stop()

            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(calls[0][0], ["tegrastats", "--interval", "500"])
        self.assertEqual(recorder.sample_count, 2)
        self.assertEqual(rows[0]["schema_version"], "1.0")
        self.assertEqual(rows[0]["host_unix_ns"], "100")
        self.assertEqual(rows[0]["host_monotonic_ns"], "10")
        self.assertEqual(rows[0]["sample_index"], "0")
        self.assertEqual(rows[0]["emc_freq_hz"], "123")
        self.assertEqual(rows[1]["parse_status"], "malformed")
        self.assertEqual(rows[1]["raw_line"], "unknown")
        self.assertTrue(fake.terminated)

    def test_stop_escalates_to_kill_after_bounded_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fake = _FakeProcess([], timeout_once=True)
            recorder = SystemTelemetryRecorder(
                Path(temp_dir) / "system.csv",
                popen_factory=lambda *_args, **_kwargs: fake,
                platform_reader=lambda: {},
                terminate_timeout_s=0.01,
                kill_timeout_s=0.01,
            )
            self.assertTrue(recorder.start())
            recorder.stop()

        self.assertTrue(fake.terminated)
        self.assertTrue(fake.killed)
        self.assertEqual(fake.wait_calls, 2)

    def test_oversized_tegrastats_record_stops_reader_and_reports_failure(self) -> None:
        errors: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            fake = _FakeProcess(["x" * (MAX_TELEMETRY_LINE_CHARS + 1) + "\n"])
            recorder = SystemTelemetryRecorder(
                Path(temp_dir) / "system.csv",
                popen_factory=lambda *_args, **_kwargs: fake,
                platform_reader=lambda: {},
                on_error=errors.append,
            )

            self.assertTrue(recorder.start())
            assert recorder._thread is not None
            recorder._thread.join(timeout=2)
            recorder.stop()

        self.assertEqual(recorder.sample_count, 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("record exceeds", errors[0])
        self.assertIn(str(MAX_TELEMETRY_LINE_CHARS), errors[0])

    def test_missing_tegrastats_is_best_effort_and_idempotent(self) -> None:
        errors: list[str] = []

        def missing(*_args, **_kwargs):
            raise FileNotFoundError("tegrastats")

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "system.csv"
            recorder = SystemTelemetryRecorder(
                output,
                popen_factory=missing,
                on_error=errors.append,
            )
            self.assertFalse(recorder.start())
            recorder.stop()
            recorder.stop()

        self.assertIn("FileNotFoundError", recorder.last_error or "")
        self.assertEqual(len(errors), 1)

    def test_early_tegrastats_exit_is_reported(self) -> None:
        class ExitedProcess(_FakeProcess):
            def poll(self):
                return 9

        errors: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            process = ExitedProcess([])
            recorder = SystemTelemetryRecorder(
                Path(temp_dir) / "system.csv",
                popen_factory=lambda *_args, **_kwargs: process,
                platform_reader=lambda: {},
                on_error=errors.append,
            )
            self.assertTrue(recorder.start())
            assert recorder._thread is not None
            recorder._thread.join(timeout=2)
            recorder.stop()

        self.assertEqual(
            errors,
            ["system telemetry process exited unexpectedly with code 9"],
        )

    def test_process_group_signal_during_parent_shutdown_is_not_degradation(self) -> None:
        class SignaledProcess(_FakeProcess):
            def poll(self):
                return -2

        errors: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            process = SignaledProcess([])
            recorder = SystemTelemetryRecorder(
                Path(temp_dir) / "system.csv",
                popen_factory=lambda *_args, **_kwargs: process,
                platform_reader=lambda: {},
                on_error=errors.append,
                shutdown_requested=lambda: True,
            )
            self.assertTrue(recorder.start())
            assert recorder._thread is not None
            recorder._thread.join(timeout=2)
            recorder.stop()

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
