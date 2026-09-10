from __future__ import annotations

import unittest
from types import SimpleNamespace

from squeakview.apps.inference.storage_reserve import (
    DEFAULT_MIN_FREE_BYTES,
    StorageReserveMonitor,
    StorageReservePolicy,
    resolve_storage_reserve_policy,
)


class OneCycleEvent:
    def __init__(self) -> None:
        self.calls = 0

    def wait(self, _timeout: float) -> bool:
        self.calls += 1
        return self.calls > 1


class StorageReserveTests(unittest.TestCase):
    def test_policy_matches_startup_reserve_and_accepts_override(self) -> None:
        default = resolve_storage_reserve_policy({})
        override = resolve_storage_reserve_policy(
            {
                "SQUEAKVIEW_MIN_RUN_FREE_BYTES": "1234",
                "SQUEAKVIEW_STORAGE_CHECK_INTERVAL_S": "2.5",
            }
        )

        self.assertEqual(default.min_free_bytes, DEFAULT_MIN_FREE_BYTES)
        self.assertEqual(override, StorageReservePolicy(1234, 2.5))
        self.assertEqual(
            resolve_storage_reserve_policy(
                {"SQUEAKVIEW_STORAGE_CHECK_INTERVAL_S": "nan"}
            ).check_interval_s,
            StorageReservePolicy().check_interval_s,
        )

    def test_low_reserve_is_fatal(self) -> None:
        failures: list[str] = []
        monitor = StorageReserveMonitor(
            "/run",
            OneCycleEvent(),
            failures.append,
            StorageReservePolicy(1000, 0.1),
            disk_usage=lambda _path: SimpleNamespace(free=999),
        )

        monitor.run()

        self.assertEqual(len(failures), 1)
        self.assertIn("999 bytes free", failures[0])

    def test_usage_failure_is_fatal(self) -> None:
        def fail(_path):
            raise OSError("filesystem unavailable")

        failures: list[str] = []
        monitor = StorageReserveMonitor(
            "/run",
            OneCycleEvent(),
            failures.append,
            StorageReservePolicy(),
            disk_usage=fail,
        )

        monitor.run()

        self.assertEqual(len(failures), 1)
        self.assertIn("could not verify", failures[0])

    def test_sufficient_reserve_does_not_stop_capture(self) -> None:
        failures: list[str] = []
        monitor = StorageReserveMonitor(
            "/run",
            OneCycleEvent(),
            failures.append,
            StorageReservePolicy(1000, 0.1),
            disk_usage=lambda _path: SimpleNamespace(free=1000),
        )

        monitor.run()

        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
