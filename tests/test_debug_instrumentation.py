from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.inference import debug_instrumentation


class _Pipeline:
    def __init__(self) -> None:
        self.attachments: list[tuple] = []

    def attach(self, *args, **kwargs):
        self.attachments.append((args, kwargs))
        return self


class DebugInstrumentationTests(unittest.TestCase):
    def test_profile_flag_is_explicit(self) -> None:
        self.assertTrue(
            debug_instrumentation.profile_enabled(
                {debug_instrumentation.DEBUG_PROFILE_ENV: "true"}
            )
        )
        self.assertFalse(debug_instrumentation.profile_enabled({}))

    def test_probe_identity_is_regular_bounded_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            modules = root / "service-maker" / "modules"
            modules.mkdir(parents=True)
            for filename in (
                "libmeasure_latency_probe.so",
                "libmeasure_fps_probe.so",
            ):
                (modules / filename).write_bytes(filename.encode())

            identities = debug_instrumentation.validate_probe_modules(sdk_root=root)

        self.assertEqual(set(identities), {"measure_latency_probe", "measure_fps_probe"})
        self.assertTrue(all(item["available"] for item in identities.values()))
        self.assertTrue(all(len(str(item["sha256"])) == 64 for item in identities.values()))

    def test_missing_probe_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "measure_latency_probe"):
                debug_instrumentation.validate_probe_modules(
                    sdk_root=Path(temp_dir)
                )

    def test_attach_uses_nvidia_factories_and_bounded_fps_interval(self) -> None:
        pipeline = _Pipeline()
        with mock.patch.object(
            debug_instrumentation, "validate_probe_modules", return_value={}
        ):
            debug_instrumentation.attach_debug_probes(pipeline, "tracker")

        self.assertEqual(
            pipeline.attachments,
            [
                (("tracker", "measure_latency_probe", "squeakview_latency"), {}),
                (
                    ("tracker", "measure_fps_probe", "squeakview_fps"),
                    {"properties": {"interval": 5}},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
