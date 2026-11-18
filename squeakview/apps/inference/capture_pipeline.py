"""Drain FLIR sources without entering Service Maker's blocking stop binding."""
from pyservicemaker import Pipeline


class CapturePipeline(Pipeline):
    """End each source at its next read boundary, preserving in-flight frames.

    Service Maker's Pipeline.stop binding retains the GIL while synchronously
    sending EOS through the graph. A streaming callback waiting for the GIL
    can therefore deadlock stop. Our source property only sets an atomic flag;
    streaming threads send EOS themselves after delivering their current frame.
    Pipeline.wait still owns the downstream drain and transition to NULL.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._capture_sources: list[str] = []

    def add(self, type_name, name, properties=None):
        super().add(type_name, name, properties)
        if type_name in {"flirspinsrc", "squeakviewlivesrc"}:
            # Reject stale plugins before recording starts, not at Stop.
            if self[name].get("graceful-eos") is not False:
                raise RuntimeError(
                    "FLIR source lacks a fresh graceful-eos control; "
                    "rebuild with bash scripts/build_native.sh"
                )
            self._capture_sources.append(name)
        return self

    def stop(self):
        failures = []
        for name in self._capture_sources:
            try:
                source = self[name]
                source.set({"graceful-eos": True})
                if source.get("graceful-eos") is not True:
                    raise RuntimeError("source did not acknowledge EOS request")
            except Exception as exc:
                failures.append(f"{name}: {exc}")
        if failures:
            raise RuntimeError("source drain request failed: " + "; ".join(failures))
        return self
