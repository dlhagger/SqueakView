"""SDK-independent recording activity snapshot."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecordingActivity:
    source_count: int
    admission_count: int
    egress_count: int
    last_source_monotonic_ns: int | None
    last_admission_monotonic_ns: int | None
    last_egress_monotonic_ns: int | None
