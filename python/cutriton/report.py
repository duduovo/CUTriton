from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SegmentReport:
    index: int
    backend: str
    nodes: tuple[str, ...]
    reason: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class CompileReport:
    model_path: str
    model_hash: str
    device_id: int
    segments: tuple[SegmentReport, ...]
    cpu_fallback_allowed: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def triton_segment_count(self) -> int:
        return sum(item.backend == "triton" for item in self.segments)

    @property
    def ort_segment_count(self) -> int:
        return sum(item.backend == "ort" for item in self.segments)
