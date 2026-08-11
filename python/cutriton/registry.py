from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class NodeLike(Protocol):
    op_type: str
    domain: str


Capability = Callable[[NodeLike, Mapping[str, Any]], tuple[bool, str]]
Executor = Callable[[NodeLike, list[Any], Mapping[str, Any]], list[Any]]


@dataclass(frozen=True)
class KernelSpec:
    kernel_id: str
    op_types: tuple[str, ...]
    execute: Executor
    capability: Capability
    domains: tuple[str, ...] = ("", "ai.onnx")
    min_opset: int = 1
    max_opset: int = 2**31 - 1
    configs: tuple[Mapping[str, int], ...] = field(default_factory=tuple)

    def supports(self, node: NodeLike, metadata: Mapping[str, Any], opset: int) -> tuple[bool, str]:
        if node.domain not in self.domains or node.op_type not in self.op_types:
            return False, "operator/domain mismatch"
        if not self.min_opset <= opset <= self.max_opset:
            return False, f"opset {opset} outside supported range"
        return self.capability(node, metadata)


class KernelRegistry:
    def __init__(self) -> None:
        self._specs: list[KernelSpec] = []

    def register(self, spec: KernelSpec) -> None:
        if not spec.kernel_id or not spec.op_types:
            raise ValueError("kernel spec identity is incomplete")
        if any(existing.kernel_id == spec.kernel_id for existing in self._specs):
            raise ValueError(f"duplicate kernel id: {spec.kernel_id}")
        self._specs.append(spec)

    def match(
        self, node: NodeLike, metadata: Mapping[str, Any], opset: int
    ) -> tuple[KernelSpec | None, str]:
        reasons: list[str] = []
        for spec in self._specs:
            if node.op_type not in spec.op_types or node.domain not in spec.domains:
                continue
            supported, reason = spec.supports(node, metadata, opset)
            if supported:
                return spec, "supported"
            reasons.append(f"{spec.kernel_id}: {reason}")
        return None, "; ".join(reasons) or "no registered Triton kernel"

    def specs(self) -> tuple[KernelSpec, ...]:
        return tuple(self._specs)

    def extend(self, specs: Iterable[KernelSpec]) -> None:
        for spec in specs:
            self.register(spec)
