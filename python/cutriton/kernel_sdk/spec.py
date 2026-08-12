from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ArgumentSpec:
    """One ordered PTX argument and its safe runtime binding source."""

    name: str
    type: str
    source: dict[str, Any]


@dataclass(frozen=True)
class KernelVariant:
    variant_id: str
    num_warps: int = 4
    num_stages: int = 1
    meta: dict[str, int] = field(default_factory=dict)
    default: bool = False


@dataclass(frozen=True)
class KernelSpec:
    kernel_id: str
    op_type: str
    version: int
    function: Callable[..., Any]
    signature: dict[str, str]
    arguments: tuple[ArgumentSpec, ...]
    grid: dict[str, Any] | tuple[dict[str, Any], ...]
    variants: tuple[KernelVariant, ...]
    dtype: str = "float32"
    layout: str = ""
    min_compute_capability: int = 80
    constraints: tuple[dict[str, Any], ...] = ()

    def validate(self) -> None:
        if not self.kernel_id or not self.op_type or self.version <= 0:
            raise ValueError("KernelSpec identity is incomplete")
        user_arguments = tuple(
            arg.name for arg in self.arguments
            if arg.source.get("kind") != "runtime_reserved"
        )
        if tuple(self.signature) != user_arguments:
            raise ValueError(f"{self.kernel_id}: signature and arguments differ")
        if not self.variants or sum(item.default for item in self.variants) != 1:
            raise ValueError(f"{self.kernel_id}: exactly one default variant is required")
        variant_ids = {item.variant_id for item in self.variants}
        if len(variant_ids) != len(self.variants):
            raise ValueError(f"{self.kernel_id}: duplicate variant id")
        allowed_sources = {
            "input", "output", "input_dim", "input_dim_product",
            "output_dim", "output_numel",
            "attribute_i64", "attribute_f64", "literal_i32",
            "runtime_reserved",
        }
        source_fields = {
            "input": {"kind", "index"},
            "output": {"kind", "index"},
            "input_dim": {"kind", "index", "axis"},
            "input_dim_product": {"kind", "index", "begin_axis"},
            "output_dim": {"kind", "index", "axis"},
            "output_numel": {"kind", "index"},
            "attribute_i64": {"kind", "name", "alternate_name", "index", "default"},
            "attribute_f64": {"kind", "name", "alternate_name", "index", "default"},
            "literal_i32": {"kind", "default"},
            "runtime_reserved": {"kind"},
        }
        for argument in self.arguments:
            kind = argument.source.get("kind")
            if kind not in allowed_sources:
                raise ValueError(
                    f"{self.kernel_id}: unsafe source for {argument.name}"
                )
            unknown = set(argument.source) - source_fields[kind]
            if unknown:
                raise ValueError(
                    f"{self.kernel_id}: unknown source field for {argument.name}: {sorted(unknown)}"
                )
        reserved_positions = [
            index
            for index, argument in enumerate(self.arguments)
            if argument.source.get("kind") == "runtime_reserved"
        ]
        if reserved_positions != [len(self.arguments) - 2, len(self.arguments) - 1]:
            raise ValueError(
                f"{self.kernel_id}: ABI schema 1 requires two trailing runtime_reserved arguments"
            )
        _validate_grid(self.kernel_id, self.grid, self.variants)
        allowed_constraints = {
            "attribute_i64_eq": {"kind", "name", "value", "default"},
            "input_count_eq": {"kind", "value"},
            "tensor_rank_eq": {"kind", "source", "index", "value"},
        }
        for constraint in self.constraints:
            kind = constraint.get("kind")
            if kind not in allowed_constraints or set(constraint) - allowed_constraints.get(kind, set()):
                raise ValueError(f"{self.kernel_id}: unsafe constraint")
            if kind == "tensor_rank_eq" and constraint.get("source") not in {"input", "output"}:
                raise ValueError(f"{self.kernel_id}: invalid tensor constraint source")


_REGISTRY: list[KernelSpec] = []


def _validate_grid(
    kernel_id: str,
    grid: dict[str, Any] | tuple[dict[str, Any], ...],
    variants: tuple[KernelVariant, ...],
) -> None:
    # ABI v1 shorthand remains accepted by the SDK and is normalized by the
    # builder before an ABI v2 pack is emitted.
    if isinstance(grid, dict) and set(grid) == {"op", "value", "divisor"}:
        if (
            grid.get("op") != "ceil_div"
            or not isinstance(grid.get("value"), dict)
            or grid["value"].get("kind") != "output_numel"
            or set(grid["value"]) - {"kind", "index"}
            or not isinstance(grid.get("divisor"), int)
            or grid["divisor"] <= 0
        ):
            raise ValueError(f"{kernel_id}: invalid legacy ceil_div grid")
        return
    if not isinstance(grid, tuple) or not 1 <= len(grid) <= 3:
        raise ValueError(f"{kernel_id}: grid must contain one to three axes")
    meta_names = {name for variant in variants for name in variant.meta}

    def validate(expression: Any, depth: int = 0) -> None:
        if depth > 8 or not isinstance(expression, dict):
            raise ValueError(f"{kernel_id}: invalid grid expression")
        kind = expression.get("kind")
        leaf_fields = {
            "literal": {"kind", "value"},
            "input_dim": {"kind", "index", "axis"},
            "output_dim": {"kind", "index", "axis"},
            "input_numel": {"kind", "index"},
            "output_numel": {"kind", "index"},
            "meta": {"kind", "name"},
        }
        if kind in leaf_fields:
            if set(expression) != leaf_fields[kind]:
                raise ValueError(f"{kernel_id}: invalid {kind} grid leaf")
            if kind == "literal" and (
                not isinstance(expression["value"], int) or expression["value"] <= 0
            ):
                raise ValueError(f"{kernel_id}: grid literals must be positive")
            if kind in {"input_dim", "output_dim"} and (
                not isinstance(expression["index"], int)
                or not isinstance(expression["axis"], int)
            ):
                raise ValueError(f"{kernel_id}: grid tensor coordinates must be integers")
            if kind in {"input_numel", "output_numel"} and not isinstance(
                expression["index"], int
            ):
                raise ValueError(f"{kernel_id}: grid Tensor index must be an integer")
            if kind == "meta" and expression["name"] not in meta_names:
                raise ValueError(f"{kernel_id}: grid references unknown variant meta")
            return
        if kind not in {"ceil_div", "mul"} or set(expression) != {"kind", "args"}:
            raise ValueError(f"{kernel_id}: unsafe grid expression")
        args = expression["args"]
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError(f"{kernel_id}: {kind} requires two operands")
        for operand in args:
            validate(operand, depth + 1)

    for axis in grid:
        validate(axis)


def register(spec: KernelSpec) -> KernelSpec:
    spec.validate()
    if any(item.kernel_id == spec.kernel_id for item in _REGISTRY):
        raise ValueError(f"duplicate kernel id: {spec.kernel_id}")
    _REGISTRY.append(spec)
    return spec


def registered_specs() -> tuple[KernelSpec, ...]:
    return tuple(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
