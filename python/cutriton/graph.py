from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .config import CompileOptions
from .errors import ModelValidationError
from .registry import KernelRegistry, KernelSpec


@dataclass(frozen=True)
class TensorMeta:
    name: str
    elem_type: int
    shape: tuple[int | str | None, ...]
    initializer: bool = False

    @property
    def dynamic(self) -> bool:
        return any(not isinstance(dimension, int) or dimension <= 0 for dimension in self.shape)


@dataclass(frozen=True)
class NodeView:
    index: int
    name: str
    op_type: str
    domain: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    proto: Any = field(repr=False, compare=False)


class SegmentKind(str, Enum):
    TRITON = "triton"
    ORT = "ort"


@dataclass(frozen=True)
class GraphSegment:
    index: int
    kind: SegmentKind
    nodes: tuple[NodeView, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    model_bytes: bytes
    specs: tuple[KernelSpec | None, ...]
    reason: str


@dataclass(frozen=True)
class ModelGraph:
    path: Path
    model: Any = field(repr=False, compare=False)
    model_hash: str
    opsets: Mapping[str, int]
    values: Mapping[str, TensorMeta]
    nodes: tuple[NodeView, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    initializers: frozenset[str]

    @property
    def default_opset(self) -> int:
        return int(self.opsets.get("", self.opsets.get("ai.onnx", 1)))


def _onnx() -> Any:
    try:
        import onnx
    except ImportError as error:
        raise ModelValidationError("ONNX support requires the 'onnx' package") from error
    return onnx


def _safe_external_files(model: Any, root: Path, limit: int) -> list[Path]:
    onnx = _onnx()
    root = root.resolve()
    files: list[Path] = []
    total = 0
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {item.key: item.value for item in tensor.external_data}
        location = fields.get("location")
        if not location:
            raise ModelValidationError(f"initializer {tensor.name!r} has no external location")
        candidate = (root / location).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ModelValidationError("external ONNX data escapes the model directory") from error
        if not candidate.is_file():
            raise ModelValidationError(f"external ONNX data not found: {candidate}")
        total += candidate.stat().st_size
        if total > limit:
            raise ModelValidationError("external ONNX data exceeds configured size limit")
        files.append(candidate)
    return files


def _tensor_meta(value: Any, initializer: bool = False) -> TensorMeta | None:
    tensor_type = value.type.tensor_type
    if not tensor_type.elem_type:
        return None
    dimensions: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return TensorMeta(value.name, int(tensor_type.elem_type), tuple(dimensions), initializer)


def load_model(path: Path | str, options: CompileOptions) -> ModelGraph:
    onnx = _onnx()
    model_path = Path(path).expanduser().resolve()
    if options.trusted_model_root is not None:
        try:
            model_path.relative_to(options.trusted_model_root)
        except ValueError as error:
            raise ModelValidationError("model path is outside trusted_model_root") from error
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if model_path.suffix.lower() != ".onnx":
        raise ModelValidationError("the JIT runtime accepts ONNX models only")
    if model_path.stat().st_size > options.max_model_bytes:
        raise ModelValidationError("ONNX model exceeds configured size limit")

    model = onnx.load(model_path, load_external_data=False)
    external_files = _safe_external_files(model, model_path.parent, options.max_external_data_bytes)
    if external_files:
        onnx.load_external_data_for_model(model, str(model_path.parent))
    try:
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    except Exception as error:
        raise ModelValidationError(f"invalid ONNX model: {error}") from error

    digest = hashlib.sha256(model_path.read_bytes())
    for external in sorted(external_files):
        digest.update(str(external.relative_to(model_path.parent)).encode())
        digest.update(external.read_bytes())

    initializer_names = frozenset(item.name for item in inferred.graph.initializer)
    values: dict[str, TensorMeta] = {}
    for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        meta = _tensor_meta(value, value.name in initializer_names)
        if meta is not None:
            values[value.name] = meta
    for tensor in inferred.graph.initializer:
        values[tensor.name] = TensorMeta(
            tensor.name, int(tensor.data_type), tuple(int(item) for item in tensor.dims), True
        )

    nodes = tuple(
        NodeView(
            index,
            node.name or f"{node.op_type}_{index}",
            node.op_type,
            node.domain,
            tuple(name for name in node.input if name),
            tuple(name for name in node.output if name),
            node,
        )
        for index, node in enumerate(inferred.graph.node)
    )
    graph_inputs = tuple(
        value.name for value in inferred.graph.input if value.name not in initializer_names
    )
    return ModelGraph(
        model_path,
        inferred,
        digest.hexdigest(),
        {item.domain: int(item.version) for item in inferred.opset_import},
        values,
        nodes,
        graph_inputs,
        tuple(value.name for value in inferred.graph.output),
        initializer_names,
    )


def _segment_boundaries(
    graph: ModelGraph, nodes: Iterable[NodeView]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected = tuple(nodes)
    produced = {name for node in selected for name in node.outputs}
    inputs: list[str] = []
    for node in selected:
        for name in node.inputs:
            if name not in produced and name not in inputs:
                inputs.append(name)

    selected_indices = {node.index for node in selected}
    outside_inputs = {
        name for node in graph.nodes if node.index not in selected_indices for name in node.inputs
    }
    outputs = [
        name
        for node in selected
        for name in node.outputs
        if name in outside_inputs or name in graph.outputs
    ]
    if not outputs and selected:
        outputs.extend(selected[-1].outputs)
    return tuple(inputs), tuple(dict.fromkeys(outputs))


def _value_info(graph: ModelGraph, name: str) -> Any:
    onnx = _onnx()
    for collection in (
        graph.model.graph.input,
        graph.model.graph.value_info,
        graph.model.graph.output,
    ):
        for value in collection:
            if value.name == name:
                result = onnx.ValueInfoProto()
                result.CopyFrom(value)
                return result
    meta = graph.values.get(name)
    if meta is None:
        raise ModelValidationError(f"missing type information for segment value {name!r}")
    shape = [item if isinstance(item, int | str) else None for item in meta.shape]
    return onnx.helper.make_tensor_value_info(name, meta.elem_type, shape)


def _segment_model(
    graph: ModelGraph,
    nodes: tuple[NodeView, ...],
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
) -> bytes:
    onnx = _onnx()
    initializer_by_name = {item.name: item for item in graph.model.graph.initializer}
    initializers = [initializer_by_name[name] for name in inputs if name in initializer_by_name]
    runtime_inputs = [
        _value_info(graph, name) for name in inputs if name not in initializer_by_name
    ]
    graph_outputs = [_value_info(graph, name) for name in outputs]
    selected_values = {name for node in nodes for name in (*node.inputs, *node.outputs)}
    value_info = [
        _value_info(graph, name)
        for name in selected_values
        if name not in inputs and name not in outputs and name in graph.values
    ]
    subgraph = onnx.helper.make_graph(
        [node.proto for node in nodes],
        f"cutriton_segment_{nodes[0].index}_{nodes[-1].index}",
        runtime_inputs,
        graph_outputs,
        initializer=initializers,
        value_info=value_info,
    )
    model = onnx.helper.make_model(
        subgraph,
        opset_imports=[
            onnx.helper.make_opsetid(domain, version) for domain, version in graph.opsets.items()
        ],
        producer_name="cutriton",
    )
    model.ir_version = graph.model.ir_version
    onnx.checker.check_model(model)
    return model.SerializeToString()


def partition_graph(graph: ModelGraph, registry: KernelRegistry) -> tuple[GraphSegment, ...]:
    control_flow = {"If", "Loop", "Scan"}
    classified: list[tuple[SegmentKind, NodeView, KernelSpec | None, str]] = []
    for node in graph.nodes:
        if node.op_type in control_flow:
            classified.append((SegmentKind.ORT, node, None, "control flow is delegated to ORT"))
            continue
        spec, reason = registry.match(node, graph.values, graph.default_opset)
        classified.append(
            (SegmentKind.TRITON if spec is not None else SegmentKind.ORT, node, spec, reason)
        )

    groups: list[list[tuple[SegmentKind, NodeView, KernelSpec | None, str]]] = []
    for item in classified:
        if not groups or groups[-1][0][0] != item[0]:
            groups.append([item])
        else:
            groups[-1].append(item)

    segments: list[GraphSegment] = []
    for index, group in enumerate(groups):
        kind = group[0][0]
        nodes = tuple(item[1] for item in group)
        inputs, outputs = _segment_boundaries(graph, nodes)
        reasons = "; ".join(dict.fromkeys(item[3] for item in group))
        segments.append(
            GraphSegment(
                index,
                kind,
                nodes,
                inputs,
                outputs,
                _segment_model(graph, nodes, inputs, outputs),
                tuple(item[2] for item in group),
                reasons,
            )
        )
    return tuple(segments)
