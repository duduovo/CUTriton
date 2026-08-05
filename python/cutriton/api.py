from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping


SUPPORTED_TARGETS = {"cuda_triton", "cpu_reference"}


@dataclass(frozen=True)
class Engine:
    """Python 侧的已编译计划。

    在 pybind11 桥接落地前，这个对象会刻意保持轻量。它对齐 C++ API 形态，
    这样后续暴露原生引擎时，用户代码不需要大幅调整。
    """

    model_path: Path
    target: str
    plan: Dict[str, Any] = field(default_factory=dict)

    def create_context(self) -> "ExecutionContext":
        return ExecutionContext(self)


class ExecutionContext:
    def __init__(self, engine: Engine):
        self.engine = engine

    def run(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        values: MutableMapping[str, Any] = dict(inputs)
        constants = self.engine.plan.get("constants", {})
        values.update(constants)

        for node in self.engine.plan.get("nodes", []):
            op_type = node["op_type"]
            node_inputs = [values[name] for name in node.get("inputs", [])]
            outputs = node.get("outputs", [])
            if not outputs:
                continue
            values[outputs[0]] = _run_reference_op(op_type, node_inputs, node)

        requested_outputs = self.engine.plan.get("outputs", [])
        if requested_outputs:
            return {name: values[name] for name in requested_outputs if name in values}
        return dict(values)


def compile(model_path: str | Path, target: str = "cuda_triton", **options: Any) -> Engine:
    """将模型文件编译成 Engine 形态的对象。

    V1 支持轻量 JSON IR，便于测试和工具链验证。ONNX 文件可以被识别，
    但需要可选的 `onnx` 包，目前只抽取图结构；完整 ONNX lowering 属于原生编译器后续工作。
    """

    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"不支持的目标后端 {target!r}；可选值为 {sorted(SUPPORTED_TARGETS)}")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        plan = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".onnx":
        plan = _load_onnx_structure(path)
    else:
        raise ValueError("CUTriton V1 Python 门面仅接受 .json 或 .onnx 模型")

    plan.setdefault("target", target)
    plan.setdefault("options", dict(options))
    return Engine(model_path=path, target=target, plan=plan)


def _load_onnx_structure(path: Path) -> Dict[str, Any]:
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("需要安装 cutriton[onnx] 才能在 Python 中解析 ONNX 模型结构") from exc

    model = onnx.load(str(path))
    graph = model.graph
    return {
        "name": graph.name or path.stem,
        "inputs": [value.name for value in graph.input],
        "outputs": [value.name for value in graph.output],
        "nodes": [
            {
                "name": node.name or f"{node.op_type}_{idx}",
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": [attr.name for attr in node.attribute],
            }
            for idx, node in enumerate(graph.node)
        ],
    }


def _run_reference_op(op_type: str, inputs: Iterable[Any], node: Mapping[str, Any]) -> Any:
    values = list(inputs)
    if op_type in {"Identity", "Relu"}:
        return values[0] if op_type == "Identity" else _relu(values[0])
    if op_type == "Gelu":
        from .kernels.gelu import gelu

        return gelu(values[0])
    if op_type == "Softmax":
        from .kernels.softmax import softmax

        axis = int(node.get("axis", -1))
        return softmax(values[0], axis=axis)
    if op_type == "LayerNormalization":
        from .kernels.layer_norm import layer_norm

        eps = float(node.get("epsilon", 1e-5))
        return layer_norm(values[0], epsilon=eps)
    if op_type == "Add":
        return _binary(values[0], values[1], lambda a, b: a + b)
    raise NotImplementedError(f"尚未实现 Python 参考算子: {op_type}")


def _relu(x: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        return np.maximum(x, 0)
    except ImportError:
        return _map_nested(x, lambda value: value if value > 0 else 0)


def _binary(a: Any, b: Any, fn):
    try:
        import numpy as np  # type: ignore

        return fn(np.asarray(a), np.asarray(b))
    except ImportError:
        return fn(a, b)


def _map_nested(value: Any, fn):
    if isinstance(value, list):
        return [_map_nested(item, fn) for item in value]
    return fn(value)
