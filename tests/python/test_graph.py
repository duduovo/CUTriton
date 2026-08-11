from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")

from cutriton.config import CompileOptions
from cutriton.graph import SegmentKind, load_model, partition_graph
from cutriton.triton_backend import create_default_registry


def _write_model(path):
    helper = onnx.helper
    tensor = onnx.TensorProto
    x = helper.make_tensor_value_info("x", tensor.FLOAT, [None, 4])
    y = helper.make_tensor_value_info("y", tensor.FLOAT, [None, 4])
    z = helper.make_tensor_value_info("z", tensor.FLOAT, [None, 4])
    output = helper.make_tensor_value_info("output", tensor.FLOAT, [None, 4])
    nodes = [
        helper.make_node("Add", ["x", "y"], ["z"], name="add"),
        helper.make_node("Relu", ["z"], ["output"], name="relu"),
    ]
    graph = helper.make_graph(nodes, "test", [x, y], [output], value_info=[z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_supported_nodes_form_a_maximal_triton_segment(tmp_path):
    path = tmp_path / "model.onnx"
    _write_model(path)
    graph = load_model(path, CompileOptions(cache_dir=tmp_path / "cache"))
    segments = partition_graph(graph, create_default_registry())
    assert len(segments) == 1
    assert segments[0].kind is SegmentKind.TRITON
    assert segments[0].inputs == ("x", "y")
    assert segments[0].outputs == ("output",)


def test_external_data_cannot_escape_model_directory(tmp_path):
    path = tmp_path / "model.onnx"
    _write_model(path)
    model = onnx.load(path)
    weight = onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT, [1], [1.0])
    weight.data_location = onnx.TensorProto.EXTERNAL
    location = weight.external_data.add()
    location.key = "location"
    location.value = "../secret.bin"
    model.graph.initializer.append(weight)
    onnx.save_model(model, path, save_as_external_data=False)
    with pytest.raises(Exception, match="escapes"):
        load_model(path, CompileOptions(cache_dir=tmp_path / "cache"))
