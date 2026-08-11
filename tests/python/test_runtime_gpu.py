from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")
torch = pytest.importorskip("torch")

from cutriton import CompileOptions, ShapeProfile, ShapeRange, compile
from cutriton.errors import InputValidationError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _model(path, op_type):
    helper = onnx.helper
    tensor = onnx.TensorProto
    x = helper.make_tensor_value_info("x", tensor.FLOAT, [None, 16])
    output = helper.make_tensor_value_info("output", tensor.FLOAT, [None, 16])
    graph = helper.make_graph(
        [helper.make_node(op_type, ["x"], ["output"], name=op_type.lower())],
        op_type.lower(),
        [x],
        [output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, path)


@pytest.mark.gpu
@pytest.mark.integration
def test_ort_cuda_segment_uses_cuda_tensor_io(tmp_path):
    path = tmp_path / "exp.onnx"
    _model(path, "Exp")
    engine = compile(path, CompileOptions(cache_dir=tmp_path / "cache"))
    context = engine.create_context()
    try:
        source = torch.randn(3, 16, device="cuda")
        output = context.run({"x": source})["output"]
        assert output.device.type == "cuda"
        torch.testing.assert_close(output, torch.exp(source))
    finally:
        context.close()
        engine.close()


@pytest.mark.gpu
@pytest.mark.integration
def test_triton_signature_is_qualified_and_cached(tmp_path):
    path = tmp_path / "relu.onnx"
    _model(path, "Relu")
    options = CompileOptions(
        cache_dir=tmp_path / "cache",
        profiles=[ShapeProfile("default", {"x": ShapeRange((1, 16), (8, 16), (32, 16))})],
    )
    engine = compile(path, options)
    try:
        engine.warmup()
        # Each min/opt/max shape has one kernel and one whole-plan decision.
        assert engine.cache.count() == 6
        assert engine.stats.snapshot()["cuda_graph_replays"] >= 1
        context = engine.create_context()
        try:
            source = torch.randn(8, 16, device="cuda")
            source.view(-1)[:3] = torch.tensor(
                [float("nan"), float("inf"), -float("inf")], device="cuda"
            )
            output = context.run({"x": source})["output"]
            torch.testing.assert_close(output, torch.relu(source), equal_nan=True)
            with pytest.raises(InputValidationError, match="ShapeProfile"):
                context.run({"x": torch.randn(64, 16, device="cuda")})
        finally:
            context.close()
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.integration
def test_layer_norm_initializer_and_triton_kernel(tmp_path):
    path = tmp_path / "layer_norm.onnx"
    helper = onnx.helper
    tensor = onnx.TensorProto
    x = helper.make_tensor_value_info("x", tensor.FLOAT, [None, 32])
    output = helper.make_tensor_value_info("output", tensor.FLOAT, [None, 32])
    scale = helper.make_tensor("scale", tensor.FLOAT, [32], [1.0] * 32)
    bias = helper.make_tensor("bias", tensor.FLOAT, [32], [0.0] * 32)
    graph = helper.make_graph(
        [helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["output"], axis=-1)],
        "layer_norm",
        [x],
        [output],
        initializer=[scale, bias],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)
    engine = compile(
        path,
        CompileOptions(
            cache_dir=tmp_path / "cache",
            profiles=[ShapeProfile("default", {"x": ShapeRange((1, 32), (4, 32), (8, 32))})],
        ),
    )
    try:
        engine.warmup()
        context = engine.create_context()
        try:
            source = torch.randn(4, 32, device="cuda")
            output_value = context.run({"x": source})["output"]
            expected = torch.nn.functional.layer_norm(source, (32,))
            torch.testing.assert_close(output_value, expected, rtol=1e-4, atol=1e-4)
        finally:
            context.close()
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize(
    ("elem_type", "dtype", "tolerance"),
    [
        (onnx.TensorProto.FLOAT16, torch.float16, 1e-2),
        (onnx.TensorProto.BFLOAT16, torch.bfloat16, 2e-2),
    ],
)
def test_jit_precision_policies(tmp_path, elem_type, dtype, tolerance):
    path = tmp_path / f"relu-{dtype}.onnx"
    helper = onnx.helper
    x = helper.make_tensor_value_info("x", elem_type, [4, 32])
    output = helper.make_tensor_value_info("output", elem_type, [4, 32])
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["output"])], "precision", [x], [output]
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)
    fixed = ShapeRange((4, 32), (4, 32), (4, 32))
    engine = compile(
        path,
        CompileOptions(
            cache_dir=tmp_path / "cache",
            profiles=[ShapeProfile("fixed", {"x": fixed})],
        ),
    )
    try:
        engine.warmup()
        context = engine.create_context()
        try:
            source = torch.randn(4, 32, dtype=dtype, device="cuda")
            actual = context.run({"x": source})["output"]
            torch.testing.assert_close(actual, torch.relu(source), rtol=tolerance, atol=tolerance)
        finally:
            context.close()
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.integration
def test_gemm_transpose_bias_epilogue(tmp_path):
    path = tmp_path / "gemm.onnx"
    helper = onnx.helper
    tensor = onnx.TensorProto
    x = helper.make_tensor_value_info("x", tensor.FLOAT16, [None, 32])
    output = helper.make_tensor_value_info("output", tensor.FLOAT16, [None, 48])
    weight_values = torch.linspace(-0.1, 0.1, 48 * 32).reshape(48, 32)
    bias_values = torch.linspace(-0.2, 0.2, 48)
    weight = helper.make_tensor(
        "weight", tensor.FLOAT16, [48, 32], weight_values.flatten().tolist()
    )
    bias = helper.make_tensor("bias", tensor.FLOAT16, [48], bias_values.tolist())
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gemm",
                ["x", "weight", "bias"],
                ["output"],
                transB=1,
                alpha=0.5,
                beta=2.0,
            )
        ],
        "gemm",
        [x],
        [output],
        initializer=[weight, bias],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)
    fixed = ShapeRange((8, 32), (8, 32), (8, 32))
    engine = compile(
        path,
        CompileOptions(
            cache_dir=tmp_path / "cache",
            profiles=[ShapeProfile("fixed", {"x": fixed})],
        ),
    )
    try:
        source = torch.randn(8, 32, dtype=torch.float16, device="cuda")
        segment_inputs = {"x": source, **engine.initializers}
        with torch.cuda.stream(torch.cuda.current_stream()):
            actual = engine._run_triton(engine.segments[0], segment_inputs)["output"]
        expected = 0.5 * (source @ engine.initializers["weight"].T)
        expected += 2.0 * engine.initializers["bias"]
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.integration
def test_global_average_pool_and_flatten_fusion(tmp_path):
    path = tmp_path / "pool_flatten.onnx"
    helper = onnx.helper
    tensor = onnx.TensorProto
    x = helper.make_tensor_value_info("x", tensor.FLOAT, [None, 3, 8, 8])
    pooled = helper.make_tensor_value_info("pooled", tensor.FLOAT, [None, 3, 1, 1])
    output = helper.make_tensor_value_info("output", tensor.FLOAT, [None, 3])
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["x"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["output"], axis=1),
        ],
        "pool_flatten",
        [x],
        [output],
        value_info=[pooled],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)
    engine = compile(path, CompileOptions(cache_dir=tmp_path / "cache"))
    try:
        assert len(engine.segments) == 1
        source = torch.randn(4, 3, 8, 8, device="cuda")
        actual = engine._run_triton(engine.segments[0], {"x": source})["output"]
        expected = source.mean(dim=(-2, -1))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        assert actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.integration
def test_rank4_elementwise_broadcast(tmp_path):
    path = tmp_path / "broadcast.onnx"
    helper = onnx.helper
    tensor = onnx.TensorProto
    x = helper.make_tensor_value_info("x", tensor.FLOAT, [2, 3, 4])
    output = helper.make_tensor_value_info("output", tensor.FLOAT, [2, 3, 4])
    bias = helper.make_tensor("bias", tensor.FLOAT, [4], [0.5, -1.0, 2.0, 3.0])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "bias"], ["output"])],
        "broadcast",
        [x],
        [output],
        initializer=[bias],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)
    engine = compile(path, CompileOptions(cache_dir=tmp_path / "cache"))
    try:
        source = torch.randn(2, 3, 4, device="cuda")
        actual = engine._run_triton(
            engine.segments[0], {"x": source, "bias": engine.initializers["bias"]}
        )["output"]
        torch.testing.assert_close(actual, source + engine.initializers["bias"])
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.integration
def test_hot_start_reuses_decisions_without_compilation(tmp_path):
    path = tmp_path / "relu.onnx"
    _model(path, "Relu")
    fixed = ShapeRange((8, 16), (8, 16), (8, 16))
    options = CompileOptions(
        cache_dir=tmp_path / "cache",
        profiles=[ShapeProfile("fixed", {"x": fixed})],
    )
    first = compile(path, options)
    first.warmup()
    first.close()
    second = compile(path, options)
    try:
        second.warmup()
        stats = second.stats.snapshot()
        assert stats["compile_seconds"] == 0
        assert stats["cache_misses"] == 0
        assert stats["cuda_graph_replays"] >= 1
    finally:
        second.close()


@pytest.mark.gpu
@pytest.mark.integration
def test_repeated_cuda_graph_runs_do_not_leak_allocated_memory(tmp_path):
    path = tmp_path / "relu.onnx"
    _model(path, "Relu")
    fixed = ShapeRange((8, 16), (8, 16), (8, 16))
    engine = compile(
        path,
        CompileOptions(
            cache_dir=tmp_path / "cache",
            profiles=[ShapeProfile("fixed", {"x": fixed})],
        ),
    )
    engine.warmup()
    context = engine.create_context()
    try:
        source = torch.randn(8, 16, device="cuda")
        for _ in range(10):
            context.run({"x": source})
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        for _ in range(200):
            output = context.run({"x": source})["output"]
            del output
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() <= baseline + 1024 * 1024
    finally:
        context.close()
        engine.close()
