from __future__ import annotations

import pytest
from cutriton import BatchingOptions, CompileOptions, FallbackPolicy, ShapeProfile, ShapeRange


def test_compile_options_normalize_values(tmp_path):
    profile = ShapeProfile(
        "resnet",
        {"input": ShapeRange((1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224))},
    )
    options = CompileOptions(fallback="ort_cuda_cpu", cache_dir=tmp_path, profiles=[profile])
    assert options.fallback is FallbackPolicy.ORT_CUDA_CPU
    assert options.cache_dir == tmp_path


@pytest.mark.parametrize(
    "shape_range",
    [
        ((0,), (1,), (2,)),
        ((2,), (1,), (3,)),
        ((1, 2), (1,), (2,)),
    ],
)
def test_invalid_shape_range_is_rejected(shape_range):
    with pytest.raises(ValueError):
        ShapeRange(*shape_range)


def test_batch_preferences_must_be_ordered_and_bounded():
    with pytest.raises(ValueError):
        BatchingOptions(max_batch_size=8, preferred_batch_sizes=(8, 4))
