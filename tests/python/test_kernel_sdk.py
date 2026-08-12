import unittest

from cutriton.kernel_sdk.spec import (
    ArgumentSpec,
    KernelSpec,
    KernelVariant,
    clear_registry,
    register,
    registered_specs,
)


def _kernel():
    return None


def _spec(*, source_kind="input", variants=None):
    return KernelSpec(
        kernel_id="test.relu.fp32",
        op_type="Relu",
        version=1,
        function=_kernel,
        signature={"x": "*fp32"},
        arguments=(
            ArgumentSpec("x", "pointer", {"kind": source_kind, "index": 0}),
            ArgumentSpec("runtime_scratch", "pointer", {"kind": "runtime_reserved"}),
            ArgumentSpec("profile_scratch", "pointer", {"kind": "runtime_reserved"}),
        ),
        grid={"op": "ceil_div", "value": {"kind": "output_numel"}, "divisor": 256},
        variants=variants
        or (KernelVariant("default", default=True),),
    )


class KernelSdkTest(unittest.TestCase):
    def tearDown(self):
        clear_registry()

    def test_valid_spec_round_trips_through_registry(self):
        spec = _spec()
        register(spec)
        self.assertEqual(registered_specs(), (spec,))
        self.assertEqual(registered_specs()[0].arguments[0].source["kind"], "input")

    def test_rejects_unsafe_argument_source(self):
        with self.assertRaisesRegex(ValueError, "unsafe source"):
            _spec(source_kind="python_callable").validate()

    def test_rejects_duplicate_kernel_id(self):
        register(_spec())
        with self.assertRaisesRegex(ValueError, "duplicate kernel id"):
            register(_spec())

    def test_requires_reserved_arguments_at_abi_tail(self):
        spec = _spec()
        object.__setattr__(spec, "arguments", spec.arguments[:-1])
        with self.assertRaisesRegex(ValueError, "two trailing"):
            spec.validate()

    def test_requires_exactly_one_default_variant(self):
        with self.assertRaisesRegex(ValueError, "exactly one default"):
            _spec(
                variants=(
                    KernelVariant("a", default=True),
                    KernelVariant("b", default=True),
                )
            ).validate()

    def test_rejects_arbitrary_grid_and_constraint_expressions(self):
        spec = _spec()
        object.__setattr__(spec, "grid", {"op": "python", "callable": "evil"})
        with self.assertRaisesRegex(ValueError, "grid"):
            spec.validate()
        spec = _spec()
        object.__setattr__(
            spec, "constraints", ({"kind": "python", "expression": "x"},)
        )
        with self.assertRaisesRegex(ValueError, "unsafe constraint"):
            spec.validate()

    def test_accepts_safe_multidimensional_grid(self):
        spec = _spec(variants=(KernelVariant("tiled", meta={"BLOCK": 64}, default=True),))
        object.__setattr__(
            spec,
            "grid",
            (
                {
                    "kind": "ceil_div",
                    "args": [
                        {"kind": "output_dim", "index": 0, "axis": 0},
                        {"kind": "meta", "name": "BLOCK"},
                    ],
                },
                {"kind": "literal", "value": 2},
            ),
        )
        spec.validate()

    def test_rejects_unknown_grid_meta(self):
        spec = _spec()
        object.__setattr__(
            spec,
            "grid",
            ({"kind": "meta", "name": "UNDECLARED"},),
        )
        with self.assertRaisesRegex(ValueError, "unknown variant meta"):
            spec.validate()


if __name__ == "__main__":
    unittest.main()
