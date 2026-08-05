from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
from typing import Any

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from .spec import registered_specs


PACK_SCHEMA_VERSION = 2
ABI_SCHEMA_VERSION = 1
GENERATOR_VERSION = "0.2.0"
DEFAULT_MODULES = ("cutriton.triton_kernels.resnet",)


def _compile_spec(spec: Any, output: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    kernel_dir = output / "kernels" / spec.kernel_id
    kernel_dir.mkdir(parents=True, exist_ok=True)
    for variant in spec.variants:
        compiled = triton.compile(
            ASTSource(spec.function, spec.signature),
            target=GPUTarget("cuda", spec.min_compute_capability, 32),
            options={
                "num_warps": variant.num_warps,
                "num_stages": variant.num_stages,
                **variant.meta,
            },
        )
        ptx = compiled.asm["ptx"]
        relative = Path("kernels") / spec.kernel_id / f"{variant.variant_id}.ptx"
        (output / relative).write_text(ptx, encoding="utf-8")
        arguments = [
            {"name": arg.name, "type": arg.type, "source": arg.source}
            for arg in spec.arguments
        ]
        entries.append(
            {
                "kernel_id": spec.kernel_id,
                "op_type": spec.op_type,
                "kernel_version": spec.version,
                "variant_id": variant.variant_id,
                "default": variant.default,
                "symbol": compiled.name,
                "binary": relative.as_posix(),
                "binary_format": "ptx",
                "sha256": hashlib.sha256(ptx.encode("utf-8")).hexdigest(),
                "dtype": spec.dtype,
                "layout": spec.layout,
                "min_compute_capability": spec.min_compute_capability,
                "num_warps": variant.num_warps,
                "num_stages": variant.num_stages,
                "shared_memory_bytes": int(compiled.metadata.shared),
                "arguments": arguments,
                "grid": spec.grid,
                "constraints": list(spec.constraints),
                "tuning": dict(variant.meta),
            }
        )
    return entries


def build_pack(output: Path, modules: tuple[str, ...] = DEFAULT_MODULES) -> None:
    for module in modules:
        importlib.import_module(module)
    specs = registered_specs()
    if not specs:
        raise RuntimeError("no KernelSpec was registered")
    if output.exists():
        # Only generated pack content is replaced. CMake points this at its build tree.
        shutil.rmtree(output)
    output.mkdir(parents=True)
    kernels: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: item.kernel_id):
        kernels.extend(_compile_spec(spec, output))
    pack = {
        "schema_version": PACK_SCHEMA_VERSION,
        "abi_schema_version": ABI_SCHEMA_VERSION,
        "pack_name": "cutriton_builtin_fp32",
        "pack_version": 1,
        "generator_version": GENERATOR_VERSION,
        "triton_version": triton.__version__,
        "kernels": kernels,
    }
    (output / "pack.json").write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", action="append", dest="modules")
    args = parser.parse_args()
    build_pack(args.output, tuple(args.modules or DEFAULT_MODULES))


if __name__ == "__main__":
    main()
