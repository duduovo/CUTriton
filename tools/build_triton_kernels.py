"""Compatibility CLI for the modular CUTriton Kernel Pack builder."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from cutriton.kernel_sdk.builder import main  # noqa: E402


if __name__ == "__main__":
    main()
