"""CUTriton 的 Python 门面。

生产运行时以 C++ 为主。这个包提供 V1 阶段的 Python 入口，以及算子开发时使用的 Triton Kernel 工作区。
"""

from .api import Engine, ExecutionContext, compile

__all__ = ["Engine", "ExecutionContext", "compile"]
