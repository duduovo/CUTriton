from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from cutriton.api import Engine


def test_same_signature_uses_one_background_compilation():
    engine = object.__new__(Engine)
    engine._compile_lock = threading.Lock()
    engine._compiling = {}
    engine._compiler = ThreadPoolExecutor(max_workers=1)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def qualify(key, segment, inputs, ready_event):
        calls.append(key)
        started.set()
        release.wait(timeout=5)

    engine._qualify = qualify
    first = engine._submit_qualification("same-shape", object(), {"x": torch.ones(1)}, object())
    assert started.wait(timeout=5)
    second = engine._submit_qualification("same-shape", object(), {"x": torch.ones(1)}, object())
    assert first is second
    release.set()
    first.result(timeout=5)
    engine._compiler.shutdown(wait=True)
    assert calls == ["same-shape"]
