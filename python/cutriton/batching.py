from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import BatchingOptions
from .errors import ServiceOverloadedError


@dataclass
class _Request:
    inputs: Mapping[str, Any]
    future: asyncio.Future[dict[str, Any]]
    deadline: float | None
    enqueued: float
    batch_size: int
    signature: tuple[Any, ...]


class DynamicBatcher:
    """Deadline-aware dynamic batching with one exclusive context per in-flight batch."""

    def __init__(self, engine: Any, options: BatchingOptions | None = None) -> None:
        self.engine = engine
        self.options = options or BatchingOptions()
        self._queue: deque[_Request] = deque()
        self._condition = asyncio.Condition()
        self._contexts: asyncio.Queue[Any] = asyncio.Queue()
        for _ in range(self.options.max_inflight_batches):
            self._contexts.put_nowait(engine.create_context())
        self._scheduler: asyncio.Task[None] | None = None
        self._active = 0
        self._draining = False
        self._closed = False

    async def start(self) -> None:
        if self._scheduler is None:
            self._scheduler = asyncio.create_task(self._schedule(), name="cutriton-batcher")

    async def infer(
        self, inputs: Mapping[str, Any], *, deadline: float | None = None
    ) -> dict[str, Any]:
        if self._closed or self._draining:
            raise ServiceOverloadedError("inference service is draining")
        if not inputs:
            raise ValueError("inputs cannot be empty")
        await self.start()
        loop = asyncio.get_running_loop()
        request = _Request(
            inputs,
            loop.create_future(),
            deadline,
            time.monotonic(),
            _batch_size(inputs),
            _signature(inputs),
        )
        if request.batch_size > self.options.max_batch_size:
            raise ServiceOverloadedError(
                f"request batch {request.batch_size} exceeds "
                f"max_batch_size={self.options.max_batch_size}"
            )
        async with self._condition:
            if len(self._queue) >= self.options.max_queue_size:
                raise ServiceOverloadedError("dynamic batching queue is full")
            self._queue.append(request)
            self._condition.notify()
        try:
            if deadline is None:
                return await request.future
            timeout = max(0.0, deadline - time.monotonic())
            return await asyncio.wait_for(request.future, timeout)
        except TimeoutError as error:
            request.future.cancel()
            raise TimeoutError("inference deadline exceeded") from error

    async def drain(self) -> None:
        self._draining = True
        async with self._condition:
            while self._queue or self._active:
                await self._condition.wait()

    async def close(self) -> None:
        if self._closed:
            return
        self._draining = True
        await self.drain()
        self._closed = True
        async with self._condition:
            self._condition.notify_all()
        if self._scheduler is not None:
            await self._scheduler
        while not self._contexts.empty():
            context = self._contexts.get_nowait()
            context.close()

    async def _schedule(self) -> None:
        running: set[asyncio.Task[None]] = set()
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._queue or self._closed)
                if self._closed and not self._queue:
                    break
                self._discard_expired()
                if not self._queue:
                    self._condition.notify_all()
                    continue
                first = self._queue[0]
                wait_seconds = self.options.max_queue_delay_us / 1_000_000
                remaining = first.enqueued + wait_seconds - time.monotonic()
                if remaining > 0 and not self._preferred_ready(first):
                    try:
                        await asyncio.wait_for(self._condition.wait(), remaining)
                        continue
                    except TimeoutError:
                        pass
                batch = self._take_compatible(first)
                self._active += 1
                self._condition.notify_all()
            context = await self._contexts.get()
            task = asyncio.create_task(self._execute(context, batch))
            running.add(task)
            task.add_done_callback(running.discard)
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    def _discard_expired(self) -> None:
        now = time.monotonic()
        retained: deque[_Request] = deque()
        for request in self._queue:
            if request.future.cancelled():
                continue
            if request.deadline is not None and request.deadline <= now:
                request.future.set_exception(TimeoutError("inference deadline exceeded in queue"))
            else:
                retained.append(request)
        self._queue = retained

    def _preferred_ready(self, first: _Request) -> bool:
        total = sum(
            request.batch_size
            for request in self._queue
            if request.signature == first.signature and not request.future.cancelled()
        )
        return any(total >= preferred for preferred in self.options.preferred_batch_sizes)

    def _take_compatible(self, first: _Request) -> list[_Request]:
        selected: list[_Request] = []
        retained: deque[_Request] = deque()
        total = 0
        for request in self._queue:
            if (
                request.signature == first.signature
                and not request.future.cancelled()
                and total + request.batch_size <= self.options.max_batch_size
            ):
                selected.append(request)
                total += request.batch_size
            else:
                retained.append(request)
        self._queue = retained
        return selected

    async def _execute(self, context: Any, requests: list[_Request]) -> None:
        try:
            torch = __import__("torch")
            stats = getattr(self.engine, "stats", None)
            if stats is not None:
                stats.add("dynamic_batches")
                stats.add("dynamic_batch_items", sum(item.batch_size for item in requests))
                stats.add(
                    "queue_seconds",
                    sum(time.monotonic() - item.enqueued for item in requests),
                )
            names = tuple(requests[0].inputs)
            merged = {
                name: torch.cat([request.inputs[name] for request in requests], dim=0)
                for name in names
            }
            outputs = await asyncio.to_thread(context.run, merged)
            offsets = [request.batch_size for request in requests]
            split_outputs = {
                name: torch.split(tensor, offsets, dim=0) for name, tensor in outputs.items()
            }
            for index, request in enumerate(requests):
                if not request.future.done():
                    request.future.set_result(
                        {name: tensors[index] for name, tensors in split_outputs.items()}
                    )
        except Exception as error:
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(error)
        finally:
            self._contexts.put_nowait(context)
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()


def _batch_size(inputs: Mapping[str, Any]) -> int:
    sizes = {int(tensor.shape[0]) for tensor in inputs.values() if tensor.ndim > 0}
    if len(sizes) != 1:
        raise ValueError("all inputs must have one identical batch dimension")
    size = sizes.pop()
    if size <= 0:
        raise ValueError("batch size must be positive")
    return size


def _signature(inputs: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (name, str(tensor.dtype), tuple(tensor.shape[1:]), str(tensor.device))
        for name, tensor in sorted(inputs.items())
    )
