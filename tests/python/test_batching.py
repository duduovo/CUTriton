from __future__ import annotations

import asyncio
import time

import torch
from cutriton.batching import DynamicBatcher
from cutriton.config import BatchingOptions


class _Context:
    def run(self, inputs):
        return {"output": inputs["input"] * 2}

    def close(self):
        pass


class _Engine:
    def create_context(self):
        return _Context()


def test_dynamic_batching_preserves_request_boundaries():
    async def scenario():
        batcher = DynamicBatcher(
            _Engine(),
            BatchingOptions(
                max_batch_size=8,
                preferred_batch_sizes=(4, 8),
                max_queue_delay_us=50_000,
                max_inflight_batches=1,
            ),
        )
        requests = [
            asyncio.create_task(batcher.infer({"input": torch.full((1, 3), float(index))}))
            for index in range(4)
        ]
        outputs = await asyncio.gather(*requests)
        assert [item["output"][0, 0].item() for item in outputs] == [0.0, 2.0, 4.0, 6.0]
        await batcher.close()

    asyncio.run(scenario())


def test_incompatible_non_batch_shapes_are_not_merged():
    async def scenario():
        batcher = DynamicBatcher(
            _Engine(), BatchingOptions(max_queue_delay_us=0, max_inflight_batches=1)
        )
        left, right = await asyncio.gather(
            batcher.infer({"input": torch.ones(1, 2)}),
            batcher.infer({"input": torch.ones(1, 3)}),
        )
        assert left["output"].shape == (1, 2)
        assert right["output"].shape == (1, 3)
        await batcher.close()

    asyncio.run(scenario())


def test_deadline_expires_while_waiting_for_a_batch():
    async def scenario():
        batcher = DynamicBatcher(
            _Engine(),
            BatchingOptions(max_queue_delay_us=100_000, max_inflight_batches=1),
        )
        try:
            try:
                await batcher.infer({"input": torch.ones(1, 2)}, deadline=time.monotonic() + 0.001)
                raise AssertionError("deadline should expire")
            except TimeoutError:
                pass
        finally:
            await batcher.close()

    asyncio.run(scenario())
