# Copyright 2026 Shirui Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for remote disconnect handling in AsyncEnvWorker."""

import asyncio
from dataclasses import dataclass

import pytest
import torch

from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.envs.remote.remote_env import RobotServerDisconnectedError
from rlinf.utils.comm_mapping import CommMapper
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.env.async_env_worker import AsyncEnvWorker


def test_async_env_worker_waits_for_staged_shutdown_on_remote_disconnect() -> None:
    worker = object.__new__(AsyncEnvWorker)
    log_messages: list[str] = []

    async def fake_run_interact_once(*args, **kwargs):
        del args, kwargs
        raise RobotServerDisconnectedError(
            "[RemoteEnv] Robot server disconnected during ChunkStep (gRPC UNAVAILABLE)."
        )

    worker._run_interact_once = fake_run_interact_once
    worker.log_warning = log_messages.append

    async def _exercise() -> None:
        task = asyncio.create_task(
            AsyncEnvWorker._interact(worker, None, None, None, None)
        )
        await asyncio.sleep(0.05)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_exercise())

    assert len(log_messages) == 1
    assert "Waiting for the staged entrypoint to stop training." in log_messages[0]


def test_recv_rollout_results_async_yields_control() -> None:
    worker = object.__new__(EnvWorker)
    worker._rank = 5
    worker.src_ranks = {"train": [(1, 1), (2, 2)]}

    rollout_shards = {
        CommMapper.build_channel_key(1, worker._rank, extra="train_rollout_results"): (
            RolloutResult(actions=torch.tensor([[1.0, 2.0]]))
        ),
        CommMapper.build_channel_key(2, worker._rank, extra="train_rollout_results"): (
            RolloutResult(actions=torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        ),
    }

    @dataclass
    class FakeAsyncWork:
        result: RolloutResult
        started: asyncio.Event
        finished: asyncio.Event

        async def async_wait(self) -> RolloutResult:
            self.started.set()
            await asyncio.sleep(0.02)
            self.finished.set()
            return self.result

    class FakeChannel:
        def __init__(self, items: dict[str, RolloutResult]):
            self._items = items
            self.calls: list[tuple[str, bool]] = []
            self.started = asyncio.Event()
            self.finished = asyncio.Event()

        def get(self, *, key: str, async_op: bool):
            self.calls.append((key, async_op))
            return FakeAsyncWork(
                result=self._items[key],
                started=self.started,
                finished=self.finished,
            )

    channel = FakeChannel(rollout_shards)

    async def _exercise() -> tuple[int, RolloutResult]:
        yield_count = 0

        async def ticker() -> None:
            nonlocal yield_count
            await channel.started.wait()
            while not channel.finished.is_set():
                yield_count += 1
                await asyncio.sleep(0)

        ticker_task = asyncio.create_task(ticker())
        merged = await worker.recv_rollout_results(channel, mode="train")
        await ticker_task
        return yield_count, merged

    yield_count, merged = asyncio.run(_exercise())

    assert yield_count >= 1
    assert merged.actions.shape == (3, 2)
    assert torch.equal(
        merged.actions,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert channel.calls == [
        (
            CommMapper.build_channel_key(
                1, worker._rank, extra="train_rollout_results"
            ),
            True,
        ),
        (
            CommMapper.build_channel_key(
                2, worker._rank, extra="train_rollout_results"
            ),
            True,
        ),
    ]
