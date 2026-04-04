# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    import ray  # noqa: F401
except Exception:
    import sys
    import types

    ray_stub = types.ModuleType("ray")
    ray_stub.get = lambda ref: ref
    sys.modules.setdefault("ray", ray_stub)

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.workers.env.vlm_planner_client import (
    VLMPlannerClient,
    _PendingSubtask,
    _PendingTopReward,
)


class _PlannerStub:
    def __init__(self, subtask_result="grasp the corner", score_result=1.25):
        self.calls = []
        self.subtask_result = subtask_result
        self.scores = [score_result]
        self.get_next_subtask = SimpleNamespace(remote=self._get_next_subtask)
        self.compute_top_reward = SimpleNamespace(remote=self._compute_top_reward)

    def _get_next_subtask(self, images, main_task):
        self.calls.append(("subtask", images, main_task))
        return self.subtask_result

    def _compute_top_reward(self, frames, instruction):
        self.calls.append(("top_reward", list(frames), instruction))
        return self.scores.pop(0)


class _SequentialPlannerStub:
    def __init__(self, subtask_results):
        self.calls = []
        self.subtask_results = list(subtask_results)
        self.get_next_subtask = SimpleNamespace(remote=self._get_next_subtask)

    def _get_next_subtask(self, images, main_task):
        self.calls.append(("subtask", images, main_task))
        return self.subtask_results.pop(0)


class _AwaitableRef:
    def __init__(self, value):
        self.value = value
        self.awaited = False

    def __await__(self):
        async def _resolve():
            self.awaited = True
            await asyncio.sleep(0)
            return self.value

        return _resolve().__await__()


def _make_env(task_description="fold the towel"):
    inner_env = SimpleNamespace(task_description=task_description)
    return SimpleNamespace(
        unwrapped=inner_env,
        last_obs={"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
    )


def _make_client() -> VLMPlannerClient:
    client = VLMPlannerClient(
        slot_count=1, worker_timer=lambda *_a, **_k: nullcontext()
    )
    client._log_info = lambda *_args, **_kwargs: None
    return client


@pytest.fixture(autouse=True)
def _mock_ray_get(monkeypatch):
    monkeypatch.setattr(ray, "get", lambda ref: ref)


def test_initial_subtask_planning_updates_env_and_observations():
    client = _make_client()
    client._subtask_interval = 10
    client._initial_task_descriptions = ["fold the towel"]
    client._vlm_planner = _PlannerStub()

    env = _make_env("fold the towel")
    env_output = EnvOutput(
        obs={"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
        final_obs={"states": torch.zeros((1, 4))},
    )

    updated = client.maybe_plan_initial_subtask(0, env_output, [env])

    assert env.unwrapped.task_description == "grasp the corner"
    assert updated.obs["task_descriptions"] == ["grasp the corner"]
    assert updated.final_obs["task_descriptions"] == ["grasp the corner"]
    assert env.last_obs["task_descriptions"] == ["grasp the corner"]


def test_adaptive_subtask_update_queues_and_resolves_sync():
    client = _make_client()
    client._subtask_interval = 10
    client._subtask_adaptive = True
    client._subtask_min_interval = 2
    client._subtask_plateau_window = 3
    client._subtask_plateau_threshold = 0.01
    client._top_reward_enabled = True
    client._top_reward_has_prev_score = True
    client._prev_top_score = -1.0
    client._recent_top_deltas = deque([0.0, 0.005, -0.003], maxlen=3)
    client._steps_since_subtask_update = 1
    client._initial_task_descriptions = ["fold the towel"]
    client._vlm_planner = _PlannerStub()

    env = _make_env("fold the towel")

    client.maybe_update_subtask(0, [env])
    assert 0 in client._pending_subtasks

    client.resolve_pending_sync(0, [env])

    assert env.unwrapped.task_description == "grasp the corner"
    assert client._steps_since_subtask_update == 0
    assert [call[0] for call in client._vlm_planner.calls] == [
        "subtask",
        "top_reward",
    ]


def test_out_of_order_slot_resolution_keeps_both_subtask_updates():
    client = VLMPlannerClient(
        slot_count=2, worker_timer=lambda *_a, **_k: nullcontext()
    )
    client._log_info = lambda *_args, **_kwargs: None
    client._subtask_interval = 1
    client._subtask_adaptive = False
    client._initial_task_descriptions = ["fold the towel", "stack the blocks"]
    client._vlm_planner = _SequentialPlannerStub(
        ["grasp the left corner", "align the top block"]
    )

    envs = [_make_env("fold the towel"), _make_env("stack the blocks")]

    client.maybe_update_subtask(0, envs)
    client.maybe_update_subtask(1, envs)

    client.resolve_pending_sync(1, envs)
    client.resolve_pending_sync(0, envs)

    assert envs[0].unwrapped.task_description == "grasp the left corner"
    assert envs[1].unwrapped.task_description == "align the top block"
    assert client._last_applied_subtask_request_id == [1, 1]


def test_subtask_switch_seeds_top_reward_baseline_for_next_chunk():
    client = _make_client()
    client._top_reward_enabled = True
    client._top_reward_max_frames = 16
    client._initial_task_descriptions = ["fold the towel"]
    client._vlm_planner = _PlannerStub(score_result=1.0)
    client._vlm_planner.scores = [1.0, 1.4]

    env = _make_env("fold the towel")

    assert client.apply_subtask_update(0, "grasp the corner", [env]) is True
    assert client._seed_top_reward_baseline_sync(0, [env]) is True
    assert client._prev_top_score == pytest.approx(1.0)
    assert client._top_reward_has_prev_score is True

    env_output = EnvOutput(
        obs={"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
        rewards=torch.zeros((1, 1), dtype=torch.float32),
        dones=torch.zeros((1, 1), dtype=torch.bool),
    )
    client.submit_top_reward(env_output, 0, [env])
    client.resolve_pending_sync(0, [env])

    assert env_output.rewards[0, -1].item() == pytest.approx(0.4)
    assert list(client._recent_top_deltas) == [pytest.approx(0.4)]


def test_submit_top_reward_is_nonblocking_until_sync_resolve():
    client = _make_client()
    client._top_reward_enabled = True
    client._top_reward_max_frames = 16
    client._initial_task_descriptions = ["fold the towel"]
    client._vlm_planner = _PlannerStub(score_result=1.25)

    env_output = EnvOutput(
        obs={"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
        rewards=torch.zeros((1, 2), dtype=torch.float32),
        dones=torch.zeros((1, 2), dtype=torch.bool),
    )

    client.submit_top_reward(env_output, 0, [_make_env()])

    assert 0 in client._pending_top_rewards
    assert env_output.rewards[0, -1].item() == pytest.approx(0.0)

    client.resolve_pending_sync(0, [_make_env()])

    assert env_output.rewards[0, -1].item() == pytest.approx(0.0)
    assert client._prev_top_score == pytest.approx(1.25)
    assert client._top_reward_has_prev_score is True


def test_async_resolution_awaits_refs_and_updates_state():
    client = _make_client()
    client._top_reward_enabled = True
    client._top_reward_has_prev_score = True
    client._prev_top_score = 1.5
    client._recent_top_deltas = deque(maxlen=3)
    client._initial_task_descriptions = ["fold the towel"]

    env = _make_env("fold the towel")
    env_output = EnvOutput(
        obs={"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
        rewards=torch.zeros((1, 2), dtype=torch.float32),
        dones=torch.zeros((1, 2), dtype=torch.bool),
    )
    score_ref = _AwaitableRef(2.0)
    subtask_ref = _AwaitableRef("grasp the corner")
    client._pending_top_rewards[0] = _PendingTopReward(
        env_output=env_output,
        score_ref=score_ref,
        done_on_step=False,
    )
    client._pending_subtasks[0] = _PendingSubtask(ref=subtask_ref, request_id=1)

    asyncio.run(client.resolve_pending_async(0, [env]))

    assert score_ref.awaited is True
    assert subtask_ref.awaited is True
    assert env_output.rewards[0, -1].item() == pytest.approx(0.5)
    assert env.unwrapped.task_description == "grasp the corner"
