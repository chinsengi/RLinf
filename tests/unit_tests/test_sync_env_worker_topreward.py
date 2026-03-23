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

"""Unit tests for sync EnvWorker TOPReward and subtask planning changes.

Covers:
- TOPReward reset does not raise on subtask update (Plan step 1)
- Main task is passed to get_next_subtask (Plan step 2)
- Empty task_description rejected when subtask_interval > 0 (Plan step 2)
- TOPReward instruction selection logic (Plan step 4)
"""

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import ray  # noqa: F401
except Exception:
    import sys
    import types

    ray_stub = types.ModuleType("ray")
    ray_actor_stub = types.ModuleType("ray.actor")
    ray_stub.get = lambda ref: ref
    ray_stub.actor = ray_actor_stub
    ray_actor_stub.ActorHandle = object
    sys.modules.setdefault("ray", ray_stub)
    sys.modules.setdefault("ray.actor", ray_actor_stub)

from rlinf.workers.env.env_worker import EnvWorker

# ---------------------------------------------------------------------------
# Plan step 1: TOPReward reset on subtask update should not raise
# ---------------------------------------------------------------------------


def test_apply_subtask_update_does_not_raise_with_top_reward():
    """_apply_subtask_update calls _reset_top_reward_state() without args."""
    inner_env = SimpleNamespace(task_description="fold the towel")
    worker = EnvWorker.__new__(EnvWorker)
    worker.env_list = [SimpleNamespace(unwrapped=inner_env)]
    worker._top_reward_enabled = True
    worker._top_reward_instruction_source = "current_task"
    worker._episode_frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    worker._prev_top_score = 1.5
    worker.log_info = lambda *_args, **_kwargs: None

    # Should not raise TypeError (the old bug passed stage_id to a no-arg method).
    result = worker._apply_subtask_update(0, "grasp the left corner")
    assert result is True
    assert inner_env.task_description == "grasp the left corner"
    # Reward state should be reset.
    assert worker._episode_frames == []
    assert worker._prev_top_score == 0.0


def test_apply_subtask_update_no_reset_for_initial_task():
    """With initial_task source, subtask update does NOT reset reward state."""
    inner_env = SimpleNamespace(task_description="fold the towel")
    worker = EnvWorker.__new__(EnvWorker)
    worker.env_list = [SimpleNamespace(unwrapped=inner_env)]
    worker._top_reward_enabled = True
    worker._top_reward_instruction_source = "initial_task"
    worker._episode_frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    worker._prev_top_score = 1.5
    worker.log_info = lambda *_args, **_kwargs: None

    result = worker._apply_subtask_update(0, "grasp the left corner")
    assert result is True
    # Reward state should NOT be reset.
    assert len(worker._episode_frames) == 1
    assert worker._prev_top_score == 1.5


# ---------------------------------------------------------------------------
# Plan step 2: Main task required for subtask planning
# ---------------------------------------------------------------------------


def test_get_next_subtask_requires_main_task():
    """VLMPlannerWorker.get_next_subtask raises ValueError on empty main_task."""
    from rlinf.workers.vlm_planner.vlm_planner_worker import VLMPlannerWorker

    planner = VLMPlannerWorker.__new__(VLMPlannerWorker)
    with pytest.raises(ValueError, match="non-empty main_task"):
        planner.get_next_subtask(images=[], main_task="")


def test_get_next_subtask_prompt_includes_main_task():
    """The subtask prompt includes the episode goal."""
    from rlinf.workers.vlm_planner.vlm_planner_worker import VLMPlannerWorker

    planner = VLMPlannerWorker.__new__(VLMPlannerWorker)
    # Mock _generate to capture the prompt.
    captured_messages = []

    def fake_generate(messages, max_new_tokens):
        captured_messages.append(messages)
        return "pick up the corner"

    planner._generate = fake_generate
    planner._max_new_tokens_subtask = 64

    # Provide a logger stub.
    planner._logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )

    result = planner.get_next_subtask(images=[], main_task="fold the towel")
    assert result == "pick up the corner"
    assert len(captured_messages) == 1
    # The user text should contain the main task.
    user_content = captured_messages[0][1]["content"]
    # user_content is a list of dicts; the last one is text.
    text_item = user_content[-1]["text"]
    assert "fold the towel" in text_item


# ---------------------------------------------------------------------------
# Plan step 4: TOPReward instruction source selection
# ---------------------------------------------------------------------------


def test_instruction_source_initial_task():
    worker = EnvWorker.__new__(EnvWorker)
    worker._top_reward_instruction_source = "initial_task"
    worker._initial_task_descriptions = ["fold the towel"]
    worker.env_list = [
        SimpleNamespace(unwrapped=SimpleNamespace(task_description="grasp the corner"))
    ]
    assert worker._get_top_reward_instruction(0) == "fold the towel"


def test_instruction_source_current_task():
    worker = EnvWorker.__new__(EnvWorker)
    worker._top_reward_instruction_source = "current_task"
    worker._initial_task_descriptions = ["fold the towel"]
    worker.env_list = [
        SimpleNamespace(unwrapped=SimpleNamespace(task_description="grasp the corner"))
    ]
    assert worker._get_top_reward_instruction(0) == "grasp the corner"
