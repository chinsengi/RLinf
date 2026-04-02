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

import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult


def test_append_transitions_keeps_live_obs_task_descriptions() -> None:
    rollout_result = EmbodiedRolloutResult(rollout_horizon_steps=10)
    curr_obs = {
        "states": torch.zeros((1, 14)),
        "task_descriptions": ["fold the towel"],
    }
    next_obs = {
        "states": torch.ones((1, 14)),
        "task_descriptions": ["fold the towel"],
    }

    rollout_result.append_transitions(curr_obs=curr_obs, next_obs=next_obs)

    assert curr_obs["task_descriptions"] == ["fold the towel"]
    assert next_obs["task_descriptions"] == ["fold the towel"]
    assert "task_descriptions" not in rollout_result.curr_obs[0]
    assert "task_descriptions" not in rollout_result.next_obs[0]


def test_rollout_horizon_steps_propagates_to_trajectory() -> None:
    rollout_result = EmbodiedRolloutResult(rollout_horizon_steps=12)

    trajectory = rollout_result.to_trajectory()

    assert trajectory.rollout_horizon_steps == 12
