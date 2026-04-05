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

from rlinf.data.embodied_io_struct import (
    EmbodiedRolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)


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


def test_convert_trajectories_to_batch_ignores_leading_empty_trajectory() -> None:
    empty_trajectory = Trajectory(rollout_horizon_steps=8)
    populated_trajectory = Trajectory(
        rollout_horizon_steps=8,
        prev_logprobs=torch.ones((1, 1, 1), dtype=torch.float32),
        rewards=torch.ones((1, 1, 1), dtype=torch.float32),
        dones=torch.zeros((2, 1, 1), dtype=torch.bool),
        forward_inputs={"action": torch.ones((1, 1, 1), dtype=torch.float32)},
        curr_obs={"states": torch.zeros((1, 1, 4), dtype=torch.float32)},
        next_obs={"states": torch.ones((1, 1, 4), dtype=torch.float32)},
    )

    batch = convert_trajectories_to_batch([empty_trajectory, populated_trajectory])

    assert batch["prev_logprobs"].shape == (1, 1, 1)
    assert batch["rewards"].shape == (1, 1, 1)
    assert batch["dones"].shape == (2, 1, 1)
    assert batch["forward_inputs"]["action"].shape == (1, 1, 1)
    assert batch["curr_obs"]["states"].shape == (1, 1, 4)
    assert batch["next_obs"]["states"].shape == (1, 1, 4)


def test_convert_trajectories_to_batch_handles_all_empty_trajectories() -> None:
    batch = convert_trajectories_to_batch(
        [Trajectory(rollout_horizon_steps=8), Trajectory(rollout_horizon_steps=8)]
    )

    assert batch == {}
