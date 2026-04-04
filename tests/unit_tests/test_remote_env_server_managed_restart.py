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

import numpy as np
import torch

from rlinf.envs.remote.proto import robot_env_pb2
from rlinf.envs.remote.remote_env import RemoteEnv


def test_remote_env_step_does_not_issue_implicit_reset() -> None:
    env = object.__new__(RemoteEnv)
    env.num_envs = 1
    env._action_dim = 2

    reset_calls: list[bool] = []

    def _fake_chunk_step(_chunk_actions):
        return (
            [{"states": "terminal"}],
            torch.tensor([[0.0]], dtype=torch.float32),
            torch.tensor([[False]], dtype=torch.bool),
            torch.tensor([[True]], dtype=torch.bool),
            [{"episode": {}}],
        )

    env.chunk_step = _fake_chunk_step
    env.reset = lambda *args, **kwargs: reset_calls.append(True)  # type: ignore[assignment]

    obs, reward, terminated, truncated, infos = env.step(
        np.array([0.1, -0.2], dtype=np.float32),
        auto_reset=True,
    )

    assert reset_calls == []
    assert obs == {"states": "terminal"}
    assert reward.tolist() == [0.0]
    assert terminated.tolist() == [False]
    assert truncated.tolist() == [True]
    assert infos == {"episode": {}}


def test_remote_env_chunk_step_collapses_done_flags_to_chunk_tail() -> None:
    env = object.__new__(RemoteEnv)
    env._timeout = 3.0
    env._task_description = "fold the towel"
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env._num_steps = 0
    env.num_envs = 1
    env.last_obs = None
    env._record_metrics = lambda reward, terminations, intervene, infos: infos

    class _Stub:
        def ChunkStep(self, request, timeout):
            assert timeout == 6.0
            assert request.chunk_size == 2
            return robot_env_pb2.ChunkStepResponse(
                step_results=[
                    robot_env_pb2.StepResult(
                        observation=robot_env_pb2.Observation(),
                        reward=0.0,
                        terminated=True,
                        truncated=False,
                    ),
                    robot_env_pb2.StepResult(
                        observation=robot_env_pb2.Observation(
                            states=np.zeros((1, 4), dtype=np.float32).tobytes(),
                            state_shape=[1, 4],
                            main_image=np.zeros((2, 2, 3), dtype=np.uint8).tobytes(),
                            img_height=2,
                            img_width=2,
                            is_compressed=False,
                            task_description="fold the towel",
                        ),
                        reward=0.0,
                        terminated=False,
                        truncated=False,
                    ),
                ]
            )

    env._stub = _Stub()

    obs_list, _, chunk_terminations, chunk_truncations, _ = env.chunk_step(
        np.zeros((1, 2, 4), dtype=np.float32)
    )

    assert len(obs_list) == 1
    assert chunk_terminations.tolist() == [[False, True]]
    assert chunk_truncations.tolist() == [[False, False]]
    assert env.last_obs["task_descriptions"] == ["fold the towel"]
