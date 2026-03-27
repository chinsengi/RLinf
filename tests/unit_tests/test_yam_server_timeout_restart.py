# Copyright 2026 Shirui Chen
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

"""Regression tests for YAM timeout/restart handling."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from rlinf.envs.remote.robot_server import RobotEnvServicer
from rlinf.envs.yam.yam_env import YAMEnv


class _FakeRobotEnvForTimeout:
    def __init__(self):
        self.step_calls = 0
        self.get_obs_calls = 0

    def step(self, action_dict):
        self.step_calls += 1
        return {"source": "step", "action_dict": action_dict}

    def get_obs(self):
        self.get_obs_calls += 1
        return {"source": "get_obs"}


def _make_timeout_env() -> tuple[YAMEnv, _FakeRobotEnvForTimeout]:
    robot_env = _FakeRobotEnvForTimeout()
    env = object.__new__(YAMEnv)
    env._is_dummy = False
    env._robot_env = robot_env
    env._episode_start_time = time.monotonic() - 5.0
    env._episode_duration_s = 1.0
    env.num_envs = 1
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env._num_steps = 0
    env._logger = Mock()
    env._task_description = ""
    env._reset_metrics = Mock()
    env._record_metrics = lambda reward, terminations, intervene, infos: infos
    env._wrap_obs = lambda raw_obs: raw_obs
    return env, robot_env


def test_yam_step_does_not_command_robot_after_wall_clock_timeout():
    env, robot_env = _make_timeout_env()

    _, _, _, truncated, _ = env.step(
        np.zeros((1, 14), dtype=np.float32), auto_reset=False
    )

    assert truncated.tolist() == [True]
    assert robot_env.step_calls == 0
    assert robot_env.get_obs_calls == 1


def test_yam_chunk_step_stops_executing_after_first_timeout():
    env, robot_env = _make_timeout_env()
    chunk_actions = np.zeros((1, 3, 14), dtype=np.float32)

    obs_list, _, _, chunk_truncations, _ = env.chunk_step(chunk_actions)

    assert len(obs_list) == 3
    assert robot_env.step_calls == 0
    assert robot_env.get_obs_calls == 1
    assert chunk_truncations.shape == (1, 3)
    assert bool(chunk_truncations[0, -1])


class _FakeServicerEnv:
    def __init__(self):
        self._img_h = 2
        self._img_w = 2
        self._episode_cooldown_s = 60.0
        self._episode_start_time = None
        self._is_dummy = True
        self.return_home_calls = 0
        self.zero_torque_calls = 0
        self.prepare_for_reconnection_calls = 0
        self.prepare_for_next_episode_calls = 0
        self.chunk_step_calls = 0

    def _dummy_obs(self):
        return {"raw": True}

    def _wrap_obs(self, raw_obs):
        return {
            "states": np.zeros((1, 14), dtype=np.float32),
            "main_images": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "task_descriptions": [""],
        }

    def return_to_home(self):
        self.return_home_calls += 1

    def enter_zero_torque_mode(self):
        self.zero_torque_calls += 1

    def prepare_for_reconnection(self):
        self.prepare_for_reconnection_calls += 1

    def prepare_for_next_episode(self):
        self.prepare_for_next_episode_calls += 1

    def chunk_step(self, actions):
        self.chunk_step_calls += 1
        raise AssertionError("chunk_step should not run while restart is pending")


def test_close_rpc_triggers_safe_recovery_without_requesting_shutdown():
    env = _FakeServicerEnv()
    request_shutdown = Mock()
    servicer = RobotEnvServicer(env, request_shutdown=request_shutdown)

    servicer.Close(None, None)

    assert env.return_home_calls == 1
    assert env.zero_torque_calls == 1
    assert env.prepare_for_reconnection_calls == 1
    assert servicer._restart_required is True
    request_shutdown.assert_not_called()


def test_chunk_step_discards_stale_actions_while_cooldown_is_active():
    env = _FakeServicerEnv()
    servicer = RobotEnvServicer(env)
    servicer._first_chunk_approved = True
    servicer._restart_required = True
    servicer._cooldown_deadline = time.monotonic() + 30.0

    request = SimpleNamespace(
        actions=np.zeros((1, 2, 14), dtype=np.float32).tobytes(),
        num_envs=1,
        chunk_size=2,
        action_dim=14,
    )

    response = servicer.ChunkStep(request, None)

    assert env.chunk_step_calls == 0
    assert len(response.step_results) == 2
    assert response.step_results[-1].truncated is True
