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
- Adaptive subtask triggering (plateau, score threshold, cooldown, fallback)
"""

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
    ray_actor_stub = types.ModuleType("ray.actor")
    ray_stub.get = lambda ref: ref
    ray_stub.actor = ray_actor_stub
    ray_actor_stub.ActorHandle = object
    sys.modules.setdefault("ray", ray_stub)
    sys.modules.setdefault("ray.actor", ray_actor_stub)

from rlinf.workers.env.env_worker import EnvWorker, _apply_action_chunk_smoothing


class _CfgNode(dict):
    """Minimal config node supporting both attr access and `.get()`."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _cfg_node(data):
    if isinstance(data, dict):
        return _CfgNode({key: _cfg_node(value) for key, value in data.items()})
    return data


class _PlannerStub:
    def __init__(self, result="pick up the corner"):
        self.calls = []
        self.result = result
        self.get_next_subtask = SimpleNamespace(remote=self._get_next_subtask)

    def _get_next_subtask(self, images, main_task):
        self.calls.append((images, main_task))
        return self.result


class _TopRewardPlannerStub:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []
        self.compute_top_reward = SimpleNamespace(remote=self._compute_top_reward)

    def _compute_top_reward(self, frames, instruction):
        self.calls.append((list(frames), instruction))
        return self.scores.pop(0)


def _make_adaptive_worker(
    *,
    subtask_adaptive=True,
    subtask_interval=10,
    subtask_min_interval=2,
    plateau_window=3,
    plateau_threshold=0.01,
    score_threshold=-0.5,
    prev_top_score=-2.0,
    recent_deltas=None,
    steps_since_update=0,
):
    inner_env = SimpleNamespace(task_description="fold the towel")
    env = SimpleNamespace(
        unwrapped=inner_env,
        last_obs={"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
    )
    worker = EnvWorker.__new__(EnvWorker)
    worker.env_list = [env]
    worker._initial_task_descriptions = ["fold the towel"]
    worker._subtask_interval = subtask_interval
    worker._subtask_adaptive = subtask_adaptive
    worker._subtask_min_interval = subtask_min_interval
    worker._subtask_plateau_window = plateau_window
    worker._subtask_plateau_threshold = plateau_threshold
    worker._subtask_score_threshold = score_threshold
    worker._top_reward_enabled = True
    worker._prev_top_score = prev_top_score
    worker._top_reward_has_prev_score = True
    worker._steps_since_subtask_update = steps_since_update
    worker._recent_top_deltas = deque(
        recent_deltas or [],
        maxlen=plateau_window,
    )
    worker._episode_frames = []
    worker._vlm_planner = _PlannerStub()
    worker.log_info = lambda *_args, **_kwargs: None
    return worker


def _make_top_reward_worker(scores, *, subtask_interval=0):
    worker = EnvWorker.__new__(EnvWorker)
    worker.env_list = [
        SimpleNamespace(unwrapped=SimpleNamespace(task_description="fold the towel"))
    ]
    worker._initial_task_descriptions = ["fold the towel"]
    worker._top_reward_enabled = True
    worker._subtask_interval = subtask_interval
    worker._top_reward_max_frames = 16
    worker._episode_frames = []
    worker._prev_top_score = 0.0
    worker._top_reward_has_prev_score = False
    worker._recent_top_deltas = deque(maxlen=3)
    worker._vlm_planner = _TopRewardPlannerStub(scores)
    worker.worker_timer = lambda *_args, **_kwargs: nullcontext()
    worker.log_info = lambda *_args, **_kwargs: None
    return worker


# ---------------------------------------------------------------------------
# Plan step 1: TOPReward reset on subtask update should not raise
# ---------------------------------------------------------------------------


def test_apply_subtask_update_does_not_raise_with_top_reward():
    """_apply_subtask_update calls _reset_top_reward_state() without args."""
    inner_env = SimpleNamespace(task_description="fold the towel")
    worker = EnvWorker.__new__(EnvWorker)
    worker.env_list = [SimpleNamespace(unwrapped=inner_env)]
    worker._top_reward_enabled = True
    worker._episode_frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    worker._prev_top_score = 1.5
    worker._recent_top_deltas = deque([0.1, 0.0], maxlen=3)
    worker.log_info = lambda *_args, **_kwargs: None

    # Should not raise TypeError (the old bug passed slot_id to a no-arg method).
    result = worker._apply_subtask_update(0, "grasp the left corner")
    assert result is True
    assert inner_env.task_description == "grasp the left corner"
    # Reward state should be reset.
    assert worker._episode_frames == []
    assert worker._prev_top_score == 0.0
    assert worker._top_reward_has_prev_score is False
    assert list(worker._recent_top_deltas) == []


def test_apply_subtask_update_resets_when_subtask_changes():
    """Subtask updates reset reward state when TOPReward is enabled."""
    inner_env = SimpleNamespace(task_description="fold the towel")
    worker = EnvWorker.__new__(EnvWorker)
    worker.env_list = [SimpleNamespace(unwrapped=inner_env)]
    worker._top_reward_enabled = True
    worker._episode_frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    worker._prev_top_score = 1.5
    worker._top_reward_has_prev_score = True
    worker._recent_top_deltas = deque([0.1, 0.0], maxlen=3)
    worker.log_info = lambda *_args, **_kwargs: None

    result = worker._apply_subtask_update(0, "grasp the left corner")
    assert result is True
    assert inner_env.task_description == "grasp the left corner"
    # Reward state should be reset.
    assert worker._episode_frames == []
    assert worker._prev_top_score == 0.0
    assert worker._top_reward_has_prev_score is False
    assert list(worker._recent_top_deltas) == []


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
# Plan step 4: TOPReward instruction selection by subtask_interval
# ---------------------------------------------------------------------------


def test_top_reward_uses_initial_task_when_subtask_planning_disabled():
    worker = EnvWorker.__new__(EnvWorker)
    worker._subtask_interval = 0
    worker._initial_task_descriptions = ["fold the towel"]
    worker.env_list = [
        SimpleNamespace(unwrapped=SimpleNamespace(task_description="grasp the corner"))
    ]
    assert worker._get_top_reward_instruction(0) == "fold the towel"


def test_top_reward_uses_current_task_when_subtask_planning_enabled():
    worker = EnvWorker.__new__(EnvWorker)
    worker._subtask_interval = 10
    worker._initial_task_descriptions = ["fold the towel"]
    worker.env_list = [
        SimpleNamespace(unwrapped=SimpleNamespace(task_description="grasp the corner"))
    ]
    assert worker._get_top_reward_instruction(0) == "grasp the corner"


def test_compute_top_reward_seeds_episode_with_zero_delta(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", lambda ref: ref)
    worker = _make_top_reward_worker([1.25, 1.75])
    first_env_output = SimpleNamespace(
        obs={"main_images": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)},
        rewards=torch.zeros((1, 1), dtype=torch.float32),
    )
    second_env_output = SimpleNamespace(
        obs={"main_images": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)},
        rewards=torch.zeros((1, 1), dtype=torch.float32),
    )

    first_output = worker._compute_top_reward(first_env_output, 0)
    second_output = worker._compute_top_reward(second_env_output, 0)

    assert first_output.rewards[:, -1].item() == pytest.approx(0.0)
    assert second_output.rewards[:, -1].item() == pytest.approx(0.5)
    assert worker._prev_top_score == pytest.approx(1.75)
    assert worker._top_reward_has_prev_score is True
    assert list(worker._recent_top_deltas) == [0.0, 0.5]


def test_compute_top_reward_seeds_zero_delta_after_subtask_reset(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", lambda ref: ref)
    worker = _make_top_reward_worker([2.0], subtask_interval=10)
    worker._episode_frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    worker._prev_top_score = 1.5
    worker._top_reward_has_prev_score = True
    worker._recent_top_deltas.append(0.25)

    assert worker._apply_subtask_update(0, "grasp the left corner") is True

    env_output = SimpleNamespace(
        obs={"main_images": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)},
        rewards=torch.zeros((1, 1), dtype=torch.float32),
    )
    updated_output = worker._compute_top_reward(env_output, 0)

    assert updated_output.rewards[:, -1].item() == pytest.approx(0.0)
    assert worker._prev_top_score == pytest.approx(2.0)
    assert worker._top_reward_has_prev_score is True
    assert list(worker._recent_top_deltas) == [0.0]


def test_bootstrap_step_reuses_last_obs_when_rollout_reset_disabled():
    worker = EnvWorker.__new__(EnvWorker)
    worker.cfg = _cfg_node(
        {
            "env": {
                "train": {
                    "auto_reset": False,
                    "reset_on_rollout_epoch": False,
                }
            },
            "actor": {"model": {"num_action_chunks": 1}},
        }
    )
    worker._top_reward_enabled = True
    worker._prev_top_score = 1.25
    worker._top_reward_has_prev_score = True
    worker._recent_top_deltas = deque([0.5], maxlen=3)
    worker.slot_count = 1
    worker.train_num_envs_per_stage = 1
    worker.last_obs_list = [{"states": "previous"}]
    worker.last_intervened_info_list = [(torch.tensor([1]), torch.tensor([0]))]
    worker._steps_since_subtask_update = 7

    reset_calls = []

    class _Env:
        is_start = False

        def reset(self):
            reset_calls.append(True)
            return {"states": "reset"}, {}

    worker.env_list = [_Env()]

    env_outputs = worker.bootstrap_step()

    assert reset_calls == []
    assert env_outputs[0].obs == {"states": "previous"}
    assert torch.equal(env_outputs[0].intervene_actions, torch.tensor([1]))
    assert torch.equal(env_outputs[0].intervene_flags, torch.tensor([0]))
    assert worker._prev_top_score == pytest.approx(1.25)
    assert worker._top_reward_has_prev_score is True
    assert list(worker._recent_top_deltas) == [0.5]
    assert worker._steps_since_subtask_update == 7


def test_bootstrap_step_resets_on_first_epoch_even_when_rollout_reset_disabled():
    worker = EnvWorker.__new__(EnvWorker)
    worker.cfg = _cfg_node(
        {
            "env": {
                "train": {
                    "auto_reset": False,
                    "reset_on_rollout_epoch": False,
                }
            },
            "actor": {"model": {"num_action_chunks": 1}},
        }
    )
    worker._top_reward_enabled = True
    worker._prev_top_score = 1.25
    worker._top_reward_has_prev_score = True
    worker._recent_top_deltas = deque([0.5], maxlen=3)
    worker._episode_frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    worker.slot_count = 1
    worker.train_num_envs_per_stage = 1
    worker.last_obs_list = []
    worker.last_intervened_info_list = []
    worker._steps_since_subtask_update = 4

    reset_calls = []

    class _Env:
        is_start = False

        def reset(self):
            reset_calls.append(True)
            return {"states": "reset"}, {}

    worker.env_list = [_Env()]

    env_outputs = worker.bootstrap_step()

    assert reset_calls == [True]
    assert env_outputs[0].obs == {"states": "reset"}
    assert worker._prev_top_score == 0.0
    assert worker._top_reward_has_prev_score is False
    assert worker._episode_frames == []
    assert list(worker._recent_top_deltas) == []
    assert worker._steps_since_subtask_update == 0


def test_env_interact_step_resets_subtask_counter_on_episode_done(monkeypatch):
    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.prepare_actions",
        lambda **kwargs: kwargs["raw_chunk_actions"],
    )

    worker = EnvWorker.__new__(EnvWorker)
    worker.cfg = _cfg_node(
        {
            "env": {
                "train": {
                    "env_type": "yam",
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "wm_env_type": None,
                }
            },
            "actor": {
                "model": {
                    "model_type": "openpi",
                    "num_action_chunks": 1,
                    "action_dim": 1,
                }
            },
        }
    )
    worker.worker_timer = lambda *_args, **_kwargs: nullcontext()
    worker._top_reward_enabled = False
    worker._vlm_planner = None
    worker._steps_since_subtask_update = 5
    worker.env_list = [
        SimpleNamespace(
            chunk_step=lambda _chunk_actions: (
                [{"states": "next"}],
                torch.zeros((1, 1), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.bool),
                torch.ones((1, 1), dtype=torch.bool),
                [{}],
            )
        )
    ]

    env_output, env_info = worker.env_interact_step(torch.zeros((1, 1)), 0)

    assert env_output.dones[:, -1].item() is True
    assert env_info == {}
    assert worker._steps_since_subtask_update == 0


def test_apply_action_chunk_smoothing_ema_numpy():
    chunk_actions = np.array([[[0.0], [10.0], [10.0]]], dtype=np.float32)

    smoothed = _apply_action_chunk_smoothing(
        chunk_actions,
        {"enabled": True, "method": "ema", "alpha": 0.5},
    )

    assert smoothed.shape == chunk_actions.shape
    assert smoothed[0, 0, 0] == 0.0
    assert smoothed[0, 1, 0] == 5.0
    assert smoothed[0, 2, 0] == 7.5


def test_env_interact_step_applies_action_chunk_smoothing(monkeypatch):
    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.prepare_actions",
        lambda **kwargs: kwargs["raw_chunk_actions"].numpy(),
    )

    captured = {}

    def _chunk_step(chunk_actions):
        captured["chunk_actions"] = chunk_actions
        return (
            [{"states": "next"}],
            torch.zeros((1, 3), dtype=torch.float32),
            torch.zeros((1, 3), dtype=torch.bool),
            torch.zeros((1, 3), dtype=torch.bool),
            [{}],
        )

    worker = EnvWorker.__new__(EnvWorker)
    worker.cfg = _cfg_node(
        {
            "env": {
                "train": {
                    "env_type": "yam",
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "wm_env_type": None,
                    "action_chunk_smoothing": {
                        "enabled": True,
                        "method": "ema",
                        "alpha": 0.5,
                    },
                }
            },
            "actor": {
                "model": {
                    "model_type": "openpi",
                    "num_action_chunks": 3,
                    "action_dim": 1,
                }
            },
        }
    )
    worker.worker_timer = lambda *_args, **_kwargs: nullcontext()
    worker._top_reward_enabled = False
    worker._vlm_planner = None
    worker._steps_since_subtask_update = 0
    worker.env_list = [SimpleNamespace(chunk_step=_chunk_step)]

    worker.env_interact_step(torch.tensor([[[0.0], [10.0], [10.0]]]), 0)

    assert captured["chunk_actions"][0, 0, 0] == 0.0
    assert captured["chunk_actions"][0, 1, 0] == 5.0
    assert captured["chunk_actions"][0, 2, 0] == 7.5


def test_maybe_update_subtask_triggers_on_plateau(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", lambda ref: ref)
    worker = _make_adaptive_worker(
        recent_deltas=[0.001, -0.002, 0.0],
        steps_since_update=1,
    )

    worker._maybe_update_subtask(0)

    assert len(worker._vlm_planner.calls) == 1
    assert worker.env_list[0].unwrapped.task_description == "pick up the corner"
    assert worker._steps_since_subtask_update == 0


def test_maybe_update_subtask_triggers_on_score_threshold(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", lambda ref: ref)
    worker = _make_adaptive_worker(
        prev_top_score=-0.1,
        recent_deltas=[0.2],
        steps_since_update=1,
    )

    worker._maybe_update_subtask(0)

    assert len(worker._vlm_planner.calls) == 1
    assert worker._steps_since_subtask_update == 0


def test_maybe_update_subtask_respects_adaptive_cooldown(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", lambda ref: ref)
    worker = _make_adaptive_worker(
        prev_top_score=-0.1,
        recent_deltas=[0.001, 0.0, -0.001],
        steps_since_update=0,
    )

    worker._maybe_update_subtask(0)

    assert len(worker._vlm_planner.calls) == 0
    assert worker._steps_since_subtask_update == 1

    worker._maybe_update_subtask(0)

    assert len(worker._vlm_planner.calls) == 1
    assert worker._steps_since_subtask_update == 0


def test_maybe_update_subtask_falls_back_to_max_interval(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", lambda ref: ref)
    worker = _make_adaptive_worker(
        subtask_interval=3,
        subtask_min_interval=2,
        prev_top_score=-2.0,
        recent_deltas=[0.1, 0.2],
        steps_since_update=2,
    )

    worker._maybe_update_subtask(0)

    assert len(worker._vlm_planner.calls) == 1
    assert worker._steps_since_subtask_update == 0


def test_maybe_update_subtask_fixed_mode_preserves_old_interval_behavior(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", lambda ref: ref)
    worker = _make_adaptive_worker(
        subtask_adaptive=False,
        subtask_interval=1,
        subtask_min_interval=99,
        prev_top_score=-0.1,
        recent_deltas=[0.001, 0.0, -0.001],
        steps_since_update=0,
    )

    worker._maybe_update_subtask(0)

    assert len(worker._vlm_planner.calls) == 1
    assert worker._steps_since_subtask_update == 0
