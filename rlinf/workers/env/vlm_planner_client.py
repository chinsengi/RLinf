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

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import EnvOutput

if TYPE_CHECKING:
    from rlinf.workers.vlm_planner.vlm_planner_worker import VLMPlannerWorker

# Must stay in sync with
# rlinf.workers.vlm_planner.vlm_planner_worker.NEW_TASK_PREFIX.
NEW_TASK_PREFIX = "NEW_TASK:"

# Qwen VL default: matches the FPS constant in qwen_vl_utils and the
# default parameter in VLMPlannerWorker.compute_top_reward.
_TOP_REWARD_DEFAULT_FPS: float = 2.0


@dataclass
class _PendingTopReward:
    env_output: EnvOutput
    score_ref: Any
    done_on_step: bool


@dataclass
class _PendingSubtask:
    ref: Any
    request_id: int


def _noop_log(*_args: Any, **_kwargs: Any) -> None:
    return None


class VLMPlannerClient:
    """Manage VLM planner and TOPReward state for one env worker.

    This helper owns all env-side VLM coordination that used to live directly
    on :class:`EnvWorker`. It handles two features:

    1. subtask planning, where a planner proposes a new
       ``env.task_description`` for a pipeline slot
    2. TOPReward, where planner scores over recent frames are converted into a
       dense reward delta

    The client is worker-scoped. Pending planner work is keyed by pipeline
    slot, and stale-request protection is also slot-scoped, so an update from
    one slot cannot suppress a valid update from another slot if results resolve
    out of order.

    Attributes:
        _log_info: Logger callback used for planner/TOPReward status messages.
        _worker_timer: Timer factory used to measure planner-side work such as
            TOPReward scoring.
        _vlm_planner: Planner actor/handle injected after worker construction.
            Expected to expose ``get_next_subtask.remote(...)`` and
            ``compute_top_reward.remote(...)``.
        _initial_task_descriptions: Per-slot episode-level task descriptions
            from config. Subtask planning treats these as the parent tasks.
        _subtask_interval: Base chunk-step interval for requesting a subtask
            refresh. ``0`` disables subtask planning.
        _subtask_adaptive: Whether subtask requests may trigger before the fixed
            interval based on recent TOPReward dynamics.
        _subtask_min_interval: Minimum number of chunk steps before adaptive
            subtask logic is allowed to trigger.
        _subtask_plateau_window: Number of recent TOPReward deltas inspected
            when detecting a plateau.
        _subtask_plateau_threshold: Absolute-delta threshold used by plateau
            detection.
        _subtask_score_threshold: TOPReward score threshold that can force an
            early subtask refresh in adaptive mode.
        _steps_since_subtask_update: Worker-level counter of chunk steps since
            the most recent applied subtask update.
        _pending_subtasks: Mapping ``slot_id -> _PendingSubtask`` for in-flight
            subtask planner requests.
        _subtask_request_counter: Per-slot monotonically increasing request ids.
            Incremented when a new subtask request is submitted.
        _last_applied_subtask_request_id: Per-slot request ids for the latest
            applied subtask result. Used to ignore stale resolutions for that
            same slot only.
        _episode_done_waiting_for_top_reward_reset: Per-slot guard that blocks
            adaptive subtask updates after a terminal step until the pending
            TOPReward result has been resolved and episode-local state has been
            cleared.
        _dense_reward_method: Name of the dense reward method to use
            (``"top_reward"`` or ``"none"``).
        _top_reward_max_frames: Maximum number of recent frames retained for a
            TOPReward planner call.
        _episode_frames: Rolling frame buffer for the current TOPReward scoring
            segment.
        _prev_top_score: Previous TOPReward score used to convert the latest
            score into a delta reward.
        _top_reward_has_prev_score: Whether ``_prev_top_score`` is initialized
            for the current TOPReward segment.
        _recent_top_deltas: Recent dense TOPReward deltas. Adaptive subtask
            planning uses this history for plateau detection.
        _pending_top_rewards: Mapping ``slot_id -> _PendingTopReward`` for
            in-flight TOPReward score requests whose rewards have not yet been
            written back into the stored ``EnvOutput``.
    """

    def __init__(
        self,
        cfg: DictConfig | None = None,
        *,
        slot_count: int = 1,
        log_info: Callable[..., None] | None = None,
        worker_timer: Callable[..., Any] | None = None,
    ) -> None:
        train_cfg = getattr(getattr(cfg, "env", None), "train", {}) if cfg else {}

        self._log_info = log_info or _noop_log
        self._worker_timer = worker_timer or (lambda *_args, **_kwargs: nullcontext())
        self._vlm_planner: "VLMPlannerWorker | None" = None

        self._init_subtask_params(train_cfg, slot_count)
        self._init_top_reward_params(train_cfg, slot_count, cfg)
        self._validate_params(slot_count)

    def _init_subtask_params(self, train_cfg, slot_count: int) -> None:
        """Initialize subtask planning parameters from config."""
        self._subtask_interval: int = int(train_cfg.get("subtask_interval", 0))
        self._subtask_adaptive: bool = bool(train_cfg.get("subtask_adaptive", True))
        self._subtask_min_interval: int = int(train_cfg.get("subtask_min_interval", 2))
        self._subtask_plateau_window: int = max(
            int(train_cfg.get("subtask_plateau_window", 3)),
            1,
        )
        self._subtask_plateau_threshold: float = float(
            train_cfg.get("subtask_plateau_threshold", 0.01)
        )
        self._subtask_score_threshold: float = float(
            train_cfg.get("subtask_score_threshold", -0.5)
        )
        self._steps_since_subtask_update: int = 0
        self._recent_top_deltas: deque[float] = deque(
            maxlen=self._subtask_plateau_window
        )
        self._pending_subtasks: dict[int, _PendingSubtask] = {}
        self._subtask_request_counter: list[int] = [0 for _ in range(slot_count)]
        self._last_applied_subtask_request_id: list[int] = [
            0 for _ in range(slot_count)
        ]
        self._initial_task_descriptions: list[str] = [
            str(train_cfg.get("task_description", "")) for _ in range(slot_count)
        ]

    @property
    def _top_reward_enabled(self) -> bool:
        return self._dense_reward_method == "top_reward"

    def _init_top_reward_params(self, train_cfg, slot_count: int, cfg=None) -> None:
        """Initialize dense-reward parameters from config."""
        self._dense_reward_method: str = str(
            train_cfg.get("dense_reward_method", "none")
        )
        self._top_reward_max_frames: int = int(
            train_cfg.get("top_reward_max_frames", 1000)
        )
        self._episode_frames: list[np.ndarray] = []
        self._prev_top_score: float = 0.0
        self._top_reward_has_prev_score: bool = False
        self._pending_top_rewards: dict[int, _PendingTopReward] = {}
        self._episode_done_waiting_for_top_reward_reset: list[bool] = [
            False for _ in range(slot_count)
        ]

        # TOPReward fps = control_rate_hz / reward_frame_interval.
        self._control_rate_hz: float = float(train_cfg.get("control_rate_hz", 0.0))
        self._reward_frame_interval: int = int(
            train_cfg.get("reward_frame_interval", 0)
        )
        self._top_reward_fps: float = (
            self._control_rate_hz / self._reward_frame_interval
            if self._top_reward_enabled
            else _TOP_REWARD_DEFAULT_FPS
        )

    def _validate_params(self, slot_count: int) -> None:
        """Validate parameter combinations."""
        if self._top_reward_enabled and slot_count != 1:
            raise ValueError(
                "TOPReward currently supports exactly one pipeline slot. "
                "Set rollout.pipeline_slot_count=1 "
                "when env.train.dense_reward_method='top_reward'."
            )
        if self._top_reward_enabled and self._control_rate_hz <= 0:
            raise ValueError(
                "dense_reward_method='top_reward' requires env.train.control_rate_hz > 0."
            )
        if self._top_reward_enabled and self._reward_frame_interval <= 0:
            raise ValueError(
                "dense_reward_method='top_reward' requires "
                "env.train.reward_frame_interval > 0."
            )
        if self._subtask_interval > 0 and not any(self._initial_task_descriptions):
            raise ValueError(
                "Subtask planning (subtask_interval > 0) requires a non-empty "
                "env.train.task_description. Set it to a concrete episode goal "
                "(e.g. 'fold the towel')."
            )

    def set_planner_handle(self, planner_handle) -> None:
        self._vlm_planner = planner_handle
        self._log_info(
            "[EnvWorker] VLM planner handle set "
            f"(subtask_interval={self._subtask_interval})."
        )

    @staticmethod
    def _get_env(env_list: list[Any], slot_id: int) -> Any:
        return env_list[slot_id]

    @staticmethod
    def _get_inner_env(env_list: list[Any], slot_id: int) -> Any:
        env = env_list[slot_id]
        return getattr(env, "unwrapped", env)

    def get_current_task_description(self, slot_id: int, env_list: list[Any]) -> str:
        inner_env = self._get_inner_env(env_list, slot_id)
        return str(getattr(inner_env, "task_description", ""))

    def reset_subtask_update_state(self) -> None:
        self._steps_since_subtask_update = 0

    def reset_top_reward_state(self) -> None:
        self._episode_frames = []
        self._prev_top_score = 0.0
        self._top_reward_has_prev_score = False
        self._recent_top_deltas.clear()

    def reset_for_env_reset(self, slot_id: int | None = None) -> None:
        if slot_id is None:
            self.reset_subtask_update_state()
            self._episode_done_waiting_for_top_reward_reset = [
                False for _ in self._episode_done_waiting_for_top_reward_reset
            ]
        else:
            self._episode_done_waiting_for_top_reward_reset[slot_id] = False
        if self._top_reward_enabled:
            self.reset_top_reward_state()

    def reset_subtask_update_state_for_episode_done(self) -> None:
        self.reset_subtask_update_state()

    def get_top_reward_instruction(self, slot_id: int, env_list: list[Any]) -> str:
        if self._subtask_interval <= 0:
            return self._initial_task_descriptions[slot_id]
        return self.get_current_task_description(slot_id, env_list)

    def apply_subtask_update(
        self,
        slot_id: int,
        new_subtask: str,
        env_list: list[Any],
    ) -> bool:
        inner_env = self._get_inner_env(env_list, slot_id)
        if not new_subtask or not hasattr(inner_env, "task_description"):
            return False

        if new_subtask.startswith(NEW_TASK_PREFIX):
            new_task_name = new_subtask[len(NEW_TASK_PREFIX) :].strip()
            if not new_task_name:
                return False
            self._initial_task_descriptions[slot_id] = new_task_name
            new_subtask = new_task_name
            self._log_info(
                f"[EnvWorker] Main task rotated for slot {slot_id}: '{new_task_name}'"
            )

        inner_env.task_description = new_subtask
        if self._top_reward_enabled:
            self.reset_top_reward_state()
        self._log_info(
            f"[EnvWorker] Subtask updated for slot {slot_id}: '{new_subtask}'"
        )
        return True

    def _seed_top_reward_baseline_state(
        self,
        slot_id: int,
        env_list: list[Any],
        obs: dict[str, Any] | None,
        score_t: float,
    ) -> bool:
        if not self._top_reward_enabled or obs is None:
            return False

        images = self.extract_planner_images(obs)
        if not images:
            return False

        self._episode_frames = [np.asarray(images[0])]
        self._prev_top_score = float(score_t)
        self._top_reward_has_prev_score = True
        self._log_info(
            "[EnvWorker] Seeded TOPReward baseline "
            f"for slot {slot_id}: score={score_t:.4f}, "
            f"instruction='{self.get_top_reward_instruction(slot_id, env_list)}'"
        )
        return True

    def _get_baseline_obs(
        self,
        slot_id: int,
        env_list: list[Any],
        obs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve baseline observation for TOPReward seeding."""
        baseline_obs = obs
        if baseline_obs is None:
            env = self._get_env(env_list, slot_id)
            baseline_obs = getattr(env, "last_obs", None)
        return baseline_obs

    def _seed_top_reward_baseline_sync(
        self,
        slot_id: int,
        env_list: list[Any],
        obs: dict[str, Any] | None = None,
    ) -> bool:
        """Seed TOPReward after a subtask switch without emitting reward."""
        if not self._top_reward_enabled or self._vlm_planner is None:
            return False

        baseline_obs = self._get_baseline_obs(slot_id, env_list, obs)
        if baseline_obs is None:
            return False

        images = self.extract_planner_images(baseline_obs)
        if not images:
            return False

        instruction = self.get_top_reward_instruction(slot_id, env_list)
        with self._worker_timer("top_reward"):
            score_ref = self._vlm_planner.compute_top_reward.remote(
                images, instruction, fps=self._top_reward_fps
            )
            score_t = ray.get(score_ref)
        return self._seed_top_reward_baseline_state(
            slot_id, env_list, baseline_obs, score_t
        )

    async def _seed_top_reward_baseline_async(
        self,
        slot_id: int,
        env_list: list[Any],
        obs: dict[str, Any] | None = None,
    ) -> bool:
        """Async counterpart to ``_seed_top_reward_baseline_sync``."""
        if not self._top_reward_enabled or self._vlm_planner is None:
            return False

        baseline_obs = self._get_baseline_obs(slot_id, env_list, obs)
        if baseline_obs is None:
            return False

        images = self.extract_planner_images(baseline_obs)
        if not images:
            return False

        instruction = self.get_top_reward_instruction(slot_id, env_list)
        with self._worker_timer("top_reward"):
            score_ref = self._vlm_planner.compute_top_reward.remote(
                images, instruction, fps=self._top_reward_fps
            )
            score_t = await score_ref
        return self._seed_top_reward_baseline_state(
            slot_id, env_list, baseline_obs, score_t
        )

    @staticmethod
    def extract_planner_images(obs: dict[str, Any]) -> list[np.ndarray]:
        main_images = obs.get("main_images", None)
        if main_images is None:
            return []

        if isinstance(main_images, torch.Tensor):
            main_images = main_images.numpy()

        if main_images.ndim == 4:
            return [main_images[0]]
        return [main_images]

    @staticmethod
    def inject_task_description_into_obs(
        obs: dict[str, Any] | None, task_description: str
    ) -> None:
        if isinstance(obs, dict):
            obs["task_descriptions"] = [str(task_description)]

    def sync_subtask_into_env_output(
        self,
        slot_id: int,
        env_output: EnvOutput,
        task_description: str,
        env_list: list[Any],
    ) -> None:
        self.inject_task_description_into_obs(env_output.obs, task_description)
        self.inject_task_description_into_obs(env_output.final_obs, task_description)
        env = self._get_env(env_list, slot_id)
        self.inject_task_description_into_obs(
            getattr(env, "last_obs", None), task_description
        )

    def request_subtask_sync(self, slot_id: int, obs: dict[str, Any]) -> str:
        images = self.extract_planner_images(obs)
        main_task = self._initial_task_descriptions[slot_id]
        subtask_ref = self._vlm_planner.get_next_subtask.remote(images, main_task)
        return ray.get(subtask_ref)

    def submit_subtask_request(self, slot_id: int, obs: dict[str, Any]) -> None:
        if slot_id in self._pending_subtasks or self._vlm_planner is None:
            return

        images = self.extract_planner_images(obs)
        main_task = self._initial_task_descriptions[slot_id]
        self._subtask_request_counter[slot_id] += 1
        subtask_ref = self._vlm_planner.get_next_subtask.remote(images, main_task)
        self._pending_subtasks[slot_id] = _PendingSubtask(
            ref=subtask_ref,
            request_id=self._subtask_request_counter[slot_id],
        )

    def maybe_plan_initial_subtask(
        self,
        slot_id: int,
        env_output: EnvOutput,
        env_list: list[Any],
    ) -> EnvOutput:
        if self._subtask_interval <= 0 or self._vlm_planner is None:
            return env_output

        main_task = self._initial_task_descriptions[slot_id]
        current_task = self.get_current_task_description(slot_id, env_list)
        should_prime = (
            self._steps_since_subtask_update == 0 and current_task == main_task
        )
        if not should_prime:
            return env_output

        new_subtask = self.request_subtask_sync(slot_id, env_output.obs)
        if self.apply_subtask_update(slot_id, new_subtask, env_list):
            task_desc = self.get_current_task_description(slot_id, env_list)
            self.sync_subtask_into_env_output(slot_id, env_output, task_desc, env_list)
            self.reset_subtask_update_state()
            self._seed_top_reward_baseline_sync(slot_id, env_list, env_output.obs)
        return env_output

    def maybe_update_subtask(self, slot_id: int, env_list: list[Any]) -> None:
        if self._subtask_interval <= 0 or self._vlm_planner is None:
            return
        if self._episode_done_waiting_for_top_reward_reset[slot_id]:
            return

        self._steps_since_subtask_update += 1
        if self._subtask_adaptive and (
            self._steps_since_subtask_update < self._subtask_min_interval
        ):
            return

        should_trigger = self._steps_since_subtask_update >= self._subtask_interval
        if self._subtask_adaptive and self._top_reward_enabled:
            recent_deltas = list(self._recent_top_deltas)
            plateau_triggered = len(recent_deltas) >= self._subtask_plateau_window and (
                all(
                    abs(delta) < self._subtask_plateau_threshold
                    for delta in recent_deltas[-self._subtask_plateau_window :]
                )
            )
            score_triggered = bool(
                self._top_reward_has_prev_score
                and self._prev_top_score > self._subtask_score_threshold
            )
            should_trigger = should_trigger or plateau_triggered or score_triggered

        if not should_trigger:
            return

        env = self._get_env(env_list, slot_id)
        obs = getattr(env, "last_obs", None) or {}
        self.submit_subtask_request(slot_id, obs)

    def apply_resolved_subtask(
        self,
        slot_id: int,
        pending: _PendingSubtask,
        new_subtask: str,
        env_list: list[Any],
    ) -> bool:
        """Apply a resolved subtask request if it is newer than the last one.

        Stale planner replies are ignored by comparing ``pending.request_id``
        against the last applied request id for this slot. When the update is
        applied successfully, this method records the request id and resets the
        subtask-update cadence state.

        Returns:
            True if ``new_subtask`` was applied and bookkeeping was updated,
            otherwise False.
        """
        if pending.request_id <= self._last_applied_subtask_request_id[slot_id]:
            return False
        if self.apply_subtask_update(slot_id, new_subtask, env_list):
            self._last_applied_subtask_request_id[slot_id] = pending.request_id
            self.reset_subtask_update_state()
            return True
        return False

    def resolve_pending_subtask_sync(self, slot_id: int, env_list: list[Any]) -> None:
        pending = self._pending_subtasks.pop(slot_id, None)
        if pending is None:
            return
        new_subtask = ray.get(pending.ref)
        if self.apply_resolved_subtask(slot_id, pending, new_subtask, env_list):
            self._seed_top_reward_baseline_sync(slot_id, env_list)

    def submit_top_reward(
        self,
        env_output: EnvOutput,
        slot_id: int,
        env_list: list[Any],
    ) -> EnvOutput:
        if not self._top_reward_enabled or self._vlm_planner is None:
            return env_output

        with self._worker_timer("top_reward"):
            # Collect lossless reward frames captured by the server during
            # chunk execution.  When present, the server already includes
            # the final-step image as the last element, so we skip the
            # JPEG-compressed obs to avoid feeding lossy images to the VLM.
            env = env_list[slot_id]
            reward_frames = getattr(env, "_last_reward_frames", [])
            if reward_frames:
                for rf in reward_frames:
                    self._episode_frames.append(np.asarray(rf))
            else:
                # Fallback: no lossless frames available — use obs image.
                main_images = env_output.obs.get("main_images", None)
                if main_images is not None:
                    if isinstance(main_images, torch.Tensor):
                        frame = main_images[0].cpu().numpy()
                    else:
                        frame = np.asarray(main_images[0])
                    self._episode_frames.append(frame)

            if len(self._episode_frames) > self._top_reward_max_frames:
                self._episode_frames = self._episode_frames[
                    -self._top_reward_max_frames :
                ]

            instruction = self.get_top_reward_instruction(slot_id, env_list)
            score_ref = self._vlm_planner.compute_top_reward.remote(
                self._episode_frames, instruction, fps=self._top_reward_fps
            )

        if env_output.rewards is not None:
            env_output.rewards[:, -1] = 0.0
        self._pending_top_rewards[slot_id] = _PendingTopReward(
            env_output=env_output,
            score_ref=score_ref,
            done_on_step=bool(env_output.dones[:, -1].any()),
        )
        return env_output

    def _push_top_reward_to_env(self, env: Any, score_t: float, reward: float) -> None:
        push = getattr(env, "push_status_info", None)
        if callable(push):
            try:
                push(
                    values={
                        "top_reward_score": float(score_t),
                        "top_reward_delta": float(reward),
                    }
                )
            except Exception:
                pass

    def apply_resolved_top_reward(
        self,
        pending: _PendingTopReward,
        score_t: float,
        env: Any = None,
    ) -> None:
        if self._top_reward_has_prev_score:
            reward = float(score_t) - self._prev_top_score
        else:
            reward = 0.0
        self._prev_top_score = float(score_t)
        self._top_reward_has_prev_score = True
        self._recent_top_deltas.append(reward)
        if pending.env_output.rewards is not None:
            pending.env_output.rewards[:, -1] = reward
        if env is not None:
            self._push_top_reward_to_env(env, score_t, reward)
        if pending.done_on_step:
            self.reset_top_reward_state()

    def resolve_pending_top_reward_sync(
        self, slot_id: int, env_list: list[Any] | None = None
    ) -> None:
        pending = self._pending_top_rewards.pop(slot_id, None)
        if pending is None:
            return
        score_t = ray.get(pending.score_ref)
        env = env_list[slot_id] if env_list is not None else None
        self.apply_resolved_top_reward(pending, score_t, env=env)
        if pending.done_on_step:
            self._episode_done_waiting_for_top_reward_reset[slot_id] = False

    def compute_top_reward_sync(
        self,
        env_output: EnvOutput,
        slot_id: int,
        env_list: list[Any],
    ) -> EnvOutput:
        if not self._top_reward_enabled or self._vlm_planner is None:
            return env_output

        with self._worker_timer("top_reward"):
            # Collect lossless reward frames from the server (includes
            # the final-step image when available).
            env = env_list[slot_id]
            reward_frames = getattr(env, "_last_reward_frames", [])
            if reward_frames:
                for rf in reward_frames:
                    self._episode_frames.append(np.asarray(rf))
            else:
                # Fallback: no lossless frames — use obs image.
                main_images = env_output.obs.get("main_images", None)
                if main_images is not None:
                    if isinstance(main_images, torch.Tensor):
                        frame = main_images[0].cpu().numpy()
                    else:
                        frame = np.asarray(main_images[0])
                    self._episode_frames.append(frame)

            if len(self._episode_frames) > self._top_reward_max_frames:
                self._episode_frames = self._episode_frames[
                    -self._top_reward_max_frames :
                ]

            instruction = self.get_top_reward_instruction(slot_id, env_list)
            score_ref = self._vlm_planner.compute_top_reward.remote(
                self._episode_frames, instruction, fps=self._top_reward_fps
            )
            score_t = ray.get(score_ref)

        if self._top_reward_has_prev_score:
            reward = float(score_t) - self._prev_top_score
        else:
            reward = 0.0
        self._prev_top_score = float(score_t)
        self._top_reward_has_prev_score = True
        self._recent_top_deltas.append(reward)

        if env_output.rewards is not None:
            env_output.rewards[:, -1] = reward
        self._log_info(
            f"[EnvWorker] TOPReward: score={score_t:.4f}, delta={reward:.4f}"
        )
        if 0 <= slot_id < len(env_list):
            self._push_top_reward_to_env(env_list[slot_id], score_t, reward)
        return env_output

    def on_env_step(
        self,
        slot_id: int,
        env_output: EnvOutput,
        env_list: list[Any],
    ) -> EnvOutput:
        if not env_output.collect_for_training:
            self.reset_for_env_reset(slot_id)
            return env_output

        env_output = self.submit_top_reward(env_output, slot_id, env_list)
        if env_output.dones[:, -1].any():
            if self._top_reward_enabled:
                self._episode_done_waiting_for_top_reward_reset[slot_id] = True
            self.reset_subtask_update_state_for_episode_done()
        return env_output

    def resolve_pending_vlm_results_sync(
        self, slot_id: int, env_list: list[Any]
    ) -> None:
        self.resolve_pending_top_reward_sync(slot_id, env_list)
        self.resolve_pending_subtask_sync(slot_id, env_list)

    async def resolve_pending_async(self, slot_id: int, env_list: list[Any]) -> None:
        pending_top_reward = self._pending_top_rewards.pop(slot_id, None)
        if pending_top_reward is not None:
            score_t = await pending_top_reward.score_ref
            env = env_list[slot_id] if 0 <= slot_id < len(env_list) else None
            self.apply_resolved_top_reward(pending_top_reward, score_t, env=env)
            if pending_top_reward.done_on_step:
                self._episode_done_waiting_for_top_reward_reset[slot_id] = False

        pending_subtask = self._pending_subtasks.pop(slot_id, None)
        if pending_subtask is not None:
            new_subtask = await pending_subtask.ref
            if self.apply_resolved_subtask(
                slot_id, pending_subtask, new_subtask, env_list
            ):
                await self._seed_top_reward_baseline_async(slot_id, env_list)
