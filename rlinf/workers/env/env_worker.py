# Copyright 2025 The RLinf Authors.
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
from collections import defaultdict, deque
from typing import Any, Literal

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement


def _apply_action_chunk_smoothing(
    chunk_actions: torch.Tensor | np.ndarray,
    smoothing_cfg: DictConfig | dict[str, Any] | None,
) -> torch.Tensor | np.ndarray:
    """Optionally smooth a chunk of actions along the time axis.

    The smoother is intentionally lightweight and post-processes already prepared
    chunk actions right before env execution so users can quickly test whether
    visible shakiness comes from noisy predictions rather than runtime jitter.
    """
    if smoothing_cfg is None:
        return chunk_actions

    enabled = bool(smoothing_cfg.get("enabled", False))
    if not enabled:
        return chunk_actions

    method = str(smoothing_cfg.get("method", "ema")).lower()
    if method not in {"ema", "exponential_moving_average"}:
        raise ValueError(
            "env.*.action_chunk_smoothing.method must be 'ema' or "
            "'exponential_moving_average'."
        )

    alpha = float(smoothing_cfg.get("alpha", 0.6))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("env.*.action_chunk_smoothing.alpha must be in [0, 1].")

    if isinstance(chunk_actions, torch.Tensor):
        if chunk_actions.ndim < 3 or chunk_actions.shape[1] <= 1:
            return chunk_actions
        smoothed = chunk_actions.clone()
        for chunk_idx in range(1, smoothed.shape[1]):
            smoothed[:, chunk_idx, :] = (
                alpha * smoothed[:, chunk_idx, :]
                + (1.0 - alpha) * smoothed[:, chunk_idx - 1, :]
            )
        return smoothed

    chunk_actions_np = np.asarray(chunk_actions)
    if chunk_actions_np.ndim < 3 or chunk_actions_np.shape[1] <= 1:
        return chunk_actions

    smoothed_np = chunk_actions_np.astype(np.float32, copy=True)
    for chunk_idx in range(1, smoothed_np.shape[1]):
        smoothed_np[:, chunk_idx, :] = (
            alpha * smoothed_np[:, chunk_idx, :]
            + (1.0 - alpha) * smoothed_np[:, chunk_idx - 1, :]
        )
    return smoothed_np.astype(chunk_actions_np.dtype, copy=False)


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self.rollout_epoch = self.cfg.algorithm.get("rollout_epoch", 1)
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.slot_count = int(self.cfg.rollout.get("pipeline_slot_count", 1))
        self.stage_num = self.slot_count  # Legacy alias for older call sites/tests.

        # Env configurations
        self.enable_offload = self.cfg.env.train.get("enable_offload", False)
        # VLM planner integration (optional slot-indexed subtask updates).
        # When subtask_interval > 0 the env worker calls the VLM planner every
        # subtask_interval chunk steps and updates env.task_description.
        self._subtask_interval: int = int(self.cfg.env.train.get("subtask_interval", 0))
        self._subtask_adaptive: bool = bool(
            self.cfg.env.train.get("subtask_adaptive", True)
        )
        self._subtask_min_interval: int = int(
            self.cfg.env.train.get("subtask_min_interval", 2)
        )
        self._subtask_plateau_window: int = max(
            int(self.cfg.env.train.get("subtask_plateau_window", 3)),
            1,
        )
        self._subtask_plateau_threshold: float = float(
            self.cfg.env.train.get("subtask_plateau_threshold", 0.01)
        )
        self._subtask_score_threshold: float = float(
            self.cfg.env.train.get("subtask_score_threshold", -0.5)
        )
        self._subtask_require_success: bool = bool(
            self.cfg.env.train.get("subtask_require_success", False)
        )
        planner_cfg = self.cfg.get("vlm_planner", {})
        self._subtask_success_threshold: float = float(
            planner_cfg.get("success_threshold", 0.5)
        )
        self._steps_since_subtask_update: int = 0
        self._recent_top_deltas: deque[float] = deque(
            maxlen=self._subtask_plateau_window
        )
        self._vlm_planner = None  # set via set_vlm_planner() after construction

        # TOPReward dense reward state.
        self._top_reward_enabled: bool = bool(
            self.cfg.env.train.get("top_reward_enabled", False)
        )
        self._top_reward_max_frames: int = int(
            self.cfg.env.train.get("top_reward_max_frames", 16)
        )
        # NOTE: _episode_frames and _prev_top_score are worker-global (not
        # per-slot). This is correct only when there is a single pipeline slot.
        self._episode_frames: list[np.ndarray] = []
        self._prev_top_score: float = 0.0
        self._top_reward_has_prev_score: bool = False
        # there can be a sequence of tasks
        # TODO: make task_description a list to avoid uniform slot task descriptions
        self._initial_task_descriptions: list[str] = [
            str(self.cfg.env.train.get("task_description", ""))
            for _ in range(self.slot_count)
        ]
        if self._top_reward_enabled and self.slot_count != 1:
            raise ValueError(
                "TOPReward currently supports exactly one pipeline slot. "
                "Set rollout.pipeline_slot_count=1 "
                "when env.train.top_reward_enabled=true."
            )
        if self._subtask_interval > 0 and not any(self._initial_task_descriptions):
            raise ValueError(
                "Subtask planning (subtask_interval > 0) requires a non-empty "
                "env.train.task_description. Set it to a concrete episode goal "
                "(e.g. 'fold the towel')."
            )
        self._log_subtask(
            f"enabled={self._subtask_interval > 0}"
            f" interval={self._subtask_interval}"
            f" adaptive={self._subtask_adaptive}"
            f" min_interval={self._subtask_min_interval}"
            f" plateau_window={self._subtask_plateau_window}"
            f" plateau_threshold={self._subtask_plateau_threshold}"
            f" score_threshold={self._subtask_score_threshold}"
            f" require_success={self._subtask_require_success}"
            f" success_threshold={self._subtask_success_threshold}"
            f" top_reward_enabled={self._top_reward_enabled}"
            f" initial_tasks={self._initial_task_descriptions!r}"
        )
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.slot_count
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.slot_count
            )
        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
            if self.enable_eval
            else 0
        )
        self.actor_split_num = self.get_actor_split_num()

    def init_worker(self):
        self._close_enabled = False
        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.cfg.env.train.total_num_envs // self.slot_count
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.cfg.env.train.total_num_envs // self.slot_count
            ),
        }

        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.cfg.env.eval.total_num_envs // self.slot_count
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.cfg.env.eval.total_num_envs // self.slot_count
            )
        self.log_info(f"Env worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Env worker initialized with src_ranks: {self.src_ranks}")
        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = (
            get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            if self.enable_eval
            else None
        )

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = (
            get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            if self.enable_eval
            else None
        )

        if not self.only_eval:
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
        if self.enable_eval:
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )

        if not self.only_eval:
            self._init_env()

        self._close_enabled = True

    def close_envs(self):
        if not getattr(self, "_close_enabled", False):
            return
        for env in getattr(self, "env_list", []):
            try:
                env.close()
            except Exception:
                pass
        for env in getattr(self, "eval_env_list", []):
            try:
                env.close()
            except Exception:
                pass

    def _log_subtask(self, message: str) -> None:
        """Route subtask planner instrumentation through the worker logger."""
        self.log_info(f"[Subtask] {message}")

    def set_vlm_planner(self, planner_handle) -> None:
        """Inject a VLMPlannerWorker Ray handle for VLM-driven features.

        Call this from the runner after both workers have been created, passing
        the Ray remote handle (e.g. ``vlm_planner_actor``).  The handle is used
        for two purposes:

        1. **TOPReward scoring** (when ``top_reward_enabled=True``): each chunk
           step calls ``planner_handle.compute_top_reward.remote()`` to get a
           dense progress reward from log P("True" | frames, instruction).
        2. **Subtask planning** (when ``subtask_interval > 0``): every
           ``subtask_interval`` chunk steps calls
           ``planner_handle.get_next_subtask.remote(images, main_task)``
           and updates ``env.task_description`` (and ``SetTaskDescription``
           on the server).

        Args:
            planner_handle: A VLMPlannerWorker Ray actor handle, or None to
                disable VLM-driven features.
        """
        self._vlm_planner = planner_handle
        self._log_subtask(
            f"planner_attached={planner_handle is not None}"
            f" interval={self._subtask_interval}"
            f" adaptive={self._subtask_adaptive}"
        )
        self.log_info(
            f"[EnvWorker] VLM planner handle set (subtask_interval="
            f"{self._subtask_interval})."
        )

    def _get_current_task_description(self, slot_id: int) -> str:
        env = self.env_list[slot_id]
        inner_env = getattr(env, "unwrapped", env)
        return str(getattr(inner_env, "task_description", ""))

    def _reset_subtask_update_state(self) -> None:
        """Reset per-episode subtask planner cadence state."""
        self._log_subtask(
            "counter_reset "
            f"previous_steps_since_update={self._steps_since_subtask_update}"
        )
        self._steps_since_subtask_update = 0

    def _get_top_reward_instruction(self, slot_id: int) -> str:
        if self._subtask_interval <= 0:
            return self._initial_task_descriptions[slot_id]
        return self._get_current_task_description(slot_id)

    def _apply_subtask_update(self, slot_id: int, new_subtask: str) -> bool:
        env = self.env_list[slot_id]
        inner_env = getattr(env, "unwrapped", env)
        if not new_subtask or not hasattr(inner_env, "task_description"):
            self._log_subtask(
                f"apply_skipped slot={slot_id}"
                f" new_subtask={new_subtask!r}"
                f" has_task_description={hasattr(inner_env, 'task_description')}"
            )
            return False

        inner_env.task_description = new_subtask
        if self._top_reward_enabled:
            self._reset_top_reward_state()
        self._log_subtask(f"apply_success slot={slot_id} subtask={new_subtask!r}")
        self.log_info(
            f"[EnvWorker] Subtask updated for slot {slot_id}: '{new_subtask}'"
        )
        return True

    @staticmethod
    async def _await_ray_ref(object_ref: Any) -> Any:
        """Resolve a Ray ref without blocking the async event loop."""
        return await asyncio.to_thread(ray.get, object_ref)

    @staticmethod
    def _extract_planner_images(obs: Any) -> list[np.ndarray]:
        images = []
        if not isinstance(obs, dict):
            return images

        main_images = obs.get("main_images", None)
        if main_images is None:
            return images

        if isinstance(main_images, torch.Tensor):
            main_images = main_images.numpy()

        if main_images.ndim == 4:
            images.append(main_images[0])
        else:
            images.append(main_images)
        return images

    @staticmethod
    def _inject_task_description_into_obs(
        obs: dict[str, Any] | None, task_description: str
    ) -> None:
        if isinstance(obs, dict):
            obs["task_descriptions"] = [str(task_description)]

    def _sync_subtask_into_env_output(
        self, slot_id: int, env_output: EnvOutput, task_description: str
    ) -> None:
        self._inject_task_description_into_obs(env_output.obs, task_description)
        self._inject_task_description_into_obs(env_output.final_obs, task_description)
        env = self.env_list[slot_id]
        self._inject_task_description_into_obs(
            getattr(env, "last_obs", None), task_description
        )

    def _restore_main_task_description(
        self, slot_id: int, obs: dict[str, Any] | None = None
    ) -> None:
        """Restore the environment-visible task text to the main episode goal."""
        main_task = self._initial_task_descriptions[slot_id]
        env = self.env_list[slot_id]
        inner_env = getattr(env, "unwrapped", env)
        if hasattr(inner_env, "task_description"):
            inner_env.task_description = main_task
        self._inject_task_description_into_obs(obs, main_task)
        self._inject_task_description_into_obs(
            getattr(env, "last_obs", None), main_task
        )

    def _request_subtask(self, slot_id: int, obs: Any, *, reason: str) -> str:
        images = self._extract_planner_images(obs)
        main_task = self._initial_task_descriptions[slot_id]
        current_task = self._get_current_task_description(slot_id)
        self._log_subtask(
            f"requesting slot={slot_id}"
            f" reason={reason}"
            f" main_task={main_task!r}"
            f" current_task={current_task!r}"
            f" num_images={len(images)}"
        )
        subtask_ref = self._vlm_planner.get_next_subtask.remote(
            images,
            main_task,
            current_task,
        )
        new_subtask: str = ray.get(subtask_ref)
        self._log_subtask(
            f"generated slot={slot_id} reason={reason} subtask={new_subtask!r}"
        )
        return new_subtask

    async def _request_subtask_async(
        self, slot_id: int, obs: Any, *, reason: str
    ) -> str:
        images = self._extract_planner_images(obs)
        main_task = self._initial_task_descriptions[slot_id]
        current_task = self._get_current_task_description(slot_id)
        self._log_subtask(
            f"requesting slot={slot_id}"
            f" reason={reason}"
            f" main_task={main_task!r}"
            f" current_task={current_task!r}"
            f" num_images={len(images)}"
        )
        subtask_ref = self._vlm_planner.get_next_subtask.remote(
            images,
            main_task,
            current_task,
        )
        new_subtask: str = await self._await_ray_ref(subtask_ref)
        self._log_subtask(
            f"generated slot={slot_id} reason={reason} subtask={new_subtask!r}"
        )
        return new_subtask

    def _is_current_subtask_complete(
        self, slot_id: int, obs: Any, *, reason: str
    ) -> bool:
        """Return whether the current subtask is complete enough to advance."""
        if not getattr(self, "_subtask_require_success", False):
            return True

        evaluator = getattr(self._vlm_planner, "evaluate_subtask", None)
        if evaluator is None or not hasattr(evaluator, "remote"):
            return True

        current_task = self._get_current_task_description(slot_id).strip()
        main_task = str(self._initial_task_descriptions[slot_id]).strip()
        if not current_task or current_task == main_task:
            return True

        images = self._extract_planner_images(obs)
        self._log_subtask(
            f"evaluating slot={slot_id}"
            f" reason={reason}"
            f" current_task={current_task!r}"
            f" num_images={len(images)}"
        )
        score_ref = evaluator.remote(images, current_task)
        success_score = float(ray.get(score_ref))
        success_threshold = float(getattr(self, "_subtask_success_threshold", 0.5))
        completed = success_score >= success_threshold
        self._log_subtask(
            f"completion_check slot={slot_id}"
            f" reason={reason}"
            f" current_task={current_task!r}"
            f" success_score={success_score:.4f}"
            f" success_threshold={success_threshold:.4f}"
            f" completed={completed}"
        )
        return completed

    async def _is_current_subtask_complete_async(
        self, slot_id: int, obs: Any, *, reason: str
    ) -> bool:
        """Async variant of `_is_current_subtask_complete`."""
        if not getattr(self, "_subtask_require_success", False):
            return True

        evaluator = getattr(self._vlm_planner, "evaluate_subtask", None)
        if evaluator is None or not hasattr(evaluator, "remote"):
            return True

        current_task = self._get_current_task_description(slot_id).strip()
        main_task = str(self._initial_task_descriptions[slot_id]).strip()
        if not current_task or current_task == main_task:
            return True

        images = self._extract_planner_images(obs)
        self._log_subtask(
            f"evaluating slot={slot_id}"
            f" reason={reason}"
            f" current_task={current_task!r}"
            f" num_images={len(images)}"
        )
        score_ref = evaluator.remote(images, current_task)
        success_score = float(await self._await_ray_ref(score_ref))
        success_threshold = float(getattr(self, "_subtask_success_threshold", 0.5))
        completed = success_score >= success_threshold
        self._log_subtask(
            f"completion_check slot={slot_id}"
            f" reason={reason}"
            f" current_task={current_task!r}"
            f" success_score={success_score:.4f}"
            f" success_threshold={success_threshold:.4f}"
            f" completed={completed}"
        )
        return completed

    def _maybe_plan_initial_subtask(
        self, slot_id: int, env_output: EnvOutput
    ) -> EnvOutput:
        if self._subtask_interval <= 0 or self._vlm_planner is None:
            self._log_subtask(
                f"initial_skip slot={slot_id}"
                f" enabled={self._subtask_interval > 0}"
                f" planner_attached={self._vlm_planner is not None}"
            )
            return env_output

        main_task = self._initial_task_descriptions[slot_id]
        current_task = self._get_current_task_description(slot_id)
        should_prime = (
            self._steps_since_subtask_update == 0 and current_task == main_task
        )
        self._log_subtask(
            f"initial_check slot={slot_id}"
            f" should_prime={should_prime}"
            f" steps_since_update={self._steps_since_subtask_update}"
            f" main_task={main_task!r}"
            f" current_task={current_task!r}"
        )
        if not should_prime:
            return env_output

        new_subtask = self._request_subtask(slot_id, env_output.obs, reason="initial")
        if self._apply_subtask_update(slot_id, new_subtask):
            self._sync_subtask_into_env_output(slot_id, env_output, new_subtask)
            self._reset_subtask_update_state()
        return env_output

    async def _maybe_plan_initial_subtask_async(
        self, slot_id: int, env_output: EnvOutput
    ) -> EnvOutput:
        if self._subtask_interval <= 0 or self._vlm_planner is None:
            self._log_subtask(
                f"initial_skip slot={slot_id}"
                f" enabled={self._subtask_interval > 0}"
                f" planner_attached={self._vlm_planner is not None}"
            )
            return env_output

        main_task = self._initial_task_descriptions[slot_id]
        current_task = self._get_current_task_description(slot_id)
        should_prime = (
            self._steps_since_subtask_update == 0 and current_task == main_task
        )
        self._log_subtask(
            f"initial_check slot={slot_id}"
            f" should_prime={should_prime}"
            f" steps_since_update={self._steps_since_subtask_update}"
            f" main_task={main_task!r}"
            f" current_task={current_task!r}"
        )
        if not should_prime:
            return env_output

        new_subtask = await self._request_subtask_async(
            slot_id, env_output.obs, reason="initial"
        )
        if self._apply_subtask_update(slot_id, new_subtask):
            self._sync_subtask_into_env_output(slot_id, env_output, new_subtask)
            self._reset_subtask_update_state()
        return env_output

    def _maybe_update_subtask(self, slot_id: int) -> None:
        """Optionally call the VLM planner to refresh the current subtask.

        Called each chunk step from :meth:`interact`.  When
        ``subtask_interval > 0`` and a VLM planner handle is available, polls
        the planner and writes the new subtask description into the env so that
        the next observation includes the updated ``task_descriptions`` text.

        Args:
            slot_id: Pipeline slot index (used to select the correct env
                from ``self.env_list``).
        """
        if self._subtask_interval <= 0 or self._vlm_planner is None:
            self._log_subtask(
                f"skip slot={slot_id}"
                f" enabled={self._subtask_interval > 0}"
                f" planner_attached={self._vlm_planner is not None}"
            )
            return

        self._steps_since_subtask_update += 1
        self._log_subtask(
            f"tick slot={slot_id}"
            f" steps_since_update={self._steps_since_subtask_update}"
            f" interval={self._subtask_interval}"
            f" adaptive={self._subtask_adaptive}"
        )
        if self._subtask_adaptive and (
            self._steps_since_subtask_update < self._subtask_min_interval
        ):
            self._log_subtask(
                f"waiting_min_interval slot={slot_id}"
                f" steps_since_update={self._steps_since_subtask_update}"
                f" min_interval={self._subtask_min_interval}"
            )
            return

        interval_triggered = self._steps_since_subtask_update >= self._subtask_interval
        plateau_triggered = False
        score_triggered = False
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
        self._log_subtask(
            f"trigger_check slot={slot_id}"
            f" should_trigger={should_trigger}"
            f" interval_triggered={interval_triggered}"
            f" plateau_triggered={plateau_triggered}"
            f" score_triggered={score_triggered}"
            f" has_prev_top_score={self._top_reward_has_prev_score}"
            f" prev_top_score={self._prev_top_score}"
            f" recent_top_deltas={list(self._recent_top_deltas)!r}"
        )

        if not should_trigger:
            self._log_subtask(
                f"not_triggered slot={slot_id}"
                f" steps_since_update={self._steps_since_subtask_update}"
            )
            return

        env = self.env_list[slot_id]
        obs = getattr(env, "last_obs", None) or {}
        if not self._is_current_subtask_complete(slot_id, obs, reason="interval"):
            self._log_subtask(
                f"hold_current slot={slot_id}"
                f" reason=interval"
                f" current_task={self._get_current_task_description(slot_id)!r}"
            )
            return
        new_subtask = self._request_subtask(slot_id, obs, reason="interval")

        if self._apply_subtask_update(slot_id, new_subtask):
            self._reset_subtask_update_state()

    async def _maybe_update_subtask_async(self, slot_id: int) -> None:
        """Async variant of subtask refresh for the interact event loop."""
        if self._subtask_interval <= 0 or self._vlm_planner is None:
            self._log_subtask(
                f"skip slot={slot_id}"
                f" enabled={self._subtask_interval > 0}"
                f" planner_attached={self._vlm_planner is not None}"
            )
            return

        self._steps_since_subtask_update += 1
        self._log_subtask(
            f"tick slot={slot_id}"
            f" steps_since_update={self._steps_since_subtask_update}"
            f" interval={self._subtask_interval}"
            f" adaptive={self._subtask_adaptive}"
        )
        if self._subtask_adaptive and (
            self._steps_since_subtask_update < self._subtask_min_interval
        ):
            self._log_subtask(
                f"waiting_min_interval slot={slot_id}"
                f" steps_since_update={self._steps_since_subtask_update}"
                f" min_interval={self._subtask_min_interval}"
            )
            return

        interval_triggered = self._steps_since_subtask_update >= self._subtask_interval
        plateau_triggered = False
        score_triggered = False
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
        self._log_subtask(
            f"trigger_check slot={slot_id}"
            f" should_trigger={should_trigger}"
            f" interval_triggered={interval_triggered}"
            f" plateau_triggered={plateau_triggered}"
            f" score_triggered={score_triggered}"
            f" has_prev_top_score={self._top_reward_has_prev_score}"
            f" prev_top_score={self._prev_top_score}"
            f" recent_top_deltas={list(self._recent_top_deltas)!r}"
        )

        if not should_trigger:
            self._log_subtask(
                f"not_triggered slot={slot_id}"
                f" steps_since_update={self._steps_since_subtask_update}"
            )
            return

        env = self.env_list[slot_id]
        obs = getattr(env, "last_obs", None) or {}
        if not await self._is_current_subtask_complete_async(
            slot_id, obs, reason="interval"
        ):
            self._log_subtask(
                f"hold_current slot={slot_id}"
                f" reason=interval"
                f" current_task={self._get_current_task_description(slot_id)!r}"
            )
            return
        new_subtask = await self._request_subtask_async(slot_id, obs, reason="interval")

        if self._apply_subtask_update(slot_id, new_subtask):
            self._reset_subtask_update_state()

    def _compute_top_reward(self, env_output: EnvOutput, slot_id: int) -> EnvOutput:
        """Compute TOPReward dense progress reward and inject into env_output.

        Extracts the current frame, queries the VLM planner for a progress
        score, and writes ``score_t - score_{t-1}`` into the reward tensor.

        Args:
            env_output: The env output from ``env_interact_step``.
            slot_id: Pipeline slot index.

        Returns:
            Modified env_output with dense reward injected.
        """
        if not self._top_reward_enabled or self._vlm_planner is None:
            return env_output

        with self.worker_timer("top_reward"):
            # Extract current frame from observation.
            main_images = env_output.obs.get("main_images", None)
            if main_images is not None:
                if isinstance(main_images, torch.Tensor):
                    frame = main_images[0].cpu().numpy()  # (H, W, 3)
                else:
                    frame = np.asarray(main_images[0])
                self._episode_frames.append(frame)

            # Trim to max frames.
            if len(self._episode_frames) > self._top_reward_max_frames:
                self._episode_frames = self._episode_frames[
                    -self._top_reward_max_frames :
                ]

            # Get instruction from the unwrapped env so wrapper attribute shadowing
            # can never return a stale value.
            instruction = self._get_top_reward_instruction(slot_id)

            score_ref = self._vlm_planner.compute_top_reward.remote(
                self._episode_frames, instruction
            )
            score_t = ray.get(score_ref)

        if self._top_reward_has_prev_score:
            reward = float(score_t) - self._prev_top_score
        else:
            # Seed the new episode/subtask segment with the first score rather
            # than comparing it against the reset sentinel.
            reward = 0.0
        self._prev_top_score = float(score_t)
        self._top_reward_has_prev_score = True
        self._recent_top_deltas.append(reward)

        # Inject reward into the last chunk position.
        if env_output.rewards is not None:
            env_output.rewards[:, -1] = reward
        self.log_info(f"[EnvWorker] TOPReward: score={score_t:.4f}, delta={reward:.4f}")

        return env_output

    async def _compute_top_reward_async(
        self, env_output: EnvOutput, slot_id: int
    ) -> EnvOutput:
        """Async variant of TOPReward scoring for the interact event loop."""
        if not self._top_reward_enabled or self._vlm_planner is None:
            return env_output

        with self.worker_timer("top_reward"):
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

            instruction = self._get_top_reward_instruction(slot_id)
            score_ref = self._vlm_planner.compute_top_reward.remote(
                self._episode_frames, instruction
            )
            score_t = await self._await_ray_ref(score_ref)

        if self._top_reward_has_prev_score:
            reward = float(score_t) - self._prev_top_score
        else:
            reward = 0.0
        self._prev_top_score = float(score_t)
        self._top_reward_has_prev_score = True
        self._recent_top_deltas.append(reward)

        if env_output.rewards is not None:
            env_output.rewards[:, -1] = reward
        self.log_info(f"[EnvWorker] TOPReward: score={score_t:.4f}, delta={reward:.4f}")

        return env_output

    def _reset_top_reward_state(self) -> None:
        """Reset TOPReward episode state for a new episode."""
        self._episode_frames = []
        self._prev_top_score = 0.0
        self._top_reward_has_prev_score = False
        self._recent_top_deltas.clear()

    def _handle_episode_boundary(self, slot_id: int, env_output: EnvOutput) -> None:
        """Reset per-episode planner state after an episode-ending step."""
        if env_output.dones is None or not env_output.dones[:, -1].any():
            return

        if self._top_reward_enabled:
            self._reset_top_reward_state()
        self._reset_subtask_update_state()
        if self.cfg.env.train.auto_reset:
            self._restore_main_task_description(slot_id, env_output.obs)

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for slot_id in range(self.slot_count):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.slot_count + slot_id,
                total_num_processes=self._world_size * self.slot_count,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    stats_sample_ratio=getattr(
                        env_cfg.data_collection, "stats_sample_ratio", 0.1
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute rollout peer ranks for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for both sending
        env outputs and receiving action chunks.

        Args:
            batch_size: Total env batch size per pipeline slot across all workers.

        Returns:
            Ordered ``(rollout_rank, batch_size)`` tuples this env worker should send
            env outputs to.
        """
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute rollout source ranks and sizes for receiving action chunks."""
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            dst_rank=self._rank,
        )

    def _init_env(self):
        if self.cfg.env.train.auto_reset:
            for i in range(self.slot_count):
                extracted_obs, _ = self.env_list[i].reset()
                self._restore_main_task_description(i, extracted_obs)
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))

                if self.enable_offload and hasattr(self.env_list[i], "offload"):
                    self.env_list[i].offload()

    def _get_zero_dones(self) -> torch.Tensor:
        return (
            torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
            .unsqueeze(1)
            .repeat(1, self.cfg.actor.model.num_action_chunks)
        )

    def _reset_envs_for_next_rollout_epoch(self) -> list[EnvOutput]:
        env_outputs: list[EnvOutput] = []
        dones = self._get_zero_dones()
        terminations = dones.clone()
        truncations = dones.clone()
        for slot_id in range(self.slot_count):
            self.env_list[slot_id].is_start = True
            extracted_obs, infos = self.env_list[slot_id].reset()
            self._restore_main_task_description(slot_id, extracted_obs)
            env_outputs.append(
                EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
            )
        return env_outputs

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self,
        chunk_actions: torch.Tensor,
        slot_id: int,
        *,
        compute_top_reward: bool = True,
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        chunk_actions = _apply_action_chunk_smoothing(
            chunk_actions,
            self.cfg.env.train.get("action_chunk_smoothing", None),
        )
        env_info = {}

        # Only time the actual environment interaction (chunk_step) so the
        # latency profiler reports pure env latency, excluding action
        # preparation and post-processing.
        with self.worker_timer("env_step"):
            (
                obs_list,
                chunk_rewards,
                chunk_terminations,
                chunk_truncations,
                infos_list,
            ) = self.env_list[slot_id].chunk_step(chunk_actions)

        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )

        if compute_top_reward:
            env_output = self._compute_top_reward(env_output, slot_id)
            self._handle_episode_boundary(slot_id, env_output)

        return env_output, env_info

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, slot_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        if not self.enable_eval:
            raise RuntimeError(
                "EnvWorker.evaluate_step called without env.eval configured."
            )

        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        chunk_actions = _apply_action_chunk_smoothing(
            chunk_actions,
            self.cfg.env.eval.get("action_chunk_smoothing", None),
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[slot_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
        )
        return env_output, env_info

    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        """Receive and merge chunked actions for the current env worker.

        The method fetches one action shard from each mapped rollout source rank
        under a deterministic channel key pattern and concatenates them on the
        batch dimension.

        Args:
            input_channel: Channel carrying rollout->env action chunks.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            Concatenated action chunk array with shape ``[num_envs_per_stage, ...]``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_actions"
                ),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        chunk_action = np.concatenate(chunk_action, axis=0)
        expected_total_size = sum(size for _, size in src_ranks_and_sizes)
        assert chunk_action.shape[0] == expected_total_size, (
            f"Expected concatenated action size {expected_total_size}, got {chunk_action.shape[0]}."
        )
        return chunk_action

    def recv_rollout_results(
        self, input_channel: Channel, mode="train"
    ) -> RolloutResult:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        rollout_results: list[RolloutResult] = []

        def _infer_rollout_batch_size(rollout_result: RolloutResult) -> int:
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(rollout_result, field_name, None)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            if rollout_result.forward_inputs:
                first_tensor = next(iter(rollout_result.forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return first_tensor.shape[0]
            raise ValueError("Cannot infer batch size from rollout result.")

        for src_rank, expected_size in src_ranks_and_sizes:
            rollout_result = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_rollout_results"
                ),
            )

            actual_size = _infer_rollout_batch_size(rollout_result)
            assert actual_size == expected_size, (
                f"Expected rollout result size {expected_size} from rollout rank {src_rank}, "
                f"got batch size {actual_size}."
            )

            rollout_results.append(rollout_result)

        return RolloutResult.merge_rollout_results(rollout_results)

    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.slot_count):
                if self.cfg.env.train.video_cfg.save_video and isinstance(
                    self.env_list[i], RecordVideo
                ):
                    self.env_list[i].flush_video()
                if hasattr(self.env_list[i], "update_reset_state_ids"):
                    self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.slot_count):
                if self.cfg.env.eval.video_cfg.save_video and isinstance(
                    self.eval_env_list[i], RecordVideo
                ):
                    self.eval_env_list[i].flush_video()
                if not self.cfg.env.eval.auto_reset:
                    if hasattr(self.eval_env_list[i], "update_reset_state_ids"):
                        self.eval_env_list[i].update_reset_state_ids()

    def split_env_batch(
        self,
        env_batch: dict[str, Any],
        sizes: list[int],
        mode: Literal["train", "eval"],
    ) -> list[dict[str, Any]]:
        """Split one env batch dict into size-specified sub-batches along dim-0.

        Tensor values are chunked on dim-0; list values are sliced proportionally;
        nested dict values are split recursively.

        Args:
            env_batch: Env output dictionary produced by ``EnvOutput.to_dict``.
            sizes: Batch sizes for each destination rank.
            mode: Rollout mode used for list-length validation.

        Returns:
            A list of split env batches, one item per destination rank.
        """
        count = len(sizes)
        total_size = sum(sizes)
        splitted_env_batches = [{} for _ in range(count)]
        for key, value in env_batch.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == total_size, (
                    f"Tensor field '{key}' expected batch size {total_size}, got {value.shape[0]}."
                )
                splitted_values = torch.split(value, sizes, dim=0)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_values[i].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.train_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.train_num_envs_per_stage} "
                        f"(train_num_envs_per_stage), got {length}"
                    )
                elif mode == "eval":
                    assert length == self.eval_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.eval_num_envs_per_stage} "
                        f"(eval_num_envs_per_stage), got {length}"
                    )
                assert length == total_size, (
                    f"List field '{key}' expected length {total_size}, got {length}."
                )
                begin = 0
                for i, size in enumerate(sizes):
                    splitted_env_batches[i][key] = value[begin : begin + size]
                    begin += size
            elif isinstance(value, dict):
                splitted_sub_batches = self.split_env_batch(value, sizes, mode)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_sub_batches[i]
            else:
                for i in range(count):
                    splitted_env_batches[i][key] = value

        return splitted_env_batches

    def send_env_batch(
        self,
        output_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Send split env batches to mapped rollout ranks.

        Each destination rank receives one split batch via a stable key built from
        ``src_rank``, ``dst_rank`` and ``mode``.

        Args:
            output_channel: Channel carrying env->rollout outputs.
            env_batch: Env output dictionary for one pipeline slot.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = self.split_env_batch(env_batch, split_sizes, mode)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            output_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=f"{mode}_obs"),
            )

    def bootstrap_step(self) -> list[EnvOutput]:
        reset_on_rollout_epoch = bool(
            self.cfg.env.train.get("reset_on_rollout_epoch", True)
        )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for slot_id in range(self.slot_count):
                should_reset = (
                    reset_on_rollout_epoch or len(self.last_obs_list) <= slot_id
                )
                if should_reset:
                    self._reset_subtask_update_state()
                    if self._top_reward_enabled:
                        self._reset_top_reward_state()
                    self.env_list[slot_id].is_start = True
                    extracted_obs, infos = self.env_list[slot_id].reset()
                    self._restore_main_task_description(slot_id, extracted_obs)
                    intervene_actions = None
                    intervene_flags = None
                else:
                    extracted_obs = self.last_obs_list[slot_id]
                    infos = {}
                    intervene_actions = self.last_intervened_info_list[slot_id][0]
                    intervene_flags = self.last_intervened_info_list[slot_id][1]
                dones = self._get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=intervene_actions,
                    intervene_flags=intervene_flags,
                )
                env_outputs.append(env_output)
        else:
            dones = self._get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for slot_id in range(self.slot_count):
                env_output = EnvOutput(
                    obs=self.last_obs_list[slot_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[slot_id][0],
                    intervene_flags=self.last_intervened_info_list[slot_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def _get_zero_dones(self) -> torch.Tensor:
        return (
            torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
            .unsqueeze(1)
            .repeat(1, self.cfg.actor.model.num_action_chunks)
        )

    def _reset_envs_for_next_rollout_epoch(self) -> list[EnvOutput]:
        env_outputs: list[EnvOutput] = []
        self._reset_subtask_update_state()
        if self._top_reward_enabled:
            self._reset_top_reward_state()
        dones = self._get_zero_dones()
        terminations = dones.clone()
        truncations = dones.clone()
        for slot_id in range(self.slot_count):
            self.env_list[slot_id].is_start = True
            extracted_obs, infos = self.env_list[slot_id].reset()
            env_outputs.append(
                EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
            )
        return env_outputs

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any], epoch: int
    ):
        for key, value in env_info.items():
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                if key in env_metrics and len(env_metrics[key]) > epoch:
                    env_metrics[key][epoch] = value
                else:
                    env_metrics[key].append(value)
            else:
                env_metrics[key].append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: Trajectory = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)

    async def _run_interact_once(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                rollout_horizon_steps=self.cfg.env.train.rollout_horizon_steps,
            )
            for _ in range(self.slot_count)
        ]
        env_metrics = defaultdict(list)

        for epoch in range(self.rollout_epoch):
            env_outputs = self.bootstrap_step()
            for slot_id in range(self.slot_count):
                env_output: EnvOutput = await self._maybe_plan_initial_subtask_async(
                    slot_id, env_outputs[slot_id]
                )
                env_outputs[slot_id] = env_output
                env_batch = env_output.to_dict()
                self.send_env_batch(
                    output_channel,
                    {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    },
                )

            for _ in range(self.n_train_chunk_steps):
                for slot_id in range(self.slot_count):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[slot_id]
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[slot_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    rollout_result = self.recv_rollout_results(
                        input_channel, mode="train"
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=rollout_result.prev_logprobs
                        if self.collect_prev_infos
                        else None,
                        prev_values=rollout_result.prev_values
                        if self.collect_prev_infos
                        else None,
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                    )
                    self.rollout_results[slot_id].append_step_result(chunk_step_result)

                    env_output, env_info = self.env_interact_step(
                        rollout_result.actions,
                        slot_id,
                        compute_top_reward=False,
                    )
                    env_output = await self._compute_top_reward_async(
                        env_output, slot_id
                    )
                    self._handle_episode_boundary(slot_id, env_output)
                    await self._maybe_update_subtask_async(slot_id)
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                    )
                    if self.collect_transitions:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[slot_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[slot_id] = env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            for slot_id in range(self.slot_count):
                env_output = env_outputs[slot_id]
                if env_output.intervene_actions is not None:
                    self.rollout_results[slot_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

                rollout_result = self.recv_rollout_results(input_channel, mode="train")
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values
                )
                chunk_step_result = ChunkStepResult(
                    prev_values=rollout_result.prev_values
                    if self.collect_prev_infos
                    else None,
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                )
                self.rollout_results[slot_id].append_step_result(chunk_step_result)

            if not self.cfg.env.train.auto_reset and self.cfg.env.train.get(
                "reset_on_rollout_epoch_end", False
            ):
                env_outputs = self._reset_envs_for_next_rollout_epoch()

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if actor_channel is not None:
            for slot_id in range(self.slot_count):
                await self.send_rollout_trajectories(
                    self.rollout_results[slot_id], actor_channel
                )

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            output_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        return env_metrics

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        if not self.enable_eval:
            raise RuntimeError("EnvWorker.evaluate called with evaluation disabled.")

        eval_metrics = defaultdict(list)

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for slot_id in range(self.slot_count):
                    self.eval_env_list[slot_id].is_start = True
                    extracted_obs, infos = self.eval_env_list[slot_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for slot_id in range(self.slot_count):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, slot_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch
                            == self.cfg.algorithm.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            self.finish_rollout(mode="eval")
        for slot_id in range(self.slot_count):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[slot_id], "offload"
            ):
                self.eval_env_list[slot_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.slot_count
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num
