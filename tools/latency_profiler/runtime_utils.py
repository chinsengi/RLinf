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

"""Lightweight helpers shared by the latency profiler runtime and tests."""

from tools.latency_profiler.profiler import LatencyProfiler, LayerTimer


def backfill_worker_durations(
    profiler: LatencyProfiler,
    step_id: int,
    env_time: dict,
    rollout_time: dict,
    actor_time: dict | None = None,
) -> None:
    """Overwrite selected layer durations with worker-side measurements."""
    if not profiler.records:
        return
    rec = profiler.records[-1]
    if rec.step_id != step_id:
        return

    def _set_duration_if_present(
        metrics: dict | None, keys: tuple[str, ...], layer_name: str
    ) -> None:
        if not metrics:
            return
        for key in keys:
            if key in metrics:
                timer = LayerTimer(layer=layer_name)
                timer.duration_ms = metrics[key] * 1000.0
                rec.timers[layer_name] = timer
                return

    for keys, layer_name in [
        (("env_step", "env_interact_step"), "env_execution"),
        (("top_reward",), "vlm_reward"),
    ]:
        if env_time:
            for key in keys:
                if key in env_time:
                    timer = LayerTimer(layer=layer_name)
                    timer.duration_ms = env_time[key] * 1000.0
                    rec.timers[layer_name] = timer
                    break

    # Prefer "predict" (pure model forward) over "generate" (includes env
    # wait time) so that rollout_inference reflects actual inference cost.
    _set_duration_if_present(
        rollout_time,
        ("predict", "generate_one_epoch", "generate"),
        "rollout_inference",
    )
    _set_duration_if_present(actor_time, ("recv_trajectory",), "recv_trajectory")


def is_async_runtime(cfg) -> bool:
    """Return whether the staged config should run with the async runtime."""
    return str(cfg.algorithm.get("loss_type", "")).lower() == "decoupled_actor_critic"
