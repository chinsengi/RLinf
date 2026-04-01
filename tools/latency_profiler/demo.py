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

"""Demo: standalone test of LatencyProfiler with simulated timings.

Run without Beaker/Ray/GPU (just tests the profiler itself):
    python -m tools.latency_profiler.demo

For real training with profiling, use the staged embodied entrypoint, e.g.:
    bash scripts/submit_yam_training.sh --interactive --config <config_name> ...
"""

from __future__ import annotations

import random
import time

from tools.latency_profiler.profiler import LatencyProfiler

# ---------------------------------------------------------------------------
# Standalone demo with simulated timings
# ---------------------------------------------------------------------------


def run_dummy_demo(n_steps: int = 5) -> LatencyProfiler:
    """Simulate a few training steps in dummy mode."""
    profiler = LatencyProfiler(mode="dummy")

    # Simulated latency ranges (ms) for dummy mode.
    dummy_latency = {
        "weight_sync": (800, 2500),
        "rollout_inference": (80, 200),
        "env_execution": (5, 30),  # dummy: no real robot
        "vlm_reward": (100, 400),
        "recv_trajectory": (1, 15),
        "advantage_computation": (30, 150),
        "actor_training": (800, 2000),
    }

    for step in range(n_steps):
        profiler.step_start(step)
        for layer_name, (lo, hi) in dummy_latency.items():
            profiler.start(layer_name)
            time.sleep(random.uniform(lo, hi) / 1000.0)
            profiler.stop(layer_name)
        profiler.step_end()

    return profiler


def run_real_demo(n_steps: int = 5) -> LatencyProfiler:
    """Simulate a few training steps in real-robot mode."""
    profiler = LatencyProfiler(mode="real")

    # Simulated latency ranges (ms) for real mode.
    real_latency = {
        "weight_sync": (1000, 3000),
        "rollout_inference": (100, 250),
        "env_execution": (2800, 4200),  # real robot: ~3s physical
        "vlm_reward": (200, 600),
        "recv_trajectory": (20, 100),
        "advantage_computation": (50, 200),
        "actor_training": (1000, 2500),
    }

    for step in range(n_steps):
        profiler.step_start(step)
        for layer_name, (lo, hi) in real_latency.items():
            profiler.start(layer_name)
            time.sleep(random.uniform(lo, hi) / 1000.0)
            profiler.stop(layer_name)
        profiler.step_end()

    return profiler


# ---------------------------------------------------------------------------
# Integration examples (copy-paste into your runner)
# ---------------------------------------------------------------------------

INTEGRATE_SYNC_EXAMPLE = """
# --- Integration with EmbodiedRunner.run() ---

from tools.latency_profiler import LatencyProfiler

# In __init__ or before run():
profiler = LatencyProfiler(mode="real")  # or "dummy"

# Inside the run() loop, wrap each stage:

profiler.step_start(_step)

profiler.start("weight_sync")
if _step % self.weight_sync_interval == 0:
    self.update_rollout_weights()
profiler.stop("weight_sync")

profiler.start("rollout_inference")
env_handle = self.env.interact(...)
rollout_handle = self.rollout.generate(...)
profiler.stop("rollout_inference")

profiler.start("env_execution")
# env_handle runs in parallel with rollout; time until actor receives data
profiler.stop("env_execution")

profiler.start("vlm_reward")
# VLM reward is computed inside env interact loop
profiler.stop("vlm_reward")

profiler.start("recv_trajectory")
self.actor.recv_rollout_trajectories(input_channel=self.actor_channel).wait()
rollout_handle.wait()
profiler.stop("recv_trajectory")

profiler.start("advantage_computation")
self.actor.compute_advantages_and_returns().wait()
profiler.stop("advantage_computation")

profiler.start("actor_training")
actor_training_handle = self.actor.run_training()
actor_training_handle.wait()
profiler.stop("actor_training")

profiler.step_end()

# After training loop:
profiler.report()
profiler.export_csv("latency_real.csv")
"""

INTEGRATE_ASYNC_EXAMPLE = """
# --- Integration with AsyncPPOEmbodiedRunner.run() ---

from tools.latency_profiler import LatencyProfiler

profiler = LatencyProfiler(mode="real")

# env and rollout are long-running background workers in async mode.
# Time each actor-side stage:

while self.global_step < self.max_steps:
    profiler.step_start(self.global_step)

    profiler.start("recv_trajectory")
    self.actor.recv_rollout_trajectories(...).wait()
    profiler.stop("recv_trajectory")

    profiler.start("advantage_computation")
    self.actor.compute_advantages_and_returns().wait()
    profiler.stop("advantage_computation")

    profiler.start("actor_training")
    self.actor.run_training().wait()
    profiler.stop("actor_training")

    profiler.start("weight_sync")
    self.update_rollout_weights()
    profiler.stop("weight_sync")

    profiler.step_end()

profiler.report()
profiler.export_csv("latency_async.csv")
"""


def main() -> None:
    print("=" * 60)
    print("  DUMMY MODE (simulated desktop, no real robot)")
    print("=" * 60)
    dummy_profiler = run_dummy_demo(n_steps=5)
    dummy_profiler.report()
    dummy_profiler.export_csv("/tmp/latency_dummy.csv")
    dummy_profiler.export_json("/tmp/latency_dummy.json")
    print("  Exported to /tmp/latency_dummy.csv and /tmp/latency_dummy.json\n")

    print("=" * 60)
    print("  REAL MODE (simulated real-robot timings)")
    print("=" * 60)
    real_profiler = run_real_demo(n_steps=5)
    real_profiler.report()
    real_profiler.export_csv("/tmp/latency_real.csv")
    real_profiler.export_json("/tmp/latency_real.json")
    print("  Exported to /tmp/latency_real.csv and /tmp/latency_real.json\n")

    # Show the dependency graph.
    print("=" * 60)
    print("  LAYER DEPENDENCY GRAPH")
    print("=" * 60)
    from tools.latency_profiler.layers import ALL_LAYERS

    for layer in ALL_LAYERS:
        deps = ", ".join(layer.depends_on) if layer.depends_on else "(none)"
        opt = " [optional]" if layer.optional else ""
        print(f"  {layer.name:<25} <- {deps}{opt}")
    print()


if __name__ == "__main__":
    main()
