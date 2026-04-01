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

"""Run embodied RL training with latency profiling enabled.

Drop-in replacement for ``train_embodied_agent_staged.py`` /
``train_embodied_agent_staged_async.py``.  Uses the same Hydra config
interface, same Beaker submission flow — just swap the entry point.

The full pipeline runs identically to normal training.  Dummy mode only
means YAMEnv does not connect to real hardware (returns zero observations);
everything else — Beaker, Ray, gRPC, actor training, rollout inference,
VLM reward — runs for real.

Topology 1 — Remote env (single Beaker node, robot server on desktop)::

    # Step 1: submit Beaker job
    bash scripts/submit_yam_training.sh --interactive \
        --config yam_ppo_openpi_sync \
        --model-path thomas0829/folding_towel_pi05

    # Step 2: start robot server on desktop (add --dummy for no real robot)
    bash scripts/start_robot_server.sh \
        --config .../yam_pi05_follower.yaml \
        --use-follower-servers --remote-host beaker-0 [--dummy]

Topology 2 — Desktop-driven (Beaker GPUs + desktop env)::

    # Step 1: submit Beaker cluster
    bash scripts/submit_yam_beaker_cluster.sh \
        --config yam_ppo_openpi_desktop_sync

    # Step 2: join from desktop (set env.train.is_dummy=True for no real robot)
    bash scripts/join_beaker_cluster.sh \
        --head-ip <beaker-tailscale-ip> \
        --config yam_ppo_openpi_desktop_sync \
        --model-path thomas0829/folding_towel_pi05

Results are saved to ``<log_path>/<experiment_name>/latency_report/``
when training finishes or is interrupted (Ctrl+C).
"""

import json
import os
import sys

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

# Ensure repo root is on sys.path so ``tools.*`` imports work.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlinf.config import validate_cfg  # noqa: E402
from rlinf.utils.logging import get_logger  # noqa: E402
from tools.latency_profiler.profiler import LatencyProfiler  # noqa: E402
from tools.latency_profiler.runtime_utils import (  # noqa: E402
    backfill_worker_durations,
    is_async_runtime,
)

mp.set_start_method("spawn", force=True)

logger = get_logger()


# ---------------------------------------------------------------------------
# Detect mode from config
# ---------------------------------------------------------------------------


def _detect_mode(cfg) -> str:
    """Return ``"dummy"`` or ``"real"`` based on the Hydra config.

    Dummy mode means YAMEnv returns zero observations without connecting to
    real hardware.  The rest of the pipeline (Beaker, Ray, gRPC, inference,
    training, VLM reward) runs identically in both modes.

    Detection:
    - Desktop-direct topology: ``env.train.is_dummy`` in the YAML config.
    - Remote env topology: the robot server is started with ``--dummy`` on the
      desktop side, which is not visible in the Hydra config.  In that case
      the user can pass ``+profiler.mode=dummy`` as a Hydra override, or this
      defaults to ``"real"`` (conservative).
    """
    # Explicit user override via Hydra: +profiler.mode=dummy
    profiler_cfg = getattr(cfg, "profiler", None)
    if profiler_cfg is not None:
        explicit_mode = getattr(profiler_cfg, "mode", None)
        if explicit_mode in ("dummy", "real"):
            return str(explicit_mode)

    # Desktop-direct topology: is_dummy flag in env config.
    if getattr(cfg.env.train, "is_dummy", False):
        return "dummy"

    # Remote env topology: cannot detect --dummy on the robot server side.
    # Default to "real"; user can override with +profiler.mode=dummy.
    return "real"


# ---------------------------------------------------------------------------
# Profiled sync runner
# ---------------------------------------------------------------------------


def _run_sync_profiled(cfg, profiler: LatencyProfiler) -> None:
    """Run the sync EmbodiedRunner with profiling instrumentation."""
    # Reuse the staged entry point for VLM planner + simulated desktop.
    from examples.embodiment.train_embodied_agent_staged import (
        _launch_vlm_planner,
    )
    from rlinf.envs.remote.simulated_desktop import (
        launch_simulated_desktop_server,
        stop_process,
    )
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.scheduler import WorkerGroupFuncResult as Handle
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.utils.runner_utils import check_progress
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    simulated_desktop_server = launch_simulated_desktop_server(cfg)
    try:
        cluster = Cluster(cluster_cfg=cfg.cluster)
        component_placement = HybridComponentPlacement(cfg, cluster)

        vlm_actor = _launch_vlm_planner(cfg, cluster)

        actor_placement = component_placement.get_strategy("actor")
        actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
        )

        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )

        env_placement = component_placement.get_strategy("env")
        env_group = EnvWorker.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

        runner = EmbodiedRunner(
            cfg=cfg,
            actor=actor_group,
            rollout=rollout_group,
            env=env_group,
        )
        runner.init_workers()

        if vlm_actor is not None:
            env_group.set_vlm_planner(vlm_actor).wait()

        # --- Profiled training loop (mirrors EmbodiedRunner.run) ---

        env_channel = runner.env_channel
        rollout_channel = runner.rollout_channel
        actor_channel = runner.actor_channel

        start_step = runner.global_step

        for _step in range(start_step, runner.max_steps):
            actor_group.set_global_step(runner.global_step)
            rollout_group.set_global_step(runner.global_step)

            profiler.step_start(_step)

            # ① Weight sync
            profiler.start("weight_sync")
            if _step % runner.weight_sync_interval == 0:
                runner.update_rollout_weights()
            profiler.stop("weight_sync")

            # ②③④ Rollout + Env + VLM (parallel workers — measure wall time)
            profiler.start("rollout_inference")
            env_handle: Handle = env_group.interact(
                input_channel=rollout_channel,
                output_channel=env_channel,
                actor_channel=actor_channel,
            )
            rollout_handle: Handle = rollout_group.generate(
                input_channel=env_channel,
                output_channel=rollout_channel,
            )
            profiler.stop("rollout_inference")

            # ③ Env execution (wall time until trajectories arrive)
            profiler.start("env_execution")
            actor_group.recv_rollout_trajectories(input_channel=actor_channel).wait()
            rollout_handle.wait()
            profiler.stop("env_execution")

            # ⑤ Recv trajectory is included in env_execution above.
            # Record VLM reward from worker-reported durations.
            profiler.start("vlm_reward")
            profiler.stop("vlm_reward")

            profiler.start("recv_trajectory")
            profiler.stop("recv_trajectory")

            # ⑥ Advantage computation
            profiler.start("advantage_computation")
            actor_group.compute_advantages_and_returns().wait()
            profiler.stop("advantage_computation")

            # ⑦ Actor training
            profiler.start("actor_training")
            actor_training_handle: Handle = actor_group.run_training()
            actor_training_handle.wait()
            profiler.stop("actor_training")

            profiler.step_end()

            runner.global_step += 1

            # Collect worker-side durations and backfill vlm/env/rollout.
            env_time, _ = env_handle.consume_durations(return_per_rank=True)
            rollout_time, _ = rollout_handle.consume_durations(return_per_rank=True)
            backfill_worker_durations(profiler, _step, env_time, rollout_time)

            _, save_model, _ = check_progress(
                runner.global_step,
                runner.max_steps,
                cfg.runner.val_check_interval,
                cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            if save_model:
                runner._save_checkpoint()

    except KeyboardInterrupt:
        logger.info("Interrupted — saving partial latency data.")
    finally:
        stop_process(simulated_desktop_server)


# ---------------------------------------------------------------------------
# Profiled async runner
# ---------------------------------------------------------------------------


def _run_async_profiled(cfg, profiler: LatencyProfiler) -> None:
    """Run the async AsyncPPOEmbodiedRunner with profiling instrumentation."""
    from examples.embodiment.train_embodied_agent_staged import (
        _launch_vlm_planner,
    )
    from rlinf.envs.remote.simulated_desktop import (
        launch_simulated_desktop_server,
        stop_process,
    )
    from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.utils.runner_utils import check_progress
    from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )

    simulated_desktop_server = launch_simulated_desktop_server(cfg)
    try:
        cluster = Cluster(cluster_cfg=cfg.cluster)
        component_placement = HybridComponentPlacement(cfg, cluster)

        vlm_actor = _launch_vlm_planner(cfg, cluster)

        actor_placement = component_placement.get_strategy("actor")
        actor_group = AsyncPPOEmbodiedFSDPActor.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
        )

        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = AsyncMultiStepRolloutWorker.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_placement,
        )

        env_placement = component_placement.get_strategy("env")
        env_group = AsyncEnvWorker.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

        runner = AsyncPPOEmbodiedRunner(
            cfg=cfg,
            actor=actor_group,
            rollout=rollout_group,
            env=env_group,
        )
        runner.init_workers()

        if vlm_actor is not None:
            env_group.set_vlm_planner(vlm_actor).wait()

        # --- Profiled async loop (mirrors AsyncPPOEmbodiedRunner.run) ---

        actor_group.set_global_step(runner.global_step).wait()
        rollout_group.set_global_step(runner.global_step).wait()
        runner.update_rollout_weights()

        env_handle = env_group.interact(
            input_channel=runner.rollout_channel,
            output_channel=runner.env_channel,
            metric_channel=runner.env_metric_channel,
            replay_channel=runner.actor_channel,
        )
        rollout_handle = rollout_group.generate(
            input_channel=runner.env_channel,
            output_channel=runner.rollout_channel,
            metric_channel=runner.rollout_metric_channel,
        )

        recompute_logprobs = bool(cfg.rollout.get("recompute_logprobs", True))

        while runner.global_step < runner.max_steps:
            profiler.step_start(runner.global_step)

            profiler.start("recv_trajectory")
            actor_group.recv_rollout_trajectories(
                input_channel=runner.actor_channel
            ).wait()
            profiler.stop("recv_trajectory")

            if recompute_logprobs:
                profiler.start("rollout_inference")
                actor_group.compute_proximal_logprobs().wait()
                profiler.stop("rollout_inference")

            profiler.start("advantage_computation")
            actor_group.compute_advantages_and_returns().wait()
            profiler.stop("advantage_computation")

            profiler.start("actor_training")
            actor_group.run_training().wait()
            profiler.stop("actor_training")

            runner.global_step += 1
            actor_group.set_global_step(runner.global_step).wait()
            rollout_group.set_global_step(runner.global_step).wait()

            profiler.start("weight_sync")
            runner.update_rollout_weights()
            profiler.stop("weight_sync")

            profiler.step_end()

            _, save_model, _ = check_progress(
                runner.global_step,
                runner.max_steps,
                cfg.runner.val_check_interval,
                cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            if save_model:
                runner._save_checkpoint()

        env_group.stop().wait()
        rollout_group.stop().wait()
        env_handle.wait()
        rollout_handle.wait()

    except KeyboardInterrupt:
        logger.info("Interrupted — saving partial latency data.")
    finally:
        stop_process(simulated_desktop_server)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_results(profiler: LatencyProfiler, cfg) -> None:
    """Save profiling results to the output directory."""
    out_dir = os.path.join(
        cfg.runner.logger.log_path,
        cfg.runner.logger.experiment_name,
        "latency_report",
    )
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"latency_{profiler.mode}.csv")
    json_path = os.path.join(out_dir, f"latency_{profiler.mode}.json")
    report_path = os.path.join(out_dir, f"latency_{profiler.mode}.txt")

    profiler.export_csv(csv_path)
    profiler.export_json(json_path)

    with open(report_path, "w") as f:
        profiler.report(f)

    # Also print to stdout.
    profiler.report()
    logger.info(f"Latency report saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.1",
    config_path="../../examples/embodiment/config",
    config_name="yam_ppo_openpi_sync",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    mode = _detect_mode(cfg)
    logger.info(f"Latency profiler: mode={mode}")
    profiler = LatencyProfiler(mode=mode)

    is_async = is_async_runtime(cfg)

    try:
        if is_async:
            _run_async_profiled(cfg, profiler)
        else:
            _run_sync_profiled(cfg, profiler)
    finally:
        _save_results(profiler, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130) from None
