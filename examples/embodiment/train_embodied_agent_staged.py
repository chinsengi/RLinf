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

"""Sync staged entry point for embodied RL with a VLM planner.

This script extends ``train_embodied_agent.py`` with staged-runtime features
that are needed by VLM-planner workflows:

- launch ``VLMPlannerWorker`` when ``env.train.subtask_interval > 0`` or
  ``env.train.top_reward_enabled`` is set,
- support the remote-desktop simulation path for ``env_type: remote``,
- keep the staged remote-disconnect handling used by the YAM/remote stack.

The command-line entrypoint in this file is sync-only. It is intended for
staged configs that should run with ``EmbodiedRunner`` +
``MultiStepRolloutWorker`` + ``EnvWorker``.

Use this script for staged configs with an explicit ``_sync`` suffix:

- ``yam_ppo_openpi_sync`` — TOPReward only (``subtask_interval: 0``)
- ``yam_ppo_openpi_subtask_sync`` — TOPReward + optional subtask planning
- ``yam_ppo_openpi_desktop_sync`` — desktop-topology staged variant

Usage::

    # Start Ray first (see CLAUDE.md: Multi-Node Setup)
    bash examples/embodiment/run_embodiment.sh <staged_sync_config>

Or directly::

    python examples/embodiment/train_embodied_agent_staged.py \
        --config-path examples/embodiment/config/ \
        --config-name <staged_sync_config>

Launch-script routing:

- ``run_embodiment.sh`` / ``run_realworld.sh`` / ``join_beaker_cluster.sh``
  route config names ending in ``_sync`` here.
- Matching ``*_async`` staged configs use
  ``train_embodied_agent_staged_async.py`` instead.

The config must contain a ``vlm_planner`` section and a node group labelled
``"beaker_vlm"`` in ``cluster.node_groups``. ``VLMPlannerWorker`` is allocated
through RLinf's placement stack so it inherits the same GPU isolation model as
actor and rollout workers instead of relying on a standalone Ray ``num_gpus``
reservation.

This file also exposes the shared ``run_with_runtime(...)`` helper used by
``train_embodied_agent_staged_async.py``. The helper remains
runtime-parameterized so the async wrapper can reuse the same staged
setup/teardown logic without duplicating the implementation.

Optional remote-desktop simulation:

- Set ``env.remote_desktop_simulation.enabled: true`` when using
  ``env_type: remote`` to have this script launch a local dummy
  ``RobotServer`` automatically.
- This simulates the robot desktop input path end to end, so ``RemoteEnv``
  still talks gRPC, but no real desktop machine or SSH tunnel is required.
"""

import json
import signal
import socket
import sys
import threading

import grpc
import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.envs.remote.proto import robot_env_pb2, robot_env_pb2_grpc
from rlinf.envs.remote.simulated_desktop import (
    launch_simulated_desktop_server,
    stop_process,
)
from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import AcceleratorUtil, Cluster, PackedPlacementStrategy
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.workers.vlm_planner import VLMPlannerWorker

mp.set_start_method("spawn", force=True)

_FORCED_ASYNC_RUNTIME = False

_VLM_PLANNER_NODE_GROUP = "beaker_vlm"
_REMOTE_MONITOR_POLL_S = 1.0
_REMOTE_MONITOR_CONNECT_TIMEOUT_S = 0.5
_REMOTE_MONITOR_FAILURE_THRESHOLD = 3
_REMOTE_SAFE_RECOVERY_TIMEOUT_S = 5.0
_REMOTE_DISCONNECT_EVENT = threading.Event()
_REMOTE_DISCONNECT_MARKERS = (
    "RobotServerDisconnectedError",
    "[RemoteEnv] Robot server disconnected during",
    "failed to connect to all addresses",
    "grpc unavailable",
)


def _parse_host_port(server_url: str) -> tuple[str, int]:
    host, port_str = str(server_url).rsplit(":", 1)
    return host, int(port_str)


def _start_remote_disconnect_monitor(cfg, on_disconnect=None):
    """Stop Beaker training when the desktop RobotServer disappears."""
    if str(cfg.env.train.get("env_type", "")).lower() != "remote":
        return None, None

    server_url = str(cfg.env.train.get("remote_server_url", "localhost:50051"))
    host, port = _parse_host_port(server_url)
    stop_event = threading.Event()

    def _monitor() -> None:
        consecutive_failures = 0
        while not stop_event.wait(_REMOTE_MONITOR_POLL_S):
            try:
                with socket.create_connection(
                    (host, port), timeout=_REMOTE_MONITOR_CONNECT_TIMEOUT_S
                ):
                    consecutive_failures = 0
            except OSError:
                consecutive_failures += 1
                if consecutive_failures < _REMOTE_MONITOR_FAILURE_THRESHOLD:
                    continue
                if _REMOTE_DISCONNECT_EVENT.is_set():
                    return
                _REMOTE_DISCONNECT_EVENT.set()
                print(
                    "[train_embodied_agent_staged] Detected remote robot server "
                    f"disconnect at {server_url}. Stopping training.",
                    file=sys.stderr,
                    flush=True,
                )
                if on_disconnect is not None:
                    try:
                        on_disconnect()
                    except Exception:
                        pass
                return

    thread = threading.Thread(
        target=_monitor,
        name="remote-robot-server-monitor",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _request_remote_safe_recovery(cfg) -> None:
    """Ask the desktop RobotServer to enter its safe idle recovery path."""
    if str(cfg.env.train.get("env_type", "")).lower() != "remote":
        return
    if _REMOTE_DISCONNECT_EVENT.is_set():
        return

    server_url = str(cfg.env.train.get("remote_server_url", "localhost:50051"))
    max_msg = int(cfg.env.train.get("grpc_max_message_size", 64 * 1024 * 1024))
    grpc_timeout = min(
        float(cfg.env.train.get("grpc_timeout", _REMOTE_SAFE_RECOVERY_TIMEOUT_S)),
        _REMOTE_SAFE_RECOVERY_TIMEOUT_S,
    )
    channel = grpc.insecure_channel(
        server_url,
        options=[
            ("grpc.max_send_message_length", max_msg),
            ("grpc.max_receive_message_length", max_msg),
        ],
    )
    try:
        stub = robot_env_pb2_grpc.RobotEnvServiceStub(channel)
        stub.Close(robot_env_pb2.Empty(), timeout=grpc_timeout)
        print(
            "[train_embodied_agent_staged] Requested remote robot safe recovery "
            f"at {server_url}.",
            file=sys.stderr,
            flush=True,
        )
    except grpc.RpcError as error:
        print(
            "[train_embodied_agent_staged] Failed to request remote robot safe "
            f"recovery at {server_url}: {error}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        channel.close()


def _shutdown_worker_group_fast(worker_group) -> None:
    """Terminate a worker group without waiting for graceful env cleanup."""
    if worker_group is None:
        return
    try:
        worker_group._close()
    except Exception:
        pass


def _suppress_worker_failure_signal() -> None:
    """Ignore SIGUSR1 during intentional fast shutdown.

    WorkerGroup wait threads send SIGUSR1 when a Ray worker disappears while a
    background ``ray.get`` is still pending. That is useful for real failures,
    but during an intentional Ctrl+C / remote-disconnect fast shutdown it turns
    an expected teardown into a noisy "worker execution failure" exit path.
    """
    try:
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    except Exception:
        pass


def _is_expected_remote_disconnect_error(error: BaseException | None) -> bool:
    """Return whether an exception is the expected remote robot disconnect path."""
    seen: set[int] = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(marker in str(current) for marker in _REMOTE_DISCONNECT_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _compute_vlm_gpu_index(cfg) -> int:
    """Return the GPU index to use for VLMPlannerWorker.

    Configs may set ``vlm_planner.placement`` to override the default. When no
    override is provided, the default remains the original heuristic:

    - if actor/rollout use fewer than two distinct GPU indices on the VLM node,
      place the VLM on GPU 0
    - otherwise place it on ``max(actor, rollout, env placements on that node)+1``

    The important difference is that the returned index is now fed into RLinf's
    placement system, which reserves and isolates the selected GPU in the same
    way as actor and rollout workers.
    """
    # Explicit placement override takes precedence over the heuristic.
    vlm_cfg = getattr(cfg, "vlm_planner", None)
    if vlm_cfg is not None:
        explicit = getattr(vlm_cfg, "placement", None)
        if explicit is not None:
            return int(explicit)

    # Find node_ranks of the beaker_vlm group.
    vlm_node_ranks: set[int] = set()
    for g in cfg.cluster.node_groups:
        if g.label == _VLM_PLANNER_NODE_GROUP:
            nr = g.node_ranks
            if isinstance(nr, int):
                vlm_node_ranks.add(nr)
            else:
                for r in str(nr).split(","):
                    vlm_node_ranks.add(int(r.strip()))
            break

    # Build label → node_ranks map for all groups.
    group_ranks: dict[str, set[int]] = {}
    for g in cfg.cluster.node_groups:
        nr = g.node_ranks
        if isinstance(nr, int):
            ranks: set[int] = {nr}
        else:
            ranks = {int(r.strip()) for r in str(nr).split(",")}
        group_ranks[g.label] = ranks

    # Collect distinct GPU placement indices used by other components on the
    # same node as beaker_vlm.
    placements_on_shared_node: set[int] = set()
    for comp_name in ("actor", "rollout", "env"):
        comp = getattr(cfg.cluster.component_placement, comp_name, None)
        if comp is None:
            continue
        comp_group_ranks = group_ranks.get(getattr(comp, "node_group", ""), set())
        if not (comp_group_ranks & vlm_node_ranks):
            continue  # Component is on a different physical node — no conflict.
        placement_val = str(getattr(comp, "placement", 0))
        # Handle range syntax "0-2" — use the high watermark.
        high = (
            int(placement_val.split("-")[-1])
            if "-" in placement_val
            else int(placement_val)
        )
        placements_on_shared_node.add(high)

    # Only one distinct GPU index in use (or dedicated node): VLM shares GPU 0.
    # Two or more distinct indices: every index is occupied — VLM needs max+1.
    if len(placements_on_shared_node) < 2:
        return 0

    return max(placements_on_shared_node) + 1


def _get_vlm_planner_placement(cfg) -> tuple[str, int]:
    """Resolve the node group label and GPU index for the VLM planner."""
    vlm_cfg = getattr(cfg, "vlm_planner", None)
    node_group = str(getattr(vlm_cfg, "node_group", _VLM_PLANNER_NODE_GROUP))
    gpu_index = _compute_vlm_gpu_index(cfg)
    return node_group, gpu_index


def _launch_vlm_planner(cfg, cluster: Cluster):
    """Create a placement-backed VLMPlannerWorker Ray actor.

    Args:
        cfg: Top-level Hydra config.
        cluster: Initialised Cluster object.

    Returns:
        Ray actor handle for VLMPlannerWorker, or None if the ``vlm_planner``
        config section is absent and neither ``env.train.subtask_interval > 0``
        nor ``env.train.top_reward_enabled`` is set.
    """
    subtask_interval = cfg.env.train.get("subtask_interval", 0)
    top_reward_enabled = cfg.env.train.get("top_reward_enabled", False)
    if subtask_interval <= 0 and not top_reward_enabled:
        return None

    if not hasattr(cfg, "vlm_planner"):
        return None

    node_group_label, vlm_gpu = _get_vlm_planner_placement(cfg)
    node_group = cluster.get_node_group(node_group_label)
    if node_group is None or not node_group.nodes:
        raise RuntimeError(
            "VLMPlannerWorker requires a node group labelled "
            f"'{node_group_label}' in cluster.node_groups. Check your YAML config."
        )

    placement_strategy = PackedPlacementStrategy(
        start_hardware_rank=vlm_gpu,
        end_hardware_rank=vlm_gpu,
        node_group=node_group_label,
    )
    placements = placement_strategy.get_placement(cluster, isolate_accelerator=True)
    if len(placements) != 1:
        raise RuntimeError(
            "Expected exactly one placement for VLMPlannerWorker, got "
            f"{len(placements)}."
        )

    placement = placements[0]
    node = cluster.get_node_info(placement.cluster_node_rank)
    env_vars = {
        "VISIBLE_DEVICES": ",".join(placement.visible_accelerators),
        "ACCELERATOR_TYPE": str(node.accelerator_type),
        "ACCELERATOR_MODEL": node.accelerator_model,
        "ISOLATE_ACCELERATOR": "1" if placement.isolate_accelerator else "0",
        "LOCAL_ACCELERATOR_RANK": str(placement.local_accelerator_rank),
        "LOCAL_HARDWARE_RANKS": ",".join(map(str, placement.local_hardware_ranks)),
        "NODE_GROUP_LABEL": placement.node_group_label,
        "NODE_RANK": str(placement.placement_node_rank),
        "CLUSTER_NODE_RANK": str(placement.cluster_node_rank),
        "NODE_LOCAL_RANK": str(placement.local_rank),
        "NODE_LOCAL_WORLD_SIZE": str(placement.local_world_size),
        "RAY_ACTOR": str(1),
    }
    env_vars.update(
        AcceleratorUtil.get_accelerator_env_var(
            node.accelerator_type, placement.visible_accelerators
        )
    )

    worker_name = f"{cfg.vlm_planner.get('group_name', 'VLMPlannerWorker')}_0"
    vlm_actor = cluster.allocate(
        cls=VLMPlannerWorker,
        worker_name=worker_name,
        worker_rank=0,
        node_rank=placement.cluster_node_rank,
        max_concurrency=1,
        env_vars=env_vars,
        node_group_label=placement.node_group_label,
        disable_distributed_log=False,
        cls_args=(cfg,),
        cls_kwargs={},
    )

    return vlm_actor


def run_with_runtime(cfg, *, use_async_runtime: bool) -> None:
    """Run staged embodied training with an explicit runtime selection."""
    simulated_desktop_server = launch_simulated_desktop_server(cfg)
    cluster = None
    actor_group = None
    rollout_group = None
    env_group = None
    vlm_actor = None
    workers_initialized = False
    remote_monitor_stop = None
    remote_monitor_thread = None
    fast_shutdown = False
    remote_disconnect_exit = False
    try:
        cluster = Cluster(cluster_cfg=cfg.cluster)
        component_placement = HybridComponentPlacement(cfg, cluster)
        # Load the VLM planner before worker init so model startup completes
        # before the desktop-side robot session begins moving.
        vlm_actor = _launch_vlm_planner(cfg, cluster)

        # Create actor worker group (FSDP training on Beaker).
        actor_placement = component_placement.get_strategy("actor")
        actor_worker_cls = (
            AsyncPPOEmbodiedFSDPActor if use_async_runtime else EmbodiedFSDPActor
        )
        actor_group = actor_worker_cls.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
        )

        # Create rollout worker group (inference).
        rollout_placement = component_placement.get_strategy("rollout")
        rollout_worker_cls = (
            AsyncMultiStepRolloutWorker if use_async_runtime else MultiStepRolloutWorker
        )
        rollout_group = rollout_worker_cls.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )

        # Create env worker group (direct YAMEnv or RemoteEnv per config).
        env_placement = component_placement.get_strategy("env")
        env_worker_cls = AsyncEnvWorker if use_async_runtime else EnvWorker
        env_group = env_worker_cls.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

        runner_cls = AsyncPPOEmbodiedRunner if use_async_runtime else EmbodiedRunner
        runner = runner_cls(
            cfg=cfg,
            actor=actor_group,
            rollout=rollout_group,
            env=env_group,
        )

        def _handle_remote_disconnect() -> None:
            _suppress_worker_failure_signal()
            _shutdown_worker_group_fast(env_group)
            _shutdown_worker_group_fast(rollout_group)
            _shutdown_worker_group_fast(actor_group)

        runner.init_workers()
        workers_initialized = True
        remote_monitor_stop, remote_monitor_thread = _start_remote_disconnect_monitor(
            cfg, on_disconnect=_handle_remote_disconnect
        )

        # Wire the VLM planner into env workers after they have initialised.
        if vlm_actor is not None:
            env_group.set_vlm_planner(vlm_actor).wait()

        runner.run()
    except KeyboardInterrupt:
        fast_shutdown = True
        raise
    except Exception as error:
        if _REMOTE_DISCONNECT_EVENT.is_set() or _is_expected_remote_disconnect_error(
            error
        ):
            fast_shutdown = True
            remote_disconnect_exit = True
            _REMOTE_DISCONNECT_EVENT.set()
            print(
                "Detected remote robot server disconnect. Training stopped.",
                file=sys.stderr,
                flush=True,
            )
        else:
            raise
    finally:
        if remote_monitor_stop is not None:
            remote_monitor_stop.set()
        if remote_monitor_thread is not None:
            remote_monitor_thread.join(timeout=1.0)
        fast_shutdown = fast_shutdown or _REMOTE_DISCONNECT_EVENT.is_set()
        _request_remote_safe_recovery(cfg)
        if workers_initialized and fast_shutdown:
            _suppress_worker_failure_signal()
            _shutdown_worker_group_fast(env_group)
            _shutdown_worker_group_fast(rollout_group)
            _shutdown_worker_group_fast(actor_group)
        else:
            if workers_initialized and env_group is not None:
                try:
                    env_group.close_envs().wait()
                except Exception:
                    pass
            if workers_initialized and rollout_group is not None:
                try:
                    rollout_group._close()
                except Exception:
                    pass
            if workers_initialized and actor_group is not None:
                try:
                    actor_group._close()
                except Exception:
                    pass
        stop_process(simulated_desktop_server)
    if remote_disconnect_exit:
        raise SystemExit(1)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="yam_ppo_openpi_subtask_sync",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    run_with_runtime(cfg, use_async_runtime=_FORCED_ASYNC_RUNTIME)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if _REMOTE_DISCONNECT_EVENT.is_set():
            print(
                "Detected remote robot server disconnect. Training stopped.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130) from None
