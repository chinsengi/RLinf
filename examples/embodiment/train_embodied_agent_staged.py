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
  ``env.train.dense_reward_method`` is not ``"none"``,
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

The shared staged runtime lives in
``rlinf.runners.staged_embodied_runtime`` so both the sync and async
entrypoints can import the same implementation directly.

Optional remote-desktop simulation:

- Set ``env.remote_desktop_simulation.enabled: true`` when using
  ``env_type: remote`` to have this script launch a local dummy
  ``RobotServer`` automatically.
- This simulates the robot desktop input path end to end, so ``RemoteEnv``
  still talks gRPC, but no real desktop machine or SSH tunnel is required.
"""

import json
import sys

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.staged_embodied_runtime import (
    REMOTE_DISCONNECT_EVENT,
    run_with_runtime,
)

mp.set_start_method("spawn", force=True)

_FORCED_ASYNC_RUNTIME = False


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
        if REMOTE_DISCONNECT_EVENT.is_set():
            print(
                "Detected remote robot server disconnect. Training stopped.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130) from None
