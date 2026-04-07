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

"""Async staged entry point for embodied RL with a VLM planner.

This script provides the staged YAM/TOPReward runtime on the async PPO stack.
It uses the same staged orchestration as ``train_embodied_agent_staged.py``
but forces the async components:

- ``AsyncPPOEmbodiedRunner``
- ``AsyncPPOEmbodiedFSDPActor``
- ``AsyncMultiStepRolloutWorker``
- ``AsyncEnvWorker``

Use this script for staged configs with an explicit ``_async`` suffix:

- ``yam_ppo_openpi_async`` — TOPReward only (``subtask_interval: 0``)
- ``yam_ppo_openpi_subtask_async`` — TOPReward + optional subtask planning
- ``yam_ppo_openpi_desktop_async`` — desktop-topology staged variant

These configs are auto-selected here by the standard launchers whenever the
config name ends in ``_async``.
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.staged_embodied_runtime import run_with_runtime

mp.set_start_method("spawn", force=True)

_FORCED_ASYNC_RUNTIME = True


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="yam_ppo_openpi_subtask_async",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    run_with_runtime(cfg, use_async_runtime=_FORCED_ASYNC_RUNTIME)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
