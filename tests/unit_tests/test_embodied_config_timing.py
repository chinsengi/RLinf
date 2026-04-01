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

from types import SimpleNamespace

from omegaconf import OmegaConf

import rlinf.config as config_mod
from rlinf.config import validate_embodied_cfg


def test_return_home_minutes_disables_rollout_boundary_resets(monkeypatch) -> None:
    monkeypatch.setattr(config_mod, "Cluster", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        config_mod,
        "HybridComponentPlacement",
        lambda *args, **kwargs: SimpleNamespace(get_world_size=lambda component: 1),
    )

    cfg = OmegaConf.create(
        {
            "env": {
                "return_home_minutes": 2,
                "server_cooldown_minutes": 1,
                "train": {
                    "control_rate_hz": 10.0,
                    "reset_on_rollout_epoch": True,
                    "reset_on_rollout_epoch_end": True,
                    "total_num_envs": 1,
                    "group_size": 1,
                    "env_type": "remote",
                },
                "eval": {
                    "control_rate_hz": 10.0,
                    "total_num_envs": 1,
                    "group_size": 1,
                    "env_type": "remote",
                },
            },
            "algorithm": {
                "loss_type": "ppo",
                "adv_type": "gae",
                "rollout_epoch": 1,
            },
            "runner": {
                "val_check_interval": 0,
                "only_eval": False,
            },
            "actor": {
                "global_batch_size": 1,
                "model": {
                    "model_type": "openpi",
                    "num_action_chunks": 30,
                },
            },
            "rollout": {
                "collect_prev_infos": True,
                "pipeline_stage_num": 1,
            },
        }
    )

    validated = validate_embodied_cfg(cfg)

    assert validated.env.server_episode_duration_s == 120.0
    assert validated.env.server_cooldown_s == 60.0
    assert validated.env.train.max_episode_steps == 1200
    assert validated.env.train.max_steps_per_rollout_epoch == 1200
    assert validated.env.eval.max_episode_steps == 1200
    assert validated.env.eval.max_steps_per_rollout_epoch == 1200
    assert validated.env.train.reset_on_rollout_epoch is False
    assert validated.env.train.reset_on_rollout_epoch_end is False
