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

"""Regression tests for async HuggingFace rollout worker initialization."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


def _make_package_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_rollout_worker_stubs() -> dict[str, types.ModuleType | None]:
    originals: dict[str, types.ModuleType | None] = {}

    def _set_module(name: str, module: types.ModuleType) -> None:
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    scheduler_module = types.ModuleType("rlinf.scheduler")

    class Worker:
        def __init__(self) -> None:
            self._rank = 0
            self._world_size = 1
            self._group_name = "RolloutGroup"
            self.torch_platform = SimpleNamespace(current_device=lambda: "cpu")

        @staticmethod
        def timer(_name: str):
            def _decorator(func):
                return func

            return _decorator

    class Cluster:
        pass

    class CollectiveGroupOptions:
        def __init__(self, accel_max_ctas=None, accel_min_ctas=None) -> None:
            self.accel_max_ctas = accel_max_ctas
            self.accel_min_ctas = accel_min_ctas

    scheduler_module.Channel = object
    scheduler_module.Cluster = Cluster
    scheduler_module.CollectiveGroupOptions = CollectiveGroupOptions
    scheduler_module.Worker = Worker
    _set_module("rlinf.scheduler", scheduler_module)

    config_module = types.ModuleType("rlinf.config")

    class SupportedModel:
        OPENPI = "openpi"
        MLP_POLICY = "mlp_policy"
        GR00T = "gr00t"
        CNN_POLICY = "cnn_policy"
        FLOW_POLICY = "flow_policy"

    config_module.SupportedModel = SupportedModel
    _set_module("rlinf.config", config_module)

    data_package = _make_package_module("rlinf.data")
    _set_module("rlinf.data", data_package)

    embodied_io_struct_module = types.ModuleType("rlinf.data.embodied_io_struct")

    class RolloutResult:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    embodied_io_struct_module.RolloutResult = RolloutResult
    _set_module("rlinf.data.embodied_io_struct", embodied_io_struct_module)

    models_package = _make_package_module("rlinf.models")
    models_package.get_model = lambda cfg: cfg
    _set_module("rlinf.models", models_package)

    embodiment_package = _make_package_module("rlinf.models.embodiment")
    _set_module("rlinf.models.embodiment", embodiment_package)

    base_policy_module = types.ModuleType("rlinf.models.embodiment.base_policy")

    class BasePolicy:
        pass

    base_policy_module.BasePolicy = BasePolicy
    _set_module("rlinf.models.embodiment.base_policy", base_policy_module)

    utils_package = _make_package_module("rlinf.utils")
    _set_module("rlinf.utils", utils_package)

    comm_mapping_module = types.ModuleType("rlinf.utils.comm_mapping")

    class CommMapper:
        @staticmethod
        def get_dst_ranks(**kwargs):
            return [(0, kwargs["batch_size"])]

        @staticmethod
        def get_src_ranks(**kwargs):
            return [(0, kwargs["batch_size"])]

        @staticmethod
        def build_channel_key(*args, **kwargs) -> str:
            del args, kwargs
            return "channel-key"

    comm_mapping_module.CommMapper = CommMapper
    _set_module("rlinf.utils.comm_mapping", comm_mapping_module)

    placement_module = types.ModuleType("rlinf.utils.placement")

    class HybridComponentPlacement:
        def __init__(self, cfg, cluster) -> None:
            del cfg, cluster

        def get_world_size(self, component: str) -> int:
            del component
            return 1

    placement_module.HybridComponentPlacement = HybridComponentPlacement
    _set_module("rlinf.utils.placement", placement_module)

    return originals


def _restore_modules(originals: dict[str, types.ModuleType | None]) -> None:
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def test_async_rollout_worker_initializes_pipeline_slot_aliases() -> None:
    """Async rollout init should honor pipeline_slot_count without missing attrs."""
    repo_root = Path(__file__).resolve().parents[2]
    originals = _install_rollout_worker_stubs()
    loaded_modules = [
        "rlinf.workers.rollout.hf.huggingface_worker",
        "rlinf.workers.rollout.hf.async_huggingface_worker",
    ]

    try:
        _load_module(
            "rlinf.workers.rollout.hf.huggingface_worker",
            repo_root / "rlinf/workers/rollout/hf/huggingface_worker.py",
        )
        async_module = _load_module(
            "rlinf.workers.rollout.hf.async_huggingface_worker",
            repo_root / "rlinf/workers/rollout/hf/async_huggingface_worker.py",
        )

        cfg = OmegaConf.create(
            {
                "actor": {
                    "group_name": "ActorGroup",
                    "model": {"num_action_chunks": 10},
                },
                "rollout": {
                    "pipeline_slot_count": 2,
                    "enable_offload": False,
                },
                "algorithm": {
                    "rollout_epoch": 1,
                    "staleness_threshold": 1,
                },
                "runner": {
                    "val_check_interval": -1,
                },
                "env": {
                    "train": {
                        "total_num_envs": 8,
                        "max_steps_per_rollout_epoch": 20,
                    }
                },
            }
        )

        worker = async_module.AsyncMultiStepRolloutWorker(cfg)

        assert worker.num_pipeline_slots == 2
        assert worker.num_pipeline_stages == 2
        assert worker.slot_count == 2
        assert worker.train_batch_size == 4
        assert worker.num_envs_per_stage == 4
    finally:
        for module_name in loaded_modules:
            sys.modules.pop(module_name, None)
        _restore_modules(originals)
