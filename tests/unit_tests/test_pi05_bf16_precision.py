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

"""Unit tests for PI05 bf16 precision plumbing."""

import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.pi05 import inference_adapter as adapter_mod
from rlinf.workers.actor.fsdp_actor_worker import _normalize_floating_parameter_dtypes


def test_config_dtype_from_precision_maps_supported_values():
    assert adapter_mod._config_dtype_from_precision("bf16") == "bfloat16"
    assert adapter_mod._config_dtype_from_precision("bfloat16") == "bfloat16"
    assert adapter_mod._config_dtype_from_precision("float32") == "float32"
    assert adapter_mod._config_dtype_from_precision(None) is None


def test_pi05_get_model_forwards_precision_to_adapter(monkeypatch):
    captured = {}

    class DummyPolicyAdapter:
        def __init__(self, checkpoint_dir, **kwargs):
            captured["checkpoint_dir"] = checkpoint_dir
            captured.update(kwargs)

        def freeze_vlm(self):
            captured["freeze_vlm_called"] = True

    monkeypatch.setattr(adapter_mod, "PI05PolicyAdapter", DummyPolicyAdapter)
    monkeypatch.setattr(adapter_mod, "_resolve_checkpoint_dir", lambda path: path)

    cfg = OmegaConf.create(
        {
            "model_path": "unused-checkpoint",
            "precision": "bf16",
            "add_value_head": True,
            "openpi": {
                "noise_method": "flow_sde",
                "value_after_vlm": True,
                "value_vlm_mode": "mean_token",
                "train_expert_only": False,
            },
        }
    )

    adapter_mod.get_model(cfg, torch_dtype=torch.bfloat16)

    assert captured["checkpoint_dir"] == "unused-checkpoint"
    assert captured["precision"] == "bf16"
    assert captured["add_value_head"] is True


def test_normalize_floating_parameter_dtypes_only_converts_floating_params():
    class ToyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4)
            self.register_buffer("int_buffer", torch.ones(2, dtype=torch.int64))

    module = ToyModule().to(dtype=torch.float32)

    converted = _normalize_floating_parameter_dtypes(module, torch.bfloat16)

    assert sorted(converted) == ["linear.bias", "linear.weight"]
    assert module.linear.weight.dtype == torch.bfloat16
    assert module.linear.bias.dtype == torch.bfloat16
    assert module.int_buffer.dtype == torch.int64
