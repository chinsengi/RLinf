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

"""Unit tests for PI05 routing and legacy OpenPI fail-closed behavior."""

import sys
from types import ModuleType

import pytest
from omegaconf import OmegaConf

import rlinf.models as models_mod
from rlinf.models.embodiment.openpi import get_model as get_legacy_openpi_model


def _fake_module(return_value):
    module = ModuleType("fake_module")

    def _get_model(cfg, torch_dtype=None):
        del cfg, torch_dtype
        return return_value

    module.get_model = _get_model
    return module


def test_top_level_model_factory_routes_pi05_openpi_to_vendored(monkeypatch):
    vendored_model = object()
    legacy_model = object()
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.pi05",
        _fake_module(vendored_model),
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.openpi",
        _fake_module(legacy_model),
    )
    monkeypatch.setattr(
        models_mod.Worker.torch_platform,
        "is_available",
        lambda: False,
    )

    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "precision": "bf16",
            "is_lora": False,
            "openpi": {"config_name": "pi05_yam_follower"},
        }
    )

    assert models_mod.get_model(cfg) is vendored_model


def test_top_level_model_factory_keeps_non_pi05_openpi_on_legacy_loader(monkeypatch):
    vendored_model = object()
    legacy_model = object()
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.pi05",
        _fake_module(vendored_model),
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.openpi",
        _fake_module(legacy_model),
    )
    monkeypatch.setattr(
        models_mod.Worker.torch_platform,
        "is_available",
        lambda: False,
    )

    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "precision": "bf16",
            "is_lora": False,
            "openpi": {"config_name": "pi0_libero"},
        }
    )

    assert models_mod.get_model(cfg) is legacy_model


def test_legacy_openpi_loader_raises_for_pi05():
    cfg = OmegaConf.create(
        {
            "model_path": "unused",
            "model_type": "openpi",
            "openpi": {"config_name": "pi05_yam_follower"},
        }
    )

    with pytest.raises(RuntimeError, match="Legacy OpenPI PI05 is disabled"):
        get_legacy_openpi_model(cfg)
