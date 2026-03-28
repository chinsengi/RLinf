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

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "embodiment"
    / "train_embodied_agent_staged.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "train_embodied_agent_staged", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_use_async_embodied_runtime = _MODULE._use_async_embodied_runtime


def test_decoupled_actor_critic_uses_async_staged_runtime() -> None:
    cfg = OmegaConf.create({"algorithm": {"loss_type": "decoupled_actor_critic"}})
    assert _use_async_embodied_runtime(cfg) is True


def test_actor_critic_uses_sync_staged_runtime() -> None:
    cfg = OmegaConf.create({"algorithm": {"loss_type": "actor_critic"}})
    assert _use_async_embodied_runtime(cfg) is False
