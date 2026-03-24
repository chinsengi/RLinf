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

from __future__ import annotations

import glob
from pathlib import Path

import pytest
from safetensors.torch import load_file

from rlinf.models.embodiment.pi05.adarms_vendor import enable_pi05_adarms_expert
from rlinf.models.embodiment.pi05.inference_adapter import (
    PI05InferenceAdapter,
    PI05PolicyAdapter,
    _build_config,
    _normalize_model_state_dict_keys,
)
from rlinf.models.embodiment.pi05.modeling_pi05 import (
    PI05Pytorch,
    _policy_no_init_context,
)


def _cached_pi05_snapshot() -> str | None:
    matches = glob.glob(
        "/home/prior/.cache/huggingface/hub/models--thomas0829--folding_towel_pi05/snapshots/*"
    )
    return matches[0] if matches else None


def test_vendored_transformers_modules_import_from_rlinf_namespace() -> None:
    import rlinf.vendor.transformers_pi05.models.gemma.modeling_gemma as gemma_mod
    import rlinf.vendor.transformers_pi05.models.paligemma.modeling_paligemma as paligemma_mod
    import rlinf.vendor.transformers_pi05.models.siglip.modeling_siglip as siglip_mod

    assert "rlinf/vendor/transformers_pi05" in gemma_mod.__file__
    assert "rlinf/vendor/transformers_pi05" in paligemma_mod.__file__
    assert "rlinf/vendor/transformers_pi05" in siglip_mod.__file__


def test_pi05_inference_adapter_is_kept_as_compatibility_alias() -> None:
    assert PI05InferenceAdapter is PI05PolicyAdapter


def test_vendored_pi05_core_loads_real_checkpoint_cleanly() -> None:
    snapshot = _cached_pi05_snapshot()
    if snapshot is None:
        pytest.skip("local cached PI05 checkpoint is unavailable")

    config = _build_config(snapshot)
    with _policy_no_init_context():
        model = PI05Pytorch(config)
    enable_pi05_adarms_expert(model.paligemma_with_expert)

    state_dict = load_file(str(Path(snapshot) / "model.safetensors"), device="cpu")
    normalized_state_dict = _normalize_model_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)

    assert missing == []
    assert unexpected == []
