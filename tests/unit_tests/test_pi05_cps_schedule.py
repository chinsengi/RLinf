# Copyright 2026 Shirui Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for PI05 constant-preserving schedule support."""

import pytest
import torch

from rlinf.models.embodiment.pi05 import inference_adapter as adapter_mod


def test_compute_flow_cps_schedule_matches_openpi_formula():
    t_input = torch.tensor([[[0.8]], [[0.6]]], dtype=torch.float32)
    delta = torch.tensor([[[0.1]], [[0.2]]], dtype=torch.float32)
    noise_level = 0.5

    x0_weight, x1_weight, x_t_std = adapter_mod._compute_flow_cps_schedule(
        t_input=t_input,
        delta=delta,
        noise_level=noise_level,
    )

    cos_term = torch.cos(torch.pi * torch.tensor(noise_level) / 2)
    sin_term = torch.sin(torch.pi * torch.tensor(noise_level) / 2)
    expected_x0 = 1 - (t_input - delta)
    expected_x1 = (t_input - delta) * cos_term
    expected_std = (t_input - delta) * sin_term

    torch.testing.assert_close(x0_weight, expected_x0)
    torch.testing.assert_close(x1_weight, expected_x1)
    torch.testing.assert_close(x_t_std, expected_std)


def test_get_num_scored_denoise_steps_skips_deterministic_flow_cps_tail() -> None:
    assert adapter_mod._get_num_scored_denoise_steps(4, "flow_cps") == 3
    assert adapter_mod._get_num_scored_denoise_steps(4, "flow_sde") == 4


def test_get_num_scored_denoise_steps_rejects_degenerate_flow_cps() -> None:
    with pytest.raises(ValueError, match="requires num_steps > 1"):
        adapter_mod._get_num_scored_denoise_steps(1, "flow_cps")
