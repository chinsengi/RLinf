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

from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.pi05 import inference_adapter as adapter_mod


def test_compute_flow_cps_schedule_matches_openpi_formula() -> None:
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


def test_compute_flow_cps_schedule_accepts_annealed_noise_level() -> None:
    scheduled_noise = adapter_mod._get_scheduled_noise_level(
        noise_level=0.5,
        noise_anneal=True,
        noise_params=(0.16, 0.12, 200.0),
        global_step=100,
        device=torch.device("cpu"),
    )
    t_input = torch.tensor([[[0.8]]], dtype=torch.float32)
    delta = torch.tensor([[[0.1]]], dtype=torch.float32)

    _, x1_weight, x_t_std = adapter_mod._compute_flow_cps_schedule(
        t_input=t_input,
        delta=delta,
        noise_level=scheduled_noise,
    )

    expected_noise = torch.tensor(0.14)
    expected_x1 = (t_input - delta) * torch.cos(torch.pi * expected_noise / 2)
    expected_std = (t_input - delta) * torch.sin(torch.pi * expected_noise / 2)

    torch.testing.assert_close(x1_weight, expected_x1)
    torch.testing.assert_close(x_t_std, expected_std)


def test_get_num_scored_denoise_steps_skips_deterministic_flow_cps_tail() -> None:
    assert adapter_mod._get_num_scored_denoise_steps(4, "flow_cps") == 3
    assert adapter_mod._get_num_scored_denoise_steps(4, "flow_sde") == 4


def test_get_num_scored_denoise_steps_rejects_degenerate_flow_cps() -> None:
    with pytest.raises(ValueError, match="requires num_steps > 1"):
        adapter_mod._get_num_scored_denoise_steps(1, "flow_cps")


def test_sample_mean_var_val_uses_annealed_flow_cps_noise() -> None:
    class DummyAdapter:
        def __init__(self) -> None:
            self.model = SimpleNamespace(
                action_out_proj=lambda suffix_out: torch.zeros_like(suffix_out)
            )
            self.add_value_head = False
            self.value_after_vlm = False
            self.noise_method = "flow_cps"
            self.noise_level = 0.5
            self.noise_anneal = True
            self.noise_params = (0.16, 0.12, 200.0)
            self.global_step = 100

        def _get_suffix_out(self, **kwargs) -> torch.Tensor:
            del kwargs
            return torch.zeros((2, 1, 3), dtype=torch.float32)

    adapter = DummyAdapter()
    x_t = torch.ones((2, 1, 3), dtype=torch.float32)
    prefix_output = torch.zeros((2, 1, 3), dtype=torch.float32)
    prefix_pad_masks = torch.ones((2, 1), dtype=torch.bool)

    x_t_mean, x_t_std, value_t = adapter_mod.PI05PolicyAdapter._sample_mean_var_val(
        adapter,
        x_t=x_t,
        idx=0,
        prefix_output=prefix_output,
        prefix_pad_masks=prefix_pad_masks,
        past_key_values=None,
        mode="train",
        denoise_steps=4,
        compute_values=False,
    )

    timesteps = torch.linspace(1, 1 / 4, 4, dtype=torch.float32)
    timesteps = torch.cat([timesteps, torch.tensor([0.0], dtype=torch.float32)])
    t_input = timesteps[0].expand(2)[:, None, None].expand_as(x_t)
    delta = (timesteps[0] - timesteps[1]).expand(2)[:, None, None].expand_as(x_t)
    scheduled_noise = adapter_mod._get_scheduled_noise_level(
        noise_level=adapter.noise_level,
        noise_anneal=adapter.noise_anneal,
        noise_params=adapter.noise_params,
        global_step=adapter.global_step,
        device=x_t.device,
    )
    _, _, expected_std = adapter_mod._compute_flow_cps_schedule(
        t_input=t_input,
        delta=delta,
        noise_level=scheduled_noise,
    )

    torch.testing.assert_close(x_t_std, expected_std)
    assert x_t_mean.shape == x_t.shape
    assert torch.all(torch.isfinite(x_t_mean))
    assert value_t.shape == (2,)
