# Copyright 2025 The RLinf Authors.
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


def get_num_scored_denoise_steps(num_steps: int, noise_method: str) -> int:
    """Return how many denoising transitions should contribute PPO log-probs."""
    if noise_method == "flow_cps":
        if num_steps <= 1:
            raise ValueError(
                "noise_method='flow_cps' requires num_steps > 1 because "
                "its last denoising step is deterministic."
            )
        return num_steps - 1
    return num_steps


def get_num_single_step_candidates(
    num_steps: int,
    noise_method: str,
    *,
    ignore_last: bool,
) -> int:
    """Return how many denoising steps can be sampled in single-step PPO mode."""
    candidates = get_num_scored_denoise_steps(num_steps, noise_method)
    if ignore_last:
        candidates -= 1
    if candidates <= 0:
        raise ValueError(
            "No stochastic denoising steps are available for PPO. "
            f"Got {num_steps=}, {noise_method=}, {ignore_last=}."
        )
    return candidates
