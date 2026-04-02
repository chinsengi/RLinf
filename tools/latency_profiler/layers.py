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

"""Layer definitions and dependency graph for the embodied RL pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    """A single pipeline layer that contributes latency."""

    name: str
    description: str
    # Names of layers that must complete before this layer can start.
    depends_on: tuple[str, ...] = ()
    # Whether this layer is optional (e.g. xsglang).
    optional: bool = False


# ---------------------------------------------------------------------------
# Layer registry — order follows the execution flow.
# ---------------------------------------------------------------------------

WEIGHT_SYNC = Layer(
    name="weight_sync",
    description="Actor -> Rollout weight synchronisation (NCCL)",
    depends_on=(),
)

ROLLOUT_INFERENCE = Layer(
    name="rollout_inference",
    description="Policy inference to generate action chunks (HF / SGLang / vLLM)",
    depends_on=("weight_sync",),
)

ENV_EXECUTION = Layer(
    name="env_execution",
    description="Environment step via RemoteEnv gRPC (real robot or dummy)",
    depends_on=("rollout_inference",),
)

VLM_REWARD = Layer(
    name="vlm_reward",
    description="VLM TOPReward dense reward computation (Qwen3-VL)",
    depends_on=("env_execution",),
)

RECV_TRAJECTORY = Layer(
    name="recv_trajectory",
    description="Actor receives trajectory data via Channel",
    depends_on=("vlm_reward",),
)

ADVANTAGE_COMPUTATION = Layer(
    name="advantage_computation",
    description="GAE / PPO advantage and return computation",
    depends_on=("recv_trajectory",),
)

ACTOR_TRAINING = Layer(
    name="actor_training",
    description="Policy update: FSDP forward + backward + optimizer step",
    depends_on=("advantage_computation",),
)

XSGLANG_INFERENCE = Layer(
    name="xsglang_inference",
    description="xSGLang inference layer (reserved, optional)",
    depends_on=("weight_sync",),
    optional=True,
)

# Ordered list used as the default layer set.
DEFAULT_LAYERS: tuple[Layer, ...] = (
    WEIGHT_SYNC,
    ROLLOUT_INFERENCE,
    ENV_EXECUTION,
    VLM_REWARD,
    RECV_TRAJECTORY,
    ADVANTAGE_COMPUTATION,
    ACTOR_TRAINING,
)

ALL_LAYERS: tuple[Layer, ...] = DEFAULT_LAYERS + (XSGLANG_INFERENCE,)

LAYER_MAP: dict[str, Layer] = {layer.name: layer for layer in ALL_LAYERS}
