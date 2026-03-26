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

"""RLinf adapter around the vendored LeRobot PI05 runtime."""

import json
import math
import os
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.pi05.configuration_pi05 import (
    FeatureType,
    PI05Config,
    PolicyFeature,
)
from rlinf.models.embodiment.pi05.modeling_pi05 import (
    PI05Pytorch,
    _policy_no_init_context,
    _restore_missing_tied_weight_keys,
    make_att_2d_masks,
    resize_with_pad_torch,
)


def _resolve_checkpoint_dir(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    return snapshot_download(repo_id=model_path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_stats(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    tensors = load_file(str(path), device="cpu")
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in tensors.items():
        feature_name, stat_name = key.rsplit(".", 1)
        stats.setdefault(feature_name, {})[stat_name] = value.detach().cpu()
    return stats


def _normalize_model_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        updated_key = key[len("model.") :] if key.startswith("model.") else key
        normalized[updated_key] = value

    lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    embed_key = (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    )
    if lm_head_key in normalized and embed_key not in normalized:
        normalized[embed_key] = normalized[lm_head_key]
    return normalized


def _feature_map_from_config(
    raw_features: dict[str, dict[str, Any]],
) -> dict[str, PolicyFeature]:
    feature_map = {}
    for name, spec in raw_features.items():
        feature_map[name] = PolicyFeature(
            type=FeatureType(spec["type"]),
            shape=tuple(spec["shape"]),
        )
    return feature_map


def _build_config(checkpoint_dir: str) -> PI05Config:
    config_path = Path(checkpoint_dir) / "config.json"
    raw_config = _load_json(config_path)
    config_field_names = {field.name for field in fields(PI05Config)}
    config_kwargs = {k: v for k, v in raw_config.items() if k in config_field_names}
    config_kwargs["input_features"] = _feature_map_from_config(
        raw_config["input_features"]
    )
    config_kwargs["output_features"] = _feature_map_from_config(
        raw_config["output_features"]
    )
    return PI05Config(**config_kwargs)


def _normalize_quantiles(
    tensor: torch.Tensor,
    stats: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    q01 = stats["q01"].to(device=tensor.device, dtype=tensor.dtype)
    q99 = stats["q99"].to(device=tensor.device, dtype=tensor.dtype)
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.full_like(denom, eps), denom)
    return 2.0 * (tensor - q01) / denom - 1.0


def _unnormalize_quantiles(
    tensor: torch.Tensor,
    stats: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    q01 = stats["q01"].to(device=tensor.device, dtype=tensor.dtype)
    q99 = stats["q99"].to(device=tensor.device, dtype=tensor.dtype)
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.full_like(denom, eps), denom)
    return (tensor + 1.0) * denom / 2.0 + q01


def _pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return torch.nn.functional.pad(vector, (0, new_dim - vector.shape[-1]))


def _clone_env_obs(env_obs: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in env_obs.items():
        if key == "task_descriptions" or value is None:
            continue
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        elif isinstance(value, list):
            cloned[key] = [
                item.detach().clone() if torch.is_tensor(item) else item
                for item in value
            ]
        else:
            cloned[key] = value
    return cloned


def _assert_no_none_forward_inputs(forward_inputs: dict[str, Any]) -> None:
    none_keys = [key for key, value in forward_inputs.items() if value is None]
    if none_keys:
        raise ValueError(
            f"PI05 forward_inputs must not contain None values. Found keys: {none_keys}"
        )


class PI05PolicyAdapter(torch.nn.Module, BasePolicy):
    """RLinf policy adapter around the vendored LeRobot PI05 core."""

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        action_chunk: int | None = None,
        action_horizon: int | None = None,
        action_env_dim: int | None = None,
        num_steps: int | None = None,
        noise_method: str = "flow_sde",
        noise_level: float = 0.5,
        noise_params: list[float] | tuple[float, ...] = (0.16, 0.12, 200.0),
        noise_logvar_range: list[float] | tuple[float, ...] = (0.1, 1.0),
        joint_logprob: bool = False,
        safe_get_logprob: bool = False,
        add_value_head: bool = False,
        value_after_vlm: bool = False,
        value_vlm_mode: str = "mean_token",
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.config = _build_config(checkpoint_dir)
        # RLinf trains the PI05 policy under FSDP, which requires each wrapped
        # module to have a uniform parameter dtype when flattening parameters.
        # The checkpoint config may request mixed bf16/fp32 weights, so force
        # the runtime model construction to use full fp32 here.
        self.config.dtype = "float32"
        self.runtime_action_chunk = action_chunk or self.config.n_action_steps
        self.requested_action_horizon = action_horizon or self.config.chunk_size
        self.runtime_action_horizon = self.config.chunk_size
        self.runtime_action_env_dim = (
            action_env_dim or self.config.output_features["action"].shape[0]
        )
        self.runtime_num_steps = num_steps or self.config.num_inference_steps
        self.noise_method = noise_method
        self.noise_level = noise_level
        self.noise_params = tuple(noise_params)
        self.noise_logvar_range = list(noise_logvar_range)
        self.joint_logprob = joint_logprob
        self.safe_get_logprob = safe_get_logprob
        self.add_value_head = add_value_head
        self.value_after_vlm = value_after_vlm and add_value_head
        self.value_vlm_mode = value_vlm_mode
        self.preprocessor_stats = _load_stats(
            Path(checkpoint_dir)
            / "policy_preprocessor_step_2_normalizer_processor.safetensors"
        )
        self.postprocessor_stats = _load_stats(
            Path(checkpoint_dir)
            / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.image_feature_order = tuple(
            key
            for key in self.config.input_features
            if key.startswith("observation.images.")
        )

        with _policy_no_init_context():
            self.model = PI05Pytorch(self.config)
        self._load_weights()

        if self.add_value_head:
            self.value_head = ValueHead(
                input_dim=2048,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
        if self.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.max_action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.noise_logvar_range,
                noise_scheduler_type="learn",
            )

    def freeze_vlm(self) -> None:
        """Freeze the PaliGemma trunk while keeping the expert/value heads trainable."""
        self.model.paligemma_with_expert.paligemma.eval()
        for param in self.model.paligemma_with_expert.paligemma.parameters():
            param.requires_grad = False

    def _load_weights(self) -> None:
        state_dict = load_file(
            str(Path(self.checkpoint_dir) / "model.safetensors"), device="cpu"
        )
        updated_state_dict = _normalize_model_state_dict_keys(state_dict)
        updated_state_dict, _ = _restore_missing_tied_weight_keys(updated_state_dict)
        missing, unexpected = self.model.load_state_dict(
            updated_state_dict, strict=False
        )
        self.model.paligemma_with_expert.paligemma.tie_weights()
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to load PI05 checkpoint cleanly: {len(missing)} missing, {len(unexpected)} unexpected"
            )

    def _obs_to_semantic_images(
        self, env_obs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        wrist_images = env_obs["wrist_images"]
        top = env_obs["main_images"]
        left = wrist_images[:, 0]
        right = wrist_images[:, 1]
        return {
            "observation.images.left": left,
            "observation.images.right": right,
            "observation.images.top": top,
        }

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return _normalize_quantiles(state, self.preprocessor_stats["observation.state"])

    def _tokenize_task(
        self, task_descriptions: list[str], state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_state = self._normalize_state(state)
        padded_state = _pad_vector(normalized_state, self.config.max_state_dim)
        discretized = (
            np.digitize(
                padded_state.detach().cpu().numpy(),
                bins=np.linspace(-1, 1, 256 + 1)[:-1],
            )
            - 1
        )
        prompts = []
        for idx, task in enumerate(task_descriptions):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized[idx]))
            prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")
        tokenized = self.tokenizer(
            prompts,
            max_length=self.config.tokenizer_max_length,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="pt",
        )
        return tokenized["input_ids"], tokenized["attention_mask"].to(dtype=torch.bool)

    def _prepare_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        original_dtype = image.dtype
        image = image.to(dtype=torch.float32, device="cpu")
        if original_dtype == torch.uint8 or image.max() > 1.0:
            image = image / 255.0
        image = resize_with_pad_torch(image, *self.config.image_resolution)
        image = image * 2.0 - 1.0
        return image.to(device=next(self.parameters()).device, dtype=torch.float32)

    def _build_model_inputs_from_obs(
        self, env_obs: dict[str, Any]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        semantic_images = self._obs_to_semantic_images(env_obs)
        images = []
        img_masks = []
        for image_key in self.image_feature_order:
            image = self._prepare_image_tensor(semantic_images[image_key])
            images.append(image)
            img_masks.append(
                torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
            )

        tasks = env_obs["task_descriptions"]
        if isinstance(tasks, str):
            tasks = [tasks]
        tokens, masks = self._tokenize_task(tasks, env_obs["states"])
        device = next(self.parameters()).device
        return images, img_masks, tokens.to(device), masks.to(device)

    def _build_model_inputs_from_forward_inputs(
        self, forward_inputs: dict[str, Any]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        semantic_images = self._obs_to_semantic_images(forward_inputs)
        images = []
        img_masks = []
        for image_key in self.image_feature_order:
            image = self._prepare_image_tensor(semantic_images[image_key])
            images.append(image)
            img_masks.append(
                torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
            )
        device = next(self.parameters()).device
        tokens = forward_inputs["tokenized_prompt"].to(device)
        masks = forward_inputs["tokenized_prompt_mask"].to(
            device=device, dtype=torch.bool
        )
        return images, img_masks, tokens, masks

    def _prepare_attention_masks_4d(self, att_masks_2d: torch.Tensor) -> torch.Tensor:
        return self.model._prepare_attention_masks_4d(att_masks_2d)

    def _compute_prefix_cache(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_output, _), past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_output, prefix_pad_masks, past_key_values

    def _get_suffix_out(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.model.embed_suffix(x_t, timestep)
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -x_t.shape[1] :].to(dtype=torch.float32)
        return suffix_out

    def _get_value_from_vlm(self, prefix_output: torch.Tensor) -> torch.Tensor:
        if not self.add_value_head:
            return torch.zeros(prefix_output.shape[0], device=prefix_output.device)
        if self.value_vlm_mode == "last_token":
            prefix_out_value = prefix_output[:, -1]
        elif self.value_vlm_mode == "first_token":
            prefix_out_value = prefix_output[:, 0]
        else:
            prefix_out_value = prefix_output.mean(dim=1)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        return self.value_head(prefix_out_value)[:, 0]

    def _gaussian_entropy(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma_safe = torch.where(sigma <= 0, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return torch.where(sigma <= 0, torch.zeros_like(entropy), entropy)

    def _get_logprob_norm(
        self, sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        if self.safe_get_logprob:
            return -torch.pow((sample - mu), 2)
        sigma_safe = torch.where(sigma <= 0, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        return torch.where(sigma <= 0, torch.zeros_like(log_prob), log_prob)

    def _sample_noise(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        return self.model.sample_noise(shape, device)

    def _sample_mean_var_val(
        self,
        x_t: torch.Tensor,
        idx: int | torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        mode: str,
        denoise_steps: int,
        compute_values: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x_t.shape[0]
        device = x_t.device
        if isinstance(idx, int):
            idx = torch.full((batch_size,), idx, device=device, dtype=torch.long)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        suffix_out = self._get_suffix_out(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=t_input,
        )
        v_t = self.model.action_out_proj(suffix_out)

        if self.add_value_head and compute_values and not self.value_after_vlm:
            suffix_out_value = torch.mean(suffix_out, dim=1)
            value_t = self.value_head(suffix_out_value.to(dtype=torch.float32))[:, 0]
        else:
            value_t = torch.zeros((batch_size), device=device, dtype=torch.float32)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.noise_method == "flow_sde":
                noise_start, noise_end, anneal_steps = self.noise_params
                del noise_end, anneal_steps
                noise_level = noise_start
                noise_level = torch.tensor(
                    noise_level, device=device, dtype=torch.float32
                )
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)
            else:
                raise ValueError(
                    f"Unsupported PI05 PPO noise method: {self.noise_method}"
                )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        if self.value_after_vlm and compute_values:
            value_t = self._get_value_from_vlm(prefix_output)
        return x_t_mean, x_t_std, value_t

    def _get_log_prob_value(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
        chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        compute_values: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = chains.shape[0]
        prefix_output, prefix_pad_masks, past_key_values = self._compute_prefix_cache(
            images, img_masks, tokens, masks
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.joint_logprob:
            num_steps = self.runtime_num_steps
            initial_log_prob = self._get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self._gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(batch_size), denoise_ind]
            chains_next = chains[torch.arange(batch_size), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self._sample_mean_var_val(
                chains_pre,
                denoise_ind,
                prefix_output,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.runtime_num_steps,
                compute_values=compute_values,
            )
            chains_log_probs.append(
                self._get_logprob_norm(chains_next, x_t_mean, x_t_std)
            )
            chains_entropy.append(self._gaussian_entropy(x_t_std))
            if not self.value_after_vlm:
                chains_values.append(value_t)

        if self.value_after_vlm:
            chains_values.append(self._get_value_from_vlm(prefix_output))

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        if self.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        compute_values: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del kwargs
        images, img_masks, tokens, masks = self._build_model_inputs_from_obs(env_obs)
        device = tokens.device
        prefix_output, prefix_pad_masks, past_key_values = self._compute_prefix_cache(
            images, img_masks, tokens, masks
        )

        noise_shape = (
            tokens.shape[0],
            self.runtime_action_horizon,
            self.config.max_action_dim,
        )
        x_t = self._sample_noise(noise_shape, device)
        chains = [x_t]
        log_probs = []
        values = []

        if self.joint_logprob:
            log_probs.append(
                self._get_logprob_norm(x_t, torch.zeros_like(x_t), torch.ones_like(x_t))
            )
        if mode == "train":
            if self.joint_logprob:
                denoise_inds = torch.arange(self.runtime_num_steps, device=device)
            else:
                denoise_inds = torch.tensor(
                    [random.randint(0, self.runtime_num_steps - 1)]
                    * self.runtime_num_steps,
                    device=device,
                )
        else:
            denoise_inds = torch.full(
                (self.runtime_num_steps,), -1, device=device, dtype=torch.long
            )
        denoise_inds = denoise_inds[None].repeat(tokens.shape[0], 1)

        for idx in range(self.runtime_num_steps):
            sample_mode = "train" if idx == denoise_inds[0][idx] else "eval"
            x_t_mean, x_t_std, value_t = self._sample_mean_var_val(
                x_t,
                idx,
                prefix_output,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                self.runtime_num_steps,
                compute_values=compute_values,
            )
            x_t = x_t_mean + self._sample_noise(x_t.shape, device) * x_t_std
            log_probs.append(self._get_logprob_norm(x_t, x_t_mean, x_t_std))
            values.append(value_t)
            chains.append(x_t)

        chains_tensor = torch.stack(chains, dim=1)
        log_probs_tensor = torch.stack(log_probs, dim=1)[
            :, :, : self.runtime_action_chunk, : self.runtime_action_env_dim
        ]
        if self.joint_logprob:
            prev_logprobs = log_probs_tensor.mean(dim=1)
        else:
            prev_logprobs = log_probs_tensor[
                torch.arange(log_probs_tensor.shape[0], device=device),
                denoise_inds[:, 0],
            ]
        if self.value_after_vlm:
            prev_values = self._get_value_from_vlm(prefix_output)[:, None]
        else:
            prev_values = torch.stack(values, dim=1).mean(dim=-1)

        actions = x_t[:, : self.runtime_action_chunk, : self.runtime_action_env_dim].to(
            dtype=torch.float32
        )
        actions = _unnormalize_quantiles(actions, self.postprocessor_stats["action"])
        forward_inputs = {
            "chains": chains_tensor.detach(),
            "denoise_inds": denoise_inds.detach(),
            "tokenized_prompt": tokens.detach(),
            "tokenized_prompt_mask": masks.detach(),
        }
        forward_inputs.update(_clone_env_obs(env_obs))
        _assert_no_none_forward_inputs(forward_inputs)
        result = {
            "prev_logprobs": prev_logprobs.to(dtype=torch.float32).detach().cpu(),
            "prev_values": prev_values.to(dtype=torch.float32).detach().cpu(),
            "forward_inputs": {
                key: value.cpu() if torch.is_tensor(value) else value
                for key, value in forward_inputs.items()
            },
        }
        return actions.cpu(), result

    def default_forward(self, **kwargs):
        forward_inputs = kwargs["forward_inputs"]
        compute_values = kwargs.get("compute_values", False)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        images, img_masks, tokens, masks = self._build_model_inputs_from_forward_inputs(
            forward_inputs
        )
        log_probs, value_t, entropy = self._get_log_prob_value(
            images,
            img_masks,
            tokens,
            masks,
            chains.to(next(self.parameters()).device),
            denoise_inds.to(next(self.parameters()).device),
            compute_values=compute_values,
        )
        log_probs = log_probs[
            :, :, : self.runtime_action_chunk, : self.runtime_action_env_dim
        ]
        entropy = entropy[
            :, :, : self.runtime_action_chunk, : self.runtime_action_env_dim
        ]
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs.float(),
            "values": value_t.float(),
            "entropy": entropy.float(),
        }

    def forward(self, forward_type=None, **kwargs):
        del forward_type
        return self.default_forward(**kwargs)


def get_model(cfg, torch_dtype=None):
    del torch_dtype
    checkpoint_dir = _resolve_checkpoint_dir(str(cfg.model_path))
    openpi_cfg = getattr(cfg, "openpi", None)
    action_chunk = getattr(openpi_cfg, "action_chunk", None)
    action_horizon = getattr(openpi_cfg, "action_horizon", None)
    action_env_dim = getattr(openpi_cfg, "action_env_dim", None)
    num_steps = getattr(openpi_cfg, "num_steps", None)
    noise_method = getattr(openpi_cfg, "noise_method", "flow_sde")
    noise_level = getattr(openpi_cfg, "noise_level", 0.5)
    noise_params = getattr(openpi_cfg, "noise_params", (0.16, 0.12, 200.0))
    noise_logvar_range = getattr(cfg, "noise_logvar_range", (0.1, 1.0))
    joint_logprob = getattr(openpi_cfg, "joint_logprob", False)
    safe_get_logprob = getattr(openpi_cfg, "safe_get_logprob", False)
    value_after_vlm = getattr(openpi_cfg, "value_after_vlm", False)
    value_vlm_mode = getattr(openpi_cfg, "value_vlm_mode", "mean_token")
    train_expert_only = getattr(openpi_cfg, "train_expert_only", False)
    model = PI05PolicyAdapter(
        checkpoint_dir,
        action_chunk=action_chunk,
        action_horizon=action_horizon,
        action_env_dim=action_env_dim,
        num_steps=num_steps,
        noise_method=noise_method,
        noise_level=noise_level,
        noise_params=noise_params,
        noise_logvar_range=noise_logvar_range,
        joint_logprob=joint_logprob,
        safe_get_logprob=safe_get_logprob,
        add_value_head=getattr(cfg, "add_value_head", False),
        value_after_vlm=value_after_vlm,
        value_vlm_mode=value_vlm_mode,
    )
    if train_expert_only:
        model.freeze_vlm()
    return model


PI05InferenceAdapter = PI05PolicyAdapter
