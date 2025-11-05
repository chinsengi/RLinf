#!/usr/bin/env python

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

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Union, get_args, get_origin

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.envs.configs import (
    GripperConfig,
    HILSerlProcessorConfig,
    HILSerlRobotEnvConfig,
    ImagePreprocessingConfig,
    InverseKinematicsConfig,
    ObservationConfig,
    ResetConfig,
    RewardClassifierConfig,
)
from lerobot.processor import TransitionKey, create_transition
from lerobot.rl.gym_manipulator import (
    RobotEnv,
    make_processors,
    step_env_and_process_transition,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from rlinf.envs.utils import to_tensor

CameraConfigType = Union[OpenCVCameraConfig, RealSenseCameraConfig]

CAMERA_CONFIG_BUILDERS: dict[str, Callable[..., CameraConfigType]] = {
    "opencv": OpenCVCameraConfig,
    "intelrealsense": RealSenseCameraConfig,
}


@dataclass(kw_only=True)
class NullTeleopConfig(TeleoperatorConfig):
    """Placeholder teleoperator config used for policy-only control."""


class NullTeleop(Teleoperator):
    """Minimal teleoperator stub that always reports no intervention."""

    config_class = NullTeleopConfig
    name = "null_teleop"

    def __init__(self, motor_names: Iterable[str]):
        super().__init__(NullTeleopConfig(id="null_teleop"))
        self._connected = False
        self._motor_names = list(motor_names)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self._motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def get_action(self) -> None:
        return None

    def send_feedback(self, feedback: dict[str, Any]) -> None:  # noqa: ARG002
        return

    def disconnect(self) -> None:
        self._connected = False

    def get_teleop_events(self) -> dict[str, bool]:
        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }


def _instantiate_dataclass(cls, data):
    if data is None:
        return None
    if dataclasses.is_dataclass(data):
        return data
    if not isinstance(data, Mapping):
        return data

    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        kwargs[field.name] = _coerce_field(field.type, value)
    return cls(**kwargs)


def _coerce_field(field_type, value):
    origin = get_origin(field_type)
    if origin is None:
        if dataclasses.is_dataclass(field_type):
            return _instantiate_dataclass(field_type, value)
        return value

    if origin is Union:
        if value is None:
            return None
        for arg in get_args(field_type):
            if arg is type(None):
                continue
            try:
                return _coerce_field(arg, value)
            except Exception:
                continue
        return value

    if origin is dict:
        key_type, item_type = get_args(field_type)
        if dataclasses.is_dataclass(item_type):
            return {
                _coerce_field(key_type, k): _instantiate_dataclass(item_type, v)
                for k, v in value.items()
            }
        return value

    if origin in (list, tuple):
        (item_type,) = get_args(field_type)
        return type(value)(_coerce_field(item_type, v) for v in value)

    return value


def _build_processor_config(cfg: Mapping[str, Any]) -> HILSerlProcessorConfig:
    processor_dict = dict(cfg)
    if "observation" in processor_dict:
        processor_dict["observation"] = _instantiate_dataclass(
            ObservationConfig, processor_dict["observation"]
        )
    if "image_preprocessing" in processor_dict:
        processor_dict["image_preprocessing"] = _instantiate_dataclass(
            ImagePreprocessingConfig, processor_dict["image_preprocessing"]
        )
    if "gripper" in processor_dict:
        processor_dict["gripper"] = _instantiate_dataclass(
            GripperConfig, processor_dict["gripper"]
        )
    if "reset" in processor_dict:
        processor_dict["reset"] = _instantiate_dataclass(
            ResetConfig, processor_dict["reset"]
        )
    if "inverse_kinematics" in processor_dict:
        processor_dict["inverse_kinematics"] = _instantiate_dataclass(
            InverseKinematicsConfig, processor_dict["inverse_kinematics"]
        )
    if "reward_classifier" in processor_dict:
        processor_dict["reward_classifier"] = _instantiate_dataclass(
            RewardClassifierConfig, processor_dict["reward_classifier"]
        )
    return HILSerlProcessorConfig(**processor_dict)


def _build_camera_configs(camera_cfg: Mapping[str, Any]) -> dict[str, CameraConfigType]:
    cameras: dict[str, CameraConfigType] = {}
    for name, cfg in camera_cfg.items():
        cfg_dict = dict(cfg)
        camera_type = cfg_dict.pop("type", None)
        if camera_type is None:
            raise ValueError(f"Camera '{name}' must define a 'type' field.")
        if camera_type not in CAMERA_CONFIG_BUILDERS:
            raise ValueError(f"Unsupported camera type '{camera_type}' for '{name}'.")
        builder = CAMERA_CONFIG_BUILDERS[camera_type]
        cameras[name] = builder(**cfg_dict)
    return cameras


class LeRobotSo100Env(gym.Env):
    """RLinf-compatible wrapper around LeRobot's SO100 follower robot."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: DictConfig,
        seed_offset: int,
        total_num_processes: int,
        record_metrics: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.record_metrics = record_metrics

        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.use_rel_reward = cfg.use_rel_reward
        self.reward_coef = cfg.reward_coef
        self.max_episode_steps = cfg.max_episode_steps

        lerobot_cfg = OmegaConf.to_container(cfg.lerobot, resolve=True) if cfg.lerobot else {}
        if lerobot_cfg is None:
            raise ValueError("`cfg.lerobot` must be provided for the LeRobot integration.")

        self.device = torch.device(
            lerobot_cfg.get("policy_device", lerobot_cfg.get("device", "cpu"))
        )
        self.env_device = torch.device(lerobot_cfg.get("device", "cpu"))

        self.task_prompt = lerobot_cfg.get("task_prompt", "")
        self.primary_camera_key = lerobot_cfg.get("primary_camera_key")
        self.wrist_camera_keys = lerobot_cfg.get("wrist_camera_keys", [])
        self.state_key = lerobot_cfg.get("state_key", "observation.state")

        self._init_robot_and_env(lerobot_cfg)
        self._init_processors(lerobot_cfg)

        self.num_envs = 1
        self._is_start = True
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32)

        if self.record_metrics:
            self._init_metrics()

    def _init_robot_and_env(self, lerobot_cfg: Mapping[str, Any]) -> None:
        robot_cfg = dict(lerobot_cfg.get("robot", {}))
        if not robot_cfg:
            raise ValueError("`cfg.lerobot.robot` configuration is required.")

        robot_type = robot_cfg.get("type", "so100_follower")
        if robot_type != "so100_follower":
            raise ValueError(
                f"Only 'so100_follower' is currently supported, but got '{robot_type}'."
            )

        cameras_cfg = robot_cfg.get("cameras", {})
        if cameras_cfg:
            robot_cfg["cameras"] = _build_camera_configs(cameras_cfg)

        self.robot_config = SO100FollowerConfig(**robot_cfg)
        self.robot = SO100Follower(self.robot_config)

        calibrate = bool(lerobot_cfg.get("calibrate_on_start", False))
        try:
            self.robot.connect(calibrate=calibrate)
        except DeviceAlreadyConnectedError:
            pass

        reset_pose = lerobot_cfg.get("reset_joint_positions")
        reset_time_s = lerobot_cfg.get("reset_time_s", 5.0)
        use_gripper = lerobot_cfg.get("use_gripper", True)
        display_cameras = lerobot_cfg.get("display_cameras", False)

        self.env = RobotEnv(
            robot=self.robot,
            use_gripper=use_gripper,
            display_cameras=display_cameras,
            reset_pose=reset_pose,
            reset_time_s=reset_time_s,
        )

        self.teleop_device = NullTeleop(self.robot.bus.motors.keys())
        self.teleop_device.connect(calibrate=False)

    def _init_processors(self, lerobot_cfg: Mapping[str, Any]) -> None:
        fps = lerobot_cfg.get("fps", 30)
        processor_cfg = _build_processor_config(lerobot_cfg.get("processor", {}))

        env_cfg = HILSerlRobotEnvConfig(
            robot=None,
            teleop=None,
            processor=processor_cfg,
            fps=fps,
        )

        self.env_processor, self.action_processor = make_processors(
            self.env,
            self.teleop_device,
            env_cfg,
            device=str(self.device),
        )

        self.control_dt = 1.0 / fps if fps > 0 else 0.1

    def _init_metrics(self) -> None:
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)

    def _reset_metrics(self) -> None:
        self.prev_step_reward[:] = 0.0
        if self.record_metrics:
            self.returns[:] = 0.0
            self.success_once[:] = False

    @property
    def device_tensor(self) -> torch.device:
        return self.device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return torch.from_numpy(self._elapsed_steps)

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    def _extract_obs_from_transition(self, transition) -> dict[str, Any]:
        obs = transition[TransitionKey.OBSERVATION]
        if not isinstance(obs, dict):
            raise ValueError("Processed observation must be a dictionary.")

        image_tensor = None
        wrist_images = []

        if self.primary_camera_key and self.primary_camera_key in obs:
            image_tensor = obs[self.primary_camera_key]
        else:
            for key, value in obs.items():
                if key.startswith("observation.images"):
                    image_tensor = value
                    break

        for key in self.wrist_camera_keys:
            if key in obs:
                wrist_images.append(obs[key])

        if image_tensor is None:
            raise KeyError(
                "Unable to find primary camera tensor. Please set "
                "`cfg.lerobot.primary_camera_key` to a valid observation key."
            )

        if image_tensor.ndim == 4:
            image_tensor = image_tensor.to(torch.float32).contiguous()
        else:
            raise ValueError(
                f"Primary image tensor expected to be 4D (B, C, H, W), got {image_tensor.shape}."
            )

        if wrist_images:
            wrist_images_tensor = torch.stack(
                [img.squeeze(0) if img.ndim == 4 else img for img in wrist_images], dim=0
            )
            wrist_images_tensor = wrist_images_tensor.to(torch.float32)
        else:
            wrist_images_tensor = None

        states_tensor = None
        if self.state_key in obs:
            states_tensor = obs[self.state_key].to(torch.float32)

        task_descriptions = [self.task_prompt] if self.task_prompt else None

        return {
            "images": image_tensor,
            "wrist_images": wrist_images_tensor,
            "states": states_tensor,
            "task_descriptions": task_descriptions,
        }

    def _record_metrics(self, step_reward: torch.Tensor, info: dict[str, Any]) -> dict[str, Any]:
        if not self.record_metrics:
            return info

        self.returns += step_reward
        success_flag = info.get(TeleopEvents.SUCCESS, False)
        if success_flag:
            self.success_once[:] = True

        episode = {
            "return": self.returns.clone(),
            "reward": self.returns.clone() / torch.clamp(
                torch.from_numpy(self._elapsed_steps), min=1
            ),
            "episode_len": torch.from_numpy(self._elapsed_steps),
            "success_once": self.success_once.clone(),
        }
        info = dict(info)
        info["episode"] = episode
        return info

    def _build_reset_info(self, transition) -> dict[str, Any]:
        info = transition.get(TransitionKey.INFO, {})
        if self.task_prompt:
            info = dict(info)
            info["task_prompt"] = self.task_prompt
        return info

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,  # noqa: ARG002
        options: Optional[dict] = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_obs, info = self.env.reset()
        complementary_data = {}
        if hasattr(self.env, "get_raw_joint_positions"):
            complementary_data["raw_joint_positions"] = self.env.get_raw_joint_positions()

        self.env_processor.reset()
        self.action_processor.reset()

        transition = create_transition(
            observation=raw_obs,
            info=info,
            complementary_data=complementary_data,
        )
        transition = self.env_processor(transition)
        self.transition = transition

        self._elapsed_steps[:] = 0
        self._reset_metrics()

        extracted_obs = self._extract_obs_from_transition(transition)
        infos = self._build_reset_info(transition)
        return extracted_obs, infos

    def _compute_step_reward(self, reward_value: Union[float, torch.Tensor]) -> torch.Tensor:
        reward_tensor = to_tensor(reward_value).view(self.num_envs).to(torch.float32)
        reward_tensor *= float(self.reward_coef)
        if self.use_rel_reward:
            reward_diff = reward_tensor - self.prev_step_reward
            self.prev_step_reward = reward_tensor
            return reward_diff
        self.prev_step_reward = reward_tensor
        return reward_tensor

    def step(
        self,
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        auto_reset: bool = True,
    ) -> tuple[dict[str, Any], Optional[torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        if actions is None:
            if not self.is_start:
                raise RuntimeError("Actions must be provided after the first reset.")
            extracted_obs, infos = self.reset()
            self.is_start = False
            terminations = torch.zeros(self.num_envs, dtype=torch.bool)
            truncations = torch.zeros(self.num_envs, dtype=torch.bool)
            return extracted_obs, None, terminations, truncations, infos

        if self.transition is None:
            raise RuntimeError("Environment must be reset before calling step.")

        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)

        transition = step_env_and_process_transition(
            env=self.env,
            transition=self.transition,
            action=action_tensor,
            env_processor=self.env_processor,
            action_processor=self.action_processor,
        )
        self.transition = transition

        reward_value = transition[TransitionKey.REWARD]
        step_reward = self._compute_step_reward(reward_value)

        terminated = bool(transition.get(TransitionKey.DONE, False))
        truncated = bool(transition.get(TransitionKey.TRUNCATED, False))

        terminations = torch.tensor([terminated], dtype=torch.bool)
        truncations = torch.tensor([truncated], dtype=torch.bool)

        info = dict(transition.get(TransitionKey.INFO, {}))

        self._elapsed_steps += 1
        info = self._record_metrics(step_reward, info)

        if self.ignore_terminations:
            terminations[:] = False

        dones = torch.logical_or(terminations, truncations)
        extracted_obs = self._extract_obs_from_transition(transition)

        if dones.any() and auto_reset and self.auto_reset:
            extracted_obs, info = self._handle_auto_reset(
                dones=dones, extracted_obs=extracted_obs, infos=info
            )

        return extracted_obs, step_reward, terminations, truncations, info

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        extracted_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_obs = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in extracted_obs.items()
            if value is not None
        }
        final_info = dict(infos)

        extracted_obs, infos = self.reset()
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def chunk_step(self, chunk_actions: torch.Tensor):
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        infos = {}

        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, infos
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos

    def close(self):
        if hasattr(self, "env"):
            try:
                self.env.close()
            except Exception:
                pass
        if hasattr(self, "robot"):
            try:
                self.robot.disconnect()
            except DeviceNotConnectedError:
                pass
