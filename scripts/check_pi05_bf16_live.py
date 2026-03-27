#!/usr/bin/env python3
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

"""Run one non-destructive PI05 bf16 inference check on YAM observations.

This script is meant for local robot bring-up and sanity-checking. It can:

1. Read the current observation from a running RobotServer without resetting.
2. Or open the three RealSense cameras directly when no server is running.
3. Optionally override the live robot state with a provided literal.

It then loads the PI05 OpenPI policy in bf16, runs one forward pass, and
prints the first predicted action chunk plus summary statistics so we can
quickly spot obviously broken outputs (NaN, Inf, absurd ranges, etc.).
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.embodiment.infer_embodied_agent import (  # noqa: E402
    connect_to_server,
    proto_to_obs,
)
from rlinf.envs.remote.proto import robot_env_pb2  # noqa: E402
from rlinf.envs.yam.realsense_camera import RealsenseCamera  # noqa: E402
from rlinf.models import get_model  # noqa: E402

_DEFAULT_TASK = "fold the towel"
_DEFAULT_SERVER_URL = "localhost:50051"
_DEFAULT_MODEL_PATH = "thomas0829/folding_towel_pi05"
_DEFAULT_SERIALS = {
    "top": "215222073684",
    "left": "218622275075",
    "right": "128422272697",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one PI05 bf16 inference pass on live YAM observations."
    )
    parser.add_argument("--server-url", default=_DEFAULT_SERVER_URL)
    parser.add_argument(
        "--observation-source",
        choices=("grpc", "realsense"),
        default="grpc",
        help="Use RobotServer's current observation or open cameras directly.",
    )
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--action-chunk", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--top-serial", default=_DEFAULT_SERIALS["top"])
    parser.add_argument("--left-serial", default=_DEFAULT_SERIALS["left"])
    parser.add_argument("--right-serial", default=_DEFAULT_SERIALS["right"])
    parser.add_argument(
        "--state-literal",
        default=None,
        help=(
            "Optional Python/JSON literal for the 14D state. Accepts either a flat "
            "14-element list or a {'left': [...7], 'right': [...7]} dict."
        ),
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to save the captured images and predicted chunk.",
    )
    return parser


def _parse_state_literal(state_literal: str | None) -> np.ndarray | None:
    if not state_literal:
        return None
    parsed = ast.literal_eval(state_literal)
    if isinstance(parsed, dict):
        if "left" not in parsed or "right" not in parsed:
            raise ValueError("state-literal dict must contain 'left' and 'right'.")
        flat = list(parsed["left"]) + list(parsed["right"])
    else:
        flat = parsed
    state = np.asarray(flat, dtype=np.float32).reshape(-1)
    if state.shape != (14,):
        raise ValueError(f"Expected 14 state values, got shape {state.shape}.")
    return state.reshape(1, 14)


def _capture_obs_from_server(server_url: str) -> dict:
    stub, spaces, channel = connect_to_server(server_url)
    try:
        proto_obs = stub.GetObservation(robot_env_pb2.Empty(), timeout=30.0)
        obs = proto_to_obs(proto_obs, spaces.img_height, spaces.img_width)
    finally:
        channel.close()
    return obs


def _capture_obs_from_realsense(
    top_serial: str, left_serial: str, right_serial: str, width: int, height: int
) -> dict:
    cameras = {
        "top": RealsenseCamera(
            serial_number=top_serial, resolution=(width, height), fps=30
        ),
        "left": RealsenseCamera(
            serial_number=left_serial, resolution=(width, height), fps=30
        ),
        "right": RealsenseCamera(
            serial_number=right_serial, resolution=(width, height), fps=30
        ),
    }
    try:
        top = cameras["top"].read().images["rgb"]
        left = cameras["left"].read().images["rgb"]
        right = cameras["right"].read().images["rgb"]
    finally:
        for camera in cameras.values():
            camera.stop()

    return {
        "main_images": torch.from_numpy(top[None].copy()),
        "wrist_images": torch.from_numpy(np.stack([left, right], axis=0)[None].copy()),
    }


def _load_bf16_model(
    model_path: str,
    precision: str,
    action_dim: int,
    action_chunk: int,
    num_steps: int,
):
    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": precision,
            "is_lora": False,
            "add_value_head": False,
            "num_action_chunks": action_chunk,
            "action_dim": action_dim,
            "num_steps": num_steps,
            "openpi": {
                "config_name": "pi05_yam_follower",
                "action_dim": action_dim,
                "action_horizon": action_chunk,
                "action_chunk": action_chunk,
                "num_steps": num_steps,
            },
        }
    )
    model = get_model(cfg)
    model.eval()
    return model


def _maybe_save_artifacts(save_dir: str | None, obs: dict, actions: np.ndarray) -> None:
    if save_dir is None:
        return
    import imageio.v2 as imageio

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_dir / "cam_top.png", obs["main_images"][0].cpu().numpy())
    wrist_images = obs["wrist_images"][0].cpu().numpy()
    imageio.imwrite(out_dir / "cam_left.png", wrist_images[0])
    imageio.imwrite(out_dir / "cam_right.png", wrist_images[1])
    np.save(out_dir / "first_chunk.npy", actions)


def _summarize_param_dtypes(model: torch.nn.Module) -> dict[str, int]:
    counts = Counter()
    for _, param in model.named_parameters():
        counts[str(param.dtype)] += param.numel()
    return dict(counts)


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.observation_source == "grpc":
        obs = _capture_obs_from_server(args.server_url)
    else:
        obs = _capture_obs_from_realsense(
            args.top_serial,
            args.left_serial,
            args.right_serial,
            args.image_width,
            args.image_height,
        )

    override_state = _parse_state_literal(args.state_literal)
    if override_state is not None:
        obs["states"] = torch.from_numpy(override_state.copy())
    elif "states" not in obs:
        raise RuntimeError(
            "No state available from the observation source. Pass --state-literal."
        )

    model = _load_bf16_model(
        model_path=args.model_path,
        precision=args.precision,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        num_steps=args.num_steps,
    )
    device = next(model.parameters()).device

    env_obs = {
        "main_images": obs["main_images"].to(device),
        "wrist_images": obs["wrist_images"].to(device),
        "states": obs["states"].to(device),
        "task_descriptions": [args.task],
    }

    with torch.no_grad():
        actions, _ = model.predict_action_batch(
            env_obs, mode="eval", compute_values=False
        )

    actions_cpu = actions.detach().float().cpu().numpy()
    first_chunk = actions_cpu[0]

    print(f"model_device={device}")
    print(f"model_param_dtype={next(model.parameters()).dtype}")
    print(f"param_numel_by_dtype={_summarize_param_dtypes(model)}")
    print(f"actions_shape={actions_cpu.shape}")
    print(f"actions_dtype={actions.dtype}")
    print(f"all_finite={bool(np.isfinite(actions_cpu).all())}")
    print(f"min={float(actions_cpu.min()):.6f}")
    print(f"max={float(actions_cpu.max()):.6f}")
    print(f"mean_abs={float(np.abs(actions_cpu).mean()):.6f}")
    print(f"step0_l2={float(np.linalg.norm(first_chunk[0])):.6f}")
    print("first_action=")
    print(np.array2string(first_chunk[0], precision=6, suppress_small=False))
    print("last_action=")
    print(np.array2string(first_chunk[-1], precision=6, suppress_small=False))
    print("first_chunk=")
    print(np.array2string(first_chunk, precision=6, suppress_small=False))

    _maybe_save_artifacts(args.save_dir, obs, actions_cpu)


if __name__ == "__main__":
    main()
