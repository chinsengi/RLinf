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

"""gRPC server that wraps YAMEnv and serves obs/actions to remote clients.

Run standalone::

    python -m rlinf.envs.remote.robot_server --config-path /path/to/env_config.yaml

The server exposes the robot environment over gRPC so that a ``RemoteEnv``
client running on a Beaker GPU node can drive the real robot over a reverse
SSH tunnel (Tailscale).
"""

import argparse
import os
import signal
import threading
import time
from concurrent import futures

import grpc
import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.remote.proto import robot_env_pb2, robot_env_pb2_grpc
from rlinf.utils.logging import get_logger

logger = get_logger()

_DEFAULT_PORT = 50051
_DEFAULT_MAX_MESSAGE_SIZE = 16 * 1024 * 1024  # 16 MB
_JPEG_QUALITY = 90


def _compress_image(img: np.ndarray, quality: int = _JPEG_QUALITY) -> bytes:
    """Compress uint8 HWC image to JPEG bytes."""
    import cv2

    _, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _to_single_image(images: np.ndarray) -> np.ndarray:
    """Normalize an observation image field to a single HWC uint8 image."""
    if hasattr(images, "cpu"):
        images = images.cpu().numpy()
    images = np.asarray(images, dtype=np.uint8)
    if images.ndim == 4:
        images = images[0]
    return images


def _to_image_list(images: np.ndarray | None) -> list[np.ndarray]:
    """Normalize an observation image field to a list of HWC uint8 images."""
    if images is None:
        return []
    if hasattr(images, "cpu"):
        images = images.cpu().numpy()
    images = np.asarray(images, dtype=np.uint8)
    if images.ndim == 5:
        images = images[0]
    elif images.ndim == 3:
        images = images[np.newaxis, ...]
    return [np.asarray(image, dtype=np.uint8) for image in images]


def _obs_to_proto(
    obs: dict,
    img_h: int,
    img_w: int,
    compress: bool = True,
    jpeg_quality: int = _JPEG_QUALITY,
) -> robot_env_pb2.Observation:
    """Convert a YAMEnv observation dict to a protobuf Observation."""
    # States: tensor → numpy → bytes
    states = obs["states"]
    if hasattr(states, "cpu"):
        states = states.cpu().numpy()
    states = np.asarray(states, dtype=np.float32)
    state_shape = list(states.shape)

    # Main image: tensor → numpy → optionally JPEG compress
    main_images = _to_single_image(obs["main_images"])

    if compress:
        img_bytes = _compress_image(main_images, jpeg_quality)
        wrist_image_bytes = [
            _compress_image(image, jpeg_quality)
            for image in _to_image_list(obs.get("wrist_images"))
        ]
        extra_view_image_bytes = [
            _compress_image(image, jpeg_quality)
            for image in _to_image_list(obs.get("extra_view_images"))
        ]
        is_compressed = True
    else:
        img_bytes = main_images.tobytes()
        wrist_image_bytes = [
            image.tobytes() for image in _to_image_list(obs.get("wrist_images"))
        ]
        extra_view_image_bytes = [
            image.tobytes() for image in _to_image_list(obs.get("extra_view_images"))
        ]
        is_compressed = False

    # Task description
    task_desc = ""
    if "task_descriptions" in obs:
        descs = obs["task_descriptions"]
        if isinstance(descs, list) and len(descs) > 0:
            task_desc = str(descs[0])

    return robot_env_pb2.Observation(
        states=states.tobytes(),
        state_shape=state_shape,
        main_image=img_bytes,
        wrist_images=wrist_image_bytes,
        extra_view_images=extra_view_image_bytes,
        img_height=img_h,
        img_width=img_w,
        is_compressed=is_compressed,
        task_description=task_desc,
    )


def _print_robot_state(env) -> None:
    """Print current robot joint states for pre-flight inspection."""
    if env._is_dummy:
        logger.info("[RobotServer] Running in dummy mode — no real robot state.")
        return

    robot_env = env._robot_env
    if robot_env is None:
        logger.warning("[RobotServer] Robot environment not initialized.")
        return

    print("\n" + "=" * 60)
    print("  Current Robot State")
    print("=" * 60)
    for name in robot_env.get_all_robots().keys():
        joint_pos = env._read_robot_joint_position(name)
        print(f"  [{name}] joint positions ({len(joint_pos)}D):")
        print(f"    {np.array2string(joint_pos, precision=4, suppress_small=True)}")
    print("=" * 60)


class RobotEnvServicer(robot_env_pb2_grpc.RobotEnvServiceServicer):
    """gRPC servicer wrapping a YAMEnv instance."""

    def __init__(
        self,
        env,
        compress: bool = True,
        jpeg_quality: int = _JPEG_QUALITY,
        verbose: bool = False,
        request_shutdown=None,
    ):
        self._env = env
        self._compress = compress
        self._jpeg_quality = jpeg_quality
        self._verbose = verbose
        self._first_chunk_approved = not verbose
        self._request_shutdown = request_shutdown

    def GetSpaces(self, request, context):
        obs_space = self._env.observation_space
        obs_spaces = obs_space.spaces
        act_space = self._env.action_space
        return robot_env_pb2.SpacesResponse(
            state_dim=obs_spaces["states"].shape[0],
            action_dim=act_space.shape[0],
            action_low=float(act_space.low[0]),
            action_high=float(act_space.high[0]),
            img_height=obs_spaces["main_images"].shape[0],
            img_width=obs_spaces["main_images"].shape[1],
            img_channels=obs_spaces["main_images"].shape[2],
            max_episode_steps=self._env._max_episode_steps,
            control_rate_hz=self._env._control_rate_hz,
            auto_reset=self._env.auto_reset,
            ignore_terminations=self._env.ignore_terminations,
            num_wrist_images=(
                obs_spaces["wrist_images"].shape[0]
                if "wrist_images" in obs_spaces
                else 0
            ),
            num_extra_view_images=(
                obs_spaces["extra_view_images"].shape[0]
                if "extra_view_images" in obs_spaces
                else 0
            ),
        )

    _APPROVAL_FILE = "/tmp/rlinf_approve_chunk"

    def _wait_for_first_chunk_approval(self, actions: np.ndarray) -> None:
        """Block until the user approves the first chunk by creating a file."""
        if os.path.exists(self._APPROVAL_FILE):
            os.remove(self._APPROVAL_FILE)

        print("\n" + "=" * 60, flush=True)
        print("  FIRST CHUNK — waiting for approval before executing", flush=True)
        print("=" * 60, flush=True)
        print(
            f"  chunk_size={actions.shape[1]}, action_dim={actions.shape[2]}",
            flush=True,
        )
        for i in range(actions.shape[1]):
            print(
                f"  step {i}: {np.array2string(actions[0, i], precision=4, suppress_small=True)}",
                flush=True,
            )
        print("=" * 60, flush=True)
        print(
            f"  To approve, run:  touch {self._APPROVAL_FILE}",
            flush=True,
        )
        print("  To abort, Ctrl+C the server.", flush=True)
        print("=" * 60 + "\n", flush=True)

        while not os.path.exists(self._APPROVAL_FILE):
            time.sleep(0.5)

        os.remove(self._APPROVAL_FILE)
        self._first_chunk_approved = True
        logger.info("[ChunkStep] First chunk approved. Executing...")

    def Reset(self, request, context):
        seed = request.seed if request.HasField("seed") else None
        obs, _ = self._env.reset(seed=seed)
        return _obs_to_proto(
            obs,
            self._env._img_h,
            self._env._img_w,
            compress=self._compress,
            jpeg_quality=self._jpeg_quality,
        )

    def ChunkStep(self, request, context):
        actions = np.frombuffer(request.actions, dtype=np.float32).reshape(
            request.num_envs, request.chunk_size, request.action_dim
        )
        if self._verbose:
            logger.info(
                f"[ChunkStep] Received chunk: num_envs={request.num_envs}, "
                f"chunk_size={request.chunk_size}, action_dim={request.action_dim}"
            )
            for i in range(request.chunk_size):
                logger.info(
                    f"[ChunkStep]   step {i}/{request.chunk_size}: "
                    f"action={np.array2string(actions[0, i], precision=4, suppress_small=True)}"
                )

        if not self._first_chunk_approved:
            self._wait_for_first_chunk_approval(actions)

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self._env.chunk_step(actions)
        )

        step_results = []
        chunk_size = request.chunk_size
        for i in range(chunk_size):
            obs_proto = _obs_to_proto(
                obs_list[i],
                self._env._img_h,
                self._env._img_w,
                compress=self._compress,
                jpeg_quality=self._jpeg_quality,
            )
            # chunk_rewards shape: (num_envs, chunk_size)
            reward = float(chunk_rewards[0, i])
            terminated = bool(chunk_terminations[0, i])
            truncated = bool(chunk_truncations[0, i])
            step_results.append(
                robot_env_pb2.StepResult(
                    observation=obs_proto,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            )

        if self._verbose:
            logger.info(
                f"[ChunkStep] Done. rewards={[float(chunk_rewards[0, i]) for i in range(chunk_size)]}, "
                f"terminated={[bool(chunk_terminations[0, i]) for i in range(chunk_size)]}, "
                f"truncated={[bool(chunk_truncations[0, i]) for i in range(chunk_size)]}"
            )
        return robot_env_pb2.ChunkStepResponse(step_results=step_results)

    def SetTaskDescription(self, request, context):
        self._env.task_description = request.task_description
        return robot_env_pb2.Empty()

    def EnterZeroTorqueMode(self, request, context):
        self._env.enter_zero_torque_mode()
        return robot_env_pb2.Empty()

    def Close(self, request, context):
        logger.info("[RobotServer] Close RPC received. Scheduling shutdown.")
        if self._request_shutdown is not None:
            self._request_shutdown(return_home=True)
        return robot_env_pb2.Empty()


def serve(
    cfg_path: str,
    port: int = _DEFAULT_PORT,
    max_message_size: int = _DEFAULT_MAX_MESSAGE_SIZE,
    dummy: bool = False,
    verbose: bool = False,
):
    """Start the gRPC server with a YAMEnv instance."""
    from rlinf.envs.yam.yam_env import YAMEnv

    cfg = OmegaConf.load(cfg_path)
    if dummy:
        OmegaConf.update(cfg, "is_dummy", True)

    compress = bool(cfg.get("compress_images", True))
    jpeg_quality = int(cfg.get("jpeg_quality", _JPEG_QUALITY))

    logger.info(f"[RobotServer] Creating YAMEnv from config: {cfg_path}")
    env = YAMEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )

    if verbose:
        _print_robot_state(env)
        ready_flag = os.environ.get("RLINF_ROBOT_SERVER_READY_FLAG")
        if ready_flag:
            with open(ready_flag, "w") as f:
                f.write("ready\n")

    stop_event = threading.Event()
    shutdown_lock = threading.Lock()
    shutdown_state = {"return_home": False}

    def _request_shutdown(return_home: bool) -> None:
        with shutdown_lock:
            if return_home:
                shutdown_state["return_home"] = True
            stop_event.set()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", max_message_size),
            ("grpc.max_receive_message_length", max_message_size),
        ],
    )
    servicer = RobotEnvServicer(
        env,
        compress=compress,
        jpeg_quality=jpeg_quality,
        verbose=verbose,
        request_shutdown=_request_shutdown,
    )
    robot_env_pb2_grpc.add_RobotEnvServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"[RobotServer] Serving on port {port}")

    def _shutdown(signum, frame):
        _request_shutdown(return_home=True)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    stop_event.wait()

    logger.info("[RobotServer] Shutting down...")
    server.stop(grace=1)
    try:
        env.close(return_home=shutdown_state["return_home"])
    except Exception:
        pass

    import multiprocessing

    for child in multiprocessing.active_children():
        logger.info(
            f"[RobotServer] Killing child process: {child.name} (pid={child.pid})"
        )
        child.kill()
        child.join(timeout=2)

    logger.info("[RobotServer] Cleanup complete.")


def main():
    parser = argparse.ArgumentParser(description="Robot gRPC server wrapping YAMEnv")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the YAM environment YAML config file",
    )
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument(
        "--max-message-size",
        type=int,
        default=_DEFAULT_MAX_MESSAGE_SIZE,
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Run in dummy mode (no real hardware, zero observations)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show robot state before serving and log every chunk step",
    )
    args = parser.parse_args()
    serve(
        args.config_path,
        args.port,
        args.max_message_size,
        dummy=args.dummy,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
