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
import sys
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

        # Track last client RPC time for disconnect detection.
        self._last_rpc_time: float = time.monotonic()
        self._client_connected: bool = False
        self._chunk_count: int = 0
        self._episode_cooldown_s: float = float(
            getattr(self._env, "_episode_cooldown_s", 0.0)
        )
        self._cooldown_deadline: float | None = None
        self._restart_required: bool = False
        self._restart_truncation_pending: bool = False
        # Protects all env operations so that safe_recover() and gRPC
        # handlers never touch the robot concurrently (the portal clients'
        # use_future flag is not thread-safe).
        self._env_lock = threading.Lock()

    def _touch(self) -> None:
        """Record that a client RPC was received."""
        self._last_rpc_time = time.monotonic()
        self._client_connected = True

    def _reset_client_session_state(self) -> None:
        """Reset per-client counters after a forced stop/recovery."""
        self._client_connected = False
        self._chunk_count = 0
        self._first_chunk_approved = not self._verbose

    def _get_current_raw_obs_locked(self) -> dict:
        """Return the latest raw observation without commanding the robot."""
        if getattr(self._env, "_is_dummy", False):
            return self._env._dummy_obs()
        robot_env = getattr(self._env, "_robot_env", None)
        if robot_env is None:
            return self._env._dummy_obs()
        return robot_env.get_obs()

    def _build_idle_chunk_response(
        self,
        obs: dict,
        chunk_size: int,
        *,
        truncated: bool,
    ) -> robot_env_pb2.ChunkStepResponse:
        """Return a no-op chunk response that surfaces the current observation."""
        step_results = []
        for step_idx in range(chunk_size):
            obs_proto = _obs_to_proto(
                obs,
                self._env._img_h,
                self._env._img_w,
                compress=self._compress,
                jpeg_quality=self._jpeg_quality,
            )
            step_results.append(
                robot_env_pb2.StepResult(
                    observation=obs_proto,
                    reward=0.0,
                    terminated=False,
                    truncated=bool(truncated and step_idx == chunk_size - 1),
                )
            )
        return robot_env_pb2.ChunkStepResponse(step_results=step_results)

    def _start_restart_flow_locked(
        self,
        *,
        reason: str,
        enter_zero_torque: bool,
        cooldown_s: float,
        prepare_for_reconnection: bool,
    ) -> None:
        """Return home and arm the next episode/session restart."""
        logger.info(reason)
        try:
            self._env.return_to_home()
            logger.info("[RobotServer] Arms returned to home.")
        except Exception as exc:
            logger.error(f"[RobotServer] Failed to return home: {exc}")

        if enter_zero_torque:
            try:
                self._env.enter_zero_torque_mode()
                logger.info("[RobotServer] Motors in zero-torque mode.")
            except Exception as exc:
                logger.error(f"[RobotServer] Failed to enter zero-torque: {exc}")

        if prepare_for_reconnection:
            self._env.prepare_for_reconnection()
        else:
            self._env.prepare_for_next_episode()

        self._restart_required = True
        self._restart_truncation_pending = True
        self._cooldown_deadline = (
            time.monotonic() + cooldown_s if cooldown_s > 0 else None
        )
        self._reset_client_session_state()

    def _finish_restart_if_ready_locked(self) -> dict | None:
        """Reset from home once cooldown has elapsed and return the fresh obs."""
        if not self._restart_required:
            return None

        if self._cooldown_deadline is not None:
            remaining_s = self._cooldown_deadline - time.monotonic()
            if remaining_s > 0:
                return None

        obs, _ = self._env.reset()
        self._restart_required = False
        self._cooldown_deadline = None
        logger.info("[RobotServer] Episode/session restarted from home.")
        return obs

    def poll_restart_if_ready(self) -> dict | None:
        """Finish a cooldown-driven restart outside request handlers when ready."""
        with self._env_lock:
            return self._finish_restart_if_ready_locked()

    def GetSpaces(self, request, context):
        self._touch()
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
        self._touch()
        with self._env_lock:
            seed = request.seed if request.HasField("seed") else None
            if self._restart_required:
                obs = self._finish_restart_if_ready_locked()
                if obs is None:
                    obs = self._env._wrap_obs(self._get_current_raw_obs_locked())
            else:
                obs, _ = self._env.reset(seed=seed)
        return _obs_to_proto(
            obs,
            self._env._img_h,
            self._env._img_w,
            compress=self._compress,
            jpeg_quality=self._jpeg_quality,
        )

    def GetObservation(self, request, context):
        self._touch()
        with self._env_lock:
            obs = self._env._wrap_obs(self._get_current_raw_obs_locked())
        return _obs_to_proto(
            obs,
            self._env._img_h,
            self._env._img_w,
            compress=self._compress,
            jpeg_quality=self._jpeg_quality,
        )

    def ChunkStep(self, request, context):
        self._touch()
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

        with self._env_lock:
            if self._restart_required:
                obs = self._finish_restart_if_ready_locked()
                if obs is None:
                    obs = self._env._wrap_obs(self._get_current_raw_obs_locked())
                emit_truncated = self._restart_truncation_pending
                self._restart_truncation_pending = False
                return self._build_idle_chunk_response(
                    obs, request.chunk_size, truncated=emit_truncated
                )

        if not self._first_chunk_approved:
            self._wait_for_first_chunk_approval(actions)

        with self._env_lock:
            (
                obs_list,
                chunk_rewards,
                chunk_terminations,
                chunk_truncations,
                infos_list,
            ) = self._env.chunk_step(actions)
        self._chunk_count += 1

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
        self._touch()
        self._env.task_description = request.task_description
        return robot_env_pb2.Empty()

    def EnterZeroTorqueMode(self, request, context):
        self._touch()
        with self._env_lock:
            self._env.enter_zero_torque_mode()
        return robot_env_pb2.Empty()

    def Close(self, request, context):
        self._touch()
        logger.info(
            "[RobotServer] Close RPC received from remote client. "
            "Returning home, entering zero-torque, and waiting for a new client."
        )
        with self._env_lock:
            self._start_restart_flow_locked(
                reason=(
                    "[RobotServer] Remote client requested session close — "
                    "starting safe recovery."
                ),
                enter_zero_torque=True,
                cooldown_s=0.0,
                prepare_for_reconnection=True,
            )
        return robot_env_pb2.Empty()

    def episode_timeout(self) -> None:
        """Return arms to home when the episode timer expires.

        Called by the timer thread.  Acquires ``_env_lock`` and re-checks
        the timer to avoid acting on a stale reading.
        """
        with self._env_lock:
            start = self._env._episode_start_time
            if start is None:
                return
            elapsed = time.monotonic() - start
            if elapsed < self._env._episode_duration_s:
                return
            self._start_restart_flow_locked(
                reason=(
                    "[RobotServer] Episode timer expired — returning home and "
                    "starting the server-side restart countdown."
                ),
                enter_zero_torque=True,
                cooldown_s=self._episode_cooldown_s,
                prepare_for_reconnection=False,
            )

    def safe_recover(self, idle_timeout_s: float) -> None:
        """Return arms to home, enter zero-torque, and prepare for reconnection.

        Called by the watchdog when the client appears to have disconnected.
        Acquires ``_env_lock`` to avoid racing with in-flight gRPC handlers,
        then re-checks the idle time — if a new RPC arrived while waiting for
        the lock the recovery is skipped.
        """
        with self._env_lock:
            # Re-check: a new RPC may have arrived while we waited for the lock.
            idle_s = time.monotonic() - self._last_rpc_time
            if idle_s < idle_timeout_s:
                logger.info(
                    "[RobotServer] Client activity detected after lock acquired "
                    "(idle=%.1fs) — skipping safe recovery.",
                    idle_s,
                )
                return

            self._start_restart_flow_locked(
                reason=(
                    "[RobotServer] Client disconnected — returning home, "
                    "entering zero-torque, and waiting for reconnection."
                ),
                enter_zero_torque=True,
                cooldown_s=0.0,
                prepare_for_reconnection=True,
            )
            logger.info(
                "[RobotServer] Safe recovery complete. "
                "Server is still listening — waiting for new client."
            )


_DEFAULT_CLIENT_IDLE_TIMEOUT_S = 120.0
_WATCHDOG_POLL_INTERVAL_S = 5.0


def _parse_client_idle_timeout_s(raw_timeout: object) -> float | None:
    """Normalize the client-idle watchdog timeout.

    ``None`` or any non-positive value disables the watchdog entirely. This is
    useful for long-running real-robot sessions where the Beaker side may go
    quiet for extended periods during planner/model work, but the desktop
    server should keep the robot session alive until an explicit stop.
    """
    if raw_timeout is None:
        return None
    timeout_s = float(raw_timeout)
    if timeout_s <= 0:
        return None
    return timeout_s


def serve(
    cfg_path: str,
    port: int = _DEFAULT_PORT,
    max_message_size: int = _DEFAULT_MAX_MESSAGE_SIZE,
    dummy: bool = False,
    verbose: bool = False,
):
    """Start the gRPC server with a YAMEnv instance.

    The server stays alive across client disconnects.  A watchdog thread
    monitors client activity; when no RPC has been received for
    ``client_idle_timeout_s`` seconds and a client was previously connected,
    the watchdog triggers safe recovery (return-to-home → zero-torque) and
    then waits for the next client.

    A local ``Ctrl+C`` (SIGINT/SIGTERM) shuts the server down for real.
    """
    from rlinf.envs.yam.yam_env import YAMEnv

    cfg = OmegaConf.load(cfg_path)
    if dummy:
        OmegaConf.update(cfg, "is_dummy", True)

    compress = bool(cfg.get("compress_images", True))
    jpeg_quality = int(cfg.get("jpeg_quality", _JPEG_QUALITY))
    client_idle_timeout_s = _parse_client_idle_timeout_s(
        cfg.get("client_idle_timeout_s", _DEFAULT_CLIENT_IDLE_TIMEOUT_S)
    )

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

    def _request_shutdown(return_home: bool) -> None:
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

    # ---- Watchdog: detect client disconnect and trigger safe recovery ----
    # Only activates after the first ChunkStep has been processed so that
    # slow model loading on the Beaker side does not trigger a false alarm.
    def _watchdog() -> None:
        while not stop_event.is_set():
            stop_event.wait(timeout=_WATCHDOG_POLL_INTERVAL_S)
            if stop_event.is_set():
                break
            if not servicer._client_connected or servicer._chunk_count == 0:
                continue
            idle_s = time.monotonic() - servicer._last_rpc_time
            if idle_s >= client_idle_timeout_s:
                logger.warning(
                    "[RobotServer] No client RPC for %.0fs "
                    "(timeout=%.0fs). Assuming client disconnected.",
                    idle_s,
                    client_idle_timeout_s,
                )
                servicer.safe_recover(idle_timeout_s=client_idle_timeout_s)

    watchdog_thread = None
    if client_idle_timeout_s is not None:
        watchdog_thread = threading.Thread(
            target=_watchdog, name="robot-server-watchdog", daemon=True
        )
        watchdog_thread.start()
    else:
        logger.info(
            "[RobotServer] Client-idle watchdog disabled; the server will keep "
            "running until an explicit stop or episode timeout."
        )

    # ---- Timer: 1-second countdown display + episode timeout handling ----
    def _timer_display() -> None:
        last_line_len = 0
        while not stop_event.is_set():
            time.sleep(1.0)
            if stop_event.is_set():
                break
            start = env._episode_start_time
            cooldown_deadline = servicer._cooldown_deadline
            if cooldown_deadline is not None:
                remaining_s = max(0.0, cooldown_deadline - time.monotonic())
                if remaining_s > 0:
                    rem_m, rem_s = divmod(int(remaining_s), 60)
                    line = f"  [Cooldown] {rem_m}:{rem_s:02d} before restart"
                    sys.stdout.write("\r" + line.ljust(last_line_len))
                    sys.stdout.flush()
                    last_line_len = len(line)
                    continue
                servicer.poll_restart_if_ready()
            if start is None:
                # Clear the timer line when no episode is active.
                if last_line_len > 0:
                    sys.stdout.write("\r" + " " * last_line_len + "\r")
                    sys.stdout.flush()
                    last_line_len = 0
                continue
            elapsed_s = time.monotonic() - start
            duration_s = env._episode_duration_s
            remaining_s = max(0.0, duration_s - elapsed_s)
            if remaining_s <= 0:
                sys.stdout.write("\r" + " " * last_line_len + "\r")
                sys.stdout.flush()
                last_line_len = 0
                servicer.episode_timeout()
                continue
            rem_m, rem_s = divmod(int(remaining_s), 60)
            tot_m, tot_s = divmod(int(duration_s), 60)
            line = f"  [Timer] {rem_m}:{rem_s:02d} remaining (of {tot_m}:{tot_s:02d})"
            sys.stdout.write("\r" + line.ljust(last_line_len))
            sys.stdout.flush()
            last_line_len = len(line)

    timer_thread = threading.Thread(
        target=_timer_display, name="robot-server-timer", daemon=True
    )
    timer_thread.start()

    # ---- Signal handler: local Ctrl+C triggers real shutdown ----
    def _shutdown(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    stop_event.wait()

    # ---- Shutdown: return home FIRST while portal connections are alive ----
    sys.stdout.write("\n")  # move past the timer line
    sys.stdout.flush()
    logger.info("[RobotServer] Shutting down...")
    server.stop(grace=2)

    logger.info("[RobotServer] Returning arms to home...")
    try:
        env.return_to_home()
        logger.info("[RobotServer] Arms at home.")
    except Exception as exc:
        logger.error(f"[RobotServer] Failed to return home: {exc}")

    logger.info("[RobotServer] Local shutdown requested — skipping zero-torque mode.")

    try:
        env.close(return_home=False)
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
