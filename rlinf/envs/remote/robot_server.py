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
import select
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
_BIND_ADDRESSES = ("[::]:{port}", "0.0.0.0:{port}")


def _bind_server(server: grpc.Server, port: int) -> str:
    """Bind the gRPC server to the first available listen address."""
    errors: list[str] = []
    for address_template in _BIND_ADDRESSES:
        address = address_template.format(port=port)
        try:
            bound_port = server.add_insecure_port(address)
        except RuntimeError as exc:
            errors.append(f"{address} ({exc})")
            continue

        if bound_port:
            return address

        errors.append(f"{address} (returned port 0)")

    joined_errors = "; ".join(errors)
    raise RuntimeError(
        "Failed to bind RobotServer to any listen address for "
        f"port {port}. Tried: {joined_errors}"
    )


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
    """gRPC servicer that exposes a ``YAMEnv`` to a remote ``RemoteEnv`` client.

    The servicer runs on the robot desktop and translates gRPC RPCs into
    ``YAMEnv`` calls (``reset``, ``chunk_step``, etc.).  A ``RemoteEnv``
    client on a Beaker GPU node connects through a reverse SSH tunnel.

    Lifecycle:
        1. Client calls ``GetSpaces`` / ``Reset`` to start an episode.
        2. Client sends ``ChunkStep`` RPCs with action chunks; the servicer
           executes them on the robot, encodes observations (JPEG for images,
           lossless PNG for reward frames), and streams results back.
        3. Episode ends via ``Reset`` (normal) or via ``Close`` /
           ``episode_timeout`` / ``safe_recover`` (recovery).
        4. Recovery paths call ``_start_restart_flow_locked`` which returns
           the robot home, optionally enters zero-torque, starts a cooldown
           countdown, and resets session state.  The next ``Reset`` from the
           client completes the restart via ``_finish_restart_if_ready_locked``.

    Modes:
        - **verbose** (``--verbose``): prints per-chunk info to the terminal.
        - **step-by-step** (``--sbs``): implies verbose; pauses before each
          chunk for user approval, saves cumulative TOPReward input video
          (``topreward_input.mp4``), per-chunk PNG frames, and the
          reconstructed prompt (``prompt.txt``) to ``sbs_reward_frames/``.
        - **no-action** (``--no-action``, requires ``--sbs``): sends zero
          actions instead of the policy output.

    Thread safety:
        All ``YAMEnv`` operations are guarded by ``_env_lock``.  The watchdog
        (``safe_recover``) and episode timer (``episode_timeout``) acquire the
        same lock, so they never race with in-flight gRPC handlers.

    Args:
        env: A ``YAMEnv`` (or dummy) instance.
        compress: Whether to JPEG-compress camera images in observations.
        jpeg_quality: JPEG quality (0-100) when ``compress`` is True.
        verbose: Print per-chunk diagnostics.
        step_by_step: Pause before each chunk and save reward frame previews.
        no_action: Replace policy actions with zeros (requires ``step_by_step``).
        request_shutdown: Callable invoked when the server should exit.
        stop_event: ``threading.Event`` signalled on shutdown to unblock
            blocking waits (SBS approval prompts, cooldown sleeps, etc.).
    """

    def __init__(
        self,
        env,
        compress: bool = True,
        jpeg_quality: int = _JPEG_QUALITY,
        verbose: bool = False,
        step_by_step: bool = False,
        no_action: bool = False,
        request_shutdown=None,
        stop_event: threading.Event | None = None,
    ):
        self._env = env
        self._compress = compress
        self._jpeg_quality = jpeg_quality
        self._verbose = verbose or step_by_step
        self._step_by_step = step_by_step
        self._no_action = no_action and step_by_step
        self._first_chunk_approved = not self._verbose
        self._request_shutdown = request_shutdown
        self._stop_event = stop_event or threading.Event()

        # Track last client RPC time for disconnect detection.
        self._last_rpc_time: float = time.monotonic()
        self._client_connected: bool = False
        self._chunk_count: int = 0
        self._episode_count: int = 0
        self._initial_task_description: str = self._read_env_task_description()
        self._episode_cooldown_s: float = float(
            getattr(self._env, "_episode_cooldown_s", 0.0)
        )
        self._cooldown_deadline: float | None = None
        self._restart_required: bool = False
        # Latest status info pushed from the training client via
        # PushStatusInfo RPC (e.g. TOPReward score, lr).
        self._client_status_values: dict[str, float] = {}
        self._client_status_text: str = ""
        self._client_status_updated_at: float = 0.0
        # Track subtask changes to annotate SBS when baseline resets.
        self._prev_subtask: str = self._initial_task_description
        # Protects all env operations so that safe_recover() and gRPC
        # handlers never touch the robot concurrently (the portal clients'
        # use_future flag is not thread-safe).
        self._env_lock = threading.Lock()
        # Cumulative episode frames mirroring what TOPReward accumulates
        # on the training side.  Used to write topreward_input.mp4 in SBS.
        self._episode_reward_frames: list[np.ndarray] = []
        # SBS preview: save reward frames locally so the user can inspect
        # exactly what the VLM receives.
        self._sbs_preview_dir: str | None = None
        if self._step_by_step:
            # Place preview frames inside the project root so they are easy
            # to browse from the IDE.
            _project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            self._sbs_preview_dir = os.path.join(_project_root, "sbs_reward_frames")
            self._maybe_clean_preview_dir()
            os.makedirs(self._sbs_preview_dir, exist_ok=True)
            logger.info(
                f"[SBS] Reward frame preview directory: {self._sbs_preview_dir}"
            )

    def _maybe_clean_preview_dir(self) -> None:
        """Prompt the user to delete old SBS preview frames on restart."""
        if self._sbs_preview_dir is None:
            return
        if not os.path.isdir(self._sbs_preview_dir):
            return
        # Check if the directory has any content.
        try:
            entries = os.listdir(self._sbs_preview_dir)
        except OSError:
            return
        if not entries:
            return
        print(
            f"[SBS] Found existing reward frames in {self._sbs_preview_dir}/ "
            f"({len(entries)} items).",
            flush=True,
        )
        print("[SBS] Delete old frames? [Y/n] ", end="", flush=True)
        try:
            with open("/dev/tty", "r") as tty:
                answer = tty.readline().strip().lower()
        except OSError:
            answer = input().strip().lower()
        if answer in ("", "y", "yes"):
            import shutil

            shutil.rmtree(self._sbs_preview_dir)
            print("[SBS] Old frames deleted.", flush=True)
        else:
            print("[SBS] Keeping old frames.", flush=True)

    def _wait_for_enter_or_shutdown(self) -> bool:
        """Block until the user presses Enter or stop_event is set.

        Returns ``True`` if the user pressed Enter, ``False`` if interrupted
        by the stop event (shutdown requested).
        """
        try:
            with open("/dev/tty", "r") as tty:
                fd = tty.fileno()
                while not self._stop_event.is_set():
                    ready, _, _ = select.select([fd], [], [], 0.5)
                    if ready:
                        tty.readline()
                        return True
                return False
        except OSError:
            # No /dev/tty (e.g. Docker without -it); fall back to stdin.
            while not self._stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if ready:
                    sys.stdin.readline()
                    return True
            return False

    def _read_env_task_description(self) -> str:
        """Return the current task/subtask string from the wrapped env."""
        task_description = getattr(self._env, "task_description", "")
        return str(task_description or "").strip()

    def _format_chunk_task_context(self) -> str:
        """Return a compact terminal string showing the current task."""
        current_task = self._read_env_task_description()
        return f'task="{current_task}"' if current_task else 'task=""'

    def _print_chunk_task_context(self) -> None:
        """Print the current task/subtask context for each received chunk."""
        chunk_index = self._chunk_count + 1
        print(
            f"[RobotServer][ChunkStep {chunk_index}] {self._format_chunk_task_context()}",
            flush=True,
        )

    def _save_reward_frames_for_preview(
        self, frames: list[np.ndarray], chunk_num: int
    ) -> None:
        """Save the cumulative TOPReward input video, prompt, and per-chunk PNGs.

        Each chunk's new frames are appended to ``_episode_reward_frames`` so
        that the resulting ``topreward_input.mp4`` mirrors the full frame
        history that ``VLMPlannerClient._episode_frames`` accumulates on the
        Beaker side for each TOPReward scoring call.

        A ``prompt.txt`` is written alongside the video containing the
        reconstructed TOPReward prompt text and metadata so users can compare
        with the official TOPReward implementation.
        """
        if not self._sbs_preview_dir:
            return
        import cv2

        self._episode_reward_frames.extend(frames)

        ep_dir = os.path.join(
            self._sbs_preview_dir, f"episode_{self._episode_count:04d}"
        )
        os.makedirs(ep_dir, exist_ok=True)

        # Write the cumulative video that matches what TOPReward receives.
        if len(self._episode_reward_frames) >= 2:
            h, w = self._episode_reward_frames[0].shape[:2]
            cumulative_path = os.path.join(ep_dir, "topreward_input.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(cumulative_path, fourcc, 2.0, (w, h))
            for frame in self._episode_reward_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

        self._save_topreward_prompt(ep_dir, chunk_num, len(frames))

        # Save per-chunk PNGs for detailed inspection.
        chunk_dir = os.path.join(ep_dir, f"chunk_{chunk_num:04d}")
        os.makedirs(chunk_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            path = os.path.join(chunk_dir, f"frame_{i:02d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Matches _TOPREWARD_PROMPT_PREFIX in vlm_planner_worker.py and
    # _PROMPT_PREFIX in rlinf/algorithms/rewards/top_reward/top_reward.py,
    # which both mirror the official TOPReward/TOPReward prompt.
    _TOPREWARD_PROMPT_PREFIX = (
        "The above video shows a robot manipulation trajectory that "
        "completes the following task: "
    )

    def _save_topreward_prompt(
        self, ep_dir: str, chunk_num: int, new_frame_count: int
    ) -> None:
        """Write the reconstructed TOPReward prompt and metadata to disk.

        The text mirrors what ``VLMPlannerWorker._prepare_top_reward_request``
        assembles on the Beaker side (without the Qwen chat-template wrapper,
        which only adds special tokens around the content).
        """
        instruction = self._read_env_task_description()
        num_frames = len(self._episode_reward_frames)
        fps = 2.0

        prompt_text = self._TOPREWARD_PROMPT_PREFIX
        instruction_suffix = (
            f"{instruction} Decide whether the above statement is True "
            "or not. The answer is: True"
        )
        full_user_text = f"{prompt_text}{instruction_suffix}"

        prompt_path = os.path.join(ep_dir, "prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(
                f"TOPReward Prompt  (chunk {chunk_num}, "
                f"episode {self._episode_count})\n"
            )
            f.write("=" * 72 + "\n\n")
            f.write(f"Instruction : {instruction}\n")
            f.write(f"Num frames  : {num_frames}  (+{new_frame_count} this chunk)\n")
            f.write(f"FPS         : {fps}\n\n")
            f.write("--- user message content (video + text) ---\n\n")
            f.write(f"[VIDEO: {num_frames} RGB frames at {fps} fps]\n\n")
            f.write(f"{full_user_text}\n\n")
            f.write("--- label target (last token only) ---\n\n")
            f.write(
                'Only the final token of "True" is unmasked; '
                "log P(True_token | everything_before) is the reward.\n\n"
            )
            f.write("--- comparison with official TOPReward ---\n\n")
            f.write(
                "Prompt prefix, instruction suffix, label masking, and\n"
                "video input format are identical to the default path in\n"
                "TOPReward/TOPReward (add_chat_template=false).\n"
                "See: github.com/TOPReward/TOPReward/blob/main/"
                "topreward/clients/qwen.py\n"
            )

    def _format_sbs_status(self) -> str:
        """Return a one-line status summary for the SBS prompt."""
        with self._env_lock:
            # Snapshot mutable state under the lock so that concurrent
            # PushStatusInfo / SetEpisodeTiming RPCs cannot mutate values
            # mid-read (dict iteration race, torn timing reads).
            duration_s = float(getattr(self._env, "_episode_duration_s", 0.0))
            start = getattr(self._env, "_episode_start_time", None)
            status_values = dict(self._client_status_values)
            status_text = self._client_status_text

        if start is not None:
            elapsed_s = max(0.0, time.monotonic() - start)
        else:
            elapsed_s = 0.0
        remaining_s = max(0.0, duration_s - elapsed_s) if duration_s > 0 else 0.0

        def _fmt(sec: float) -> str:
            m, s = divmod(int(sec), 60)
            return f"{m}:{s:02d}"

        next_chunk = self._chunk_count + 1
        parts = [
            f"chunk#{next_chunk}",
            f"ep_time={_fmt(elapsed_s)}/{_fmt(duration_s)} (left={_fmt(remaining_s)})",
        ]
        # Append client-pushed values (TOPReward score/delta, lr, ...).
        # For chunk#1 (no values pushed yet), show delta=0 with score
        # pending, since the first async reward hasn't resolved yet.
        if status_values:
            kv_strs = [f"{k}={v:+.4f}" for k, v in status_values.items()]
            parts.append(", ".join(kv_strs))
        else:
            parts.append("top_reward_delta=+0.0000, top_reward_score=--")
        # Detect subtask change: the displayed delta/score may use a
        # freshly seeded baseline, so annotate the discontinuity.
        # Skip chunk#1 — subtask is already set before the first chunk
        # arrives, so comparing against base task would be a false positive.
        current_subtask = self._read_env_task_description()
        if self._chunk_count > 0 and current_subtask != self._prev_subtask:
            parts.append("(baseline reset)")
        self._prev_subtask = current_subtask
        if status_text:
            parts.append(status_text)
        return " | ".join(parts)

    def _touch(self) -> None:
        """Record that a client RPC was received."""
        self._last_rpc_time = time.monotonic()
        self._client_connected = True

    def _reset_client_session_state(self) -> None:
        """Reset per-client counters after a forced stop/recovery."""
        self._client_connected = False
        self._chunk_count = 0
        self._first_chunk_approved = not self._verbose
        self._client_status_values = {}
        self._client_status_text = ""
        self._client_status_updated_at = 0.0
        self._episode_reward_frames = []
        self._episode_count += 1

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
                    reward=float("nan"),
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
        """Return home and prepare for the next client connection."""
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
        self._cooldown_deadline = (
            time.monotonic() + cooldown_s if cooldown_s > 0 else None
        )
        self._reset_client_session_state()

    def _finish_restart_if_ready_locked(self) -> dict | None:
        """Complete a pending restart once the cooldown has finished."""
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

    def _wait_for_first_chunk_approval(self, actions: np.ndarray) -> None:
        """Block until the user approves the first chunk by pressing Enter."""
        print("\n" + "=" * 60, flush=True)
        print("  FIRST CHUNK — waiting for approval before executing", flush=True)
        print("=" * 60, flush=True)
        if not self._no_action:
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
        print("  Press Enter to approve, or Ctrl+C the server to abort.", flush=True)
        print("=" * 60 + "\n", flush=True)

        if not self._wait_for_enter_or_shutdown():
            logger.info("[ChunkStep] First chunk aborted by shutdown.")
            return

        self._first_chunk_approved = True
        logger.info("[ChunkStep] First chunk approved. Executing...")

    def Reset(self, request, context):
        self._touch()
        self._episode_reward_frames = []
        with self._env_lock:
            seed = request.seed if request.HasField("seed") else None
            if self._restart_required:
                obs = self._finish_restart_if_ready_locked()
                if obs is None:
                    obs = self._env._wrap_obs(self._get_current_raw_obs_locked())
            else:
                self._episode_count += 1
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
        # If shutdown is in progress, return idle immediately so the gRPC
        # thread does not block on SBS prompts or env operations.
        if self._stop_event.is_set():
            return self._build_idle_chunk_response(
                self._env._wrap_obs(self._env._dummy_obs()),
                request.chunk_size,
                truncated=True,
            )
        actions = np.frombuffer(request.actions, dtype=np.float32).reshape(
            request.num_envs, request.chunk_size, request.action_dim
        )
        self._print_chunk_task_context()
        if self._verbose and not self._no_action:
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
                return self._build_idle_chunk_response(
                    obs, request.chunk_size, truncated=True
                )

        if not self._first_chunk_approved:
            self._wait_for_first_chunk_approval(actions)
            if self._stop_event.is_set():
                return self._build_idle_chunk_response(
                    self._env._wrap_obs(self._env._dummy_obs()),
                    request.chunk_size,
                    truncated=True,
                )

        if self._step_by_step and not self._stop_event.is_set():
            sbs_status = self._format_sbs_status()
            task_ctx = self._format_chunk_task_context()
            print(f"[SBS] {sbs_status}", flush=True)
            print(f"[SBS] {task_ctx}", flush=True)
            print("[SBS] Press Enter to execute this chunk...", flush=True)
            if not self._wait_for_enter_or_shutdown():
                logger.info("[SBS] Chunk aborted by shutdown.")
                return self._build_idle_chunk_response(
                    self._env._wrap_obs(self._env._dummy_obs()),
                    request.chunk_size,
                    truncated=True,
                )

        with self._env_lock:
            # Re-check after the approval / step-by-step gap: episode_timeout()
            # may have started the restart flow while the lock was not held.
            if self._restart_required:
                obs = self._finish_restart_if_ready_locked()
                if obs is None:
                    obs = self._env._wrap_obs(self._get_current_raw_obs_locked())
                return self._build_idle_chunk_response(
                    obs, request.chunk_size, truncated=True
                )

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
        # The env's chunk_step only returns obs for the last step (or
        # the step where the episode ended).  Use the last entry in
        # obs_list for the final StepResult and send empty observations
        # for intermediate steps to save bandwidth.
        last_obs = obs_list[-1] if obs_list else None
        for i in range(chunk_size):
            if i == chunk_size - 1 and last_obs is not None:
                obs_proto = _obs_to_proto(
                    last_obs,
                    self._env._img_h,
                    self._env._img_w,
                    compress=self._compress,
                    jpeg_quality=self._jpeg_quality,
                )
            else:
                obs_proto = robot_env_pb2.Observation()
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

        # In SBS mode the YAM env reward is always 0 (dense reward is computed
        # on the training side via TOPReward), so dumping the zero-filled
        # rewards/terminated/truncated arrays only adds noise; skip it.
        if self._verbose and not self._step_by_step:
            logger.info(
                f"[ChunkStep] Done. rewards={[float(chunk_rewards[0, i]) for i in range(chunk_size)]}, "
                f"terminated={[bool(chunk_terminations[0, i]) for i in range(chunk_size)]}, "
                f"truncated={[bool(chunk_truncations[0, i]) for i in range(chunk_size)]}"
            )

        # Encode reward frames as lossless PNG so that TOPReward sees
        # full-quality images (JPEG artifacts degrade VLM scores).
        # Includes both intermediate frames (every N steps) AND the
        # final-step observation so the client can skip the lossy JPEG obs.
        reward_frames_bytes = []
        raw_reward_frames = getattr(self._env, "_last_chunk_reward_frames", [])
        # Append the final-step image from the wrapped observation.
        if last_obs is not None:
            final_img = last_obs.get("main_images")
            if final_img is not None:
                final_img = np.asarray(final_img)
                if final_img.ndim == 4:
                    final_img = final_img[0]
                raw_reward_frames = list(raw_reward_frames) + [final_img]
        if raw_reward_frames:
            import cv2

            for frame in raw_reward_frames:
                _, buf = cv2.imencode(
                    ".png",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
                reward_frames_bytes.append(bytes(buf))

            if self._step_by_step:
                self._save_reward_frames_for_preview(
                    raw_reward_frames, self._chunk_count
                )

        return robot_env_pb2.ChunkStepResponse(
            step_results=step_results,
            reward_frames=reward_frames_bytes,
        )

    def SetTaskDescription(self, request, context):
        self._touch()
        self._env.task_description = request.task_description
        return robot_env_pb2.Empty()

    def EnterZeroTorqueMode(self, request, context):
        self._touch()
        with self._env_lock:
            self._env.enter_zero_torque_mode()
        return robot_env_pb2.Empty()

    def SetEpisodeTiming(self, request, context):
        self._touch()
        duration_s = float(request.episode_duration_s)
        cooldown_s = float(request.episode_cooldown_s)
        with self._env_lock:
            changes = []
            if duration_s > 0:
                old = float(getattr(self._env, "_episode_duration_s", 0.0))
                self._env._episode_duration_s = duration_s
                changes.append(f"episode_duration_s: {old:.1f} -> {duration_s:.1f}")
            if cooldown_s >= 0:
                old = self._episode_cooldown_s
                self._episode_cooldown_s = cooldown_s
                # Keep env-side value in sync for any code that reads it directly.
                if hasattr(self._env, "_episode_cooldown_s"):
                    self._env._episode_cooldown_s = cooldown_s
                changes.append(f"episode_cooldown_s: {old:.1f} -> {cooldown_s:.1f}")
        if changes:
            logger.info(
                "[RobotServer] Episode timing updated from client: %s",
                ", ".join(changes),
            )
        return robot_env_pb2.Empty()

    def PushStatusInfo(self, request, context):
        self._touch()
        with self._env_lock:
            self._client_status_values = {
                k: float(v) for k, v in request.values.items()
            }
            self._client_status_text = str(request.text or "")
            self._client_status_updated_at = time.monotonic()
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
        """Start cooldown when the episode timer expires.

        Called by the timer thread.  Acquires ``_env_lock`` and directly
        starts the restart flow (return-to-home, zero-torque, cooldown
        countdown) so that the robot is made safe even when no further
        ``ChunkStep`` arrives.  Because the lock is held, no concurrent
        ``chunk_step()`` can be in progress.
        """
        with self._env_lock:
            start = self._env._episode_start_time
            if start is None:
                return
            elapsed = time.monotonic() - start
            if elapsed < self._env._episode_duration_s:
                return
            if not self._restart_required:
                self._start_restart_flow_locked(
                    reason=(
                        "[RobotServer] Episode timer expired — returning "
                        "home and starting the server-side restart "
                        "countdown."
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
    step_by_step: bool = False,
    no_action: bool = False,
    reward_frame_interval: int = 5,
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

    # Enable intermediate frame capture for TOPReward scoring.
    env._reward_frame_interval = int(reward_frame_interval)
    if reward_frame_interval > 0:
        logger.info(
            f"[RobotServer] Reward frame capture enabled: every {reward_frame_interval} steps"
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
        step_by_step=step_by_step,
        no_action=no_action,
        request_shutdown=_request_shutdown,
        stop_event=stop_event,
    )
    robot_env_pb2_grpc.add_RobotEnvServiceServicer_to_server(servicer, server)

    bound_address = _bind_server(server, port)
    server.start()
    logger.info(f"[RobotServer] Serving on {bound_address}")

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
    with servicer._env_lock:
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
    parser.add_argument(
        "--sbs",
        action="store_true",
        help="Step-by-step mode: wait for Enter before executing each chunk",
    )
    parser.add_argument(
        "--no-action",
        action="store_true",
        help="Suppress per-step action logging in SBS mode. Only effective with --sbs.",
    )
    parser.add_argument(
        "--reward-frame-interval",
        type=int,
        default=5,
        help="Capture a camera frame every N low-level steps during chunk_step "
        "for TOPReward scoring. 0 to disable. Default: 5 (6 frames per 30-step chunk).",
    )
    args = parser.parse_args()
    serve(
        args.config_path,
        args.port,
        args.max_message_size,
        dummy=args.dummy,
        verbose=args.verbose,
        step_by_step=args.sbs,
        no_action=args.no_action,
        reward_frame_interval=args.reward_frame_interval,
    )


if __name__ == "__main__":
    main()
