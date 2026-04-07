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

"""VLM Planner Worker for slot-indexed embodied RL.

This Ray actor runs on the Beaker GPU node and exposes three methods:

1. **Subtask generation** ``get_next_subtask()`` - given recent robot
   observations (images) and the episode-level main task, generates the next
   subtask instruction (e.g. "pick up the red block").
   Called by ``EnvWorker._maybe_update_subtask()`` when ``subtask_interval > 0``.

2. **Subtask reward evaluation** ``evaluate_subtask()`` - given the same
   context plus the completed subtask description, decides whether the subtask
   succeeded (1.0) or failed (0.0).
   **Not currently called** by EnvWorker in the YAM pipeline; available for
   future use.

3. **TOPReward scoring** ``compute_top_reward()`` - given accumulated
   trajectory frames and the current task instruction, returns
   log P("True" | frames, instruction) as a dense progress reward signal.
   Called by ``EnvWorker._compute_top_reward()`` every chunk step when
   ``dense_reward_method: top_reward`` (both YAM training configs).

The worker either loads a local Qwen3-VL vision-language model or performs
local Qwen-VL preprocessing before calling an external SGLang server. In staged
embodied training it is placed on a dedicated GPU node.

Architecture (active call paths for YAM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    EnvWorker (YAM node)
        │  every subtask_interval steps          (subtask_interval > 0 only)
        │  images + main_task ───────────────▶  VLMPlannerWorker (Beaker)
        │                                           │
        │  ◀── new subtask text ─────────────────   │  get_next_subtask()
        │                                           │
        │  every chunk step (dense_reward_method):   │
        │  episode frames + instruction ──────────▶ │  compute_top_reward()
        │  ◀── log P("True" | frames, instr) ─────  │

Configuration (under ``vlm_planner`` in the top-level YAML):
    model_path: str
        HuggingFace model ID or local path, e.g. "Qwen/Qwen3-VL-8B-Instruct".
        Must be a vision-language model — image inputs are required for
        subtask planning and TOPReward scoring.
    backend: str
        Inference backend:
        - ``"transformers"`` (default): load the VLM directly in this worker.
        - ``"sglang_http"``: call an external SGLang HTTP server running in a
          separate Python environment / process.
    server_url: str
        Base URL of the external SGLang server when ``backend: sglang_http``.
    request_timeout: float
        Timeout in seconds for SGLang HTTP requests.
    dtype: str
        Torch dtype string: "bfloat16" (default), "float16", or "float32".
    max_new_tokens_subtask: int
        Maximum tokens to generate for subtask instructions (default: 64).
    max_new_tokens_reward: int
        Maximum tokens to generate for reward verdicts (default: 16).
    success_threshold: float
        Confidence threshold [0, 1] above which the VLM vote counts as success.
    dense_reward_method: str
        Dense reward method name (default: ``"none"``).  Set to
        ``"top_reward"`` to enable TOPReward scoring via log P("True" |
        frames, instruction).  Should match ``env.train.dense_reward_method``.
    top_reward_max_frames: int
        Maximum trajectory frames to pass to the TOPReward VLM (default: 1000).
        Older frames are dropped when the buffer exceeds this limit.

Example YAML::

    vlm_planner:
      model_path: "Qwen/Qwen3-VL-8B-Instruct"
      backend: "transformers"
      dtype: "bfloat16"
      max_new_tokens_subtask: 64
      max_new_tokens_reward: 16
      success_threshold: 0.5
      dense_reward_method: top_reward
      top_reward_max_frames: 1000
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from omegaconf import DictConfig
from PIL import Image

from rlinf.utils.logging import get_logger

NEW_TASK_PREFIX = "NEW_TASK:"
"""Prefix returned by :meth:`get_next_subtask` when the VLM judges the
current goal achieved and proposes a new one.

The :class:`~rlinf.workers.env.vlm_planner_client.VLMPlannerClient` detects
this prefix, strips it, and rotates ``_initial_task_descriptions`` so that
subsequent planning calls use the creative task as the new main goal.
"""

_SUBTASK_SYSTEM_PROMPT = """\
You are an AI assistant controlling a bimanual robot arm. \
You will be shown images from the robot's cameras and the overall episode goal. \
Your job is to identify the single most appropriate next subtask for the robot to execute. \
If the overall episode goal has NOT been fully achieved yet, reply with ONLY the subtask \
instruction as a short imperative sentence (5-15 words). \
If the overall episode goal appears to be fully achieved based on the current observation, \
reply with "NEW TASK:" followed by a creative new task the robot can attempt next, \
given what you see in the scene (e.g. tidying up, rearranging objects, \
or practicing a related manipulation skill). \
Do not add any other explanation or formatting."""

_REWARD_SYSTEM_PROMPT = """\
You are an AI evaluator for a bimanual robot arm. \
You will be shown images from the robot's cameras and a description of the subtask that was attempted. \
Decide whether the robot successfully completed the subtask. \
Reply with ONLY "success" or "failure" — no other text."""
_TOPREWARD_PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory that completes "
    "the following task: "
)

_THINK_CLOSE_TAG = "</think>"


@dataclass(slots=True)
class _PreparedSGLangRequest:
    """One preprocessed SGLang request ready for token-in generation."""

    input_ids: list[int]
    image_data: Optional[list[dict[str, Any]]] = None
    video_data: Optional[list[dict[str, Any]]] = None


def _strip_thinking_output(text: str) -> str:
    """Return the final assistant answer after an optional reasoning block."""
    stripped = text.strip()
    if _THINK_CLOSE_TAG in stripped:
        return stripped.split(_THINK_CLOSE_TAG, 1)[1].strip()
    return stripped


def _process_vision_info_compat(process_vision_info, messages):
    """Handle qwen_vl_utils versions that return either 2 or 3 values."""
    video_kwargs = {}
    try:
        processed = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except TypeError:
        try:
            processed = process_vision_info(messages, return_video_kwargs=True)
        except TypeError:
            processed = process_vision_info(messages)

    if len(processed) == 3:
        image_inputs, video_inputs, video_kwargs = processed
    elif len(processed) == 2:
        image_inputs, video_inputs = processed
    else:
        raise ValueError(
            "process_vision_info returned an unexpected number of values: "
            f"{len(processed)}"
        )

    if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 0:
        video_kwargs.pop("fps", None)
    elif isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
        video_kwargs["fps"] = video_kwargs["fps"][0]

    if (
        video_inputs
        and isinstance(video_inputs[0], tuple)
        and len(video_inputs[0]) == 2
    ):
        videos = []
        video_metadata = []
        for video, metadata in video_inputs:
            videos.append(video)
            video_metadata.append(metadata)
        video_inputs = videos
        video_kwargs["video_metadata"] = video_metadata
        video_kwargs.pop("fps", None)

    return image_inputs, video_inputs, video_kwargs


class VLMPlannerWorker:
    """Ray actor class that hosts a local Qwen VLM for subtask planning and reward evaluation.

    Placement is controlled by the caller. In staged embodied training, the
    worker is allocated through RLinf's placement stack so the selected GPU is
    reserved and isolated consistently with actor and rollout workers.

    Args:
        cfg: Top-level Hydra config.  The worker reads ``cfg.vlm_planner``.
    """

    def __init__(self, cfg: DictConfig):
        self._logger = get_logger()
        planner_cfg = cfg.get("vlm_planner", {})

        # Backward-compat deprecation warnings for old Anthropic config fields.
        if "model" in planner_cfg:
            warnings.warn(
                "[VLMPlannerWorker] Config field 'vlm_planner.model' is deprecated. "
                "Use 'vlm_planner.model_path' instead (e.g. 'Qwen/Qwen3-VL-8B-Instruct').",
                DeprecationWarning,
                stacklevel=2,
            )
        if "api_key_env" in planner_cfg:
            warnings.warn(
                "[VLMPlannerWorker] Config field 'vlm_planner.api_key_env' is deprecated. "
                "The worker now uses a local Qwen model and requires no API key.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._model_path: str = planner_cfg.get(
            "model_path", "Qwen/Qwen3-VL-8B-Instruct"
        )
        self._backend: str = str(planner_cfg.get("backend", "transformers")).lower()
        self._dtype_str: str = planner_cfg.get("dtype", "bfloat16")
        self._max_new_tokens_subtask: int = int(
            planner_cfg.get("max_new_tokens_subtask", 64)
        )
        self._max_new_tokens_reward: int = int(
            planner_cfg.get("max_new_tokens_reward", 16)
        )
        self._success_threshold: float = float(
            planner_cfg.get("success_threshold", 0.5)
        )

        # Dense reward configuration
        self._dense_reward_method: str = str(
            planner_cfg.get("dense_reward_method", "none")
        )
        self._top_reward_max_frames: int = int(
            planner_cfg.get("top_reward_max_frames", 1000)
        )
        self._top_reward_reward_scale: float = float(
            planner_cfg.get("reward_scale", 1.0)
        )
        self._top_reward_label: str = str(planner_cfg.get("top_reward_label", "True"))
        self._transformers_runtime_path: Optional[str] = planner_cfg.get(
            "transformers_runtime_path", None
        )
        self._sglang_server_url: str = str(planner_cfg.get("server_url", "")).strip()
        self._sglang_request_timeout: float = float(
            planner_cfg.get("request_timeout", 120.0)
        )
        self._sglang_generate_path: str = str(
            planner_cfg.get("generate_path", "/generate")
        )
        self._sglang_api_key: Optional[str] = planner_cfg.get("api_key", None)

        if self._backend in {"sglang", "sglang_local", "sglang_server"}:
            if not self._sglang_server_url:
                raise ValueError(
                    "Local in-process SGLang backends ('sglang', "
                    "'sglang_local', 'sglang_server') were removed. "
                    "Use backend='sglang_http' and set vlm_planner.server_url."
                )
            self._logger.warning(
                "[VLMPlannerWorker] backend='%s' is deprecated. "
                "Using backend='sglang_http' because server_url is set.",
                self._backend,
            )
            self._backend = "sglang_http"
            self._load_sglang_http_backend()
        elif self._backend == "sglang_http":
            self._load_sglang_http_backend()
        else:
            self._backend = "transformers"
            self._load_transformers_backend()

        if self._dense_reward_method == "top_reward":
            if self._backend == "transformers":
                from rlinf.algorithms.rewards.top_reward import TOPReward

                self._top_reward = TOPReward(
                    planner_cfg, model=self._model, processor=self._processor
                )
            self._logger.info("[VLMPlannerWorker] TOPReward enabled.")

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_transformers_backend(self) -> None:
        """Load Qwen model via HuggingFace Transformers."""
        import torch
        from transformers import AutoModelForImageTextToText

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

        self._logger.info(
            f"[VLMPlannerWorker] Loading '{self._model_path}' "
            f"with dtype={self._dtype_str} via transformers."
        )
        self._load_qwen_processor()
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self._model.eval()
        self._logger.info("[VLMPlannerWorker] Model loaded.")

    def _activate_transformers_runtime_path(self) -> None:
        """Optionally prepend an isolated Transformers runtime for Qwen3-VL."""
        runtime_path = self._transformers_runtime_path

        if not runtime_path:
            return

        runtime_dir = Path(runtime_path)
        if not runtime_dir.is_absolute():
            runtime_dir = (Path(__file__).resolve().parents[3] / runtime_dir).resolve()
        if not runtime_dir.exists():
            self._logger.warning(
                "[VLMPlannerWorker] transformers_runtime_path '%s' does not exist.",
                runtime_dir,
            )
            return

        runtime_str = str(runtime_dir)
        if runtime_str not in sys.path:
            sys.path.insert(0, runtime_str)
            os.environ["PYTHONPATH"] = (
                runtime_str
                if not os.environ.get("PYTHONPATH")
                else f"{runtime_str}:{os.environ['PYTHONPATH']}"
            )
            self._logger.info(
                "[VLMPlannerWorker] Prepending isolated Transformers runtime: %s",
                runtime_str,
            )

    def _load_sglang_http_backend(self) -> None:
        """Use an externally managed SGLang HTTP server."""
        if not self._sglang_server_url:
            raise ValueError(
                "Backend 'sglang_http' requires vlm_planner.server_url, e.g. "
                "'http://127.0.0.1:30000'."
            )
        self._load_qwen_processor()
        self._sglang_server_url = self._sglang_server_url.rstrip("/")
        self._logger.info(
            "[VLMPlannerWorker] Using external SGLang server at '%s'.",
            self._sglang_server_url,
        )

    def _load_qwen_processor(self) -> None:
        """Load the Qwen multimodal processor used for local preprocessing."""
        self._activate_transformers_runtime_path()

        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            trust_remote_code=True,
        )
        if not (
            hasattr(self._processor, "tokenizer")
            and hasattr(self._processor, "image_processor")
            and hasattr(self._processor, "video_processor")
        ):
            raise RuntimeError(
                f"[VLMPlannerWorker] '{self._model_path}' did not expose a "
                "multimodal processor in the active transformers runtime. "
                "Install a Qwen3-VL-capable transformers build or set "
                "'vlm_planner.transformers_runtime_path' to one."
            )

    # ------------------------------------------------------------------
    # Subtask generation
    # ------------------------------------------------------------------

    def get_next_subtask(
        self,
        images: list[np.ndarray],
        main_task: str = "",
    ) -> str:
        """Generate the next subtask instruction from observations.

        Args:
            images: List of uint8 RGB images (H, W, 3) from robot cameras.
            main_task: The episode-level goal (e.g. "fold the towel").
                Required for meaningful subtask planning.

        Returns:
            Subtask instruction string, e.g. ``"pick up the red block"``.
            If the VLM judges the current goal achieved it proposes a new
            task, returned as ``"NEW_TASK: <creative task>"``.

        Raises:
            ValueError: If *main_task* is empty.
        """
        if not main_task or not main_task.strip():
            raise ValueError(
                "get_next_subtask() requires a non-empty main_task. "
                "Set env.train.task_description to a concrete episode goal."
            )

        user_text = (
            f"The overall episode goal is: {main_task}\n\n"
            "Given the current observation, what is the single best next "
            "subtask for the robot to execute?"
        )
        messages = self._build_qwen_messages(_SUBTASK_SYSTEM_PROMPT, images, user_text)

        try:
            subtask = self._generate(messages, self._max_new_tokens_subtask)
        except Exception as exc:
            self._logger.warning(
                f"[VLMPlannerWorker] get_next_subtask failed: {exc}. "
                "Returning empty subtask."
            )
            subtask = ""

        stripped = subtask.strip()
        if stripped.upper().startswith("NEW TASK:"):
            creative_task = stripped.split(":", 1)[1].strip()
            if creative_task:
                self._logger.info(
                    f"[VLMPlannerWorker] Goal achieved, new task: '{creative_task}'"
                )
                return f"{NEW_TASK_PREFIX} {creative_task}"

        self._logger.info(f"[VLMPlannerWorker] Next subtask: '{subtask}'")
        return subtask

    # ------------------------------------------------------------------
    # Subtask reward evaluation
    # ------------------------------------------------------------------

    def evaluate_subtask(
        self,
        images: list[np.ndarray],
        subtask: str,
    ) -> float:
        """Evaluate whether a subtask was completed, returning a binary reward.

        Args:
            images: List of uint8 RGB images from robot cameras (post-subtask).
            subtask: The subtask instruction that was attempted.

        Returns:
            1.0 if the subtask was completed, 0.0 otherwise.
        """
        user_text = (
            f'Subtask attempted: "{subtask}"\n\n'
            "Did the robot successfully complete this subtask?"
        )
        messages = self._build_qwen_messages(_REWARD_SYSTEM_PROMPT, images, user_text)

        try:
            verdict = self._generate(messages, self._max_new_tokens_reward).lower()
            reward = 1.0 if "success" in verdict else 0.0
        except Exception as exc:
            self._logger.warning(
                f"[VLMPlannerWorker] evaluate_subtask failed: {exc}. "
                "Returning 0.0 reward."
            )
            reward = 0.0

        self._logger.info(f"[VLMPlannerWorker] Subtask '{subtask}' → reward={reward}")
        return reward

    # ------------------------------------------------------------------
    # TOPReward: dense progress reward via True-token log-probability
    # ------------------------------------------------------------------

    def compute_top_reward(
        self,
        frames: list[np.ndarray],
        instruction: str,
        reduction: str = "mean",
        fps: float = 2.0,
    ) -> float:
        """Compute a TOPReward progress score for the given trajectory frames.

        Delegates to :class:`rlinf.algorithms.rewards.top_reward.TOPReward`
        for the actual scoring logic (prompt construction, label masking,
        log-prob extraction).

        Args:
            frames: List of uint8 RGB images ``(H, W, 3)`` representing the
                trajectory so far (most recent last).
            instruction: Task description string.
            reduction: How to aggregate per-token log-probs: ``"mean"`` or
                ``"sum"``.
            fps: Frames per second metadata for video input.

        Returns:
            Log-probability of the final "True" token given the video frames
            and instruction (a float, typically negative).  All preceding
            tokens are masked; only the last token of the sequence is scored.
        """
        if self._dense_reward_method != "top_reward":
            return 0.0

        if self._backend == "sglang_http":
            return float(
                self._compute_top_reward_sglang(
                    frames,
                    instruction,
                    reduction=reduction,
                    fps=fps,
                )
            )

        # Trim frames to the configured maximum.
        if len(frames) > self._top_reward_max_frames:
            frames = frames[-self._top_reward_max_frames :]

        score = self._top_reward.compute_score(
            frames, instruction, reduction=reduction, fps=fps
        )
        return float(score)

    # ------------------------------------------------------------------
    # Inference dispatch
    # ------------------------------------------------------------------

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        """Dispatch to the active backend and return a stripped response string.

        Args:
            messages: ChatML message list (system + user with images).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Stripped generated text string.
        """
        if self._backend == "sglang_http":
            return self._generate_sglang_http(messages, max_new_tokens)
        return self._generate_transformers(messages, max_new_tokens)

    def _generate_transformers(self, messages: list[dict], max_new_tokens: int) -> str:
        """Run inference using HuggingFace Transformers.

        Args:
            messages: ChatML message list.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Stripped generated text string.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = _process_vision_info_compat(
            process_vision_info, messages
        )
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        ).to(next(self._model.parameters()).device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )

        # Trim the prompt prefix from each sequence before decoding.
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = [seq[prompt_len:] for seq in generated_ids]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _generate_sglang_http(self, messages: list[dict], max_new_tokens: int) -> str:
        """Run inference through an external SGLang HTTP /generate endpoint."""
        prepared = self._prepare_chat_request(
            messages,
            add_generation_prompt=True,
        )
        payload = {
            "input_ids": prepared.input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        }
        if prepared.image_data is not None:
            payload["image_data"] = prepared.image_data
        if prepared.video_data is not None:
            payload["video_data"] = prepared.video_data

        response = self._post_json(self._sglang_generate_path, payload)
        return self._extract_generation_text(response)

    def _compute_top_reward_sglang(
        self,
        frames: list[np.ndarray],
        instruction: str,
        reduction: str = "mean",
        fps: float = 2.0,
    ) -> float:
        """Compute TOPReward through SGLang generation with prompt-side logprobs."""
        del reduction

        if len(frames) > self._top_reward_max_frames:
            frames = frames[-self._top_reward_max_frames :]

        prepared, expected_token_id = self._prepare_top_reward_request(
            frames,
            instruction,
            fps=fps,
        )
        payload: dict[str, Any] = {
            "input_ids": prepared.input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0,
            },
            "return_logprob": True,
            "logprob_start_len": max(len(prepared.input_ids) - 2, 0),
            "token_ids_logprob": [expected_token_id],
        }
        if prepared.image_data is not None:
            payload["image_data"] = prepared.image_data
        if prepared.video_data is not None:
            payload["video_data"] = prepared.video_data

        response = self._post_json(self._sglang_generate_path, payload)
        score = self._extract_top_reward_logprob(
            response,
            expected_token_id=expected_token_id,
        )
        return float(score) * self._top_reward_reward_scale

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_generation_text(self, response: dict[str, Any]) -> str:
        """Read text from SGLang response or decode output_ids locally."""
        text = response.get("text", None)
        if text is not None:
            if isinstance(text, list):
                if not text:
                    raise RuntimeError(
                        "[VLMPlannerWorker] SGLang /generate returned an empty text list."
                    )
                text = text[0]
            return _strip_thinking_output(str(text))

        output_ids = response.get("output_ids", None)
        if output_ids is None:
            raise RuntimeError(
                "[VLMPlannerWorker] SGLang /generate returned neither text nor output_ids."
            )
        if output_ids and isinstance(output_ids[0], list):
            output_ids = output_ids[0]

        decoded = self._processor.batch_decode(
            [output_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return _strip_thinking_output(str(decoded).strip())

    @staticmethod
    def _to_pil_image(image: np.ndarray) -> Image.Image:
        """Convert a uint8 RGB numpy array to a PIL Image.

        Args:
            image: uint8 ndarray of shape (H, W, 3).

        Returns:
            PIL Image in RGB mode.
        """
        return Image.fromarray(image.astype(np.uint8))

    @staticmethod
    def _normalize_http_path(base_url: str, path: str) -> str:
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the configured SGLang server and parse the response."""
        request = urllib_request.Request(
            self._normalize_http_path(self._sglang_server_url, path),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_http_headers(),
            method="POST",
        )
        try:
            with urllib_request.urlopen(
                request, timeout=self._sglang_request_timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"[VLMPlannerWorker] SGLang HTTP {path} failed with "
                f"status={exc.code}: {body}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(
                f"[VLMPlannerWorker] Failed to reach SGLang server at "
                f"'{self._sglang_server_url}': {exc}"
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"[VLMPlannerWorker] SGLang HTTP {path} returned invalid JSON: {body}"
            ) from exc

        if isinstance(parsed, dict) and "error" in parsed:
            raise RuntimeError(
                f"[VLMPlannerWorker] SGLang HTTP {path} returned error: "
                f"{parsed['error']}"
            )
        return parsed

    def _build_http_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._sglang_api_key:
            headers["Authorization"] = f"Bearer {self._sglang_api_key}"
        return headers

    def _prepare_chat_request(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool,
    ) -> _PreparedSGLangRequest:
        """Render ChatML + vision inputs locally and build a token-in request."""
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        image_inputs, video_inputs, video_kwargs = _process_vision_info_compat(
            process_vision_info, messages
        )
        processor_inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        return self._prepare_processor_output_request(processor_inputs)

    def _prepare_top_reward_request(
        self,
        frames: list[np.ndarray],
        instruction: str,
        *,
        fps: float,
    ) -> tuple[_PreparedSGLangRequest, int]:
        """Build the strict TOPReward prompt and resolve the final target token."""
        from qwen_vl_utils import process_vision_info

        pil_frames = [self._to_pil_image(frame) for frame in frames]
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "fps": fps},
                    {"type": "text", "text": _TOPREWARD_PROMPT_PREFIX},
                ],
            }
        ]
        prompt_chat = self._processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        eos_token = self._processor.tokenizer.eos_token
        if eos_token is not None:
            prompt_chat = prompt_chat.split(eos_token)[0]
        instruction_suffix = (
            f"{instruction} Decide whether the above statement is True or not. "
            f"The answer is: {self._top_reward_label}"
        )
        image_inputs, video_inputs, video_kwargs = _process_vision_info_compat(
            process_vision_info, user_messages
        )
        processor_inputs = self._processor(
            text=[f"{prompt_chat}{instruction_suffix}"],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        prepared = self._prepare_processor_output_request(processor_inputs)
        return prepared, int(prepared.input_ids[-1])

    def _prepare_processor_output_request(
        self, processor_inputs
    ) -> _PreparedSGLangRequest:
        """Convert HF processor output into a JSON-safe SGLang request payload."""
        payload: dict[str, Any] = {"format": "processor_output"}
        for key, value in processor_inputs.items():
            payload[key] = self._to_json_compatible(value)

        input_ids = payload["input_ids"]
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        has_video = any(
            key in payload
            for key in (
                "pixel_values_videos",
                "video_grid_thw",
                "second_per_grid_ts",
                "video_second_per_grid",
            )
        )
        has_image = any(
            key in payload
            for key in (
                "pixel_values",
                "image_grid_thw",
            )
        )

        return _PreparedSGLangRequest(
            input_ids=list(input_ids),
            image_data=[payload] if has_image and not has_video else None,
            video_data=[payload] if has_video else None,
        )

    @staticmethod
    def _to_json_compatible(value: Any) -> Any:
        """Convert tensors / numpy objects into plain Python containers."""
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                key: VLMPlannerWorker._to_json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [VLMPlannerWorker._to_json_compatible(item) for item in value]
        return value

    def _extract_top_reward_logprob(
        self,
        response: dict[str, Any],
        *,
        expected_token_id: int,
    ) -> float:
        """Extract the requested input-side token logprob from /generate output."""
        meta_info = response.get("meta_info", {})
        scored_positions = meta_info.get("input_token_ids_logprobs") or []
        score_entry = next(
            (position for position in reversed(scored_positions) if position),
            None,
        )
        if score_entry is None:
            raise RuntimeError(
                "[VLMPlannerWorker] SGLang did not return input-side "
                "token_ids_logprob data for TOPReward."
            )

        matching_entry = next(
            (
                entry
                for entry in score_entry
                if len(entry) >= 2 and int(entry[1]) == expected_token_id
            ),
            None,
        )
        if matching_entry is None:
            raise RuntimeError(
                "[VLMPlannerWorker] SGLang TOPReward response did not include "
                f"the requested final token id {expected_token_id}."
            )

        return float(matching_entry[0])

    def _build_qwen_messages(
        self,
        system_prompt: str,
        images: list[np.ndarray],
        user_text: str,
    ) -> list[dict]:
        """Build a ChatML message list for Qwen multimodal inference.

        Args:
            system_prompt: System instruction string.
            images: List of uint8 RGB numpy arrays from robot cameras.
            user_text: User-facing text prompt (history + question).

        Returns:
            ChatML list with system and user roles; user content interleaves
            image dicts and a trailing text dict.
        """
        user_content = []
        for img in images:
            if img is None:
                continue
            user_content.append(
                {
                    "type": "image",
                    "image": self._to_pil_image(np.asarray(img, dtype=np.uint8)),
                }
            )
        user_content.append({"type": "text", "text": user_text})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
