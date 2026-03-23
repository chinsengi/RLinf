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

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request

import numpy as np
from PIL import Image


@dataclass(slots=True)
class MarlImage:
    """One image payload sent to the marl ingress endpoint."""

    camera_name: str
    image: np.ndarray
    capture_ts_ms: int | None = None


class MarlClient:
    """Small synchronous client for the local marl HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 30.0,
        image_format: str = "jpeg",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        normalized_format = image_format.strip().lower()
        if normalized_format not in {"jpeg", "jpg", "png"}:
            raise ValueError(
                "marl image_format must be one of: jpeg, jpg, png. "
                f"Got '{image_format}'."
            )
        self.image_format = "jpeg" if normalized_format == "jpg" else normalized_format

    def healthz(self) -> dict[str, Any]:
        """Probe the marl service health endpoint."""
        return self._request_json("GET", "/healthz")

    def create_image_set(
        self,
        *,
        run_id: str,
        episode_id: str,
        step_id: int,
        task_description: str | None,
        images: list[MarlImage],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload one logical image set for a rollout step."""
        payload = {
            "run_id": run_id,
            "episode_id": episode_id,
            "step_id": int(step_id),
            "task_description": task_description,
            "metadata": metadata or {},
            "images": [self._encode_image(image) for image in images],
        }
        return self._request_json("POST", "/image-sets", payload)

    def plan(
        self,
        *,
        run_id: str,
        episode_id: str,
        step_id: int,
        memory_text: str = "",
    ) -> dict[str, Any]:
        """Request the next subtask from marl."""
        payload = {
            "run_id": run_id,
            "episode_id": episode_id,
            "step_id": int(step_id),
            "memory_text": memory_text,
        }
        return self._request_json("POST", "/plan", payload)

    def score_topreward(
        self,
        *,
        run_id: str,
        episode_id: str,
        end_step_id: int,
        instruction: str,
        start_step_id: int | None = None,
        max_frames: int | None = None,
        camera_name: str | None = None,
        fps: float | None = None,
    ) -> dict[str, Any]:
        """Request one absolute TOPReward score for the trajectory prefix."""
        payload = {
            "run_id": run_id,
            "episode_id": episode_id,
            "end_step_id": int(end_step_id),
            "instruction": instruction,
            "start_step_id": start_step_id,
            "max_frames": max_frames,
            "camera_name": camera_name,
            "fps": fps,
        }
        return self._request_json("POST", "/topreward", payload)

    def _encode_image(self, image: MarlImage) -> dict[str, Any]:
        frame = np.asarray(image.image)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"marl expects RGB images shaped (H, W, 3), got {frame.shape}."
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(frame, mode="RGB")
        buffer = io.BytesIO()
        if self.image_format == "png":
            pil_image.save(buffer, format="PNG")
            content_type = "image/png"
        else:
            pil_image.save(buffer, format="JPEG", quality=95)
            content_type = "image/jpeg"

        return {
            "camera_name": image.camera_name,
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "content_type": content_type,
            "capture_ts_ms": image.capture_ts_ms,
        }

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                raw_body = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"marl request {method} {path} failed with status {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"marl request {method} {path} failed: {exc.reason}"
            ) from exc

        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))
