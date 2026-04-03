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

"""Unit tests for the SGLang-backed VLM planner paths."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    sys.modules["torch"] = torch_stub

import rlinf.workers.vlm_planner.vlm_planner_worker as vlm_planner_worker
from rlinf.workers.vlm_planner.vlm_planner_worker import VLMPlannerWorker


class _FakeProcessor:
    """Small processor stub that records local preprocessing calls."""

    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token="<eos>")
        self.template_calls = []
        self.processor_calls = []
        self.decode_calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "rendered prompt"

    def __call__(
        self,
        *,
        text,
        images,
        videos,
        padding,
        return_tensors,
        **video_kwargs,
    ):
        self.processor_calls.append(
            {
                "text": text,
                "images": images,
                "videos": videos,
                "padding": padding,
                "return_tensors": return_tensors,
                "video_kwargs": video_kwargs,
            }
        )
        return {
            "input_ids": np.asarray([[101, 102, 103]], dtype=np.int64),
            "pixel_values": np.asarray([[[[1.0]]]], dtype=np.float32),
            "image_grid_thw": np.asarray([[1, 1, 1]], dtype=np.int64),
        }

    def batch_decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        self.decode_calls.append(
            {
                "token_ids": token_ids,
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        return ["<think>hidden</think>move to the red block"]


def test_prepare_chat_request_uses_local_processor_preprocessing(monkeypatch):
    planner = VLMPlannerWorker.__new__(VLMPlannerWorker)
    planner._processor = _FakeProcessor()

    def fake_process_vision_info(messages, **_kwargs):
        assert messages[1]["role"] == "user"
        return (["processed-image"], None, {})

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=fake_process_vision_info),
    )

    messages = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": object()},
                {"type": "text", "text": "describe the scene"},
            ],
        },
    ]

    prepared = planner._prepare_chat_request(messages, add_generation_prompt=True)

    assert planner._processor.template_calls[0]["messages"] == messages
    assert planner._processor.processor_calls[0]["text"] == ["rendered prompt"]
    assert planner._processor.processor_calls[0]["images"] == ["processed-image"]
    assert prepared.input_ids == [101, 102, 103]
    assert prepared.image_data is not None
    assert prepared.image_data[0]["format"] == "processor_output"
    assert prepared.video_data is None


def test_generate_sglang_http_posts_preprocessed_generate_request():
    planner = VLMPlannerWorker.__new__(VLMPlannerWorker)
    planner._sglang_generate_path = "/generate"

    prepared = SimpleNamespace(
        input_ids=[11, 12, 13],
        image_data=[{"format": "processor_output", "pixel_values": [[1.0]]}],
        video_data=None,
    )
    planner._prepare_chat_request = lambda messages, add_generation_prompt: prepared

    captured = {}

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"text": "<think>hidden</think>pick up the towel corner"}

    planner._post_json = fake_post

    text = planner._generate_sglang_http(
        messages=[{"role": "user", "content": "x"}], max_new_tokens=32
    )

    assert captured["path"] == "/generate"
    assert captured["payload"]["input_ids"] == [11, 12, 13]
    assert captured["payload"]["image_data"] == prepared.image_data
    assert captured["payload"]["sampling_params"] == {
        "temperature": 0,
        "max_new_tokens": 32,
    }
    assert text == "pick up the towel corner"


def test_generate_sglang_http_decodes_output_ids_when_text_is_missing():
    planner = VLMPlannerWorker.__new__(VLMPlannerWorker)
    planner._sglang_generate_path = "/generate"
    planner._processor = _FakeProcessor()

    prepared = SimpleNamespace(
        input_ids=[11, 12, 13],
        image_data=[{"format": "processor_output", "pixel_values": [[1.0]]}],
        video_data=None,
    )
    planner._prepare_chat_request = lambda messages, add_generation_prompt: prepared
    planner._post_json = lambda path, payload: {"output_ids": [91, 92, 93]}

    text = planner._generate_sglang_http(
        messages=[{"role": "user", "content": "x"}], max_new_tokens=32
    )

    assert planner._processor.decode_calls[0]["token_ids"] == [[91, 92, 93]]
    assert text == "move to the red block"


def test_compute_top_reward_sglang_uses_input_side_logprob_request():
    planner = VLMPlannerWorker.__new__(VLMPlannerWorker)
    planner._backend = "sglang_http"
    planner._sglang_generate_path = "/generate"
    planner._top_reward_reward_scale = 2.0
    planner._top_reward_max_frames = 16

    prepared = SimpleNamespace(
        input_ids=[201, 202, 203],
        image_data=None,
        video_data=[{"format": "processor_output", "pixel_values_videos": [[1.0]]}],
    )
    planner._prepare_top_reward_request = lambda frames, instruction, fps: (
        prepared,
        203,
    )

    captured = {}

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"meta_info": {"input_token_ids_logprobs": [None, [[-0.75, 203, None]]]}}

    planner._post_json = fake_post

    score = planner._compute_top_reward_sglang(
        frames=[],
        instruction="fold the towel",
        fps=2.0,
    )

    assert captured["path"] == "/generate"
    assert captured["payload"]["input_ids"] == [201, 202, 203]
    assert captured["payload"]["video_data"] == prepared.video_data
    assert captured["payload"]["sampling_params"] == {
        "temperature": 0,
        "max_new_tokens": 1,
    }
    assert captured["payload"]["return_logprob"] is True
    assert captured["payload"]["logprob_start_len"] == 1
    assert captured["payload"]["token_ids_logprob"] == [203]
    assert score == pytest.approx(-1.5)


def test_prepare_top_reward_request_uses_supplied_instruction(monkeypatch):
    planner = VLMPlannerWorker.__new__(VLMPlannerWorker)
    planner._processor = _FakeProcessor()
    planner._top_reward_label = "True"
    debug_calls = []
    planner._logger = SimpleNamespace(
        debug=lambda msg, *args: debug_calls.append(msg % args)
    )

    def fake_process_vision_info(messages, **_kwargs):
        assert messages[0]["content"][1]["text"].startswith(
            "The above video shows a robot manipulation trajectory"
        )
        return (None, ["processed-video"], {"fps": 2.0})

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=fake_process_vision_info),
    )

    prepared, expected_token_id = planner._prepare_top_reward_request(
        [np.zeros((8, 8, 3), dtype=np.uint8)],
        "pick up the blue one",
        fps=2.0,
    )

    assert planner._processor.processor_calls[0]["text"] == [
        "rendered promptpick up the blue one Decide whether the above statement "
        "is True or not. The answer is: True"
    ]
    assert planner._processor.processor_calls[0]["videos"] == ["processed-video"]
    assert prepared.input_ids == [101, 102, 103]
    assert expected_token_id == 103
    assert debug_calls == [
        "[VLMPlannerWorker] TOPReward full prompt: rendered promptpick up the "
        "blue one Decide whether the above statement is True or not. "
        "The answer is: True"
    ]


def test_extract_top_reward_logprob_requires_input_side_data():
    planner = VLMPlannerWorker.__new__(VLMPlannerWorker)

    with pytest.raises(RuntimeError, match="input-side token_ids_logprob data"):
        planner._extract_top_reward_logprob(
            {"meta_info": {"input_token_ids_logprobs": [None]}},
            expected_token_id=203,
        )


def test_extract_top_reward_logprob_returns_requested_input_token():
    planner = VLMPlannerWorker.__new__(VLMPlannerWorker)

    score = planner._extract_top_reward_logprob(
        {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [[-1.25, 3007, None]],
                ]
            }
        },
        expected_token_id=3007,
    )

    assert score == pytest.approx(-1.25)


def test_removed_local_sglang_backend_requires_http_server(monkeypatch):
    monkeypatch.setattr(
        vlm_planner_worker,
        "get_logger",
        lambda: SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        ),
    )

    with pytest.raises(ValueError, match="removed"):
        VLMPlannerWorker({"vlm_planner": {"backend": "sglang"}})


def test_legacy_sglang_backend_alias_uses_http_when_server_url_is_set(monkeypatch):
    monkeypatch.setattr(
        vlm_planner_worker,
        "get_logger",
        lambda: SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setattr(
        VLMPlannerWorker,
        "_load_sglang_http_backend",
        lambda self: setattr(self, "_loaded_backend", "sglang_http"),
    )
    monkeypatch.setattr(
        VLMPlannerWorker,
        "_load_transformers_backend",
        lambda self: setattr(self, "_loaded_backend", "transformers"),
    )

    planner = VLMPlannerWorker(
        {
            "vlm_planner": {
                "backend": "sglang",
                "server_url": "http://127.0.0.1:30000",
            }
        }
    )

    assert planner._backend == "sglang_http"
    assert planner._loaded_backend == "sglang_http"
