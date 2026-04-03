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

import numpy as np

try:
    import torch  # noqa: F401
except Exception:
    import sys
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.Tensor = object
    sys.modules["torch"] = torch_stub

from rlinf.workers.vlm_planner.vlm_planner_worker import VLMPlannerWorker


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def test_get_next_subtask_prompt_includes_main_task():
    worker = VLMPlannerWorker.__new__(VLMPlannerWorker)
    worker._logger = _FakeLogger()
    worker._max_new_tokens_subtask = 64
    captured = {}

    def fake_build_qwen_messages(_system, _images, user_text):
        captured["user_text"] = user_text
        return user_text

    worker._build_qwen_messages = fake_build_qwen_messages
    worker._generate = lambda _messages, _tokens: "grasp the towel corner"

    _ = worker.get_next_subtask(
        images=[np.zeros((8, 8, 3), dtype=np.uint8)],
        main_task="fold the towel",
    )

    prompt = captured["user_text"]
    assert "The overall episode goal is: fold the towel" in prompt
    assert "single best next subtask" in prompt
    assert "If a grasp is still missing" in prompt
    assert "align or center the object with the target" in prompt
    assert "put down or release the object" in prompt
    assert "left or right side" in prompt
    assert "left gripper" in prompt
    assert "do not invent a left/right assignment" in prompt


def test_get_next_subtask_prompt_includes_current_subtask_context():
    worker = VLMPlannerWorker.__new__(VLMPlannerWorker)
    worker._logger = _FakeLogger()
    worker._max_new_tokens_subtask = 64
    captured = {}

    def fake_build_qwen_messages(_system, _images, user_text):
        captured["user_text"] = user_text
        return user_text

    worker._build_qwen_messages = fake_build_qwen_messages
    worker._generate = lambda _messages, _tokens: "align the can to the target"

    _ = worker.get_next_subtask(
        images=[np.zeros((8, 8, 3), dtype=np.uint8)],
        main_task="put the can in the middle of the black tape",
        current_subtask="move the can toward the black tape",
    )

    prompt = captured["user_text"]
    assert "Current active subtask: move the can toward the black tape" in prompt
    assert "If the current subtask already names a gripper" in prompt
    assert "Do not skip prerequisite stages" in prompt


def test_get_next_subtask_normalizes_bulleted_output():
    worker = VLMPlannerWorker.__new__(VLMPlannerWorker)
    worker._logger = _FakeLogger()
    worker._max_new_tokens_subtask = 64
    worker._build_qwen_messages = lambda _system, _images, user_text: user_text
    worker._generate = (
        lambda _messages,
        _tokens: '1. "align the can to the middle of the black tape"\n'
    )

    subtask = worker.get_next_subtask(
        images=[np.zeros((8, 8, 3), dtype=np.uint8)],
        main_task="put the can in the middle of the black tape",
    )

    assert subtask == "align the can to the middle of the black tape"


def test_get_next_subtask_requires_main_task():
    worker = VLMPlannerWorker.__new__(VLMPlannerWorker)
    worker._logger = _FakeLogger()
    worker._max_new_tokens_subtask = 64
    worker._build_qwen_messages = lambda _system, _images, user_text: user_text
    worker._generate = lambda messages, _tokens: messages

    try:
        worker.get_next_subtask(
            images=[np.zeros((8, 8, 3), dtype=np.uint8)],
            main_task="  ",
        )
    except ValueError as exc:
        assert "non-empty main_task" in str(exc)
    else:
        raise AssertionError("Expected get_next_subtask() to require main_task.")
