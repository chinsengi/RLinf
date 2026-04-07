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

import sys
import types

import numpy as np

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    sys.modules["torch"] = torch_stub

from rlinf.workers.vlm_planner.vlm_planner_worker import (
    NEW_TASK_PREFIX,
    VLMPlannerWorker,
)


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def test_get_next_subtask_prompt_includes_main_task():
    worker = VLMPlannerWorker.__new__(VLMPlannerWorker)
    worker._logger = _FakeLogger()
    worker._max_new_tokens_subtask = 64
    worker._build_qwen_messages = lambda _system, _images, user_text: user_text
    worker._generate = lambda messages, _tokens: messages

    prompt = worker.get_next_subtask(
        images=[np.zeros((8, 8, 3), dtype=np.uint8)],
        main_task="fold the towel",
    )

    assert "The overall episode goal is: fold the towel" in prompt
    assert "single best next subtask" in prompt


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


def test_get_next_subtask_returns_vlm_response_as_subtask():
    """VLM response is returned as-is, including creative task proposals."""
    worker = VLMPlannerWorker.__new__(VLMPlannerWorker)
    worker._logger = _FakeLogger()
    worker._max_new_tokens_subtask = 64
    worker._build_qwen_messages = lambda _system, _images, user_text: user_text
    worker._generate = lambda _messages, _tokens: "grasp the left corner"

    result = worker.get_next_subtask(
        images=[np.zeros((8, 8, 3), dtype=np.uint8)],
        main_task="fold the towel",
    )
    assert result == "grasp the left corner"


def test_new_task_prefix_returned_when_vlm_signals_goal_achieved():
    """VLM 'NEW TASK: ...' response is normalized to the canonical prefix."""
    for vlm_reply in (
        "NEW TASK: tidy up the scattered items",
        "new task: tidy up the scattered items",
        "New Task:  tidy up the scattered items",
    ):
        worker = VLMPlannerWorker.__new__(VLMPlannerWorker)
        worker._logger = _FakeLogger()
        worker._max_new_tokens_subtask = 64
        worker._build_qwen_messages = lambda _s, _i, _u: _u
        worker._generate = lambda _m, _t, _r=vlm_reply: _r

        result = worker.get_next_subtask(
            images=[np.zeros((8, 8, 3), dtype=np.uint8)],
            main_task="fold the towel",
        )
        assert result.startswith(NEW_TASK_PREFIX), (
            f"Expected NEW_TASK_PREFIX for '{vlm_reply}', got '{result}'"
        )
        assert "tidy up the scattered items" in result


def test_normal_subtask_not_tagged_as_new_task():
    """A regular subtask that doesn't start with 'NEW TASK:' passes through."""
    worker = VLMPlannerWorker.__new__(VLMPlannerWorker)
    worker._logger = _FakeLogger()
    worker._max_new_tokens_subtask = 64
    worker._build_qwen_messages = lambda _s, _i, _u: _u
    worker._generate = lambda _m, _t: "grasp the left corner"

    result = worker.get_next_subtask(
        images=[np.zeros((8, 8, 3), dtype=np.uint8)],
        main_task="fold the towel",
    )
    assert not result.startswith(NEW_TASK_PREFIX)
    assert result == "grasp the left corner"
