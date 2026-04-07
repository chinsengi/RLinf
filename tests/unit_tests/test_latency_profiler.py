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

import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tools.latency_profiler.profiler import LatencyProfiler
from tools.latency_profiler.runtime_utils import (
    backfill_worker_durations,
    is_async_runtime,
)


def test_profiler_exports_and_summary(tmp_path):
    profiler = LatencyProfiler(mode="dummy")

    profiler.step_start(0)
    profiler.start("weight_sync")
    profiler.stop("weight_sync")
    profiler.start("actor_training")
    profiler.stop("actor_training")
    profiler.step_end()

    summary = profiler.summary()
    assert summary["weight_sync"]["count"] == 1
    assert summary["actor_training"]["count"] == 1
    assert summary["total_step"]["count"] == 1

    csv_path = tmp_path / "latency.csv"
    json_path = tmp_path / "latency.json"
    profiler.export_csv(str(csv_path))
    profiler.export_json(str(json_path))

    csv_text = csv_path.read_text()
    assert "weight_sync" in csv_text
    assert "actor_training" in csv_text

    json_data = json.loads(json_path.read_text())
    assert json_data["mode"] == "dummy"
    assert json_data["steps"][0]["step"] == 0
    assert "weight_sync" in json_data["steps"][0]["layers"]


def test_backfill_worker_durations_overwrites_latest_step():
    profiler = LatencyProfiler(mode="dummy")

    profiler.step_start(3)
    profiler.start("rollout_inference")
    profiler.stop("rollout_inference")
    profiler.start("env_execution")
    profiler.stop("env_execution")
    profiler.start("vlm_reward")
    profiler.stop("vlm_reward")
    profiler.step_end()

    backfill_worker_durations(
        profiler,
        step_id=3,
        env_time={"env_step": 1.5, "top_reward": 0.25},
        rollout_time={"generate": 0.75},
    )

    rec = profiler.records[-1]
    assert rec.layer_ms("env_execution") == 1500.0
    assert rec.layer_ms("vlm_reward") == 250.0
    assert rec.layer_ms("rollout_inference") == 750.0


def test_backfill_worker_durations_accepts_runtime_timer_aliases():
    profiler = LatencyProfiler(mode="dummy")

    profiler.step_start(4)
    profiler.start("rollout_inference")
    profiler.stop("rollout_inference")
    profiler.start("env_execution")
    profiler.stop("env_execution")
    profiler.start("recv_trajectory")
    profiler.stop("recv_trajectory")
    profiler.start("vlm_reward")
    profiler.stop("vlm_reward")
    profiler.step_end()

    backfill_worker_durations(
        profiler,
        step_id=4,
        env_time={"env_interact_step": 1.2, "top_reward": 0.4},
        rollout_time={"generate_one_epoch": 0.8},
        actor_time={"recv_trajectory": 0.15},
    )

    rec = profiler.records[-1]
    assert rec.layer_ms("env_execution") == 1200.0
    assert rec.layer_ms("vlm_reward") == 400.0
    assert rec.layer_ms("rollout_inference") == 800.0
    assert rec.layer_ms("recv_trajectory") == 150.0


def test_is_async_runtime_uses_loss_type():
    class AlgoCfg(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class Cfg:
        def __init__(self, loss_type):
            self.algorithm = AlgoCfg(loss_type=loss_type)

    assert is_async_runtime(Cfg("decoupled_actor_critic")) is True
    assert is_async_runtime(Cfg("actor_critic")) is False


def test_staged_runtime_helpers_import_from_shared_runtime():
    run_py = Path(__file__).resolve().parents[2] / "tools/latency_profiler/run.py"
    module = ast.parse(run_py.read_text())
    helper_fn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_staged_runtime_helpers"
    )
    helper_imports = {
        node.module for node in ast.walk(helper_fn) if isinstance(node, ast.ImportFrom)
    }

    assert "rlinf.runners.staged_embodied_runtime" in helper_imports
    assert "examples.embodiment.train_embodied_agent_staged" not in helper_imports
