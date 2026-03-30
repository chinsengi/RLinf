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

import importlib.util
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "embodiment"
    / "train_embodied_agent_staged.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "train_embodied_agent_staged", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_is_expected_remote_disconnect_error = _MODULE._is_expected_remote_disconnect_error
_start_remote_disconnect_monitor = _MODULE._start_remote_disconnect_monitor
_SYNC_FORCED_ASYNC_RUNTIME = _MODULE._FORCED_ASYNC_RUNTIME

_ASYNC_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "embodiment"
    / "train_embodied_agent_staged_async.py"
)
_ASYNC_SPEC = importlib.util.spec_from_file_location(
    "train_embodied_agent_staged_async", _ASYNC_SCRIPT_PATH
)
assert _ASYNC_SPEC is not None and _ASYNC_SPEC.loader is not None
_ASYNC_MODULE = importlib.util.module_from_spec(_ASYNC_SPEC)
_ASYNC_SPEC.loader.exec_module(_ASYNC_MODULE)
_ASYNC_FORCED_ASYNC_RUNTIME = _ASYNC_MODULE._FORCED_ASYNC_RUNTIME
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


def test_sync_staged_entrypoint_forces_sync_runtime() -> None:
    assert _SYNC_FORCED_ASYNC_RUNTIME is False


def test_async_staged_entrypoint_forces_async_runtime() -> None:
    assert _ASYNC_FORCED_ASYNC_RUNTIME is True


def test_explicit_staged_sync_async_configs_validate() -> None:
    os.environ["EMBODIED_PATH"] = str(_CONFIG_DIR.parent)
    async_config_names = [
        "yam_ppo_openpi_async",
        "yam_ppo_openpi_subtask_async",
        "yam_ppo_openpi_desktop_async",
    ]
    sync_config_names = [
        "yam_ppo_openpi_sync",
        "yam_ppo_openpi_subtask_sync",
        "yam_ppo_openpi_desktop_sync",
    ]

    with initialize_config_dir(version_base="1.1", config_dir=str(_CONFIG_DIR)):
        for config_name in async_config_names:
            cfg = compose(config_name=config_name)
            assert cfg.algorithm.loss_type == "decoupled_actor_critic"
            if "desktop" in config_name:
                assert cfg.cluster.num_nodes == 2
                assert cfg.env.train.env_type == "yam"
            else:
                assert cfg.cluster.num_nodes == 1
                assert cfg.env.train.env_type == "remote"
        for config_name in sync_config_names:
            cfg = compose(config_name=config_name)
            assert cfg.algorithm.loss_type == "actor_critic"
            if "desktop" in config_name:
                assert cfg.cluster.num_nodes == 2
                assert cfg.env.train.env_type == "yam"
            else:
                assert cfg.cluster.num_nodes == 1
                assert cfg.env.train.env_type == "remote"


def test_remote_disconnect_error_detector_matches_remote_env_disconnect() -> None:
    error = RuntimeError(
        "ray worker failed: "
        "[RemoteEnv] Robot server disconnected during ChunkStep (gRPC UNAVAILABLE)."
    )

    assert _is_expected_remote_disconnect_error(error) is True


def test_remote_disconnect_error_detector_ignores_unrelated_failures() -> None:
    assert _is_expected_remote_disconnect_error(ValueError("ordinary failure")) is False


def test_remote_disconnect_monitor_invokes_shutdown_callback(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "env_type": "remote",
                    "remote_server_url": "localhost:50051",
                }
            }
        }
    )
    calls = []

    class _FailingSocket:
        def __enter__(self):
            raise OSError("disconnect")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(_MODULE, "_REMOTE_MONITOR_POLL_S", 0.01)
    monkeypatch.setattr(_MODULE, "_REMOTE_MONITOR_CONNECT_TIMEOUT_S", 0.01)
    monkeypatch.setattr(_MODULE, "_REMOTE_MONITOR_FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(
        _MODULE.socket, "create_connection", lambda *args, **kwargs: _FailingSocket()
    )
    _MODULE._REMOTE_DISCONNECT_EVENT.clear()

    stop_event, thread = _start_remote_disconnect_monitor(
        cfg, on_disconnect=lambda: calls.append("shutdown")
    )
    thread.join(timeout=0.2)
    stop_event.set()

    assert calls == ["shutdown"]
    assert _MODULE._REMOTE_DISCONNECT_EVENT.is_set() is True
    _MODULE._REMOTE_DISCONNECT_EVENT.clear()
