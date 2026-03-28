# Copyright 2026 Shirui Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for background process cleanup helpers."""

from yam_realtime.utils.portal_utils import shutdown_background_process


class _FakePortalProcess:
    def __init__(self, running: bool = True):
        self.running = running
        self.kill_calls: list[float] = []
        self.join_calls: list[float] = []

    def kill(self, timeout: float = 1.0) -> None:
        self.kill_calls.append(timeout)
        self.running = False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


class _FakeSubprocess:
    def __init__(self) -> None:
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float] = []

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def test_shutdown_background_process_uses_portal_kill_for_portal_process() -> None:
    proc = _FakePortalProcess(running=True)

    shutdown_background_process(proc, timeout=3.0)

    assert proc.kill_calls == [3.0]
    assert proc.running is False


def test_shutdown_background_process_joins_finished_portal_process() -> None:
    proc = _FakePortalProcess(running=False)

    shutdown_background_process(proc, timeout=2.5)

    assert proc.kill_calls == []
    assert proc.join_calls == [2.5]


def test_shutdown_background_process_falls_back_to_subprocess_api() -> None:
    proc = _FakeSubprocess()

    shutdown_background_process(proc, timeout=1.5)

    assert proc.terminate_calls == 1
    assert proc.wait_calls == [1.5]
    assert proc.kill_calls == 0
