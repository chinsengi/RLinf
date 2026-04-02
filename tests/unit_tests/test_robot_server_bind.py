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

"""Tests for RobotServer gRPC listen-address binding."""

import pytest

from rlinf.envs.remote.robot_server import _bind_server


class _FakeGrpcServer:
    def __init__(self, results):
        self._results = list(results)
        self.calls: list[str] = []

    def add_insecure_port(self, address: str) -> int:
        self.calls.append(address)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_bind_server_falls_back_to_ipv4_when_ipv6_bind_fails() -> None:
    server = _FakeGrpcServer([RuntimeError("ipv6 unavailable"), 50051])

    bound_address = _bind_server(server, 50051)

    assert bound_address == "0.0.0.0:50051"
    assert server.calls == ["[::]:50051", "0.0.0.0:50051"]


def test_bind_server_raises_clear_error_when_all_addresses_fail() -> None:
    server = _FakeGrpcServer([RuntimeError("ipv6 unavailable"), 0])

    with pytest.raises(RuntimeError, match="Failed to bind RobotServer"):
        _bind_server(server, 50051)
