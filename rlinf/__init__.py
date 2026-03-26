# Copyright 2025 The RLinf Authors.
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

import os

# RLinf manages Ray worker interpreters explicitly via `runtime_env.py_executable`.
# When the driver itself is launched with `uv run`, Ray's automatic uv
# propagation creates a second temporary environment for workers, which can
# drift from the driver's pinned Ray version. Keep that hook disabled by
# default unless the user explicitly opts back in.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from .utils.omega_resolver import omegaconf_register

omegaconf_register()
