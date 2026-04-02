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

"""Latency profiler for the embodied RL pipeline.

Records per-layer timing across training steps and produces summary
reports.  Supports both *dummy* mode (simulated desktop, no real robot)
and *real* mode (gRPC to physical robot arm).

Usage::

    from tools.latency_profiler import LatencyProfiler

    profiler = LatencyProfiler(mode="dummy")  # or "real"

    for step in range(n_steps):
        profiler.step_start(step)

        profiler.start("weight_sync")
        # ... actual weight sync ...
        profiler.stop("weight_sync")

        profiler.start("rollout_inference")
        # ... inference ...
        profiler.stop("rollout_inference")

        # ... remaining layers ...

        profiler.step_end()

    profiler.report()
    profiler.export_csv("latency.csv")
"""

from __future__ import annotations

import csv
import io
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from tools.latency_profiler.layers import DEFAULT_LAYERS, Layer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class LayerTimer:
    """Accumulated timing for a single layer within one step."""

    layer: str
    start_time: float | None = None
    end_time: float | None = None
    duration_ms: float | None = None

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        if self.start_time is None:
            raise RuntimeError(f"Layer '{self.layer}' was never started.")
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0


@dataclass
class StepRecord:
    """All layer timings for a single training step."""

    step_id: int
    mode: str
    timers: dict[str, LayerTimer] = field(default_factory=dict)
    step_start_time: float | None = None
    step_end_time: float | None = None
    total_ms: float | None = None

    def layer_ms(self, layer_name: str) -> float | None:
        timer = self.timers.get(layer_name)
        return timer.duration_ms if timer else None


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


class LatencyProfiler:
    """Records and reports per-layer latency for the embodied RL pipeline.

    Parameters
    ----------
    mode : str
        ``"dummy"`` for simulated desktop (no real robot) or
        ``"real"`` for physical robot arm via gRPC.
    layers : sequence of Layer, optional
        Which layers to track.  Defaults to ``DEFAULT_LAYERS``.
    enabled : bool
        Set to ``False`` to make all timing calls no-ops (zero overhead
        when profiling is not needed).
    """

    def __init__(
        self,
        mode: str = "real",
        layers: tuple[Layer, ...] | None = None,
        enabled: bool = True,
    ) -> None:
        if mode not in ("dummy", "real"):
            raise ValueError(f"mode must be 'dummy' or 'real', got {mode!r}")
        self.mode = mode
        self.enabled = enabled
        self._layers = layers or DEFAULT_LAYERS
        self._layer_names: set[str] = {l.name for l in self._layers}
        self._records: list[StepRecord] = []
        self._current: StepRecord | None = None

    # -- step lifecycle -----------------------------------------------------

    def step_start(self, step_id: int) -> None:
        """Mark the beginning of a training step."""
        if not self.enabled:
            return
        self._current = StepRecord(
            step_id=step_id,
            mode=self.mode,
            step_start_time=time.perf_counter(),
        )

    def step_end(self) -> None:
        """Mark the end of a training step and store the record."""
        if not self.enabled or self._current is None:
            return
        self._current.step_end_time = time.perf_counter()
        self._current.total_ms = (
            self._current.step_end_time - self._current.step_start_time
        ) * 1000.0
        self._records.append(self._current)
        self._current = None

    # -- layer timing -------------------------------------------------------

    def start(self, layer_name: str) -> None:
        """Begin timing *layer_name* in the current step."""
        if not self.enabled or self._current is None:
            return
        if layer_name not in self._layer_names:
            raise ValueError(
                f"Unknown layer '{layer_name}'. Known: {sorted(self._layer_names)}"
            )
        timer = LayerTimer(layer=layer_name)
        timer.start()
        self._current.timers[layer_name] = timer

    def stop(self, layer_name: str) -> None:
        """Finish timing *layer_name* in the current step."""
        if not self.enabled or self._current is None:
            return
        timer = self._current.timers.get(layer_name)
        if timer is None:
            raise RuntimeError(
                f"Layer '{layer_name}' was never started in step "
                f"{self._current.step_id}."
            )
        timer.stop()

    @contextmanager
    def track(self, layer_name: str) -> Generator[None, None, None]:
        """Context manager shorthand for ``start`` / ``stop``."""
        self.start(layer_name)
        try:
            yield
        finally:
            self.stop(layer_name)

    # -- data access --------------------------------------------------------

    @property
    def records(self) -> list[StepRecord]:
        return list(self._records)

    def clear(self) -> None:
        """Drop all recorded data."""
        self._records.clear()
        self._current = None

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute per-layer statistics across all recorded steps.

        Returns a dict ``{layer_name: {"mean": …, "min": …, "max": …,
        "std": …, "count": …}}``.
        """
        from math import sqrt

        stats: dict[str, dict[str, float]] = {}
        for layer in self._layers:
            durations = [
                r.layer_ms(layer.name)
                for r in self._records
                if r.layer_ms(layer.name) is not None
            ]
            if not durations:
                stats[layer.name] = {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "std": 0.0,
                    "count": 0,
                }
                continue
            n = len(durations)
            mean = sum(durations) / n
            variance = sum((d - mean) ** 2 for d in durations) / n if n > 1 else 0.0
            stats[layer.name] = {
                "mean": mean,
                "min": min(durations),
                "max": max(durations),
                "std": sqrt(variance),
                "count": n,
            }

        # Total step time.
        totals = [r.total_ms for r in self._records if r.total_ms is not None]
        if totals:
            n = len(totals)
            mean = sum(totals) / n
            variance = sum((d - mean) ** 2 for d in totals) / n if n > 1 else 0.0
            stats["total_step"] = {
                "mean": mean,
                "min": min(totals),
                "max": max(totals),
                "std": sqrt(variance),
                "count": n,
            }
        return stats

    def report(self, file: io.TextIOBase | None = None) -> str:
        """Print a human-readable latency table and return it as a string.

        Parameters
        ----------
        file : text stream, optional
            If given, the report is written there instead of stdout.
        """
        stats = self.summary()
        if not stats:
            msg = "No latency data recorded.\n"
            if file:
                file.write(msg)
            else:
                print(msg)
            return msg

        lines: list[str] = []
        header = (
            f"{'Layer':<25} {'Mean (ms)':>10} {'Min (ms)':>10} "
            f"{'Max (ms)':>10} {'Std (ms)':>10} {'Count':>6}"
        )
        sep = "-" * len(header)
        lines.append(f"\n  Latency Report  (mode={self.mode})")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for layer in self._layers:
            s = stats.get(layer.name)
            if s is None or s["count"] == 0:
                lines.append(f"{layer.name:<25} {'(no data)':>10}")
                continue
            lines.append(
                f"{layer.name:<25} {s['mean']:>10.2f} {s['min']:>10.2f} "
                f"{s['max']:>10.2f} {s['std']:>10.2f} {int(s['count']):>6}"
            )

        # Total row.
        ts = stats.get("total_step")
        if ts and ts["count"] > 0:
            lines.append(sep)
            lines.append(
                f"{'TOTAL STEP':<25} {ts['mean']:>10.2f} {ts['min']:>10.2f} "
                f"{ts['max']:>10.2f} {ts['std']:>10.2f} {int(ts['count']):>6}"
            )
        lines.append(sep)
        lines.append("")

        text = "\n".join(lines)
        if file:
            file.write(text)
        else:
            print(text)
        return text

    # -- export -------------------------------------------------------------

    def export_csv(self, path: str) -> None:
        """Write per-step, per-layer timings to a CSV file."""
        layer_names = [l.name for l in self._layers]
        fieldnames = ["step", "mode", *layer_names, "total_ms"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._records:
                row: dict[str, object] = {
                    "step": rec.step_id,
                    "mode": rec.mode,
                    "total_ms": f"{rec.total_ms:.2f}" if rec.total_ms else "",
                }
                for name in layer_names:
                    ms = rec.layer_ms(name)
                    row[name] = f"{ms:.2f}" if ms is not None else ""
                writer.writerow(row)

    def export_json(self, path: str) -> None:
        """Write all records to a JSON file."""
        import json

        data = {
            "mode": self.mode,
            "layers": [l.name for l in self._layers],
            "steps": [],
        }
        for rec in self._records:
            step_data: dict[str, object] = {
                "step": rec.step_id,
                "total_ms": rec.total_ms,
                "layers": {},
            }
            for l in self._layers:
                step_data["layers"][l.name] = rec.layer_ms(l.name)
            data["steps"].append(step_data)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
