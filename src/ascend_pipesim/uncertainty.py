"""Bounded environmental parameters for reproducible performance sweeps.

These are analytical assumptions, not claims about guaranteed hardware limits.
All corners are evaluated so reported best/worst timings refer to the declared
box. Interior scenario points can be added for dominance and regime analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.low)
            or not math.isfinite(self.high)
            or self.high < self.low
        ):
            raise ValueError("interval requires finite low <= high")

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    def to_dict(self) -> list[float]:
        return [self.low, self.high]

    @classmethod
    def parse(cls, value: "Interval | list[float] | tuple[float, float]") -> "Interval":
        return value if isinstance(value, cls) else cls(*value)


@dataclass(frozen=True)
class EnvironmentScenario:
    dma_latency_cycles: float
    effective_hbm_bandwidth_fraction: float
    cross_core_sync_cycles: float
    simt_cache_miss_rate_drift: float

    def to_dict(self) -> dict[str, float]:
        return dict(vars(self))


@dataclass(frozen=True)
class UncertaintySpace:
    dma_latency_cycles: Interval = Interval(180.0, 320.0)
    effective_hbm_bandwidth_fraction: Interval = Interval(0.6, 1.0)
    cross_core_sync_cycles: Interval = Interval(20.0, 80.0)
    simt_cache_miss_rate_drift: Interval = Interval(0.0, 0.2)

    def __post_init__(self) -> None:
        if self.dma_latency_cycles.low < 0 or self.cross_core_sync_cycles.low < 0:
            raise ValueError(
                "DMA latency and synchronization cycles must be nonnegative"
            )
        bandwidth = self.effective_hbm_bandwidth_fraction
        if bandwidth.low <= 0 or bandwidth.high > 1:
            raise ValueError("effective HBM bandwidth fraction must be in (0, 1]")
        if (
            self.simt_cache_miss_rate_drift.low < -1
            or self.simt_cache_miss_rate_drift.high > 1
        ):
            raise ValueError("SIMT miss-rate drift must lie in [-1, 1]")

    def nominal(self) -> EnvironmentScenario:
        return EnvironmentScenario(*(getattr(self, key).midpoint for key in vars(self)))

    def scenarios(self, *, interior: bool = True) -> tuple[EnvironmentScenario, ...]:
        """All box corners plus center and axial centers, with duplicates removed."""
        axes = tuple(vars(self).values())
        points = list(product(*((axis.low, axis.high) for axis in axes)))
        center = tuple(axis.midpoint for axis in axes)
        if interior:
            points.append(center)
            for axis_id, axis in enumerate(axes):
                for edge in (axis.low, axis.high):
                    point = list(center)
                    point[axis_id] = edge
                    points.append(tuple(point))
        return tuple(EnvironmentScenario(*point) for point in dict.fromkeys(points))

    def to_dict(self) -> dict[str, list[float]]:
        return {key: value.to_dict() for key, value in vars(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UncertaintySpace":
        return cls(**{key: Interval.parse(value) for key, value in data.items()})
