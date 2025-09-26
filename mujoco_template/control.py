from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import mujoco as mj
from typing import Protocol


class ControlSpace:
    """Canonical control-space tokens aligned with MuJoCo actuator expectations."""

    TORQUE = "torque"
    POSITION = "position"
    VELOCITY = "velocity"
    INTVELOCITY = "intvelocity"


@dataclass(frozen=True)
class ControllerCapabilities:
    control_space: str = ControlSpace.TORQUE
    needs_linearization: bool = False
    needs_jacobians: Iterable[str] = field(default_factory=tuple)
    actuator_groups: Iterable[int] | None = None


class Controller(Protocol):
    """Minimal controller protocol: native and strict."""

    capabilities: ControllerCapabilities

    def prepare(self, model: mj.MjModel, data: mj.MjData) -> None: ...
    def __call__(self, model: mj.MjModel, data: mj.MjData, t: float) -> None: ...


__all__ = [
    "ControlSpace",
    "ControllerCapabilities",
    "Controller",
]
