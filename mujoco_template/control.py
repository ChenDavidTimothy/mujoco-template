from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, cast

import mujoco as mj


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


def controller_from_callable(
    fn: Callable[[mj.MjModel, mj.MjData, float], None],
    *,
    control_space: str = ControlSpace.TORQUE,
    actuator_groups: Iterable[int] | None = None,
    needs_linearization: bool = False,
    needs_jacobians: Iterable[str] = (),
    initializer: Callable[[mj.MjModel, mj.MjData], None] | None = None,
) -> Controller:
    """Wrap a simple callable into a strict ``Controller`` implementation.

    Advanced users can keep providing a full ``Controller`` instance.  For quick
    experiments the callable may ignore ``t`` entirely; the wrapper keeps the
    MuJoCo-native capability negotiation intact so ``Env`` can continue to
    honour actuator group and Jacobian/linearization requests.
    """

    caps = ControllerCapabilities(
        control_space=control_space,
        needs_linearization=bool(needs_linearization),
        needs_jacobians=tuple(needs_jacobians),
        actuator_groups=tuple(int(g) for g in actuator_groups) if actuator_groups else None,
    )

    class _CallableController:
        capabilities = caps

        def __init__(self, callback: Callable[[mj.MjModel, mj.MjData, float], None]):
            self._callback = callback

        def prepare(self, model: mj.MjModel, data: mj.MjData) -> None:  # pragma: no cover - thin wrapper
            if initializer is not None:
                initializer(model, data)

        def __call__(self, model: mj.MjModel, data: mj.MjData, t: float) -> None:
            self._callback(model, data, t)

    return cast(Controller, _CallableController(fn))


__all__ = [
    "ControlSpace",
    "ControllerCapabilities",
    "Controller",
    "controller_from_callable",
]
