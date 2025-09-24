from __future__ import annotations

from dataclasses import dataclass

import mujoco as mj
import numpy as np

from .control import ControlSpace, ControllerCapabilities
from .exceptions import CompatibilityError, ConfigError, TemplateError


@dataclass
class ZeroController:
    capabilities: ControllerCapabilities = ControllerCapabilities(
        control_space=ControlSpace.TORQUE
    )

    def prepare(self, model: mj.MjModel, data: mj.MjData) -> None:
        if model.nu == 0:
            raise CompatibilityError("ZeroController requires nu>0 to write controls.")

    def __call__(self, model: mj.MjModel, data: mj.MjData, t: float) -> None:
        if data.ctrl.shape[0] != model.nu:
            raise TemplateError("data.ctrl size does not match model.nu")
        data.ctrl[:] = 0.0


@dataclass
class PositionTargetDemo:
    targets: np.ndarray | None = None
    capabilities: ControllerCapabilities = ControllerCapabilities(
        control_space=ControlSpace.POSITION
    )

    def prepare(self, model: mj.MjModel, data: mj.MjData) -> None:
        if model.nu == 0:
            raise CompatibilityError("PositionTargetDemo requires nu>0.")
        if self.targets is None:
            self.targets = np.array(data.ctrl) if data.ctrl.size == model.nu else np.zeros(model.nu)
        if self.targets.shape[0] != model.nu:
            raise ConfigError("targets must have length model.nu")

    def __call__(self, model: mj.MjModel, data: mj.MjData, t: float) -> None:
        if data.ctrl.shape[0] != model.nu:
            raise TemplateError("data.ctrl size does not match model.nu")
        data.ctrl[:] = self.targets


__all__ = ["ZeroController", "PositionTargetDemo"]
