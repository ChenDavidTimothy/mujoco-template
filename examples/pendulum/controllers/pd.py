from __future__ import annotations

import numpy as np

import mujoco_template as mt

from ..pendulum_config import ControllerConfig


class PendulumPDController:
    """Simple PD torque controller that stabilizes the pendulum upright."""

    def __init__(self, config: ControllerConfig) -> None:
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self.kp = float(config.kp)
        self.kd = float(config.kd)
        self.target = float(np.deg2rad(config.target_angle_deg))

    def prepare(self, model: mt.mj.MjModel, _data: mt.mj.MjData) -> None:
        if model.nu != 1:
            raise mt.CompatibilityError("PendulumPDController expects a single actuator.")

    def __call__(self, model: mt.mj.MjModel, data: mt.mj.MjData, _t: float) -> None:
        angle = float(data.qpos[0])
        velocity = float(data.qvel[0])
        torque = -self.kp * (angle - self.target) - self.kd * velocity
        if hasattr(model, "actuator_forcerange") and model.actuator_forcerange.size >= 2:
            bounds = np.asarray(model.actuator_forcerange)[0]
            torque = float(np.clip(torque, bounds[0], bounds[1]))
        data.ctrl[0] = torque


__all__ = ["PendulumPDController"]

