import numpy as np

import mujoco_template as mt


class CartPolePIDController:
    """PID balance controller applying horizontal force to the cart."""

    def __init__(self, config):
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self.angle_kp = float(config.angle_kp)
        self.angle_kd = float(config.angle_kd)
        self.angle_ki = float(config.angle_ki)
        self.position_kp = float(config.position_kp)
        self.position_kd = float(config.position_kd)
        self.integral_limit = float(abs(config.integral_limit))
        self._integral_term = 0.0
        self._dt = 0.0

    def prepare(self, model, _data):
        if model.nu != 1:
            raise mt.CompatibilityError("CartPolePIDController expects a single actuator driving the cart.")
        self._integral_term = 0.0
        self._dt = float(model.opt.timestep)

    def __call__(self, model, data, _t):
        cart_x = float(data.qpos[0])
        pole_angle = float(data.qpos[1])
        cart_vel = float(data.qvel[0])
        pole_ang_vel = float(data.qvel[1])

        if self.angle_ki != 0.0:
            self._integral_term += pole_angle * self._dt
            self._integral_term = float(np.clip(self._integral_term, -self.integral_limit, self.integral_limit))
        else:
            self._integral_term = 0.0

        force = (
            self.position_kp * cart_x
            + self.position_kd * cart_vel
            + self.angle_kp * pole_angle
            + self.angle_kd * pole_ang_vel
            + self.angle_ki * self._integral_term
        )

        if hasattr(model, "actuator_ctrlrange") and model.actuator_ctrlrange.size >= 2:
            low, high = model.actuator_ctrlrange[0]
            force = float(np.clip(force, low, high))
        data.ctrl[0] = force


__all__ = ["CartPolePIDController"]

