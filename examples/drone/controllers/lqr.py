from __future__ import annotations

from typing import Iterable
from types import SimpleNamespace

import numpy as np
import scipy.linalg
from numpy.linalg import LinAlgError

import mujoco_template as mt


class DroneLQRController:
    """Linear-quadratic regulator that steers the drone to a Cartesian target."""

    def __init__(
        self,
        config: SimpleNamespace,
        *,
        start_position: Iterable[float] | None = None,
        start_orientation: Iterable[float] | None = None,
        start_velocity: Iterable[float] | None = None,
        start_angular_velocity: Iterable[float] | None = None,
    ) -> None:
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self._config = config
        self._start_position_cfg = None if start_position is None else np.asarray(start_position, dtype=float)
        self._start_orientation_cfg = (
            None if start_orientation is None else self._normalize_quat(np.asarray(start_orientation, dtype=float))
        )
        self._start_velocity_cfg = (
            None if start_velocity is None else self._normalize_vector(start_velocity, "linear velocity")
        )
        self._start_angular_velocity_cfg = (
            None
            if start_angular_velocity is None
            else self._normalize_vector(start_angular_velocity, "angular velocity")
        )

        self._prepared = False
        self._nu = 0
        self._nv = 0
        self._pos_dim = 0
        self._rot_dim = 0

        self._qpos_goal: np.ndarray | None = None
        self._qpos_start: np.ndarray | None = None
        self._qvel_goal: np.ndarray | None = None
        self._qvel_start: np.ndarray | None = None
        self._ctrl0: np.ndarray | None = None
        self._goal_position: np.ndarray | None = None
        self._goal_orientation: np.ndarray | None = None
        self._goal_velocity: np.ndarray | None = None
        self._goal_angular_velocity: np.ndarray | None = None
        self._K: np.ndarray | None = None
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._dq_scratch: np.ndarray | None = None
        self._dx_scratch: np.ndarray | None = None
        self._ctrl_delta: np.ndarray | None = None
        self._ctrl_buffer: np.ndarray | None = None
        self._ctrl_low: np.ndarray | None = None
        self._ctrl_high: np.ndarray | None = None
        self._position_feedback_scale: np.ndarray | None = None
        self._orientation_feedback_scale: np.ndarray | None = None
        self._velocity_feedback_scale: np.ndarray | None = None
        self._angular_velocity_feedback_scale: np.ndarray | None = None
        self._yaw_control_scale: float = 1.0
        self._yaw_direction: np.ndarray | None = None
        self._yaw_direction_norm: float = 0.0
        self._yaw_proportional_gain: float = 0.0
        self._yaw_derivative_gain: float = 0.0
        self._yaw_integral_gain: float = 0.0
        self._yaw_integral_limit: float = 0.0
        self._yaw_integral: float = 0.0

    def prepare(self, model: mt.mj.MjModel, _data: mt.mj.MjData) -> None:
        cfg = self._config
        if model.nu == 0:
            raise mt.CompatibilityError("DroneLQRController requires force-producing actuators.")

        self._prepared = False
        self._nu = int(model.nu)
        self._nv = int(model.nv)
        pos_dim = min(3, self._nv)
        rot_dim = min(3, max(0, self._nv - pos_dim))
        self._pos_dim = pos_dim
        self._rot_dim = rot_dim

        work = mt.mj.MjData(model)
        keyframe_id = self._resolve_keyframe(model, cfg.keyframe)
        mt.mj.mj_resetDataKeyframe(model, work, keyframe_id)
        mt.mj.mj_forward(model, work)

        qpos_hover = np.array(work.qpos, dtype=float)
        ctrl_hover = np.array(work.ctrl[: self._nu], dtype=float)

        goal_position = self._normalize_position(cfg.goal_position_m)
        goal_orientation = self._normalize_quat(np.asarray(cfg.goal_orientation_wxyz, dtype=float))
        goal_velocity = self._normalize_vector(getattr(cfg, "goal_velocity_mps", (0.0, 0.0, 0.0)), "linear velocity")
        goal_angular_velocity = self._normalize_vector(
            getattr(cfg, "goal_angular_velocity_radps", (0.0, 0.0, 0.0)), "angular velocity"
        )

        qpos_goal = np.array(qpos_hover, copy=True)
        qpos_goal[:3] = goal_position
        qpos_goal[3:7] = goal_orientation

        start_position = (
            goal_position if self._start_position_cfg is None else self._normalize_position(self._start_position_cfg)
        )
        start_orientation = goal_orientation if self._start_orientation_cfg is None else self._start_orientation_cfg
        start_velocity = goal_velocity if self._start_velocity_cfg is None else self._start_velocity_cfg
        start_angular_velocity = (
            goal_angular_velocity if self._start_angular_velocity_cfg is None else self._start_angular_velocity_cfg
        )

        qvel_goal = self._compose_qvel(goal_velocity, goal_angular_velocity)
        qvel_start = self._compose_qvel(start_velocity, start_angular_velocity)

        qpos_start = np.array(qpos_goal, copy=True)
        qpos_start[:3] = start_position
        qpos_start[3:7] = start_orientation

        work.qpos[:] = qpos_goal
        work.qvel[:] = qvel_goal
        work.ctrl[: self._nu] = ctrl_hover
        mt.mj.mj_forward(model, work)

        eps = float(cfg.linearization_eps)
        if eps <= 0.0:
            raise mt.ConfigError("linearization_eps must be positive.")
        A, B = mt.linearize_discrete(model, work, eps=eps)

        Q = self._build_state_cost(cfg)
        R = self._build_ctrl_cost(cfg)

        _P, K = self._solve_lqr_gains(A, B, Q, R)

        closed_loop = A - B @ K
        eigvals = np.linalg.eigvals(closed_loop)
        spectral_radius = float(np.max(np.abs(eigvals)))
        print(f"Closed-loop spectral radius: {spectral_radius:.9f}")

        self._qpos_goal = qpos_goal
        self._qpos_start = qpos_start
        self._qvel_goal = qvel_goal
        self._qvel_start = qvel_start
        self._ctrl0 = ctrl_hover
        self._goal_position = goal_position
        self._goal_orientation = goal_orientation
        self._goal_velocity = goal_velocity
        self._goal_angular_velocity = goal_angular_velocity
        self._K = K
        self._A = A
        self._B = B
        self._dq_scratch = np.zeros(self._nv)
        self._dx_scratch = np.zeros(2 * self._nv)
        self._ctrl_delta = np.zeros(self._nu)
        self._ctrl_buffer = np.zeros(self._nu)
        self._position_feedback_scale = self._resolve_feedback_scale(
            getattr(cfg, "position_feedback_scale", 1.0), pos_dim, "position feedback"
        )
        self._orientation_feedback_scale = self._resolve_feedback_scale(
            getattr(cfg, "orientation_feedback_scale", 1.0), rot_dim, "orientation feedback"
        )
        self._velocity_feedback_scale = self._resolve_feedback_scale(
            getattr(cfg, "velocity_feedback_scale", 1.0), pos_dim, "velocity feedback"
        )
        self._angular_velocity_feedback_scale = self._resolve_feedback_scale(
            getattr(cfg, "angular_velocity_feedback_scale", 1.0), rot_dim, "angular-velocity feedback"
        )
        yaw_scale = float(getattr(cfg, "yaw_control_scale", 1.0))
        if yaw_scale <= 0.0:
            raise mt.ConfigError("yaw_control_scale must be positive.")
        self._yaw_control_scale = yaw_scale
        yaw_integral_gain = float(getattr(cfg, "yaw_integral_gain", 0.0))
        if yaw_integral_gain < 0.0:
            raise mt.ConfigError("yaw_integral_gain must be non-negative.")
        yaw_integral_limit = float(getattr(cfg, "yaw_integral_limit", np.inf))
        if yaw_integral_limit <= 0.0 and not np.isinf(yaw_integral_limit):
            raise mt.ConfigError("yaw_integral_limit must be positive when finite.")
        yaw_proportional_gain = float(getattr(cfg, "yaw_proportional_gain", 0.0))
        if yaw_proportional_gain < 0.0:
            raise mt.ConfigError("yaw_proportional_gain must be non-negative.")
        yaw_derivative_gain = float(getattr(cfg, "yaw_derivative_gain", 0.0))
        if yaw_derivative_gain < 0.0:
            raise mt.ConfigError("yaw_derivative_gain must be non-negative.")
        self._yaw_integral_gain = yaw_integral_gain
        self._yaw_integral_limit = yaw_integral_limit
        self._yaw_proportional_gain = yaw_proportional_gain
        self._yaw_derivative_gain = yaw_derivative_gain
        self._yaw_integral = 0.0
        if rot_dim > 0:
            yaw_idx = self._nv + pos_dim + rot_dim - 1
            yaw_dir = B[yaw_idx, : self._nu].astype(float, copy=True)
            yaw_norm = float(np.dot(yaw_dir, yaw_dir))
            if yaw_norm > 0.0:
                self._yaw_direction = yaw_dir
                self._yaw_direction_norm = yaw_norm
            else:
                self._yaw_direction = None
                self._yaw_direction_norm = 0.0
        else:
            self._yaw_direction = None
            self._yaw_direction_norm = 0.0

        if cfg.clip_controls and hasattr(model, "actuator_ctrllimited") and hasattr(model, "actuator_ctrlrange"):
            limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
            ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=float)
            if ctrl_range.shape == (self._nu, 2):
                low = np.where(limited, ctrl_range[:, 0], -np.inf)
                high = np.where(limited, ctrl_range[:, 1], np.inf)
                self._ctrl_low = low
                self._ctrl_high = high
            else:
                self._ctrl_low = None
                self._ctrl_high = None
        else:
            self._ctrl_low = None
            self._ctrl_high = None

        self._prepared = True

    def __call__(self, model: mt.mj.MjModel, data: mt.mj.MjData, _t: float) -> None:
        if not self._prepared or self._qpos_goal is None or self._ctrl0 is None or self._K is None:
            raise mt.TemplateError("DroneLQRController invoked before prepare().")
        assert self._dq_scratch is not None and self._dx_scratch is not None
        assert self._ctrl_delta is not None and self._ctrl_buffer is not None
        assert self._qvel_goal is not None

        mt.mj.mj_differentiatePos(model, self._dq_scratch, 1.0, self._qpos_goal, data.qpos)
        self._dx_scratch[: self._nv] = self._dq_scratch
        self._dx_scratch[self._nv :] = data.qvel - self._qvel_goal

        pos_dim = self._pos_dim
        rot_dim = self._rot_dim
        assert self._position_feedback_scale is not None
        assert self._velocity_feedback_scale is not None
        assert self._orientation_feedback_scale is not None
        assert self._angular_velocity_feedback_scale is not None

        if pos_dim > 0:
            self._dx_scratch[:pos_dim] *= self._position_feedback_scale
            self._dx_scratch[self._nv : self._nv + pos_dim] *= self._velocity_feedback_scale
        if rot_dim > 0:
            start = pos_dim
            stop = pos_dim + rot_dim
            self._dx_scratch[start:stop] *= self._orientation_feedback_scale
            vel_start = self._nv + pos_dim
            vel_stop = vel_start + rot_dim
            self._dx_scratch[vel_start:vel_stop] *= self._angular_velocity_feedback_scale

        self._ctrl_delta[:] = self._K @ self._dx_scratch
        yaw_dir = self._yaw_direction
        yaw_norm = self._yaw_direction_norm
        yaw_gain = self._yaw_integral_gain
        yaw_prop = self._yaw_proportional_gain
        yaw_deriv = self._yaw_derivative_gain
        yaw_error = 0.0
        yaw_rate_error = 0.0
        if yaw_dir is not None and yaw_norm > 0.0 and self._rot_dim > 0:
            yaw_error = float(self._dx_scratch[self._pos_dim + self._rot_dim - 1])
            yaw_rate_error = float(self._dx_scratch[self._nv + self._pos_dim + self._rot_dim - 1])
            if yaw_gain > 0.0:
                dt = float(model.opt.timestep)
                self._yaw_integral += yaw_error * dt
                limit = self._yaw_integral_limit
                if not np.isinf(limit):
                    self._yaw_integral = float(np.clip(self._yaw_integral, -limit, limit))
                self._ctrl_delta += yaw_gain * self._yaw_integral * yaw_dir
            if yaw_prop > 0.0 or yaw_deriv > 0.0:
                yaw_feedback = yaw_prop * yaw_error + yaw_deriv * yaw_rate_error
                self._ctrl_delta += yaw_feedback * yaw_dir
        if yaw_dir is not None and yaw_norm > 0.0 and self._yaw_control_scale != 1.0:
            yaw_component = float(np.dot(self._ctrl_delta, yaw_dir)) / yaw_norm
            self._ctrl_delta += (self._yaw_control_scale - 1.0) * yaw_component * yaw_dir
        self._ctrl_buffer[:] = self._ctrl0 - self._ctrl_delta

        if self._ctrl_low is not None and self._ctrl_high is not None:
            np.clip(self._ctrl_buffer, self._ctrl_low, self._ctrl_high, out=self._ctrl_buffer)

        data.ctrl[: self._nu] = self._ctrl_buffer

    @staticmethod
    def _resolve_keyframe(model: mt.mj.MjModel, keyframe: int | str) -> int:
        if isinstance(keyframe, str):
            key_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_KEY, keyframe))
            if key_id < 0:
                raise mt.ConfigError(f"Keyframe '{keyframe}' not found in model.")
            return key_id
        key_id = int(keyframe)
        if key_id < 0 or key_id >= model.nkey:
            raise mt.ConfigError(f"Keyframe index {key_id} out of range for model with {model.nkey} keyframes.")
        return key_id

    @staticmethod
    def _normalize_position(position: Iterable[float]) -> np.ndarray:
        pos = np.asarray(position, dtype=float).reshape(-1)
        if pos.size != 3:
            raise mt.ConfigError("Positions must be length-3 XYZ tuples.")
        return pos

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=float).reshape(-1)
        if quat.size != 4:
            raise mt.ConfigError("Quaternions must contain four components (w, x, y, z).")
        norm = float(np.linalg.norm(quat))
        if norm == 0.0:
            raise mt.ConfigError("Quaternion magnitude must be non-zero.")
        return quat / norm

    @staticmethod
    def _normalize_vector(vector: Iterable[float], kind: str) -> np.ndarray:
        vec = np.asarray(vector, dtype=float).reshape(-1)
        if vec.size != 3:
            raise mt.ConfigError(f"{kind.capitalize()} specifications must be length-3 tuples.")
        return vec

    def _resolve_feedback_scale(self, scale: Iterable[float] | float, size: int, label: str) -> np.ndarray:
        arr = np.asarray(scale, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr, size)
        if arr.size != size:
            raise mt.ConfigError(f"{label.capitalize()} scale must broadcast to dimension {size}.")
        if np.any(arr <= 0.0):
            raise mt.ConfigError(f"{label.capitalize()} scale entries must be positive.")
        return arr

    @staticmethod
    def _compose_qvel(linear: np.ndarray, angular: np.ndarray) -> np.ndarray:
        linear = np.asarray(linear, dtype=float).reshape(-1)
        angular = np.asarray(angular, dtype=float).reshape(-1)
        return np.concatenate([linear, angular])

    def _build_state_cost(self, cfg: SimpleNamespace) -> np.ndarray:
        pos_weights = self._resolve_feedback_scale(cfg.position_weight, self._pos_dim, "position weight")
        rot_weights = self._resolve_feedback_scale(cfg.orientation_weight, self._rot_dim, "orientation weight")
        vel_weights = self._resolve_feedback_scale(cfg.velocity_weight, self._pos_dim, "velocity weight")
        ang_weights = self._resolve_feedback_scale(
            cfg.angular_velocity_weight, self._rot_dim, "angular velocity weight"
        )

        q_weights = np.concatenate([pos_weights, rot_weights])
        qd_weights = np.concatenate([vel_weights, ang_weights])
        return np.diag(np.concatenate([q_weights, qd_weights]))

    def _build_ctrl_cost(self, cfg: SimpleNamespace) -> np.ndarray:
        control_weight = float(cfg.control_weight)
        if control_weight < 0.0:
            raise mt.ConfigError("control_weight must be non-negative.")
        return control_weight * np.eye(self._nu)

    def _solve_lqr_gains(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            P = scipy.linalg.solve_discrete_are(A, B, Q, R)
            K = scipy.linalg.solve(R + B.T @ P @ B, B.T @ P @ A, assume_a="sym")
            return P, K
        except LinAlgError as err:
            last_error: LinAlgError | None = err

        # Retry with progressively inflated state costs for numerical stability.
        scales = (10.0, 100.0, 1000.0)
        last_error = None
        for scale in scales:
            try:
                P = scipy.linalg.solve_discrete_are(A, B, scale * Q, R)
                K = scipy.linalg.solve(R + B.T @ P @ B, B.T @ P @ A, assume_a="sym")
                return P, K
            except LinAlgError as err:  # pragma: no cover - rare failure path
                last_error = err

        raise mt.TemplateError(
            "Unable to compute discrete LQR gains – try increasing the cost weights or"
            " adjusting the linearization epsilon."
        ) from last_error

    @property
    def qpos_goal(self) -> np.ndarray:
        if self._qpos_goal is None:
            raise mt.TemplateError("Goal configuration requested before prepare().")
        return np.array(self._qpos_goal, copy=True)

    @property
    def qpos_start(self) -> np.ndarray:
        if self._qpos_start is None:
            raise mt.TemplateError("Start configuration requested before prepare().")
        return np.array(self._qpos_start, copy=True)

    @property
    def qvel_goal(self) -> np.ndarray:
        if self._qvel_goal is None:
            raise mt.TemplateError("Goal velocity requested before prepare().")
        return np.array(self._qvel_goal, copy=True)

    @property
    def qvel_start(self) -> np.ndarray:
        if self._qvel_start is None:
            raise mt.TemplateError("Start velocity requested before prepare().")
        return np.array(self._qvel_start, copy=True)

    @property
    def ctrl_equilibrium(self) -> np.ndarray:
        if self._ctrl0 is None:
            raise mt.TemplateError("Equilibrium controls requested before prepare().")
        return np.array(self._ctrl0, copy=True)

    @property
    def goal_position(self) -> np.ndarray:
        if self._goal_position is None:
            raise mt.TemplateError("Goal position requested before prepare().")
        return np.array(self._goal_position, copy=True)

    @property
    def goal_orientation(self) -> np.ndarray:
        if self._goal_orientation is None:
            raise mt.TemplateError("Goal orientation requested before prepare().")
        return np.array(self._goal_orientation, copy=True)

    @property
    def goal_velocity(self) -> np.ndarray:
        if self._goal_velocity is None:
            raise mt.TemplateError("Goal velocity requested before prepare().")
        return np.array(self._goal_velocity, copy=True)

    @property
    def goal_angular_velocity(self) -> np.ndarray:
        if self._goal_angular_velocity is None:
            raise mt.TemplateError("Goal angular velocity requested before prepare().")
        return np.array(self._goal_angular_velocity, copy=True)

    @property
    def gains(self) -> np.ndarray:
        if self._K is None:
            raise mt.TemplateError("Controller gains requested before prepare().")
        return np.array(self._K, copy=True)


__all__ = ["DroneLQRController"]

