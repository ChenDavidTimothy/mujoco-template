import sys
from typing import Iterable

import numpy as np
import scipy.linalg

from numpy.linalg import LinAlgError

import mujoco_template as mt

from drone_common import make_env, make_navigation_probes
from drone_config import CONFIG


class DroneLQRController:
    """Linear-quadratic regulator that steers the drone to a Cartesian target."""

    def __init__(
        self,
        config,
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

        P, K = self._solve_lqr_gains(A, B, Q, R)

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

    def _compose_qvel(self, linear_velocity: np.ndarray, angular_velocity: np.ndarray) -> np.ndarray:
        qvel = np.zeros(self._nv)
        pos_dim = min(3, self._nv)
        rot_dim = min(3, max(0, self._nv - pos_dim))
        if pos_dim > 0:
            qvel[:pos_dim] = linear_velocity[:pos_dim]
        if rot_dim > 0:
            qvel[pos_dim : pos_dim + rot_dim] = angular_velocity[:rot_dim]
        return qvel

    def _build_state_cost(self, cfg) -> np.ndarray:
        nv = self._nv
        Q = np.zeros((2 * nv, 2 * nv))

        pos_dim = min(3, nv)
        rot_dim = min(3, max(0, nv - pos_dim))

        if pos_dim > 0:
            pos_diag = self._resolve_weight_diagonal(cfg.position_weight, pos_dim, "position")
            vel_diag = self._resolve_weight_diagonal(cfg.velocity_weight, pos_dim, "velocity")
            Q[:pos_dim, :pos_dim] = np.diag(pos_diag)
            Q[nv : nv + pos_dim, nv : nv + pos_dim] = np.diag(vel_diag)
        if rot_dim > 0:
            start = pos_dim
            stop = pos_dim + rot_dim
            ori_diag = self._resolve_weight_diagonal(cfg.orientation_weight, rot_dim, "orientation")
            ang_diag = self._resolve_weight_diagonal(
                cfg.angular_velocity_weight, rot_dim, "angular velocity"
            )
            Q[start:stop, start:stop] = np.diag(ori_diag)
            vel_start = nv + pos_dim
            vel_stop = vel_start + rot_dim
            Q[vel_start:vel_stop, vel_start:vel_stop] = np.diag(ang_diag)

        return Q

    def _build_ctrl_cost(self, cfg) -> np.ndarray:
        weight = float(cfg.control_weight)
        if weight <= 0.0:
            raise mt.ConfigError("control_weight must be positive.")
        return weight * np.eye(self._nu)

    @staticmethod
    def _resolve_weight_diagonal(weight, dim: int, name: str) -> np.ndarray:
        values = np.asarray(weight, dtype=float)
        if values.ndim == 0:
            values = np.full(dim, float(values))
        else:
            values = values.reshape(-1)
            if values.size == 1:
                values = np.full(dim, float(values[0]))
            elif values.size != dim:
                raise mt.ConfigError(
                    f"{name.capitalize()} weight must be a scalar or length-{dim} sequence; got {values.size}."
                )

        if np.any(~np.isfinite(values)):
            raise mt.ConfigError(f"{name.capitalize()} weights must be finite numbers.")
        if np.any(values < 0.0):
            raise mt.ConfigError(f"{name.capitalize()} weights must be non-negative.")
        return values

    @staticmethod
    def _resolve_feedback_scale(scale, dim: int, name: str) -> np.ndarray:
        if dim == 0:
            return np.ones(0)
        values = np.asarray(scale, dtype=float)
        if values.ndim == 0:
            values = np.full(dim, float(values))
        else:
            values = values.reshape(-1)
            if values.size == 1:
                values = np.full(dim, float(values[0]))
            elif values.size != dim:
                raise mt.ConfigError(
                    f"{name.capitalize()} must be a scalar or length-{dim} sequence; got {values.size}."
                )
        if np.any(~np.isfinite(values)):
            raise mt.ConfigError(f"{name.capitalize()} must contain finite numbers.")
        if np.any(values <= 0.0):
            raise mt.ConfigError(f"{name.capitalize()} must be strictly positive.")
        return values

    def _solve_lqr_gains(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve the discrete-time algebraic Riccati equation with fallbacks.

        SciPy's :func:`solve_discrete_are` occasionally raises ``LinAlgError`` when
        the associated symplectic pencil has eigenvalues close to the unit
        circle.  This happens for the drone linearization due to near-marginally
        stable dynamics around hover.  To make the example robust across
        platforms we progressively regularize the state and control penalties and
        retry the solver.  The additional penalty is tiny and has a negligible
        effect on the resulting controller but nudges the eigenvalues away from
        the problematic regime.
        """

        try:
            P = scipy.linalg.solve_discrete_are(A, B, Q, R)
            K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            return P, K
        except LinAlgError:
            pass

        eye_Q = np.eye(Q.shape[0])
        eye_R = np.eye(R.shape[0])
        last_error: LinAlgError | None = None

        # Retry with gradually increasing diagonal regularization.
        for reg in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
            try:
                Q_reg = Q + reg * eye_Q
                R_reg = R + reg * eye_R
                P = scipy.linalg.solve_discrete_are(A, B, Q_reg, R_reg)
                K = np.linalg.solve(R_reg + B.T @ P @ B, B.T @ P @ A)
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


def _require_lqr_controller(controller: mt.Controller) -> DroneLQRController:
    if not isinstance(controller, DroneLQRController):
        raise mt.TemplateError("DroneLQRController is required for this harness.")
    return controller


def build_env() -> mt.Env:
    ctrl_cfg = CONFIG.controller
    traj_cfg = CONFIG.trajectory
    controller = DroneLQRController(
        ctrl_cfg,
        start_position=traj_cfg.start_position_m,
        start_orientation=traj_cfg.start_orientation_wxyz,
        start_velocity=traj_cfg.start_velocity_mps,
        start_angular_velocity=traj_cfg.start_angular_velocity_radps,
    )
    obs_spec = mt.ObservationSpec(include_ctrl=True, include_sensordata=False, include_time=True)
    return make_env(obs_spec=obs_spec, controller=controller)


def seed_env(env: mt.Env) -> None:
    env.reset()
    controller = _require_lqr_controller(env.controller)
    env.data.qpos[:] = controller.qpos_start
    env.data.qvel[:] = controller.qvel_start
    env.data.ctrl[: controller.ctrl_equilibrium.shape[0]] = controller.ctrl_equilibrium
    env.handle.forward()


def summarize(result: mt.PassiveRunResult) -> None:
    controller = _require_lqr_controller(result.env.controller)
    recorder = result.recorder
    rows = recorder.rows
    goal = controller.goal_position

    if rows:
        column_index = recorder.column_index
        distance_idx = column_index.get("goal_distance_m")
        if distance_idx is not None:
            distances = np.array([row[distance_idx] for row in rows], dtype=float)
            print(
                "Goal distance min {:.3f} m | max {:.3f} m | final {:.3f} m".format(
                    float(distances.min()), float(distances.max()), float(distances[-1])
                )
            )

    final_position = np.array(result.env.data.qpos[:3], dtype=float)
    final_orientation = np.array(result.env.data.qpos[3:7], dtype=float)
    final_velocity = np.array(result.env.data.qvel[:3], dtype=float)
    goal_velocity = controller.goal_velocity
    goal_angular_velocity = controller.goal_angular_velocity
    goal_orientation = controller.goal_orientation

    qvel_goal = controller.qvel_goal
    pos_dim = min(3, qvel_goal.size)
    rot_dim = min(3, max(0, qvel_goal.size - pos_dim))
    final_qvel = np.array(result.env.data.qvel, dtype=float)
    final_linear = np.zeros(3)
    final_linear[:pos_dim] = final_qvel[:pos_dim]
    final_angular = np.zeros(3)
    if rot_dim > 0:
        final_angular[:rot_dim] = final_qvel[pos_dim : pos_dim + rot_dim]

    print(
        "Final position [{:.3f}, {:.3f}, {:.3f}] m | target [{:.3f}, {:.3f}, {:.3f}] m".format(
            *final_position, *goal
        )
    )
    print(
        "Final orientation [{:.3f}, {:.3f}, {:.3f}, {:.3f}] wxyz | target [{:.3f}, {:.3f}, {:.3f}, {:.3f}] wxyz".format(
            *final_orientation,
            *goal_orientation,
        )
    )
    print(
        "Final velocity [{:.3f}, {:.3f}, {:.3f}] m/s | target [{:.3f}, {:.3f}, {:.3f}] m/s".format(
            *final_linear,
            *goal_velocity,
        )
    )
    print(
        "Final angular velocity [{:.3f}, {:.3f}, {:.3f}] rad/s | target [{:.3f}, {:.3f}, {:.3f}] rad/s".format(
            *final_angular,
            *goal_angular_velocity,
        )
    )
    print(
        "Translational speed {:.3f} m/s | simulated {:.3f} s over {} steps".format(
            float(np.linalg.norm(final_velocity)), float(result.env.data.time), result.steps
        )
    )


HARNESS = mt.PassiveRunHarness(
    build_env,
    description="Skydio X2 drone point-to-point flight via LQR (MuJoCo Template)",
    seed_fn=seed_env,
    probes=make_navigation_probes,
    start_message="Running drone LQR rollout...",
)


def main(argv=None) -> None:
    ctrl_cfg = CONFIG.controller
    traj_cfg = CONFIG.trajectory
    print(
        "Preparing drone LQR controller (keyframe {} | target [{:.2f}, {:.2f}, {:.2f}] m)".format(
            ctrl_cfg.keyframe,
            *np.asarray(ctrl_cfg.goal_position_m, dtype=float),
        )
    )
    print(
        "Start position [{:.2f}, {:.2f}, {:.2f}] m | duration {} steps".format(
            *np.asarray(traj_cfg.start_position_m, dtype=float),
            CONFIG.run.simulation.max_steps,
        )
    )
    print(
        "Start orientation [{:.3f}, {:.3f}, {:.3f}, {:.3f}] wxyz".format(
            *np.asarray(traj_cfg.start_orientation_wxyz, dtype=float)
        )
    )
    print(
        "Start velocity [{:.2f}, {:.2f}, {:.2f}] m/s | angular [{:.2f}, {:.2f}, {:.2f}] rad/s".format(
            *np.asarray(traj_cfg.start_velocity_mps, dtype=float),
            *np.asarray(traj_cfg.start_angular_velocity_radps, dtype=float),
        )
    )
    print(
        "Goal velocity [{:.2f}, {:.2f}, {:.2f}] m/s | angular [{:.2f}, {:.2f}, {:.2f}] rad/s".format(
            *np.asarray(ctrl_cfg.goal_velocity_mps, dtype=float),
            *np.asarray(ctrl_cfg.goal_angular_velocity_radps, dtype=float),
        )
    )
    result = HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize(result)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(130)
