import sys
from typing import Iterable

import numpy as np
import scipy.linalg

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
    ) -> None:
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self._config = config
        self._start_position_cfg = None if start_position is None else np.asarray(start_position, dtype=float)
        self._start_orientation_cfg = (
            None if start_orientation is None else self._normalize_quat(np.asarray(start_orientation, dtype=float))
        )

        self._prepared = False
        self._nu = 0
        self._nv = 0

        self._qpos_goal: np.ndarray | None = None
        self._qpos_start: np.ndarray | None = None
        self._ctrl0: np.ndarray | None = None
        self._goal_position: np.ndarray | None = None
        self._goal_orientation: np.ndarray | None = None
        self._K: np.ndarray | None = None
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._dq_scratch: np.ndarray | None = None
        self._dx_scratch: np.ndarray | None = None
        self._ctrl_delta: np.ndarray | None = None
        self._ctrl_buffer: np.ndarray | None = None
        self._ctrl_low: np.ndarray | None = None
        self._ctrl_high: np.ndarray | None = None

    def prepare(self, model: mt.mj.MjModel, _data: mt.mj.MjData) -> None:
        cfg = self._config
        if model.nu == 0:
            raise mt.CompatibilityError("DroneLQRController requires force-producing actuators.")

        self._prepared = False
        self._nu = int(model.nu)
        self._nv = int(model.nv)

        work = mt.mj.MjData(model)
        keyframe_id = self._resolve_keyframe(model, cfg.keyframe)
        mt.mj.mj_resetDataKeyframe(model, work, keyframe_id)
        mt.mj.mj_forward(model, work)

        qpos_hover = np.array(work.qpos, dtype=float)
        ctrl_hover = np.array(work.ctrl[: self._nu], dtype=float)

        goal_position = self._normalize_position(cfg.goal_position_m)
        goal_orientation = self._normalize_quat(np.asarray(cfg.goal_orientation_wxyz, dtype=float))

        qpos_goal = np.array(qpos_hover, copy=True)
        qpos_goal[:3] = goal_position
        qpos_goal[3:7] = goal_orientation

        start_position = (
            goal_position if self._start_position_cfg is None else self._normalize_position(self._start_position_cfg)
        )
        start_orientation = goal_orientation if self._start_orientation_cfg is None else self._start_orientation_cfg

        qpos_start = np.array(qpos_goal, copy=True)
        qpos_start[:3] = start_position
        qpos_start[3:7] = start_orientation

        work.qpos[:] = qpos_goal
        work.qvel[:] = 0.0
        work.ctrl[: self._nu] = ctrl_hover
        mt.mj.mj_forward(model, work)

        Q = self._build_state_cost(cfg)
        R = self._build_ctrl_cost(cfg)

        A, B, K = self._stabilizing_lqr(model, work, Q, R)

        self._qpos_goal = qpos_goal
        self._qpos_start = qpos_start
        self._ctrl0 = ctrl_hover
        self._goal_position = goal_position
        self._goal_orientation = goal_orientation
        self._K = K
        self._A = A
        self._B = B
        self._dq_scratch = np.zeros(self._nv)
        self._dx_scratch = np.zeros(2 * self._nv)
        self._ctrl_delta = np.zeros(self._nu)
        self._ctrl_buffer = np.zeros(self._nu)

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

        mt.mj.mj_differentiatePos(model, self._dq_scratch, 1.0, self._qpos_goal, data.qpos)
        self._dx_scratch[: self._nv] = self._dq_scratch
        self._dx_scratch[self._nv :] = data.qvel

        self._ctrl_delta[:] = self._K @ self._dx_scratch
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

    def _build_state_cost(self, cfg) -> np.ndarray:
        nv = self._nv
        Q = np.zeros((2 * nv, 2 * nv))

        pos_dim = min(3, nv)
        rot_dim = min(3, max(0, nv - pos_dim))

        if pos_dim > 0:
            Q[:pos_dim, :pos_dim] = float(cfg.position_weight) * np.eye(pos_dim)
            Q[nv : nv + pos_dim, nv : nv + pos_dim] = float(cfg.velocity_weight) * np.eye(pos_dim)
        if rot_dim > 0:
            start = pos_dim
            stop = pos_dim + rot_dim
            Q[start:stop, start:stop] = float(cfg.orientation_weight) * np.eye(rot_dim)
            vel_start = nv + pos_dim
            vel_stop = vel_start + rot_dim
            Q[vel_start:vel_stop, vel_start:vel_stop] = float(cfg.angular_velocity_weight) * np.eye(rot_dim)

        return Q

    def _build_ctrl_cost(self, cfg) -> np.ndarray:
        weight = float(cfg.control_weight)
        if weight <= 0.0:
            raise mt.ConfigError("control_weight must be positive.")
        return weight * np.eye(self._nu)

    def _stabilizing_lqr(
        self,
        model: mt.mj.MjModel,
        data: mt.mj.MjData,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self._config

        eps = float(cfg.linearization_eps)
        if eps <= 0.0:
            raise mt.ConfigError("linearization_eps must be positive.")

        eps_growth = float(getattr(cfg, "linearization_eps_growth", 5.0))
        eps_max = float(getattr(cfg, "linearization_eps_max", eps))
        if eps_growth <= 1.0:
            eps_growth = 1.0
        if eps_max < eps:
            eps_max = eps

        last_error: Exception | None = None

        while True:
            mt.mj.mj_forward(model, data)
            try:
                A, B = mt.linearize_discrete(model, data, eps=eps)
            except Exception as exc:  # mt.LinearizationError or other unexpected errors
                last_error = exc
            else:
                try:
                    K = self._solve_discrete_lqr(A, B, Q, R)
                    return A, B, K
                except np.linalg.LinAlgError as exc:
                    last_error = exc

            if eps >= eps_max:
                message = "Failed to compute stabilizing LQR gains"
                if last_error is not None:
                    message += f": {last_error}"
                raise mt.TemplateError(message)

            prev_eps = eps
            eps = min(eps * eps_growth, eps_max)
            if eps == prev_eps:
                message = "Failed to compute stabilizing LQR gains"
                if last_error is not None:
                    message += f": {last_error}"
                raise mt.TemplateError(message)

    def _solve_discrete_lqr(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        cfg = self._config

        reg = float(getattr(cfg, "are_regularization", 0.0))
        reg_growth = float(getattr(cfg, "are_regularization_growth", 10.0))
        reg_max = float(getattr(cfg, "are_regularization_max", 1e-2))

        if reg_growth <= 1.0:
            reg_growth = 10.0
        if reg_max < 0.0:
            reg_max = 0.0

        attempt_reg = 0.0
        if reg > 0.0:
            attempt_reg = reg

        last_error: Exception | None = None

        while True:
            if attempt_reg > 0.0:
                I_x = np.eye(Q.shape[0])
                I_u = np.eye(R.shape[0])
                Q_eff = Q + attempt_reg * I_x
                R_eff = R + attempt_reg * I_u
            else:
                Q_eff = Q
                R_eff = R

            try:
                P = scipy.linalg.solve_discrete_are(A, B, Q_eff, R_eff)
                return np.linalg.solve(R_eff + B.T @ P @ B, B.T @ P @ A)
            except np.linalg.LinAlgError as exc:
                last_error = exc

            if attempt_reg == 0.0:
                attempt_reg = reg if reg > 0.0 else 1e-8
            else:
                attempt_reg *= reg_growth

            if attempt_reg > reg_max > 0.0:
                break
            if attempt_reg > 1.0 and reg_max == 0.0:
                break

        message = "solve_discrete_are failed to converge"
        if last_error is not None:
            message += f": {last_error}"
        raise np.linalg.LinAlgError(message)

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
    )
    obs_spec = mt.ObservationSpec(include_ctrl=True, include_sensordata=False, include_time=True)
    return make_env(obs_spec=obs_spec, controller=controller)


def seed_env(env: mt.Env) -> None:
    env.reset()
    controller = _require_lqr_controller(env.controller)
    env.data.qpos[:] = controller.qpos_start
    env.data.qvel[:] = 0.0
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
    final_velocity = np.array(result.env.data.qvel[:3], dtype=float)
    print(
        "Final position [{:.3f}, {:.3f}, {:.3f}] m | target [{:.3f}, {:.3f}, {:.3f}] m".format(
            *final_position, *goal
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
    result = HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize(result)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(130)
