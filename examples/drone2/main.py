"""
Straightforward Skydio X2 LQR example (minimal, control-engineering focused).
Removes typing/dataclass ceremony; plain Python with clear math and MuJoCo calls.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation

# Ensure repo root on path (so `mujoco_template` resolves when running from examples/)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import mujoco_template as mt

SCENE_XML = Path(__file__).with_name("scene.xml")


# -------- Parameters (simple container, manual __init__) ----------------------
class _DroneParameters:
    def __init__(
        self,
        keyframe="hover",
        linearization_eps=1e-4,
        max_steps=4000,
        duration_seconds=8.0,
        position_weight=(10.0, 10.0, 10.0),
        orientation_weight=(10.0, 10.0, 20.0),
        velocity_weight=(8.0, 8.0, 6.0),
        angular_velocity_weight=(6.0, 6.0, 10.0),
        control_weight=2.0,
        yaw_control_scale=6.0,
        yaw_proportional_gain=18.0,
        yaw_derivative_gain=4.5,
        yaw_integral_gain=4.0,
        yaw_integral_limit=6.0,
        clip_controls=True,
        start_position_m=(0.0, 0.0, 0.7),
        start_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        start_velocity_mps=(0.0, 0.0, 0.0),
        start_angular_velocity_radps=(0.0, 0.0, 0.0),
        goal_position_m=(5.0, -4.0, 2.3),
        goal_orientation_wxyz=(0.0, 0.0, 0.0, 1.0),
        goal_velocity_mps=(0.0, 0.0, 0.0),
        goal_angular_velocity_radps=(0.0, 0.0, 0.0),
        log_path=Path("drone_lqr.csv"),
        video_path=Path("drone_lqr.mp4"),
    ):
        self.keyframe = keyframe
        self.linearization_eps = linearization_eps
        self.max_steps = max_steps
        self.duration_seconds = duration_seconds
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.velocity_weight = velocity_weight
        self.angular_velocity_weight = angular_velocity_weight
        self.control_weight = control_weight
        self.yaw_control_scale = yaw_control_scale
        self.yaw_proportional_gain = yaw_proportional_gain
        self.yaw_derivative_gain = yaw_derivative_gain
        self.yaw_integral_gain = yaw_integral_gain
        self.yaw_integral_limit = yaw_integral_limit
        self.clip_controls = clip_controls
        self.start_position_m = start_position_m
        self.start_orientation_wxyz = start_orientation_wxyz
        self.start_velocity_mps = start_velocity_mps
        self.start_angular_velocity_radps = start_angular_velocity_radps
        self.goal_position_m = goal_position_m
        self.goal_orientation_wxyz = goal_orientation_wxyz
        self.goal_velocity_mps = goal_velocity_mps
        self.goal_angular_velocity_radps = goal_angular_velocity_radps
        self.log_path = log_path
        self.video_path = video_path


def _quat_from_body_yaw(yaw_deg):
    r = Rotation.from_euler("xyz", [0.0, 0.0, yaw_deg], degrees=True)
    x, y, z, w = r.as_quat()
    return float(w), float(x), float(y), float(z)


PARAMS = _DroneParameters(
    start_orientation_wxyz=_quat_from_body_yaw(0.0),
    goal_orientation_wxyz=_quat_from_body_yaw(180.0),
)

RUN_SETTINGS = mt.PassiveRunSettings.from_flags(
    viewer=False,
    video=False,
    logging=True,
    simulation_overrides=dict(
        max_steps=PARAMS.max_steps,
        duration_seconds=PARAMS.duration_seconds,
    ),
    video_overrides=dict(
        path=PARAMS.video_path,
        fps=60.0,
        width=1280,
        height=720,
        crf=20,
        preset="medium",
        tune=None,
        faststart=True,
        capture_initial_frame=True,
        adaptive_camera=mt.AdaptiveCameraSettings(
            enabled=True,
            zoom_policy="distance",
            azimuth=120.0,
            elevation=-15.0,
            distance=4.0,
            fovy=None,
            ortho_height=None,
            lookat=(0.0, 0.0, 1.0),
            safety_margin=0.2,
            widen_threshold=0.82,
            tighten_threshold=0.62,
            smoothing_time_constant=0.05,
            min_distance=4.0,
            max_distance=14.0,
            min_fovy=20.0,
            max_fovy=80.0,
            min_ortho_height=0.5,
            max_ortho_height=25.0,
            recenter_axis="x,y,z",
            recenter_time_constant=0.5,
            points_of_interest=("body:x2",),
        ),
    ),
    viewer_overrides=dict(duration_seconds=10.0),
    logging_overrides=dict(path=PARAMS.log_path, store_rows=True),
)


# -------- Small helpers -------------------------------------------------------
def _broadcast(scale, size):
    arr = np.asarray(scale, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, size)
    if arr.size != size:
        raise mt.ConfigError(f"Expected {size} entries, received {arr.size}.")
    if np.any(arr <= 0.0):
        raise mt.ConfigError("Feedback weights and scales must be positive.")
    return arr


def _solve_lqr(A, B, Q, R):
    last_error = None
    for scale in (1.0, 10.0, 100.0, 1000.0):
        try:
            P = scipy.linalg.solve_discrete_are(A, B, scale * Q, R)
            # K = (R + B^T P B)^{-1} B^T P A
            return scipy.linalg.solve(R + B.T @ P @ B, B.T @ P @ A, assume_a="sym")
        except Exception as exc:
            last_error = exc
    raise mt.TemplateError("Failed to compute LQR gains.") from last_error


def _compose_qvel(linear, angular):
    return np.concatenate([linear, angular])


def _normalize_quat(quat):
    quat = np.asarray(quat, dtype=float).reshape(4)
    n = float(np.linalg.norm(quat))
    if n <= 0.0:
        raise mt.ConfigError("Quaternion must have positive norm.")
    return quat / n


def _normalize_vector(vec, label):
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size != 3:
        raise mt.ConfigError(f"{label} must provide exactly 3 entries.")
    return arr


def _normalize_position(vec):
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size != 3:
        raise mt.ConfigError("Positions must provide exactly 3 entries.")
    return arr


def _site_id(model, name):
    site_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_SITE, name))
    if site_id < 0:
        raise mt.NameLookupError(f"Site not found: {name}")
    return site_id


# -------- Controller ----------------------------------------------------------
class _InlineDroneLQR(mt.Controller):
    capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)

    def __init__(self, cfg):
        self._cfg = cfg
        self._prepared = False
        self._printed_spectral_radius = False
        self._last_spectral_radius = 0.0
        self._nu = 0
        self._nv = 0
        self._pos_dim = 0
        self._rot_dim = 0
        self._qpos_goal = None
        self._qvel_goal = None
        self._qpos_start = None
        self._qvel_start = None
        self._ctrl_equilibrium = None
        self._goal_position = None
        self._goal_orientation = None
        self._goal_velocity = None
        self._goal_angular_velocity = None
        self._K = None
        self._dq_scratch = None
        self._dx_scratch = None
        self._ctrl_delta = None
        self._ctrl_command = None
        self._position_feedback_scale = None
        self._orientation_feedback_scale = None
        self._velocity_feedback_scale = None
        self._angular_velocity_feedback_scale = None
        self._yaw_control_scale = 1.0
        self._yaw_direction = None
        self._yaw_direction_norm = 0.0
        self._yaw_proportional_gain = 0.0
        self._yaw_derivative_gain = 0.0
        self._yaw_integral_gain = 0.0
        self._yaw_integral_limit = np.inf
        self._yaw_integral = 0.0
        self._ctrl_low = None
        self._ctrl_high = None
        self._timestep = 0.0

    def prepare(self, model, data):
        cfg = self._cfg
        if model.nu == 0:
            raise mt.compat.CompatibilityError(
                "Drone LQR controller requires actuators."
            ) if hasattr(mt, "compat") else mt.CompatibilityError(
                "Drone LQR controller requires actuators."
            )

        self._prepared = False
        self._nu = int(model.nu)
        self._nv = int(model.nv)
        self._pos_dim = min(3, self._nv)
        self._rot_dim = min(3, max(0, self._nv - self._pos_dim))
        self._timestep = float(model.opt.timestep)

        # Load hover keyframe (equilibrium)
        work = mt.mj.MjData(model)
        keyframe_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_KEY, cfg.keyframe))
        if keyframe_id < 0:
            raise mt.ConfigError(f"Keyframe '{cfg.keyframe}' not found in model.")
        mt.mj.mj_resetDataKeyframe(model, work, keyframe_id)
        mt.mj.mj_forward(model, work)

        qpos_hover = np.array(work.qpos, dtype=float)
        ctrl_hover = np.array(work.ctrl[: self._nu], dtype=float)

        # Build goal/start states
        goal_position = _normalize_position(cfg.goal_position_m)
        goal_orientation = _normalize_quat(np.asarray(cfg.goal_orientation_wxyz, dtype=float))
        goal_velocity = _normalize_vector(cfg.goal_velocity_mps, "linear velocity")
        goal_angular_velocity = _normalize_vector(
            cfg.goal_angular_velocity_radps, "angular velocity"
        )

        qpos_goal = np.array(qpos_hover, copy=True)
        qpos_goal[:3] = goal_position
        qpos_goal[3:7] = goal_orientation

        start_position = _normalize_position(cfg.start_position_m)
        start_orientation = _normalize_quat(np.asarray(cfg.start_orientation_wxyz, dtype=float))
        start_velocity = _normalize_vector(cfg.start_velocity_mps, "start linear velocity")
        start_angular_velocity = _normalize_vector(
            cfg.start_angular_velocity_radps, "start angular velocity"
        )

        qvel_goal = _compose_qvel(goal_velocity, goal_angular_velocity)
        qvel_start = _compose_qvel(start_velocity, start_angular_velocity)

        qpos_start = np.array(qpos_goal, copy=True)
        qpos_start[:3] = start_position
        qpos_start[3:7] = start_orientation

        # Linearize at goal equilibrium (with hover control)
        work.qpos[:] = qpos_goal
        work.qvel[:] = qvel_goal
        work.ctrl[: self._nu] = ctrl_hover
        mt.mj.mj_forward(model, work)

        eps = float(cfg.linearization_eps)
        if eps <= 0.0:
            raise mt.ConfigError("linearization_eps must be positive.")
        A, B = mt.linearize_discrete(model, work, eps=eps)

        # LQR design
        Q = np.diag(
            np.concatenate(
                [
                    _broadcast(cfg.position_weight, self._pos_dim),
                    _broadcast(cfg.orientation_weight, self._rot_dim),
                    _broadcast(cfg.velocity_weight, self._pos_dim),
                    _broadcast(cfg.angular_velocity_weight, self._rot_dim),
                ]
            )
        )
        if cfg.control_weight < 0.0:
            raise mt.ConfigError("control_weight must be non-negative.")
        R = float(cfg.control_weight) * np.eye(self._nu)

        K = _solve_lqr(A, B, Q, R)
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
        if (not self._printed_spectral_radius) or (
            not np.isclose(spectral_radius, self._last_spectral_radius)
        ):
            print(f"Closed-loop spectral radius: {spectral_radius:.9f}")
            self._printed_spectral_radius = True
            self._last_spectral_radius = spectral_radius

        # Cache
        self._qpos_goal = qpos_goal
        self._qvel_goal = qvel_goal
        self._qpos_start = qpos_start
        self._qvel_start = qvel_start
        self._ctrl_equilibrium = ctrl_hover
        self._goal_position = goal_position
        self._goal_orientation = goal_orientation
        self._goal_velocity = goal_velocity
        self._goal_angular_velocity = goal_angular_velocity
        self._K = K

        # Scratch and scaling
        self._dq_scratch = np.zeros(self._nv)
        self._dx_scratch = np.zeros(2 * self._nv)
        self._ctrl_delta = np.zeros(self._nu)
        self._ctrl_command = np.zeros(self._nu)
        self._position_feedback_scale = _broadcast((1.0,), self._pos_dim)
        self._velocity_feedback_scale = _broadcast((1.0,), self._pos_dim)
        if self._rot_dim > 0:
            self._orientation_feedback_scale = _broadcast((1.0,), self._rot_dim)
            self._angular_velocity_feedback_scale = _broadcast((1.0,), self._rot_dim)
        else:
            self._orientation_feedback_scale = np.zeros(0)
            self._angular_velocity_feedback_scale = np.zeros(0)

        # Yaw shaping (optional)
        self._yaw_control_scale = float(cfg.yaw_control_scale)
        if self._yaw_control_scale <= 0.0:
            raise mt.ConfigError("yaw_control_scale must be positive.")
        self._yaw_proportional_gain = float(cfg.yaw_proportional_gain)
        if self._yaw_proportional_gain < 0.0:
            raise mt.ConfigError("yaw_proportional_gain must be non-negative.")
        self._yaw_derivative_gain = float(cfg.yaw_derivative_gain)
        if self._yaw_derivative_gain < 0.0:
            raise mt.ConfigError("yaw_derivative_gain must be non-negative.")
        self._yaw_integral_gain = float(cfg.yaw_integral_gain)
        if self._yaw_integral_gain < 0.0:
            raise mt.ConfigError("yaw_integral_gain must be non-negative.")
        yaw_limit = float(cfg.yaw_integral_limit)
        if yaw_limit <= 0.0:
            raise mt.ConfigError("yaw_integral_limit must be positive.")
        self._yaw_integral_limit = yaw_limit
        self._yaw_integral = 0.0

        self._yaw_direction = None
        self._yaw_direction_norm = 0.0
        if self._rot_dim > 0:
            yaw_idx = self._nv + self._pos_dim + self._rot_dim - 1
            candidate = np.array(B[yaw_idx, : self._nu], dtype=float)
            norm = float(np.dot(candidate, candidate))
            if norm > 0.0:
                self._yaw_direction = candidate
                self._yaw_direction_norm = norm

        # Actuator limits (optional)
        self._ctrl_low = None
        self._ctrl_high = None
        if cfg.clip_controls and hasattr(model, "actuator_ctrllimited") and hasattr(
            model, "actuator_ctrlrange"
        ):
            limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
            ranges = np.asarray(model.actuator_ctrlrange, dtype=float)
            if ranges.shape == (self._nu, 2):
                self._ctrl_low = np.where(limited, ranges[:, 0], -np.inf)
                self._ctrl_high = np.where(limited, ranges[:, 1], np.inf)

        # Initialize sim state
        data.qpos[:] = qpos_start
        data.qvel[:] = qvel_start
        data.ctrl[: self._nu] = ctrl_hover
        mt.mj.mj_forward(model, data)
        self._prepared = True

    def __call__(self, model, data, _t):
        if not self._prepared:
            raise mt.TemplateError("Controller must be prepared before use.")
        # Shorthands
        mt.mj.mj_differentiatePos(
            model, self._dq_scratch, 1.0, self._qpos_goal, data.qpos
        )
        self._dx_scratch[: self._nv] = self._dq_scratch
        self._dx_scratch[self._nv :] = data.qvel - self._qvel_goal

        # Optional per-axis scaling
        if self._pos_dim > 0:
            self._dx_scratch[: self._pos_dim] *= self._position_feedback_scale
            self._dx_scratch[self._nv : self._nv + self._pos_dim] *= (
                self._velocity_feedback_scale
            )
        if self._rot_dim > 0:
            ps = self._pos_dim
            pe = ps + self._rot_dim
            self._dx_scratch[ps:pe] *= self._orientation_feedback_scale
            vs = self._nv + self._pos_dim
            ve = vs + self._rot_dim
            self._dx_scratch[vs:ve] *= self._angular_velocity_feedback_scale

        # LQR
        self._ctrl_delta[:] = self._K @ self._dx_scratch

        # Yaw PID shaping along identified control direction
        if (
            self._yaw_direction is not None
            and self._yaw_direction_norm > 0.0
            and self._rot_dim > 0
        ):
            yaw_err_idx = self._pos_dim + self._rot_dim - 1
            yaw_rate_idx = self._nv + self._pos_dim + self._rot_dim - 1
            yaw_error = float(self._dx_scratch[yaw_err_idx])
            yaw_rate_error = float(self._dx_scratch[yaw_rate_idx])
            if self._yaw_integral_gain > 0.0:
                self._yaw_integral += yaw_error * self._timestep
                self._yaw_integral = float(
                    np.clip(
                        self._yaw_integral,
                        -self._yaw_integral_limit,
                        self._yaw_integral_limit,
                    )
                )
                self._ctrl_delta += (
                    self._yaw_integral_gain * self._yaw_integral * self._yaw_direction
                )
            if self._yaw_proportional_gain > 0.0 or self._yaw_derivative_gain > 0.0:
                yaw_feedback = (
                    self._yaw_proportional_gain * yaw_error
                    + self._yaw_derivative_gain * yaw_rate_error
                )
                self._ctrl_delta += yaw_feedback * self._yaw_direction
            if self._yaw_control_scale != 1.0:
                yaw_component = float(
                    np.dot(self._ctrl_delta, self._yaw_direction)
                ) / self._yaw_direction_norm
                self._ctrl_delta += (
                    (self._yaw_control_scale - 1.0)
                    * yaw_component
                    * self._yaw_direction
                )

        # Command = equilibrium - delta; apply limits if known
        self._ctrl_command[:] = self._ctrl_equilibrium - self._ctrl_delta
        if self._ctrl_low is not None and self._ctrl_high is not None:
            np.clip(
                self._ctrl_command, self._ctrl_low, self._ctrl_high, out=self._ctrl_command
            )
        data.ctrl[: self._nu] = self._ctrl_command

    # Convenience properties for seeding and reporting
    @property
    def qpos_start(self):
        return np.array(self._qpos_start, copy=True)

    @property
    def qvel_start(self):
        return np.array(self._qvel_start, copy=True)

    @property
    def ctrl_equilibrium(self):
        return np.array(self._ctrl_equilibrium, copy=True)

    @property
    def goal_position(self):
        return np.array(self._goal_position, copy=True)

    @property
    def goal_orientation(self):
        return np.array(self._goal_orientation, copy=True)

    @property
    def goal_velocity(self):
        return np.array(self._goal_velocity, copy=True)

    @property
    def goal_angular_velocity(self):
        return np.array(self._goal_angular_velocity, copy=True)

    @property
    def qvel_goal(self):
        return np.array(self._qvel_goal, copy=True)


# -------- Env, harness, probes ------------------------------------------------
def _make_env():
    controller = _InlineDroneLQR(PARAMS)
    obs_spec = mt.ObservationSpec(include_ctrl=True, include_time=True)
    return mt.Env.from_xml_path(
        str(SCENE_XML),
        obs_spec=obs_spec,
        controller=controller,
        auto_reset=False,
    )


def _seed_env(env):
    controller = env.controller
    if not isinstance(controller, _InlineDroneLQR):
        raise mt.TemplateError("Inline drone harness requires the bundled controller.")
    env.reset()
    env.data.qpos[:] = controller.qpos_start
    env.data.qvel[:] = controller.qvel_start
    env.data.ctrl[: controller.ctrl_equilibrium.shape[0]] = controller.ctrl_equilibrium
    env.handle.forward()


def _make_navigation_probes(env):
    controller = env.controller
    if not isinstance(controller, _InlineDroneLQR):
        raise mt.TemplateError("Inline drone harness requires the bundled controller.")
    imu_site = _site_id(env.model, "imu")

    def site_component(axis):
        return lambda e, _result=None: float(e.data.site_xpos[imu_site, axis])

    def goal_distance(e, _result=None):
        position = e.data.site_xpos[imu_site]
        return float(np.linalg.norm(position - controller.goal_position))

    return (
        mt.DataProbe("imu_x_m", site_component(0)),
        mt.DataProbe("imu_y_m", site_component(1)),
        mt.DataProbe("imu_z_m", site_component(2)),
        mt.DataProbe("goal_distance_m", goal_distance),
    )


# -------- Reporting -----------------------------------------------------------
def summarize(result):
    controller = result.env.controller
    if not isinstance(controller, _InlineDroneLQR):
        raise mt.TemplateError("Inline drone harness requires the bundled controller.")

    recorder = result.recorder
    if recorder is not None and recorder.rows:
        column_index = recorder.column_index
        distance_idx = column_index.get("goal_distance_m")
        if distance_idx is not None:
            distances = np.array([row[distance_idx] for row in recorder.rows], dtype=float)
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
    goal_position = controller.goal_position

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
            *final_position,
            *goal_position,
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
            float(np.linalg.norm(final_velocity)),
            float(result.env.data.time),
            result.steps,
        )
    )


SCENARIO = mt.PassiveScenario(
    settings=RUN_SETTINGS,
    env_factory=_make_env,
    seed_fn=_seed_env,
    probes=_make_navigation_probes,
    summarize=summarize,
    description="Skydio X2 LQR point-to-point flight",
    start_message="Running drone LQR rollout...",
)


# -------- CLI -----------------------------------------------------------------
def main(argv=None):
    cfg = PARAMS
    print(
        "Preparing drone LQR controller (keyframe {} | target [{:.2f}, {:.2f}, {:.2f}] m)".format(
            cfg.keyframe,
            *np.asarray(cfg.goal_position_m, dtype=float),
        )
    )
    print(
        "Start position [{:.2f}, {:.2f}, {:.2f}] m | duration {} steps".format(
            *np.asarray(cfg.start_position_m, dtype=float),
            cfg.max_steps,
        )
    )
    print(
        "Start orientation [{:.3f}, {:.3f}, {:.3f}, {:.3f}] wxyz".format(
            *np.asarray(cfg.start_orientation_wxyz, dtype=float)
        )
    )
    print(
        "Start velocity [{:.2f}, {:.2f}, {:.2f}] m/s | angular [{:.2f}, {:.2f}, {:.2f}] rad/s".format(
            *np.asarray(cfg.start_velocity_mps, dtype=float),
            *np.asarray(cfg.start_angular_velocity_radps, dtype=float),
        )
    )
    print(
        "Goal velocity [{:.2f}, {:.2f}, {:.2f}] m/s | angular [{:.2f}, {:.2f}, {:.2f}] rad/s".format(
            *np.asarray(cfg.goal_velocity_mps, dtype=float),
            *np.asarray(cfg.goal_angular_velocity_radps, dtype=float),
        )
    )

    return SCENARIO.cli(argv)


if __name__ == "__main__":  # pragma: no cover
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


__all__ = ["RUN_SETTINGS", "SCENARIO", "summarize", "main"]
