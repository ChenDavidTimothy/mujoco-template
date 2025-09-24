from __future__ import annotations

import sys

import numpy as np

import mujoco_template as mt
from cartpole_common import initialize_state as seed_cartpole, make_env as make_cartpole_env

LOG_COLUMNS = (
    "time_s",
    "cart_x_m",
    "cart_xdot_mps",
    "pole_angle_rad",
    "pole_ang_vel_rps",
    "cart_force_N",
    "tip_x_m",
    "tip_z_m",
)
DEFAULT_HEADLESS_STEPS = 2000
SAMPLE_STRIDE = 50
INITIAL_CART_POS = 0.0
INITIAL_CART_VEL = 0.0
INITIAL_POLE_ANGLE_DEG = 5.0
INITIAL_POLE_VEL_DEG = 0.0


class CartPolePIDController:
    """PID balance controller applying horizontal force to the cart."""

    def __init__(
        self,
        *,
        angle_kp: float = 120.0,
        angle_kd: float = 25.0,
        angle_ki: float = 5.0,
        position_kp: float = 0.8,
        position_kd: float = 4.0,
        integral_limit: float = 10.0,
    ) -> None:
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self.angle_kp = float(angle_kp)
        self.angle_kd = float(angle_kd)
        self.angle_ki = float(angle_ki)
        self.position_kp = float(position_kp)
        self.position_kd = float(position_kd)
        self.integral_limit = float(abs(integral_limit))
        self._integral_term = 0.0
        self._dt = 0.0

    def prepare(self, model: mt.mj.MjModel, data: mt.mj.MjData) -> None:
        if model.nu != 1:
            raise mt.CompatibilityError("CartPolePIDController expects a single actuator driving the cart.")
        self._integral_term = 0.0
        self._dt = float(model.opt.timestep)

    def __call__(self, model: mt.mj.MjModel, data: mt.mj.MjData, t: float) -> None:
        cart_x = float(data.qpos[0])
        pole_angle = float(data.qpos[1])
        cart_vel = float(data.qvel[0])
        pole_ang_vel = float(data.qvel[1])

        self._integral_term += pole_angle * self._dt
        self._integral_term = float(np.clip(self._integral_term, -self.integral_limit, self.integral_limit))

        force = (
            self.angle_kp * pole_angle
            + self.angle_kd * pole_ang_vel
            + self.angle_ki * self._integral_term
            + self.position_kp * cart_x
            + self.position_kd * cart_vel
        )
        force = -force

        if hasattr(model, "actuator_ctrlrange") and model.actuator_ctrlrange.size >= 2:
            low, high = model.actuator_ctrlrange[0]
            force = float(np.clip(force, low, high))
        data.ctrl[0] = force


def _extract_sample(result: mt.StepResult) -> tuple[float, ...]:
    obs = result.obs
    time_s = float(obs["time"][0])
    cart_x = float(obs["qpos"][0])
    pole_angle = float(obs["qpos"][1])
    cart_vel = float(obs["qvel"][0])
    pole_ang_vel = float(obs["qvel"][1])
    force = float(obs["ctrl"][0])
    tip = obs["sites_pos"][0]
    return (time_s, cart_x, cart_vel, pole_angle, pole_ang_vel, force, float(tip[0]), float(tip[2]))


def build_env() -> mt.Env:
    controller = CartPolePIDController()
    obs_spec = mt.ObservationSpec(
        include_ctrl=True,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return make_cartpole_env(obs_spec=obs_spec, controller=controller)


def run_headless(env: mt.Env, options: mt.PassiveRunCLIOptions) -> None:
    timestep = float(env.model.opt.timestep)
    if options.duration is None:
        max_steps = DEFAULT_HEADLESS_STEPS
    else:
        max_steps = max(1, int(round(options.duration / timestep)))

    print("Running cartpole PID rollout (headless)...")
    samples: list[tuple[float, ...]] = []

    with mt.TrajectoryLogger(options.log_path, LOG_COLUMNS, _extract_sample) as logger:
        def on_step(result: mt.StepResult) -> None:
            row = logger.log(result)
            samples.append(row)

        mt.run_passive_headless(env, max_steps=max_steps, hooks=on_step)

    for idx in range(0, len(samples), SAMPLE_STRIDE):
        time_s, cart_x, cart_vel, pole_angle, pole_ang_vel, force, tip_x, tip_z = samples[idx]
        print(
            "t={:5.2f}s cart={:6.3f}m cartdot={:6.3f}m/s pole={:6.2f}deg poledot={:6.2f}deg/s force={:6.2f}N tip_z={:6.3f}m".format(
                time_s,
                cart_x,
                cart_vel,
                np.rad2deg(pole_angle),
                np.rad2deg(pole_ang_vel),
                force,
                tip_z,
            )
        )

    if samples:
        pole_angles = np.array([np.rad2deg(s[3]) for s in samples], dtype=float)
        tip_z = np.array([s[7] for s in samples], dtype=float)
        print(
            "Pole angle range: {:.2f} deg to {:.2f} deg | Tip height range: {:.3f} m to {:.3f} m".format(
                float(pole_angles.min()), float(pole_angles.max()), float(tip_z.min()), float(tip_z.max())
            )
        )


def run_viewer(env: mt.Env, options: mt.PassiveRunCLIOptions) -> None:
    print("Launching MuJoCo viewer... close the window to exit.")

    with mt.TrajectoryLogger(options.log_path, LOG_COLUMNS, _extract_sample) as logger:
        def on_step(result: mt.StepResult) -> None:
            logger.log(result)

        try:
            mt.run_passive_viewer(env, duration=options.duration, hooks=on_step)
        except mt.TemplateError as exc:  # pragma: no cover - viewer availability depends on platform
            raise SystemExit(str(exc)) from exc

    print("Viewer closed. Final simulated time: {:.3f}s".format(env.data.time))


def main() -> None:
    options = mt.parse_passive_run_cli("CartPole PID balance demo")
    env = build_env()
    seed_cartpole(
        env,
        cart_position=INITIAL_CART_POS,
        cart_velocity=INITIAL_CART_VEL,
        pole_angle_rad=np.deg2rad(INITIAL_POLE_ANGLE_DEG),
        pole_velocity=np.deg2rad(INITIAL_POLE_VEL_DEG),
    )

    print(
        "Initial cart x: {:.3f} m | pole angle: {:.2f} deg | pole velocity: {:.2f} deg/s".format(
            INITIAL_CART_POS,
            INITIAL_POLE_ANGLE_DEG,
            INITIAL_POLE_VEL_DEG,
        )
    )

    if options.viewer:
        run_viewer(env, options)
    else:
        run_headless(env, options)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
