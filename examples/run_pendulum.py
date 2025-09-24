from __future__ import annotations

from pathlib import Path

import numpy as np

import mujoco_template as mt


class PendulumPDController:
    """Simple PD torque controller that stabilizes the pendulum upright."""

    def __init__(self, kp: float = 20.0, kd: float = 4.0, target: float = 0.0) -> None:
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self.kp = float(kp)
        self.kd = float(kd)
        self.target = float(target)

    def prepare(self, model: mt.mj.MjModel, data: mt.mj.MjData) -> None:
        if model.nu != 1:
            raise mt.CompatibilityError("PendulumPDController expects a single actuator.")

    def __call__(self, model: mt.mj.MjModel, data: mt.mj.MjData, t: float) -> None:
        angle = float(data.qpos[0])
        velocity = float(data.qvel[0])
        torque = -self.kp * (angle - self.target) - self.kd * velocity
        if hasattr(model, "actuator_forcerange") and model.actuator_forcerange.size >= 2:
            bounds = np.asarray(model.actuator_forcerange)[0]
            torque = float(np.clip(torque, bounds[0], bounds[1]))
        data.ctrl[0] = torque


def main() -> None:
    xml_path = Path(__file__).with_name("pendulum.xml")
    handle = mt.ModelHandle.from_xml_path(str(xml_path))
    controller = PendulumPDController()
    obs_spec = mt.ObservationSpec(
        include_ctrl=True,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    env = mt.Env(handle, obs_spec=obs_spec, controller=controller)

    env.reset()
    env.data.qpos[0] = np.deg2rad(60.0)
    env.data.qvel[0] = 0.0
    env.handle.forward()

    print("Running pendulum rollout...")
    samples = []
    for step in range(400):
        res = env.step()
        obs = res.obs
        samples.append(
            (
                float(obs["time"][0]),
                float(obs["qpos"][0]),
                float(obs["qvel"][0]),
                float(obs["ctrl"][0]),
                float(obs["sites_pos"][0, 2]),
            )
        )

    for idx in range(0, len(samples), 80):
        t, angle, vel, torque, tip_z = samples[idx]
        print(
            f"t={t:5.3f}s angle={np.rad2deg(angle):6.2f}deg vel={np.rad2deg(vel):6.2f}deg/s torque={torque:6.3f}Nm tip_z={tip_z:6.3f}m"
        )

    times = np.array([s[0] for s in samples])
    tip_z = np.array([s[4] for s in samples])
    print(f"Tip height range: {tip_z.min():.4f}  m to {tip_z.max():.4f} m over {times[-1]:.3f}s")


if __name__ == "__main__":
    main()