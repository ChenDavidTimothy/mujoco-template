"""Cartpole example showcasing the SimulationSession façade."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mujoco_template as mt

ASSETS = Path(__file__).resolve().parent
MODEL_PATH = ASSETS / "cartpole.xml"


class CartPoleController:
    def __init__(self) -> None:
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self._integral = 0.0
        self._dt = 0.0

    def prepare(self, model, data) -> None:
        if model.nu != 1:
            raise mt.CompatibilityError("Cartpole model must expose a single actuator")
        self._integral = 0.0
        self._dt = float(model.opt.timestep)

    def __call__(self, model, data, _t) -> None:
        cart_x = float(data.qpos[0])
        pole_angle = float(data.qpos[1])
        cart_vel = float(data.qvel[0])
        pole_ang_vel = float(data.qvel[1])
        self._integral += pole_angle * self._dt
        self._integral = float(np.clip(self._integral, -2.0, 2.0))
        force = (
            -1.0 * cart_x
            - 2.0 * cart_vel
            - 30.0 * pole_angle
            - 6.0 * pole_ang_vel
            - 5.0 * self._integral
        )
        if hasattr(model, "actuator_ctrlrange"):
            low, high = model.actuator_ctrlrange[0]
            force = float(np.clip(force, low, high))
        data.ctrl[0] = force


def build_session(controller: mt.Controller | None = None) -> mt.SimulationSession:
    controller = controller or CartPoleController()
    spec = mt.ObservationSpec.basic().with_ctrl().with_time().with_sites("tip")
    return mt.SimulationSession.from_xml_path(str(MODEL_PATH), controller=controller, observation=spec)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0, help="Duration to simulate")
    args = parser.parse_args(argv)

    session = build_session()
    session.reset()
    result = session.run(duration_seconds=args.seconds, sample_stride=20)
    tip_heights = [float(sample.obs["sites_pos"][0, 2]) for sample in result.samples]
    print(f"Simulated {result.steps} steps; final time {session.data.time:.3f}s")
    print(
        "Tip height range: {:.3f} m – {:.3f} m".format(
            float(min(tip_heights, default=0.0)),
            float(max(tip_heights, default=0.0)),
        )
    )


if __name__ == "__main__":
    main()
