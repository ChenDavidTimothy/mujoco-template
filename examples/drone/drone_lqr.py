"""Hover a quadrotor at a fixed target using SimulationSession."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mujoco_template as mt

ASSETS = Path(__file__).resolve().parent
MODEL_PATH = ASSETS / "scene.xml"


def make_session(target_height: float = 1.5) -> mt.SimulationSession:
    def _controller(model, data, _t):
        # Simple proportional-derivative hover around the target height.
        position = np.array(data.qpos[:3], copy=False)
        velocity = np.array(data.qvel[:3], copy=False)
        error = position - np.array([0.0, 0.0, target_height], dtype=float)
        thrust = -25.0 * error - 8.0 * velocity
        thrust_z = thrust[2] + 9.81
        data.ctrl[:] = 0.0
        data.ctrl[:4] = thrust_z / 4.0
        if hasattr(model, "actuator_ctrlrange"):
            low, high = model.actuator_ctrlrange[:4].T
            data.ctrl[:4] = np.clip(data.ctrl[:4], low, high)

    spec = (
        mt.ObservationSpec.basic()
        .with_time()
        .with_ctrl()
        .with_sites("imu")
        .with_bodies("x2", inertial=True)
    )
    return mt.SimulationSession.from_xml_path(
        str(MODEL_PATH),
        controller=mt.controller_from_callable(_controller, actuator_groups=[0]),
        observation=spec,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=6.0, help="Duration to simulate")
    parser.add_argument("--height", type=float, default=1.5, help="Hover target height (m)")
    args = parser.parse_args(argv)

    session = make_session(target_height=args.height)
    session.reset()
    result = session.run(duration_seconds=args.seconds, sample_stride=25)
    heights = [float(sample.obs["bodies_pos"][0, 2]) for sample in result.samples]
    print(f"Simulated {result.steps} steps; final time {session.data.time:.3f}s")
    print(
        "Body height range: {:.3f} m – {:.3f} m".format(
            float(min(heights, default=0.0)),
            float(max(heights, default=0.0)),
        )
    )


if __name__ == "__main__":
    main()
